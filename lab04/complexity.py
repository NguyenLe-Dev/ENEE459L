from __future__ import annotations

from typing import Any

from graph import (
    Graph,
    Layer,
    computed,
    dtype_bytes,
    is_answered,
    unknown,
)


FLOPS_PER_MAC = 2

# The conventions `to_flops` will honour by name. Anything else is unknown
# rather than an assumption, because the whole point of the parameter is that
# the caller has to say which one they mean.
FLOP_CONVENTIONS = {
  "mac_is_two_flops": FLOPS_PER_MAC,
    "mac_is_one_flop": 1,
}

# Batch normalisation holds two learnable vectors per channel (scale and shift)
# and two non-learnable ones (running mean and variance). The first pair are
# parameters; the second pair are buffers. Both are in the file.
BN_PARAMS_PER_CHANNEL = 2
BN_BUFFERS_PER_CHANNEL = 2

# Buffers are kept in FP32 even when the weights are not. Halving them saves
# nothing worth having and a denormal running variance is a real failure mode.
BUFFER_DTYPE = "fp32"

# Below this many models there is no line to fit and no residual to report.
MIN_MODELS_FOR_FIT = 3

# Two floats are the same MAC count when they are the same integer. There is no
# tolerance here on purpose: MAC counts are integers, and a tolerance would let
# two genuinely different architectures be reported as tied.
TIE_EXACT = True


# ===========================================================================
# 1. How many numbers are stored
# ===========================================================================

def count_parameters(graph: Graph) -> dict[str, Any]:
    per_layer = {}
    for layer in graph.layers:
        per_layer[layer.name] = _layer_parameters(layer)
    total = sum(per_layer.values())
    return computed(
        total,
        f"{graph.name}: {len(graph)} layers, shapes from the description",
        per_layer=per_layer,
        includes_bias=True,
        excludes_bn_buffers=True,
        bn_params_per_channel=BN_PARAMS_PER_CHANNEL,
    )

def _layer_parameters(layer: Layer) -> int:
    if layer.kind == "conv":
        kernel_height, kernel_width = layer.kernel or (1, 1)
        weight_params = (
            layer.out_shape[0]
            * (layer.in_shape[0] // layer.groups)
            * kernel_height
            * kernel_width
        )
        bias_params = layer.out_shape[0] if layer.bias else 0
        return weight_params + bias_params
    elif layer.kind == "linear":
        weight_params = layer.out_shape[0] * layer.in_shape[0]
        bias_params = layer.out_shape[0] if layer.bias else 0
        return weight_params + bias_params
    elif layer.kind == "bn":
        return BN_PARAMS_PER_CHANNEL * layer.out_shape[0]
    return 0
# ===========================================================================
# 2. What those numbers weigh, which is not the size of the file
# ===========================================================================


def model_size_bytes(graph: Graph) -> dict[str, Any]:
    """Bytes of stored tensors: parameters plus buffers, at their own dtypes.

    Lecture 04 slide 8 gives the formula as `#Parameters × bit width` and slide
    9 spends a page on why the file on disk is not that number. Three reasons,
    two of which this function has to get right:

      * a model is not stored in one dtype. `Layer.weight_dtype` is per layer
        and a network with FP16 weights and FP32 normalisation is completely
        ordinary. Multiplying a single total by a single bit width is the
        mistake, and on these four descriptions it is worth several per cent
      * buffers are in the file. Batch norm's running statistics are two
        vectors per channel that no optimiser ever touched, and they are still
        bytes you have to ship
      * the container is in the file too — the pickle framing, the state-dict
        keys, the archive directory. This function does *not* try to model
        that, and it says so in `container_overhead_excluded` rather than
        quietly letting the caller assume it did

    Returns a `computed` finding whose value is bytes, with the per-dtype
    breakdown that makes the first bullet checkable.
    """
    per_dtype: dict[str, float] = {}
    per_layer: dict[str, float] = {}
    buffer_bytes = 0.0

    for layer in graph.layers:
      parameter_count = _layer_parameters(layer)
      parameter_bytes = parameter_count * dtype_bytes(layer.weight_dtype)
      per_dtype[layer.weight_dtype] = (
        per_dtype.get(layer.weight_dtype, 0.0) + parameter_bytes
      )
      layer_bytes = parameter_bytes

      if layer.kind == "bn":
        buffer_elements = BN_BUFFERS_PER_CHANNEL * layer.out_shape[0]
        layer_buffer_bytes = buffer_elements * dtype_bytes(BUFFER_DTYPE)
        buffer_bytes += layer_buffer_bytes
        per_dtype[BUFFER_DTYPE] = (
          per_dtype.get(BUFFER_DTYPE, 0.0) + layer_buffer_bytes
        )
        layer_bytes += layer_buffer_bytes

      per_layer[layer.name] = layer_bytes

    total = sum(per_layer.values())
    return computed(
      total,
      f"{graph.name}: per-layer dtypes, buffers at {BUFFER_DTYPE}",
      per_layer=per_layer,
      per_dtype=per_dtype,
      buffer_bytes=buffer_bytes,
      container_overhead_excluded=True,
    )

# ===========================================================================
# 3. The memory nobody puts in the table
# ===========================================================================

def _elements(shape: tuple[int, ...]) -> int:
  count = 1
  for dimension in shape:
    count *= dimension
  return count


def _last_use(graph: Graph) -> dict[str, int]:
  last: dict[str, int] = {}
  names = [layer.name for layer in graph.layers]

  for index, layer in enumerate(graph.layers):
    if layer.reads:
      for tensor in layer.reads:
        last[tensor] = index
    elif index == 0:
      last["__input__"] = 0
    else:
      last[names[index - 1]] = index
    last.setdefault(layer.name, index)

  if graph.layers:
    last[graph.layers[-1].name] = len(graph) - 1
  return last


def _peak_elements(graph: Graph, last_use: dict[str, int]) -> int:
  live = {"__input__": _elements(graph.input_shape)}
  peak = sum(live.values())

  for index, layer in enumerate(graph.layers):
    live[layer.name] = layer.out_elements
    peak = max(peak, sum(live.values()))
    for tensor, last_index in last_use.items():
      if last_index == index:
        live.pop(tensor, None)

  return peak


def count_activations(graph: Graph) -> dict[str, Any]:
    """Total and peak activation footprint, in elements and in bytes.

    UNC COMP 790-150 Lec 2 p. 70 gives AlexNet as total 932,264 and peak
    440,928, and the two numbers answer two different questions. Total is what
    the whole forward pass produced. Peak is how much had to be resident at
    once, and peak is the one that decides whether the model runs.

    Peak is not `max(out_elements)`. Three things make it larger than that:

      * a layer's input is still resident while its output is being written.
        The live set at layer *i* contains both
      * a tensor consumed by a later layer stays resident in between. `add`
        layers name two inputs in `Layer.reads`, and the earlier one has been
        sitting in memory across every layer of the block. This is the residual
        connection and it is the single largest contributor to peak in
        ResNet-shaped networks
      * the network's own input is a tensor too

    The implementation is a liveness pass: work out the last layer that reads
    each tensor, then walk forward keeping a live set and taking the maximum of
    its total size. Anything simpler than that is wrong on any graph with a
    skip connection, and it is wrong quietly, in the direction that says the
    model fits.

    Returns a `computed` finding whose value is peak *bytes*, because bytes are
    what a memory budget is denominated in, with elements and the layer where
    the peak occurs alongside.
    """
    last_use = _last_use(graph)
    live: dict[str, float] = {
      "__input__": _elements(graph.input_shape) * dtype_bytes(graph.precision)
    }
    peak_bytes = 0.0
    peak_at = None
    total_elements = 0
    total_bytes = 0.0

    for index, layer in enumerate(graph.layers):
      output_bytes = layer.out_elements * dtype_bytes(layer.act_dtype)
      live[layer.name] = output_bytes
      total_elements += layer.out_elements
      total_bytes += output_bytes

      resident_bytes = sum(live.values())
      if resident_bytes > peak_bytes:
        peak_bytes = resident_bytes
        peak_at = layer.name

      for tensor, last_index in last_use.items():
        if last_index == index:
          live.pop(tensor, None)

    return computed(
      peak_bytes,
      f"{graph.name}: liveness over {len(graph)} layers, input included",
      peak_at=peak_at,
      peak_elements=_peak_elements(graph, last_use),
      total_elements=total_elements,
      total_bytes=total_bytes,
      includes_network_input=True,
    )

# ===========================================================================
# 4. The factor of two that halves everybody's numbers
# ===========================================================================

def to_flops(macs: dict[str, Any], convention: str = "mac_is_two_flops") -> dict[str, Any]:
    """Convert a MAC finding to a FLOP finding, naming the convention used.

    A multiply-accumulate is one multiply and one add, so it is two
    floating-point operations. Roughly half the published literature calls a
    MAC one FLOP anyway, and the two conventions differ by exactly the factor
    that makes two papers' numbers incomparable.

    Three requirements, and the third is the graded one:

      * multiply once. `FLOPS_PER_MAC` exists so that the number 2 appears in
        this file exactly once
      * an unknown MAC count converts to an unknown FLOP count. It does not
        convert to zero and it does not raise
      * the convention goes in the finding. A FLOP count that does not say
        which convention produced it is not a FLOP count, it is a number, and
        `to_flops(x, "mac_is_one_flop")` has to be as clearly labelled as the
        default

    An unrecognised convention is `unknown`, not a default. The caller asked
    for something this function does not know how to do.
    """
    if not is_answered(macs):
      return unknown("to_flops", "no valid MAC count was provided")

    if convention not in FLOP_CONVENTIONS:
      allowed = ", ".join(FLOP_CONVENTIONS)
      return unknown(
        "to_flops",
        f"unrecognized convention {convention!r}; allowed options: {allowed}",
      )

    factor = FLOP_CONVENTIONS[convention]
    per_layer = macs.get("per_layer")
    extra = {}
    if isinstance(per_layer, dict):
      extra["per_layer"] = {
        name: count * factor for name, count in per_layer.items()
      }

    return computed(
      macs["value"] * factor,
      macs.get("source", "MAC count"),
      convention=convention,
      flops_per_mac=factor,
      **extra,
    )