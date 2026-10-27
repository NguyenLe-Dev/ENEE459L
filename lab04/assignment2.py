from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

FLOPS_PER_MAC = 2
BN_PARAMS_PER_CHANNEL = 2
BN_BUFFERS_PER_CHANNEL = 2
BUFFER_DTYPE = "fp32"
DTYPE_BYTES = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
    "int4": 0.5,
}


@dataclass(frozen=True)
class Layer:
    name: str
    kind: str
    in_shape: tuple[int, ...]
    out_shape: tuple[int, ...]
    kernel: tuple[int, int] | None = None
    groups: int = 1
    bias: bool = False
    weight_dtype: str = "fp32"
    act_dtype: str = "fp32"
    reads: tuple[str, ...] = ()

    @staticmethod
    def _elements(shape: tuple[int, ...]) -> int:
        count = 1
        for dimension in shape:
            count *= dimension
        return count

    @property
    def in_elements(self) -> int:
        return self._elements(self.in_shape)

    @property
    def out_elements(self) -> int:
        return self._elements(self.out_shape)


@dataclass
class Graph:
    name: str
    input_shape: tuple[int, ...]
    layers: list[Layer]
    precision: str = "fp32"

    def __iter__(self):
        return iter(self.layers)


def dtype_bytes(dtype: str) -> float:
    if dtype not in DTYPE_BYTES:
        raise KeyError(f"unknown dtype {dtype!r}; known: {sorted(DTYPE_BYTES)}")
    return DTYPE_BYTES[dtype]


def _layer_parameters(layer: Layer) -> int:
    if layer.kind == "conv":
        kernel_height, kernel_width = layer.kernel or (1, 1)
        weights = (
            layer.out_shape[0]
            * (layer.in_shape[0] // layer.groups)
            * kernel_height
            * kernel_width
        )
        return weights + (layer.out_shape[0] if layer.bias else 0)
    if layer.kind == "linear":
        weights = layer.out_shape[0] * layer.in_shape[0]
        return weights + (layer.out_shape[0] if layer.bias else 0)
    if layer.kind == "bn":
        return BN_PARAMS_PER_CHANNEL * layer.out_shape[0]
    return 0


def build_synthetic_graph() -> Graph:
    layers = [
        Layer("conv1", "conv", (3, 32, 32), (16, 32, 32),
              kernel=(3, 3), weight_dtype="fp32", act_dtype="fp32"),
        Layer("bn1", "bn", (16, 32, 32), (16, 32, 32)),
        Layer("relu1", "relu", (16, 32, 32), (16, 32, 32)),
        Layer("res_conv", "conv", (16, 32, 32), (16, 32, 32),
              kernel=(3, 3), reads=("relu1",)),
        Layer("add1", "add", (16, 32, 32), (16, 32, 32),
              reads=("relu1", "res_conv")),
        Layer("pool", "pool", (16, 32, 32), (16, 1, 1)),
        Layer("flatten", "flatten", (16, 1, 1), (16,)),
        Layer("fc", "linear", (16,), (10,), bias=True),
    ]
    return Graph("toy_resnet", (3, 32, 32), layers)


def _layer_macs(layer: Layer) -> int:
    if layer.kind == "conv":
        kernel_height, kernel_width = layer.kernel or (1, 1)
        input_channels_per_group = layer.in_shape[0] // layer.groups
        return (
            layer.out_elements
            * input_channels_per_group
            * kernel_height
            * kernel_width
        )
    if layer.kind == "linear":
        return layer.in_elements * layer.out_elements
    return 0


def computed(value: Any, message: str, **metadata: Any) -> dict[str, Any]:
    return {
        "type": "computed",
        "value": value,
        "message": message,
        **metadata,
    }


def arithmetic_intensity(graph: Graph) -> dict[str, Any]:
    per_layer: dict[str, dict[str, int | float]] = {}
    layers = list(graph)
    layers_by_name: dict[str, Layer] = {}
    total_flops = 0
    total_bytes = 0.0

    for index, layer in enumerate(layers):
        if layer.reads:
            producer = layers_by_name.get(layer.reads[0])
            if producer is None:
                raise ValueError(
                    f"{layer.name}: input producer {layer.reads[0]!r} "
                    "is not an earlier layer"
                )
            input_dtype = producer.act_dtype
        elif index > 0:
            input_dtype = layers[index - 1].act_dtype
        else:
            input_dtype = graph.precision

        input_count = layer.in_elements
        if layer.kind == "add" and len(layer.reads) == 2:
            input_count *= 2
        input_bytes = input_count * dtype_bytes(input_dtype)

        parameter_bytes = (
            _layer_parameters(layer) * dtype_bytes(layer.weight_dtype)
        )
        if layer.kind == "bn":
            parameter_bytes += (
                BN_BUFFERS_PER_CHANNEL
                * layer.out_shape[0]
                * dtype_bytes(BUFFER_DTYPE)
            )

        output_bytes = layer.out_elements * dtype_bytes(layer.act_dtype)
        layer_bytes = float(input_bytes + parameter_bytes + output_bytes)
        flops = _layer_macs(layer) * FLOPS_PER_MAC
        intensity = round(flops / layer_bytes, 4) if layer_bytes else 0.0

        per_layer[layer.name] = {
            "flops": flops,
            "bytes": layer_bytes,
            "intensity": intensity,
        }
        total_flops += flops
        total_bytes += layer_bytes
        layers_by_name[layer.name] = layer

    model_intensity = round(total_flops / total_bytes, 4) if total_bytes else 0.0
    return computed(
        model_intensity,
        f"{graph.name}: FLOPs over bytes, every byte counted once",
        per_layer=per_layer,
        total_flops=total_flops,
        total_bytes=float(total_bytes),
        flops_per_mac=FLOPS_PER_MAC,
        cache_reuse_assumed=False,
        note="no fusion and no cache reuse; both would lower the byte column",
    )


def main() -> None:
    graph = build_synthetic_graph()
    print(json.dumps(arithmetic_intensity(graph), indent=2))


if __name__ == "__main__":
    main()
