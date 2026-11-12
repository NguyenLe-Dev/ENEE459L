
from __future__ import annotations

from math import fsum
from typing import Any, Iterable, Sequence

from schema import LATENCY_PERCENTILES, REL_FLOOR

import json
from pathlib import Path

########################################################
##########   Helper functions         ##################
########################################################


def deviation_sentence(abs_res: dict[str, Any], rel_res: dict[str, Any]) -> str:
    """The two deviation metrics in one line, with the caveat. Provided complete.

    Read it before writing `max_rel_dev`. The clause after the semicolon is the
    honest disclosure and it appears whenever any denominator was floored --
    which, on a classification head, is almost always.
    """
    line = (f"max |a-b| = {abs_res['max_abs']:.6g} at index {abs_res['index']}; "
            f"max relative = {rel_res['max_rel']:.6g}")
    if rel_res["floored"]:
        line += (f" (denominator floored on {rel_res['floored']} of "
                 f"{rel_res['n']} entries; smallest |b| was "
                 f"{rel_res['denom_min']:.3g})")
    return line


def decision_sentence(dm: dict[str, Any], tol: float) -> str:
    """The claim `decision_margin` licenses, or the one it refuses. Provided."""
    if dm["safe"]:
        return (f"no decision flips: the smallest margin over {dm['n']} inputs "
                f"is {dm['min']:.6g} and the deviation is {tol:.6g}")
    return (f"{dm['below_tol']} of {dm['n']} decisions sit inside the "
            f"deviation: smallest margin {dm['min']:.6g} at row {dm['argmin']} "
            f"against a deviation of {tol:.6g}")


class AgreementError(ValueError):
    """A comparison this module cannot make. `engine.py` exits 3."""


def _finite(v: Any, where: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise AgreementError(f"{where} is {v!r}, which is not a number.")
    try:
        f = float(v)
    except OverflowError as exc:
        raise AgreementError(
            f"{where} is {v!r}, which cannot be represented as a finite number."
        ) from exc
    if f != f or f in (float("inf"), float("-inf")):
        raise AgreementError(
            f"{where} is {f}. A NaN or an infinity in an output array is the "
            "run failing quietly; it is not a value to take a maximum over."
        )
    return f


def _percentile(xs: Sequence[float], q: float) -> float:
    """Linear interpolation between the two nearest ranks. Provided complete.

    The same implementation Labs 04 through 08 use, here for the same reason:
    a lab whose percentile is nearest-rank and a lab whose percentile
    interpolates produce different p95s from the same samples, and a student
    comparing this week's table against last week's should be comparing the
    board and not the arithmetic.
    """
    s = sorted(float(x) for x in xs)
    if not s:
        raise AgreementError("no samples.")
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * (float(q) / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


########################################################
##########    Start from here          #################
########################################################


def latency_report(ms: Iterable[float]) -> dict[str, Any]:
    values: list[float] = []
    for i, value in enumerate(ms):
        sample = _finite(value, f"ms[{i}]")
        if sample < 0:
            raise AgreementError(f"ms[{i}] is {sample}, which is negative.")
        values.append(sample)

    if not values:
        raise AgreementError("no latency samples.")

    n = len(values)
    percentiles = {
        q: _percentile(values, q) for q in LATENCY_PERCENTILES
    }
    mean = fsum(value / n for value in values)
    return {
        "n": n,
        "mean": mean,
        "worst": max(values),
        "p50": percentiles[50],
        "p95": percentiles[95],
        "p99": percentiles[99],
        "jitter": percentiles[95] - percentiles[50],
    }


def speedup(base_ms: float, other_ms: float) -> dict[str, Any]:
    base = _finite(base_ms, "base_ms")
    other = _finite(other_ms, "other_ms")
    if base <= 0:
        raise AgreementError(f"base_ms is {base}, which must be positive.")
    if other <= 0:
        raise AgreementError(f"other_ms is {other}, which must be positive.")

    factor = _finite(base / other, "speedup factor")
    return {
        "factor": factor,
        "faster": factor > 1.0,
        "saved_ms": base - other,
    }


def max_abs_dev(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    if not a or not b:
        raise AgreementError("output sequences must be non-empty.")
    if len(a) != len(b):
        raise AgreementError(
            f"output lengths differ: len(a)={len(a)}, len(b)={len(b)}."
        )

    first_a = _finite(a[0], "a[0]")
    first_b = _finite(b[0], "b[0]")
    best = _finite(abs(first_a - first_b), "absolute deviation at index 0")
    best_i = 0
    best_a = first_a
    best_b = first_b

    for i in range(1, len(a)):
        value_a = _finite(a[i], f"a[{i}]")
        value_b = _finite(b[i], f"b[{i}]")
        deviation = _finite(abs(value_a - value_b),
                            f"absolute deviation at index {i}")
        if deviation > best:
            best = deviation
            best_i = i
            best_a = value_a
            best_b = value_b

    return {
        "max_abs": best,
        "index": best_i,
        "a": best_a,
        "b": best_b,
        "n": len(a),
    }


def max_rel_dev(a: Sequence[float], b: Sequence[float],
                floor: float = REL_FLOOR) -> dict[str, Any]:
    denominator_floor = _finite(floor, "floor")
    if denominator_floor <= 0:
        raise AgreementError(
            f"floor is {denominator_floor}, which must be positive."
        )
    if not a or not b:
        raise AgreementError("output sequences must be non-empty.")
    if len(a) != len(b):
        raise AgreementError(
            f"output lengths differ: len(a)={len(a)}, len(b)={len(b)}."
        )

    best = -1.0
    best_i = 0
    floored = 0
    denom_min = float("inf")
    for i, (raw_a, raw_b) in enumerate(zip(a, b)):
        value_a = _finite(raw_a, f"a[{i}]")
        value_b = _finite(raw_b, f"b[{i}]")
        denominator = abs(value_b)
        denom_min = min(denom_min, denominator)
        if denominator < denominator_floor:
            floored += 1
        relative = _finite(
            abs(value_a - value_b) / max(denominator, denominator_floor),
            f"relative deviation at index {i}",
        )
        if relative > best:
            best = relative
            best_i = i

    return {
        "max_rel": best,
        "index": best_i,
        "floored": floored,
        "denom_min": denom_min,
        "n": len(a),
    }


def top1_agreement(rows_a: Sequence[Sequence[float]],
                   rows_b: Sequence[Sequence[float]]) -> dict[str, Any]:
    if not rows_a or not rows_b:
        raise AgreementError("score row sequences must be non-empty.")
    if len(rows_a) != len(rows_b):
        raise AgreementError(
            f"row counts differ: len(rows_a)={len(rows_a)}, "
            f"len(rows_b)={len(rows_b)}."
        )

    agree = 0
    first_disagreement = None
    for k, (raw_a, raw_b) in enumerate(zip(rows_a, rows_b)):
        if not raw_a or not raw_b:
            raise AgreementError(f"score rows at index {k} must be non-empty.")
        if len(raw_a) != len(raw_b):
            raise AgreementError(
                f"score row widths differ at index {k}: "
                f"len(rows_a[{k}])={len(raw_a)}, len(rows_b[{k}])={len(raw_b)}."
            )

        row_a = [_finite(value, f"rows_a[{k}][{c}]")
                 for c, value in enumerate(raw_a)]
        row_b = [_finite(value, f"rows_b[{k}][{c}]")
                 for c, value in enumerate(raw_b)]
        top_a = max(range(len(row_a)), key=row_a.__getitem__)
        top_b = max(range(len(row_b)), key=row_b.__getitem__)
        if top_a == top_b:
            agree += 1
        elif first_disagreement is None:
            first_disagreement = k

    n = len(rows_a)
    return {
        "n": n,
        "agree": agree,
        "disagree": n - agree,
        "rate": agree / n,
        "first_disagreement": first_disagreement,
    }


def decision_margin(rows: Sequence[Sequence[float]],
                    tol: float) -> dict[str, Any]:
    tolerance = _finite(tol, "tol")
    if tolerance < 0:
        raise AgreementError(f"tol is {tolerance}, which must be non-negative.")
    if not rows:
        raise AgreementError("score rows must be non-empty.")

    margins: list[float] = []
    for k, raw_row in enumerate(rows):
        if len(raw_row) < 2:
            raise AgreementError(
                f"rows[{k}] has fewer than two classes; decision margin "
                "is undefined."
            )
        row = [_finite(value, f"rows[{k}][{c}]")
               for c, value in enumerate(raw_row)]
        top_two = sorted(row, reverse=True)[:2]
        margins.append(
            _finite(top_two[0] - top_two[1], f"decision margin at row {k}")
        )

    min_margin = margins[0]
    argmin = 0
    for k, margin in enumerate(margins[1:], start=1):
        if margin < min_margin:
            min_margin = margin
            argmin = k

    n = len(margins)
    below_tol = sum(margin < tolerance for margin in margins)
    return {
        "n": n,
        "min": min_margin,
        "median": _percentile(margins, 50),
        "mean": fsum(margin / n for margin in margins),
        "argmin": argmin,
        "below_tol": below_tol,
        "safe": below_tol == 0,
    }

#############################################################
###############  Main function starts here ##################
#############################################################

def main(
    input_path: str | Path = "benchmark_input_data.json",
    output_path: str | Path = "benchmark_results.json",
) -> None:
    in_file = Path(input_path)
    out_file = Path(output_path)

    # 1. Load synthetic data from JSON
    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {in_file.resolve()}")

    with in_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    timings_base = data["timings_base"]
    timings_optimized = data["timings_optimized"]
    flat_logits_base = data["flat_logits_base"]
    flat_logits_opt = data["flat_logits_opt"]
    batched_scores_base = data["batched_scores_base"]
    batched_scores_opt = data["batched_scores_opt"]

    # 2. Execute metric and report functions
    report_base = latency_report(timings_base)
    report_opt = latency_report(timings_optimized)
    perf_speedup = speedup(base_ms=report_base["p50"], other_ms=report_opt["p50"])

    abs_dev = max_abs_dev(flat_logits_opt, flat_logits_base)
    rel_dev = max_rel_dev(flat_logits_opt, flat_logits_base, floor=REL_FLOOR)

    agreement = top1_agreement(batched_scores_opt, batched_scores_base)
    margin = decision_margin(batched_scores_base, tol=abs_dev["max_abs"])

    # 3. Structure output results
    results: dict[str, Any] = {
        "latency": {
            "base": report_base,
            "optimized": report_opt,
            "speedup_p50": perf_speedup,
        },
        "deviations": {
            "max_abs_dev": abs_dev,
            "max_rel_dev": rel_dev,
            "summary_sentence": deviation_sentence(abs_dev, rel_dev),
        },
        "agreement": {
            "top1_agreement": agreement,
            "decision_margin": margin,
            "decision_sentence": decision_sentence(margin, tol=abs_dev["max_abs"]),
        },
    }

    # 4. Save results to JSON
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 5. Print a console summary
    print(f"Loaded input from: {in_file.resolve()}")
    print(f"Results saved to:  {out_file.resolve()}\n")
    print("--- Summary ---")
    print(results["deviations"]["summary_sentence"])
    print(results["agreement"]["decision_sentence"])
    print(
        f"Speedup (p50): {perf_speedup['factor']:.2f}x "
        f"({perf_speedup['saved_ms']:.2f} ms saved)"
    )

if __name__ == "__main__":
    main()