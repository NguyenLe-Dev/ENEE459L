from __future__ import annotations

import statistics
from typing import Any

from lab04.graph import measured, unknown

MIN_SAMPLES_FOR_STAIONARITY = 12
STATIONARITY_TOL = 0.10


def is_stationary(samples: list[float]) -> dict[str, Any]:
    if len(samples) < MIN_SAMPLES_FOR_STAIONARITY:
        return unknown("is_stationary", "too few samples to divide into thirds.")

    overall_median = statistics.median(samples)
    if overall_median <= 0:
        return unknown("is_stationary", "median is not positive.")

    k = len(samples) // 3
    first_third = samples[:k]
    last_third = samples[-k:]
    first_median = statistics.median(first_third)
    last_median = statistics.median(last_third)

    drift_ms = last_median - first_median
    relative_drift = abs(drift_ms) / overall_median

    if drift_ms > 0:
        direction = "slower"
    elif drift_ms < 0:
        direction = "faster"
    else:
        direction = "flat"

    result = relative_drift <= STATIONARITY_TOL
    return measured(
        result,
        "is_stationary",
        first_third_median_ms=round(first_median, 4),
        last_third_median_ms=round(last_median, 4),
        drift_ms=round(drift_ms, 4),
        drift_relative=round(relative_drift, 4),
        direction=direction,
        tolerance=STATIONARITY_TOL,
    )


__all__ = ["MIN_SAMPLES_FOR_STAIONARITY", "STATIONARITY_TOL", "is_stationary"]
