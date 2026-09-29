from __future__ import annotations

import statistics
from typing import Any

from bench import Bench, measured, read_first, read_text, unknown

import json

# A sample is still warm-up while it exceeds the settled rate by this fraction.
WARMUP_TOL = 0.5

# How many samples must sit strictly above a quantile before that quantile is an
# estimate rather than "the biggest number we saw, wearing a hat".
MIN_SAMPLES_ABOVE = 5

# Percentiles the record carries, in the order the schema lists them.
PERCENTILES = (50, 95, 99)

# The widest gap between neighbouring measurements, as a multiple of the typical
# gap, beyond which the sample is treated as coming from two populations.
MULTIMODAL_GAP_RATIO = 20.0

# Neither side of that gap is a mode unless it holds at least this fraction.
MIN_MODE_FRACTION = 0.10

# Below this many retained samples, modality is not a question worth answering.
MIN_SAMPLES_FOR_MODALITY = 20

# How far the last third of a run may drift from the first third, relative to
# the run's own median, before the run is not one population either.
STATIONARITY_TOL = 0.10
MIN_SAMPLES_FOR_STATIONARITY = 12

THERMAL_ZONES = "sys/devices/virtual/thermal"

POWER_RAIL_CANDIDATES = (
    "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/in1_input",
    "sys/bus/i2c/drivers/ina3221/1-0040/iio:device0/in_power0_input",
    "sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
)

GPU_LOAD_CANDIDATES = (
    "sys/devices/platform/gpu.0/load",
    "sys/devices/gpu.0/load",
)

CPUFREQ_MIN = "sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq"
CPUFREQ_MAX = "sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"


# ===========================================================================
# 1. The loop
# ===========================================================================
def run_timed_iterations(bench: Bench, repeats: int = 100) -> list[float]:
    bench.workload.synchronize()
    elapsed_times = []
    for _ in range(repeats):
        start = bench.clock()
        bench.workload.run()
        bench.workload.synchronize()
        end = bench.clock()
        duration = (end - start)/1000000.0 #nano to mili
        elapsed_times.append(duration)

    return elapsed_times

def find_warmup_boundary(samples: list[float]) -> dict[str, Any]:
    if len(samples) < 4:
        return unknown("find_warmup_boundary", "not enough samples")
    half_index = len(samples) // 2
    second_half = samples[half_index:]
    settled_median = statistics.median(second_half)
    if settled_median <= 0:
        return unknown("find_warmup_boundary", "non-positive settled median")
    threshold = settled_median * (1 + WARMUP_TOL)
    discard_count = 0
    for sample in samples:
        if sample > threshold:
            discard_count += 1
        else:
            break
    retained_count = len(samples) - discard_count
    return measured(
        value=discard_count,
        source="leading prefix above (1 + 0.5) x median of the run's second half",
        settled_rate_ms=settled_median,
        threshold_ms=threshold,
        tolerance=WARMUP_TOL,
        retained=retained_count,
    )

def summarize(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }
    n = len(samples)
    sorted_samples = sorted(samples)
    mean = statistics.fmean(sorted_samples)
    min = sorted_samples[0]
    max = sorted_samples[-1]
    if n>2:
        std = statistics.stdev(sorted_samples)
    else:
        std = 0.0
    def calc_percentile(q: float) -> float:
        h = (n - 1) * q / 100.0
        i = int(h)
        if i >= n - 1:
            return float(sorted_samples[-1])
        return sorted_samples[i] + (h-i) * (sorted_samples[i + 1] - sorted_samples[i])
    p50 = calc_percentile(PERCENTILES[0])
    p95 = calc_percentile(PERCENTILES[1])
    p99 = calc_percentile(PERCENTILES[2])
    return {
        "n": n,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(min, 4),
        "max": round(max, 4),
        "p50": round(p50, 4),
        "p95": round(p95, 4),
        "p99": round(p99, 4),
    }
def is_multimodal(samples: list[float]) -> dict[str, Any]:
    n = len(samples)
    if n < MIN_SAMPLES_FOR_MODALITY:
        return unknown("is_multimodal", "not enough samples")
    sorted_samples = sorted(samples)
    trim_count = int(n * 0.05)
    
    # Slice off top and bottom 5%
    if trim_count > 0:
        trimmed = sorted_samples[trim_count:-trim_count]
    else:
        trimmed = sorted_samples
    gaps = [trimmed[i + 1] - trimmed[i] for i in range(len(trimmed) - 1)]
    if not gaps:
        return unknown("is_multimodal", "no gaps :(")
    median_gap = statistics.median(gaps)
    if median_gap <= 0:
        return unknown("is_multimodal", "non-positive median gap")
    widest_gap = max(gaps)
    gap_ratio = widest_gap / median_gap
    split_index = gaps.index(widest_gap) + 1
    left_group = trimmed[:split_index]
    right_group = trimmed[split_index:]
    left_count = len(left_group)
    right_count = len(right_group)
    min_fraction_count = n * MIN_MODE_FRACTION
    is_multi = (gap_ratio >= MULTIMODAL_GAP_RATIO) and (
        left_count >= min_fraction_count and right_count >= min_fraction_count
    )
    modes = [
        {
            "n": left_count,
            "share": round(left_count / len(trimmed), 2),
            "median_ms": round(statistics.median(left_group), 4) if left_group else None,
        },
        {
            "n": right_count,
            "share": round(right_count / len(trimmed), 2),
            "median_ms": round(statistics.median(right_group), 4) if right_group else None,
        },
    ]
    return {
        "value": is_multi,
        "source": "widest trimmed gap >= 20.0x the median gap, with >= 10% of samples on each side",
        "status": "ok",
        "gap_ratio": round(gap_ratio, 2),
        "widest_gap_ms": round(widest_gap, 5),
        "typical_gap_ms": round(median_gap, 5),
        "modes": modes,
    }
# ===========================================================================
# 7. The clock ceiling the run happened under
# ===========================================================================


def probe_power_state(bench: Bench) -> dict[str, Any]:
    res = bench.runner(["nvpmodel", "-q"])

    # Handle command failure
    if not res.ok or res.returncode != 0:
        return unknown("nvpmodel -q", res.error or f"returncode {res.returncode}")

    # Parse nvpmodel output for power mode name and mode index
    
    lines = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
    power_mode_name = None
    mode_index = None

    for i, line in enumerate(lines):
        if "NV Power Mode:" in line:
            # Extract power mode string after the colon
            power_mode_name = line.split("NV Power Mode:", 1)[1].strip()
            # The mode index is typically on the line immediately following
            if i + 1 < len(lines):
                try:
                    mode_index = int(lines[i + 1])
                except ValueError:
                    mode_index = None
            break

    if power_mode_name is None:
        return unknown("nvpmodel -q", "failed to parse NV Power Mode from output")

    # Read CPU minimum and maximum scaling frequencies using read_text
    min_freq_str = read_text(bench.telemetry, CPUFREQ_MIN)
    max_freq_str = read_text(bench.telemetry, CPUFREQ_MAX)

    # Evaluate jetson_clocks state
    if min_freq_str is not None and max_freq_str is not None:
        jetson_clocks = (min_freq_str == max_freq_str)
        clock_source = {
            "value": f"scaling_min_freq={min_freq_str}, scaling_max_freq={max_freq_str}",
            "source": f"{CPUFREQ_MIN} vs {CPUFREQ_MAX}",
            "status": "ok"
        }
    else:
        jetson_clocks = None
        clock_source = unknown(
            f"{CPUFREQ_MIN} vs {CPUFREQ_MAX}",
            "could not read scaling frequencies"
        )
    return measured(
        value=power_mode_name,
        source="nvpmodel -q",
        mode_index=mode_index,
        jetson_clocks=jetson_clocks,
        jetson_clocks_source=clock_source,)



def probe_telemetry(bench: Bench) -> dict[str, Any]:
    root = bench.telemetry

    thermal_dir = root / THERMAL_ZONES
    highest_temp = -float("inf")
    highest_zone = None
    zones_read = 0

    if thermal_dir.exists() and thermal_dir.is_dir():
        for zone_folder in thermal_dir.iterdir():
            if zone_folder.name.startswith("thermal_zone"):
                # Construct clean relative paths for read_text
                rel_temp = f"{THERMAL_ZONES}/{zone_folder.name}/temp"
                rel_type = f"{THERMAL_ZONES}/{zone_folder.name}/type"

                temp_str = read_text(root, rel_temp)
                if temp_str:
                    try:
                        temp_mdeg = float(temp_str)
                        # Ignore disabled sensors reporting values <= -1000
                        if temp_mdeg > -1000:
                            temp_c = temp_mdeg / 1000.0
                            zones_read += 1
                            if temp_c > highest_temp:
                                highest_temp = temp_c
                                zone_type = read_text(root, rel_type)
                                highest_zone = zone_type if zone_type else zone_folder.name
                    except ValueError:
                        pass
    if zones_read > 0:
        temp_record = {
            "value": round(highest_temp, 2),
            "source": "sys/devices/virtual/thermal/*/temp",
            "status": "ok",
            "zone": highest_zone,
            "zones_read": zones_read,
        }
    else:
        temp_record = unknown(
            "sys/devices/virtual/thermal/*/temp",
            "no valid thermal zones could be read",
        )

    # Power Draw
    power_match = read_first(root, POWER_RAIL_CANDIDATES)
    power_source_str = " | ".join(POWER_RAIL_CANDIDATES)

    if power_match is not None:
        rel_path, raw_power = power_match
        try:
            power_val = int(raw_power)
            power_record = {
                "value": power_val,
                "source": rel_path,
                "status": "ok",
            }
        except ValueError:
            power_record = {
                "value": None,
                "source": power_source_str,
                "status": "unknown",
                "detail": "failed to parse power measurement as integer",
            }
    else:
        power_record = {
            "value": None,
            "source": power_source_str,
            "status": "unknown",
            "detail": "none of the documented INA3221 rail paths could be read",
        }

    # GPU
    gpu_match = read_first(root, GPU_LOAD_CANDIDATES)

    if gpu_match is not None:
        rel_path, raw_load = gpu_match
        try:
            # Converted from per-mille (tenths of a percent) to percentage
            gpu_pct = float(raw_load) / 10.0
            gpu_record = {
                "value": round(gpu_pct, 1),
                "source": rel_path,
                "status": "ok",
                "units": "per-mille / 10",
            }
        except ValueError:
            gpu_record = unknown(rel_path, "failed to parse GPU load value")
    else:
        gpu_record = unknown(
            " | ".join(GPU_LOAD_CANDIDATES),
            "no GPU load file found",
        )

    return {
        "temperature_c": temp_record,
        "power_mw": power_record,
        "gpu_utilization_percent": gpu_record,
    }

## for debugging - uncomment the following lines for debugging.
# if __name__ == "__main__":
    # env = Bench.real()
    # out = find_warmup_boundary(samples)
    # print(out)

# for generating system_report.json
if __name__ == "__main__":
    # calling base environment
    env = Bench.real()

    # get your samples
    samples = run_timed_iterations(env, repeats=100)

    # testing measurments and probes
    report = {
        "warmup_boundary": find_warmup_boundary(samples),
        "summarize_setup": summarize(samples),
        "is_multimodal": is_multimodal(samples),
        "probe_power_state": probe_power_state(env),
        "probe_telemetry": probe_telemetry(env),
    }

    # save samples
    path = "samples_analysis.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4)

    # save report
    path = "system_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)