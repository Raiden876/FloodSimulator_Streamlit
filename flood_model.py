"""Event hydrology shared by the notebook, Streamlit app, and terrain model.

CN tables: USDA NRCS TR-55 (via HEC-HMS CN Tables). WorldCover-to-TR-55
cover crosswalks are screening assumptions, not calibrated Philippine values.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

DISPLAY_DEPTH_THRESHOLD_M = 0.05
EXPOSURE_DEPTH_THRESHOLD_M = 0.10
EXPOSURE_SENSITIVITY_THRESHOLDS_M = (0.05, 0.10, 0.30, 0.50)
UNMODELED_EXPOSURE_LABEL = "Outside model coverage"

# Ordered A/B/C/D. Good woods, fair brush, good pasture, straight-row crops
# in good condition, and newly graded bare soil respectively (TR-55 2-2).
CN_BY_LAND_COVER = {
    10: (30.0, 55.0, 70.0, 77.0),
    20: (48.0, 67.0, 77.0, 83.0),
    30: (39.0, 61.0, 74.0, 80.0),
    40: (67.0, 78.0, 85.0, 89.0),
    50: (39.0, 61.0, 74.0, 80.0),  # pervious portion of built-up cells
    60: (77.0, 86.0, 91.0, 94.0),
    70: (100.0,) * 4,  # snow/ice: no-infiltration screening proxy
    80: (100.0,) * 4,  # open water
    90: (98.0,) * 4,  # saturated wetland proxy; requires local calibration
    95: (98.0,) * 4,  # saturated mangrove proxy; requires local calibration
    100: (39.0, 61.0, 74.0, 80.0),  # moss/lichen: grassland proxy
}
LAND_COVER_LABELS = {
    10: "Trees", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
    50: "Built-up", 60: "Bare ground", 70: "Snow / ice", 80: "Open water",
    90: "Herbaceous wetland", 95: "Mangroves", 100: "Moss / lichen",
}


def _finite_nonnegative(value, name):
    result = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def validate_cn(cn):
    values = np.asarray(cn, dtype=float)
    if np.any(~np.isfinite(values)) or np.any((values <= 0) | (values > 100)):
        raise ValueError("Curve Numbers must be finite and in (0, 100]")
    return values


def adjust_cn_amc(cn_ii, prior_rain_mm):
    """Apply the existing five-day AMC convention once, scalar or array."""
    cn = validate_cn(cn_ii)
    prior = float(_finite_nonnegative(prior_rain_mm, "prior_rain_mm"))
    if prior < 35.0:
        adjusted = 4.2 * cn / (10.0 - 0.058 * cn)
    elif prior > 52.5:
        adjusted = 23.0 * cn / (10.0 + 0.13 * cn)
    else:
        adjusted = cn.copy()
    adjusted = np.clip(adjusted, np.finfo(float).tiny, 100.)
    return float(adjusted) if adjusted.ndim == 0 else adjusted


def get_effective_cn(row, prior_rain_mm):
    """Respect an explicit effective CN; missing/NaN values use CN_base."""
    if "CN_amc" in row and pd.notna(row["CN_amc"]):
        return float(validate_cn(row["CN_amc"]))
    return float(adjust_cn_amc(row["CN_base"], prior_rain_mm))


def effective_cn_values(frame, prior_rain_mm):
    """Vector equivalent of get_effective_cn, including partial CN_amc data."""
    cn = np.empty(len(frame), dtype=float)
    supplied = (frame["CN_amc"].notna().to_numpy() if "CN_amc" in frame
                else np.zeros(len(frame), dtype=bool))
    if supplied.any():
        cn[supplied] = validate_cn(frame.loc[supplied, "CN_amc"].to_numpy())
    if (~supplied).any():
        cn[~supplied] = adjust_cn_amc(frame.loc[~supplied, "CN_base"].to_numpy(), prior_rain_mm)
    return cn


def scs_runoff_mm(cumulative_rain_mm, cn):
    """Cumulative event excess; Ia=0.2S. CN=100 is valid, including dry input."""
    rain = _finite_nonnegative(cumulative_rain_mm, "rain_mm")
    cn = validate_cn(cn)
    retention = 25400.0 / cn - 254.0
    excess = np.maximum(rain - 0.2 * retention, 0.0)
    denominator = excess + retention
    out = np.zeros(np.broadcast_shapes(rain.shape, cn.shape), dtype=float)
    np.divide(excess ** 2, denominator, out=out, where=denominator > 0)
    return float(out) if out.ndim == 0 else out


def empirical_depth_m(cn, base_depth_m, rain_mm, duration_h, benchmark_rain_mm=180.0):
    """Preserve the old vectorized depth relationship as the comparison baseline."""
    duration = float(_finite_nonnegative(duration_h, "duration_h"))
    if duration <= 0:
        raise ValueError("duration_h must be greater than zero")
    base = _finite_nonnegative(base_depth_m, "base_depth_m")
    benchmark = float(_finite_nonnegative(benchmark_rain_mm, "benchmark_rain_mm"))
    if benchmark <= 0:
        raise ValueError("benchmark_rain_mm must be greater than zero")
    actual = scs_runoff_mm(rain_mm, cn)
    reference = scs_runoff_mm(benchmark, cn)
    ratio = np.divide(actual, reference, out=np.zeros_like(np.asarray(reference)), where=np.asarray(reference) > 0)
    return np.round(base * ratio * (1.0 - np.exp(-0.75 * duration)), 2)


def normalize_hsg(group, drained=False):
    """Resolve A/D, B/D, C/D only when adequate drainage is explicitly supplied."""
    group = str(group).upper().replace(" ", "")
    if group in {"A/D", "B/D", "C/D"}:
        return group[0] if drained else "D"
    if group not in {"A", "B", "C", "D"}:
        raise ValueError(f"Unsupported HSG: {group}")
    return group


def hsg_from_properties(ksat_0_50_um_s, ksat_0_100_um_s,
                        restrictive_depth_cm, water_table_depth_cm):
    """Undrained NEH 630 chapter 7 criteria; Ksat is the minimum by profile.

    Uses the shallow profile limits when a restriction or water table is within
    100 cm, and the deeper-profile limits only when both are deeper than 100 cm.
    Missing diagnostics produce NaN, never a texture-derived soil group.
    """
    k50, k100, restriction, water = np.broadcast_arrays(*[
        np.asarray(v, dtype=float) for v in (ksat_0_50_um_s, ksat_0_100_um_s,
                                           restrictive_depth_cm, water_table_depth_cm)
    ])
    deep = (restriction > 100.0) & (water > 100.0)
    ksat = np.where(deep, k100, k50)
    valid = np.isfinite(ksat) & np.isfinite(restriction) & np.isfinite(water)
    valid &= (ksat >= 0) & (restriction >= 0) & (water >= 0)
    a, b, c = (np.where(deep, d, s) for d, s in ((10., 40.), (4., 10.), (.4, 1.)))
    groups = np.select([ksat > a, ksat > b, ksat > c], [1., 2., 3.], default=4.)
    groups[(restriction <= 50.) | (water <= 60.)] = 4.
    return np.where(valid, groups, np.nan)


def curve_numbers(land_cover, hsg, impervious_fraction=None, built_up_fraction=0.65):
    """CN from aligned WorldCover codes and numeric HSG 1..4.

    An explicit impervious raster mixes pervious CN with paved CN=98 in every
    terrestrial cell. Otherwise only built-up cells use built_up_fraction.
    """
    cover, groups = np.broadcast_arrays(np.asarray(land_cover), np.asarray(hsg, dtype=float))
    if np.any(~np.isin(cover, list(CN_BY_LAND_COVER))):
        raise ValueError("Land-cover values must be supported ESA WorldCover codes")
    if np.any(~np.isin(groups, [1, 2, 3, 4])):
        raise ValueError("HSG raster values must be 1=A, 2=B, 3=C, or 4=D")
    if not np.isfinite(built_up_fraction) or not 0 <= built_up_fraction <= 1:
        raise ValueError("built_up_fraction must be in [0, 1]")
    cn = np.empty(cover.shape, dtype=float)
    for code, table in CN_BY_LAND_COVER.items():
        chosen = cover == code
        cn[chosen] = np.asarray(table)[groups[chosen].astype(int) - 1]
    fraction = (np.where(cover == 50, built_up_fraction, 0.) if impervious_fraction is None
                else np.broadcast_to(np.asarray(impervious_fraction, dtype=float), cover.shape))
    if np.any(~np.isfinite(fraction)) or np.any((fraction < 0) | (fraction > 1)):
        raise ValueError("Impervious fractions must be finite and in [0, 1]")
    terrestrial = ~np.isin(cover, [70, 80, 90, 95])
    cn[terrestrial] = (cn * (1. - fraction) + 98. * fraction)[terrestrial]
    return cn


@dataclass(frozen=True)
class RainfallEvent:
    """Rain depth (mm) in consecutive intervals, each with a duration (hours)."""
    rain_mm: tuple
    interval_h: tuple

    def __post_init__(self):
        rain = _finite_nonnegative(self.rain_mm, "rainfall intervals")
        times = _finite_nonnegative(self.interval_h, "rainfall interval durations")
        if rain.ndim != 1 or times.ndim != 1 or not len(rain) or len(rain) != len(times):
            raise ValueError("Rainfall and interval_h must be nonempty equal-length sequences")
        if np.any(times <= 0):
            raise ValueError("Every interval_h must be greater than zero")
        if len(rain) > 2000 or times.sum() > 168:
            raise ValueError("Event limit: 2000 intervals and 168 hours")
        object.__setattr__(self, "rain_mm", tuple(float(x) for x in rain))
        object.__setattr__(self, "interval_h", tuple(float(x) for x in times))

    @property
    def total_mm(self):
        return float(sum(self.rain_mm))

    @property
    def duration_h(self):
        return float(sum(self.interval_h))

    @classmethod
    def from_frame(cls, frame):
        if not {"rain_mm", "interval_h"}.issubset(frame.columns):
            raise ValueError("Rainfall CSV requires rain_mm and interval_h columns")
        return cls(tuple(pd.to_numeric(frame["rain_mm"], errors="raise")),
                   tuple(pd.to_numeric(frame["interval_h"], errors="raise")))

    def to_frame(self):
        return pd.DataFrame({"time_h": np.cumsum(self.interval_h),
                             "interval_h": self.interval_h, "rain_mm": self.rain_mm,
                             "intensity_mm_h": np.asarray(self.rain_mm) / self.interval_h})


def make_rainfall_event(total_mm, duration_h, pattern="Uniform", interval_h=0.5):
    """Synthetic timing scenarios, normalized to the requested total depth."""
    total = float(_finite_nonnegative(total_mm, "rain_mm"))
    duration = float(_finite_nonnegative(duration_h, "duration_h"))
    if duration <= 0 or not np.isfinite(interval_h) or interval_h <= 0:
        raise ValueError("Duration and interval_h must be greater than zero")
    count = int(np.ceil(duration / interval_h))
    times = np.full(count, float(interval_h))
    times[-1] = duration - float(interval_h) * (count - 1)
    centers = (np.cumsum(times) - times / 2.) / duration
    if pattern == "Uniform":
        weights = times.copy()
    elif pattern in {"Early peak", "Middle peak", "Late peak"}:
        peak = {"Early peak": .2, "Middle peak": .5, "Late peak": .8}[pattern]
        weights = times * (.05 + np.exp(-.5 * ((centers - peak) / .12) ** 2))
    else:
        raise ValueError(f"Unknown rainfall pattern: {pattern}")
    return RainfallEvent(tuple(total * weights / weights.sum()), tuple(times))


def incremental_runoff_mm(event, cn):
    cn = validate_cn(cn)
    cumulative = np.cumsum(event.rain_mm)
    excess = scs_runoff_mm(cumulative.reshape((-1,) + (1,) * cn.ndim), cn)
    return np.maximum(np.diff(excess, axis=0, prepend=np.zeros_like(excess[:1])), 0.)


def route_runoff(event, cn, routing_time_h=1.0, recession_h=None):
    """Exact linear-reservoir balance for piecewise-constant excess inflow.

    S = I*k + (S0-I*k)*exp(-dt/k). Interval outflow volume equals input plus
    starting storage minus ending storage. Volumes are retained through the
    recession tail, including the residual storage at its end.
    """
    lag = float(routing_time_h)
    if not np.isfinite(lag) or lag <= 0:
        raise ValueError("routing_time_h must be finite and greater than zero")
    recession = 6. * lag if recession_h is None else float(recession_h)
    if not np.isfinite(recession) or recession < 0 or recession > 168:
        raise ValueError("recession_h must be in [0, 168]")
    increments = incremental_runoff_mm(event, cn)
    shape = increments.shape[1:]
    times = list(event.interval_h)
    rain = list(event.rain_mm)
    tail_count = int(np.ceil(recession / .5))
    tail_times = [min(.5, recession - i * .5) for i in range(tail_count)]
    times.extend(tail_times)
    rain.extend([0.] * tail_count)
    increments = np.concatenate([increments, np.zeros((tail_count,) + shape)])
    storage = np.zeros(shape)
    peak = np.zeros(shape)
    peak_time = np.zeros(shape)
    max_storage = np.zeros(shape)
    outflows, storages, outlet_rates = [], [], []
    elapsed = 0.
    for dt, runoff in zip(times, increments):
        inflow = runoff / dt
        end = inflow * lag + (storage - inflow * lag) * np.exp(-dt / lag)
        outflow = runoff + storage - end
        # With constant inflow, reservoir discharge is monotonic in the interval.
        start_rate, end_rate = storage / lag, end / lag
        candidate = np.maximum(start_rate, end_rate)
        candidate_time = np.where(end_rate >= start_rate, elapsed + dt, elapsed)
        peak_time = np.where(candidate > peak, candidate_time, peak_time)
        peak = np.maximum(peak, candidate)
        max_storage = np.maximum(max_storage, end)
        outflows.append(outflow)
        storages.append(end.copy())
        outlet_rates.append(end_rate)
        storage = end
        elapsed += dt
    outflows = np.asarray(outflows)
    return {
        "time_h": np.cumsum(times), "interval_h": np.asarray(times),
        "rain_mm": np.asarray(rain), "incremental_runoff_mm": increments,
        "outflow_mm": outflows, "storage_mm": np.asarray(storages),
        "outlet_mm_h": np.asarray(outlet_rates), "peak_mm_h": peak,
        "peak_time_h": peak_time, "peak_storage_mm": max_storage,
        "runoff_mm": increments.sum(axis=0), "residual_storage_mm": storage,
        "balance_error_mm": increments.sum(axis=0) - outflows.sum(axis=0) - storage,
    }


def validation_metrics(predicted, observed, threshold_m=EXPOSURE_DEPTH_THRESHOLD_M, valid_mask=None):
    """Independent raster depth validation, excluding missing/invalid observations."""
    pred, obs = np.asarray(predicted, float), np.asarray(observed, float)
    if pred.shape != obs.shape:
        raise ValueError("Predicted and observed arrays must share a grid")
    threshold = float(_finite_nonnegative(threshold_m, "threshold_m"))
    valid = np.isfinite(pred) & np.isfinite(obs) & (pred >= 0) & (obs >= 0)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, bool)
        if mask.shape != pred.shape:
            raise ValueError("valid_mask must share the prediction grid")
        valid &= mask
    p, o = pred[valid], obs[valid]
    wet_p, wet_o = p >= threshold, o >= threshold
    tp = int(np.count_nonzero(wet_p & wet_o))
    fp = int(np.count_nonzero(wet_p & ~wet_o))
    fn = int(np.count_nonzero(~wet_p & wet_o))
    union = tp + fp + fn
    return {"valid_cells": len(p), "hits": tp, "false_alarms": fp, "misses": fn,
            "inundation_iou": tp / union if union else np.nan,
            "critical_success_index": tp / union if union else np.nan,
            "depth_mae_m": float(np.mean(np.abs(p - o))) if len(p) else np.nan,
            "depth_rmse_m": float(np.sqrt(np.mean((p - o) ** 2))) if len(p) else np.nan}


def rainfall_runoff_analysis(event, hydrograph):
    """Separate rainfall intervals and outlet-flow samples on a shared time axis.

    Apparent peak lag uses the midpoint of a unique peak-intensity interval.
    The lag range retains the input interval's timing uncertainty. Flat or tied
    rainfall peaks, and zero outlet flow, have no unique lag estimate.
    """
    rainfall = event.to_frame()
    rainfall["start_h"] = rainfall["time_h"] - rainfall["interval_h"]
    runoff = hydrograph[["time_h", "outflow_m3_s"]].copy()
    if {"peak_outflow_time_h", "peak_outflow_m3_s"}.issubset(hydrograph.columns):
        peaks = hydrograph[["peak_outflow_time_h", "peak_outflow_m3_s"]].rename(
            columns={"peak_outflow_time_h": "time_h", "peak_outflow_m3_s": "outflow_m3_s"})
        runoff = pd.concat([runoff, peaks], ignore_index=True)
    runoff = pd.concat([pd.DataFrame({"time_h": [0.], "outflow_m3_s": [0.]}), runoff], ignore_index=True)
    runoff = runoff.groupby("time_h", as_index=False)["outflow_m3_s"].max().sort_values("time_h")
    if not np.all(np.isfinite(runoff.to_numpy())) or (runoff.to_numpy() < 0).any():
        raise ValueError("Hydrograph samples must be finite and non-negative")
    peak_row = runoff.loc[runoff["outflow_m3_s"].idxmax()]
    peak_rain = float(rainfall["intensity_mm_h"].max())
    tied = np.flatnonzero(np.isclose(rainfall["intensity_mm_h"], peak_rain))
    timing = {"axis_end_h": max(event.duration_h, float(runoff["time_h"].max())),
              "peak_runoff_time_h": float(peak_row.time_h),
              "peak_runoff_m3_s": float(peak_row.outflow_m3_s),
              "peak_rain_start_h": None, "peak_rain_end_h": None,
              "apparent_lag_h": None, "lag_min_h": None, "lag_max_h": None}
    if len(tied) == 1 and peak_rain > 0. and timing["peak_runoff_m3_s"] > 0.:
        row = rainfall.iloc[int(tied[0])]
        start, end = float(row.start_h), float(row.time_h)
        peak_time = timing["peak_runoff_time_h"]
        timing.update(peak_rain_start_h=start, peak_rain_end_h=end,
                      apparent_lag_h=peak_time - (start + end) / 2.,
                      lag_min_h=peak_time - end, lag_max_h=peak_time - start)
    return rainfall, runoff, timing
