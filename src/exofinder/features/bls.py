"""Box Least Squares transit search and candidate feature extraction."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from astropy.timeseries import BoxLeastSquares
from scipy.ndimage import uniform_filter1d

from ..config import AUX_FEATURES, DEFAULT_VIEW, ViewConfig

DURATIONS = np.array([0.04, 0.06, 0.09, 0.13, 0.19, 0.27])  # days
MAX_PERIODS = 250_000


@dataclass
class Candidate:
    period: float
    t0: float
    duration: float
    depth: float  # fractional
    depth_err: float
    snr: float
    sde: float  # signal detection efficiency of the periodogram peak
    odd_depth: float = np.nan
    odd_err: float = np.nan
    even_depth: float = np.nan
    even_err: float = np.nan
    secondary_depth: float = np.nan
    secondary_err: float = np.nan
    n_transits: int = 0
    # Normalised periodogram for plotting; empty when the ephemeris was supplied.
    grid_period: np.ndarray = field(default_factory=lambda: np.zeros(0))
    grid_sde: np.ndarray = field(default_factory=lambda: np.zeros(0))


def _pair(x) -> tuple[float, float]:
    try:
        v, e = float(x[0]), float(x[1])
    except (TypeError, IndexError, ValueError):
        v, e = float(x), np.nan
    return v, e


def _fill_stats(c: Candidate, bls: BoxLeastSquares) -> Candidate:
    try:
        s = bls.compute_stats(c.period, c.duration, c.t0)
    except Exception:  # astropy raises on degenerate inputs; features fall back to zeros
        return c
    c.odd_depth, c.odd_err = _pair(s.get("depth_odd", (np.nan, np.nan)))
    c.even_depth, c.even_err = _pair(s.get("depth_even", (np.nan, np.nan)))
    c.secondary_depth, c.secondary_err = _pair(s.get("depth_phased", (np.nan, np.nan)))
    counts = np.asarray(s.get("per_transit_count", []))
    c.n_transits = int(np.sum(counts > 0)) if counts.size else 0
    return c


def _bin_uniform(time: np.ndarray, flux: np.ndarray, max_points: int):
    if time.size <= max_points:
        return time, flux
    width = max((time[-1] - time[0]) / max_points, float(np.median(np.diff(time))))
    idx = np.floor((time - time[0]) / width).astype(int)
    uniq, inv = np.unique(idx, return_inverse=True)
    cnt = np.bincount(inv)
    return (
        np.bincount(inv, weights=time) / cnt,
        np.bincount(inv, weights=flux) / cnt,
    )


def period_grid(baseline: float, cfg: ViewConfig, oversample: float = 3.0) -> np.ndarray:
    """Log-spaced grid fine enough that a transit cannot slip between trial periods."""
    pmin = cfg.bls_min_period
    pmax = min(cfg.bls_max_period, baseline / 2.0)
    if pmax <= pmin:
        raise ValueError("Baseline too short for the configured minimum period.")
    dln = DURATIONS.min() / (baseline * oversample)
    n = int(np.clip(np.log(pmax / pmin) / dln, 2000, MAX_PERIODS))
    return np.exp(np.linspace(np.log(pmin), np.log(pmax), n))


def search(
    time: np.ndarray,
    flat: np.ndarray,
    sigma: float,
    cfg: ViewConfig = DEFAULT_VIEW,
    oversample: float = 3.0,
) -> Candidate:
    """Blind BLS search. Returns the strongest periodic dip with periodogram attached."""
    t, y = _bin_uniform(time, flat, cfg.bls_max_points)
    periods = period_grid(float(time[-1] - time[0]), cfg, oversample)
    durations = DURATIONS[DURATIONS < 0.5 * periods.min()]
    dy = np.full_like(y, sigma)
    bls = BoxLeastSquares(t, y, dy=dy)
    res = bls.power(periods, durations, objective="likelihood", oversample=5)

    power = np.asarray(res.power, dtype=float)
    win = int(max(101, (len(power) // 50) | 1))
    resid = power - uniform_filter1d(power, size=win, mode="nearest")
    mad = 1.4826 * np.median(np.abs(resid - np.median(resid))) or 1.0
    sde_curve = (resid - np.median(resid)) / mad
    i = int(np.argmax(sde_curve))

    cand = Candidate(
        period=float(periods[i]),
        t0=float(res.transit_time[i]),
        duration=float(res.duration[i]),
        depth=float(res.depth[i]),
        depth_err=float(res.depth_err[i]),
        snr=float(res.depth_snr[i]),
        sde=float(sde_curve[i]),
        grid_period=periods,
        grid_sde=sde_curve.astype(np.float32),
    )
    return _fill_stats(cand, BoxLeastSquares(time, flat, dy=np.full_like(flat, sigma)))


def refine(
    time: np.ndarray, flat: np.ndarray, sigma: float, cand: Candidate, cfg: ViewConfig = DEFAULT_VIEW
) -> Candidate:
    """Narrow BLS around the detected peak on the re-detrended curve."""
    p = cand.period
    periods = np.linspace(p * 0.99, p * 1.01, 801)
    durations = np.clip(cand.duration * np.array([0.6, 0.8, 1.0, 1.25, 1.6]), 0.02, 0.45 * p)
    bls = BoxLeastSquares(time, flat, dy=np.full_like(flat, sigma))
    res = bls.power(periods, durations, objective="likelihood", oversample=10)
    i = int(np.argmax(res.power))
    out = Candidate(
        period=float(periods[i]),
        t0=float(res.transit_time[i]),
        duration=float(res.duration[i]),
        depth=float(res.depth[i]),
        depth_err=float(res.depth_err[i]),
        snr=float(res.depth_snr[i]),
        sde=cand.sde,
        grid_period=cand.grid_period,
        grid_sde=cand.grid_sde,
    )
    return _fill_stats(out, bls)


def from_ephemeris(
    time: np.ndarray, flat: np.ndarray, sigma: float, period: float, t0: float, duration: float
) -> Candidate:
    """Evaluate a catalogue ephemeris (e.g. a KOI) instead of searching blindly."""
    bls = BoxLeastSquares(time, flat, dy=np.full_like(flat, sigma))
    model = bls.model(time, period, duration, t0)
    in_tr = model < 1.0
    depth = float(1.0 - np.mean(flat[in_tr])) if in_tr.any() else 0.0
    err = float(sigma / np.sqrt(max(int(in_tr.sum()), 1)))
    cand = Candidate(
        period=float(period),
        t0=float(t0),
        duration=float(duration),
        depth=depth,
        depth_err=err,
        snr=depth / err if err > 0 else 0.0,
        sde=0.0,
    )
    return _fill_stats(cand, bls)


def feature_dict(c: Candidate, sigma: float) -> dict[str, float]:
    depth = max(c.depth, 1e-7)
    if np.isfinite(c.odd_depth) and np.isfinite(c.even_depth):
        err = np.hypot(c.odd_err if np.isfinite(c.odd_err) else 0.0,
                       c.even_err if np.isfinite(c.even_err) else 0.0)
        odd_even = abs(c.odd_depth - c.even_depth) / err if err > 0 else 0.0
    else:
        odd_even = 0.0
    sec = c.secondary_depth if np.isfinite(c.secondary_depth) else 0.0
    return {
        "log_period": float(np.log10(c.period)),
        "log_duration_hr": float(np.log10(max(c.duration * 24.0, 1e-3))),
        "duty_cycle": float(c.duration / c.period),
        "log_depth_ppm": float(np.log10(depth * 1e6)),
        "log_snr": float(np.log10(1.0 + max(c.snr, 0.0))),
        "odd_even_sigma": float(min(odd_even, 30.0)),
        "secondary_ratio": float(np.clip(sec / depth, -1.0, 2.0)),
        "log_n_transits": float(np.log10(1.0 + c.n_transits)),
        "log_noise_ppm": float(np.log10(max(sigma, 1e-7) * 1e6)),
    }


def feature_vector(c: Candidate, sigma: float) -> np.ndarray:
    d = feature_dict(c, sigma)
    return np.array([d[k] for k in AUX_FEATURES], dtype=np.float32)
