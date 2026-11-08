"""Light curve cleaning, detrending, phase folding and binning.

Everything downstream (BLS, CNN input views, baseline features) consumes the
``LightCurve`` produced here, so training and serving see identical data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from ..config import DEFAULT_VIEW, ViewConfig


@dataclass
class LightCurve:
    time: np.ndarray  # days (BKJD / BTJD / arbitrary), sorted
    flux: np.ndarray  # normalised so the median is 1.0
    trend: np.ndarray  # slow stellar + instrument trend
    flat: np.ndarray  # flux / trend, centred on 1.0
    sigma: float  # point-to-point scatter of ``flat``

    @property
    def baseline(self) -> float:
        return float(self.time[-1] - self.time[0])

    @property
    def cadence(self) -> float:
        return float(np.median(np.diff(self.time)))


def robust_sigma(flat: np.ndarray) -> float:
    """Point-to-point scatter, insensitive to slow variability and to transits."""
    d = np.diff(flat)
    if d.size == 0:
        return 1e-4
    return float(max(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0), 1e-7))


def _segments(time: np.ndarray, gap_days: float) -> list[np.ndarray]:
    breaks = np.where(np.diff(time) > gap_days)[0] + 1
    return np.split(np.arange(time.size), breaks)


def running_median_trend(
    time: np.ndarray,
    flux: np.ndarray,
    window_days: float,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Running-median trend computed separately on each gap-free segment.

    ``mask`` marks in-transit points; they are replaced by linear interpolation
    before filtering so the trend does not dip into the transit.
    """
    trend = np.empty_like(flux)
    cadence = float(np.median(np.diff(time))) if time.size > 1 else 1.0
    gap = max(0.3, 20.0 * cadence)
    for idx in _segments(time, gap):
        t, f = time[idx], flux[idx].copy()
        if idx.size < 7:
            trend[idx] = np.median(f)
            continue
        dt = float(np.median(np.diff(t)))
        n = int(round(window_days / dt)) | 1
        n = max(3, min(n, idx.size if idx.size % 2 else idx.size - 1))
        if mask is not None:
            m = mask[idx]
            if m.any() and (~m).sum() > 3:
                f[m] = np.interp(t[m], t[~m], f[~m])
        tr = median_filter(f, size=n, mode="nearest")
        smooth = max(3, n // 3)
        trend[idx] = uniform_filter1d(tr, size=smooth, mode="nearest")
    return trend


def prepare(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray | None = None,
    cfg: ViewConfig = DEFAULT_VIEW,
) -> LightCurve:
    """Clean, normalise, detrend and clip a raw light curve."""
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    ok = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[ok], flux[ok]
    if time.size < 50:
        raise ValueError(f"Light curve has only {time.size} valid points; need at least 50.")
    order = np.argsort(time, kind="stable")
    time, flux = time[order], flux[order]
    keep = np.concatenate([[True], np.diff(time) > 0])  # drop duplicate timestamps
    time, flux = time[keep], flux[keep]

    med = float(np.median(flux))
    if not np.isfinite(med) or med == 0.0:
        raise ValueError("Flux median is zero or invalid; cannot normalise.")
    flux = flux / med

    trend = running_median_trend(time, flux, cfg.detrend_window_days)
    flat = flux / trend
    resid = flat - 1.0
    scatter = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    good = resid < cfg.clip_sigma_upper * max(scatter, 1e-7)  # clip flares, keep dips
    time, flux, trend, flat = time[good], flux[good], trend[good], flat[good]

    if time[-1] - time[0] < 2.0 * cfg.bls_min_period:
        raise ValueError("Light curve baseline is too short to search for periodic transits.")
    return LightCurve(time, flux, trend, flat, robust_sigma(flat))


def transit_mask(time: np.ndarray, period: float, t0: float, half_width: float) -> np.ndarray:
    return np.abs(fold(time, period, t0)) <= half_width


def redetrend(
    lc: LightCurve, period: float, t0: float, duration: float, cfg: ViewConfig = DEFAULT_VIEW
) -> LightCurve:
    """Second detrending pass with the detected transits masked out."""
    mask = transit_mask(lc.time, period, t0, 1.0 * duration)
    trend = running_median_trend(lc.time, lc.flux, cfg.detrend_window_days, mask)
    flat = lc.flux / trend
    return LightCurve(lc.time, lc.flux, trend, flat, robust_sigma(flat))


def fold(time: np.ndarray, period: float, t0: float) -> np.ndarray:
    """Phase in days, wrapped to [-period/2, period/2). Transit is at 0."""
    return ((time - t0 + 0.5 * period) % period) - 0.5 * period


def bin_median(x: np.ndarray, y: np.ndarray, lo: float, hi: float, nbins: int) -> np.ndarray:
    """Median of ``y`` in ``nbins`` equal bins over [lo, hi]; empty bins are interpolated."""
    out = np.full(nbins, np.nan)
    if x.size:
        idx = np.floor((x - lo) / (hi - lo) * nbins).astype(int)
        valid = (idx >= 0) & (idx < nbins)
        idx, yy = idx[valid], y[valid]
        if idx.size:
            order = np.argsort(idx, kind="stable")
            idx, yy = idx[order], yy[order]
            starts = np.flatnonzero(np.r_[True, np.diff(idx) > 0])
            for s, chunk in zip(idx[starts], np.split(yy, starts[1:]), strict=True):
                out[s] = np.median(chunk)
    finite = np.isfinite(out)
    if not finite.any():
        return np.zeros(nbins)
    if not finite.all():
        pos = np.arange(nbins)
        out = np.interp(pos, pos[finite], out[finite])
    return out


def normalize_view(v: np.ndarray) -> np.ndarray:
    """Median 0, deepest point about -1 (shape matters; depth lives in the scalar features)."""
    v = v - np.median(v)
    scale = max(-float(np.percentile(v, 1)), 1e-9)
    return np.clip(v / scale, -5.0, 5.0).astype(np.float32)


def make_views(
    time: np.ndarray,
    flat: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    cfg: ViewConfig = DEFAULT_VIEW,
) -> tuple[np.ndarray, np.ndarray]:
    """Global (whole orbit) and local (zoomed on transit) phase-folded views."""
    phase = fold(time, period, t0)
    g = bin_median(phase, flat, -period / 2, period / 2, cfg.global_bins)
    half = min(cfg.local_half_width_durations * duration, period / 2)
    sel = np.abs(phase) <= half
    loc = bin_median(phase[sel], flat[sel], -half, half, cfg.local_bins)
    return normalize_view(g), normalize_view(loc)
