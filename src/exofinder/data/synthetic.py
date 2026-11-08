"""Synthetic light curve generator.

Produces Kepler/TESS-like 30-minute photometry with stellar variability, red and
white noise, flares and gaps, then injects one of several signal types:

* ``planet``           U-shaped transit, no secondary eclipse (label 1)
* ``eclipsing_binary`` deep, V-ish primary plus a secondary eclipse (label 0)
* ``grazing_eb``       shallow V-shaped dips that mimic planets (label 0)
* ``systematic``       sharp-edged periodic dropouts with unstable depth (label 0)
* ``noise``            no periodic signal; BLS locks onto noise or variability (label 0)

It exists so the whole stack runs, trains and demos without internet access.
Real Kepler/TESS data goes through the same pipeline (see ``download.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d

KINDS = ("planet", "eclipsing_binary", "grazing_eb", "systematic", "noise")
FALSE_POSITIVE_WEIGHTS = {
    "eclipsing_binary": 0.35,
    "grazing_eb": 0.25,
    "systematic": 0.20,
    "noise": 0.20,
}
CADENCE = 30.0 / 1440.0  # days


@dataclass
class SyntheticCurve:
    time: np.ndarray
    flux: np.ndarray
    kind: str
    label: int
    truth: dict = field(default_factory=dict)


def _time_grid(rng: np.random.Generator) -> np.ndarray:
    baseline = rng.uniform(60.0, 120.0)
    t = np.arange(0.0, baseline, CADENCE)
    for _ in range(int(rng.integers(1, 4))):
        start = rng.uniform(0.1, 0.9) * baseline
        t = t[(t < start) | (t > start + rng.uniform(0.5, 2.0))]
    return t[rng.random(t.size) > 0.02]


def _stellar(t: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    n = t.size
    sigma_w = 10 ** rng.uniform(np.log10(150), np.log10(1200)) * 1e-6
    flux = np.ones(n)
    for _ in range(int(rng.integers(1, 3))):  # rotational modulation
        p = rng.uniform(1.0, 20.0)
        amp = 10 ** rng.uniform(2.0, 3.5) * 1e-6 * (0.15 if p < 3.0 else 1.0)
        flux += amp * np.sin(2 * np.pi * t / p + rng.uniform(0, 2 * np.pi))
    red = gaussian_filter1d(rng.normal(size=n), sigma=rng.uniform(10, 60))
    flux += red * (10 ** rng.uniform(2.0, 2.7) * 1e-6) / (red.std() + 1e-12)
    flux += rng.normal(0.0, 3e-4) * ((t - t.mean()) / max(np.ptp(t), 1.0)) ** 2  # slow drift
    for _ in range(int(rng.integers(0, 6))):  # flares: fast rise, exponential decay
        i = int(rng.integers(0, n))
        amp = 10 ** rng.uniform(3.0, 4.0) * 1e-6
        flux += amp * np.exp(-np.clip(t - t[i], 0, None) / 0.03) * (t >= t[i])
    flux += rng.normal(0.0, sigma_w, n)
    return flux, sigma_w


def _trapezoid(dt: np.ndarray, t14: float, tau: float, depth: float, curvature: float = 0.0):
    a = np.abs(dt)
    half = t14 / 2.0
    ramp = np.clip((half - a) / np.clip(tau, 1e-4, half), 0.0, 1.0)
    limb = 1.0 + curvature * (1.0 - np.clip(a / half, 0.0, 1.0) ** 2)
    return depth * ramp * limb


def _duration(period: float, rng: np.random.Generator) -> float:
    return 13.0 / 24.0 * (period / 365.0) ** (1.0 / 3.0) * rng.uniform(0.5, 1.25)


def _depth_for_snr(sigma: float, n_in: int, snr: float, cap: float) -> float:
    return float(min(snr * sigma / np.sqrt(max(n_in, 1)), cap))


def _inject(kind: str, t: np.ndarray, sigma: float, rng: np.random.Generator):
    """Return (dip fraction array, truth dict) for one signal type."""
    base = t[-1] - t[0]
    for _ in range(50):
        period = float(np.exp(rng.uniform(np.log(0.7), np.log(base / 2.5))))
        t14 = float(min(_duration(period, rng), 0.3 * period, 0.4))
        t0 = float(t[0] + rng.uniform(0.0, period))
        phase = ((t - t0 + period / 2) % period) - period / 2
        n_in = int(np.sum(np.abs(phase) < t14 / 2))
        n_events = len(np.unique(np.round((t[np.abs(phase) < t14 / 2] - t0) / period)))
        if n_events >= 2 and n_in >= 4:
            break
    snr = float(np.exp(rng.uniform(np.log(8.0), np.log(60.0))))
    truth = {"period": period, "t0": t0, "duration": t14}

    if kind == "planet":
        depth = _depth_for_snr(sigma, n_in, snr, 0.03)
        tau = t14 * rng.uniform(0.03, 0.2)
        dip = _trapezoid(phase, t14, tau, depth, curvature=0.18)
        truth["depth"] = depth
    elif kind == "grazing_eb":
        depth = _depth_for_snr(sigma, n_in, snr, 0.04)
        dip = _trapezoid(phase, t14, t14 * rng.uniform(0.42, 0.5), depth)
        truth["depth"] = depth
    elif kind == "eclipsing_binary":
        depth = float(np.exp(rng.uniform(np.log(0.005), np.log(0.3))))
        sec = depth * rng.uniform(0.15, 1.0)
        tau = t14 * rng.uniform(0.25, 0.5)
        dip = _trapezoid(phase, t14, tau, depth)
        sec_phase = ((t - t0) % period) - period / 2
        dip = dip + _trapezoid(sec_phase, t14, tau, sec)
        truth["depth"] = depth
    elif kind == "systematic":
        depth = _depth_for_snr(sigma, n_in, snr, 0.02)
        t14 = float(rng.uniform(0.04, 0.25))
        k = np.round((t - t0) / period).astype(int)
        scale = rng.uniform(0.3, 1.7, size=k.max() - k.min() + 1)[k - k.min()]
        dip = _trapezoid(phase, t14, t14 * 0.02, depth) * scale
        truth.update(depth=depth, duration=t14)
    else:  # noise
        dip = np.zeros_like(t)
        truth = {"period": 0.0, "t0": 0.0, "duration": 0.0, "depth": 0.0}
    return dip, truth


def make_curve(kind: str, seed: int) -> SyntheticCurve:
    if kind not in KINDS:
        raise ValueError(f"Unknown kind {kind!r}; choose from {KINDS}")
    rng = np.random.default_rng(seed)
    t = _time_grid(rng)
    flux, sigma = _stellar(t, rng)
    dip, truth = _inject(kind, t, sigma, rng)
    truth["noise_ppm"] = sigma * 1e6
    return SyntheticCurve(
        time=t,
        flux=flux * (1.0 - dip),
        kind=kind,
        label=int(kind == "planet"),
        truth=truth,
    )


def sample_kinds(n: int, positive_fraction: float, rng: np.random.Generator) -> list[str]:
    names = list(FALSE_POSITIVE_WEIGHTS)
    probs = np.array(list(FALSE_POSITIVE_WEIGHTS.values()))
    out = []
    for _ in range(n):
        if rng.random() < positive_fraction:
            out.append("planet")
        else:
            out.append(str(rng.choice(names, p=probs / probs.sum())))
    return out
