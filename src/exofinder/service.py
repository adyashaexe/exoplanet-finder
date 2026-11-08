"""Glue between pipeline, models and the API: parse inputs, run, shape the JSON the dashboard uses."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from .data import synthetic
from .features import pipeline, preprocess
from .predictor import Predictor

MAX_LC_POINTS = 5000
MAX_PERIODOGRAM_POINTS = 1500
R_SUN_IN_R_EARTH = 109.1


class InputError(ValueError):
    """The caller's data cannot be analysed; the message is safe to show to them."""


# --------------------------------------------------------------------------- input parsing
def parse_table(content: bytes, filename: str = "") -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """CSV/TXT (time, flux[, flux_err]) or a Kepler/TESS light-curve FITS file."""
    name = filename.lower()
    if name.endswith((".fits", ".fit", ".fits.gz")):
        return _parse_fits(content)
    try:
        df = pd.read_csv(io.BytesIO(content), sep=None, engine="python", comment="#")
    except Exception as e:
        raise InputError(f"Could not read the file as a table: {e}") from e
    cols = {str(c).strip().lower(): c for c in df.columns}
    t_key = next((cols[k] for k in ("time", "t", "bjd", "btjd", "bkjd", "jd") if k in cols), None)
    f_key = next(
        (cols[k] for k in ("flux", "pdcsap_flux", "sap_flux", "f", "normalized_flux") if k in cols), None
    )
    if t_key is None or f_key is None:
        # Headerless file: treat columns as time, flux[, flux_err].
        try:
            df = pd.read_csv(io.BytesIO(content), sep=None, engine="python", comment="#", header=None)
            df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all", axis=1)
        except Exception as e:
            raise InputError(f"Could not read the file as a table: {e}") from e
        if df.shape[1] < 2 or df.iloc[:, :2].isna().all().any():
            raise InputError("Need columns named 'time' and 'flux' (optional: 'flux_err').")
        cols = {}
        t_key, f_key = df.columns[:2]
        e_key = df.columns[2] if df.shape[1] > 2 else None
    else:
        e_key = next((cols[k] for k in ("flux_err", "err", "error", "flux_error") if k in cols), None)
    t = pd.to_numeric(df[t_key], errors="coerce").to_numpy(float)
    f = pd.to_numeric(df[f_key], errors="coerce").to_numpy(float)
    e = pd.to_numeric(df[e_key], errors="coerce").to_numpy(float) if e_key is not None else None
    return t, f, e


def _parse_fits(content: bytes):
    from astropy.io import fits

    try:
        with fits.open(io.BytesIO(content)) as hdul:
            data = hdul["LIGHTCURVE"].data if "LIGHTCURVE" in [h.name for h in hdul] else hdul[1].data
            names = {n.upper(): n for n in data.columns.names}
            f_name = next((names[k] for k in ("PDCSAP_FLUX", "SAP_FLUX", "FLUX") if k in names), None)
            if "TIME" not in names or f_name is None:
                raise InputError("FITS file has no TIME / PDCSAP_FLUX columns.")
            e_name = names.get(f_name.upper() + "_ERR") or names.get("FLUX_ERR")
            t = np.array(data[names["TIME"]], dtype=float)
            f = np.array(data[f_name], dtype=float)
            e = np.array(data[e_name], dtype=float) if e_name else None
            return t, f, e
    except InputError:
        raise
    except Exception as e:
        raise InputError(f"Could not read FITS file: {e}") from e


def parse_target(target: str, mission: str) -> int:
    digits = "".join(ch for ch in str(target) if ch.isdigit())
    if not digits:
        raise InputError("Enter a numeric ID such as 'KIC 8462852' or 'TIC 261136679'.")
    return int(digits)


# ------------------------------------------------------------------------------ analysis
def run(
    time,
    flux,
    flux_err,
    predictor: Predictor,
    name: str,
    source: str,
    model: str = "auto",
    ephemeris=None,
    truth: dict | None = None,
) -> dict:
    try:
        a = pipeline.analyze(time, flux, flux_err, ephemeris=ephemeris)
    except ValueError as e:
        raise InputError(str(e)) from e
    prediction = predictor.predict(a, model)
    attention = predictor.attention(a)
    return build_report(a, prediction, attention, name, source, truth)


def demo(predictor: Predictor, kind: str, seed: int, model: str = "auto") -> dict:
    c = synthetic.make_curve(kind, seed)
    truth = {"kind": kind, "is_planet": bool(c.label), **{k: float(v) for k, v in c.truth.items()}}
    return run(c.time, c.flux, None, predictor, f"Synthetic {kind.replace('_', ' ')} #{seed}", "synthetic", model, truth=truth)


# ---------------------------------------------------------------------------- reporting
def _r(x, nd=4):
    return np.round(np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0), nd).tolist()


def _bin_mean(time: np.ndarray, series: list[np.ndarray], max_points: int):
    if len(time) <= max_points:
        return time, series
    width = (time[-1] - time[0]) / max_points
    idx = np.floor((time - time[0]) / width).astype(int)
    _, inv = np.unique(idx, return_inverse=True)
    cnt = np.bincount(inv)
    return np.bincount(inv, weights=time) / cnt, [np.bincount(inv, weights=s) / cnt for s in series]


def _peak_downsample(x: np.ndarray, y: np.ndarray, k: int):
    if len(x) <= k:
        return x, y
    edges = np.linspace(0, len(x), k + 1).astype(int)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        if hi > lo:
            j = lo + int(np.argmax(y[lo:hi]))
            xs.append(x[j])
            ys.append(y[j])
    return np.array(xs), np.array(ys)


def vetting_checks(a: pipeline.Analysis) -> list[dict]:
    """Rule-of-thumb checks astronomers apply to a transit candidate (not model output)."""
    c, f = a.candidate, a.features
    checks = []

    issues = {
        "snr": "a weak signal",
        "repeats": "too few repeats",
        "odd_even": "odd and even dips differ",
        "secondary": "a secondary eclipse",
        "depth": "a very deep dip",
        "duration": "an unusual transit length",
    }

    def add(id_, label, status, detail):
        checks.append({"id": id_, "label": label, "status": status, "detail": detail, "issue": issues[id_]})

    snr = c.snr
    add("snr", "Signal strength", "pass" if snr >= 7.1 else "warn" if snr >= 5 else "fail",
        f"Transit signal-to-noise is {snr:.1f}; 7.1 or more is the usual detection bar.")

    n = c.n_transits
    add("repeats", "Repeats", "pass" if n >= 3 else "warn" if n == 2 else "fail",
        f"{n} transit{'s' if n != 1 else ''} fall inside the data; three or more make a convincing periodic signal.")

    if np.isfinite(c.odd_depth) and np.isfinite(c.even_depth):
        s = f["odd_even_sigma"]
        add("odd_even", "Odd and even dips match", "pass" if s < 3 else "warn" if s < 5 else "fail",
            f"Odd dips are {c.odd_depth * 1e6:,.0f} ppm deep and even dips {c.even_depth * 1e6:,.0f} ppm ({s:.1f} sigma apart). "
            "Different depths suggest an eclipsing binary at twice the period.")
    else:
        add("odd_even", "Odd and even dips match", "warn", "Not enough transits to compare odd and even dips.")

    sec = c.secondary_depth if np.isfinite(c.secondary_depth) else 0.0
    ratio = sec / c.depth if c.depth > 0 else 0.0
    add("secondary", "No secondary eclipse", "pass" if ratio < 0.25 else "warn" if ratio < 0.5 else "fail",
        f"At phase 0.5 the star dims by {max(sec, 0) * 1e6:,.0f} ppm ({max(ratio, 0) * 100:.0f}% of the main dip). "
        "A real secondary points to a stellar companion.")

    rr = float(np.sqrt(max(c.depth, 0)))
    add("depth", "Planet-sized depth", "pass" if c.depth < 0.03 else "warn" if c.depth < 0.1 else "fail",
        f"Depth {c.depth * 1e6:,.0f} ppm implies a body about {rr:.2f} times the star's radius "
        f"({rr * R_SUN_IN_R_EARTH:,.1f} Earth radii if the star is Sun-like).")

    expected = 13.0 * (c.period / 365.0) ** (1 / 3)
    dur = c.duration * 24
    status = "pass" if 0.2 * expected <= dur <= 1.6 * expected else "warn"
    add("duration", "Transit length fits the orbit", status,
        f"The transit lasts {dur:.1f} h; a Sun-like star with a {c.period:.2f}-day orbit gives at most about {expected:.1f} h.")
    return checks


def build_report(a: pipeline.Analysis, pred: dict, attention: dict | None, name: str, source: str, truth: dict | None) -> dict:
    lc, c = a.lc, a.candidate
    ppm_raw = (lc.flux - 1.0) * 1e6
    ppm_trend = (lc.trend - 1.0) * 1e6
    ppm_flat = (lc.flat - 1.0) * 1e6
    t_ds, (raw_ds, trend_ds, flat_ds) = _bin_mean(lc.time, [ppm_raw, ppm_trend, ppm_flat], MAX_LC_POINTS)

    k0 = int(np.ceil((lc.time[0] - c.t0) / c.period))
    k1 = int(np.floor((lc.time[-1] - c.t0) / c.period))
    tt = c.t0 + c.period * np.arange(k0, k1 + 1)
    tt = [float(x) for x in tt if np.any(np.abs(lc.time - x) < max(c.duration, 0.05))]

    pg_p, pg_s = _peak_downsample(c.grid_period, c.grid_sde, MAX_PERIODOGRAM_POINTS) if len(c.grid_period) else (np.zeros(0), np.zeros(0))

    phase = preprocess.fold(lc.time, c.period, c.t0)
    rng = np.random.default_rng(0)

    def sample(mask: np.ndarray, k: int) -> np.ndarray:
        idx = np.flatnonzero(mask)
        if len(idx) > k:
            idx = np.sort(rng.choice(idx, k, replace=False))
        return idx

    half = min(3.0 * c.duration, c.period / 2)
    gi, li = sample(np.ones_like(phase, bool), 2500), sample(np.abs(phase) <= half, 2500)
    gb = preprocess.bin_median(phase, ppm_flat, -c.period / 2, c.period / 2, 200)
    lb = preprocess.bin_median(phase[np.abs(phase) <= half], ppm_flat[np.abs(phase) <= half], -half, half, 60)
    g_centers = -c.period / 2 + (np.arange(200) + 0.5) * c.period / 200
    l_centers = -half + (np.arange(60) + 0.5) * 2 * half / 60

    n_l, n_g = len(a.local_view), len(a.global_view)
    return {
        "target": {
            "name": name,
            "source": source,
            "points": int(len(lc.time)),
            "baseline_days": round(lc.baseline, 2),
            "cadence_minutes": round(lc.cadence * 1440, 1),
            "noise_ppm": round(lc.sigma * 1e6, 1),
        },
        "verdict": {
            "score": round(pred["score"], 4),
            "threshold": round(pred["threshold"], 4),
            "planet_like": pred["planet_like"],
            "model": pred["model"],
            "per_model": {k: {m: round(x, 4) for m, x in v.items()} for k, v in pred["per_model"].items()},
        },
        "candidate": {
            "period_days": round(c.period, 5),
            "t0": round(c.t0, 4),
            "duration_hours": round(c.duration * 24, 2),
            "depth_ppm": round(c.depth * 1e6, 1),
            "snr": round(c.snr, 2),
            "sde": round(c.sde, 2),
            "transits_observed": c.n_transits,
        },
        "checks": vetting_checks(a),
        "lightcurve": {
            "time": _r(t_ds, 4), "raw": _r(raw_ds, 1), "trend": _r(trend_ds, 1), "flat": _r(flat_ds, 1),
            "transit_times": _r(tt, 4), "duration_days": round(c.duration, 4),
        },
        "periodogram": {"period": _r(pg_p, 4), "sde": _r(pg_s, 2), "best": round(c.period, 5)},
        "fold": {
            "global": {"phase_days": _r(phase[gi], 4), "flux": _r(ppm_flat[gi], 1),
                       "bin_phase_days": _r(g_centers, 4), "bin_flux": _r(gb, 1)},
            "local": {"phase_hours": _r(phase[li] * 24, 3), "flux": _r(ppm_flat[li], 1),
                      "bin_phase_hours": _r(l_centers * 24, 3), "bin_flux": _r(lb, 1)},
        },
        "views": {"global": _r(a.global_view, 3), "local": _r(a.local_view, 3), "local_bins": n_l, "global_bins": n_g},
        "attention": attention,
        "features": {k: round(float(v), 4) for k, v in a.features.items()},
        "truth": truth,
    }
