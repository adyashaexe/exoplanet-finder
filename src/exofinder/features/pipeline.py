"""One entry point: raw light curve in, model-ready analysis out."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import DEFAULT_VIEW, ViewConfig
from . import bls, preprocess
from .preprocess import LightCurve


@dataclass
class Analysis:
    lc: LightCurve
    candidate: bls.Candidate
    global_view: np.ndarray
    local_view: np.ndarray
    aux: np.ndarray  # raw (unscaled) feature vector in config.AUX_FEATURES order
    features: dict[str, float]
    ephemeris_given: bool = False


def analyze(
    time,
    flux,
    flux_err=None,
    cfg: ViewConfig = DEFAULT_VIEW,
    ephemeris: tuple[float, float, float] | None = None,
) -> Analysis:
    """Detrend, find (or take) the transit ephemeris, and build views and features.

    ``ephemeris`` is ``(period_days, t0, duration_days)``. When omitted, a blind
    BLS search is run, which is what the API does for arbitrary uploads.
    """
    lc = preprocess.prepare(time, flux, flux_err, cfg)

    if ephemeris is None:
        cand = bls.search(lc.time, lc.flat, lc.sigma, cfg)
        lc = preprocess.redetrend(lc, cand.period, cand.t0, cand.duration, cfg)
        cand = bls.refine(lc.time, lc.flat, lc.sigma, cand, cfg)
    else:
        period, t0, duration = ephemeris
        lc = preprocess.redetrend(lc, period, t0, duration, cfg)
        cand = bls.from_ephemeris(lc.time, lc.flat, lc.sigma, period, t0, duration)

    g, loc = preprocess.make_views(lc.time, lc.flat, cand.period, cand.t0, cand.duration, cfg)
    return Analysis(
        lc=lc,
        candidate=cand,
        global_view=g,
        local_view=loc,
        aux=bls.feature_vector(cand, lc.sigma),
        features=bls.feature_dict(cand, lc.sigma),
        ephemeris_given=ephemeris is not None,
    )
