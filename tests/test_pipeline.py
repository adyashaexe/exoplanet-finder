import numpy as np
import pytest

from exofinder.data import synthetic
from exofinder.features import bls, pipeline, preprocess


def test_fold_centres_transit_at_zero():
    t = np.linspace(0, 30, 3000)
    phase = preprocess.fold(t, period=5.0, t0=1.0)
    assert phase.min() >= -2.5 and phase.max() < 2.5
    assert abs(preprocess.fold(np.array([1.0, 6.0, 11.0]), 5.0, 1.0)).max() < 1e-9


def test_bin_median_fills_empty_bins():
    x = np.array([0.1, 0.2, 0.9])
    out = preprocess.bin_median(x, np.array([1.0, 1.0, 3.0]), 0.0, 1.0, 10)
    assert out.shape == (10,) and np.isfinite(out).all()


def test_prepare_rejects_short_curve():
    with pytest.raises(ValueError):
        preprocess.prepare(np.arange(10.0), np.ones(10))


def test_planet_period_recovered():
    c = synthetic.make_curve("planet", 1)
    a = pipeline.analyze(c.time, c.flux)
    ratio = a.candidate.period / c.truth["period"]
    assert abs(ratio - 1.0) < 0.01
    assert a.candidate.snr > 7
    assert a.global_view.shape == (2001,) and a.local_view.shape == (201,)
    assert a.aux.shape == (9,)


def test_eclipsing_binary_has_secondary_or_odd_even_signal():
    c = synthetic.make_curve("eclipsing_binary", 1)
    a = pipeline.analyze(c.time, c.flux)
    assert a.features["odd_even_sigma"] > 3 or a.features["secondary_ratio"] > 0.2


def test_ephemeris_path_matches_blind_search():
    c = synthetic.make_curve("planet", 3)
    t = c.truth
    a = pipeline.analyze(c.time, c.flux, ephemeris=(t["period"], t["t0"], t["duration"]))
    assert a.ephemeris_given and a.candidate.snr > 7


def test_synthetic_labels():
    assert synthetic.make_curve("planet", 2).label == 1
    assert synthetic.make_curve("noise", 2).label == 0
    with pytest.raises(ValueError):
        synthetic.make_curve("comet", 0)
