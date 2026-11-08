"""Central configuration: paths and the constants every stage must agree on."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(os.environ.get("EXOFINDER_HOME", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = ROOT / "artifacts"

DATASET_PATH = PROCESSED_DIR / "dataset.npz"
LABELS_PATH = DATA_DIR / "labels.csv"
CNN_PATH = ARTIFACT_DIR / "model.pt"
BASELINE_PATH = ARTIFACT_DIR / "baseline.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

# Order matters: the model, baseline and API all index features by position.
AUX_FEATURES: tuple[str, ...] = (
    "log_period",
    "log_duration_hr",
    "duty_cycle",
    "log_depth_ppm",
    "log_snr",
    "odd_even_sigma",
    "secondary_ratio",
    "log_n_transits",
    "log_noise_ppm",
)


@dataclass(frozen=True)
class ViewConfig:
    """How a light curve becomes model input. Changing this invalidates trained models."""

    global_bins: int = 2001
    local_bins: int = 201
    local_half_width_durations: float = 3.0  # local view spans +/- this many durations
    detrend_window_days: float = 2.0
    clip_sigma_upper: float = 5.0
    bls_min_period: float = 0.6
    bls_max_period: float = 100.0
    bls_max_points: int = 20000

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_VIEW = ViewConfig()
