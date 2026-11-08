"""Fetch and standardise NASA Exoplanet Archive disposition tables (Kepler KOI, TESS TOI).

Both are reduced to one schema:

    target_id, name, label, period, t0, duration_days, depth_ppm, mission

``label`` is 1 for confirmed planets, 0 for false positives, -1 for unresolved
candidates (kept out of training, useful for scoring later).
``t0`` is in the light curve's own time system: BKJD for Kepler, BTJD for TESS.
"""

from __future__ import annotations

import io
from pathlib import Path

import httpx
import pandas as pd

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
KOI_QUERY = (
    "select kepid,kepoi_name,koi_disposition,koi_period,koi_time0bk,koi_duration,koi_depth "
    "from cumulative"
)
TOI_QUERY = (
    "select tid,toi,tfopwg_disp,pl_orbper,pl_tranmid,pl_trandurh,pl_trandep from toi"
)
SCHEMA = ["target_id", "name", "label", "period", "t0", "duration_days", "depth_ppm", "mission"]


def _tap(query: str) -> pd.DataFrame:
    r = httpx.get(
        TAP_URL, params={"query": query, "format": "csv"}, timeout=120, follow_redirects=True
    )
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def standardize_koi(df: pd.DataFrame) -> pd.DataFrame:
    label = df["koi_disposition"].map({"CONFIRMED": 1, "FALSE POSITIVE": 0, "CANDIDATE": -1})
    out = pd.DataFrame(
        {
            "target_id": df["kepid"].astype("int64"),
            "name": df["kepoi_name"],
            "label": label,
            "period": df["koi_period"],
            "t0": df["koi_time0bk"],
            "duration_days": df["koi_duration"] / 24.0,  # archive gives hours
            "depth_ppm": df["koi_depth"],
            "mission": "kepler",
        }
    )
    return out.dropna(subset=["label", "period", "t0", "duration_days"]).astype({"label": int})[SCHEMA]


def standardize_toi(df: pd.DataFrame) -> pd.DataFrame:
    label = df["tfopwg_disp"].map(
        {"CP": 1, "KP": 1, "FP": 0, "FA": 0, "PC": -1, "APC": -1}
    )
    out = pd.DataFrame(
        {
            "target_id": df["tid"].astype("int64"),
            "name": df["toi"].astype(str),
            "label": label,
            "period": df["pl_orbper"],
            "t0": df["pl_tranmid"] - 2457000.0,  # BJD -> BTJD
            "duration_days": df["pl_trandurh"] / 24.0,
            "depth_ppm": df["pl_trandep"],
            "mission": "tess",
        }
    )
    return out.dropna(subset=["label", "period", "t0", "duration_days"]).astype({"label": int})[SCHEMA]


def fetch_labels(mission: str = "kepler", out_path: str | Path | None = None) -> pd.DataFrame:
    if mission == "kepler":
        df = standardize_koi(_tap(KOI_QUERY))
    elif mission == "tess":
        df = standardize_toi(_tap(TOI_QUERY))
    else:
        raise ValueError("mission must be 'kepler' or 'tess'")
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
    return df
