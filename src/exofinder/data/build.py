"""Turn light curves (synthetic or real) into the training dataset (.npz of views + features)."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import AUX_FEATURES, DATASET_PATH, RAW_DIR, DEFAULT_VIEW
from ..features import pipeline
from . import synthetic
from .download import raw_path


def _row(a: pipeline.Analysis, label: int, kind: str, target_id: str, truth_period: float) -> dict:
    return {
        "global_view": a.global_view,
        "local_view": a.local_view,
        "aux": a.aux,
        "label": label,
        "kind": kind,
        "target_id": target_id,
        "true_period": truth_period,
        "bls_period": a.candidate.period,
        "snr": a.candidate.snr,
    }


def _synthetic_job(args) -> dict | None:
    seed, kind = args
    try:
        c = synthetic.make_curve(kind, seed)
        a = pipeline.analyze(c.time, c.flux)
    except Exception:
        return None
    return _row(a, c.label, kind, f"SYN-{seed:08d}", c.truth["period"])


def _real_job(args) -> dict | None:
    path, label, name, target_id, period, t0, duration = args
    try:
        z = np.load(path)
        a = pipeline.analyze(z["time"], z["flux"], z["flux_err"], ephemeris=(period, t0, duration))
    except Exception:
        return None
    return _row(a, int(label), "confirmed" if label == 1 else "false_positive", f"{name}|{target_id}", period)


def _map(fn, jobs, workers: int, progress: bool = True):
    from tqdm import tqdm

    if workers <= 1:
        return [fn(j) for j in tqdm(jobs, disable=not progress)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(tqdm(ex.map(fn, jobs, chunksize=4), total=len(jobs), disable=not progress))


def save_dataset(rows: list[dict], path: str | Path = DATASET_PATH) -> dict:
    rows = [r for r in rows if r is not None]
    if not rows:
        raise RuntimeError("No light curves could be processed; dataset would be empty.")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        global_views=np.stack([r["global_view"] for r in rows]).astype(np.float32),
        local_views=np.stack([r["local_view"] for r in rows]).astype(np.float32),
        aux=np.stack([r["aux"] for r in rows]).astype(np.float32),
        labels=np.array([r["label"] for r in rows], dtype=np.int64),
        kinds=np.array([r["kind"] for r in rows]),
        target_ids=np.array([r["target_id"] for r in rows]),
        true_period=np.array([r["true_period"] for r in rows], dtype=np.float64),
        bls_period=np.array([r["bls_period"] for r in rows], dtype=np.float64),
        snr=np.array([r["snr"] for r in rows], dtype=np.float32),
        aux_names=np.array(AUX_FEATURES),
        view_config=np.array(str(DEFAULT_VIEW.to_dict())),
    )
    labels = np.array([r["label"] for r in rows])
    return {"n": len(rows), "positives": int(labels.sum()), "path": str(path)}


def load_dataset(path: str | Path = DATASET_PATH) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def build_synthetic(
    n: int = 1500,
    seed: int = 0,
    positive_fraction: float = 0.4,
    workers: int | None = None,
    out_path: str | Path = DATASET_PATH,
) -> dict:
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    rng = np.random.default_rng(seed)
    kinds = synthetic.sample_kinds(n, positive_fraction, rng)
    jobs = [(seed * 1_000_003 + i, k) for i, k in enumerate(kinds)]
    rows = _map(_synthetic_job, jobs, workers)
    info = save_dataset(rows, out_path)
    ok = [r for r in rows if r is not None]
    # Fraction of injected periodic signals whose period BLS recovered (or a harmonic of it).
    hits = tot = 0
    for r in ok:
        if r["true_period"] > 0:
            tot += 1
            ratio = r["bls_period"] / r["true_period"]
            hits += any(abs(ratio - h) < 0.02 * h for h in (0.5, 1.0, 2.0))
    info["failed"] = n - len(ok)
    info["bls_recovery"] = round(hits / tot, 3) if tot else None
    return info


def build_real(
    labels_csv: str | Path,
    mission: str = "kepler",
    raw_dir: str | Path = RAW_DIR,
    workers: int | None = None,
    out_path: str | Path = DATASET_PATH,
) -> dict:
    """Build a dataset from downloaded light curves, using each catalogue entry's ephemeris."""
    df = pd.read_csv(labels_csv)
    df = df[(df["label"] >= 0) & (df["mission"] == mission)]
    jobs = []
    for r in df.itertuples():
        p = raw_path(raw_dir, mission, r.target_id)
        if p.exists():
            jobs.append((str(p), r.label, r.name, r.target_id, r.period, r.t0, r.duration_days))
    if not jobs:
        raise RuntimeError(f"No downloaded light curves found in {raw_dir}; run `exofinder download` first.")
    rows = _map(_real_job, jobs, workers or max(1, (os.cpu_count() or 2) - 1))
    info = save_dataset(rows, out_path)
    info["failed"] = len(jobs) - info["n"]
    return info
