"""Download Kepler / TESS light curves from MAST with lightkurve and cache them as .npz."""

from __future__ import annotations

import time as _time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PREFIX = {"kepler": "kic", "tess": "tic"}


def raw_path(raw_dir: str | Path, mission: str, target_id: int) -> Path:
    return Path(raw_dir) / f"{PREFIX[mission]}{int(target_id)}.npz"


def fetch_lightcurve(target_id: int, mission: str = "kepler"):
    """Return (time, flux, flux_err) for one star: all quarters/sectors, PDCSAP, stitched."""
    try:
        import lightkurve as lk
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("lightkurve is required: pip install 'exoplanet-finder[data]'") from e

    if mission == "kepler":
        search = lk.search_lightcurve(
            f"KIC {int(target_id)}", mission="Kepler", author="Kepler", cadence="long"
        )
    elif mission == "tess":
        search = lk.search_lightcurve(
            f"TIC {int(target_id)}", mission="TESS", author="SPOC", exptime=120
        )
    else:
        raise ValueError("mission must be 'kepler' or 'tess'")
    if len(search) == 0:
        raise LookupError(f"No {mission} light curves found for target {target_id}")

    collection = search.download_all(flux_column="pdcsap_flux")
    if collection is None or len(collection) == 0:
        raise LookupError(f"Download returned nothing for target {target_id}")
    lc = collection.stitch().remove_nans()
    if mission == "tess":  # 2-minute cadence -> 10 minutes keeps arrays manageable
        import astropy.units as u

        lc = lc.bin(time_bin_size=10 * u.min).remove_nans()
    return (
        np.asarray(lc.time.value, dtype=float),
        np.asarray(lc.flux.value, dtype=float),
        np.asarray(lc.flux_err.value, dtype=float),
    )


def download_target(target_id: int, mission: str, raw_dir: str | Path, attempts: int = 3):
    """Download one target to ``raw_dir``. Returns the path, or None if it failed."""
    path = raw_path(raw_dir, mission, target_id)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    for i in range(attempts):
        try:
            t, f, e = fetch_lightcurve(target_id, mission)
            np.savez_compressed(path, time=t, flux=f, flux_err=e)
            return path
        except LookupError:
            return None  # no data exists; retrying will not help
        except Exception:
            _time.sleep(2.0 * (i + 1))
    return None


def download_many(
    target_ids, mission: str, raw_dir: str | Path, workers: int = 4, progress: bool = True
) -> dict[int, Path | None]:
    from tqdm import tqdm

    ids = [int(x) for x in dict.fromkeys(target_ids)]
    results: dict[int, Path | None] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_target, i, mission, raw_dir): i for i in ids}
        for fut, tid in tqdm(futures.items(), total=len(ids), disable=not progress):
            results[tid] = fut.result()
    return results
