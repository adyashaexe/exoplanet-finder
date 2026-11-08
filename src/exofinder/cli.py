"""Command line interface: ``exofinder <command>``."""

from __future__ import annotations

import argparse
import json
import sys

from .config import DATASET_PATH, LABELS_PATH, RAW_DIR


def _cmd_make_synthetic(a):
    from .data.build import build_synthetic

    info = build_synthetic(a.n, a.seed, a.positive_fraction, a.workers, a.out)
    print(json.dumps(info, indent=2))


def _cmd_fetch_labels(a):
    from .data.labels import fetch_labels

    df = fetch_labels(a.mission, a.out)
    print(f"saved {len(df)} rows to {a.out}")
    print(df["label"].value_counts().rename({1: "confirmed", 0: "false positive", -1: "candidate"}))


def _cmd_download(a):
    import pandas as pd

    from .data.download import download_many

    df = pd.read_csv(a.labels)
    df = df[(df["mission"] == a.mission) & (df["label"] >= 0)]
    if a.balanced:  # equal planets and false positives, sampled reproducibly
        n = a.limit // 2 if a.limit else df["label"].value_counts().min()
        df = pd.concat([g.sample(min(n, len(g)), random_state=0) for _, g in df.groupby("label")])
    elif a.limit:
        df = df.sample(min(a.limit, len(df)), random_state=0)
    res = download_many(df["target_id"].unique(), a.mission, a.raw_dir, a.workers)
    ok = sum(v is not None for v in res.values())
    print(f"downloaded {ok}/{len(res)} targets to {a.raw_dir}")


def _cmd_build_dataset(a):
    from .data.build import build_real

    print(json.dumps(build_real(a.labels, a.mission, a.raw_dir, a.workers, a.out), indent=2))


def _cmd_train(a):
    from .models.train import train_all

    train_all(a.dataset, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, seed=a.seed, use_cnn=not a.baseline_only)


def _cmd_predict(a):
    from . import service
    from .predictor import Predictor

    t, f, e = service.parse_table(open(a.file, "rb").read(), a.file)
    rep = service.run(t, f, e, Predictor(), a.file, "cli", a.model)
    v, c = rep["verdict"], rep["candidate"]
    print(f"{'PLANET-LIKE' if v['planet_like'] else 'LIKELY FALSE POSITIVE'}  score {v['score']:.3f} (threshold {v['threshold']:.3f}, {v['model']})")
    print(f"period {c['period_days']} d | depth {c['depth_ppm']} ppm | duration {c['duration_hours']} h | SNR {c['snr']}")
    for ch in rep["checks"]:
        print(f"  [{ch['status']:4s}] {ch['label']}: {ch['detail']}")
    if a.json:
        json.dump(rep, open(a.json, "w"))
        print(f"full report written to {a.json}")


def _cmd_serve(a):
    import uvicorn

    uvicorn.run("exofinder.api.main:app", host=a.host, port=a.port, reload=a.reload)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="exofinder", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("make-synthetic", help="generate and process synthetic light curves")
    s.add_argument("--n", type=int, default=1500)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--positive-fraction", type=float, default=0.4)
    s.add_argument("--workers", type=int, default=None)
    s.add_argument("--out", default=str(DATASET_PATH))
    s.set_defaults(fn=_cmd_make_synthetic)

    s = sub.add_parser("fetch-labels", help="download KOI/TOI dispositions from the NASA Exoplanet Archive")
    s.add_argument("--mission", choices=["kepler", "tess"], default="kepler")
    s.add_argument("--out", default=str(LABELS_PATH))
    s.set_defaults(fn=_cmd_fetch_labels)

    s = sub.add_parser("download", help="download light curves from MAST for labelled targets")
    s.add_argument("--labels", default=str(LABELS_PATH))
    s.add_argument("--mission", choices=["kepler", "tess"], default="kepler")
    s.add_argument("--raw-dir", default=str(RAW_DIR))
    s.add_argument("--limit", type=int, default=600, help="max targets (0 = all)")
    s.add_argument("--balanced", action="store_true", help="equal confirmed and false-positive counts")
    s.add_argument("--workers", type=int, default=4)
    s.set_defaults(fn=_cmd_download)

    s = sub.add_parser("build-dataset", help="process downloaded light curves into a training set")
    s.add_argument("--labels", default=str(LABELS_PATH))
    s.add_argument("--mission", choices=["kepler", "tess"], default="kepler")
    s.add_argument("--raw-dir", default=str(RAW_DIR))
    s.add_argument("--workers", type=int, default=None)
    s.add_argument("--out", default=str(DATASET_PATH))
    s.set_defaults(fn=_cmd_build_dataset)

    s = sub.add_parser("train", help="train baseline + CNN and write evaluation metrics")
    s.add_argument("--dataset", default=str(DATASET_PATH))
    s.add_argument("--epochs", type=int, default=25)
    s.add_argument("--batch-size", type=int, default=64)
    s.add_argument("--lr", type=float, default=2e-3)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--baseline-only", action="store_true")
    s.set_defaults(fn=_cmd_train)

    s = sub.add_parser("predict", help="score one light curve file (CSV or FITS)")
    s.add_argument("file")
    s.add_argument("--model", choices=["auto", "cnn", "baseline", "ensemble"], default="auto")
    s.add_argument("--json", help="also write the full report as JSON")
    s.set_defaults(fn=_cmd_predict)

    s = sub.add_parser("serve", help="run the API and dashboard")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(fn=_cmd_serve)

    args = p.parse_args(argv)
    if getattr(args, "limit", None) == 0:
        args.limit = None
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
