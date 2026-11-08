"""Training: star-grouped split, gradient-boosting baseline, CNN, evaluation, artifacts."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from ..config import (
    AUX_FEATURES,
    BASELINE_PATH,
    CNN_PATH,
    DATASET_PATH,
    DEFAULT_VIEW,
    METRICS_PATH,
)
from ..data.build import load_dataset
from . import baseline, metrics


def group_split(target_ids, labels, val_frac=0.15, test_frac=0.15, seed=0):
    """Split by star so one star's several candidates never straddle train and test."""
    groups = np.array([str(t).split("|")[-1] for t in target_ids])
    uniq, inv = np.unique(groups, return_inverse=True)
    glabel = np.zeros(len(uniq), dtype=int)
    np.maximum.at(glabel, inv, np.asarray(labels).astype(int))
    g_idx = np.arange(len(uniq))
    g_train, g_rest = train_test_split(
        g_idx, test_size=val_frac + test_frac, stratify=glabel, random_state=seed
    )
    g_val, g_test = train_test_split(
        g_rest, test_size=test_frac / (val_frac + test_frac), stratify=glabel[g_rest], random_state=seed
    )
    return tuple(np.flatnonzero(np.isin(inv, g)) for g in (g_train, g_val, g_test))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def train_all(
    dataset_path: str | Path = DATASET_PATH,
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 2e-3,
    patience: int = 8,
    seed: int = 0,
    use_cnn: bool = True,
    log=print,
) -> dict:
    ds = load_dataset(dataset_path)
    y = ds["labels"].astype(int)
    tr, va, te = group_split(ds["target_ids"], y, seed=seed)
    log(f"dataset: {len(y)} examples ({y.sum()} planets) | train {len(tr)} val {len(va)} test {len(te)}")

    report: dict = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "path": str(dataset_path),
            "n": int(len(y)),
            "positives": int(y.sum()),
            "splits": {"train": len(tr), "val": len(va), "test": len(te)},
            "kinds": {str(k): int((ds["kinds"] == k).sum()) for k in np.unique(ds["kinds"])},
        },
        "view_config": DEFAULT_VIEW.to_dict(),
        "aux_features": list(AUX_FEATURES),
    }

    # ---- baseline -------------------------------------------------------------------------
    x_all = baseline.feature_matrix(ds["aux"], ds["local_views"], ds["global_views"])
    clf = baseline.fit(x_all[tr], y[tr], seed=seed)
    p_val = clf.predict_proba(x_all[va])[:, 1]
    thr_b = metrics.best_f1_threshold(y[va], p_val)
    p_te_b = clf.predict_proba(x_all[te])[:, 1]
    report["baseline"] = {
        "val": metrics.evaluate(y[va], p_val, thr_b),
        "test": metrics.evaluate(y[te], p_te_b, thr_b),
        "by_kind": metrics.per_kind(ds["kinds"][te], p_te_b, thr_b),
    }
    baseline.save(BASELINE_PATH, clf, thr_b, report["baseline"]["test"])
    log(f"baseline  test PR-AUC {report['baseline']['test']['pr_auc']:.3f}  "
        f"recall {report['baseline']['test']['recall']:.3f}  precision {report['baseline']['test']['precision']:.3f}")

    # ---- CNN ------------------------------------------------------------------------------
    if use_cnn:
        try:
            import torch  # noqa: F401
        except ImportError:
            log("torch not installed: skipping CNN (baseline only)")
            use_cnn = False
    if use_cnn:
        p_cnn_val, p_cnn_te, info = _train_cnn(ds, y, tr, va, te, epochs, batch_size, lr, patience, seed, log)
        thr_c = metrics.best_f1_threshold(y[va], p_cnn_val)
        report["cnn"] = {
            "val": metrics.evaluate(y[va], p_cnn_val, thr_c),
            "test": metrics.evaluate(y[te], p_cnn_te, thr_c),
            "by_kind": metrics.per_kind(ds["kinds"][te], p_cnn_te, thr_c),
            "training": info["history"],
            "device": info["device"],
            "seconds": info["seconds"],
        }
        _patch_threshold(CNN_PATH, thr_c, report["cnn"]["test"])
        ens_val = 0.5 * (p_cnn_val + p_val)
        ens_te = 0.5 * (p_cnn_te + p_te_b)
        thr_e = metrics.best_f1_threshold(y[va], ens_val)
        report["ensemble"] = {"test": metrics.evaluate(y[te], ens_te, thr_e)}
        log(f"cnn       test PR-AUC {report['cnn']['test']['pr_auc']:.3f}  "
            f"recall {report['cnn']['test']['recall']:.3f}  precision {report['cnn']['test']['precision']:.3f}")
        log(f"ensemble  test PR-AUC {report['ensemble']['test']['pr_auc']:.3f}")

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(report, indent=2))
    return report


def _patch_threshold(path, threshold, test_metrics):
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ckpt["threshold"] = float(threshold)
    ckpt["test_metrics"] = test_metrics
    torch.save(ckpt, path)


def _train_cnn(ds, y, tr, va, te, epochs, batch_size, lr, patience, seed, log):
    import torch
    from torch import nn

    from . import cnn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    aux = ds["aux"]
    mean, std = aux[tr].mean(0), aux[tr].std(0)
    aux_s = cnn.scale_aux(aux, mean, std)

    def tensors(idx):
        return (
            torch.from_numpy(ds["global_views"][idx]),
            torch.from_numpy(ds["local_views"][idx]),
            torch.from_numpy(aux_s[idx]),
            torch.from_numpy(y[idx].astype(np.float32)),
        )

    g_tr, l_tr, a_tr, y_tr = tensors(tr)
    val_t, test_t = tensors(va), tensors(te)

    model = cnn.ExoNet().to(device)
    pos_weight = torch.tensor((y_tr == 0).sum().item() / max((y_tr == 1).sum().item(), 1), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    steps = int(np.ceil(len(tr) / batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps)

    def predict(t):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(t[0]), 256):
                out.append(model(*(x[i : i + 256].to(device) for x in t[:3])).cpu())
        return _sigmoid(torch.cat(out).numpy())

    best, best_state, bad, history = -1.0, None, 0, []
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.from_numpy(rng.permutation(len(tr)))
        total = 0.0
        for i in range(0, len(perm), batch_size):
            b = perm[i : i + batch_size]
            if len(b) < 2:
                continue
            g, loc, a, yb = g_tr[b].clone(), l_tr[b].clone(), a_tr[b], y_tr[b]
            flip = torch.rand(len(b)) < 0.5  # transits are symmetric about phase 0
            g[flip], loc[flip] = g[flip].flip(-1), loc[flip].flip(-1)
            g += 0.03 * torch.randn_like(g)
            loc += 0.03 * torch.randn_like(loc)
            opt.zero_grad()
            loss = loss_fn(model(g.to(device), loc.to(device), a.to(device)), yb.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            total += loss.item() * len(b)
        p_val = predict(val_t)
        score = metrics.evaluate(y[va], p_val)["pr_auc"] or 0.0
        history.append({"epoch": epoch, "train_loss": total / len(tr), "val_pr_auc": score})
        log(f"epoch {epoch:3d}/{epochs}  loss {total / len(tr):.4f}  val PR-AUC {score:.4f}")
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                log("early stopping")
                break

    model.load_state_dict(best_state)
    cnn.save_checkpoint(
        CNN_PATH,
        model,
        aux_mean=mean,
        aux_std=std,
        threshold=0.5,
        view_config=DEFAULT_VIEW.to_dict(),
        aux_names=list(AUX_FEATURES),
    )
    info = {"history": history, "device": device, "seconds": round(time.time() - t_start, 1)}
    return predict(val_t), predict(test_t), info
