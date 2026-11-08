"""Loads trained models and scores an ``Analysis``. Used by the API and the CLI."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .config import BASELINE_PATH, CNN_PATH, DEFAULT_VIEW, METRICS_PATH
from .features.pipeline import Analysis
from .models import baseline as baseline_model

log = logging.getLogger("exofinder")


class ModelNotAvailable(RuntimeError):
    pass


class Predictor:
    def __init__(self, cnn_path=CNN_PATH, baseline_path=BASELINE_PATH, metrics_path=METRICS_PATH):
        self.cnn_path, self.baseline_path, self.metrics_path = map(Path, (cnn_path, baseline_path, metrics_path))
        self.cnn = None
        self.cnn_meta: dict = {}
        self.baseline: dict | None = None
        self.reload()

    # ------------------------------------------------------------------ loading
    def reload(self) -> None:
        self.cnn, self.cnn_meta, self.baseline = None, {}, None
        if self.cnn_path.exists():
            try:
                from .models import cnn

                self.cnn, self.cnn_meta = cnn.load_checkpoint(self.cnn_path)
                if self.cnn_meta.get("view_config") != DEFAULT_VIEW.to_dict():
                    log.warning("CNN was trained with a different ViewConfig; results may be unreliable.")
            except ImportError:
                log.warning("torch not installed; CNN checkpoint ignored.")
        if self.baseline_path.exists():
            self.baseline = baseline_model.load(self.baseline_path)

    @property
    def models(self) -> list[str]:
        out = [m for m, ok in (("cnn", self.cnn is not None), ("baseline", self.baseline is not None)) if ok]
        if len(out) == 2:
            out.append("ensemble")
        return out

    def info(self) -> dict:
        metrics = json.loads(self.metrics_path.read_text()) if self.metrics_path.exists() else None
        return {
            "models": self.models,
            "default": self._resolve("auto") if self.models else None,
            "thresholds": {
                "cnn": self.cnn_meta.get("threshold"),
                "baseline": self.baseline["threshold"] if self.baseline else None,
            },
            "metrics": metrics,
        }

    def _resolve(self, which: str) -> str:
        if not self.models:
            raise ModelNotAvailable("No trained model found. Run `exofinder train` first.")
        if which == "auto":
            return "cnn" if self.cnn is not None else "baseline"
        if which not in self.models:
            raise ModelNotAvailable(f"Model {which!r} is not available; have {self.models}.")
        return which

    # --------------------------------------------------------------- inference
    def _cnn_probs(self, g: np.ndarray, loc: np.ndarray, aux: np.ndarray) -> np.ndarray:
        import torch

        from .models.cnn import scale_aux

        aux_s = scale_aux(aux, self.cnn_meta["aux_mean"], self.cnn_meta["aux_std"])
        with torch.no_grad():
            logits = self.cnn(
                torch.from_numpy(np.ascontiguousarray(g, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(loc, dtype=np.float32)),
                torch.from_numpy(aux_s),
            )
        return torch.sigmoid(logits).numpy()

    def _score(self, a: Analysis, model: str) -> tuple[float, float]:
        g, loc, aux = a.global_view[None], a.local_view[None], a.aux[None]
        if model == "cnn":
            return float(self._cnn_probs(g, loc, aux)[0]), float(self.cnn_meta.get("threshold", 0.5))
        p = float(baseline_model.predict_proba(self.baseline, aux, loc, g)[0])
        return p, float(self.baseline["threshold"])

    def predict(self, a: Analysis, which: str = "auto") -> dict:
        model = self._resolve(which)
        per_model = {}
        for m in ("cnn", "baseline"):
            if m in self.models:
                s, t = self._score(a, m)
                per_model[m] = {"score": s, "threshold": t}
        if model == "ensemble":
            score = float(np.mean([v["score"] for v in per_model.values()]))
            thr = float(np.mean([v["threshold"] for v in per_model.values()]))
        else:
            score, thr = per_model[model]["score"], per_model[model]["threshold"]
        return {
            "model": model,
            "score": score,
            "threshold": thr,
            "planet_like": bool(score >= thr),
            "per_model": per_model,
        }

    def attention(self, a: Analysis) -> dict | None:
        """Occlusion sensitivity: how much the CNN score drops when a window of the view is erased."""
        if self.cnn is None:
            return None
        g, loc, aux = a.global_view, a.local_view, a.aux

        def occlude(view_name: str, win: int, stride: int) -> list[float]:
            n = len(g) if view_name == "global" else len(loc)
            starts = list(range(0, n - win + 1, stride))
            if starts[-1] != n - win:
                starts.append(n - win)
            k = len(starts)
            G = np.repeat(g[None], k + 1, 0)
            L = np.repeat(loc[None], k + 1, 0)
            A = np.repeat(aux[None], k + 1, 0)
            for i, s in enumerate(starts):
                (G if view_name == "global" else L)[i, s : s + win] = 0.0
            p = self._cnn_probs(G, L, A)
            base = p[-1]
            imp, cnt = np.zeros(n), np.zeros(n)
            for i, s in enumerate(starts):
                imp[s : s + win] += base - p[i]
                cnt[s : s + win] += 1
            return (imp / np.maximum(cnt, 1)).round(4).tolist()

        return {"local": occlude("local", 15, 4), "global": occlude("global", 81, 20)}
