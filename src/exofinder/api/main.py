"""FastAPI backend. Serves the JSON API and the dashboard.

    GET  /api/health
    GET  /api/model                       trained models, thresholds, evaluation metrics
    GET  /api/demo?kind=planet&seed=7     synthetic light curve through the full pipeline
    POST /api/analyze/upload              CSV / FITS file (time, flux[, flux_err])
    POST /api/analyze/target              KIC / TIC id fetched from MAST (needs internet)
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__, service
from ..data import synthetic
from ..predictor import ModelNotAvailable, Predictor

log = logging.getLogger("exofinder")
STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 40 * 1024 * 1024

ModelChoice = Literal["auto", "cnn", "baseline", "ensemble"]


class TargetRequest(BaseModel):
    target: str = Field(..., examples=["KIC 8462852"], description="Numeric KIC or TIC id")
    mission: Literal["kepler", "tess"] = "kepler"
    model: ModelChoice = "auto"
    period_days: float | None = Field(None, gt=0, description="Optional known period; skips the blind search")
    t0: float | None = None
    duration_hours: float | None = Field(None, gt=0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = Predictor()
    log.info("models available: %s", app.state.predictor.models)
    yield


app = FastAPI(title="Exoplanet Finder", version=__version__, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _predictor() -> Predictor:
    return app.state.predictor


def _guard(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ModelNotAvailable as e:
        raise HTTPException(503, str(e)) from e
    except service.InputError as e:
        raise HTTPException(422, str(e)) from e


@app.get("/api/health")
def health():
    return {"status": "ok", "version": __version__, "models": _predictor().models}


@app.get("/api/model")
def model_info():
    return _predictor().info()


@app.post("/api/model/reload")
def reload_models():
    _predictor().reload()
    return {"models": _predictor().models}


@app.get("/api/demo/kinds")
def demo_kinds():
    return {"kinds": list(synthetic.KINDS)}


@app.get("/api/demo")
def demo(
    kind: Literal["planet", "eclipsing_binary", "grazing_eb", "systematic", "noise"] = "planet",
    seed: int = Query(1, ge=0, le=10_000_000),
    model: ModelChoice = "auto",
):
    return _guard(service.demo, _predictor(), kind, seed, model)


@app.post("/api/analyze/upload")
async def analyze_upload(file: UploadFile = File(...), model: ModelChoice = Form("auto")):
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is larger than 40 MB.")

    def work():
        t, f, e = service.parse_table(content, file.filename or "")
        return service.run(t, f, e, _predictor(), file.filename or "uploaded light curve", "upload", model)

    return await asyncio.to_thread(_guard, work)


@app.post("/api/analyze/target")
async def analyze_target(req: TargetRequest):
    def work():
        from ..data.download import fetch_lightcurve

        tid = service.parse_target(req.target, req.mission)
        try:
            t, f, e = fetch_lightcurve(tid, req.mission)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(502, f"Could not fetch data from MAST: {exc}") from exc
        eph = None
        if req.period_days and req.t0 is not None and req.duration_hours:
            eph = (req.period_days, req.t0, req.duration_hours / 24.0)
        prefix = "KIC" if req.mission == "kepler" else "TIC"
        return service.run(t, f, e, _predictor(), f"{prefix} {tid}", "mast", req.model, ephemeris=eph)

    return await asyncio.to_thread(_guard, work)


# ---- dashboard ------------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")
