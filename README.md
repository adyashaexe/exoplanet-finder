# Exoplanet Finder

Find planets hiding in starlight. Give it a brightness-over-time curve from a star and it searches for repeating dips, decides whether they look like a planet or something else (a pair of eclipsing stars, an instrument glitch, plain noise), and explains why on a dashboard.

- **Search:** Box Least Squares finds the orbital period, dip depth and dip length.
- **Classify:** a two-branch 1D CNN reads the phase-folded curve (whole orbit + zoom on the dip), and a gradient-boosting model on the measured features gives a baseline to beat.
- **Explain:** vetting checks (odd/even depth, secondary eclipse, repeats, depth, duration) and an occlusion map showing where the CNN looked.
- **Serve:** FastAPI backend and a single-page dashboard that works offline (Plotly and fonts are vendored).

## Quick start (no internet needed)

```bash
pip install torch            # CPU build is fine: --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

exofinder make-synthetic --n 1500 --workers 4   # simulate + process light curves
exofinder train --epochs 25                      # baseline + CNN, writes artifacts/
exofinder serve                                  # open http://127.0.0.1:8000
```

`make demo` runs all three. The synthetic data lets the whole stack run anywhere, but accuracy on it is optimistic: it only tells you the code works.

## Train on real Kepler data

```bash
pip install -e ".[data]"
exofinder fetch-labels --mission kepler          # NASA Exoplanet Archive KOI table
exofinder download --mission kepler --limit 800 --balanced --workers 4   # light curves from MAST
exofinder build-dataset --mission kepler
exofinder train
```

Labels: confirmed planet = 1, false positive = 0. Unresolved candidates are left out of training. For TESS use `--mission tess`. Downloads need access to `mast.stsci.edu` and `exoplanetarchive.ipac.caltech.edu`.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | status and loaded models |
| `GET /api/model` | thresholds and evaluation metrics |
| `GET /api/demo?kind=planet&seed=7` | synthetic star through the full pipeline |
| `POST /api/analyze/upload` | CSV or FITS (`time`, `flux`, optional `flux_err`) |
| `POST /api/analyze/target` | `{"target": "KIC 8462852", "mission": "kepler"}` fetched from MAST |

Interactive docs at `/docs`.

## How it works

```
raw light curve ─► clean + detrend ─► BLS period search ─► re-detrend with transits masked
                                                │
                              fold at best period, bin
                                   ┌────────────┴─────────────┐
                           global view (2001)          local view (201)
                                   └──► CNN ◄── 9 scalar features (SNR, depth, odd/even, secondary…)
                                                │
                                         planet-likeness score
```

Design choices worth knowing:

- **Split by star**, never by row, so one star's several candidates cannot leak across train and test.
- **Thresholds** are tuned for best F1 on the validation split and stored with each model.
- **Imbalance:** loss is class-weighted; judge models by PR-AUC and recall, not accuracy.
- **Real vs blind:** for catalogue stars the known ephemeris is used; uploads get a blind BLS search.

## Layout

```
src/exofinder/
  config.py            paths, view sizes, feature order
  data/                synthetic.py, labels.py, download.py, build.py
  features/            preprocess.py, bls.py, pipeline.py
  models/              cnn.py, baseline.py, train.py, metrics.py
  predictor.py         loads models, scores, occlusion map
  service.py           parsing, vetting checks, JSON report
  api/                 main.py + static/ (dashboard)
  cli.py               exofinder command
tests/  Dockerfile  Makefile  .github/workflows/ci.yml
```

## Limits

- Trained on synthetic data by default. Retrain on real labels before trusting any score.
- BLS finds the strongest signal only; multi-planet systems need iterative masking (a good next step).
- Kepler long-cadence assumptions (30 min) are baked into the synthetic generator and the default view config.

MIT licensed. Plotly (MIT), Atkinson Hyperlegible and Spectral (SIL OFL) are vendored with their licenses under `api/static/vendor/`.
