# Exoplanet Finder

<p align="center">
  <strong>Find planetary transit signals in stellar light curves.</strong><br>
  A research-oriented pipeline combining NASA archival observations, Box Least Squares (BLS), machine learning, and an interactive web dashboard.
</p>

<p align="center">
  <a href="https://github.com/adyashaexe/exoplanet-finder/actions"><img src="https://img.shields.io/github/actions/workflow/status/adyashaexe/exoplanet-finder/ci.yml?branch=main&label=CI" alt="CI status"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Missions-Kepler%20%7C%20TESS-6b5bd2" alt="Kepler and TESS">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License">
</p>

Exoplanet Finder analyses a star's brightness over time (a **light curve**) and searches for the small, repeating dips that can occur when a planet passes in front of its host star. It combines a classical transit-search method with machine-learning classification and diagnostic checks, then presents the results in a local web dashboard.

> **Research note:** This is an experimental project, not a validated planet-discovery system. A planet-like score is a candidate-ranking signal—not confirmation of an exoplanet. Follow-up observations and scientific vetting are required.

## Contents

- [What it does](#what-it-does)
- [How the pipeline works](#how-the-pipeline-works)
- [Data sources](#data-sources)
- [Quick start](#quick-start)
- [Use NASA Kepler or TESS data](#use-nasa-kepler-or-tess-data)
- [Run the dashboard](#run-the-dashboard)
- [API](#api)
- [Project structure](#project-structure)
- [Reproducibility and evaluation](#reproducibility-and-evaluation)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## What it does

- **Searches for transits:** uses Box Least Squares (BLS) to identify repeating dips and estimate the period, depth, duration, and signal-to-noise ratio.
- **Classifies candidate signals:** a two-view 1D convolutional neural network (CNN) analyses the folded light curve at two scales. A gradient-boosting baseline provides a comparison model.
- **Vets detections:** reports diagnostic checks such as odd/even transit depth, secondary eclipses, repeatability, depth, and duration.
- **Visualises evidence:** the dashboard presents the light curve, candidate information, model output, and an occlusion-based view of which parts of the input influence the CNN.
- **Works with real archive data:** can retrieve Kepler and TESS light curves from the Mikulski Archive for Space Telescopes (MAST) through Lightkurve.
- **Supports local files:** analyse uploaded CSV or FITS light curves as well as supported MAST targets.
- **Includes synthetic data:** generate simulated examples to exercise the complete pipeline without downloading telescope observations.

## How the pipeline works

```text
Light curve (NASA archive / CSV / FITS / synthetic)
                    |
                    v
       Cleaning, normalisation, detrending
                    |
                    v
       Box Least Squares (BLS) search
                    |
                    v
       Candidate period and transit features
                    |
             Phase folding
                    |
          +---------+---------+
          |                   |
     Global view          Local transit view
          |                   |
          +---------+---------+
                    |
             1D CNN + features
                    |
          Baseline / ensemble
                    |
                    v
        Score, vetting checks and plots
```

The model uses a global phase-folded view (2,001 bins), a local transit view (201 bins), and nine auxiliary scalar features. The preprocessing and feature configuration is shared across training and inference; changing it can invalidate previously trained model artifacts.

## Data sources

| Source | Used for | Access |
|---|---|---|
| [MAST](https://archive.stsci.edu/) | Kepler and TESS photometric light curves | [MAST Portal](https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html) |
| [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) | Kepler KOI and TESS TOI catalogue dispositions and ephemerides | [TAP service](https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html) |
| [Lightkurve](https://lightkurve.github.io/lightkurve/) | Python search and download interface for MAST light curves | [Documentation](https://lightkurve.github.io/lightkurve/) |

**About “live” data:** target analysis queries NASA's public archives when a request is made. These are archived observations made available through MAST, not a real-time stream from a telescope. Archive coverage, cadence, products, and availability vary by target and mission.

## Quick start

### Requirements

- Python 3.10 or newer
- Internet access for NASA archive retrieval
- A terminal (the commands below use macOS/Linux shell syntax)
- Optional: PyTorch for CNN training and inference

### 1. Clone the repository

```bash
git clone https://github.com/adyashaexe/exoplanet-finder.git
cd exoplanet-finder
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install the project

For the full development and data workflow:

```bash
pip install -e ".[all]"
```

Install PyTorch separately if you want CNN training/inference and it is not installed in your environment:

```bash
pip install torch
```

For a CPU-only PyTorch installation, use the appropriate command from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

### 4. Generate a local synthetic dataset (optional)

This validates the processing and training workflow without contacting NASA:

```bash
exofinder make-synthetic --n 1500 --workers 4
exofinder train --epochs 25
```

### 5. Start the application

```bash
exofinder serve
```

Open **http://127.0.0.1:8000** in your browser. API documentation is available at **http://127.0.0.1:8000/docs**.

The dashboard's synthetic/demo workflow can be used to explore the interface. Synthetic performance is not evidence of performance on real telescope observations.

## Use NASA Kepler or TESS data

The project uses two separate NASA archive sources for different purposes:

1. **NASA Exoplanet Archive** provides catalogue labels and transit ephemerides used to prepare labelled training data.
2. **MAST** provides the observed light curves used for analysis and training.

### Analyse one target in the dashboard

Start the server, select the Star ID workflow, choose a mission, and enter a numeric KIC (Kepler) or TIC (TESS) identifier. The backend searches MAST and downloads available matching light-curve products.

You can also call the API directly:

```bash
curl -X POST http://127.0.0.1:8000/api/analyze/target \
  -H "Content-Type: application/json" \
  -d '{"target":"8462852","mission":"kepler","model":"auto"}'
```

For TESS, use a TIC identifier and set `"mission":"tess"`.

### Build a real labelled Kepler dataset

The following commands fetch catalogue labels, download light curves for labelled targets, process the downloaded observations, and train the models:

```bash
# Retrieve Kepler KOI dispositions from the NASA Exoplanet Archive
exofinder fetch-labels --mission kepler

# Download a balanced sample of confirmed planets and false positives from MAST
exofinder download --mission kepler --limit 800 --balanced --workers 4

# Convert downloaded observations into model inputs
exofinder build-dataset --mission kepler

# Train the baseline and CNN
exofinder train --epochs 25
```

For TESS, replace `kepler` with `tess`:

```bash
exofinder fetch-labels --mission tess
exofinder download --mission tess --limit 800 --balanced --workers 4
exofinder build-dataset --mission tess
exofinder train --epochs 25
```

The download command caches successful light curves under `data/raw/`. Re-running it can reuse files already present. The `--balanced` option samples equal counts from the two labelled classes (confirmed and false positive); unresolved candidates are excluded from this training download.

**Start small:** before requesting hundreds of targets, test the archive connection and workflow with a small `--limit` (for example, 10 or 20). MAST availability and network conditions can vary. A target may have no matching product, and a temporary connection failure does not necessarily indicate a code defect.

## Run the dashboard

The dashboard is served by the FastAPI application. The static interface, styles, Plotly bundle, and fonts are packaged with the project.

Useful commands:

```bash
# Default local address: 127.0.0.1:8000
exofinder serve

# Enable auto-reload during development
exofinder serve --reload

# Choose another port
exofinder serve --host 127.0.0.1 --port 8080
```

To expose the server on your local network, the application can bind to a non-loopback address, for example `--host 0.0.0.0`. Do this only on a trusted network and configure appropriate access controls before exposing it to the public internet.

## API

Interactive Swagger documentation: `/docs`.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Application status, version, and available models |
| `GET` | `/api/model` | Model information, thresholds, and evaluation metrics |
| `POST` | `/api/model/reload` | Reload model artifacts |
| `GET` | `/api/demo/kinds` | Available synthetic signal classes |
| `GET` | `/api/demo` | Run a synthetic example through the pipeline |
| `POST` | `/api/analyze/upload` | Analyse an uploaded CSV or FITS light curve |
| `POST` | `/api/analyze/target` | Retrieve and analyse a KIC/TIC target from MAST |

### Example: analyse a MAST target

```bash
curl -X POST http://127.0.0.1:8000/api/analyze/target \
  -H "Content-Type: application/json" \
  -d '{
    "target": "8462852",
    "mission": "kepler",
    "model": "auto"
  }'
```

### Example: analyse a local CSV

The upload endpoint accepts a CSV or FITS file containing a time column and a flux column. An optional flux-error column may also be supplied.

```bash
curl -X POST http://127.0.0.1:8000/api/analyze/upload \
  -F "file=@lightcurve.csv" \
  -F "model=auto"
```

Available model choices are `auto`, `cnn`, `baseline`, and `ensemble`.

## Command-line reference

| Command | Purpose |
|---|---|
| `exofinder make-synthetic` | Generate and process a synthetic dataset |
| `exofinder fetch-labels` | Retrieve Kepler KOI or TESS TOI dispositions |
| `exofinder download` | Download labelled targets' light curves from MAST |
| `exofinder build-dataset` | Process downloaded light curves into a training dataset |
| `exofinder train` | Train the baseline and, when enabled, CNN |
| `exofinder predict <file>` | Analyse a local CSV/FITS light curve |
| `exofinder serve` | Start the API and dashboard |

Run `exofinder <command> --help` for command-specific options.

## Project structure

```text
exoplanet-finder/
├── src/exofinder/
│   ├── api/
│   │   ├── main.py              # FastAPI routes
│   │   └── static/              # Dashboard, styles and vendored assets
│   ├── data/
│   │   ├── download.py          # MAST light-curve retrieval
│   │   ├── labels.py            # NASA Exoplanet Archive labels
│   │   ├── build.py             # Dataset construction
│   │   └── synthetic.py         # Synthetic light-curve generation
│   ├── features/
│   │   ├── preprocess.py        # Cleaning and detrending
│   │   ├── bls.py               # Box Least Squares search
│   │   └── pipeline.py          # Feature and view construction
│   ├── models/
│   │   ├── cnn.py               # Two-view 1D CNN
│   │   ├── baseline.py          # Gradient-boosting baseline
│   │   ├── train.py             # Training and evaluation
│   │   └── metrics.py           # Evaluation metrics
│   ├── cli.py                   # exofinder command
│   ├── config.py                # Paths and shared view configuration
│   ├── predictor.py             # Model loading and scoring
│   └── service.py               # Analysis and report assembly
├── tests/
├── artifacts/                   # Local trained model artifacts
├── data/                        # Local raw and processed data
├── pyproject.toml
├── Dockerfile
└── README.md
```

## Reproducibility and evaluation

- **Split by target star:** the training code splits by star rather than by individual candidate row to reduce leakage between examples from the same target.
- **Class imbalance:** training uses class weighting; precision-recall metrics are more informative than accuracy alone for imbalanced labels.
- **Stored configuration:** model artifacts are associated with the feature/view configuration used to train them.
- **Catalogue ephemerides:** real catalogue targets use the archive-provided period and transit epoch when building labelled examples. General uploads use a blind BLS search.
- **Synthetic vs. observed data:** synthetic examples are useful for software tests and demonstrations, but they do not establish real-world detection performance.

After training, inspect the generated metrics and validate on held-out real stars before interpreting model scores scientifically.

## Troubleshooting

### `Could not fetch data from MAST` / connection reset

The application could not complete a connection to the MAST service. Check internet access, VPN/proxy/firewall settings, and whether `https://mast.stsci.edu` is reachable from your machine. Retry later if the archive or route is temporarily unavailable. Try one target first before starting a large multi-worker download.

### No light curves found

A valid KIC/TIC identifier does not guarantee that the selected mission has a matching light-curve product. Check the identifier and mission, and try a target known to have observations in that archive.

### `lightkurve is required`

Install the data extra in the active virtual environment:

```bash
pip install -e ".[data]"
```

### PyTorch is unavailable

Install PyTorch using the official selector at [pytorch.org](https://pytorch.org/get-started/locally/). The baseline model can be used without CNN training when configured accordingly.

### Download progress appears stuck at zero

The first archive search/download may take time, and each target can involve remote requests. Test with a small limit, check network access, and inspect the final downloaded/failed target count rather than assuming the warning messages are fatal. Optional Lightkurve PRF/TPF modelling warnings do not, by themselves, indicate that ordinary light-curve retrieval has failed.

## Limitations

- The default/demo dataset is synthetic. A model trained only on synthetic curves should not be considered validated on telescope data.
- A planet-like classification is not a confirmed discovery. Stellar variability, eclipsing binaries, instrumental systematics, and data-quality problems can mimic transit signals.
- The current BLS workflow reports the strongest detected signal; iterative multi-planet searches are not implemented as a complete system.
- Cadence and preprocessing assumptions affect sensitivity. Kepler and TESS products have different observing patterns and data characteristics.
- MAST queries retrieve archived observations on demand; this project does not implement a continuous real-time telescope data stream.
- Network availability, archive coverage, data volume, and download speed are outside the application's control.

## Roadmap

Potential next steps for the project:

- [ ] Improve retry/backoff and caching for transient MAST failures.
- [ ] Add richer archive-search filters and show observation provenance in the dashboard.
- [ ] Expand evaluation on held-out Kepler and TESS targets.
- [ ] Add iterative searches for multi-planet systems.
- [ ] Add more detailed data-quality and stellar-variability diagnostics.
- [ ] Explore deployment with suitable resource limits and access controls.

## Acknowledgements

- **NASA / STScI MAST** for access to Kepler and TESS observations.
- **NASA Exoplanet Archive** for catalogue data and dispositions.
- **Lightkurve** for Python tools to work with astronomical time-series data.
- **Plotly** for interactive visualisation.

Third-party assets and their respective license notices are included under `src/exofinder/api/static/vendor/`.

## License

This project is released under the [MIT License](LICENSE).
