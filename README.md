# siddha-lifestyle

`siddha-lifestyle` is a Python-oriented data and experimentation workspace for exploratory analytics, dataset-backed analysis, and small utility code. The repository currently contains a set of Python source packages, Jupyter notebooks, bundled datasets, and supporting documentation.

## Repository layout

```text
siddha-lifestyle/
├── data/                     # Public/statistical datasets used by the project
│   └── covid_19_India/       # India-specific COVID-19 CSV data
├── docs/                     # Feature documentation and project notes
├── notebooks/                # Experiment notebooks (for analysis and visualization)
├── src/                      # Core Python modules and package scaffolding
│   ├── covid19/              # COVID-19 data/config helpers
│   ├── mypro/                # Example package/config module
│   └── utils.py              # Root-level utility imports
├── assets/                   # Static images and legacy visual assets
├── environment.yml           # Conda environment specification
├── main.py                   # Small argparse-based Python entry point
└── requirements.txt          # Python dependencies for the project
```

## Project contents

- `src/covid19/india_covid19.py` contains exploratory data-processing and Plotly visualization code for COVID-19 trend analysis.
- `src/covid19/config.py` exposes dataset state used by the notebooks and analysis scripts.
- `src/mypro/config.py` is a minimal package/config example.
- `data/` includes India COVID-19 CSV files and a representative sample dataset.
- `notebooks/` contains analysis notebooks for exploration and experimentation.
- `docs/features/` collects notes that describe the evolving feature set and project history.

## Getting started

### Clone the repository

```bash
git clone git@github.com:ravioshankar/siddha-lifestyle.git
cd siddha-lifestyle
```

### Create a Python environment

Use either Conda or a standard virtual environment.

#### Using Conda

```bash
conda env create -f environment.yml
conda activate base
```

#### Using pip/venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the CLI entry point

`main.py` is a lightweight command-line entry point. It accepts a positional argument and optional flags such as `-f`, `-n`, and `-v`.

```bash
python main.py --help
python main.py sample --name demo -v
```

## Notes

- This repository is not a packaged application in the usual service/API sense. It is organized as a research and data exploration workspace.
- The project documentation and naming inside the repository still carries some legacy references from earlier work; the source tree and data files are the main source of truth for current structure.
