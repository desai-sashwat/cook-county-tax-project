# Cook County Property Tax Shift Analysis

ML analysis of the effects of a revenue-neutral land value tax shift on residential properties in Cook County, IL.

## Overview

This project uses Cook County Assessor's Office (CCAO) open data to:
1. Train an Automated Valuation Model (AVM) for residential properties
2. Decompose property values into land and improvement components using hedonic modeling
3. Simulate a revenue-neutral shift from property tax to land value tax
4. Classify winners and losers from the tax shift across property tiers and townships

## Data Sources

- **CCAO Training Data** — ~413K residential sales (9 years) with ~200 raw features
- **CCAO Assessment Data** — ~1.1M residential properties
- **CCAO Land Rate Data** — 5,040 neighborhood-level land rates ($/sqft)

## Analysis Pipeline

| Notebook | Description |
|----------|-------------|
| `01_data_ingestion_eda.ipynb` | Download CCAO data, exploratory analysis, geographic and price distributions |
| `02_feature_engineering.ipynb` | Clean data, engineer 123 features, create train/val/test splits (324K/40.5K/40.5K) |
| `03_avm_model_training.ipynb` | Train Ridge, XGBoost, and LightGBM AVMs; evaluate with IAAO metrics |
| `04_land_value_decomposition.ipynb` | Decompose property values into land + improvement via hedonic LightGBM model |
| `05_tax_shift_sim.ipynb` | Simulate 3 revenue-neutral tax scenarios, classify winners/losers across 673K homeowners |

## Key Results

**AVM Performance (LightGBM, test set)**:
- R² = 0.829 | MAE = $83K | MAPE = 29.4%

**Tax Shift Simulation**:
| Scenario | Description | Homeowners Paying Less |
|----------|-------------|----------------------|
| A | Flat rate on total value (no classification) | 69.4% |
| B | Land-only tax, single flat rate | 18.5% |
| C | Land-only tax with current classification system | 18.5% |

## Project Structure

```
cook-county-tax-shift/
├── data/
│   ├── raw/               # Parquet files from CCAO open data
│   └── processed/         # Cleaned splits, feature config, decomposed values
├── notebooks/             # 5-stage analysis pipeline
├── outputs/
│   ├── figures/           # 19 PNGs (EDA, model, decomposition, simulation)
│   ├── models/            # Trained LightGBM, XGBoost, Ridge models
│   ├── reports/           # Simulation summary and tier comparison (JSON)
│   └── metrics/           # Model comparison and feature importance (CSV)
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Requirements

- **ML**: scikit-learn, lightgbm, xgboost, shap
- **Data**: pandas, numpy, pyarrow, scipy
- **Visualization**: matplotlib, seaborn, plotly
- **Geospatial** (optional): geopandas, folium
