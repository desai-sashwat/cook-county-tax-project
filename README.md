# Cook County Property Tax Shift Analysis

ML analysis of the effects of a revenue-neutral land value tax shift on residential properties in Cook County, IL, with cross-metro comparison and Topological Data Analysis (TDA).

## Overview

This project uses Cook County Assessor's Office (CCAO) open data to:
1. Train an Automated Valuation Model (AVM) for residential properties
2. Decompose property values into land and improvement components using CCAO assessed ratios
3. Simulate a revenue-neutral shift from property tax to land value tax
4. Classify winners and losers using a LightGBM classifier with SHAP explanations
5. Compare outcomes across 5 U.S. metros and analyze urban structure via TDA (persistent homology)

## Data Sources

- **CCAO Training Data** — ~413K residential sales (9 years) with ~200 raw features
- **CCAO Assessment Data** — ~1.1M residential properties
- **CCAO Land Rate Data** — 5,040 neighborhood-level land rates ($/sqft)
- **Philadelphia** — OpenDataPhilly OPA properties (300K parcels)
- **Washington D.C.** — DC Open Data integrated tax system (ArcGIS export)
- **New York City** — NYC PLUTO dataset (853K parcels)
- **Boston** — Boston assessment data
- **Detroit** — Detroit parcel data

## Analysis Pipeline

| Notebook | Description |
|----------|-------------|
| `data_ingestion_eda.ipynb` | Download CCAO data, exploratory analysis, geographic and price distributions |
| `feature_engineering.ipynb` | Clean data, engineer 123 features, create train/val/test splits (324K/40.5K/40.5K) |
| `avm_model_training.ipynb` | Train Ridge, XGBoost, and LightGBM AVMs; evaluate with IAAO metrics |
| `land_value_decomposition.ipynb` | Hybrid decomposition: CCAO assessed land ratios × AVM total value predictions |
| `tax_shift_sim.ipynb` | Simulate 3 revenue-neutral tax scenarios across 673K homeowners |
| `winner_loser_comparison.ipynb` | LightGBM classifier + SHAP to profile winners vs losers |
| `cross_metro_comparison.ipynb` | Cross-metro fiscal comparison + TDA (persistent homology H0/H1) |

## Key Results

**AVM Performance (LightGBM, test set)**:
- R² = 0.829 | MAE = $83K | MAPE = 29.4%

**Tax Shift Simulation (Cook County, 672K homeowners)**:

| Scenario | Description | Homeowners Paying Less | Median Savings |
|----------|-------------|----------------------|----------------|
| A | Flat rate on total assessed value | 67.6% | ~$0 |
| B | Land-only tax, single flat rate | 57.5% | -$1,656 |
| C | Land-only tax with classification | 57.5% | -$1,656 |

**Winner/Loser Classifier**: ROC-AUC = 1.000 (near-perfect separability by land ratio)

**Cross-Metro Comparison**:

| Metro | Parcels | % Pay Less (Scenario B) | Median Site Ratio |
|-------|---------|------------------------|-------------------|
| Cook County, IL | 1,100,150 | 57.5% | 20.8% |
| Philadelphia, PA | 299,986 | 13.0% | 20.0% |
| New York City, NY | 853,624 | 49.8% | 21.2% |
| Boston, MA | 82 | 0.0% | 100.0% |
| Detroit, MI | 289,900 | 0.0% | 100.0% |

## Land Value Decomposition: Methodological Notes

- **v1/v2 (hedonic subtraction)**: Failed — spatial location dominates Cook County valuations, causing land-only models to reconstruct ~100% of total value, leaving nothing for improvements
- **v3 (hybrid, used in final analysis)**: CCAO's own assessed land/building ratios applied to AVM-predicted total values — combines assessor domain expertise with ML accuracy

## Project Structure

```
cook-county-tax-shift/
├── data/
│   ├── raw/               # CCAO parquet files (via Git LFS)
│   ├── processed/         # Cleaned splits, feature config, decomposed values
│   └── external/          # Philadelphia, DC, NYC, Boston, Detroit datasets
├── notebooks/             # 7-stage analysis pipeline + download script
├── outputs/
│   ├── figures/           # 42 PNGs (EDA, model, decomposition, simulation, TDA)
│   ├── models/            # Trained LightGBM, XGBoost, Ridge models
│   └── reports/           # Simulation summary, cross-metro comparison (JSON/CSV)
├── requirements.txt
└── .gitattributes         # Git LFS tracking for *.parquet, *.joblib, *.csv
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
- **TDA**: ripser, persim
