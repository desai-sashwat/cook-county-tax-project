# Revenue-Neutral Property Tax Shift: An ML Analysis of Site Value Taxation in U.S. Metropolitan Areas

**Machine Learning 2 — Final Project | Northeastern University | Spring 2026**

**Authors:** Sashwat Desai, Kanishka Sharma
**Sponsor:** James Frederiksen, Duke University

---

## Research Question

> If Cook County, IL eliminated all taxes on improvements and raised the tax rate on all taxable sites in a revenue-neutral manner, would the majority of homeowners pay lower property taxes?

The current U.S. property tax system taxes both land and buildings equally. This penalizes private investment in improvements (new construction, renovations) while subsidizing speculative holding of vacant or underutilized parcels. This project tests whether shifting to a land-only tax — while keeping total revenue constant — would lower property taxes for the majority of homeowners in Cook County, IL and five additional U.S. metro areas.

**Result: Hypothesis supported.** 57.5% of Cook County homeowners would pay less under a revenue-neutral land-only tax shift, with median annual savings of $1,656.

---

## Key Findings

| Finding | Detail |
|---------|--------|
| **Hypothesis supported** | 57.5% of 672,719 homeowners pay less under Scenario B (land-only, flat rate) |
| **Median savings for winners** | $1,656/year |
| **Median increase for losers** | $2,302/year |
| **Key mechanism** | The site-to-total value ratio determines winners vs losers (median: 20.8%) |
| **AVM performance** | LightGBM: R² = 0.83, MAPE = 29.4%, COD = 30.66 |
| **Spatial dominance** | ML-based hedonic decomposition fails — location alone reconstructs ~100% of property value |
| **Cross-metro** | Hypothesis holds where genuine land/building splits exist; fails where data is estimated uniformly |

---

## Pipeline Architecture

Seven modular Jupyter notebooks form a reproducible pipeline:

```
01 Data Ingestion & EDA ──► 02 Feature Engineering ──► 03 AVM Training
                                                            │
04 Land Value Decomposition ◄───────────────────────────────┘
        │
        ▼
05 Tax Shift Simulation ──► 06 Winner/Loser Classification
        │
        ▼
07 Cross-Metro Comparison + TDA
```

| Notebook | Description | Key Output |
|----------|-------------|------------|
| `01 — data_ingestion_eda.ipynb` | Download CCAO parquet files, explore 1.1M properties, analyze distributions and correlations | EDA figures, data profiling |
| `02 — feature_engineering.ipynb` | Clean data, engineer features (age, ratios, log transforms), time-based 80/10/10 split | `data/processed/` parquets, `feature_config.json` |
| `03 — avm_model_training.ipynb` | Train Ridge, XGBoost, LightGBM; evaluate with ML + IAAO metrics; SHAP analysis | Model files, test predictions, feature importance |
| `04 — land_value_decomposition.ipynb` | Attempt hedonic subtraction (v1, v2 — both fail), implement hybrid CCAO-ratio approach (v3) | `assessment_decomposed.parquet`, decomposition metadata |
| `05 — tax_shift_sim.ipynb` | Revenue-neutral simulation of Scenarios A, B, C; hypothesis test across 672K homeowners | `tax_simulation_results.parquet`, simulation summary |
| `06 — winner_loser_comparison.ipynb` | LightGBM classifier for winner prediction; SHAP profiling; data leakage analysis | Classifier model, classification metrics |
| `07 — cross_metro_comparison.ipynb` | Fiscal simulation for 5 additional metros; persistent homology (TDA) for urban topology | Cross-metro CSV, topological features, Wasserstein distances |

---

## Data Sources

### Primary — Cook County, IL

All data downloaded programmatically from the [Cook County Assessor's Office](https://datacatalog.cookcountyil.gov/) public S3 bucket.

| Dataset | Records | Description |
|---------|---------|-------------|
| `training_data.parquet` | ~400,000 | Residential sales (9-year window) with ~100 features |
| `assessment_data.parquet` | 1,100,150 | Full residential assessment universe (sold + unsold) |
| `land_nbhd_rate_data.parquet` | — | CCAO neighborhood-level land rates ($/sqft) |

Features span 6 families: `char_` (physical characteristics), `loc_` (location identifiers), `prox_` (proximity to amenities), `acs5_` (ACS/Census socioeconomic data), `shp_` (parcel geometry), `time_` (sale timing).

### Cross-Metro Comparison

| Metro | Source | Key Columns |
|-------|--------|-------------|
| Philadelphia, PA | OpenDataPhilly | `market_value`, `taxable_land`, `taxable_building` |
| New York City, NY | NYC MapPLUTO | `AssessLand`, `AssessTot`, `Latitude`, `Longitude` |
| Boston, MA | Analyze Boston | `AV_LAND`, `AV_BLDG`, `AV_TOTAL` |
| Detroit, MI | Detroit Open Data | `Assessed Value`, `Property Class` |
| Washington, D.C. | DC Open Data | `LAND`, `BLDG`, `LATITUDE`, `LONGITUDE` |

---

## Methodology

### Automated Valuation Model (Notebook 03)

Three models compared on the test set:

| Metric | Ridge | XGBoost | LightGBM ★ |
|--------|-------|---------|------------|
| RMSE | $154,593 | $119,688 | $120,662 |
| MAE | $107,268 | $82,194 | $83,130 |
| MAPE | 36.15% | 30.29% | **29.39%** |
| R² | 0.7197 | 0.8321 | 0.8293 |
| COD | 38.41 | 31.37 | **30.66** |
| PRD | 1.1544 | 1.1566 | 1.1465 |

LightGBM selected as primary model (best MAPE and COD). Matches the CCAO's production model choice.

### Land Value Decomposition (Notebook 04)

The central technical challenge: splitting total property value into site (land) and improvement (building) components.

- **v1 — Full hedonic subtraction:** FAILED. A land-features-only LightGBM reconstructed ~100% of total value from spatial features, leaving $0 for improvements.
- **v2 — Constrained hedonic (spatial only):** FAILED. Even with only lat/lon and proximity features, median site ratio = 100%.
- **v3 — Hybrid CCAO-ratio approach:** SUCCEEDED. Applied the CCAO Assessor's land/building ratio to AVM-predicted total values.

**Root cause of failure:** In Cook County, high-cardinality location features (school district GEOIDs, neighborhood codes) implicitly encode improvement quality because neighborhoods are self-sorted by income and housing quality. This *spatial dominance problem* makes ML-based hedonic decomposition unreliable in dense urban markets.

### Tax Shift Simulation (Notebook 05)

Three revenue-neutral scenarios (total revenue held constant at $10.76B):

| Scenario | Description | % Homeowners Paying Less |
|----------|-------------|--------------------------|
| A | Flat rate on total value (eliminate classification) | 67.6% |
| **B** | **Land-only tax, flat rate (main proposal)** | **57.5%** |
| C | Land-only tax, keep classification | 57.5% |

Scenarios B and C are identical for residential properties because all residential classes share the same 10% assessment level.

### Cross-Metro Results (Notebook 07)

| Metro | Homeowners | % Paying Less | Median Site Ratio | Result |
|-------|-----------|---------------|-------------------|--------|
| Cook County, IL | 672,719 | 57.5% | 20.8% | ✓ Supported |
| New York City, NY | 767,839 | 49.8% | 21.2% | ~ Borderline |
| Philadelphia, PA | 294,710 | 13.0% | 20.0% | ✗ Not supported |
| Boston, MA | 55 | 0.0% | 100.0% | ⚠ Data limitation |
| Detroit, MI | 263,474 | 0.0% | 100.0% | ⚠ Data limitation |

### Topological Data Analysis (Notebook 07)

Persistent homology (Vietoris-Rips complex) applied to lat/lon point clouds of 3,000 sampled residential properties. Available for Philadelphia and NYC only.

| Metric | Philadelphia | NYC |
|--------|-------------|-----|
| H0 Components | 2,921 | 2,999 |
| H0 Max Persistence | 1.08 km | 5.02 km |
| H1 Loops | 580 | 630 |
| H1 Max Persistence | 1.81 km | 2.01 km |
| Wasserstein H0 Distance | — | 311.0 |
| Wasserstein H1 Distance | — | 52.9 |

NYC shows ~5× greater H0 max persistence (more fragmented development) and more H1 loops (larger undeveloped voids).

---

## Project Structure

```
cook-county-tax-shift/
├── data/
│   ├── external/                          # Cross-metro datasets
│   │   ├── boston_assessment.csv
│   │   ├── dc_properties.parquet
│   │   ├── detroit_parcels.csv
│   │   ├── nyc_pluto.csv
│   │   └── philadelphia_properties.parquet
│   ├── processed/                         # Cleaned and engineered data
│   │   ├── assessment_clean.parquet
│   │   ├── assessment_decomposed.parquet
│   │   ├── feature_config.json
│   │   ├── tax_simulation_results.parquet
│   │   ├── test_split.parquet
│   │   ├── train_split.parquet
│   │   └── val_split.parquet
│   └── raw/                               # Raw CCAO downloads
│       ├── assessment_data.parquet
│       ├── land_nbhd_rate_data.parquet
│       └── training_data.parquet
├── notebooks/
│   ├── data_ingestion_eda.ipynb           # 01 — EDA
│   ├── feature_engineering.ipynb          # 02 — Feature engineering
│   ├── avm_model_training.ipynb           # 03 — AVM training
│   ├── land_value_decomposition.ipynb     # 04 — Decomposition (v3)
│   ├── tax_shift_sim.ipynb                # 05 — Tax simulation
│   ├── winner_loser_comparison.ipynb      # 06 — Classification
│   ├── cross_metro_comparison.ipynb       # 07 — Cross-metro + TDA
│   └── download_metro_data.py             # Data download script
├── outputs/
│   ├── figures/                           # All visualization outputs
│   │   ├── 01_*.png                       # EDA figures
│   │   ├── 03_*.png                       # AVM model figures
│   │   ├── 04_*.png, 04v2_*, 04v3_*      # Decomposition figures
│   │   ├── 05_*.png                       # Tax simulation figures
│   │   ├── 06_*.png                       # Classification figures
│   │   └── 07_*.png                       # Cross-metro + TDA figures
│   ├── models/                            # Trained models and predictions
│   │   ├── lgb_avm_model.txt              # LightGBM AVM (primary)
│   │   ├── lgb_land_model.txt             # Land model v1 (failed)
│   │   ├── lgb_land_model_v2.txt          # Land model v2 (failed)
│   │   ├── winner_classifier.txt          # Winner/loser classifier
│   │   ├── xgb_avm_model.joblib           # XGBoost AVM
│   │   ├── xgb_preprocessor.joblib        # XGBoost preprocessor
│   │   ├── ridge_avm_model.joblib         # Ridge AVM
│   │   ├── test_predictions.parquet       # Test set predictions
│   │   ├── model_comparison.csv           # Model performance comparison
│   │   └── feature_importance.csv         # LightGBM feature importance
│   └── reports/                           # Summary statistics and metadata
│       ├── simulation_summary.json
│       ├── classification_metrics.json
│       ├── decomposition_metadata.json
│       ├── tier_comparison.json
│       ├── cross_metro_comparison.csv
│       ├── topological_features.csv
│       └── combined_fiscal_topo.csv
├── reports/                               # Project reports
│   ├── MATH_7339_Machine_Learning_2_Project_Report.pdf
│   ├── Revenue-Neutral-Property-Tax-Shift-An-ML-Analysis-of-Site-Value-Taxation-in-US-Metropolitan-Areas.pdf
│   └── Summary.pdf
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Setup & Installation

### Requirements

- Python 3.9+
- ~8 GB RAM (for full assessment dataset)
- ~2 GB disk space for raw data downloads

### Install Dependencies

```bash
git clone https://github.com/desai-sashwat/cook-county-tax-project.git
cd cook-county-tax-project
pip install -r requirements.txt
```

### Key Dependencies

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
lightgbm>=4.0
xgboost>=2.0
shap>=0.43
matplotlib>=3.7
seaborn>=0.12
ripser>=0.6
persim>=0.3
pyarrow>=12.0
joblib>=1.3
```

### Run the Pipeline

Execute notebooks in order (01 → 07). Each notebook reads from the outputs of previous notebooks.

```bash
cd notebooks/
jupyter notebook data_ingestion_eda.ipynb          # Downloads data automatically
jupyter notebook feature_engineering.ipynb
jupyter notebook avm_model_training.ipynb
jupyter notebook land_value_decomposition.ipynb
jupyter notebook tax_shift_sim.ipynb
jupyter notebook winner_loser_comparison.ipynb
jupyter notebook cross_metro_comparison.ipynb       # Requires external metro data
```

Notebook 01 downloads CCAO data automatically from their public S3 bucket. Cross-metro datasets (Notebook 07) require separate downloads from each city's open data portal — see the prerequisites section in that notebook.

---

## Known Limitations

1. **AVM COD (30.66) exceeds the IAAO target of 15.** Our model does not apply the post-processing corrections CCAO uses in production (ratio studies, appeals adjustments, multi-year smoothing). It is a research instrument, not a production assessment system.

2. **Data leakage in the winner/loser classifier.** Including `site_value_ratio` and decomposition-derived features as classification inputs produces AUC = 1.0 because these features deterministically generate the labels. The SHAP analysis remains valuable as a descriptive tool.

3. **Cross-metro data heterogeneity.** Only 2 of 5 comparison metros (Cook County, NYC) had genuine assessed land/building splits. Boston and Detroit lack separate land valuations; Philadelphia's splits may reflect estimation artifacts.

4. **TDA limited to 2 metros.** Only Philadelphia and NYC had usable lat/lon coordinates. Findings are suggestive, not causal.

5. **CCAO ratio approach inherits assessment biases.** The hybrid decomposition relies on CCAO's assessed land/building ratios, which may contain systematic biases from the assessment process.

6. **Static simulation.** The revenue-neutral model does not capture dynamic effects of a land-only tax (reduced speculation, increased construction, land price adjustments over time).

---

## References

- Ganong, P. & Shoag, D. (2017). "Why Has Regional Income Convergence in the U.S. Declined?" *Journal of Urban Economics*, 102, 76–90.
- Cook County Assessor's Office Open Data: [datacatalog.cookcountyil.gov](https://datacatalog.cookcountyil.gov/)
- CCAO Residential AVM: [github.com/ccao-data/model-res-avm](https://github.com/ccao-data/model-res-avm)
- Jacobs, J. (1961). *The Death and Life of Great American Cities*. Random House.
- Project framework: James Frederiksen, Duke University.

---

## License

This project is for academic purposes. See [LICENSE](LICENSE) for details.
