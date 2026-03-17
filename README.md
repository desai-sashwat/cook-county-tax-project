# Cook County Property Tax Shift Analysis

ML analysis of the effects of a revenue-neutral land value tax shift on residential properties in Cook County, IL.

## Overview

This project uses Cook County Assessor's Office (CCAO) open data to:
1. Train an Automated Valuation Model (AVM) for residential properties
2. Decompose property values into land and improvement components
3. Simulate a revenue-neutral shift from property tax to land value tax
4. Classify winners and losers from the tax shift using SHAP analysis
5. Compare results across metro areas (Philadelphia, Detroit, DC)

## Data Sources

- **CCAO Training Data** — ~400K residential sales with ~100 features
- **CCAO Assessment Data** — ~1.1M residential properties
- **CCAO Land Rate Data** — Neighborhood-level land rates ($/sqft)

## Project Structure

```
cook-county-tax-shift/
├── data/                  # Raw & processed datasets (gitignored)
├── notebooks/             # Jupyter notebooks for each analysis stage
├── src/                   # Reusable Python modules
└── outputs/               # Figures, models, reports (gitignored)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
