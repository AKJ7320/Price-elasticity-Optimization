# Price-elasticity-Optimization

Synthetic retail pricing analysis using hierarchical elasticity estimation and bounded price optimization.

This existing project is organized into six logical stages using the requested January–February 2026 commit timeline.

## Project scaffold

The initial scaffold provides dependency requirements, ignore rules, and the 2026 MIT license.

Install dependencies with `python -m pip install -r requirements.txt`.

## Synthetic data

Run `python src/generate_synthetic_data.py` to generate 10,800 SKUs across 26 synthetic weeks from 2025-06-30 to 2025-12-22. The final week ends 2025-12-28. Numeric week indices remain the seasonal model input. Previews are available in `data/sample/`; full generated CSVs under `data/raw/` are ignored by Git.

## Elasticity estimation

`src/mllib.py` supplies NumPy regression primitives; `src/elasticity_model.py` implements category pooling and SKU shrinkage. After generating data, run `python src/elasticity_model.py` to evaluate parameter recovery against separate synthetic ground truth. This is not a future-period forecast test.

## 2026 optimization scenarios

Run `python src/train.py` after data generation to evaluate elasticity estimates and produce bounded (±12%) pricing scenarios. Metrics and per-SKU results are written under `reports/`; figures are generated under `screenshots/`. Run `python src/business_impact.py` for the annual estimate based on explicit illustrative assumptions. These are scenarios, not observed outcomes. Run `python -m pytest tests/` for optimizer checks.

# Author

Adveidh Krishna

Financial Analyst

[LinkedIn](https://www.linkedin.com/in/akthotarath)
