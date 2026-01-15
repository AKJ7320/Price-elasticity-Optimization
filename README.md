# Price-elasticity-Optimization

Synthetic retail pricing analysis using hierarchical elasticity estimation and bounded price optimization.

This existing project is organized into six logical stages using the requested January–February 2026 commit timeline.

## Project scaffold

The initial scaffold provides dependency requirements, ignore rules, and the 2026 MIT license.

Install dependencies with `python -m pip install -r requirements.txt`.

## Synthetic data

Run `python src/generate_synthetic_data.py` to generate 10,800 SKUs across 26 synthetic weeks from 2025-06-30 to 2025-12-22. The final week ends 2025-12-28. Numeric week indices remain the seasonal model input. Previews are available in `data/sample/`; full generated CSVs under `data/raw/` are ignored by Git.

# Author

Adveidh Krishna

Financial Analyst

[LinkedIn](https://www.linkedin.com/in/akthotarath)
