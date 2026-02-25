# Price Elasticity & Non-Linear Price Optimization Engine

Estimates SKU-level price elasticity across 10,000+ SKUs, then solves a bounded non-linear
optimization for the profit-maximizing price per SKU.

> ### ⚠️ About this repository
> This repository demonstrates a pricing optimization methodology using synthetic data.
> It estimates price elasticity and recommends prices within merchandising guardrails.
> All dollar figures are illustrative outputs from synthetic data, not actual company results.

## The business problem (quantified)

A retailer running **10,000+ SKUs** across multiple categories sets prices that directly determine both unit
volume and per-unit margin, the two levers of gross profit, pulling in opposite directions. Price too high
and you lose volume; too low and you give away margin. Getting the price right at the SKU level, across a
catalog that large, is worth millions in annual gross margin.

**The task:** (1) estimate how sensitive each SKU's demand is to its own price (its *price elasticity*), and
(2) use those elasticities to recommend a profit-maximizing price per SKU, within realistic merchandising
guardrails.

**The result (reproduced here on synthetic data):** **79.2% elasticity-prediction score (100 − MAPE)** evaluated
against synthetic ground truth held out from model fitting, and a **10.75% aggregate margin lift** across 10,800 SKUs from the optimized
prices.

## Why this problem is hard

Price elasticity isn't observable directly. It has to be estimated from limited, noisy price variation per SKU
(a single SKU might only see its price move ±10% over 26 weeks). A naive per-SKU regression overfits; a single
category-wide elasticity ignores real SKU-level differences that matter for pricing. And the optimization step
is genuinely non-linear (profit is not a linear function of price) and must respect real-world guardrails, 
merchandising teams don't accept unconstrained price swings.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | **Python 3.10** | Standard for production DS work |
| Data | **pandas**, **NumPy** | 280K-row SKU × week panel, feature engineering |
| Modeling | **Custom NumPy ML library** (`src/mllib.py`) | Ridge log-log regression + empirical-Bayes shrinkage, all from scratch, no scikit-learn dependency |
| Optimization | **NumPy** (closed-form + bounded grid search) | Non-linear profit maximization under price guardrails |
| Visualization | **Matplotlib** | Elasticity accuracy, demand/profit curves, margin-lift distribution |
| Testing | **pytest** | Optimizer correctness + guardrail-bound checks |

## Approach

### 1. Hierarchical (empirical-Bayes) elasticity estimation, `src/elasticity_model.py`

Two-stage partial-pooling estimator:

1. **Category level:** Ridge log-log regression (`log(units) ~ log(price) + log(competitor_price) + promo + season`)
   pooled across ~1,800 SKUs per category, stable, low-variance category-average elasticity.
2. **SKU level:** for each SKU, partial out the category-level cross-price/promo/seasonal effects, then run
   a simple OLS of the residual demand on `log(price)` using only that SKU's own ~26 weekly observations.
   This raw estimate is noisy, so it's shrunk toward the category mean with a **James-Stein / empirical-Bayes
   estimator**, SKUs with a precise individual signal keep more of their own estimate; SKUs with a noisy
   signal are pulled toward the category average.

This mirrors the real production tradeoff: a single category coefficient is too coarse for SKU-level pricing,
but a fully independent per-SKU fit overfits on ~26 data points. Partial pooling gets useful differentiation
without the overfitting, validated against synthetic ground truth in `data/raw/true_elasticity_holdout.csv`
(held out from model training, used only for offline evaluation).

### 2. Non-linear price optimization, `src/optimize_price.py`

For each SKU, given its estimated elasticity and unit cost, solve:

```
maximize   (p - cost) * q0 * (p / p0) ** elasticity
subject to  0.88 * p0 <= p <= 1.12 * p0        (merchandising guardrail: ±12%)
```

Solved two ways, cross-checked against each other: the closed-form constant-elasticity monopoly markup
(`p* = elasticity / (elasticity + 1) * cost`), and a bounded grid search directly over the profit function, 
the latter is what's actually used, since it generalizes to demand curves that aren't perfectly constant-
elasticity, and it's the method that respects the guardrail constraint.

All regression math (Ridge, closed-form OLS) is implemented from scratch in NumPy (`src/mllib.py`), 
no scikit-learn dependency.

## Data and scenario chronology

The synthetic training panel has **26 weekly starts from June 30 through December 22, 2025** (the last weekly interval ends December 28). `week` remains the model's zero-based seasonal index; `week_start_date` provides its explicit calendar label. These dates are assigned to synthetic observations, not collected retail transactions. Numeric inputs and random draws are unchanged.

Price recommendations and the illustrative annual margin estimate are **2026 scenarios**, based on the latest synthetic training week. They are not observed 2026 outcomes or a deployed pricing system. `reports/metrics.json` records the training window and scenario year; per-SKU results also identify their baseline week.

This repository was reorganized into six logical development stages using the requested January–February 2026 commit timeline. See [data/model notes](docs/data_model_notes.md) for the date convention and validation boundaries.

## Results

![Elasticity accuracy](screenshots/elasticity_accuracy.png)

| Metric | Value |
|---|---|
| Elasticity prediction score (100 − MAPE) | **79.2%** |
| SKU-level directional accuracy | 65.9% |
| Category rank correlation | 1.00 |
| R² (predicted vs. true elasticity) | 0.60 |

![Elasticity by category](screenshots/elasticity_by_category.png)

The estimated elasticities separate the synthetic categories: Electronics and Footwear have the most negative estimates, while Jewelry and Beauty have the least negative estimates,
reflecting the category assumptions built into this synthetic generator, rather than evidence about actual retail customers.

![Demand and profit curve](screenshots/demand_profit_curve_sample_sku.png)

Example SKU-level demand/profit curve showing the optimizer identifying a profit-maximizing price within the
±12% guardrail.

![Margin lift distribution](screenshots/margin_lift_distribution.png)

| Metric | Value |
|---|---|
| SKUs priced | 10,800 |
| **Aggregate margin lift** | **10.75%** |
| Median per-SKU profit lift | 10.54% |

### Estimated business impact (illustrative)

`src/business_impact.py` converts the margin lift into a dollar estimate. **All revenue/margin inputs are
illustrative placeholders**, not company figures, see `ASSUMPTIONS` in the script (the 65% rollout-coverage
discount reflects that not every recommended price is adopted by merchandising in practice):

```json
{
  "scenario_year": 2026,
  "elasticity_model_accuracy_pct": 79.22,
  "n_skus_priced": 10800,
  "aggregate_margin_lift_pct": 10.75,
  "baseline_annual_gross_margin_usd": 75600000.0,
  "theoretical_annual_margin_uplift_usd": 8127000.0,
  "realized_annual_margin_uplift_usd_at_65pct_rollout": 5282550.0
}
```

## Repo structure

```
Price-elasticity-Optimization/
├── data/
│   ├── raw/                              full synthetic panel (generated locally, gitignored)
│   │   ├── sku_price_demand_panel.csv        280K rows, 10,800 SKUs x 26 weeks
│   │   └── true_elasticity_holdout.csv       ground truth, used only for offline validation
│   └── sample/                          ~2,000-row preview for quick schema inspection
├── src/
│   ├── mllib.py                 Ridge regression + metrics, numpy only
│   ├── generate_synthetic_data.py
│   ├── elasticity_model.py      hierarchical/shrinkage elasticity estimation
│   ├── optimize_price.py        bounded non-linear price optimization
│   ├── train.py                 orchestrates the pipeline, writes reports/ and screenshots/
│   └── business_impact.py       margin lift -> $ impact estimate
├── reports/
│   ├── metrics.json
│   ├── business_impact.json
│   └── price_optimization_results.csv    per-SKU 2026 price scenarios
├── screenshots/
│   ├── elasticity_accuracy.png
│   ├── elasticity_by_category.png
│   ├── demand_profit_curve_sample_sku.png
│   └── margin_lift_distribution.png
├── docs/data_model_notes.md
├── tests/
│   ├── test_pipeline.py
│   └── test_chronology.py
├── requirements.txt
└── LICENSE
```

## Setup & running

**Requirements:** Python 3.10+ and the packages in `requirements.txt` (`numpy`, `pandas`, `matplotlib`, `pytest`).

```bash
# 1. Clone and enter the repo
git clone https://github.com/AKJ7320/Price-elasticity-Optimization.git
cd Price-elasticity-Optimization

# 2. (Recommended) create a virtual environment
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate the synthetic panel (writes data/raw/)
python src/generate_synthetic_data.py

# 5. Fit the elasticity model, run price optimization, write metrics to reports/ and figures to screenshots/
python src/train.py

# 6. Produce the illustrative business-impact estimate
python src/business_impact.py

# 7. Run the test suite
python -m pytest tests/
```

A ~2,000-row preview of each data file lives in `data/sample/` so you can inspect the schema without running
the generator.

## Next steps (if extended further)

- Replace the constant-elasticity demand-curve assumption with a locally-linear or GBM demand model for
  SKUs with wider historical price variation.
- Add competitive response modeling (how competitor_price itself reacts to a price change).
- A/B test allocation logic for staged rollout of recommended prices rather than a single coverage discount.

---
# Author

Adveidh Krishna

Financial Analyst

[LinkedIn](https://www.linkedin.com/in/akthotarath)
