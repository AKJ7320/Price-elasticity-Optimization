"""Validate the synthetic calendar and its separation from model inputs."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import business_impact
import elasticity_model
import train
from generate_synthetic_data import generate
from optimize_price import optimize_all_skus


def test_calendar_labels_leave_numeric_draws_unchanged():
    panel, truth = generate(n_skus_per_cat=2)
    shifted, shifted_truth = generate(n_skus_per_cat=2, start_date="2024-06-30")
    assert panel.week_start_date.min() == "2025-06-30"
    assert panel.week_start_date.max() == "2025-12-22"
    assert panel.groupby("sku_id").size().eq(26).all()
    assert (pd.to_datetime(panel.week_start_date).dt.dayofweek == 0).all()
    assert_frame_equal(panel.drop(columns="week_start_date"),
                       shifted.drop(columns="week_start_date"))
    assert_frame_equal(truth, shifted_truth)
    assert_frame_equal(elasticity_model.build_features(panel).drop(columns="week_start_date"),
                       elasticity_model.build_features(shifted).drop(columns="week_start_date"))


def test_scenario_uses_last_training_week_without_changing_prices():
    panel, truth = generate(n_skus_per_cat=2)
    estimates = truth.rename(columns={"true_elasticity": "predicted_elasticity"})
    result = optimize_all_skus(panel, estimates)
    undated = optimize_all_skus(panel.drop(columns="week_start_date"), estimates)
    assert result.scenario_year.eq(2026).all()
    assert result.baseline_week_start_date.eq("2025-12-22").all()
    assert_frame_equal(result.drop(columns="baseline_week_start_date"), undated)
    assert np.isfinite(result.optimal_profit_estimate).all()
    assert result.price_change_pct.between(-12, 12).all()


def test_end_to_end_reports_and_business_impact(tmp_path, monkeypatch):
    panel, truth = generate(n_skus_per_cat=5)
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    panel.to_csv(raw / "sku_price_demand_panel.csv", index=False)
    truth.to_csv(raw / "true_elasticity_holdout.csv", index=False)
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(train, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(train, "FIGURES", tmp_path / "screenshots")
    monkeypatch.setattr(elasticity_model, "ROOT", tmp_path)
    monkeypatch.setattr(business_impact, "ROOT", tmp_path)
    metrics = train.main()
    assert metrics["n_skus_priced"] == 30
    assert metrics["training_period"] == {
        "week_start_min": "2025-06-30", "week_start_max": "2025-12-22"
    }
    assert metrics["scenario_year"] == 2026
    assert {p.name for p in (tmp_path / "screenshots").glob("*.png")} == {
        "elasticity_accuracy.png", "elasticity_by_category.png",
        "demand_profit_curve_sample_sku.png", "margin_lift_distribution.png"
    }
    impact = business_impact.estimate_annual_uplift()
    assert impact["scenario_year"] == 2026
    assert impact["theoretical_annual_margin_uplift_usd"] == round(
        impact["baseline_annual_gross_margin_usd"] * metrics["aggregate_margin_lift_pct"] / 100, 0
    )
