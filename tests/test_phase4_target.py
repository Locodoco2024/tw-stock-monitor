from __future__ import annotations

import numpy as np
import pandas as pd

from research.institutional_model.phase4_horizon import classify_horizon_return
from research.institutional_model.phase4_stability import CORE_FEATURE_COLUMNS
from research.institutional_model.phase4_target import (
    Phase4ETargetSettings,
    binary_probability_metrics,
    evaluate_target_frame,
    feature_group,
    fit_binary_calibrator,
    prepare_target_frame,
    sigmoid,
)


def test_binary_calibrator_keeps_order_and_reduces_shifted_loss() -> None:
    logits = np.linspace(-2.0, 2.0, 400)
    labels = (logits + 0.7 > 0).astype(np.uint8)
    raw = sigmoid(logits)
    calibrator = fit_binary_calibrator(logits, labels)
    calibrated = calibrator.transform_logits(logits)

    assert calibrator.slope > 0
    assert np.all(np.diff(calibrated) >= 0)
    assert binary_probability_metrics(labels, calibrated)["log_loss"] < (
        binary_probability_metrics(labels, raw)["log_loss"]
    )


def test_prepare_target_frame_builds_daily_future_return_percentile() -> None:
    frame = pd.DataFrame(
        {
            "stock_id": [f"S{index:02d}" for index in range(10)],
            "stock_name": [f"股票{index:02d}" for index in range(10)],
            "signal_date": ["2024-01-02"] * 10,
            "signal_year": [2024] * 10,
            "adjusted_return_20d": np.linspace(-0.1, 0.1, 10),
            "label_status_20d": ["ok"] * 10,
            "label_20d": [
                classify_horizon_return(value, threshold=0.05)
                for value in np.linspace(-0.1, 0.1, 10)
            ],
            "target_date_20d": ["2024-01-30"] * 10,
            "adjusted_return_40d": np.linspace(-0.2, 0.2, 10),
        }
    )
    selected = prepare_target_frame(
        frame,
        settings=Phase4ETargetSettings(minimum_daily_stocks=10),
    )

    assert selected["future_return_rank_pct_20d"].is_monotonic_increasing
    assert selected.iloc[-1]["actual_top20_20d"] == 1
    assert selected.iloc[0]["actual_bottom20_20d"] == 1
    assert selected["target_up"].sum() > 0
    assert selected["target_down"].sum() > 0


def test_feature_groups_cover_all_core_features() -> None:
    groups = {feature_group(feature) for feature in CORE_FEATURE_COLUMNS}
    assert groups == {
        "foreign",
        "investment_trust",
        "dealer_self",
        "institutional_consensus",
    }


def test_phase4e_compares_binary_and_ranking_targets() -> None:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(1234)
    for year in range(2015, 2025):
        for month in (2, 5, 8, 11):
            signal_date = f"{year}-{month:02d}-03"
            for stock_index in range(30):
                latent = (stock_index - 14.5) / 14.5
                noise = rng.normal(0.0, 0.005)
                return_20d = 0.11 * latent + noise
                return_40d = 0.16 * latent + noise
                row: dict[str, object] = {
                    "stock_id": f"S{stock_index:02d}",
                    "stock_name": f"股票{stock_index:02d}",
                    "signal_date": signal_date,
                    "signal_year": year,
                    "adjusted_return_20d": return_20d,
                    "target_date_20d": f"{year}-{month:02d}-25",
                    "label_status_20d": "ok",
                    "label_20d": classify_horizon_return(
                        return_20d, threshold=0.05
                    ),
                    "adjusted_return_40d": return_40d,
                    "target_date_40d": f"{year}-{month:02d}-27",
                    "label_status_40d": "ok",
                    "label_40d": classify_horizon_return(
                        return_40d, threshold=0.05
                    ),
                }
                for feature_index, feature in enumerate(CORE_FEATURE_COLUMNS):
                    row[feature] = (
                        latent * (1.0 if feature_index < 6 else 0.25)
                        + feature_index * 0.001
                        + rng.normal(0.0, 0.01)
                    )
                rows.append(row)
    frame = pd.DataFrame(rows)
    settings = Phase4ETargetSettings(
        minimum_daily_stocks=10,
        quantile_sample_size=1_000,
        training_chunk_size=500,
        batch_size=128,
        maximum_epochs=3,
        minimum_epochs=2,
        early_stopping_patience=1,
        bootstrap_iterations=200,
        bootstrap_block_months=2,
    )

    reports = evaluate_target_frame(frame=frame, settings=settings)

    assert reports["fold_summary"]["status"].eq("complete").all()
    assert len(reports["fold_summary"]) == 6
    assert set(reports["probability_metrics"]["target_name"]) == {
        "up_20d",
        "down_20d",
    }
    assert set(reports["yearly_ranking"]["score_variant"]) == {
        "multinomial_index",
        "p_up_platt",
        "binary_net_score",
        "return_rank_score",
    }
    rank_rows = reports["yearly_ranking"][
        (reports["yearly_ranking"]["score_variant"] == "return_rank_score")
        & (reports["yearly_ranking"]["evaluation_horizon_days"] == 20)
    ]
    assert rank_rows["top20_minus_bottom20"].gt(0).all()
    assert rank_rows["average_daily_spearman"].gt(0).all()
    assert reports["group_contributions"]["model_target"].nunique() == 3
    assert reports["oos_scores"].duplicated(["stock_id", "signal_date"]).sum() == 0
