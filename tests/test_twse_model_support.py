from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from research.institutional_model.market_model_spec import market_model_spec
from research.institutional_model.phase4_horizon import (
    HorizonFold,
    Phase4DHorizonSettings,
)
from research.institutional_model.phase4_target import (
    Phase4ETargetSettings,
    build_target_summary,
)
from research.institutional_model.phase5_final_model import validate_phase5d_inputs


def _metric_map(frame: pd.DataFrame) -> dict[str, str]:
    return dict(
        zip(
            frame["metric"].astype(str),
            frame["value"].astype(str),
            strict=True,
        )
    )




def test_market_specs_preserve_phase4b_selected_feature_sets() -> None:
    twse = market_model_spec("twse")
    tpex = market_model_spec("tpex")
    assert twse.candidate_id == "full40_l2_1e-3"
    assert len(twse.feature_columns) == 40
    assert tpex.candidate_id == "core22_l2_1e-3"
    assert len(tpex.feature_columns) == 22

def test_twse_horizon_settings_and_fold_id() -> None:
    settings = Phase4DHorizonSettings(target_market="twse")
    settings.validate()
    fold = HorizonFold(
        horizon_days=20,
        train_start_year=2015,
        train_end_year=2022,
        calibration_year=2023,
        test_year=2024,
        market="twse",
    )
    assert fold.fold_id == "twse_20d_2024"


def test_twse_return_rank_gate_passes_with_stable_confirmation() -> None:
    summary = build_target_summary(
        frame=pd.DataFrame(
            {
                "stock_id": ["2330"],
                "signal_date": ["2025-01-02"],
            }
        ),
        fold_summary=pd.DataFrame({"status": ["complete"]}),
        comparison=pd.DataFrame(
            [
                {
                    "comparison_type": "ranking",
                    "period": "confirmation_ex_latest",
                    "target_or_score": "return_rank_score",
                    "variant_or_horizon": 20,
                    "top20_minus_bottom20": 0.012,
                    "positive_years": 3,
                    "total_years": 3,
                    "average_daily_spearman": 0.04,
                }
            ]
        ),
        bootstrap=pd.DataFrame(
            [
                {
                    "period": "confirmation_ex_latest",
                    "score_variant": "return_rank_score",
                    "evaluation_horizon_days": 20,
                    "ci_lower": 0.002,
                    "ci_upper": 0.021,
                    "ci_excludes_zero_positive": 1,
                }
            ]
        ),
        settings=Phase4ETargetSettings(target_market="twse"),
    )
    metrics = _metric_map(summary)
    assert metrics["market"] == "twse"
    assert metrics["candidate"] == "full40_l2_1e-3"
    assert metrics["model_features"] == "40"
    assert metrics["return_rank_validation_pass"] == "1"


def test_twse_return_rank_gate_fails_when_bootstrap_crosses_zero() -> None:
    summary = build_target_summary(
        frame=pd.DataFrame(
            {
                "stock_id": ["2330"],
                "signal_date": ["2025-01-02"],
            }
        ),
        fold_summary=pd.DataFrame({"status": ["complete"]}),
        comparison=pd.DataFrame(
            [
                {
                    "comparison_type": "ranking",
                    "period": "confirmation_ex_latest",
                    "target_or_score": "return_rank_score",
                    "variant_or_horizon": 20,
                    "top20_minus_bottom20": 0.012,
                    "positive_years": 3,
                    "total_years": 3,
                    "average_daily_spearman": 0.04,
                }
            ]
        ),
        bootstrap=pd.DataFrame(
            [
                {
                    "period": "confirmation_ex_latest",
                    "score_variant": "return_rank_score",
                    "evaluation_horizon_days": 20,
                    "ci_lower": -0.001,
                    "ci_upper": 0.021,
                    "ci_excludes_zero_positive": 0,
                }
            ]
        ),
        settings=Phase4ETargetSettings(target_market="twse"),
    )
    assert _metric_map(summary)["return_rank_validation_pass"] == "0"


def _write_metric_csv(path: Path, values: dict[str, object]) -> None:
    pd.DataFrame(
        [{"metric": key, "value": value} for key, value in values.items()]
    ).to_csv(path, index=False, encoding="utf-8-sig")


def test_twse_final_model_requires_rank_validation_pass(tmp_path: Path) -> None:
    _write_metric_csv(
        tmp_path / "phase3b_summary.csv",
        {"status": "PASS", "ready_for_modeling": 1},
    )
    _write_metric_csv(
        tmp_path / "phase4e_summary.csv",
        {
            "market": "twse",
            "primary_horizon_days": 20,
            "ready_for_target_decision": 1,
            "return_rank_validation_pass": 0,
        },
    )
    pd.DataFrame([{"status": "complete"}]).to_csv(
        tmp_path / "phase4e_model_comparison.csv", index=False
    )
    pd.DataFrame([{"status": "complete"}]).to_csv(
        tmp_path / "phase4e_coefficient_stability.csv", index=False
    )
    pd.DataFrame([{"stock_id": "2330"}]).to_csv(
        tmp_path / "phase4e_oos_scores.csv.gz", index=False, compression="gzip"
    )
    _write_metric_csv(
        tmp_path / "phase4f_summary.csv",
        {
            "market": "twse",
            "pipeline_status": "PASS",
            "ready_for_lifecycle_decision": 1,
            "score_column": "return_rank_score_daily_percentile",
        },
    )
    pd.DataFrame([{"decision_status": "review"}]).to_csv(
        tmp_path / "phase4f_rule_candidates.csv", index=False
    )

    with pytest.raises(RuntimeError, match="樣本外驗證未通過"):
        validate_phase5d_inputs(tmp_path, target_market="twse")
