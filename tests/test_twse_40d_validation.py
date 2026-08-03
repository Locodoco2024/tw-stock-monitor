from __future__ import annotations

import pandas as pd

from research.institutional_model.cli import build_parser
from research.institutional_model.market_model_spec import market_model_spec
from research.institutional_model.phase6_twse_40d import (
    Phase6BTWSE40DSettings,
    build_phase6b_summary,
    prepare_twse_40d_frame,
)


def _metric_map(frame: pd.DataFrame) -> dict[str, str]:
    return dict(zip(frame["metric"].astype(str), frame["value"].astype(str), strict=True))


def test_phase6b_cli_command_is_available() -> None:
    args = build_parser().parse_args(["phase6b"])
    assert args.command == "phase6b"


def test_prepare_twse_40d_frame_builds_40d_cross_sectional_target() -> None:
    features = market_model_spec("twse").feature_columns
    rows = []
    for stock_id, value in (("1101", -0.10), ("2330", 0.00), ("2603", 0.20)):
        row = {
            "stock_id": stock_id,
            "stock_name": stock_id,
            "signal_date": "2025-01-02",
            "signal_year": 2025,
            "target_date_40d": "2025-03-05",
            "label_status_40d": "ok",
            "adjusted_return_20d": value / 2,
            "adjusted_return_40d": value,
        }
        row.update({feature: 0.0 for feature in features})
        rows.append(row)
    result = prepare_twse_40d_frame(
        pd.DataFrame(rows),
        settings=Phase6BTWSE40DSettings(minimum_daily_stocks=3),
    )
    ranks = result.set_index("stock_id")["future_return_rank_pct_40d"]
    assert ranks["1101"] < ranks["2330"] < ranks["2603"]
    assert result["target_date"].eq("2025-03-05").all()


def test_phase6b_gate_passes_only_for_40d_primary_result() -> None:
    summary = build_phase6b_summary(
        frame=pd.DataFrame({"stock_id": ["2330"], "signal_date": ["2025-01-02"]}),
        fold_summary=pd.DataFrame({"status": ["complete"]}),
        comparison=pd.DataFrame(
            [
                {
                    "period": "confirmation_ex_latest",
                    "score_variant": "return_rank_40d_score",
                    "evaluation_horizon_days": 40,
                    "top20_minus_bottom20": 0.012,
                    "positive_years": 3,
                    "total_years": 3,
                    "average_daily_spearman": 0.04,
                },
                {
                    "period": "confirmation_ex_latest",
                    "score_variant": "return_rank_40d_score",
                    "evaluation_horizon_days": 20,
                    "top20_minus_bottom20": -0.02,
                    "positive_years": 0,
                    "total_years": 3,
                    "average_daily_spearman": -0.04,
                },
            ]
        ),
        bootstrap=pd.DataFrame(
            [
                {
                    "period": "confirmation_ex_latest",
                    "score_variant": "return_rank_40d_score",
                    "evaluation_horizon_days": 40,
                    "ci_lower": 0.002,
                    "ci_upper": 0.021,
                }
            ]
        ),
        settings=Phase6BTWSE40DSettings(),
    )
    metrics = _metric_map(summary)
    assert metrics["model_target"] == "same_day_future_return_rank_40d"
    assert metrics["return_rank_40d_validation_pass"] == "1"
    assert metrics["deployment_ready"] == "0"


def test_phase6b_gate_fails_when_40d_bootstrap_crosses_zero() -> None:
    summary = build_phase6b_summary(
        frame=pd.DataFrame({"stock_id": ["2330"], "signal_date": ["2025-01-02"]}),
        fold_summary=pd.DataFrame({"status": ["complete"]}),
        comparison=pd.DataFrame(
            [
                {
                    "period": "confirmation_ex_latest",
                    "score_variant": "return_rank_40d_score",
                    "evaluation_horizon_days": 40,
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
                    "score_variant": "return_rank_40d_score",
                    "evaluation_horizon_days": 40,
                    "ci_lower": -0.001,
                    "ci_upper": 0.021,
                }
            ]
        ),
        settings=Phase6BTWSE40DSettings(),
    )
    assert _metric_map(summary)["return_rank_40d_validation_pass"] == "0"
