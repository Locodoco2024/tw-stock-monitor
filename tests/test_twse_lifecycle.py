from __future__ import annotations

import numpy as np
import pandas as pd

from research.institutional_model.phase6_twse_lifecycle import (
    Phase6CTWSELifecycleSettings,
    annotate_runs,
    apply_cooldown,
    build_liquidity_universe,
    build_rule_candidates,
    build_universe_entry_events,
)


def _base_rows() -> pd.DataFrame:
    rows = []
    dates = [f"2024-01-{day:02d}" for day in range(2, 8)]
    for day_index, signal_date in enumerate(dates):
        for stock_index, stock_id in enumerate(("1101", "1102", "1103", "1104")):
            rows.append(
                {
                    "stock_id": stock_id,
                    "stock_name": stock_id,
                    "signal_date": signal_date,
                    "signal_year": 2024,
                    "test_year": 2024,
                    "market_day_index": day_index,
                    "return_rank_40d_score": float(stock_index),
                    "return_rank_40d_score_daily_percentile": (stock_index + 1) * 25,
                    "adjusted_return_20d": 0.01 * stock_index,
                    "adjusted_return_40d": 0.02 * stock_index,
                    "median_trading_money_20d": 120_000_000.0,
                    "median_trading_volume_lots_20d": 400.0,
                    "liquidity_pass_20m": 1,
                    "liquidity_pass_50m": 1,
                    "liquidity_pass_100m": 1,
                    "signal_close": 100.0,
                    "signal_trading_volume_lots": 500.0,
                    "signal_trading_money": 50_000_000.0,
                }
            )
    return pd.DataFrame(rows)


def test_stricter_liquidity_universe_recomputes_percentile() -> None:
    frame = _base_rows()
    frame.loc[frame["stock_id"] == "1104", "median_trading_volume_lots_20d"] = 50
    ranked, coverage = build_liquidity_universe(
        frame,
        universe_id="test",
        minimum_money_million=100,
        minimum_volume_lots=100,
        minimum_daily_stocks=3,
    )
    assert set(ranked["stock_id"]) == {"1101", "1102", "1103"}
    top = ranked[ranked["stock_id"] == "1103"]
    assert np.allclose(top["universe_percentile"], 100.0)
    assert int(coverage.iloc[0]["minimum_daily_stocks"]) == 3


def test_run_requires_consecutive_market_days() -> None:
    frame = _base_rows()
    selected = frame[frame["stock_id"] == "1104"].copy()
    selected["universe_percentile"] = 95.0
    selected = selected[selected["market_day_index"] != 2]
    annotated = annotate_runs(selected, threshold=90.0)
    assert annotated["run_length"].tolist() == [1, 2, 1, 2, 3]


def test_entry_event_does_not_use_future_return_to_trigger() -> None:
    frame = _base_rows()
    ranked, _ = build_liquidity_universe(
        frame,
        universe_id="test",
        minimum_money_million=20,
        minimum_volume_lots=0,
        minimum_daily_stocks=4,
    )
    settings = Phase6CTWSELifecycleSettings(
        minimum_daily_stocks=20,
        entry_thresholds=(75.0,),
        confirmation_days=(1, 3),
        bootstrap_iterations=200,
    )
    first = build_universe_entry_events(ranked, settings=settings)[
        ["event_id", "signal_date"]
    ]
    ranked["adjusted_return_40d"] = np.linspace(-0.9, 0.9, len(ranked))
    second = build_universe_entry_events(ranked, settings=settings)[
        ["event_id", "signal_date"]
    ]
    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))


def test_cooldown_filters_same_stock_events() -> None:
    events = pd.DataFrame(
        {
            "stock_id": ["1101", "1101", "1101", "1102"],
            "market_day_index": [1, 5, 12, 2],
        }
    )
    selected = apply_cooldown(events, cooldown_days=10)
    assert selected.index.tolist() == [0, 2, 3]


def test_candidate_requires_positive_bootstrap_and_three_years() -> None:
    comparison = pd.DataFrame(
        [
            {
                "period": "confirmation_ex_latest",
                "universe_id": "money100m_volume100lots",
                "entry_threshold": 90.0,
                "confirmation_days": 5,
                "entry_rule": "top10_confirm5d",
                "event_count": 150,
                "unique_stocks": 60,
                "excess_return_40d_daily_equal_weight": 0.02,
                "maximum_drawdown_40d_daily_equal_weight": -0.08,
            }
        ]
    )
    yearly = pd.DataFrame(
        [
            {
                "universe_id": "money100m_volume100lots",
                "entry_threshold": 90.0,
                "confirmation_days": 5,
                "entry_rule": "top10_confirm5d",
                "test_year": year,
                "excess_return_40d_daily_equal_weight": value,
            }
            for year, value in ((2023, 0.01), (2024, -0.001), (2025, 0.03), (2026, 0.04))
        ]
    )
    bootstrap = pd.DataFrame(
        [
            {
                "period": "confirmation_ex_latest",
                "universe_id": "money100m_volume100lots",
                "entry_threshold": 90.0,
                "confirmation_days": 5,
                "entry_rule": "top10_confirm5d",
                "ci_lower": 0.002,
                "ci_upper": 0.04,
                "ci_excludes_zero_positive": 1,
            }
        ]
    )
    result = build_rule_candidates(
        comparison=comparison,
        yearly=yearly,
        bootstrap=bootstrap,
        settings=Phase6CTWSELifecycleSettings(bootstrap_iterations=200),
    )
    assert result.iloc[0]["candidate_status"] == "strong_candidate"


def test_phase6c_cli_command_is_available() -> None:
    from research.institutional_model.cli import build_parser

    args = build_parser().parse_args(["phase6c"])
    assert args.command == "phase6c"
