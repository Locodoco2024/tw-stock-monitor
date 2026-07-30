from __future__ import annotations

import numpy as np
import pandas as pd

from research.institutional_model.phase4_lifecycle import (
    Phase4FLifecycleSettings,
    annotate_threshold_runs,
    build_entry_events,
    evaluate_lifecycle_scores,
    prepare_lifecycle_frame,
)


def _base_rows(dates: list[str], stocks: int = 10) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_index, signal_date in enumerate(dates):
        year = int(signal_date[:4])
        for stock_index in range(stocks):
            score = 100.0 * (stock_index + 1) / stocks
            rows.append(
                {
                    "stock_id": f"S{stock_index:02d}",
                    "stock_name": f"股票{stock_index:02d}",
                    "signal_date": signal_date,
                    "signal_year": year,
                    "test_year": year,
                    "adjusted_return_20d": (score - 50.0) / 1000.0,
                    "adjusted_return_40d": (score - 50.0) / 600.0,
                    "return_rank_score_daily_percentile": score,
                }
            )
    return pd.DataFrame(rows)


def test_threshold_run_breaks_when_stock_misses_market_day() -> None:
    frame = _base_rows(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        stocks=10,
    )
    frame = frame[
        ~((frame["stock_id"] == "S09") & (frame["signal_date"] == "2024-01-03"))
    ]
    settings = Phase4FLifecycleSettings(minimum_daily_stocks=9)
    prepared = prepare_lifecycle_frame(frame, settings=settings)
    annotated = annotate_threshold_runs(
        prepared,
        threshold=90.0,
        momentum_lookback_days=1,
    )
    stock = annotated[
        (annotated["stock_id"] == "S09") & (annotated["qualifies"] == 1)
    ].sort_values("signal_date")

    assert stock["run_length"].tolist() == [1, 1, 2]


def test_confirmation_event_occurs_on_confirmation_day() -> None:
    frame = _base_rows(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        stocks=10,
    )
    settings = Phase4FLifecycleSettings(minimum_daily_stocks=10)
    prepared = prepare_lifecycle_frame(frame, settings=settings)
    annotated = annotate_threshold_runs(
        prepared,
        threshold=90.0,
        momentum_lookback_days=1,
    )
    events = build_entry_events(
        annotated,
        threshold=90.0,
        confirmation_days=(1, 3),
    )
    stock = events[events["stock_id"] == "S09"].sort_values("confirmation_days")

    assert stock["signal_date"].tolist() == ["2024-01-02", "2024-01-04"]
    assert stock["confirmation_days"].tolist() == [1, 3]
    assert (
        stock["market_day_index"]
        == stock["episode_start_index"] + stock["confirmation_days"] - 1
    ).all()


def test_phase4f_generates_lifecycle_reports_without_future_entry_features() -> None:
    rng = np.random.default_rng(2026)
    dates: list[str] = []
    for year in range(2022, 2026):
        dates.extend(
            pd.bdate_range(f"{year}-01-03", periods=35).strftime("%Y-%m-%d").tolist()
        )
    rows: list[dict[str, object]] = []
    stock_count = 30
    for date_index, signal_date in enumerate(dates):
        year = int(signal_date[:4])
        latent_values = np.array(
            [
                (stock_index - (stock_count - 1) / 2) / stock_count
                + 0.9 * np.sin(date_index / 5.0 + stock_index * 0.7)
                for stock_index in range(stock_count)
            ]
        )
        order = pd.Series(latent_values).rank(method="average", pct=True).to_numpy() * 100
        for stock_index, percentile in enumerate(order):
            future_signal = (percentile - 50.0) / 100.0
            rows.append(
                {
                    "stock_id": f"S{stock_index:02d}",
                    "stock_name": f"股票{stock_index:02d}",
                    "signal_date": signal_date,
                    "signal_year": year,
                    "test_year": year,
                    "adjusted_return_20d": 0.025 * future_signal
                    + rng.normal(0.0, 0.01),
                    "adjusted_return_40d": 0.045 * future_signal
                    + rng.normal(0.0, 0.015),
                    "return_rank_score_daily_percentile": percentile,
                }
            )
    frame = pd.DataFrame(rows)
    settings = Phase4FLifecycleSettings(
        minimum_daily_stocks=20,
        bootstrap_iterations=200,
        bootstrap_block_months=2,
    )

    reports = evaluate_lifecycle_scores(frame, settings=settings)

    assert len(reports["entry_events"]) > 0
    assert set(reports["entry_events"]["entry_threshold"]) == {80.0, 90.0, 95.0}
    assert set(reports["entry_events"]["confirmation_days"]) == {1, 2, 3, 5}
    assert not reports["entry_rule_comparison"].empty
    assert not reports["signal_age"].empty
    assert not reports["extension"].empty
    assert not reports["breakdown"].empty
    assert not reports["cooldown"].empty
    assert len(reports["rule_candidates"]) == 12
    summary = dict(zip(reports["summary"]["metric"], reports["summary"]["value"]))
    assert int(summary["ready_for_lifecycle_decision"]) == 1


def test_daily_excess_return_uses_same_day_universe_benchmark() -> None:
    frame = _base_rows(
        ["2024-01-02", "2024-01-03"],
        stocks=10,
    )
    prepared = prepare_lifecycle_frame(
        frame,
        settings=Phase4FLifecycleSettings(minimum_daily_stocks=10),
    )

    daily_excess = prepared.groupby("signal_date")[
        ["excess_adjusted_return_20d", "excess_adjusted_return_40d"]
    ].mean()
    assert np.allclose(daily_excess.to_numpy(dtype=float), 0.0, atol=1e-12)
