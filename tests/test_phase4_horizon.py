from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research.institutional_model.database import ResearchDatabase
from research.institutional_model.phase4_horizon import (
    HorizonFold,
    Phase4DHorizonSettings,
    _calculate_stock_horizon_outcomes,
    assign_same_day_ranks,
    build_purged_masks,
    classify_horizon_return,
    daily_rank_group_rows,
    evaluate_horizon_frame,
)
from research.institutional_model.phase4_stability import CORE_FEATURE_COLUMNS


def test_horizon_label_uses_rounded_10_decimal_boundary() -> None:
    assert classify_horizon_return(0.04999999996, threshold=0.05) == "UP"
    assert classify_horizon_return(-0.04999999996, threshold=0.05) == "DOWN"
    assert classify_horizon_return(0.04999999994, threshold=0.05) == "FLAT"
    assert classify_horizon_return(-0.04999999994, threshold=0.05) == "FLAT"


def test_purged_masks_remove_targets_crossing_next_period() -> None:
    frame = pd.DataFrame(
        {
            "signal_year": [2017, 2017, 2018, 2018, 2019],
            "target_date": [
                "2017-11-30",
                "2018-01-15",
                "2018-11-30",
                "2019-01-20",
                "2019-03-01",
            ],
        }
    )
    fold = HorizonFold(
        horizon_days=40,
        train_start_year=2015,
        train_end_year=2017,
        calibration_year=2018,
        test_year=2019,
    )
    train, calibration, test, purge = build_purged_masks(frame, fold)

    assert train.tolist() == [True, False, False, False, False]
    assert calibration.tolist() == [False, False, True, False, False]
    assert test.tolist() == [False, False, False, False, True]
    assert purge["train_rows_purged"] == 1
    assert purge["calibration_rows_purged"] == 1


def test_same_day_ranking_and_group_rows() -> None:
    frame = pd.DataFrame(
        {
            "stock_id": [f"S{index:02d}" for index in range(20)],
            "signal_date": ["2024-01-02"] * 20,
            "institutional_index_raw": np.arange(20, dtype=float),
            "adjusted_return": np.linspace(-0.10, 0.10, 20),
            "label": ["DOWN"] * 5 + ["FLAT"] * 10 + ["UP"] * 5,
        }
    )
    ranked = assign_same_day_ranks(frame, minimum_daily_stocks=10)
    rows = daily_rank_group_rows(ranked, horizon=20, test_year=2024)

    assert ranked["daily_rank"].tolist() == list(range(1, 21))
    assert ranked.iloc[0]["daily_percentile"] == 0
    assert ranked.iloc[-1]["daily_percentile"] == 100
    assert len(rows) == 10
    assert rows[-1]["average_return"] > rows[0]["average_return"]


def test_calculate_40d_outcome_uses_t_plus_one_open(tmp_path: Path) -> None:
    database = ResearchDatabase(tmp_path / "research.sqlite3")
    database.initialize()
    dates = pd.bdate_range("2020-01-02", periods=45).strftime("%Y-%m-%d").tolist()
    database.executemany(
        "INSERT INTO market_calendar(date) VALUES (?)",
        [(value,) for value in dates],
    )
    price_rows = []
    for index, value in enumerate(dates):
        price = 100.0 + index
        price_rows.append(("1234", value, 1000, 1_000_000, price, price, price, price))
    database.executemany(
        """
        INSERT INTO stock_prices(
            stock_id, date, trading_volume, trading_money, open, high, low, close
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        price_rows,
    )

    outcome = _calculate_stock_horizon_outcomes(
        database=database,
        stock_id="1234",
        signal_dates=[dates[0]],
        market_dates=dates,
        horizon=40,
        threshold=0.05,
    )

    assert outcome.iloc[0]["label_status_40d"] == "ok"
    assert outcome.iloc[0]["entry_date_40d"] == dates[1]
    assert outcome.iloc[0]["target_date_40d"] == dates[40]
    expected = (100.0 + 40) / (100.0 + 1) - 1
    assert np.isclose(outcome.iloc[0]["adjusted_return_40d"], expected)
    assert outcome.iloc[0]["label_40d"] == "UP"


def test_phase4d_evaluates_10_20_40_with_same_features() -> None:
    rows: list[dict[str, object]] = []
    for year in range(2015, 2025):
        for month in (2, 5, 8, 11):
            signal_date = f"{year}-{month:02d}-03"
            for stock_index in range(12):
                class_index = stock_index % 3
                direction = (-1.0, 0.0, 1.0)[class_index]
                row: dict[str, object] = {
                    "stock_id": f"S{stock_index:02d}",
                    "stock_name": f"股票{stock_index:02d}",
                    "signal_date": signal_date,
                    "signal_year": year,
                }
                for feature_index, feature in enumerate(CORE_FEATURE_COLUMNS):
                    row[feature] = direction if feature_index == 0 else (
                        direction * 0.1 + feature_index * 0.001
                    )
                for horizon, magnitude in ((10, 0.06), (20, 0.08), (40, 0.10)):
                    value = direction * magnitude
                    row[f"adjusted_return_{horizon}d"] = value
                    row[f"target_date_{horizon}d"] = f"{year}-{month:02d}-25"
                    row[f"label_status_{horizon}d"] = "ok"
                    row[f"label_{horizon}d"] = classify_horizon_return(
                        value,
                        threshold=0.05,
                    )
                rows.append(row)
    frame = pd.DataFrame(rows)
    settings = Phase4DHorizonSettings(
        minimum_daily_stocks=10,
        quantile_sample_size=500,
        training_chunk_size=500,
        batch_size=128,
        maximum_epochs=3,
        minimum_epochs=2,
        early_stopping_patience=1,
    )

    reports = evaluate_horizon_frame(frame=frame, settings=settings)

    summary = reports["fold_summary"]
    assert set(summary["horizon_days"]) == {10, 20, 40}
    assert summary["status"].eq("complete").all()
    assert len(summary) == 18
    assert reports["yearly"]["top20_minus_bottom20"].gt(0).all()
    assert reports["boundary_purge"]["train_rows_purged"].eq(0).all()
