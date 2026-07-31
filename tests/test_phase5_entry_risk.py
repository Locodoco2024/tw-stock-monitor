from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research.institutional_model.phase5_entry_risk import (
    Phase5ISettings,
    build_event_cost_features,
    build_quantile_analysis,
    _spearman,
    load_phase3_event_outcomes,
    prepare_phase5i_events,
)


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "A:top10_confirm1d:2024-01-08",
                "entry_rule": "top10_confirm1d",
                "stock_id": "A",
                "stock_name": "甲",
                "signal_date": "2024-01-08",
                "test_year": 2024,
                "return_rank_score_daily_percentile": 95.0,
                "adjusted_return_20d": 0.03,
                "adjusted_return_40d": 0.05,
                "excess_adjusted_return_20d": 0.02,
                "excess_adjusted_return_40d": 0.03,
            },
            {
                "event_id": "B:top20_confirm5d:2024-01-08",
                "entry_rule": "top20_confirm5d",
                "stock_id": "B",
                "stock_name": "乙",
                "signal_date": "2024-01-08",
                "test_year": 2024,
                "return_rank_score_daily_percentile": 87.0,
                "adjusted_return_20d": 0.02,
                "adjusted_return_40d": 0.04,
                "excess_adjusted_return_20d": 0.01,
                "excess_adjusted_return_40d": 0.02,
            },
            {
                "event_id": "C:top5_confirm2d:2024-01-08",
                "entry_rule": "top5_confirm2d",
                "stock_id": "C",
                "stock_name": "丙",
                "signal_date": "2024-01-08",
                "test_year": 2024,
                "return_rank_score_daily_percentile": 98.0,
                "adjusted_return_20d": 0.04,
                "adjusted_return_40d": 0.06,
                "excess_adjusted_return_20d": 0.03,
                "excess_adjusted_return_40d": 0.04,
            },
        ]
    )


def _history() -> pd.DataFrame:
    dates = pd.bdate_range("2023-12-04", periods=27)
    rows: list[dict[str, object]] = []
    for stock_id, offset in (("A", 0.0), ("B", 10.0)):
        for index, date in enumerate(dates):
            close = 100.0 + offset + index
            rows.append(
                {
                    "stock_id": stock_id,
                    "date": date,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "trading_volume": 100_000,
                    "trading_money": 10_000_000,
                    "foreign_net": 100 if index % 2 == 0 else -50,
                    "investment_trust_net": 50,
                    "dealer_self_net": 25,
                    "selected_total_net": 175 if index % 2 == 0 else 25,
                }
            )
    return pd.DataFrame(rows)


def test_prepare_phase5i_events_keeps_only_formal_rules() -> None:
    prepared = prepare_phase5i_events(_events(), settings=Phase5ISettings())

    assert prepared["stock_id"].tolist() == ["A", "B"]
    assert set(prepared["event_type"]) == {
        "NEW_CANDIDATE",
        "LAYOUT_CONFIRMED_DIRECT",
    }


def test_cost_proxy_uses_positive_net_buy_shares_and_entry_open() -> None:
    events = prepare_phase5i_events(_events(), settings=Phase5ISettings())
    result = build_event_cost_features(
        events,
        history=_history(),
        settings=Phase5ISettings(),
    )
    row = result[result["stock_id"] == "A"].iloc[0]
    history = _history()
    stock = history[history["stock_id"] == "A"].copy().reset_index(drop=True)
    signal_index = stock.index[stock["date"] == pd.Timestamp("2024-01-08")][0]
    window = stock.iloc[signal_index - 4 : signal_index + 1]
    weights = window["foreign_net"].clip(lower=0)
    typical = (window["high"] + window["low"] + window["close"]) / 3.0
    expected = float((weights * typical).sum() / weights.sum())

    assert np.isclose(row["foreign_cost_mid_5d"], expected)
    assert row["foreign_buy_days_5d"] == int((weights > 0).sum())
    next_row = stock.iloc[signal_index + 1]
    assert np.isclose(row["entry_open_computed"], next_row["open"])


def test_cost_proxy_does_not_use_future_returns() -> None:
    events = prepare_phase5i_events(_events(), settings=Phase5ISettings())
    original = build_event_cost_features(
        events,
        history=_history(),
        settings=Phase5ISettings(),
    )
    changed = events.copy()
    changed["adjusted_return_20d"] = -9.0
    changed["adjusted_return_40d"] = 9.0
    replay = build_event_cost_features(
        changed,
        history=_history(),
        settings=Phase5ISettings(),
    )
    columns = [column for column in original.columns if "cost_" in column]

    pd.testing.assert_frame_equal(original[columns], replay[columns])


def test_quantile_analysis_orders_entry_deviation() -> None:
    rows: list[dict[str, object]] = []
    for index in range(100):
        deviation = index / 100.0
        rows.append(
            {
                "event_id": f"E{index}",
                "event_type": "NEW_CANDIDATE",
                "entry_rule": "top10_confirm1d",
                "stock_id": f"S{index:03d}",
                "stock_name": "測試",
                "signal_date": f"2024-01-{index % 20 + 1:02d}",
                "test_year": 2024,
                "return_rank_score_daily_percentile": 95.0,
                "signal_close": 100.0,
                "entry_open": 100.0,
                "return_5d_before_signal": 0.0,
                "return_20d_before_signal": 0.0,
                "range_position_20d": 0.5,
                "adjusted_return_20d": 0.10 - deviation * 0.20,
                "adjusted_return_40d": 0.10 - deviation * 0.15,
                "excess_adjusted_return_20d": 0.10 - deviation * 0.20,
                "excess_adjusted_return_40d": 0.10 - deviation * 0.15,
                "max_adjusted_return_20d": 0.15,
                "min_adjusted_return_20d": -deviation * 0.10,
                "max_adjusted_return_40d": 0.20,
                "min_adjusted_return_40d": -deviation * 0.12,
                "actor": "selected_total",
                "actor_label": "三法人合計",
                "window_days": 20,
                "proxy_name": "selected_total_20d",
                "estimated_cost_mid": 100.0,
                "estimated_cost_low": 99.0,
                "estimated_cost_high": 101.0,
                "positive_net_shares": 1000,
                "buy_days": 10,
                "entry_deviation": deviation,
                "signal_deviation": deviation,
                "cost_band_width": 0.02,
                "proxy_available": 1,
            }
        )
    quantiles = build_quantile_analysis(pd.DataFrame(rows))
    confirmation = quantiles[quantiles["period"] == "confirmation"]
    low = confirmation[confirmation["deviation_quintile"] == 1].iloc[0]
    high = confirmation[confirmation["deviation_quintile"] == 5].iloc[0]

    assert low["average_entry_deviation"] < high["average_entry_deviation"]
    assert low["average_excess_return_20d"] > high["average_excess_return_20d"]
    assert low["average_min_return_20d"] > high["average_min_return_20d"]


def _write_phase3_shard(shard_dir: Path) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "stock_id": "A",
                "signal_date": "2024-01-08",
                "entry_date_20d": "2024-01-09",
                "entry_open_20d": 101.0,
                "max_adjusted_return_20d": 0.12,
                "min_adjusted_return_20d": -0.08,
            }
        ]
    ).to_csv(shard_dir / "A.csv.gz", index=False, compression="gzip")


def test_phase3_outcomes_allow_missing_40d_cache(tmp_path: Path) -> None:
    shard_dir = tmp_path / "phase3_shards" / "signature"
    _write_phase3_shard(shard_dir)
    events = pd.DataFrame([{"stock_id": "A", "signal_date": "2024-01-08"}])

    result = load_phase3_event_outcomes(events, shard_dir=shard_dir)

    assert np.isclose(result.iloc[0]["max_adjusted_return_20d"], 0.12)
    assert pd.isna(result.iloc[0]["max_adjusted_return_40d"])
    assert pd.isna(result.iloc[0]["min_adjusted_return_40d"])


def test_phase3_outcomes_load_40d_extrema_from_phase4d_cache(tmp_path: Path) -> None:
    shard_dir = tmp_path / "phase3_shards" / "signature"
    _write_phase3_shard(shard_dir)
    cache_dir = tmp_path / "phase4d_cache" / "cache-signature"
    cache_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "stock_id": "A",
                "signal_date": "2024-01-08",
                "max_adjusted_return_40d": 0.25,
                "min_adjusted_return_40d": -0.15,
            }
        ]
    ).to_csv(cache_dir / "A_40d.csv.gz", index=False, compression="gzip")
    events = pd.DataFrame([{"stock_id": "A", "signal_date": "2024-01-08"}])

    result = load_phase3_event_outcomes(events, shard_dir=shard_dir)

    assert np.isclose(result.iloc[0]["max_adjusted_return_40d"], 0.25)
    assert np.isclose(result.iloc[0]["min_adjusted_return_40d"], -0.15)


def test_spearman_does_not_require_scipy(monkeypatch) -> None:
    import builtins

    original_import = builtins.__import__

    def reject_scipy(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            raise ModuleNotFoundError("scipy intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_scipy)

    result = _spearman(
        pd.Series([10.0, 20.0, 20.0, 40.0]),
        pd.Series([1.0, 2.0, 2.0, 4.0]),
    )

    assert np.isclose(result, 1.0)


def test_spearman_returns_nan_for_constant_input() -> None:
    result = _spearman(
        pd.Series([1.0, 1.0, 1.0]),
        pd.Series([1.0, 2.0, 3.0]),
    )

    assert pd.isna(result)
