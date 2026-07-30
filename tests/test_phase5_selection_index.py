from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
import pytest

from research.institutional_model.phase3_dataset import FEATURE_COLUMNS, sha256_file
from research.institutional_model.phase4_stability import CORE_FEATURE_COLUMNS
from research.institutional_model.phase5_selection_index import (
    Phase5ASettings,
    run_phase5a_selection_index,
)


def test_phase5a_builds_latest_same_day_selection_index(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    signature = "b" * 64
    stock_ids = [str(7000 + index) for index in range(30)]
    training_rows = _training_rows(stock_ids)
    training_path = output / "phase3_training_tpex.csv.gz"
    _write_gzip_csv(training_path, training_rows)
    source_sha = sha256_file(training_path)

    _write_csv(
        output / "phase3_dataset_manifest.csv",
        [
            {
                "file_name": training_path.name,
                "market_type": "tpex",
                "row_count": len(training_rows),
                "size_bytes": training_path.stat().st_size,
                "sha256": source_sha,
                "config_signature": signature,
                "label_rule_version": "rounded-return-10dp-v1",
            }
        ],
    )
    _write_metric_csv(output / "phase3_summary.csv", {"config_signature": signature})
    _write_metric_csv(
        output / "phase3b_summary.csv",
        {"status": "PASS", "ready_for_modeling": 1},
    )
    _write_csv(
        output / "phase3_stock_summary.csv",
        [
            {
                "stock_id": stock_id,
                "stock_name": f"測試{stock_id}",
                "market_type": "tpex",
                "phase3_status": "complete",
            }
            for stock_id in stock_ids
        ],
    )
    _write_csv(
        output / "phase4b_selected_candidates.csv",
        [
            {
                "market_type": "tpex",
                "selected_candidate_id": "core22_l2_1e-3",
            }
        ],
    )
    _write_csv(
        output / "phase4b_feature_sets.csv",
        [
            {
                "candidate_id": "core22_l2_1e-3",
                "feature_set": "core22",
                "l2_penalty": 0.001,
                "feature_order": index,
                "feature": feature,
            }
            for index, feature in enumerate(CORE_FEATURE_COLUMNS, start=1)
        ],
    )
    _write_csv(output / "phase4b_training_history.csv", _training_history())
    _write_metric_csv(
        output / "phase4c_summary.csv",
        {
            "pipeline_status": "PASS",
            "ready_for_selection_index": 1,
            "decision": "PROCEED_TO_SELECTION_INDEX_BUILD",
            "last_signal_date": "2026-07-08",
            "confirmation_10d_top20_minus_bottom20": 0.007,
            "confirmation_ex_latest_10d_top20_minus_bottom20": 0.005,
        },
    )
    _write_csv(output / "phase4c_horizon_behavior.csv", _behavior_lookup())

    shard_dir = tmp_path / "phase3_shards" / signature[:16]
    shard_dir.mkdir(parents=True)
    for position, stock_id in enumerate(stock_ids):
        _write_gzip_csv(
            shard_dir / f"{stock_id}.csv.gz",
            [
                _latest_row(
                    stock_id,
                    position,
                    count=len(stock_ids),
                    signal_date="2025-10-15",
                ),
                _latest_row(stock_id, position, count=len(stock_ids)),
            ],
        )

    result = run_phase5a_selection_index(
        output_dir=output,
        cache_root=tmp_path / "phase4_cache",
        shard_root=tmp_path / "phase3_shards",
        model_root=tmp_path / "phase5_models",
        settings=Phase5ASettings(
            chunk_size=97,
            quantile_sample_size=300,
            batch_size=64,
            training_epochs=3,
            minimum_daily_stocks=20,
            random_seed=123,
        ),
    )

    assert result.status == "PASS"
    assert result.signal_date == "2026-07-08"
    assert result.selected_rows == 30
    assert (output / "phase5a_selection_index_reports.zip").exists()

    selection = pd.read_csv(output / "phase5a_selection_index.csv", encoding="utf-8-sig")
    assert selection["rank_20m"].tolist() == list(range(1, 31))
    assert selection["percentile_20m"].max() == 100.0
    assert selection["percentile_20m"].min() == 0.0
    assert selection["institutional_selection_index"].equals(
        selection["percentile_20m"]
    )
    assert selection["stock_id"].nunique() == 30
    assert selection["history_10d_sample_count"].notna().all()
    assert (selection["entry_price_available"] == 0).all()
    assert (selection["entry_feasibility_status"] == "待T+1確認或無有效開盤價").all()
    assert set(selection["liquidity_tier"]) == {
        "2,000萬元以上",
        "5,000萬元以上",
        "1億元以上",
    }

    summary = pd.read_csv(output / "phase5a_summary.csv", encoding="utf-8-sig", dtype=str)
    summary_map = dict(zip(summary["metric"], summary["value"], strict=True))
    assert summary_map["pipeline_status"] == "PASS"
    assert summary_map["ready_for_local_reference"] == "1"

    rerun = run_phase5a_selection_index(
        output_dir=output,
        cache_root=tmp_path / "phase4_cache",
        shard_root=tmp_path / "phase3_shards",
        model_root=tmp_path / "phase5_models",
        settings=Phase5ASettings(
            chunk_size=97,
            quantile_sample_size=300,
            batch_size=64,
            training_epochs=3,
            minimum_daily_stocks=20,
            random_seed=123,
        ),
    )
    assert rerun.signal_date == result.signal_date
    assert rerun.selected_rows == result.selected_rows
    assert len(list((tmp_path / "phase5_models").glob("*/phase5a_model_arrays.npz"))) == 1

    with pytest.raises(RuntimeError, match="不可回看訓練截止日以前"):
        run_phase5a_selection_index(
            output_dir=output,
            cache_root=tmp_path / "phase4_cache",
            shard_root=tmp_path / "phase3_shards",
            model_root=tmp_path / "phase5_models",
            settings=Phase5ASettings(
                chunk_size=97,
                quantile_sample_size=300,
                batch_size=64,
                training_epochs=3,
                minimum_daily_stocks=20,
                random_seed=123,
            ),
            signal_date="2025-10-15",
        )


def _training_rows(stock_ids: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for year in range(2015, 2026):
        for month in (2, 6, 10):
            for position, stock_id in enumerate(stock_ids):
                signal = -1.0 + 2.0 * position / (len(stock_ids) - 1)
                adjusted = 0.08 * signal
                if adjusted >= 0.05:
                    label = "UP"
                elif adjusted <= -0.05:
                    label = "DOWN"
                else:
                    label = "FLAT"
                row: dict[str, object] = {
                    "signal_date": f"{year}-{month:02d}-15",
                    "signal_year": year,
                    "label_10d": label,
                    "adjusted_return_10d": adjusted,
                }
                for feature in FEATURE_COLUMNS:
                    row[feature] = signal if feature == "foreign_flow_pct_1d" else 0.0
                rows.append(row)
    return rows


def _latest_row(
    stock_id: str,
    position: int,
    *,
    count: int,
    signal_date: str = "2026-07-08",
) -> dict[str, object]:
    signal = -1.0 + 2.0 * position / (count - 1)
    row: dict[str, object] = {
        "stock_id": stock_id,
        "stock_name": f"測試{stock_id}",
        "market_type": "tpex",
        "signal_date": signal_date,
        "signal_year": 2026,
        "feature_status": "ok",
        "history_market_days_20": 20,
        "history_valid_price_days_20": 20,
        "history_flow_rows_20": 20,
        "median_trading_money_20d": 25_000_000 + position * 5_000_000,
        "max_zero_volume_streak_20d": 0,
        "normal_trading_days_20d": 20,
        "entry_price_available": 0,
        "liquidity_pass_20m": 0,
        "liquidity_pass_50m": 0,
        "liquidity_pass_100m": 0,
    }
    for feature in CORE_FEATURE_COLUMNS:
        row[feature] = signal if feature == "foreign_flow_pct_1d" else 0.0
    return row


def _training_history() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for year in range(2019, 2027):
        for epoch, loss in ((1, 1.1), (2, 1.05), (3, 1.0)):
            rows.append(
                {
                    "epoch": epoch,
                    "training_log_loss": loss,
                    "calibration_log_loss": loss,
                    "batches": 1,
                    "examples": 100,
                    "market_type": "tpex",
                    "test_year": year,
                    "candidate_id": "core22_l2_1e-3",
                }
            )
    return rows


def _behavior_lookup() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for horizon in (5, 10, 20):
        for decile in range(1, 11):
            value = (decile - 5.5) * horizon / 10000
            rows.append(
                {
                    "period": "confirmation_ex_latest_year",
                    "horizon_days": horizon,
                    "group_scheme": "decile",
                    "group_value": decile,
                    "equal_day_average_return": value,
                    "equal_day_median_return": value,
                    "sample_count": 1000,
                    "signal_dates": 500,
                    "unique_stocks": 300,
                    "average_return": value,
                    "median_return": value,
                    "positive_return_rate": 0.4 + decile / 100,
                    "up_5pct_rate": 0.2 + decile / 1000,
                    "down_5pct_rate": 0.3 - decile / 1000,
                    "average_max_adjusted_return": value + 0.05,
                    "average_min_adjusted_return": value - 0.05,
                    "return_p10": value - 0.08,
                    "return_p25": value - 0.04,
                    "return_p75": value + 0.04,
                    "return_p90": value + 0.08,
                    "label_up_rate": 0.2 + decile / 100 if horizon == 10 else None,
                    "label_flat_rate": 0.5 if horizon == 10 else None,
                    "label_down_rate": 0.3 - decile / 100 if horizon == 10 else None,
                }
            )
    return rows


def _write_metric_csv(path: Path, values: dict[str, object]) -> None:
    _write_csv(path, [{"metric": key, "value": value} for key, value in values.items()])


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _write_gzip_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8-sig", newline="") as handle:
        pd.DataFrame(rows).to_csv(handle, index=False)
