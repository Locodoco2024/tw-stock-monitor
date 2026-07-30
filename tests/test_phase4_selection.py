from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from research.institutional_model.phase3_dataset import sha256_file
from research.institutional_model.phase4_selection import (
    Phase4CSettings,
    assign_same_day_ranks,
    daily_top_bottom_spreads,
    moving_block_bootstrap_mean_ci,
    run_phase4c_selection_validation,
)
from research.institutional_model.phase4_stability import CORE_FEATURE_COLUMNS


def test_same_day_ranking_resets_for_each_signal_date() -> None:
    frame = pd.DataFrame(
        {
            "signal_date": ["2023-01-02"] * 4 + ["2023-01-03"] * 4,
            "stock_id": ["1", "2", "3", "4"] * 2,
            "institutional_index_raw": [1.0, 4.0, 2.0, 3.0, 40.0, 10.0, 30.0, 20.0],
        }
    )
    ranked, dropped = assign_same_day_ranks(frame, minimum_daily_stocks=4)

    assert dropped == 0
    for _, day in ranked.groupby("signal_date"):
        assert sorted(day["daily_rank"].tolist()) == [1, 2, 3, 4]
        assert day["daily_percentile"].min() == 0.0
        assert day["daily_percentile"].max() == 100.0


def test_daily_top_bottom_spread_uses_same_day_groups() -> None:
    rows = []
    for date, shift in (("2023-01-02", 0.0), ("2023-01-03", 1.0)):
        for index in range(20):
            rows.append(
                {
                    "signal_date": date,
                    "stock_id": f"{index:02d}",
                    "institutional_index_raw": float(index),
                    "adjusted_return_10d": shift + index / 100.0,
                }
            )
    ranked, _ = assign_same_day_ranks(pd.DataFrame(rows), minimum_daily_stocks=20)
    spread = daily_top_bottom_spreads(ranked, horizon=10, band_percent=20)

    assert len(spread) == 2
    assert np.allclose(spread.to_numpy(), 0.16)


def test_moving_block_bootstrap_is_deterministic_and_positive() -> None:
    values = np.linspace(0.001, 0.010, 24)
    first = moving_block_bootstrap_mean_ci(
        values,
        iterations=500,
        block_length=3,
        random_seed=123,
    )
    second = moving_block_bootstrap_mean_ci(
        values,
        iterations=500,
        block_length=3,
        random_seed=123,
    )

    assert first == second
    assert first[0] > 0
    assert first[1] > first[0]


def test_phase4c_pipeline_builds_same_day_selection_reports(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    signature = "a" * 64
    source_sha = ""
    stock_ids = [f"{7000 + index}" for index in range(60)]
    all_rows = _synthetic_rows(stock_ids)
    training_rows = [row for row in all_rows if row["liquidity_pass_20m"] == 1]

    training_path = output / "phase3_training_tpex.csv.gz"
    _write_gzip_csv(training_path, training_rows)
    source_sha = sha256_file(training_path)
    _write_csv(
        output / "phase3_dataset_manifest.csv",
        [
            {
                "file_name": "phase3_training_tpex.csv.gz",
                "market_type": "tpex",
                "row_count": len(training_rows),
                "size_bytes": 1,
                "sha256": source_sha,
                "config_signature": signature,
                "label_rule_version": "rounded-return-10dp-v1",
            }
        ],
    )
    _write_metric_csv(
        output / "phase3_summary.csv",
        {"config_signature": signature},
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
    _write_metric_csv(
        output / "phase3b_summary.csv",
        {"status": "PASS", "ready_for_modeling": 1},
    )
    _write_metric_csv(
        output / "phase4b_summary.csv",
        {
            "pipeline_status": "PASS",
            "ready_for_phase4c": 1,
            "run_signature": "phase4b-test",
            "tpex_source_sha256": source_sha,
        },
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
        output / "phase4b_market_decisions.csv",
        [
            {
                "market_type": "tpex",
                "selected_candidate_id": "core22_l2_1e-3",
                "decision": "RANKING_ONLY",
            }
        ],
    )
    _write_feature_and_model_reports(output)

    shard_dir = tmp_path / "phase3_shards" / signature[:16]
    shard_dir.mkdir(parents=True)
    rows_by_stock: dict[str, list[dict[str, object]]] = {stock_id: [] for stock_id in stock_ids}
    for row in all_rows:
        rows_by_stock[str(row["stock_id"])].append(row)
    for stock_id, rows in rows_by_stock.items():
        _write_gzip_csv(shard_dir / f"{stock_id}.csv.gz", rows)

    result = run_phase4c_selection_validation(
        output_dir=output,
        shard_root=tmp_path / "phase3_shards",
        settings=Phase4CSettings(
            chunk_size=113,
            minimum_daily_stocks=20,
            bootstrap_iterations=200,
            bootstrap_block_months=2,
            random_seed=456,
        ),
    )

    assert result.status == "PASS"
    assert result.ready_for_selection_index
    assert result.scored_dates == 24
    assert (output / "phase4c_validation_reports.zip").exists()
    assert (output / "phase4c_oos_scores.csv.gz").exists()

    summary = pd.read_csv(output / "phase4c_summary.csv", encoding="utf-8-sig")
    summary_map = dict(zip(summary["metric"], summary["value"], strict=True))
    assert str(summary_map["ready_for_selection_index"]) in {"1", "1.0"}

    liquidity = pd.read_csv(
        output / "phase4c_liquidity_sensitivity.csv", encoding="utf-8-sig"
    )
    assert set(liquidity["threshold_ntd_million"]) == {10, 20, 50, 100}
    confirmation = liquidity[
        (liquidity["period"] == "confirmation")
        & (liquidity["horizon_days"] == 10)
        & (liquidity["band_percent"] == 20)
    ]
    assert (confirmation["average_daily_high_minus_low_return"] > 0).all()


def _synthetic_rows(stock_ids: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for year in range(2019, 2027):
        for month in (1, 6, 11):
            date = f"{year}-{month:02d}-15"
            for position, stock_id in enumerate(stock_ids):
                signal = -1.0 + 2.0 * position / (len(stock_ids) - 1)
                adjusted = 0.10 * signal + 0.002 * (month - 6)
                if adjusted >= 0.05:
                    label = "UP"
                elif adjusted <= -0.05:
                    label = "DOWN"
                else:
                    label = "FLAT"
                row: dict[str, object] = {
                    "stock_id": stock_id,
                    "stock_name": f"測試{stock_id}",
                    "market_type": "tpex",
                    "signal_date": date,
                    "signal_year": year,
                    "feature_status": "ok",
                    "label_10d": label,
                    "liquidity_pass_10m": 1,
                    "liquidity_pass_20m": int(position >= 10),
                    "liquidity_pass_50m": int(position >= 20),
                    "liquidity_pass_100m": int(position >= 30),
                }
                for horizon, multiplier in ((5, 0.6), (10, 1.0), (20, 1.4)):
                    value = adjusted * multiplier
                    row[f"label_status_{horizon}d"] = "ok"
                    row[f"adjusted_return_{horizon}d"] = value
                    row[f"max_adjusted_return_{horizon}d"] = value + 0.03
                    row[f"min_adjusted_return_{horizon}d"] = value - 0.03
                for feature in CORE_FEATURE_COLUMNS:
                    row[feature] = signal if feature == "foreign_flow_pct_1d" else 0.0
                rows.append(row)
    return rows


def _write_feature_and_model_reports(output: Path) -> None:
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
    coefficient_rows = []
    preprocessing_rows = []
    for year in range(2019, 2027):
        for label in ("DOWN", "FLAT", "UP"):
            coefficient_rows.append(
                {
                    "market_type": "tpex",
                    "test_year": year,
                    "candidate_id": "core22_l2_1e-3",
                    "class_label": label,
                    "feature": "(intercept)",
                    "standardized_coefficient": 0.0,
                }
            )
            for feature in CORE_FEATURE_COLUMNS:
                if feature == "foreign_flow_pct_1d":
                    coefficient = {"DOWN": -2.0, "FLAT": 0.0, "UP": 2.0}[label]
                else:
                    coefficient = 0.0
                coefficient_rows.append(
                    {
                        "market_type": "tpex",
                        "test_year": year,
                        "candidate_id": "core22_l2_1e-3",
                        "class_label": label,
                        "feature": feature,
                        "standardized_coefficient": coefficient,
                    }
                )
        for feature in CORE_FEATURE_COLUMNS:
            preprocessing_rows.append(
                {
                    "market_type": "tpex",
                    "test_year": year,
                    "candidate_id": "core22_l2_1e-3",
                    "feature": feature,
                    "clip_lower": -10.0,
                    "clip_upper": 10.0,
                    "training_mean_after_clip": 0.0,
                    "training_std_after_clip": 1.0,
                    "quantile_sample_rows": 100,
                    "training_rows": 1000,
                }
            )
    _write_csv(output / "phase4b_coefficients.csv", coefficient_rows)
    _write_csv(output / "phase4b_preprocessing.csv", preprocessing_rows)


def _write_metric_csv(path: Path, values: dict[str, object]) -> None:
    _write_csv(path, [{"metric": key, "value": value} for key, value in values.items()])


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_gzip_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
