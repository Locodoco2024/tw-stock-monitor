from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from research.institutional_model.phase4_stability import CORE_FEATURE_COLUMNS
from research.institutional_model.phase5_final_model import (
    FinalRankModelBundle,
    Phase5DSettings,
    fit_all_data_preprocessor,
    fit_all_data_rank_model,
    load_final_rank_model,
    replay_lifecycle_engine,
    run_phase5d_final_model,
    save_final_rank_model,
    score_rank_cross_section,
)


def test_replay_engine_uses_track_only_and_lifecycle_milestones() -> None:
    dates = pd.bdate_range("2023-01-02", periods=70).strftime("%Y-%m-%d")
    rows: list[dict[str, object]] = []
    stock_count = 25
    for day_index, signal_date in enumerate(dates):
        for stock_index in range(stock_count):
            percentile = 100.0 * (stock_index + 1) / stock_count
            rows.append(
                {
                    "stock_id": f"S{stock_index:02d}",
                    "stock_name": f"股票{stock_index:02d}",
                    "signal_date": signal_date,
                    "test_year": 2023,
                    "adjusted_return_20d": (percentile - 50.0) / 1000.0,
                    "adjusted_return_40d": (percentile - 50.0) / 600.0,
                    "return_rank_score_daily_percentile": percentile,
                }
            )
    reports = replay_lifecycle_engine(
        pd.DataFrame(rows),
        settings=Phase5DSettings(minimum_daily_stocks=20),
    )

    notifications = reports["notifications"]
    assert notifications["trade_action"].eq("TRACK_ONLY").all()
    assert notifications.duplicated(
        ["event_id", "signal_date", "notification_type"]
    ).sum() == 0
    assert "NEW_CANDIDATE" in set(notifications["notification_type"])
    assert "LAYOUT_CONFIRMED_DIRECT" in set(notifications["notification_type"])
    assert "DAY20_EXTEND_STRONG" in set(notifications["notification_type"])
    assert "DAY20_EXTEND" in set(notifications["notification_type"])
    assert "DAY40_END" in set(notifications["notification_type"])
    assert not reports["performance"].empty


def test_final_rank_model_round_trip_and_group_scoring(tmp_path: Path) -> None:
    rng = np.random.default_rng(123)
    features = rng.normal(size=(500, len(CORE_FEATURE_COLUMNS)))
    targets = pd.Series(features[:, 0] - 0.5 * features[:, 1]).rank(pct=True).to_numpy()
    settings = Phase5DSettings(
        minimum_daily_stocks=20,
        quantile_sample_size=300,
        random_seed=123,
    )
    preprocessor = fit_all_data_preprocessor(features, settings=settings)
    model = fit_all_data_rank_model(
        features=features,
        targets=targets,
        preprocessor=preprocessor,
        l2_penalty=0.001,
    )
    bundle = FinalRankModelBundle(
        signature="test-signature",
        feature_columns=tuple(CORE_FEATURE_COLUMNS),
        preprocessor=preprocessor,
        model=model,
        training_rows=len(features),
        signal_dates=20,
        first_training_date="2020-01-01",
        last_training_date="2025-12-31",
        source_sha256="a" * 64,
        target_mean=float(targets.mean()),
        target_std=float(targets.std()),
    )
    manifest = tmp_path / "model.json"
    arrays = tmp_path / "model.npz"
    save_final_rank_model(bundle=bundle, manifest_path=manifest, arrays_path=arrays)
    loaded = load_final_rank_model(manifest_path=manifest, arrays_path=arrays)

    frame = pd.DataFrame(features[:30], columns=CORE_FEATURE_COLUMNS)
    frame["stock_id"] = [f"S{index:02d}" for index in range(30)]
    scored = score_rank_cross_section(frame, bundle=loaded)

    assert np.allclose(loaded.model.weights, bundle.model.weights)
    assert scored["return_rank_daily_percentile"].between(0, 100).all()
    assert scored["return_rank_daily_percentile"].nunique() == 30
    for column in (
        "foreign_contribution",
        "investment_trust_contribution",
        "dealer_self_contribution",
        "institutional_consensus_contribution",
    ):
        assert column in scored.columns


def test_phase5d_builds_final_model_and_replay_reports(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    signature = "d" * 64
    _write_metric_csv(output / "phase3_summary.csv", {"config_signature": signature})
    _write_metric_csv(
        output / "phase3b_summary.csv",
        {"status": "PASS", "ready_for_modeling": 1},
    )
    _write_metric_csv(
        output / "phase4e_summary.csv",
        {
            "phase4e_version": "phase4e-v1",
            "primary_horizon_days": 20,
            "research_rows": 1000,
            "ready_for_target_decision": 1,
        },
    )
    _write_csv(output / "phase4e_model_comparison.csv", [{"status": "complete"}])
    _write_csv(output / "phase4e_coefficient_stability.csv", [{"status": "complete"}])
    _write_metric_csv(
        output / "phase4f_summary.csv",
        {
            "phase4f_version": "phase4f-v1",
            "pipeline_status": "PASS",
            "score_column": "return_rank_score_daily_percentile",
            "source_rows": 1000,
            "ready_for_lifecycle_decision": 1,
        },
    )
    _write_csv(output / "phase4f_rule_candidates.csv", [{"decision_status": "review"}])

    stock_count = 30
    dates = pd.bdate_range("2023-01-02", periods=70).strftime("%Y-%m-%d")
    shard_dir = tmp_path / "phase3_shards" / signature[:16]
    shard_dir.mkdir(parents=True)
    oos_rows: list[dict[str, object]] = []
    for stock_index in range(stock_count):
        stock_id = f"S{stock_index:02d}"
        shard_rows: list[dict[str, object]] = []
        for day_index, signal_date in enumerate(dates):
            signal = -1.0 + 2.0 * stock_index / (stock_count - 1)
            adjusted_20d = 0.04 * signal + 0.002 * np.sin(day_index / 5.0)
            adjusted_40d = 0.07 * signal + 0.003 * np.sin(day_index / 7.0)
            percentile = 100.0 * (stock_index + 1) / stock_count
            row: dict[str, object] = {
                "stock_id": stock_id,
                "stock_name": f"股票{stock_index:02d}",
                "market_type": "tpex",
                "signal_date": signal_date,
                "feature_status": "ok",
                "liquidity_pass_20m": 1,
                "label_status_20d": "ok",
                "adjusted_return_20d": adjusted_20d,
            }
            for feature in CORE_FEATURE_COLUMNS:
                row[feature] = signal if feature == "foreign_flow_pct_1d" else 0.0
            shard_rows.append(row)
            oos_rows.append(
                {
                    "stock_id": stock_id,
                    "stock_name": f"股票{stock_index:02d}",
                    "signal_date": signal_date,
                    "test_year": 2023,
                    "adjusted_return_20d": adjusted_20d,
                    "adjusted_return_40d": adjusted_40d,
                    "return_rank_score_daily_percentile": percentile,
                }
            )
        _write_gzip_csv(shard_dir / f"{stock_id}.csv.gz", shard_rows)
    _write_gzip_csv(output / "phase4e_oos_scores.csv.gz", oos_rows)

    settings = Phase5DSettings(
        minimum_daily_stocks=20,
        quantile_sample_size=500,
        random_seed=123,
    )
    result = run_phase5d_final_model(
        output_dir=output,
        shard_root=tmp_path / "phase3_shards",
        model_root=tmp_path / "phase5_models",
        settings=settings,
    )

    assert result.status == "PASS"
    assert result.training_rows == stock_count * len(dates)
    assert result.replay_events > 0
    assert result.replay_notifications > result.replay_events
    assert (output / "phase5d_final_model_reports.zip").exists()
    assert (output / "phase5d_model_coefficients.csv").exists()
    notifications = pd.read_csv(
        output / "phase5d_notification_replay.csv", encoding="utf-8-sig"
    )
    assert notifications["trade_action"].eq("TRACK_ONLY").all()

    rerun = run_phase5d_final_model(
        output_dir=output,
        shard_root=tmp_path / "phase3_shards",
        model_root=tmp_path / "phase5_models",
        settings=settings,
    )
    assert rerun.training_rows == result.training_rows
    assert rerun.replay_events == result.replay_events


def _write_metric_csv(path: Path, values: dict[str, object]) -> None:
    _write_csv(path, [{"metric": key, "value": value} for key, value in values.items()])


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _write_gzip_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8-sig", newline="") as handle:
        pd.DataFrame(rows).to_csv(handle, index=False)
