from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

from research.institutional_model.database import ResearchDatabase
from research.institutional_model.phase3_dataset import FEATURE_COLUMNS, sha256_file
from research.institutional_model.phase4_stability import CORE_FEATURE_COLUMNS
from research.institutional_model.phase5_daily_reference import (
    Phase5BSettings,
    run_phase5b_daily_reference,
)
from research.institutional_model.phase5_selection_index import (
    Phase5ASettings,
    run_phase5a_selection_index,
)


def test_phase5b_exports_ready_daily_reference(tmp_path: Path) -> None:
    environment = _build_environment(tmp_path, incomplete_newer_date=False)
    database = environment["database"]
    _insert_calendar_and_sources(
        database,
        environment["stock_ids"],
        calendar_dates=["2026-07-07", "2026-07-08", "2026-07-09"],
        source_date="2026-07-08",
    )

    result = _run_phase5b(environment, as_of_date="2026-07-09")

    assert result.status == "READY"
    assert result.signal_date == "2026-07-08"
    assert result.selected_rows == 30

    selection = pd.read_csv(
        environment["output"] / "phase5b_selection_index.csv",
        encoding="utf-8-sig",
    )
    diagnostics = pd.read_csv(
        environment["output"] / "phase5b_selection_index_diagnostic.csv",
        encoding="utf-8-sig",
    )
    top20_stocks = pd.read_csv(
        environment["output"] / "phase5b_selection_index_top20_stocks.csv",
        encoding="utf-8-sig",
    )
    top20_percent = pd.read_csv(
        environment["output"] / "phase5b_selection_index_top20_percent.csv",
        encoding="utf-8-sig",
    )
    assert len(selection) == 30
    assert len(diagnostics) == 30
    assert len(top20_stocks) == 20
    assert len(top20_percent) == 6
    assert set(selection["selection_band"]) == {
        "前10%",
        "前10%～20%",
        "中間60%",
        "後10%～20%",
        "後10%",
    }
    assert (selection["selection_readiness_status"] == "READY").all()
    assert (selection["data_freshness_status"] == "FRESH").all()

    summary = _metric_map(environment["output"] / "phase5b_summary.csv")
    assert summary["pipeline_status"] == "PASS"
    assert summary["selection_readiness_status"] == "READY"
    assert summary["ready_for_local_reference"] == "1"
    assert summary["trading_day_lag"] == "1"
    assert (environment["output"] / "phase5b_selection_index_reports.zip").exists()


def test_phase5b_blocks_stale_selection_but_keeps_diagnostics(tmp_path: Path) -> None:
    environment = _build_environment(tmp_path, incomplete_newer_date=False)
    database = environment["database"]
    _insert_calendar_and_sources(
        database,
        environment["stock_ids"],
        calendar_dates=["2026-07-07", "2026-07-08"],
        source_date="2026-07-08",
    )

    result = _run_phase5b(environment, as_of_date="2026-07-15")

    assert result.status == "BLOCKED_STALE_DATA"
    assert result.selected_rows == 0
    usable = pd.read_csv(
        environment["output"] / "phase5b_selection_index.csv",
        encoding="utf-8-sig",
    )
    diagnostic = pd.read_csv(
        environment["output"] / "phase5b_selection_index_diagnostic.csv",
        encoding="utf-8-sig",
    )
    top20 = pd.read_csv(
        environment["output"] / "phase5b_selection_index_top20_stocks.csv",
        encoding="utf-8-sig",
    )
    assert usable.empty
    assert top20.empty
    assert len(diagnostic) == 30
    assert set(diagnostic["data_status"]) == {"STALE_OR_INCOMPLETE"}
    summary = _metric_map(environment["output"] / "phase5b_summary.csv")
    assert summary["data_freshness_status"] == "STALE_MARKET_CALENDAR"
    assert summary["ready_for_local_reference"] == "0"
    markdown = (
        environment["output"] / "phase5b_daily_selection_summary.md"
    ).read_text(encoding="utf-8")
    assert "沒有產生可使用的選股名單" in markdown


def test_phase5b_blocks_incomplete_newer_cross_section(tmp_path: Path) -> None:
    environment = _build_environment(tmp_path, incomplete_newer_date=True)
    database = environment["database"]
    _insert_calendar_and_sources(
        database,
        environment["stock_ids"],
        calendar_dates=["2026-07-07", "2026-07-08", "2026-07-09"],
        source_date="2026-07-09",
    )

    result = _run_phase5b(environment, as_of_date="2026-07-09")

    assert result.status == "BLOCKED_INCOMPLETE_DATA"
    assert result.signal_date == "2026-07-08"
    recent = pd.read_csv(
        environment["output"] / "phase5b_recent_date_diagnostics.csv",
        encoding="utf-8-sig",
        dtype={"signal_date": str},
    )
    latest = recent.iloc[0]
    assert latest["signal_date"] == "2026-07-09"
    assert int(latest["raw_stock_count"]) == 10
    assert int(latest["eligible_20m"]) == 0
    summary = _metric_map(environment["output"] / "phase5b_summary.csv")
    assert summary["latest_raw_signal_date"] == "2026-07-09"
    assert summary["latest_eligible_signal_date"] == "2026-07-08"
    assert summary["data_freshness_status"] == "INCOMPLETE_LATEST_CROSS_SECTION"


def test_phase5b_blocks_when_sqlite_sources_are_newer_than_phase3(tmp_path: Path) -> None:
    environment = _build_environment(tmp_path, incomplete_newer_date=False)
    database = environment["database"]
    _insert_calendar_and_sources(
        database,
        environment["stock_ids"],
        calendar_dates=["2026-07-07", "2026-07-08", "2026-07-09"],
        source_date="2026-07-09",
    )

    result = _run_phase5b(environment, as_of_date="2026-07-09")

    assert result.status == "BLOCKED_INCOMPLETE_DATA"
    summary = _metric_map(environment["output"] / "phase5b_summary.csv")
    assert summary["data_freshness_status"] == "PHASE3_NOT_UPDATED_TO_LATEST_SOURCE"
    source = _metric_map(
        environment["output"] / "phase5b_data_source_freshness.csv"
    )
    assert source["latest_common_tpex_source_date"] == "2026-07-09"
    assert source["phase3_latest_raw_signal_date"] == "2026-07-08"



def test_phase5b_prefers_phase5c_live_shards(tmp_path: Path) -> None:
    environment = _build_environment(tmp_path, incomplete_newer_date=False)
    database = environment["database"]
    _insert_calendar_and_sources(
        database,
        environment["stock_ids"],
        calendar_dates=["2026-07-07", "2026-07-08", "2026-07-09"],
        source_date="2026-07-09",
    )

    live_root = tmp_path / "phase5c_live_shards"
    live_dir = live_root / "20260709"
    live_dir.mkdir(parents=True)
    for position, stock_id in enumerate(environment["stock_ids"]):
        _write_gzip_csv(
            live_dir / f"{stock_id}.csv.gz",
            [
                _latest_row(
                    stock_id,
                    position,
                    count=len(environment["stock_ids"]),
                    signal_date="2026-07-09",
                )
            ],
        )
    (live_root / "latest.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "target_date": "2026-07-09",
                "shard_directory": live_dir.name,
            }
        ),
        encoding="utf-8",
    )

    result = run_phase5b_daily_reference(
        database=database,
        output_dir=environment["output"],
        shard_root=environment["shard_root"],
        live_shard_root=live_root,
        model_root=environment["model_root"],
        model_settings=environment["model_settings"],
        settings=Phase5BSettings(
            chunk_size=97,
            minimum_daily_stocks=20,
            recent_rows_per_stock=30,
            recent_date_count=10,
            maximum_trading_day_lag=1,
            maximum_calendar_age_days=4,
        ),
        as_of_date="2026-07-09",
    )

    assert result.status == "READY"
    assert result.signal_date == "2026-07-09"
    summary = _metric_map(environment["output"] / "phase5b_summary.csv")
    assert summary["cross_section_source"] == "phase5c_live_shards"
    assert summary["cross_section_target_date"] == "2026-07-09"


def _run_phase5b(environment: dict[str, object], *, as_of_date: str):
    model_settings = environment["model_settings"]
    return run_phase5b_daily_reference(
        database=environment["database"],
        output_dir=environment["output"],
        shard_root=environment["shard_root"],
        model_root=environment["model_root"],
        model_settings=model_settings,
        settings=Phase5BSettings(
            chunk_size=97,
            minimum_daily_stocks=20,
            recent_rows_per_stock=30,
            recent_date_count=10,
            maximum_trading_day_lag=1,
            maximum_calendar_age_days=4,
        ),
        as_of_date=as_of_date,
    )


def _build_environment(
    tmp_path: Path,
    *,
    incomplete_newer_date: bool,
) -> dict[str, object]:
    output = tmp_path / "output"
    output.mkdir()
    database = ResearchDatabase(tmp_path / "institutional.sqlite3")
    database.initialize()
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
    _write_metric_csv(
        output / "phase3_summary.csv",
        {"config_signature": signature},
    )
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
        [{"market_type": "tpex", "selected_candidate_id": "core22_l2_1e-3"}],
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
            "last_signal_date": "2025-10-15",
            "confirmation_10d_top20_minus_bottom20": 0.007,
            "confirmation_ex_latest_10d_top20_minus_bottom20": 0.005,
        },
    )
    _write_csv(output / "phase4c_horizon_behavior.csv", _behavior_lookup())

    shard_root = tmp_path / "phase3_shards"
    shard_dir = shard_root / signature[:16]
    shard_dir.mkdir(parents=True)
    for position, stock_id in enumerate(stock_ids):
        rows = [_latest_row(stock_id, position, count=len(stock_ids))]
        if incomplete_newer_date and position < 10:
            rows.append(
                _latest_row(
                    stock_id,
                    position,
                    count=len(stock_ids),
                    signal_date="2026-07-09",
                )
            )
        _write_gzip_csv(shard_dir / f"{stock_id}.csv.gz", rows)

    model_root = tmp_path / "phase5_models"
    model_settings = Phase5ASettings(
        chunk_size=97,
        quantile_sample_size=300,
        batch_size=64,
        training_epochs=3,
        minimum_daily_stocks=20,
        random_seed=123,
    )
    run_phase5a_selection_index(
        output_dir=output,
        cache_root=tmp_path / "phase4_cache",
        shard_root=shard_root,
        model_root=model_root,
        settings=model_settings,
    )
    return {
        "output": output,
        "database": database,
        "stock_ids": stock_ids,
        "shard_root": shard_root,
        "model_root": model_root,
        "model_settings": model_settings,
    }


def _insert_calendar_and_sources(
    database: ResearchDatabase,
    stock_ids: list[str],
    *,
    calendar_dates: list[str],
    source_date: str,
) -> None:
    database.executemany(
        "INSERT OR IGNORE INTO market_calendar(date) VALUES (?)",
        [(value,) for value in calendar_dates],
    )
    database.executemany(
        """
        INSERT OR REPLACE INTO model_universe (
            stock_id, stock_name, market_type, current_status,
            download_enabled, training_enabled, inclusion_status,
            inclusion_reason, source
        ) VALUES (?, ?, 'tpex', 'listed', 1, 1, 'included', 'test', 'test')
        """,
        [(stock_id, f"測試{stock_id}") for stock_id in stock_ids],
    )
    database.executemany(
        """
        INSERT OR REPLACE INTO stock_prices (
            stock_id, date, trading_volume, trading_money, open, high, low, close
        ) VALUES (?, ?, 1000, 30000000, 10, 11, 9, 10)
        """,
        [(stock_id, source_date) for stock_id in stock_ids],
    )
    database.executemany(
        """
        INSERT OR REPLACE INTO institutional_flows (
            stock_id, date, foreign_net, investment_trust_net, dealer_self_net,
            dealer_hedging_net, foreign_dealer_self_net
        ) VALUES (?, ?, 1, 1, 1, 0, 0)
        """,
        [(stock_id, source_date) for stock_id in stock_ids],
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


def _metric_map(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    return dict(zip(frame["metric"], frame["value"], strict=True))


def _write_metric_csv(path: Path, values: dict[str, object]) -> None:
    _write_csv(path, [{"metric": key, "value": value} for key, value in values.items()])


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _write_gzip_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8-sig", newline="") as handle:
        pd.DataFrame(rows).to_csv(handle, index=False)
