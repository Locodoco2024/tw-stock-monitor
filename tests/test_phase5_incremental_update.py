from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from research.institutional_model.database import ResearchDatabase
from research.institutional_model.phase5_daily_reference import Phase5BResult
from research.institutional_model.phase5_incremental_update import (
    FLOW_DATASET,
    PRICE_DATASET,
    Phase5CSettings,
    resolve_target_market_date,
    run_phase5c_incremental_update,
)


TAIPEI = ZoneInfo("Asia/Taipei")


class FakeFinMindClient:
    def __init__(
        self,
        *,
        calendar_rows: list[dict[str, object]],
        price_rows: dict[str, list[dict[str, object]]],
        flow_rows: dict[str, list[dict[str, object]]],
    ) -> None:
        self.calendar_rows = calendar_rows
        self.price_rows = price_rows
        self.flow_rows = flow_rows
        self.request_count = 0
        self.calls: list[tuple[str, str | None, str | None, str | None]] = []

    def fetch(
        self,
        dataset: str,
        *,
        data_id: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, object]]:
        self.request_count += 1
        self.calls.append((dataset, data_id, start_date, end_date))
        if dataset == "TaiwanStockTradingDate":
            return self.calendar_rows
        if dataset == PRICE_DATASET:
            return self.price_rows.get(str(data_id), [])
        if dataset == FLOW_DATASET:
            return self.flow_rows.get(str(data_id), [])
        raise AssertionError(dataset)


def test_resolve_target_date_uses_safe_prior_day_without_cutoff(tmp_path: Path) -> None:
    database = ResearchDatabase(tmp_path / "research.sqlite")
    database.initialize()
    database.executemany(
        "INSERT INTO market_calendar(date) VALUES (?)",
        [("2026-07-27",), ("2026-07-28",)],
    )
    now = datetime(2026, 7, 28, 19, 0, tzinfo=TAIPEI)

    safe = resolve_target_market_date(
        database=database,
        now=now,
        as_of_date=None,
        explicit_target_date=None,
        publish_cutoff_time=None,
    )
    after_cutoff = resolve_target_market_date(
        database=database,
        now=now,
        as_of_date=None,
        explicit_target_date=None,
        publish_cutoff_time="18:00",
    )

    assert safe.target_date == "2026-07-27"
    assert safe.source == "latest_completed_market_date_before_as_of"
    assert after_cutoff.target_date == "2026-07-28"
    assert after_cutoff.source == "latest_market_date_at_or_before_as_of_after_cutoff"


def test_phase5c_incremental_update_builds_live_shards_without_touching_phase2_status(
    tmp_path: Path,
    monkeypatch,
) -> None:
    environment = _build_incremental_environment(tmp_path, target_rows_for_all=True)
    database = environment["database"]
    database.mark_download(
        dataset=PRICE_DATASET,
        stock_id="7001",
        requested_start="2015-01-01",
        requested_end="2026-07-23",
        status="complete",
        row_count=100,
    )
    selection_calls: list[dict[str, object]] = []

    def fake_phase5b(**kwargs):
        selection_calls.append(kwargs)
        return Phase5BResult(
            status="READY",
            signal_date="2026-07-28",
            selected_rows=2,
            output_paths=(),
        )

    monkeypatch.setattr(
        "research.institutional_model.phase5_incremental_update.run_phase5b_daily_reference",
        fake_phase5b,
    )
    result = run_phase5c_incremental_update(
        database=database,
        client=environment["client"],
        output_dir=environment["output"],
        live_shard_root=environment["live_root"],
        phase3_shard_root=tmp_path / "phase3_shards",
        model_root=tmp_path / "phase5_models",
        settings=Phase5CSettings(
            max_stocks_per_batch=0,
            minimum_source_stocks=2,
            minimum_source_coverage_ratio=0.5,
            continuous=True,
            recent_rows_per_stock=30,
            quota_wait_minutes=0,
        ),
        as_of_date="2026-07-28",
        target_date="2026-07-28",
        now=datetime(2026, 7, 28, 22, 0, tzinfo=TAIPEI),
    )

    assert result.status == "READY"
    assert result.completed_requests == 4
    assert result.expected_requests == 4
    assert result.source_coverage_status == "PASS"
    assert result.selection_status == "READY"
    assert len(selection_calls) == 1
    assert selection_calls[0]["live_shard_root"] == environment["live_root"]

    status = pd.read_csv(
        environment["output"] / "phase5c_incremental_update_status.csv",
        encoding="utf-8-sig",
        dtype=str,
    )
    assert set(status["status"]) == {"complete"}
    assert set(status["latest_after"]) == {"2026-07-28"}

    pointer = json.loads(
        (environment["live_root"] / "latest.json").read_text(encoding="utf-8")
    )
    live_dir = environment["live_root"] / pointer["shard_directory"]
    assert pointer["status"] == "PASS"
    assert len(list(live_dir.glob("*.csv.gz"))) == 2

    historical = database.query(
        "SELECT requested_start, requested_end FROM download_status WHERE dataset=? AND stock_id=?",
        (PRICE_DATASET, "7001"),
    )[0]
    assert historical["requested_start"] == "2015-01-01"
    assert historical["requested_end"] == "2026-07-23"

    calls = {
        (dataset, stock_id): start
        for dataset, stock_id, start, _ in environment["client"].calls
        if stock_id
    }
    assert calls[(PRICE_DATASET, "7001")] == "2026-07-28"
    assert calls[(FLOW_DATASET, "7001")] == "2026-07-28"
    assert (environment["output"] / "phase5c_daily_update_reports.zip").exists()


def test_phase5c_blocks_partial_provider_coverage(tmp_path: Path, monkeypatch) -> None:
    environment = _build_incremental_environment(tmp_path, target_rows_for_all=False)

    def unexpected_phase5b(**kwargs):  # pragma: no cover - should not run
        raise AssertionError(kwargs)

    monkeypatch.setattr(
        "research.institutional_model.phase5_incremental_update.run_phase5b_daily_reference",
        unexpected_phase5b,
    )
    result = run_phase5c_incremental_update(
        database=environment["database"],
        client=environment["client"],
        output_dir=environment["output"],
        live_shard_root=environment["live_root"],
        phase3_shard_root=tmp_path / "phase3_shards",
        model_root=tmp_path / "phase5_models",
        settings=Phase5CSettings(
            max_stocks_per_batch=0,
            minimum_source_stocks=2,
            minimum_source_coverage_ratio=0.8,
            continuous=True,
            recent_rows_per_stock=30,
            quota_wait_minutes=0,
        ),
        as_of_date="2026-07-28",
        target_date="2026-07-28",
        now=datetime(2026, 7, 28, 22, 0, tzinfo=TAIPEI),
    )

    assert result.status == "BLOCKED_SOURCE_COVERAGE"
    assert result.source_coverage_status == "INCOMPLETE_TARGET_COVERAGE"
    assert result.selection_status == "NOT_RUN"
    assert not (environment["live_root"] / "latest.json").exists()
    summary = _metric_map(environment["output"] / "phase5c_summary.csv")
    assert summary["target_common_stock_count"] == "1"
    assert summary["required_common_stock_count"] == "2"


def test_phase5c_retries_prior_partial_response_on_next_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    environment = _build_incremental_environment(tmp_path, target_rows_for_all=True)
    database = environment["database"]
    database.execute(
        "DELETE FROM stock_prices WHERE stock_id='7002' AND date='2026-07-28'"
    )
    database.execute(
        "DELETE FROM institutional_flows WHERE stock_id='7002' AND date='2026-07-28'"
    )
    database.execute(
        """
        INSERT INTO phase5c_update_status (
            target_date, stock_id, dataset, requested_start, requested_end,
            status, row_count, latest_before, latest_after, error
        ) VALUES ('2026-07-28', '7002', ?, '2026-07-28', '2026-07-28',
                  'complete', 1, '2026-07-27', '2026-07-27', '')
        """,
        (PRICE_DATASET,),
    )
    database.execute(
        """
        INSERT INTO phase5c_update_status (
            target_date, stock_id, dataset, requested_start, requested_end,
            status, row_count, latest_before, latest_after, error
        ) VALUES ('2026-07-28', '7002', ?, '2026-07-28', '2026-07-28',
                  'complete', 3, '2026-07-27', '2026-07-27', '')
        """,
        (FLOW_DATASET,),
    )

    monkeypatch.setattr(
        "research.institutional_model.phase5_incremental_update.run_phase5b_daily_reference",
        lambda **kwargs: Phase5BResult(
            status="READY", signal_date="2026-07-28", selected_rows=2, output_paths=()
        ),
    )
    result = run_phase5c_incremental_update(
        database=database,
        client=environment["client"],
        output_dir=environment["output"],
        live_shard_root=environment["live_root"],
        phase3_shard_root=tmp_path / "phase3_shards",
        model_root=tmp_path / "phase5_models",
        settings=Phase5CSettings(
            max_stocks_per_batch=0,
            minimum_source_stocks=2,
            minimum_source_coverage_ratio=0.8,
            continuous=True,
            recent_rows_per_stock=30,
            quota_wait_minutes=0,
        ),
        as_of_date="2026-07-28",
        target_date="2026-07-28",
        now=datetime(2026, 7, 28, 22, 0, tzinfo=TAIPEI),
    )

    assert result.status == "READY"
    retried = {
        (dataset, stock_id)
        for dataset, stock_id, _, _ in environment["client"].calls
        if stock_id == "7002"
    }
    assert retried == {(PRICE_DATASET, "7002"), (FLOW_DATASET, "7002")}


def _build_incremental_environment(
    tmp_path: Path,
    *,
    target_rows_for_all: bool,
) -> dict[str, object]:
    database = ResearchDatabase(tmp_path / "institutional.sqlite")
    database.initialize()
    output = tmp_path / "output"
    output.mkdir()
    live_root = tmp_path / "phase5c_live_shards"
    stock_ids = ["7001", "7002"]
    database.executemany(
        """
        INSERT INTO model_universe (
            stock_id, stock_name, market_type, listing_date, current_status,
            download_enabled, training_enabled, inclusion_status,
            inclusion_reason, source
        ) VALUES (?, ?, 'tpex', '2020-01-01', 'active', 1, 1, 'included', 'test', 'test')
        """,
        [(stock_id, f"測試{stock_id}") for stock_id in stock_ids],
    )

    start = datetime(2026, 6, 29)
    calendar_dates = [
        (start + timedelta(days=index)).date().isoformat()
        for index in range(30)
        if (start + timedelta(days=index)).weekday() < 5
    ]
    calendar_dates = [value for value in calendar_dates if value <= "2026-07-28"]
    database.executemany(
        "INSERT INTO market_calendar(date) VALUES (?)",
        [(value,) for value in calendar_dates],
    )
    prior_dates = [value for value in calendar_dates if value < "2026-07-28"]
    price_rows = []
    flow_rows = []
    for stock_position, stock_id in enumerate(stock_ids, start=1):
        for day_position, market_date in enumerate(prior_dates, start=1):
            price_rows.append(
                (
                    stock_id,
                    market_date,
                    1_000_000 + day_position,
                    30_000_000 + stock_position * 1_000_000,
                    10.0,
                    11.0,
                    9.0,
                    10.0,
                )
            )
            flow_rows.append(
                (
                    stock_id,
                    market_date,
                    stock_position * 100 + day_position,
                    stock_position * 20,
                    stock_position * 10,
                    stock_position * 130 + day_position,
                )
            )
    database.executemany(
        """
        INSERT INTO stock_prices (
            stock_id, date, trading_volume, trading_money, open, high, low, close
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        price_rows,
    )
    database.executemany(
        """
        INSERT INTO institutional_flows (
            stock_id, date, foreign_net, investment_trust_net,
            dealer_self_net, selected_total_net
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        flow_rows,
    )

    target_price_rows: dict[str, list[dict[str, object]]] = {}
    target_flow_rows: dict[str, list[dict[str, object]]] = {}
    for position, stock_id in enumerate(stock_ids, start=1):
        has_target = target_rows_for_all or position == 1
        target_price_rows[stock_id] = (
            [
                {
                    "stock_id": stock_id,
                    "date": "2026-07-28",
                    "Trading_Volume": 1_200_000,
                    "Trading_money": 40_000_000 + position * 1_000_000,
                    "open": 10.0,
                    "max": 11.0,
                    "min": 9.0,
                    "close": 10.5,
                    "spread": 0.5,
                    "Trading_turnover": 1000,
                }
            ]
            if has_target
            else []
        )
        target_flow_rows[stock_id] = (
            [
                {
                    "stock_id": stock_id,
                    "date": "2026-07-28",
                    "name": "Foreign_Investor",
                    "buy": 1000 + position,
                    "sell": 100,
                },
                {
                    "stock_id": stock_id,
                    "date": "2026-07-28",
                    "name": "Investment_Trust",
                    "buy": 300,
                    "sell": 100,
                },
                {
                    "stock_id": stock_id,
                    "date": "2026-07-28",
                    "name": "Dealer_self",
                    "buy": 200,
                    "sell": 100,
                },
            ]
            if has_target
            else []
        )
    client = FakeFinMindClient(
        calendar_rows=[{"date": value} for value in calendar_dates],
        price_rows=target_price_rows,
        flow_rows=target_flow_rows,
    )
    return {
        "database": database,
        "output": output,
        "live_root": live_root,
        "client": client,
    }


def _metric_map(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    return dict(zip(frame["metric"], frame["value"], strict=True))
