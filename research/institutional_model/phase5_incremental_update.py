from __future__ import annotations

import csv
import gzip
import math
import os
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import date, datetime, time as clock_time, timedelta
from pathlib import Path
from statistics import median
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd

from research.institutional_model.database import ResearchDatabase
from research.institutional_model.downloader import (
    upsert_institutional,
    upsert_market_calendar,
    upsert_prices,
)
from research.institutional_model.finmind_client import (
    FinMindQuotaExceeded,
    FinMindResearchClient,
)
from research.institutional_model.phase3_dataset import (
    ALL_COLUMNS,
    build_recent_stock_feature_rows,
)
from research.institutional_model.phase5_daily_reference import (
    Phase5BResult,
    Phase5BSettings,
    run_phase5b_daily_reference,
)
from research.institutional_model.phase5_selection_index import (
    Phase5ASettings,
    write_csv,
    write_json_atomic,
)


PHASE5C_VERSION = "phase5c-v1"
TARGET_MARKET = "tpex"
TAIPEI_TIMEZONE = ZoneInfo("Asia/Taipei")
PRICE_DATASET = "TaiwanStockPrice"
FLOW_DATASET = "TaiwanStockInstitutionalInvestorsBuySell"
CALENDAR_DATASET = "TaiwanStockTradingDate"
DATASET_TABLE = {
    PRICE_DATASET: "stock_prices",
    FLOW_DATASET: "institutional_flows",
}


@dataclass(frozen=True)
class Phase5CSettings:
    max_stocks_per_batch: int = 100
    recent_rows_per_stock: int = 90
    recent_coverage_dates: int = 5
    minimum_source_stocks: int = 50
    minimum_source_coverage_ratio: float = 0.80
    warmup_calendar_days: int = 60
    maximum_batches: int = 0
    continuous: bool = True
    quota_wait_minutes: int = 65
    stop_on_error: bool = False

    def validate(self) -> None:
        if self.max_stocks_per_batch < 0:
            raise ValueError("Phase 5C 每批股票數不可小於 0")
        if self.recent_rows_per_stock < 20:
            raise ValueError("Phase 5C 即時分片至少保留 20 列")
        if self.recent_coverage_dates <= 0:
            raise ValueError("Phase 5C 覆蓋率比較日期數必須大於 0")
        if self.minimum_source_stocks < 1:
            raise ValueError("Phase 5C 最低來源股票數必須大於 0")
        if not 0 < self.minimum_source_coverage_ratio <= 1:
            raise ValueError("Phase 5C 來源覆蓋率門檻必須介於 0 與 1")
        if self.warmup_calendar_days < 30:
            raise ValueError("Phase 5C 新股票暖機區間不可少於 30 日")
        if self.maximum_batches < 0:
            raise ValueError("Phase 5C 最大批次數不可小於 0")
        if self.quota_wait_minutes < 0:
            raise ValueError("Phase 5C 額度等待分鐘不可小於 0")


@dataclass(frozen=True)
class TargetDateDecision:
    as_of_date: str
    target_date: str
    source: str
    publish_cutoff_time: str


@dataclass(frozen=True)
class LiveShardBuildResult:
    status: str
    shard_directory: str
    stock_files: int
    total_rows: int
    latest_signal_date: str
    failed_stocks: int
    created_at: str


@dataclass(frozen=True)
class Phase5CResult:
    status: str
    target_date: str
    completed_requests: int
    expected_requests: int
    source_coverage_status: str
    selection_status: str
    output_paths: tuple[Path, ...]


def run_phase5c_incremental_update(
    *,
    database: ResearchDatabase,
    client: FinMindResearchClient,
    output_dir: Path | str,
    live_shard_root: Path | str,
    phase3_shard_root: Path | str,
    model_root: Path | str,
    model_settings: Phase5ASettings | None = None,
    phase5b_settings: Phase5BSettings | None = None,
    settings: Phase5CSettings | None = None,
    as_of_date: str | None = None,
    target_date: str | None = None,
    publish_cutoff_time: str | None = None,
    force: bool = False,
    now: datetime | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Phase5CResult:
    config = settings or Phase5CSettings()
    config.validate()
    model_config = model_settings or Phase5ASettings()
    model_config.validate()
    daily_config = phase5b_settings or Phase5BSettings()
    daily_config.validate()

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    live_root = Path(live_shard_root)
    live_root.mkdir(parents=True, exist_ok=True)
    effective_now = now or datetime.now(TAIPEI_TIMEZONE)
    if effective_now.tzinfo is None:
        effective_now = effective_now.replace(tzinfo=TAIPEI_TIMEZONE)
    else:
        effective_now = effective_now.astimezone(TAIPEI_TIMEZONE)

    calendar_status = "PASS"
    calendar_error = ""
    try:
        calendar_rows = client.fetch(CALENDAR_DATASET)
        upsert_market_calendar(database, calendar_rows)
    except FinMindQuotaExceeded as exc:
        calendar_status = "QUOTA_EXCEEDED"
        calendar_error = str(exc)
    except Exception as exc:  # pragma: no cover - network branch
        calendar_status = "FAILED"
        calendar_error = str(exc)

    if calendar_status != "PASS":
        decision = TargetDateDecision(
            as_of_date=as_of_date or effective_now.date().isoformat(),
            target_date=target_date or "",
            source="calendar_refresh_failed",
            publish_cutoff_time=publish_cutoff_time or "",
        )
        return _export_incomplete_result(
            output_dir=output,
            decision=decision,
            calendar_status=calendar_status,
            calendar_error=calendar_error,
            client=client,
            settings=config,
        )

    decision = resolve_target_market_date(
        database=database,
        now=effective_now,
        as_of_date=as_of_date,
        explicit_target_date=target_date,
        publish_cutoff_time=publish_cutoff_time,
    )
    candidates = load_active_tpex_candidates(database, decision.target_date)
    if not candidates:
        raise RuntimeError("Phase 5C 找不到可更新的現行 TPEx 普通股")

    if force:
        database.execute(
            "DELETE FROM phase5c_update_status WHERE target_date=?",
            (decision.target_date,),
        )

    attempted: set[tuple[str, str]] = set()
    batch_number = 0
    quota_exhausted = False
    while True:
        pending = select_pending_requests(
            database=database,
            candidates=candidates,
            target_date=decision.target_date,
            warmup_calendar_days=config.warmup_calendar_days,
            attempted=attempted,
        )
        if not pending:
            break
        if config.maximum_batches and batch_number >= config.maximum_batches:
            break

        batch_number += 1
        selected_stock_ids: list[str] = []
        for item in pending:
            if item["stock_id"] not in selected_stock_ids:
                selected_stock_ids.append(item["stock_id"])
            if (
                config.max_stocks_per_batch > 0
                and len(selected_stock_ids) >= config.max_stocks_per_batch
            ):
                break
        selected_set = set(selected_stock_ids)
        batch = [item for item in pending if item["stock_id"] in selected_set]
        print(
            f"Phase 5C 批次 {batch_number}：股票 {len(selected_set)}、"
            f"資料請求 {len(batch)}、目標日 {decision.target_date}"
        )

        for item in batch:
            key = (str(item["stock_id"]), str(item["dataset"]))
            attempted.add(key)
            try:
                process_incremental_request(
                    database=database,
                    client=client,
                    request=item,
                    target_date=decision.target_date,
                )
            except FinMindQuotaExceeded as exc:
                quota_exhausted = True
                record_update_status(
                    database=database,
                    target_date=decision.target_date,
                    stock_id=str(item["stock_id"]),
                    dataset=str(item["dataset"]),
                    requested_start=str(item["requested_start"]),
                    requested_end=decision.target_date,
                    status="quota_exceeded",
                    row_count=0,
                    latest_before=str(item["latest_before"]),
                    latest_after=str(item["latest_before"]),
                    error=str(exc),
                )
                break
            except Exception as exc:
                record_update_status(
                    database=database,
                    target_date=decision.target_date,
                    stock_id=str(item["stock_id"]),
                    dataset=str(item["dataset"]),
                    requested_start=str(item["requested_start"]),
                    requested_end=decision.target_date,
                    status="failed",
                    row_count=0,
                    latest_before=str(item["latest_before"]),
                    latest_after=latest_stock_date(
                        database,
                        table=str(item["table"]),
                        stock_id=str(item["stock_id"]),
                        target_date=decision.target_date,
                    ),
                    error=str(exc),
                )
                if config.stop_on_error:
                    raise

        if quota_exhausted:
            if not config.continuous or config.quota_wait_minutes <= 0:
                break
            print(
                f"FinMind 額度已滿，等待 {config.quota_wait_minutes} 分鐘後續傳。"
            )
            sleep_fn(config.quota_wait_minutes * 60)
            attempted.clear()
            quota_exhausted = False
            continue
        if not config.continuous:
            break

    request_report = build_update_status_report(
        database=database,
        candidates=candidates,
        target_date=decision.target_date,
    )
    coverage = build_source_coverage(
        database=database,
        target_date=decision.target_date,
        recent_dates=config.recent_coverage_dates,
        minimum_source_stocks=config.minimum_source_stocks,
        minimum_source_coverage_ratio=config.minimum_source_coverage_ratio,
    )
    expected_requests = len(candidates) * len(DATASET_TABLE)
    completed_requests = int(
        request_report["status"].isin(["complete", "already_current"]).sum()
    )
    failed_requests = int(
        request_report["status"].isin(["failed", "quota_exceeded"]).sum()
    )
    source_status = coverage_status(coverage)
    download_complete = completed_requests == expected_requests and failed_requests == 0

    live_result = LiveShardBuildResult(
        status="NOT_BUILT",
        shard_directory="",
        stock_files=0,
        total_rows=0,
        latest_signal_date="",
        failed_stocks=0,
        created_at="",
    )
    phase5b_result: Phase5BResult | None = None
    if download_complete and source_status == "PASS":
        live_result = build_live_feature_shards(
            database=database,
            candidates=candidates,
            target_date=decision.target_date,
            output_dir=output,
            live_shard_root=live_root,
            recent_rows_per_stock=config.recent_rows_per_stock,
        )
        if live_result.status == "PASS":
            phase5b_result = run_phase5b_daily_reference(
                database=database,
                output_dir=output,
                shard_root=Path(phase3_shard_root),
                live_shard_root=live_root,
                model_root=Path(model_root),
                model_settings=model_config,
                settings=daily_config,
                as_of_date=decision.as_of_date,
            )

    if not download_complete:
        overall_status = "INCOMPLETE_DOWNLOAD"
    elif source_status != "PASS":
        overall_status = "BLOCKED_SOURCE_COVERAGE"
    elif live_result.status != "PASS":
        overall_status = "LIVE_SHARD_BUILD_FAILED"
    elif phase5b_result is None:
        overall_status = "SELECTION_NOT_RUN"
    elif phase5b_result.status == "READY":
        overall_status = "READY"
    else:
        overall_status = phase5b_result.status

    summary = build_phase5c_summary(
        decision=decision,
        settings=config,
        candidate_count=len(candidates),
        request_report=request_report,
        coverage=coverage,
        source_status=source_status,
        live_result=live_result,
        phase5b_result=phase5b_result,
        overall_status=overall_status,
        calendar_status=calendar_status,
        calendar_error=calendar_error,
        request_count=client.request_count,
    )
    human_summary = build_phase5c_human_summary(summary, coverage)
    paths = export_phase5c_reports(
        output_dir=output,
        request_report=request_report,
        coverage=coverage,
        summary=summary,
        human_summary=human_summary,
        include_phase5b=phase5b_result is not None,
    )
    return Phase5CResult(
        status=overall_status,
        target_date=decision.target_date,
        completed_requests=completed_requests,
        expected_requests=expected_requests,
        source_coverage_status=source_status,
        selection_status=phase5b_result.status if phase5b_result else "NOT_RUN",
        output_paths=tuple(paths),
    )


def resolve_target_market_date(
    *,
    database: ResearchDatabase,
    now: datetime,
    as_of_date: str | None,
    explicit_target_date: str | None,
    publish_cutoff_time: str | None,
) -> TargetDateDecision:
    effective_as_of = _parse_iso_date(as_of_date) if as_of_date else now.date()
    if effective_as_of > now.date():
        raise ValueError("Phase 5C as-of date 不可晚於台北當地日期")
    calendar_dates = [
        str(row["date"])
        for row in database.query(
            "SELECT date FROM market_calendar WHERE date <= ? ORDER BY date",
            (effective_as_of.isoformat(),),
        )
    ]
    if not calendar_dates:
        raise RuntimeError("SQLite market_calendar 沒有可用交易日")

    cutoff_text = publish_cutoff_time or ""
    if explicit_target_date:
        target = _parse_iso_date(explicit_target_date)
        if target > effective_as_of:
            raise ValueError("Phase 5C target date 不可晚於 as-of date")
        if target.isoformat() not in calendar_dates:
            raise ValueError(f"Phase 5C target date 不是市場交易日：{target}")
        source = "explicit_target_date"
    else:
        include_as_of = effective_as_of < now.date()
        if effective_as_of == now.date() and publish_cutoff_time:
            cutoff = _parse_clock_time(publish_cutoff_time)
            include_as_of = now.timetz().replace(tzinfo=None) >= cutoff
        candidates = [
            value
            for value in calendar_dates
            if include_as_of or value < effective_as_of.isoformat()
        ]
        if not candidates:
            raise RuntimeError("找不到已完整結束的市場交易日")
        target = date.fromisoformat(candidates[-1])
        source = (
            "latest_market_date_at_or_before_as_of_after_cutoff"
            if include_as_of
            else "latest_completed_market_date_before_as_of"
        )
    return TargetDateDecision(
        as_of_date=effective_as_of.isoformat(),
        target_date=target.isoformat(),
        source=source,
        publish_cutoff_time=cutoff_text,
    )


def load_active_tpex_candidates(
    database: ResearchDatabase, target_date: str
) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in database.query(
            """
            SELECT stock_id, stock_name, market_type, listing_date,
                   delisting_date, current_status, training_enabled
            FROM model_universe
            WHERE LOWER(market_type)='tpex'
              AND current_status='active'
              AND training_enabled=1
              AND listing_date IS NOT NULL
              AND listing_date <= ?
            ORDER BY stock_id
            """,
            (target_date,),
        )
    ]


def select_pending_requests(
    *,
    database: ResearchDatabase,
    candidates: list[dict[str, Any]],
    target_date: str,
    warmup_calendar_days: int,
    attempted: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    status_rows = {
        (str(row["stock_id"]), str(row["dataset"])): dict(row)
        for row in database.query(
            "SELECT * FROM phase5c_update_status WHERE target_date=?",
            (target_date,),
        )
    }
    result: list[dict[str, Any]] = []
    for stock in candidates:
        stock_id = str(stock["stock_id"])
        for dataset, table in DATASET_TABLE.items():
            key = (stock_id, dataset)
            if key in attempted:
                continue
            latest_before = latest_stock_date(
                database,
                table=table,
                stock_id=stock_id,
                target_date=target_date,
            )
            if latest_before >= target_date:
                if key not in status_rows:
                    record_update_status(
                        database=database,
                        target_date=target_date,
                        stock_id=stock_id,
                        dataset=dataset,
                        requested_start=target_date,
                        requested_end=target_date,
                        status="already_current",
                        row_count=0,
                        latest_before=latest_before,
                        latest_after=latest_before,
                        error="",
                    )
                continue

            requested_start = incremental_start_date(
                latest_before=latest_before,
                listing_date=str(stock.get("listing_date") or ""),
                target_date=target_date,
                warmup_calendar_days=warmup_calendar_days,
            )
            result.append(
                {
                    **stock,
                    "dataset": dataset,
                    "table": table,
                    "latest_before": latest_before,
                    "requested_start": requested_start,
                }
            )
    return result


def incremental_start_date(
    *,
    latest_before: str,
    listing_date: str,
    target_date: str,
    warmup_calendar_days: int,
) -> str:
    target = date.fromisoformat(target_date)
    if latest_before:
        return (date.fromisoformat(latest_before) + timedelta(days=1)).isoformat()
    warmup_start = target - timedelta(days=warmup_calendar_days)
    if listing_date:
        warmup_start = max(warmup_start, date.fromisoformat(listing_date))
    return warmup_start.isoformat()


def process_incremental_request(
    *,
    database: ResearchDatabase,
    client: FinMindResearchClient,
    request: dict[str, Any],
    target_date: str,
) -> None:
    dataset = str(request["dataset"])
    stock_id = str(request["stock_id"])
    requested_start = str(request["requested_start"])
    latest_before = str(request["latest_before"])
    if requested_start > target_date:
        record_update_status(
            database=database,
            target_date=target_date,
            stock_id=stock_id,
            dataset=dataset,
            requested_start=requested_start,
            requested_end=target_date,
            status="already_current",
            row_count=0,
            latest_before=latest_before,
            latest_after=latest_before,
            error="",
        )
        return

    rows = client.fetch(
        dataset,
        data_id=stock_id,
        start_date=requested_start,
        end_date=target_date,
    )
    if dataset == PRICE_DATASET:
        row_count = upsert_prices(database, rows)
    elif dataset == FLOW_DATASET:
        row_count = upsert_institutional(database, rows)
    else:  # pragma: no cover - internal invariant
        raise ValueError(f"Phase 5C 不支援資料集：{dataset}")
    latest_after = latest_stock_date(
        database,
        table=str(request["table"]),
        stock_id=stock_id,
        target_date=target_date,
    )
    record_update_status(
        database=database,
        target_date=target_date,
        stock_id=stock_id,
        dataset=dataset,
        requested_start=requested_start,
        requested_end=target_date,
        status="complete",
        row_count=row_count,
        latest_before=latest_before,
        latest_after=latest_after,
        error="",
    )


def latest_stock_date(
    database: ResearchDatabase,
    *,
    table: str,
    stock_id: str,
    target_date: str,
) -> str:
    if table not in {"stock_prices", "institutional_flows"}:
        raise ValueError(f"不支援的來源表：{table}")
    value = database.scalar(
        f"SELECT MAX(date) FROM {table} WHERE stock_id=? AND date <= ?",
        (stock_id, target_date),
    )
    return str(value or "")


def record_update_status(
    *,
    database: ResearchDatabase,
    target_date: str,
    stock_id: str,
    dataset: str,
    requested_start: str,
    requested_end: str,
    status: str,
    row_count: int,
    latest_before: str,
    latest_after: str,
    error: str,
) -> None:
    database.execute(
        """
        INSERT INTO phase5c_update_status (
            target_date, stock_id, dataset, requested_start, requested_end,
            status, row_count, latest_before, latest_after, error, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(target_date, stock_id, dataset) DO UPDATE SET
            requested_start=excluded.requested_start,
            requested_end=excluded.requested_end,
            status=excluded.status,
            row_count=excluded.row_count,
            latest_before=excluded.latest_before,
            latest_after=excluded.latest_after,
            error=excluded.error,
            updated_at=CURRENT_TIMESTAMP
        """,
        (
            target_date,
            stock_id,
            dataset,
            requested_start,
            requested_end,
            status,
            row_count,
            latest_before,
            latest_after,
            error,
        ),
    )


def build_update_status_report(
    *,
    database: ResearchDatabase,
    candidates: list[dict[str, Any]],
    target_date: str,
) -> pd.DataFrame:
    names = {str(row["stock_id"]): str(row.get("stock_name") or "") for row in candidates}
    rows = [
        dict(row)
        for row in database.query(
            """
            SELECT target_date, stock_id, dataset, requested_start, requested_end,
                   status, row_count, latest_before, latest_after, error, updated_at
            FROM phase5c_update_status
            WHERE target_date=?
            ORDER BY stock_id, dataset
            """,
            (target_date,),
        )
    ]
    existing = {(str(row["stock_id"]), str(row["dataset"])) for row in rows}
    for stock in candidates:
        for dataset in DATASET_TABLE:
            key = (str(stock["stock_id"]), dataset)
            if key in existing:
                continue
            rows.append(
                {
                    "target_date": target_date,
                    "stock_id": key[0],
                    "dataset": dataset,
                    "requested_start": "",
                    "requested_end": target_date,
                    "status": "pending",
                    "row_count": 0,
                    "latest_before": latest_stock_date(
                        database,
                        table=DATASET_TABLE[dataset],
                        stock_id=key[0],
                        target_date=target_date,
                    ),
                    "latest_after": "",
                    "error": "",
                    "updated_at": "",
                }
            )
    frame = pd.DataFrame(rows)
    frame.insert(2, "stock_name", frame["stock_id"].map(names).fillna(""))
    frame["reached_target_date"] = (
        frame["latest_after"].astype(str) >= target_date
    ).astype(int)
    return frame.sort_values(["stock_id", "dataset"], kind="mergesort").reset_index(
        drop=True
    )


def build_source_coverage(
    *,
    database: ResearchDatabase,
    target_date: str,
    recent_dates: int,
    minimum_source_stocks: int,
    minimum_source_coverage_ratio: float,
) -> pd.DataFrame:
    market_dates = [
        str(row["date"])
        for row in database.query(
            "SELECT date FROM market_calendar WHERE date <= ? ORDER BY date DESC LIMIT ?",
            (target_date, recent_dates + 1),
        )
    ]
    market_dates = sorted(market_dates)
    rows: list[dict[str, Any]] = []
    for market_date in market_dates:
        price_count = int(
            database.scalar(
                """
                SELECT COUNT(DISTINCT u.stock_id)
                FROM model_universe u
                JOIN stock_prices p ON p.stock_id=u.stock_id AND p.date=?
                WHERE LOWER(u.market_type)='tpex'
                  AND u.current_status='active' AND u.training_enabled=1
                """,
                (market_date,),
            )
            or 0
        )
        flow_count = int(
            database.scalar(
                """
                SELECT COUNT(DISTINCT u.stock_id)
                FROM model_universe u
                JOIN institutional_flows f ON f.stock_id=u.stock_id AND f.date=?
                WHERE LOWER(u.market_type)='tpex'
                  AND u.current_status='active' AND u.training_enabled=1
                """,
                (market_date,),
            )
            or 0
        )
        common_count = int(
            database.scalar(
                """
                SELECT COUNT(DISTINCT u.stock_id)
                FROM model_universe u
                JOIN stock_prices p ON p.stock_id=u.stock_id AND p.date=?
                JOIN institutional_flows f ON f.stock_id=u.stock_id AND f.date=?
                WHERE LOWER(u.market_type)='tpex'
                  AND u.current_status='active' AND u.training_enabled=1
                """,
                (market_date, market_date),
            )
            or 0
        )
        rows.append(
            {
                "market_date": market_date,
                "price_stock_count": price_count,
                "institutional_flow_stock_count": flow_count,
                "common_stock_count": common_count,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "market_date",
                "price_stock_count",
                "institutional_flow_stock_count",
                "common_stock_count",
                "required_common_stock_count",
                "coverage_ratio_vs_prior_median",
                "is_target_date",
                "source_coverage_pass",
            ]
        )
    prior = frame[frame["market_date"] < target_date]["common_stock_count"].tolist()
    prior_median = float(median(prior)) if prior else 0.0
    required = max(
        minimum_source_stocks,
        int(math.floor(prior_median * minimum_source_coverage_ratio)),
    )
    frame["required_common_stock_count"] = required
    frame["coverage_ratio_vs_prior_median"] = frame["common_stock_count"] / max(
        prior_median, 1.0
    )
    frame["is_target_date"] = (frame["market_date"] == target_date).astype(int)
    frame["source_coverage_pass"] = (
        (frame["common_stock_count"] >= required)
        & (frame["price_stock_count"] >= required)
        & (frame["institutional_flow_stock_count"] >= required)
    ).astype(int)
    return frame.sort_values("market_date", ascending=False, kind="mergesort").reset_index(
        drop=True
    )


def coverage_status(coverage: pd.DataFrame) -> str:
    target = coverage[coverage["is_target_date"] == 1] if not coverage.empty else coverage
    if target.empty:
        return "TARGET_DATE_MISSING"
    return (
        "PASS"
        if int(target.iloc[0]["source_coverage_pass"]) == 1
        else "INCOMPLETE_TARGET_COVERAGE"
    )


def build_live_feature_shards(
    *,
    database: ResearchDatabase,
    candidates: list[dict[str, Any]],
    target_date: str,
    output_dir: Path,
    live_shard_root: Path,
    recent_rows_per_stock: int,
) -> LiveShardBuildResult:
    market_dates = [
        str(row["date"])
        for row in database.query(
            "SELECT date FROM market_calendar WHERE date <= ? ORDER BY date",
            (target_date,),
        )
    ]
    if not market_dates:
        raise RuntimeError("Phase 5C 無法建立 live shard：市場交易日曆為空")

    shard_name = target_date.replace("-", "")
    final_dir = live_shard_root / shard_name
    temporary_dir = live_shard_root / f".{shard_name}.tmp"
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True)

    total_rows = 0
    stock_files = 0
    failed_stocks = 0
    latest_signal_date = ""
    failures: list[dict[str, str]] = []
    for position, stock in enumerate(candidates, start=1):
        stock_id = str(stock["stock_id"])
        try:
            rows = build_recent_stock_feature_rows(
                database=database,
                stock=stock,
                end_date=target_date,
                recent_rows=recent_rows_per_stock,
                market_dates=market_dates,
            )
            if not rows:
                continue
            path = temporary_dir / f"{stock_id}.csv.gz"
            with gzip.open(path, "wt", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=ALL_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            stock_files += 1
            total_rows += len(rows)
            latest_signal_date = max(latest_signal_date, str(rows[-1]["signal_date"]))
        except Exception as exc:
            failed_stocks += 1
            failures.append({"stock_id": stock_id, "error": str(exc)})
        if position % 100 == 0 or position == len(candidates):
            print(f"  Phase 5C 即時特徵分片：{position}/{len(candidates)}")

    created_at = datetime.now(TAIPEI_TIMEZONE).isoformat(timespec="seconds")
    if failures:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        write_csv(output_dir / "phase5c_live_shard_failures.csv", pd.DataFrame(failures))
        return LiveShardBuildResult(
            status="FAILED",
            shard_directory="",
            stock_files=stock_files,
            total_rows=total_rows,
            latest_signal_date=latest_signal_date,
            failed_stocks=failed_stocks,
            created_at=created_at,
        )

    if final_dir.exists():
        shutil.rmtree(final_dir)
    os.replace(temporary_dir, final_dir)
    pointer = {
        "phase5c_version": PHASE5C_VERSION,
        "status": "PASS",
        "target_date": target_date,
        "shard_directory": final_dir.name,
        "stock_files": stock_files,
        "total_rows": total_rows,
        "latest_signal_date": latest_signal_date,
        "recent_rows_per_stock": recent_rows_per_stock,
        "created_at": created_at,
    }
    write_json_atomic(live_shard_root / "latest.json", pointer)
    write_csv(output_dir / "phase5c_live_manifest.csv", pd.DataFrame([pointer]))
    return LiveShardBuildResult(
        status="PASS",
        shard_directory=final_dir.name,
        stock_files=stock_files,
        total_rows=total_rows,
        latest_signal_date=latest_signal_date,
        failed_stocks=0,
        created_at=created_at,
    )


def build_phase5c_summary(
    *,
    decision: TargetDateDecision,
    settings: Phase5CSettings,
    candidate_count: int,
    request_report: pd.DataFrame,
    coverage: pd.DataFrame,
    source_status: str,
    live_result: LiveShardBuildResult,
    phase5b_result: Phase5BResult | None,
    overall_status: str,
    calendar_status: str,
    calendar_error: str,
    request_count: int,
) -> pd.DataFrame:
    target_coverage = coverage[coverage["is_target_date"] == 1]
    target_row = target_coverage.iloc[0].to_dict() if not target_coverage.empty else {}
    expected_requests = candidate_count * len(DATASET_TABLE)
    completed = int(request_report["status"].isin(["complete", "already_current"]).sum())
    failed = int(request_report["status"].isin(["failed", "quota_exceeded"]).sum())
    values: list[tuple[str, Any]] = [
        ("phase5c_version", PHASE5C_VERSION),
        ("pipeline_status", overall_status),
        ("target_market", TARGET_MARKET),
        ("as_of_date", decision.as_of_date),
        ("target_date", decision.target_date),
        ("target_date_source", decision.source),
        ("publish_cutoff_time", decision.publish_cutoff_time),
        ("calendar_refresh_status", calendar_status),
        ("calendar_refresh_error", calendar_error),
        ("candidate_stocks", candidate_count),
        ("expected_dataset_requests", expected_requests),
        ("completed_dataset_requests", completed),
        ("failed_dataset_requests", failed),
        ("pending_dataset_requests", max(0, expected_requests - completed)),
        ("finmind_requests_this_run", request_count),
        ("source_coverage_status", source_status),
        ("target_price_stock_count", target_row.get("price_stock_count", 0)),
        (
            "target_institutional_flow_stock_count",
            target_row.get("institutional_flow_stock_count", 0),
        ),
        ("target_common_stock_count", target_row.get("common_stock_count", 0)),
        (
            "required_common_stock_count",
            target_row.get("required_common_stock_count", settings.minimum_source_stocks),
        ),
        (
            "target_coverage_ratio_vs_prior_median",
            target_row.get("coverage_ratio_vs_prior_median", 0),
        ),
        ("live_shard_status", live_result.status),
        ("live_shard_directory", live_result.shard_directory),
        ("live_shard_stock_files", live_result.stock_files),
        ("live_shard_total_rows", live_result.total_rows),
        ("live_shard_latest_signal_date", live_result.latest_signal_date),
        ("live_shard_failed_stocks", live_result.failed_stocks),
        ("phase5b_selection_status", phase5b_result.status if phase5b_result else "NOT_RUN"),
        ("phase5b_selection_signal_date", phase5b_result.signal_date if phase5b_result else ""),
        ("phase5b_usable_selection_rows", phase5b_result.selected_rows if phase5b_result else 0),
        ("deployment_status", "LOCAL_RESEARCH_ONLY"),
    ]
    for key, value in asdict(settings).items():
        values.append((f"setting_{key}", value))
    return pd.DataFrame(values, columns=["metric", "value"])


def build_phase5c_human_summary(summary: pd.DataFrame, coverage: pd.DataFrame) -> str:
    values = {str(row.metric): str(row.value) for row in summary.itertuples(index=False)}
    status = values.get("pipeline_status", "")
    lines = [
        "# Phase 5C TPEx 每日增量更新摘要",
        "",
        f"- 執行狀態：`{status}`",
        f"- 目標市場日：`{values.get('target_date', '')}`",
        f"- 資料請求：{values.get('completed_dataset_requests', '0')} / "
        f"{values.get('expected_dataset_requests', '0')}",
        f"- 目標日股價股票數：{values.get('target_price_stock_count', '0')}",
        f"- 目標日法人股票數：{values.get('target_institutional_flow_stock_count', '0')}",
        f"- 股價與法人共同股票數：{values.get('target_common_stock_count', '0')}",
        f"- 來源覆蓋狀態：`{values.get('source_coverage_status', '')}`",
        f"- 選股名單狀態：`{values.get('phase5b_selection_status', 'NOT_RUN')}`",
        "",
    ]
    if status == "READY":
        lines.extend(
            [
                "本次資料更新與即時特徵分片均完成，Phase 5B 已產生可使用的本機選股參考名單。",
                "該排名仍是法人行為的相對選股指數，不是上漲機率或買進指令。",
            ]
        )
    elif status == "INCOMPLETE_DOWNLOAD":
        lines.append("尚有股票或資料集未完成；重新執行相同命令會沿用 SQLite 進度續傳。")
    elif status == "BLOCKED_SOURCE_COVERAGE":
        lines.append("API 請求雖已完成，但目標日來源股票數不足，未建立新選股名單。")
    else:
        lines.append("本次沒有產生 READY 名單，請依 phase5c_summary.csv 的狀態處理。")
    if not coverage.empty:
        lines.extend(["", "## 最近市場日來源覆蓋", ""])
        for row in coverage.sort_values("market_date", ascending=False).itertuples(index=False):
            lines.append(
                f"- {row.market_date}：股價 {row.price_stock_count}、法人 "
                f"{row.institutional_flow_stock_count}、共同 {row.common_stock_count}"
            )
    return "\n".join(lines) + "\n"


def export_phase5c_reports(
    *,
    output_dir: Path,
    request_report: pd.DataFrame,
    coverage: pd.DataFrame,
    summary: pd.DataFrame,
    human_summary: str,
    include_phase5b: bool,
) -> list[Path]:
    paths = [
        write_csv(output_dir / "phase5c_incremental_update_status.csv", request_report),
        write_csv(output_dir / "phase5c_source_coverage.csv", coverage),
        write_csv(output_dir / "phase5c_summary.csv", summary),
    ]
    manifest_path = output_dir / "phase5c_live_manifest.csv"
    if manifest_path.exists():
        paths.append(manifest_path)
    markdown_path = output_dir / "phase5c_daily_update_summary.md"
    markdown_path.write_text(human_summary, encoding="utf-8")
    paths.append(markdown_path)

    if include_phase5b:
        for name in (
            "phase5b_summary.csv",
            "phase5b_data_source_freshness.csv",
            "phase5b_recent_date_diagnostics.csv",
            "phase5b_selection_index.csv",
            "phase5b_selection_index_top20_stocks.csv",
            "phase5b_selection_index_top20_percent.csv",
            "phase5b_daily_selection_summary.md",
        ):
            path = output_dir / name
            if path.exists():
                paths.append(path)

    archive = output_dir / "phase5c_daily_update_reports.zip"
    temporary = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in paths:
            handle.write(path, arcname=path.name)
    os.replace(temporary, archive)
    paths.append(archive)
    return paths


def _export_incomplete_result(
    *,
    output_dir: Path,
    decision: TargetDateDecision,
    calendar_status: str,
    calendar_error: str,
    client: FinMindResearchClient,
    settings: Phase5CSettings,
) -> Phase5CResult:
    summary = pd.DataFrame(
        [
            ("phase5c_version", PHASE5C_VERSION),
            ("pipeline_status", "CALENDAR_REFRESH_FAILED"),
            ("as_of_date", decision.as_of_date),
            ("target_date", decision.target_date),
            ("calendar_refresh_status", calendar_status),
            ("calendar_refresh_error", calendar_error),
            ("finmind_requests_this_run", client.request_count),
            ("deployment_status", "LOCAL_RESEARCH_ONLY"),
        ],
        columns=["metric", "value"],
    )
    empty_status = pd.DataFrame(
        columns=[
            "target_date",
            "stock_id",
            "stock_name",
            "dataset",
            "requested_start",
            "requested_end",
            "status",
            "row_count",
            "latest_before",
            "latest_after",
            "error",
            "updated_at",
            "reached_target_date",
        ]
    )
    empty_coverage = pd.DataFrame(
        columns=[
            "market_date",
            "price_stock_count",
            "institutional_flow_stock_count",
            "common_stock_count",
            "required_common_stock_count",
            "coverage_ratio_vs_prior_median",
            "is_target_date",
            "source_coverage_pass",
        ]
    )
    paths = export_phase5c_reports(
        output_dir=output_dir,
        request_report=empty_status,
        coverage=empty_coverage,
        summary=summary,
        human_summary=(
            "# Phase 5C TPEx 每日增量更新摘要\n\n"
            f"市場交易日曆更新失敗：{calendar_error}\n"
        ),
        include_phase5b=False,
    )
    return Phase5CResult(
        status="CALENDAR_REFRESH_FAILED",
        target_date=decision.target_date,
        completed_requests=0,
        expected_requests=0,
        source_coverage_status="NOT_CHECKED",
        selection_status="NOT_RUN",
        output_paths=tuple(paths),
    )


def _parse_iso_date(value: str | None) -> date:
    try:
        return date.fromisoformat(str(value or ""))
    except ValueError as exc:
        raise ValueError(f"日期格式必須是 YYYY-MM-DD：{value}") from exc


def _parse_clock_time(value: str) -> clock_time:
    try:
        return clock_time.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"發布截止時間格式必須是 HH:MM：{value}") from exc
