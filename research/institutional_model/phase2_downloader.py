from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from research.institutional_model.corporate_actions import cleanup_duplicate_corporate_actions
from research.institutional_model.database import ResearchDatabase
from research.institutional_model.downloader import (
    PER_STOCK_DATASETS,
    _download_stock_dataset,
)
from research.institutional_model.finmind_client import (
    FinMindQuotaExceeded,
    FinMindResearchClient,
)


@dataclass
class Phase2BatchResult:
    selected_stocks: int = 0
    completed_stocks: int = 0
    failed_stocks: int = 0
    skipped_stocks: int = 0
    quota_exhausted: bool = False
    requests_made: int = 0


@dataclass(frozen=True)
class Phase2Progress:
    total_stocks: int
    completed_stocks: int
    remaining_stocks: int
    total_datasets: int
    completed_datasets: int
    remaining_requests: int


def get_phase2_progress(
    *,
    database: ResearchDatabase,
    start_date: str,
    end_date: str,
    include_unclassified: bool = False,
) -> Phase2Progress:
    conditions = ["download_enabled=1"]
    if not include_unclassified:
        conditions.append("market_type IN ('twse', 'tpex')")
    rows = [
        dict(row)
        for row in database.query(
            f"""
            SELECT *
            FROM model_universe
            WHERE {' AND '.join(conditions)}
            """
        )
    ]

    completed_stocks = 0
    completed_datasets = 0
    remaining_requests = 0
    for row in rows:
        request_start, request_end = effective_request_range(row, start_date, end_date)
        if request_start > request_end:
            completed_stocks += 1
            completed_datasets += len(PER_STOCK_DATASETS)
            continue
        stock_complete = True
        for dataset in PER_STOCK_DATASETS:
            complete = database.download_is_complete(
                dataset=dataset,
                stock_id=row["stock_id"],
                requested_start=request_start,
                requested_end=request_end,
            )
            if complete:
                completed_datasets += 1
            else:
                stock_complete = False
                remaining_requests += 1
        if stock_complete:
            completed_stocks += 1

    total_stocks = len(rows)
    return Phase2Progress(
        total_stocks=total_stocks,
        completed_stocks=completed_stocks,
        remaining_stocks=max(0, total_stocks - completed_stocks),
        total_datasets=total_stocks * len(PER_STOCK_DATASETS),
        completed_datasets=completed_datasets,
        remaining_requests=remaining_requests,
    )


def download_phase2_batch(
    *,
    database: ResearchDatabase,
    client: FinMindResearchClient,
    start_date: str,
    end_date: str,
    max_stocks: int = 100,
    market: str = "all",
    include_unclassified: bool = False,
    force: bool = False,
    stop_on_error: bool = False,
    overall_completed_before: int = 0,
    overall_total_stocks: int = 0,
    remaining_requests_before: int = 0,
) -> Phase2BatchResult:
    candidates = select_pending_stocks(
        database=database,
        start_date=start_date,
        end_date=end_date,
        market=market,
        include_unclassified=include_unclassified,
        force=force,
    )
    if max_stocks > 0:
        candidates = candidates[:max_stocks]

    request_count_at_start = client.request_count
    result = Phase2BatchResult(selected_stocks=len(candidates))
    processed_symbols: list[str] = []
    for index, row in enumerate(candidates, start=1):
        stock_id = row["stock_id"]
        stock_name = row["stock_name"] or ""
        request_start, request_end = effective_request_range(row, start_date, end_date)
        if request_start > request_end:
            result.skipped_stocks += 1
            continue

        approximate_overall = overall_completed_before + result.completed_stocks + 1
        total_text = str(overall_total_stocks) if overall_total_stocks else "?"
        print(
            f"[本批 {index}/{len(candidates)} | 全體約 {approximate_overall}/{total_text}] "
            f"{stock_id} {stock_name} {row['market_type']} "
            f"{request_start}~{request_end}"
        )
        try:
            for dataset in PER_STOCK_DATASETS:
                _download_stock_dataset(
                    database=database,
                    client=client,
                    dataset=dataset,
                    stock_id=stock_id,
                    start_date=request_start,
                    end_date=request_end,
                    force=force,
                )
            result.completed_stocks += 1
            processed_symbols.append(stock_id)
            batch_requests = client.request_count - request_count_at_start
            estimated_remaining = max(0, remaining_requests_before - batch_requests)
            print(
                f"  完成股票：{overall_completed_before + result.completed_stocks}/"
                f"{total_text}；本次 API：{batch_requests}；"
                f"預估剩餘 API：{estimated_remaining}"
            )
        except FinMindQuotaExceeded as exc:
            result.quota_exhausted = True
            print(f"FinMind API 額度已用完，本批次安全停止：{exc}")
            break
        except Exception as exc:
            result.failed_stocks += 1
            print(f"{stock_id} 下載失敗：{exc}")
            if stop_on_error:
                raise

    if processed_symbols:
        removed = cleanup_duplicate_corporate_actions(database, processed_symbols)
        if removed:
            print(f"已清除重複或失效的公司行動日期：{removed} 筆")
    result.requests_made = client.request_count - request_count_at_start
    return result


def select_pending_stocks(
    *,
    database: ResearchDatabase,
    start_date: str,
    end_date: str,
    market: str,
    include_unclassified: bool,
    force: bool,
) -> list[dict[str, Any]]:
    conditions = ["u.download_enabled=1"]
    params: list[Any] = []
    if market != "all":
        conditions.append("u.market_type=?")
        params.append(market)
    elif not include_unclassified:
        conditions.append("u.market_type IN ('twse', 'tpex')")

    rows = database.query(
        f"""
        SELECT u.*
        FROM model_universe u
        WHERE {' AND '.join(conditions)}
        ORDER BY
            CASE WHEN u.current_status='active' THEN 0 ELSE 1 END,
            CASE u.market_type WHEN 'twse' THEN 0 WHEN 'tpex' THEN 1 ELSE 2 END,
            u.stock_id
        """,
        tuple(params),
    )

    result: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        request_start, request_end = effective_request_range(row, start_date, end_date)
        if request_start > request_end:
            continue
        if force or not _all_datasets_complete(
            database, row["stock_id"], request_start, request_end
        ):
            result.append(row)
    return result


def effective_request_range(
    row: dict[str, Any], start_date: str, end_date: str
) -> tuple[str, str]:
    request_start = start_date
    listing_date = str(row.get("listing_date") or "")
    if listing_date and listing_date > request_start:
        request_start = listing_date

    request_end = end_date
    delisting_date = str(row.get("delisting_date") or "")
    if delisting_date:
        try:
            last_trade_candidate = (
                date.fromisoformat(delisting_date) - timedelta(days=1)
            ).isoformat()
            if last_trade_candidate < request_end:
                request_end = last_trade_candidate
        except ValueError:
            pass
    return request_start, request_end


def count_remaining_requests(
    *,
    database: ResearchDatabase,
    start_date: str,
    end_date: str,
    include_unclassified: bool = False,
) -> int:
    return get_phase2_progress(
        database=database,
        start_date=start_date,
        end_date=end_date,
        include_unclassified=include_unclassified,
    ).remaining_requests


def _all_datasets_complete(
    database: ResearchDatabase, stock_id: str, start_date: str, end_date: str
) -> bool:
    return all(
        database.download_is_complete(
            dataset=dataset,
            stock_id=stock_id,
            requested_start=start_date,
            requested_end=end_date,
        )
        for dataset in PER_STOCK_DATASETS
    )
