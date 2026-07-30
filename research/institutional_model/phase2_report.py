from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from research.institutional_model.database import ResearchDatabase
from research.institutional_model.downloader import PER_STOCK_DATASETS
from research.institutional_model.phase2_downloader import (
    count_remaining_requests,
    effective_request_range,
)


def export_phase2_reports(
    *,
    database: ResearchDatabase,
    output_dir: Path | str,
    start_date: str,
    end_date: str,
    include_unclassified: bool = False,
) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    universe_rows = [
        dict(row)
        for row in database.query(
            """
            SELECT * FROM model_universe
            ORDER BY market_type, current_status, stock_id
            """
        )
    ]
    progress_rows = _build_progress_rows(database, universe_rows, start_date, end_date)
    unresolved_rows = [
        row
        for row in universe_rows
        if row.get("inclusion_status") == "review_required"
    ]
    remaining = count_remaining_requests(
        database=database,
        start_date=start_date,
        end_date=end_date,
        include_unclassified=include_unclassified,
    )
    history_rows = [
        dict(row)
        for row in database.query(
            """
            SELECT *
            FROM phase2_batch_history
            ORDER BY id
            """
        )
    ]
    summary_rows = _summary_rows(
        universe_rows,
        progress_rows,
        remaining,
        history_rows,
    )

    paths = [
        _write_csv(output / "phase2_universe.csv", universe_rows),
        _write_csv(output / "phase2_download_progress.csv", progress_rows),
        _write_csv(output / "phase2_unresolved_universe.csv", unresolved_rows),
        _write_csv(output / "phase2_summary.csv", summary_rows),
        _write_csv(output / "phase2_batch_history.csv", history_rows),
    ]
    return paths


def _build_progress_rows(
    database: ResearchDatabase,
    universe_rows: list[dict[str, Any]],
    start_date: str,
    end_date: str,
) -> list[dict[str, Any]]:
    status_rows = database.query("SELECT * FROM download_status")
    status_map = {
        (row["dataset"], row["stock_id"]): dict(row) for row in status_rows
    }
    result: list[dict[str, Any]] = []
    for universe in universe_rows:
        request_start, request_end = effective_request_range(
            universe, start_date, end_date
        )
        row: dict[str, Any] = {
            "stock_id": universe["stock_id"],
            "stock_name": universe["stock_name"],
            "market_type": universe["market_type"],
            "current_status": universe["current_status"],
            "request_start": request_start,
            "request_end": request_end,
            "download_enabled": universe["download_enabled"],
            "training_enabled": universe["training_enabled"],
        }
        complete_count = 0
        for dataset in PER_STOCK_DATASETS:
            item = status_map.get((dataset, universe["stock_id"]), {})
            complete = (
                item.get("status") == "complete"
                and str(item.get("requested_start") or "") <= request_start
                and str(item.get("requested_end") or "") >= request_end
            )
            if complete:
                complete_count += 1
            short = _dataset_short_name(dataset)
            row[f"{short}_status"] = item.get("status") or "pending"
            row[f"{short}_rows"] = item.get("row_count") or 0
            row[f"{short}_error"] = item.get("error") or ""
        row["complete_datasets"] = complete_count
        row["remaining_datasets"] = len(PER_STOCK_DATASETS) - complete_count
        row["stock_complete"] = 1 if complete_count == len(PER_STOCK_DATASETS) else 0
        result.append(row)
    return result


def _summary_rows(
    universe_rows: list[dict[str, Any]],
    progress_rows: list[dict[str, Any]],
    remaining_requests: int,
    history_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    def count(**conditions: Any) -> int:
        return sum(
            1
            for row in universe_rows
            if all(row.get(key) == value for key, value in conditions.items())
        )

    return [
        {"metric": "universe_total", "value": len(universe_rows)},
        {"metric": "twse_total", "value": count(market_type="twse")},
        {"metric": "tpex_total", "value": count(market_type="tpex")},
        {"metric": "unknown_market_total", "value": count(market_type="unknown")},
        {"metric": "active_total", "value": count(current_status="active")},
        {"metric": "delisted_total", "value": count(current_status="delisted")},
        {
            "metric": "training_enabled_total",
            "value": sum(int(row.get("training_enabled") or 0) for row in universe_rows),
        },
        {
            "metric": "training_enabled_twse",
            "value": sum(
                1
                for row in universe_rows
                if row.get("market_type") == "twse"
                and int(row.get("training_enabled") or 0)
            ),
        },
        {
            "metric": "training_enabled_tpex",
            "value": sum(
                1
                for row in universe_rows
                if row.get("market_type") == "tpex"
                and int(row.get("training_enabled") or 0)
            ),
        },
        {
            "metric": "review_required_total",
            "value": count(inclusion_status="review_required"),
        },
        {
            "metric": "excluded_total",
            "value": count(inclusion_status="excluded"),
        },
        {
            "metric": "download_complete_stocks",
            "value": sum(int(row.get("stock_complete") or 0) for row in progress_rows),
        },
        {
            "metric": "download_remaining_stocks",
            "value": sum(1 for row in progress_rows if not int(row.get("stock_complete") or 0)),
        },
        {
            "metric": "completed_datasets",
            "value": sum(int(row.get("complete_datasets") or 0) for row in progress_rows),
        },
        {
            "metric": "total_datasets",
            "value": len(progress_rows) * len(PER_STOCK_DATASETS),
        },
        {"metric": "remaining_api_requests_estimate", "value": remaining_requests},
        {"metric": "recorded_batch_count", "value": len(history_rows)},
        {
            "metric": "recorded_api_requests_since_v0_2_1",
            "value": sum(int(row.get("requests_made") or 0) for row in history_rows),
        },
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    if rows:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    else:
        fieldnames = ["stock_id"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _dataset_short_name(dataset: str) -> str:
    return {
        "TaiwanStockPrice": "price",
        "TaiwanStockInstitutionalInvestorsBuySell": "institutional",
        "TaiwanStockDividendResult": "dividend",
        "TaiwanStockCapitalReductionReferencePrice": "capital_reduction",
    }[dataset]
