from __future__ import annotations

import csv
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from research.institutional_model.database import ResearchDatabase
from research.institutional_model.universe import is_common_stock


VALID_MARKETS = {"twse", "tpex"}
REQUIRED_STOCK_DATASETS = {
    "TaiwanStockPrice",
    "TaiwanStockInstitutionalInvestorsBuySell",
    "TaiwanStockDividendResult",
    "TaiwanStockCapitalReductionReferencePrice",
}


def build_phase2_universe(
    *,
    database: ResearchDatabase,
    start_date: str,
    end_date: str,
    overrides_path: Path | str | None = None,
) -> dict[str, int]:
    overrides = load_market_overrides(overrides_path) if overrides_path else {}
    stock_info = {
        row["stock_id"]: dict(row)
        for row in database.query("SELECT * FROM stock_info")
    }
    official_rows = [
        dict(row)
        for row in database.query(
            """
            SELECT *
            FROM official_company_info
            WHERE market_type IN ('twse', 'tpex')
            """
        )
    ]
    delistings = {
        row["stock_id"]: dict(row)
        for row in database.query(
            """
            SELECT stock_id, MAX(date) AS date, MAX(stock_name) AS stock_name
            FROM delistings
            WHERE date >= ? AND date <= ?
            GROUP BY stock_id
            """,
            (start_date, end_date),
        )
    }
    twse_delisted = {
        row["stock_id"]: dict(row)
        for row in database.query("SELECT * FROM official_twse_delisted")
    }
    first_flow_dates = {
        row["stock_id"]: row["first_date"]
        for row in database.query(
            """
            SELECT f.stock_id, MIN(f.date) AS first_date
            FROM institutional_flows f
            JOIN market_calendar c ON c.date=f.date
            JOIN stock_prices p
              ON p.stock_id=f.stock_id AND p.date=f.date
            WHERE p.open > 0 AND p.close > 0 AND p.trading_volume > 0
            GROUP BY f.stock_id
            """
        )
    }
    first_price_dates = {
        row["stock_id"]: row["first_date"]
        for row in database.query(
            """
            SELECT stock_id, MIN(date) AS first_date
            FROM stock_prices
            WHERE open > 0 AND close > 0 AND trading_volume > 0
            GROUP BY stock_id
            """
        )
    }
    download_status: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in database.query(
        """
        SELECT stock_id, dataset, requested_start, requested_end, status
        FROM download_status
        WHERE stock_id <> '*'
        """
    ):
        download_status[row["stock_id"]][row["dataset"]] = dict(row)

    universe: dict[str, dict[str, Any]] = {}
    for official in official_rows:
        stock_id = str(official["stock_id"])
        market_type = str(official["market_type"] or "").lower()
        info = stock_info.get(stock_id, {})
        normalized = {
            "stock_id": stock_id,
            "stock_name": official.get("stock_name") or info.get("stock_name"),
            "industry_category": info.get("industry_category"),
            "type": market_type,
        }
        if not is_common_stock(normalized):
            continue

        listing_date = official.get("listing_date")
        future_listing = bool(listing_date and str(listing_date) > end_date)
        prior_delisting = delistings.get(stock_id)
        transfer_record = bool(
            prior_delisting
            and listing_date
            and str(prior_delisting.get("date") or "") <= str(listing_date)
        )
        listing_ready = bool(listing_date)
        if future_listing:
            reason = f"掛牌日 {listing_date} 晚於研究截止日 {end_date}，不納入本次快照"
            source = "official_company_info:future_listing_boundary"
        elif transfer_record:
            reason = (
                "目前仍在官方掛牌清單；曾有終止上市櫃紀錄，"
                "僅納入目前掛牌日起資料"
            )
            source = "official_company_info+FinMind:market_transfer"
        elif listing_ready:
            reason = "目前為上市／上櫃普通股，且已取得官方掛牌日期"
            source = "official_company_info+FinMind:TaiwanStockInfo"
        else:
            reason = "目前為上市／上櫃普通股，但缺少官方掛牌日期，暫不進模型"
            source = "official_company_info+FinMind:TaiwanStockInfo"

        universe[stock_id] = {
            "stock_id": stock_id,
            "stock_name": official.get("stock_name") or info.get("stock_name") or "",
            "market_type": market_type,
            "industry_category": info.get("industry_category") or "",
            "listing_date": listing_date,
            "delisting_date": None,
            "current_status": "active",
            "download_enabled": 0 if future_listing else 1,
            "training_enabled": 1 if listing_ready and not future_listing else 0,
            "inclusion_status": (
                "excluded" if future_listing
                else "included" if listing_ready
                else "review_required"
            ),
            "inclusion_reason": reason,
            "source": source,
        }

    for stock_id, delisting in delistings.items():
        if stock_id in universe:
            continue

        info = stock_info.get(stock_id, {})
        override = overrides.get(stock_id, {})
        market_type = str(
            override.get("market_type")
            or info.get("market_type")
            or ("twse" if stock_id in twse_delisted else "unknown")
        ).lower()
        industry = str(info.get("industry_category") or "")
        normalized = {
            "stock_id": stock_id,
            "stock_name": info.get("stock_name") or delisting.get("stock_name"),
            "industry_category": industry,
            "type": market_type,
        }
        if not is_common_stock(normalized):
            continue

        delisting_date = str(delisting.get("date") or "")
        expected_end = _day_before(delisting_date) or end_date
        coverage_complete = _datasets_cover_range(
            download_status.get(stock_id, {}),
            start_date,
            min(end_date, expected_end),
        )
        first_flow = str(first_flow_dates.get(stock_id) or "")
        inferred_listing_date = (
            first_flow
            if coverage_complete and first_flow and first_flow < delisting_date
            else None
        )
        listing_date = override.get("listing_date") or inferred_listing_date
        has_price = bool(first_price_dates.get(stock_id))

        if listing_date:
            training_enabled = 1
            inclusion_status = "included"
            if override.get("listing_date"):
                reason = override.get("note") or "歷史掛牌日期由人工覆寫提供"
                source = "market_overrides_v1.csv+FinMind:TaiwanStockDelisting"
            else:
                reason = (
                    "歷史終止上市櫃普通股；為避免納入興櫃期間，"
                    "可用起日保守採首筆有效法人交易日"
                )
                source = "FinMind:TaiwanStockDelisting+first_institutional_date"
        elif coverage_complete and not has_price:
            training_enabled = 0
            inclusion_status = "excluded"
            reason = "研究期間內沒有有效交易價格，無法產生模型樣本"
            source = "FinMind:TaiwanStockDelisting+download_audit"
        elif coverage_complete:
            training_enabled = 0
            inclusion_status = "excluded"
            reason = "研究期間內沒有法人交易資料，無法產生法人行為訊號"
            source = "FinMind:TaiwanStockDelisting+download_audit"
        else:
            training_enabled = 0
            inclusion_status = "review_required"
            reason = "歷史終止上市櫃普通股，待完成下載後確認首筆有效法人交易日"
            source = "FinMind:TaiwanStockDelisting+TaiwanStockInfo"

        universe[stock_id] = {
            "stock_id": stock_id,
            "stock_name": info.get("stock_name") or delisting.get("stock_name") or "",
            "market_type": market_type,
            "industry_category": industry,
            "listing_date": listing_date,
            "delisting_date": delisting_date,
            "current_status": "delisted",
            "download_enabled": 1,
            "training_enabled": training_enabled,
            "inclusion_status": inclusion_status,
            "inclusion_reason": reason,
            "source": source,
        }

    _replace_model_universe(database, universe)
    return {
        "total": len(universe),
        "twse": sum(1 for item in universe.values() if item["market_type"] == "twse"),
        "tpex": sum(1 for item in universe.values() if item["market_type"] == "tpex"),
        "unknown": sum(
            1 for item in universe.values() if item["market_type"] not in VALID_MARKETS
        ),
        "active": sum(1 for item in universe.values() if item["current_status"] == "active"),
        "delisted": sum(
            1 for item in universe.values() if item["current_status"] == "delisted"
        ),
        "training_enabled": sum(
            int(item["training_enabled"]) for item in universe.values()
        ),
        "review_required": sum(
            1 for item in universe.values() if item["inclusion_status"] == "review_required"
        ),
        "excluded": sum(
            1 for item in universe.values() if item["inclusion_status"] == "excluded"
        ),
    }



def _datasets_cover_range(
    status_by_dataset: dict[str, dict[str, Any]],
    requested_start: str,
    requested_end: str,
) -> bool:
    for dataset in REQUIRED_STOCK_DATASETS:
        row = status_by_dataset.get(dataset)
        if not row:
            return False
        if row.get("status") != "complete":
            return False
        if str(row.get("requested_start") or "") > requested_start:
            return False
        if str(row.get("requested_end") or "") < requested_end:
            return False
    return True


def _day_before(value: str) -> str | None:
    try:
        return (date.fromisoformat(value) - timedelta(days=1)).isoformat()
    except ValueError:
        return None


def _replace_model_universe(
    database: ResearchDatabase, universe: dict[str, dict[str, Any]]
) -> None:
    with database.connect() as connection:
        connection.execute("DELETE FROM model_universe")
        connection.executemany(
            """
            INSERT INTO model_universe (
                stock_id, stock_name, market_type, industry_category,
                listing_date, delisting_date, current_status,
                download_enabled, training_enabled, inclusion_status,
                inclusion_reason, source, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                (
                    item["stock_id"],
                    item["stock_name"],
                    item["market_type"],
                    item["industry_category"],
                    item["listing_date"],
                    item["delisting_date"],
                    item["current_status"],
                    item["download_enabled"],
                    item["training_enabled"],
                    item["inclusion_status"],
                    item["inclusion_reason"],
                    item["source"],
                )
                for item in universe.values()
            ),
        )


def load_market_overrides(path: Path | str | None) -> dict[str, dict[str, str]]:
    if not path:
        return {}
    override_path = Path(path)
    if not override_path.exists():
        return {}
    with override_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        stock_id = str(row.get("stock_id") or "").strip()
        market_type = str(row.get("market_type") or "").strip().lower()
        if not stock_id:
            continue
        if market_type and market_type not in VALID_MARKETS:
            raise ValueError(f"{stock_id} 的 market_type 只能是 twse 或 tpex")
        result[stock_id] = {
            "market_type": market_type,
            "listing_date": str(row.get("listing_date") or "").strip(),
            "note": str(row.get("note") or "").strip(),
        }
    return result
