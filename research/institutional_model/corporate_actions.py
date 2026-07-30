from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable

from research.institutional_model.database import ResearchDatabase


@dataclass(frozen=True)
class CorporateAction:
    date: str
    action_type: str
    reference_price: float | None
    description: str
    source_dataset: str


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _first_positive(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _number(row.get(key))
        if value is not None:
            return value
    return None


def normalize_dividend_rows(rows: Iterable[dict[str, Any]]) -> list[tuple[Any, ...]]:
    normalized: list[tuple[Any, ...]] = []
    for row in rows:
        stock_id = str(row.get("stock_id") or "")
        event_date = str(row.get("date") or "")
        if not stock_id or not event_date:
            continue
        normalized.append(
            (
                stock_id,
                event_date,
                "dividend",
                _number(row.get("before_price")),
                _first_positive(row, "reference_price", "open_price", "after_price"),
                str(row.get("stock_or_cache_dividend") or "除權息"),
                "TaiwanStockDividendResult",
                json.dumps(row, ensure_ascii=False, sort_keys=True),
            )
        )
    return normalized


def normalize_capital_reduction_rows(rows: Iterable[dict[str, Any]]) -> list[tuple[Any, ...]]:
    normalized: list[tuple[Any, ...]] = []
    for row in rows:
        stock_id = str(row.get("stock_id") or "")
        event_date = str(row.get("date") or "")
        if not stock_id or not event_date:
            continue
        normalized.append(
            (
                stock_id,
                event_date,
                "capital_reduction",
                _number(row.get("ClosingPriceonTheLastTradingDay")),
                _first_positive(
                    row,
                    "OpeningReferencePrice",
                    "PostReductionReferencePrice",
                    "ExrightReferencePrice",
                ),
                str(row.get("ReasonforCapitalReduction") or "減資恢復交易"),
                "TaiwanStockCapitalReductionReferencePrice",
                json.dumps(row, ensure_ascii=False, sort_keys=True),
            )
        )
    return normalized


def normalize_split_rows(rows: Iterable[dict[str, Any]]) -> list[tuple[Any, ...]]:
    normalized: list[tuple[Any, ...]] = []
    for row in rows:
        stock_id = str(row.get("stock_id") or "")
        event_date = str(row.get("date") or "")
        if not stock_id or not event_date:
            continue
        normalized.append(
            (
                stock_id,
                event_date,
                "split",
                _number(row.get("before_price")),
                _first_positive(row, "open_price", "after_price"),
                str(row.get("type") or "股票分割／反分割"),
                "TaiwanStockSplitPrice",
                json.dumps(row, ensure_ascii=False, sort_keys=True),
            )
        )
    return normalized


def normalize_par_value_rows(rows: Iterable[dict[str, Any]]) -> list[tuple[Any, ...]]:
    normalized: list[tuple[Any, ...]] = []
    for row in rows:
        stock_id = str(row.get("stock_id") or "")
        event_date = str(row.get("date") or "")
        if not stock_id or not event_date:
            continue
        normalized.append(
            (
                stock_id,
                event_date,
                "par_value_change",
                _number(row.get("before_close")),
                _first_positive(row, "after_ref_open", "after_ref_close"),
                "變更面額恢復交易",
                "TaiwanStockParValueChange",
                json.dumps(row, ensure_ascii=False, sort_keys=True),
            )
        )
    return normalized


def upsert_corporate_actions(
    database: ResearchDatabase, rows: Iterable[tuple[Any, ...]]
) -> int:
    return database.executemany(
        """
        INSERT INTO corporate_actions (
            stock_id, date, action_type, before_price, reference_price,
            description, source_dataset, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(stock_id, date, action_type, source_dataset) DO UPDATE SET
            before_price=excluded.before_price,
            reference_price=excluded.reference_price,
            description=excluded.description,
            raw_json=excluded.raw_json
        """,
        rows,
    )


def replace_corporate_actions(
    *,
    database: ResearchDatabase,
    rows: Iterable[tuple[Any, ...]],
    source_dataset: str,
    start_date: str,
    end_date: str,
    stock_id: str | None = None,
) -> int:
    """Replace one downloaded corporate-action slice so corrected API rows remove stale DB rows."""
    materialized = list(rows)
    with database.connect() as connection:
        if stock_id:
            connection.execute(
                """
                DELETE FROM corporate_actions
                WHERE stock_id=? AND source_dataset=? AND date BETWEEN ? AND ?
                """,
                (stock_id, source_dataset, start_date, end_date),
            )
        else:
            connection.execute(
                """
                DELETE FROM corporate_actions
                WHERE source_dataset=? AND date BETWEEN ? AND ?
                """,
                (source_dataset, start_date, end_date),
            )
        if materialized:
            connection.executemany(
                """
                INSERT INTO corporate_actions (
                    stock_id, date, action_type, before_price, reference_price,
                    description, source_dataset, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(stock_id, date, action_type, source_dataset) DO UPDATE SET
                    before_price=excluded.before_price,
                    reference_price=excluded.reference_price,
                    description=excluded.description,
                    raw_json=excluded.raw_json
                """,
                materialized,
            )
    return len(materialized)


def cleanup_duplicate_corporate_actions(
    database: ResearchDatabase,
    symbols: list[str] | None = None,
    max_gap_days: int = 45,
) -> int:
    """Delete stale duplicate action dates when one nearby row is the real trading date."""
    params: list[Any] = []
    symbol_filter = ""
    if symbols:
        placeholders = ",".join("?" for _ in symbols)
        symbol_filter = f"WHERE a.stock_id IN ({placeholders})"
        params.extend(symbols)

    rows = database.query(
        f"""
        SELECT
            a.stock_id,
            a.date,
            a.action_type,
            a.before_price,
            a.reference_price,
            a.description,
            a.source_dataset,
            CASE WHEN c.date IS NOT NULL THEN 1 ELSE 0 END AS is_market_date,
            CASE
                WHEN COALESCE(p.open, 0) > 0
                 AND COALESCE(p.close, 0) > 0
                 AND COALESCE(p.trading_volume, 0) > 0
                THEN 1 ELSE 0
            END AS has_valid_price
        FROM corporate_actions a
        LEFT JOIN market_calendar c ON c.date=a.date
        LEFT JOIN stock_prices p ON p.stock_id=a.stock_id AND p.date=a.date
        {symbol_filter}
        ORDER BY a.stock_id, a.action_type, a.source_dataset, a.date
        """,
        tuple(params),
    )

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        item = dict(row)
        signature = (
            item["stock_id"],
            item["action_type"],
            item["source_dataset"],
            _rounded(item["before_price"]),
            _rounded(item["reference_price"]),
            item["description"] or "",
        )
        groups.setdefault(signature, []).append(item)

    to_delete: list[tuple[Any, ...]] = []
    for items in groups.values():
        if len(items) < 2:
            continue
        valid = [
            item
            for item in items
            if item["is_market_date"] == 1 and item["has_valid_price"] == 1
        ]
        if len(valid) != 1:
            continue
        kept = valid[0]
        kept_date = _iso_date(kept["date"])
        if kept_date is None:
            continue
        for item in items:
            if item is kept:
                continue
            item_date = _iso_date(item["date"])
            if item_date is None:
                continue
            if abs((item_date - kept_date).days) > max_gap_days:
                continue
            if item["is_market_date"] == 1 and item["has_valid_price"] == 1:
                continue
            to_delete.append(
                (
                    item["stock_id"],
                    item["date"],
                    item["action_type"],
                    item["source_dataset"],
                )
            )

    if not to_delete:
        return 0
    return database.executemany(
        """
        DELETE FROM corporate_actions
        WHERE stock_id=? AND date=? AND action_type=? AND source_dataset=?
        """,
        to_delete,
    )


def load_action_map(database: ResearchDatabase, stock_id: str) -> dict[str, list[CorporateAction]]:
    rows = database.query(
        """
        SELECT date, action_type, reference_price, description, source_dataset
        FROM corporate_actions
        WHERE stock_id=?
        ORDER BY date, action_type
        """,
        (stock_id,),
    )
    result: dict[str, list[CorporateAction]] = {}
    for row in rows:
        result.setdefault(row["date"], []).append(
            CorporateAction(
                date=row["date"],
                action_type=row["action_type"],
                reference_price=row["reference_price"],
                description=row["description"] or "",
                source_dataset=row["source_dataset"],
            )
        )
    return result


def resolve_reference_price(actions: list[CorporateAction]) -> tuple[float | None, str | None]:
    """Return one usable official reference price or an explicit validation error."""
    if not actions:
        return None, None
    valid = [action for action in actions if action.reference_price and action.reference_price > 0]
    if not valid:
        return None, "公司行動存在，但沒有有效參考價"

    priority = {
        "par_value_change": 4,
        "split": 3,
        "capital_reduction": 2,
        "dividend": 1,
    }
    chosen = max(valid, key=lambda item: priority.get(item.action_type, 0))
    for action in valid:
        difference = abs(action.reference_price - chosen.reference_price) / chosen.reference_price
        if difference > 0.01:
            names = ", ".join(
                f"{item.action_type}:{item.reference_price}" for item in valid
            )
            return None, f"同日公司行動參考價不一致（{names}）"
    return chosen.reference_price, None


def _rounded(value: Any) -> float | None:
    try:
        return round(float(value), 8)
    except (TypeError, ValueError):
        return None


def _iso_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
