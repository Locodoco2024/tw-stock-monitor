from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import yaml

from research.institutional_model.database import ResearchDatabase


EXCLUDED_INDUSTRY_CATEGORIES = {
    "大盤",
    "Index",
    "所有證券",
    "ETF",
    "ETN",
    "TDR",
    "存託憑證",
    "特別股",
    "受益證券",
    "認購權證",
    "認售權證",
}


def load_validation_universe(path: Path | str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "stock_id",
        "stock_name",
        "market_hint",
        "selection_group",
        "selection_reason",
        "expected_cases",
    }
    if not rows:
        raise ValueError("驗證股票清單是空的")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"驗證股票清單缺少欄位: {sorted(missing)}")
    symbols = [row["stock_id"].strip() for row in rows]
    if len(symbols) != len(set(symbols)):
        raise ValueError("驗證股票清單含有重複股票代號")
    return rows


def sync_universe(database: ResearchDatabase, rows: list[dict[str, str]]) -> None:
    database.executemany(
        """
        INSERT INTO stock_universe (
            stock_id, stock_name, market_hint, selection_group,
            selection_reason, expected_cases, enabled
        ) VALUES (?, ?, ?, ?, ?, ?, 1)
        ON CONFLICT(stock_id) DO UPDATE SET
            stock_name=excluded.stock_name,
            market_hint=excluded.market_hint,
            selection_group=excluded.selection_group,
            selection_reason=excluded.selection_reason,
            expected_cases=excluded.expected_cases,
            enabled=1
        """,
        (
            (
                row["stock_id"].strip(),
                row["stock_name"].strip(),
                row["market_hint"].strip(),
                row["selection_group"].strip(),
                row["selection_reason"].strip(),
                row["expected_cases"].strip(),
            )
            for row in rows
        ),
    )


def configured_holding_symbols(config_dir: Path | str) -> set[str]:
    symbols: set[str] = set()
    for path in sorted(Path(config_dir).glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not (payload.get("user") or {}).get("enabled", True):
            continue
        for stock in payload.get("stocks") or []:
            symbol = str(stock.get("symbol") or "").strip()
            if symbol:
                symbols.add(symbol)
    return symbols


def compare_with_current_holdings(
    rows: list[dict[str, str]], config_dir: Path | str
) -> tuple[set[str], set[str]]:
    expected = {
        row["stock_id"].strip()
        for row in rows
        if row["selection_group"].strip() == "current_holding"
    }
    actual = configured_holding_symbols(config_dir)
    return actual - expected, expected - actual


def is_common_stock(row: dict[str, Any]) -> bool:
    stock_id = str(row.get("stock_id") or "")
    market_type = str(row.get("type") or "").lower()
    industry = str(row.get("industry_category") or "")
    excluded_by_text = any(
        keyword in industry
        for keyword in ("ETF", "ETN", "存託憑證", "特別股", "權證", "受益證券")
    )
    return (
        market_type in {"twse", "tpex"}
        and len(stock_id) == 4
        and stock_id.isdigit()
        and not stock_id.startswith("00")
        and industry not in EXCLUDED_INDUSTRY_CATEGORIES
        and not excluded_by_text
    )
