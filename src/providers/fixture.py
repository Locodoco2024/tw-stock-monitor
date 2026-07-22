from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.models import OfficialEvent, Quote, StockInputBundle
from src.providers.metadata import resolve_stock_metadata


def load_fixture_bundle(
    path: str | Path,
    symbol: str,
    stock_config: dict[str, Any] | None = None,
    default_market_symbol: str = "TAIEX",
) -> StockInputBundle:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    stock = payload.get("stocks", {}).get(symbol)
    if not stock:
        raise ValueError(f"離線資料沒有股票 {symbol}")
    quote_data = stock.get("quote")
    quote = Quote(**quote_data) if quote_data else None
    events = [OfficialEvent(**item) for item in stock.get("events", [])]
    metadata_payload = stock.get("metadata") or {}
    resolved = resolve_stock_metadata(
        stock_config or {"symbol": symbol},
        quote,
        metadata_payload,
        default_market_symbol,
    )
    return StockInputBundle(
        quote=quote,
        resolved_name=resolved.name,
        resolved_market=resolved.market,
        benchmark=resolved.benchmark,
        prices=stock.get("prices", []),
        market_prices=payload.get("market_prices", []),
        peer_prices={
            peer: payload.get("stocks", {}).get(peer, {}).get("prices", [])
            for peer in stock.get("peers", [])
        },
        monthly_revenue=stock.get("monthly_revenue", []),
        peer_monthly_revenue={
            peer: payload.get("stocks", {}).get(peer, {}).get("monthly_revenue", [])
            for peer in stock.get("peers", [])
        },
        financial_statements=stock.get("financial_statements", []),
        events=events,
        errors=[],
    )
