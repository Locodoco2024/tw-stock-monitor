from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.models import Quote


def load_fixture_quote(path: str | Path, symbol: str) -> Quote:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    stock = payload.get("stocks", {}).get(symbol)
    if not stock:
        raise ValueError(f"離線資料沒有股票 {symbol}")
    quote_data = stock.get("quote")
    if not quote_data:
        raise ValueError(f"離線資料沒有股票 {symbol} 的報價")
    return Quote(**quote_data)
