from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.models import Quote
from src.providers.http import ProviderError


@dataclass(slots=True, frozen=True)
class StockMetadata:
    symbol: str
    name: str
    market: str
    benchmark: str


_MARKET_ALIASES = {
    "TWSE": "TWSE",
    "TSE": "TWSE",
    "LISTED": "TWSE",
    "上市": "TWSE",
    "TPEX": "TPEx",
    "OTC": "TPEx",
    "TPEx": "TPEx",
    "上櫃": "TPEx",
}


def normalize_market(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return _MARKET_ALIASES.get(text, _MARKET_ALIASES.get(text.upper()))


def resolve_stock_metadata(
    stock: dict[str, Any],
    quote: Quote | None,
    finmind_info: dict[str, Any] | None,
    default_market_symbol: str,
) -> StockMetadata:
    symbol = str(stock["symbol"]).strip()
    explicit_name = str(stock.get("name") or "").strip()
    quote_name = quote.name.strip() if quote and quote.name else ""
    info_name = str((finmind_info or {}).get("stock_name") or "").strip()
    name = explicit_name or quote_name or info_name or symbol

    explicit_market = normalize_market(stock.get("market"))
    quote_market = normalize_market(quote.exchange if quote else None) or normalize_market(
        quote.market if quote else None
    )
    info_market = normalize_market((finmind_info or {}).get("type"))
    market = explicit_market or quote_market or info_market
    if market is None:
        raise ProviderError(
            f"無法自動辨識 {symbol} 的上市／上櫃市場；可在配置中選填 market: TWSE 或 TPEx"
        )

    benchmark = str(stock.get("benchmark") or "").strip()
    if not benchmark:
        benchmark = "TPEx" if market == "TPEx" else default_market_symbol

    return StockMetadata(
        symbol=symbol,
        name=name,
        market=market,
        benchmark=benchmark,
    )
