from __future__ import annotations

from typing import Any

from src.models import Quote
from src.providers.http import HttpClient, ProviderError


class FugleProvider:
    BASE_URL = "https://api.fugle.tw/marketdata/v1.0/stock"

    def __init__(self, api_key: str | None, http: HttpClient | None = None) -> None:
        self.api_key = api_key
        self.http = http or HttpClient()

    def quote(self, symbol: str) -> Quote:
        if not self.api_key:
            raise ProviderError("缺少 FUGLE_API_KEY，無法取得盤中即時報價")
        data: dict[str, Any] = self.http.get_json(
            f"{self.BASE_URL}/intraday/quote/{symbol}",
            headers={"X-API-KEY": self.api_key},
        )
        price = data.get("lastPrice") or data.get("closePrice") or data.get("referencePrice")
        if price is None:
            raise ProviderError(f"Fugle 未回傳 {symbol} 的有效價格")
        total = data.get("total") or {}
        return Quote(
            symbol=str(data.get("symbol") or symbol),
            name=str(data.get("name") or symbol),
            price=float(price),
            previous_close=_number(data.get("previousClose")),
            open_price=_number(data.get("openPrice")),
            high_price=_number(data.get("highPrice")),
            low_price=_number(data.get("lowPrice")),
            change_percent=_number(data.get("changePercent")),
            trade_volume=_number(total.get("tradeVolume")),
            as_of=str(data.get("lastUpdated") or data.get("date") or ""),
            exchange=str(data.get("exchange") or "") or None,
            market=str(data.get("market") or "") or None,
            raw=data,
        )


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
