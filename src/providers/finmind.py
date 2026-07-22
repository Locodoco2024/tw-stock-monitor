from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from src.providers.http import HttpClient, ProviderError


class FinMindProvider:
    BASE_URL = "https://api.finmindtrade.com/api/v4/data"

    def __init__(self, token: str | None = None, http: HttpClient | None = None) -> None:
        self.token = token
        self.http = http or HttpClient(timeout=30)
        self._cache: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
        self._stock_info_cache: dict[str, dict[str, Any]] | None = None

    def fetch(
        self,
        dataset: str,
        data_id: str,
        start_date: str,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        key = (dataset, data_id, start_date, end_date or "")
        if key in self._cache:
            return self._cache[key]
        params: dict[str, Any] = {
            "dataset": dataset,
            "data_id": data_id,
            "start_date": start_date,
        }
        if end_date:
            params["end_date"] = end_date
        if self.token:
            params["token"] = self.token
        payload = self.http.get_json(self.BASE_URL, params=params)
        if not isinstance(payload, dict):
            raise ProviderError(f"FinMind {dataset}/{data_id} 回傳格式異常")
        if payload.get("status") not in (None, 200):
            raise ProviderError(
                f"FinMind {dataset}/{data_id} 查詢失敗: {payload.get('msg') or payload}"
            )
        data = payload.get("data") or []
        if not isinstance(data, list):
            raise ProviderError(f"FinMind {dataset}/{data_id} data 不是陣列")
        self._cache[key] = data
        return data


    def stock_info(self, symbol: str) -> dict[str, Any] | None:
        if self._stock_info_cache is None:
            params: dict[str, Any] = {"dataset": "TaiwanStockInfo"}
            if self.token:
                params["token"] = self.token
            payload = self.http.get_json(self.BASE_URL, params=params)
            if not isinstance(payload, dict):
                raise ProviderError("FinMind TaiwanStockInfo 回傳格式異常")
            if payload.get("status") not in (None, 200):
                raise ProviderError(
                    f"FinMind TaiwanStockInfo 查詢失敗: {payload.get('msg') or payload}"
                )
            rows = payload.get("data") or []
            if not isinstance(rows, list):
                raise ProviderError("FinMind TaiwanStockInfo data 不是陣列")
            self._stock_info_cache = {
                str(row.get("stock_id")): row
                for row in rows
                if isinstance(row, dict) and row.get("stock_id")
            }
        return self._stock_info_cache.get(str(symbol))

    def stock_prices(self, symbol: str, days: int = 430) -> list[dict[str, Any]]:
        start = (date.today() - timedelta(days=days)).isoformat()
        return self.fetch("TaiwanStockPrice", symbol, start)

    def market_prices(self, index_id: str = "TAIEX", days: int = 430) -> list[dict[str, Any]]:
        start = (date.today() - timedelta(days=days)).isoformat()
        return self.fetch("TaiwanStockTotalReturnIndex", index_id, start)

    def monthly_revenue(self, symbol: str, months: int = 30) -> list[dict[str, Any]]:
        start = (date.today() - timedelta(days=months * 32)).replace(day=1).isoformat()
        return self.fetch("TaiwanStockMonthRevenue", symbol, start)

    def financial_statements(self, symbol: str, years: int = 4) -> list[dict[str, Any]]:
        start = date(date.today().year - years, 1, 1).isoformat()
        return self.fetch("TaiwanStockFinancialStatements", symbol, start)
