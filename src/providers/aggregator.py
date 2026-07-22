from __future__ import annotations

from typing import Any, Callable, TypeVar

from src.models import StockInputBundle
from src.providers.finmind import FinMindProvider
from src.providers.fugle import FugleProvider
from src.providers.metadata import resolve_stock_metadata
from src.providers.official_events import OfficialEventProvider


T = TypeVar("T")


class DataAggregator:
    def __init__(
        self,
        fugle: FugleProvider,
        finmind: FinMindProvider,
        official_events: OfficialEventProvider,
    ) -> None:
        self.fugle = fugle
        self.finmind = finmind
        self.official_events = official_events

    def collect(self, stock: dict[str, Any], default_market_symbol: str) -> StockInputBundle:
        symbol = str(stock["symbol"])
        peers = [str(peer) for peer in stock.get("peers", [])]
        errors: list[str] = []

        quote = self._safe(lambda: self.fugle.quote(symbol), errors, "Fugle 即時報價")
        needs_fallback_info = not stock.get("name") or not stock.get("market")
        finmind_info = None
        if needs_fallback_info and not self._quote_has_metadata(quote):
            finmind_info = self._safe(
                lambda: self.finmind.stock_info(symbol),
                errors,
                "股票基本資料",
            )
        metadata = resolve_stock_metadata(stock, quote, finmind_info, default_market_symbol)

        prices = self._safe(lambda: self.finmind.stock_prices(symbol), errors, "個股歷史行情") or []
        market_prices = (
            self._safe(
                lambda: self.finmind.market_prices(metadata.benchmark),
                errors,
                f"大盤 {metadata.benchmark} 歷史行情",
            )
            or []
        )
        peer_prices: dict[str, list[dict[str, Any]]] = {}
        peer_monthly_revenue: dict[str, list[dict[str, Any]]] = {}
        for peer in peers:
            peer_prices[peer] = (
                self._safe(
                    lambda peer_symbol=peer: self.finmind.stock_prices(peer_symbol),
                    errors,
                    f"同業 {peer} 歷史行情",
                )
                or []
            )
            peer_monthly_revenue[peer] = (
                self._safe(
                    lambda peer_symbol=peer: self.finmind.monthly_revenue(peer_symbol),
                    errors,
                    f"同業 {peer} 月營收",
                )
                or []
            )
        monthly_revenue = (
            self._safe(lambda: self.finmind.monthly_revenue(symbol), errors, "個股月營收") or []
        )
        financial_statements = (
            self._safe(
                lambda: self.finmind.financial_statements(symbol),
                errors,
                "個股財務報表",
            )
            or []
        )
        events = (
            self._safe(
                lambda: self.official_events.events(symbol, metadata.market),
                errors,
                "官方重大訊息",
            )
            or []
        )
        return StockInputBundle(
            quote=quote,
            resolved_name=metadata.name,
            resolved_market=metadata.market,
            benchmark=metadata.benchmark,
            prices=prices,
            market_prices=market_prices,
            peer_prices=peer_prices,
            monthly_revenue=monthly_revenue,
            peer_monthly_revenue=peer_monthly_revenue,
            financial_statements=financial_statements,
            events=events,
            errors=errors,
        )

    @staticmethod
    def _quote_has_metadata(quote: Any) -> bool:
        return bool(
            quote
            and quote.name
            and (quote.exchange or quote.market)
        )

    @staticmethod
    def _safe(factory: Callable[[], T], errors: list[str], label: str) -> T | None:
        try:
            return factory()
        except Exception as exc:  # Individual providers must not stop the whole analysis.
            errors.append(f"{label}: {exc}")
            return None
