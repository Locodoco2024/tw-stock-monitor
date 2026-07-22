from __future__ import annotations

import pytest

from src.models import Quote
from src.providers.http import ProviderError
from src.providers.metadata import normalize_market, resolve_stock_metadata


def _quote(**overrides: object) -> Quote:
    values = {
        "symbol": "2330",
        "name": "台積電",
        "price": 100.0,
        "previous_close": 99.0,
        "open_price": 99.5,
        "high_price": 101.0,
        "low_price": 98.5,
        "change_percent": 1.0,
        "trade_volume": 1000.0,
        "as_of": "2026-07-21",
        "exchange": "TWSE",
        "market": "TSE",
    }
    values.update(overrides)
    return Quote(**values)  # type: ignore[arg-type]


def test_symbol_only_uses_fugle_metadata() -> None:
    metadata = resolve_stock_metadata(
        {"symbol": "2330"},
        _quote(),
        None,
        "TAIEX",
    )
    assert metadata.name == "台積電"
    assert metadata.market == "TWSE"
    assert metadata.benchmark == "TAIEX"


def test_tpex_symbol_uses_tpex_benchmark() -> None:
    metadata = resolve_stock_metadata(
        {"symbol": "5347"},
        _quote(symbol="5347", name="世界", exchange="TPEx", market="OTC"),
        None,
        "TAIEX",
    )
    assert metadata.market == "TPEx"
    assert metadata.benchmark == "TPEx"


def test_finmind_is_fallback_when_quote_is_unavailable() -> None:
    metadata = resolve_stock_metadata(
        {"symbol": "5347"},
        None,
        {"stock_id": "5347", "stock_name": "世界", "type": "tpex"},
        "TAIEX",
    )
    assert metadata.name == "世界"
    assert metadata.market == "TPEx"


def test_explicit_override_has_priority() -> None:
    metadata = resolve_stock_metadata(
        {"symbol": "2330", "name": "自訂名稱", "market": "TPEx", "benchmark": "CUSTOM"},
        _quote(),
        None,
        "TAIEX",
    )
    assert metadata.name == "自訂名稱"
    assert metadata.market == "TPEx"
    assert metadata.benchmark == "CUSTOM"


def test_unknown_market_requires_optional_override() -> None:
    with pytest.raises(ProviderError, match="無法自動辨識"):
        resolve_stock_metadata({"symbol": "9999"}, None, None, "TAIEX")


def test_market_aliases() -> None:
    assert normalize_market("TSE") == "TWSE"
    assert normalize_market("otc") == "TPEx"
