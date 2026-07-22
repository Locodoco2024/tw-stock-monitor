from __future__ import annotations

from datetime import date, timedelta

from src.indicators import calculate_indicators


def test_calculate_indicators_for_uptrend() -> None:
    rows = []
    start = date(2025, 1, 1)
    for index in range(160):
        close = 100 + index
        rows.append(
            {
                "date": (start + timedelta(days=index)).isoformat(),
                "open": close - 1,
                "max": close + 2,
                "min": close - 2,
                "close": close,
                "Trading_Volume": 1000 + index,
            }
        )
    result = calculate_indicators(rows)
    assert result["return_20d"] > 0
    assert result["ema20"] > result["ema60"] > result["ema120"]
    assert result["rsi14"] >= 90
    assert result["atr14"] is not None
