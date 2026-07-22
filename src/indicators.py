from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


PRICE_COLUMNS = {
    "date": ["date", "Date"],
    "open": ["open", "Open", "open_price"],
    "high": ["max", "high", "High", "high_price"],
    "low": ["min", "low", "Low", "low_price"],
    "close": ["close", "Close", "close_price", "price"],
    "volume": ["Trading_Volume", "volume", "Volume", "trade_volume"],
}


def _first_existing(row: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def price_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append(
            {
                key: _first_existing(row, candidates)
                for key, candidates in PRICE_COLUMNS.items()
            }
        )
    frame = pd.DataFrame(normalized)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date")


def calculate_indicators(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    frame = price_frame(rows)
    if frame.empty:
        return {}

    close = frame["close"]
    result: dict[str, float | None] = {
        "close": _last(close),
        "return_5d": period_return(close, 5),
        "return_20d": period_return(close, 20),
        "return_60d": period_return(close, 60),
        "ema20": _last(close.ewm(span=20, adjust=False).mean()) if len(close) >= 20 else None,
        "ema60": _last(close.ewm(span=60, adjust=False).mean()) if len(close) >= 60 else None,
        "ema120": _last(close.ewm(span=120, adjust=False).mean()) if len(close) >= 120 else None,
        "ma20": _last(close.rolling(20).mean()) if len(close) >= 20 else None,
        "ma60": _last(close.rolling(60).mean()) if len(close) >= 60 else None,
        "rsi14": rsi(close, 14),
        "atr14": atr(frame, 14),
        "volume_ratio20": volume_ratio(frame, 20),
        "ema20_slope": slope(close.ewm(span=20, adjust=False).mean(), 5) if len(close) >= 25 else None,
        "ema60_slope": slope(close.ewm(span=60, adjust=False).mean(), 10) if len(close) >= 70 else None,
    }
    atr_value = result.get("atr14")
    ema20 = result.get("ema20")
    current_close = result.get("close")
    result["distance_from_ema20_atr"] = (
        (float(current_close) - float(ema20)) / float(atr_value)
        if current_close is not None and ema20 is not None and atr_value not in (None, 0)
        else None
    )
    return result


def period_return(series: pd.Series, periods: int) -> float | None:
    clean = series.dropna()
    if len(clean) <= periods:
        return None
    start = float(clean.iloc[-periods - 1])
    end = float(clean.iloc[-1])
    if start == 0:
        return None
    return (end / start - 1.0) * 100.0


def rsi(close: pd.Series, period: int = 14) -> float | None:
    if len(close.dropna()) <= period:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    value = 100 - (100 / (1 + rs))
    last = value.iloc[-1]
    if pd.isna(last):
        return 100.0 if gain.iloc[-1] > 0 else 50.0
    return float(last)


def atr(frame: pd.DataFrame, period: int = 14) -> float | None:
    if len(frame) <= period or frame[["high", "low", "close"]].isna().any(axis=None):
        return None
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return _last(true_range.ewm(alpha=1 / period, adjust=False).mean())


def volume_ratio(frame: pd.DataFrame, period: int = 20) -> float | None:
    if len(frame) < period + 1 or frame["volume"].dropna().empty:
        return None
    previous_average = frame["volume"].iloc[-period - 1 : -1].mean()
    current = frame["volume"].iloc[-1]
    if pd.isna(previous_average) or previous_average == 0 or pd.isna(current):
        return None
    return float(current / previous_average)


def slope(series: pd.Series, periods: int) -> float | None:
    clean = series.dropna()
    if len(clean) <= periods:
        return None
    start = float(clean.iloc[-periods - 1])
    end = float(clean.iloc[-1])
    if start == 0:
        return None
    return (end / start - 1.0) * 100.0


def median(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    return float(np.median(clean)) if clean else None


def _last(series: pd.Series) -> float | None:
    clean = series.dropna()
    return float(clean.iloc[-1]) if not clean.empty else None
