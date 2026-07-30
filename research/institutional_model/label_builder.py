from __future__ import annotations

import bisect
import json
from typing import Any

from research.institutional_model.adjusted_returns import calculate_holding_return
from research.institutional_model.corporate_actions import load_action_map
from research.institutional_model.database import ResearchDatabase


def build_labels(
    *,
    database: ResearchDatabase,
    symbols: list[str],
    horizons: tuple[int, ...] = (5, 10, 20),
    primary_horizon: int = 10,
    threshold: float = 0.05,
) -> None:
    market_dates = [
        str(row["date"])
        for row in database.query("SELECT date FROM market_calendar ORDER BY date")
    ]
    if not market_dates:
        raise RuntimeError(
            "市場交易日曆是空的，請先重新執行 download 或完整 phase1。"
        )

    for position, stock_id in enumerate(symbols, start=1):
        print(f"[{position}/{len(symbols)}] 產生標籤 {stock_id}")
        _build_stock_labels(
            database=database,
            stock_id=stock_id,
            market_dates=market_dates,
            horizons=horizons,
            primary_horizon=primary_horizon,
            threshold=threshold,
        )


def _build_stock_labels(
    *,
    database: ResearchDatabase,
    stock_id: str,
    market_dates: list[str],
    horizons: tuple[int, ...],
    primary_horizon: int,
    threshold: float,
) -> None:
    database.execute("DELETE FROM label_results WHERE stock_id=?", (stock_id,))

    delisting_date = database.scalar(
        "SELECT MAX(date) FROM delistings WHERE stock_id=?", (stock_id,)
    )
    price_rows = [
        dict(row)
        for row in database.query(
            """
            SELECT date, open, close, high, low, trading_volume, trading_money
            FROM stock_prices
            WHERE stock_id=?
              AND (? IS NULL OR date < ?)
            ORDER BY date
            """,
            (stock_id, delisting_date, delisting_date),
        )
    ]
    price_by_date = {str(row["date"]): row for row in price_rows}
    signal_dates = [
        str(row["date"])
        for row in database.query(
            """
            SELECT f.date
            FROM institutional_flows f
            JOIN market_calendar c ON c.date=f.date
            JOIN stock_prices p ON p.stock_id=f.stock_id AND p.date=f.date
            WHERE f.stock_id=?
              AND COALESCE(p.open, 0) > 0
              AND COALESCE(p.close, 0) > 0
              AND COALESCE(p.trading_volume, 0) > 0
            ORDER BY f.date
            """,
            (stock_id,),
        )
        if not delisting_date or str(row["date"]) < str(delisting_date)
    ]
    action_map = load_action_map(database, stock_id)
    output: list[tuple[Any, ...]] = []

    for signal_date in signal_dates:
        entry_calendar_index = bisect.bisect_right(market_dates, signal_date)
        for horizon in horizons:
            target_calendar_index = entry_calendar_index + horizon - 1
            if (
                entry_calendar_index >= len(market_dates)
                or target_calendar_index >= len(market_dates)
            ):
                output.append(
                    _result_row(
                        stock_id=stock_id,
                        signal_date=signal_date,
                        horizon=horizon,
                        status="insufficient_future_data",
                        error="市場交易日曆的未來交易日不足",
                    )
                )
                continue

            entry_date = market_dates[entry_calendar_index]
            target_date = market_dates[target_calendar_index]
            if delisting_date and entry_date >= str(delisting_date):
                output.append(
                    _result_row(
                        stock_id=stock_id,
                        signal_date=signal_date,
                        horizon=horizon,
                        entry_date=entry_date,
                        target_date=target_date,
                        status="delisted_before_entry",
                        error=f"股票已於 {delisting_date} 終止上市櫃",
                    )
                )
                continue
            if delisting_date and target_date >= str(delisting_date):
                output.append(
                    _result_row(
                        stock_id=stock_id,
                        signal_date=signal_date,
                        horizon=horizon,
                        entry_date=entry_date,
                        target_date=target_date,
                        status="delisted_before_target",
                        error=f"持有期間跨越 {delisting_date} 終止上市櫃日",
                    )
                )
                continue

            entry_row = price_by_date.get(entry_date)
            if not _valid_entry_row(entry_row):
                output.append(
                    _result_row(
                        stock_id=stock_id,
                        signal_date=signal_date,
                        horizon=horizon,
                        entry_date=entry_date,
                        target_date=target_date,
                        status="unavailable_entry_price",
                        error="T+1 市場交易日沒有有效開盤與收盤資料",
                    )
                )
                continue

            target_row = price_by_date.get(target_date)
            if not _valid_close_row(target_row):
                output.append(
                    _result_row(
                        stock_id=stock_id,
                        signal_date=signal_date,
                        horizon=horizon,
                        entry_date=entry_date,
                        target_date=target_date,
                        entry_open=float(entry_row["open"]),
                        status="unavailable_target_price",
                        error=f"第 {horizon} 個市場交易日沒有有效收盤資料",
                    )
                )
                continue

            calendar_path = market_dates[
                entry_calendar_index : target_calendar_index + 1
            ]
            path = [
                price_by_date[date]
                for date in calendar_path
                if _valid_close_row(price_by_date.get(date))
            ]
            missing_action_dates = [
                date
                for date in action_map
                if entry_date < date <= target_date
                and not _valid_close_row(price_by_date.get(date))
            ]
            if missing_action_dates:
                output.append(
                    _result_row(
                        stock_id=stock_id,
                        signal_date=signal_date,
                        horizon=horizon,
                        entry_date=entry_date,
                        target_date=target_date,
                        entry_open=float(entry_row["open"]),
                        target_close=float(target_row["close"]),
                        status="invalid_data",
                        error=(
                            "公司行動日缺少有效價格："
                            + ",".join(sorted(missing_action_dates))
                        ),
                    )
                )
                continue

            try:
                result = calculate_holding_return(path, action_map)
                label = None
                if horizon == primary_horizon:
                    if result.adjusted_return >= threshold:
                        label = "UP"
                    elif result.adjusted_return <= -threshold:
                        label = "DOWN"
                    else:
                        label = "FLAT"
                output.append(
                    _result_row(
                        stock_id=stock_id,
                        signal_date=signal_date,
                        horizon=horizon,
                        entry_date=entry_date,
                        target_date=target_date,
                        entry_open=float(entry_row["open"]),
                        target_close=float(target_row["close"]),
                        raw_return=result.raw_return,
                        adjusted_return=result.adjusted_return,
                        max_adjusted_return=result.max_adjusted_return,
                        min_adjusted_return=result.min_adjusted_return,
                        label=label,
                        action_types=json.dumps(
                            result.action_types, ensure_ascii=False
                        ),
                        entry_day_action_ignored=int(
                            result.entry_day_action_ignored
                        ),
                        status="ok",
                    )
                )
            except ValueError as exc:
                output.append(
                    _result_row(
                        stock_id=stock_id,
                        signal_date=signal_date,
                        horizon=horizon,
                        entry_date=entry_date,
                        target_date=target_date,
                        entry_open=entry_row.get("open"),
                        target_close=target_row.get("close"),
                        status="invalid_data",
                        error=str(exc),
                    )
                )

    database.executemany(
        """
        INSERT INTO label_results (
            stock_id, signal_date, horizon, entry_date, target_date,
            entry_open, target_close, raw_return, adjusted_return,
            max_adjusted_return, min_adjusted_return, label, action_types,
            entry_day_action_ignored, status, error, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(stock_id, signal_date, horizon) DO UPDATE SET
            entry_date=excluded.entry_date,
            target_date=excluded.target_date,
            entry_open=excluded.entry_open,
            target_close=excluded.target_close,
            raw_return=excluded.raw_return,
            adjusted_return=excluded.adjusted_return,
            max_adjusted_return=excluded.max_adjusted_return,
            min_adjusted_return=excluded.min_adjusted_return,
            label=excluded.label,
            action_types=excluded.action_types,
            entry_day_action_ignored=excluded.entry_day_action_ignored,
            status=excluded.status,
            error=excluded.error,
            updated_at=CURRENT_TIMESTAMP
        """,
        output,
    )


def _valid_entry_row(row: dict[str, Any] | None) -> bool:
    return bool(
        row
        and _positive(row.get("open"))
        and _positive(row.get("close"))
        and _positive(row.get("trading_volume"))
    )


def _valid_close_row(row: dict[str, Any] | None) -> bool:
    return bool(row and _positive(row.get("close")))


def _positive(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _result_row(
    *,
    stock_id: str,
    signal_date: str,
    horizon: int,
    entry_date: str | None = None,
    target_date: str | None = None,
    entry_open: Any = None,
    target_close: Any = None,
    raw_return: Any = None,
    adjusted_return: Any = None,
    max_adjusted_return: Any = None,
    min_adjusted_return: Any = None,
    label: str | None = None,
    action_types: str = "[]",
    entry_day_action_ignored: int = 0,
    status: str,
    error: str | None = None,
) -> tuple[Any, ...]:
    return (
        stock_id,
        signal_date,
        horizon,
        entry_date,
        target_date,
        entry_open,
        target_close,
        raw_return,
        adjusted_return,
        max_adjusted_return,
        min_adjusted_return,
        label,
        action_types,
        entry_day_action_ignored,
        status,
        error,
    )
