from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

from research.institutional_model.corporate_actions import (
    CorporateAction,
    resolve_reference_price,
)


@dataclass(frozen=True)
class ReturnResult:
    raw_return: float
    adjusted_return: float
    max_adjusted_return: float
    min_adjusted_return: float
    action_types: tuple[str, ...]
    entry_day_action_ignored: bool


def calculate_holding_return(
    price_rows: list[dict[str, Any]],
    action_map: dict[str, list[CorporateAction]],
) -> ReturnResult:
    """Calculate return from first row's open to last row's close.

    The first row represents the T+1 entry day. A corporate action on the entry
    day is intentionally ignored because the position is opened after the
    action-adjusted opening auction.
    """
    if not price_rows:
        raise ValueError("價格路徑不可為空")
    entry_open = _positive(price_rows[0].get("open"), "進場開盤價")
    entry_close = _positive(price_rows[0].get("close"), "進場日收盤價")

    factors = [entry_close / entry_open]
    cumulative = [factors[0] - 1]
    seen_actions: list[str] = []
    entry_actions = action_map.get(str(price_rows[0]["date"]), [])
    if entry_actions:
        seen_actions.extend(f"entry_ignored:{item.action_type}" for item in entry_actions)

    previous_close = entry_close
    for row in price_rows[1:]:
        current_close = _positive(row.get("close"), f"{row.get('date')} 收盤價")
        actions = action_map.get(str(row["date"]), [])
        if actions:
            reference_price, error = resolve_reference_price(actions)
            if error:
                raise ValueError(f"{row['date']}：{error}")
            denominator = _positive(reference_price, f"{row['date']} 公司行動參考價")
            seen_actions.extend(item.action_type for item in actions)
        else:
            denominator = previous_close
        factors.append(current_close / denominator)
        cumulative.append(prod(factors) - 1)
        previous_close = current_close

    target_close = _positive(price_rows[-1].get("close"), "目標日收盤價")
    return ReturnResult(
        raw_return=target_close / entry_open - 1,
        adjusted_return=prod(factors) - 1,
        max_adjusted_return=max(cumulative),
        min_adjusted_return=min(cumulative),
        action_types=tuple(dict.fromkeys(seen_actions)),
        entry_day_action_ignored=bool(entry_actions),
    )


def _positive(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}不是有效數字: {value}") from exc
    if number <= 0:
        raise ValueError(f"{label}必須大於 0: {number}")
    return number
