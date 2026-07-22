from __future__ import annotations

from typing import Any

from src.indicators import calculate_indicators, median
from src.models import ModuleResult, RuleResult
from src.scoring.common import cap_module, number


def score_industry_peers(
    peer_prices: dict[str, list[dict[str, Any]]],
    peer_monthly_revenue: dict[str, list[dict[str, Any]]],
    weight: float,
) -> ModuleResult:
    indicators = {
        symbol: calculate_indicators(rows) for symbol, rows in peer_prices.items() if rows
    }
    if not indicators:
        return cap_module(
            "industry_peers", weight, 0.0, [], ["未設定同業或同業行情資料不足"]
        )

    rules: list[RuleResult] = []
    return_20d = median([item.get("return_20d") for item in indicators.values()])
    if return_20d is not None:
        if return_20d >= 8:
            score = 6
        elif return_20d >= 2:
            score = 3
        elif return_20d <= -8:
            score = -6
        elif return_20d <= -2:
            score = -3
        else:
            score = 0
        if score:
            rules.append(
                RuleResult(
                    rule_id="industry.peer_median_return_20d",
                    module="industry_peers",
                    score=score,
                    message=f"同業近二十日報酬中位數為 {return_20d:+.1f}%",
                    values={"peer_return_20d_median_pct": return_20d},
                )
            )

    above_ma20 = _breadth(indicators, "ma20")
    above_ma60 = _breadth(indicators, "ma60")
    if above_ma20 is not None:
        score = 4 if above_ma20 >= 0.7 else -4 if above_ma20 <= 0.3 else 0
        if score:
            rules.append(
                RuleResult(
                    rule_id="industry.breadth_ma20",
                    module="industry_peers",
                    score=score,
                    message=f"同業有 {above_ma20 * 100:.0f}% 站上 MA20",
                    values={"breadth_ma20_pct": above_ma20 * 100},
                )
            )
    if above_ma60 is not None:
        score = 4 if above_ma60 >= 0.7 else -4 if above_ma60 <= 0.3 else 0
        if score:
            rules.append(
                RuleResult(
                    rule_id="industry.breadth_ma60",
                    module="industry_peers",
                    score=score,
                    message=f"同業有 {above_ma60 * 100:.0f}% 站上 MA60",
                    values={"breadth_ma60_pct": above_ma60 * 100},
                )
            )

    yoy_values = [
        _latest_revenue_yoy(rows) for rows in peer_monthly_revenue.values() if rows
    ]
    revenue_yoy = median(yoy_values)
    if revenue_yoy is not None:
        if revenue_yoy >= 15:
            score = 5
        elif revenue_yoy >= 5:
            score = 3
        elif revenue_yoy <= -15:
            score = -5
        elif revenue_yoy <= -5:
            score = -3
        else:
            score = 0
        if score:
            rules.append(
                RuleResult(
                    rule_id="industry.revenue_yoy",
                    module="industry_peers",
                    score=score,
                    message=f"同業最新月營收年增率中位數為 {revenue_yoy:+.1f}%",
                    values={"peer_revenue_yoy_median_pct": revenue_yoy},
                )
            )

    return cap_module("industry_peers", weight, weight, rules)


def _breadth(indicators: dict[str, dict[str, Any]], average_key: str) -> float | None:
    values: list[bool] = []
    for item in indicators.values():
        close = item.get("close")
        average = item.get(average_key)
        if close is not None and average is not None:
            values.append(close > average)
    return sum(values) / len(values) if values else None


def _latest_revenue_yoy(rows: list[dict[str, Any]]) -> float | None:
    sorted_rows = sorted(rows, key=lambda row: str(row.get("date") or row.get("revenue_month") or ""))
    for row in reversed(sorted_rows):
        for key in ["revenue_year_growth_rate", "YoY", "yoy", "revenue_yoy"]:
            value = number(row.get(key))
            if value is not None:
                return value
    return None
