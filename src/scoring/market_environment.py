from __future__ import annotations

from typing import Any

from src.indicators import calculate_indicators
from src.models import ModuleResult, RuleResult
from src.scoring.common import cap_module


def score_market_environment(
    market_prices: list[dict[str, Any]], weight: float
) -> ModuleResult:
    indicators = calculate_indicators(market_prices)
    if not indicators:
        return cap_module(
            "market_environment", weight, 0.0, [], ["大盤歷史行情不足"]
        )
    rules: list[RuleResult] = []
    close = indicators.get("close")
    ema20 = indicators.get("ema20")
    ema60 = indicators.get("ema60")
    ema120 = indicators.get("ema120")
    if close is not None and ema20 is not None:
        rules.append(
            RuleResult(
                rule_id="market.price_vs_ema20",
                module="market_environment",
                score=2 if close > ema20 else -2,
                message=f"大盤目前{'高於' if close > ema20 else '低於'} EMA20",
                values={"close": close, "ema20": ema20},
            )
        )
    if ema20 is not None and ema60 is not None:
        rules.append(
            RuleResult(
                rule_id="market.ema20_vs_ema60",
                module="market_environment",
                score=3 if ema20 > ema60 else -3,
                message=f"大盤 EMA20 {'高於' if ema20 > ema60 else '低於'} EMA60",
                values={"ema20": ema20, "ema60": ema60},
            )
        )
    if ema60 is not None and ema120 is not None:
        rules.append(
            RuleResult(
                rule_id="market.ema60_vs_ema120",
                module="market_environment",
                score=2 if ema60 > ema120 else -2,
                message=f"大盤 EMA60 {'高於' if ema60 > ema120 else '低於'} EMA120",
                values={"ema60": ema60, "ema120": ema120},
            )
        )
    return_20d = indicators.get("return_20d")
    if return_20d is not None:
        score = 2 if return_20d >= 3 else -2 if return_20d <= -3 else 0
        if score:
            rules.append(
                RuleResult(
                    rule_id="market.return_20d",
                    module="market_environment",
                    score=score,
                    message=f"大盤近二十日報酬為 {return_20d:+.1f}%",
                    values={"return_20d_pct": return_20d},
                )
            )
    return cap_module("market_environment", weight, weight, rules)
