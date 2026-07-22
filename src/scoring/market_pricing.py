from __future__ import annotations

from typing import Any

from src.indicators import calculate_indicators, median
from src.models import ModuleResult, Quote, RuleResult
from src.scoring.common import cap_module


def score_market_pricing(
    prices: list[dict[str, Any]],
    market_prices: list[dict[str, Any]],
    peer_prices: dict[str, list[dict[str, Any]]],
    quote: Quote | None,
    scoring: dict[str, Any],
    weight: float,
) -> tuple[ModuleResult, dict[str, Any]]:
    indicators = calculate_indicators(prices)
    if not indicators:
        return cap_module(
            "market_pricing", weight, 0.0, [], ["歷史行情不足，無法計算市場定價狀態"]
        ), {}

    current_price = quote.price if quote else indicators.get("close")
    market = calculate_indicators(market_prices)
    peer_indicators = [calculate_indicators(rows) for rows in peer_prices.values() if rows]
    peer_return_5d = median([item.get("return_5d") for item in peer_indicators])
    peer_return_20d = median([item.get("return_20d") for item in peer_indicators])
    rules: list[RuleResult] = []

    ema20 = indicators.get("ema20")
    ema60 = indicators.get("ema60")
    ema120 = indicators.get("ema120")
    if current_price is not None and ema20 is not None:
        score = 3 if current_price > ema20 else -3
        rules.append(
            RuleResult(
                rule_id="pricing.price_vs_ema20",
                module="market_pricing",
                score=score,
                message=f"目前價格{'高於' if score > 0 else '低於'} EMA20",
                values={"price": current_price, "ema20": ema20},
            )
        )
    if ema20 is not None and ema60 is not None:
        score = 4 if ema20 > ema60 else -4
        rules.append(
            RuleResult(
                rule_id="pricing.ema20_vs_ema60",
                module="market_pricing",
                score=score,
                message=f"EMA20 {'高於' if score > 0 else '低於'} EMA60",
                values={"ema20": ema20, "ema60": ema60},
            )
        )
    if ema60 is not None and ema120 is not None:
        rules.append(
            RuleResult(
                rule_id="pricing.ema60_vs_ema120",
                module="market_pricing",
                score=3 if ema60 > ema120 else -3,
                message=f"EMA60 {'高於' if ema60 > ema120 else '低於'} EMA120",
                values={"ema60": ema60, "ema120": ema120},
            )
        )

    return_5d = indicators.get("return_5d")
    market_return_5d = market.get("return_5d")
    if return_5d is not None and market_return_5d is not None:
        has_peer_comparison = peer_return_5d is not None
        comparison_basis = "大盤與同業" if has_peer_comparison else "大盤"
        relative = return_5d - market_return_5d - (peer_return_5d or 0)
        relative_score = 0.0
        if relative >= 8:
            relative_score = 5
        elif relative >= 3:
            relative_score = 3
        elif relative <= -8:
            relative_score = -5
        elif relative <= -3:
            relative_score = -3
        if relative_score:
            rules.append(
                RuleResult(
                    rule_id="pricing.relative_return_5d",
                    module="market_pricing",
                    score=relative_score,
                    message=(
                        f"近五日相對{comparison_basis}表現"
                        f"{'較強' if relative_score > 0 else '較弱'} "
                        f"({relative:+.1f} 個百分點)"
                    ),
                    values={
                        "relative_return_5d_pct": relative,
                        "comparison_basis": comparison_basis,
                    },
                )
            )
        for threshold in scoring.get("market_pricing", {}).get(
            "pre_event_relative_return_penalties", []
        ):
            if relative >= float(threshold["min_pct"]):
                rules.append(
                    RuleResult(
                        rule_id="pricing.already_reflected",
                        module="market_pricing",
                        score=float(threshold["score"]),
                        message=(
                            f"近五日已相對{comparison_basis}上漲 {relative:.1f}%，"
                            "部分正面因素可能已提前反映"
                        ),
                        values={
                            "relative_return_5d_pct": relative,
                            "comparison_basis": comparison_basis,
                        },
                    )
                )
                break

    distance = indicators.get("distance_from_ema20_atr")
    if distance is not None and distance > 0:
        for threshold in scoring.get("market_pricing", {}).get(
            "price_extension_atr_penalties", []
        ):
            if distance >= float(threshold["min_atr"]):
                rules.append(
                    RuleResult(
                        rule_id="pricing.extension_from_ema20",
                        module="market_pricing",
                        score=float(threshold["score"]),
                        message=f"價格距 EMA20 已達 {distance:.2f} ATR，追價風險提高",
                        values={"distance_atr": distance},
                    )
                )
                break

    volume_ratio = indicators.get("volume_ratio20")
    if volume_ratio is not None:
        if volume_ratio >= 1.5 and (return_5d or 0) > 0:
            rules.append(
                RuleResult(
                    rule_id="pricing.volume_confirmation",
                    module="market_pricing",
                    score=3,
                    message=f"成交量為二十日均量 {volume_ratio:.2f} 倍，量價獲得確認",
                    values={"volume_ratio20": volume_ratio},
                )
            )
        elif volume_ratio >= 1.5 and (return_5d or 0) < 0:
            rules.append(
                RuleResult(
                    rule_id="pricing.volume_decline",
                    module="market_pricing",
                    score=-4,
                    message=f"下跌時成交量放大至二十日均量 {volume_ratio:.2f} 倍",
                    values={"volume_ratio20": volume_ratio},
                )
            )

    rsi14 = indicators.get("rsi14")
    if rsi14 is not None:
        if 45 <= rsi14 <= 65:
            rules.append(
                RuleResult(
                    rule_id="pricing.rsi_moderate",
                    module="market_pricing",
                    score=2,
                    message=f"RSI14 為 {rsi14:.1f}，動能正向且尚未極端過熱",
                    values={"rsi14": rsi14},
                )
            )
        elif rsi14 >= 75:
            rules.append(
                RuleResult(
                    rule_id="pricing.rsi_overheated",
                    module="market_pricing",
                    score=-3,
                    message=f"RSI14 為 {rsi14:.1f}，短期動能偏熱",
                    values={"rsi14": rsi14},
                )
            )

    context = {
        "stock": indicators,
        "market": market,
        "peer_return_5d": peer_return_5d,
        "peer_return_20d": peer_return_20d,
        "current_price": current_price,
    }
    available_weight = weight if quote is not None else weight * 0.8
    notes = [] if quote is not None else ["即時報價不可用，本次以最新歷史收盤資料計算"]
    return cap_module("market_pricing", weight, available_weight, rules, notes), context
