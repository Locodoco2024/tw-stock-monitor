from __future__ import annotations

from src.models import ModuleResult, RuleResult


def build_summary(
    modules: list[ModuleResult],
    holding_rules: list[RuleResult],
    completeness: float,
) -> str:
    rules = [rule for module in modules for rule in module.rules] + holding_rules
    positive = sorted((rule for rule in rules if rule.score > 0), key=lambda item: item.score, reverse=True)
    negative = sorted((rule for rule in rules if rule.score < 0), key=lambda item: item.score)
    neutral_holding = [
        rule for rule in holding_rules if rule.score == 0 and rule.rule_id != "holding.price_unavailable"
    ]

    sentences: list[str] = []
    if positive:
        sentences.append(_clean(positive[0].message))
    if negative:
        sentences.append(_clean(negative[0].message))
    elif len(positive) >= 2:
        sentences.append(_clean(positive[1].message))
    elif neutral_holding:
        sentences.append(_clean(neutral_holding[0].message))
    if not sentences:
        sentences.append("目前正負面規則互相抵銷，暫時沒有明確方向")
    if completeness < 80:
        sentences.append(f"本次資料完整度為 {completeness:.0f}%，結果需保守解讀")
    return "；".join(sentences[:3]) + "。"


def _clean(message: str) -> str:
    return message.rstrip("。；; ")
