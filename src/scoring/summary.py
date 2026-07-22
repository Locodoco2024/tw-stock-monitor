from __future__ import annotations

from src.models import ModuleResult, RuleResult
from src.scoring.plain_language import plain_message


_HOLDING_PRIORITY = [
    "holding.strong_profit_take",
    "holding.profit_take",
    "holding.profit_watch",
    "holding.conditional_add_on",
    "holding.add_on_not_eligible",
]

_RULE_PRIORITY = {
    "official_events": 10,
    "company_capacity": 20,
    "industry_peers": 30,
    "market_pricing": 40,
    "market_environment": 50,
    "holding": 0,
}


def build_summary(
    modules: list[ModuleResult],
    holding_rules: list[RuleResult],
    completeness: float,
    objective_score: float,
    holding_enabled: bool,
) -> str:
    rules = [rule for module in modules for rule in module.rules]
    holding_rule = _first_holding_rule(holding_rules)

    if holding_rule:
        sentences = [plain_message(holding_rule)]
        sentences.append(_holding_context(holding_rule, objective_score))
    else:
        sentences = _general_summary(rules, objective_score, holding_enabled)

    if completeness < 80:
        sentences.append(
            f"本次分析涵蓋度僅 {completeness:.0f}%，部分面向未納入，結論需保守解讀"
        )
    return "；".join(_deduplicate(sentences)[:3]) + "。"


def _first_holding_rule(rules: list[RuleResult]) -> RuleResult | None:
    by_id = {rule.rule_id: rule for rule in rules}
    for rule_id in _HOLDING_PRIORITY:
        if rule_id in by_id:
            return by_id[rule_id]
    return None


def _holding_context(rule: RuleResult, objective_score: float) -> str:
    if rule.rule_id in {
        "holding.strong_profit_take",
        "holding.profit_take",
        "holding.profit_watch",
    }:
        if objective_score >= 15:
            return "公司與市場條件仍偏正面，因此目前較接近分批停利，而不是因營運明顯惡化而退出"
        if objective_score <= -15:
            return "公司或市場條件也同步偏弱，停利與減碼理由同時增加"
        return "股票本身目前沒有明顯轉強或轉弱，可優先依原先停利規則分批處理"
    if rule.rule_id == "holding.conditional_add_on":
        return "下跌本身不是加碼理由，本次是因股票本身的分析仍偏正面才列入觀察"
    if rule.rule_id == "holding.add_on_not_eligible":
        return "下跌本身不會自動提高加碼傾向，仍需等待公司或市場條件改善"
    return "目前先以續抱觀察為主"


def _general_summary(
    rules: list[RuleResult], objective_score: float, holding_enabled: bool
) -> list[str]:
    positive = _ordered([rule for rule in rules if rule.score > 0])
    negative = _ordered([rule for rule in rules if rule.score < 0])
    trend = _trend_sentence(rules)

    if -15 < objective_score < 15:
        sentences: list[str] = []
        if trend:
            sentences.append(trend)
        elif positive and negative:
            sentences.extend([plain_message(positive[0]), plain_message(negative[0])])
        elif positive:
            sentences.append(plain_message(positive[0]))
        elif negative:
            sentences.append(plain_message(negative[0]))
        sentences.append(
            "目前沒有足夠證據支持明顯加碼或減碼"
            if holding_enabled
            else "目前沒有足夠證據支持明顯買進或避開"
        )
        return sentences

    if objective_score >= 15:
        sentences = [plain_message(positive[0])] if positive else []
        if negative:
            sentences.append(plain_message(negative[0]))
        elif trend:
            sentences.append(trend)
        return sentences or ["目前多項條件偏正面，但仍需持續觀察後續變化"]

    sentences = [plain_message(negative[0])] if negative else []
    if positive:
        sentences.append(plain_message(positive[0]))
    elif trend:
        sentences.append(trend)
    return sentences or ["目前多項條件偏弱，應提高風險警覺"]


def _trend_sentence(rules: list[RuleResult]) -> str | None:
    values = {rule.rule_id: rule.score for rule in rules}
    short = values.get("pricing.price_vs_ema20")
    medium = values.get("pricing.ema20_vs_ema60")
    long_term = values.get("pricing.ema60_vs_ema120")

    if short is not None and medium is not None:
        if short > 0 and medium > 0:
            return "股價短期與中期走勢同步偏強"
        if short < 0 and medium < 0:
            return "股價短期與中期走勢同步轉弱"
        if short > 0 and medium < 0:
            return "股價短線略有回穩，但中期走勢尚未明顯轉強"
        if short < 0 and medium > 0:
            return "股價短線轉弱，但中期趨勢仍有支撐"
    if long_term is not None:
        return "中長期走勢仍維持正向" if long_term > 0 else "中長期走勢仍承受壓力"
    return None


def _ordered(rules: list[RuleResult]) -> list[RuleResult]:
    return sorted(
        rules,
        key=lambda rule: (
            _RULE_PRIORITY.get(rule.module, 99),
            -abs(rule.score),
            rule.rule_id,
        ),
    )


def _deduplicate(sentences: list[str]) -> list[str]:
    result: list[str] = []
    for sentence in sentences:
        cleaned = sentence.rstrip("。；; ")
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return result
