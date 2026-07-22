from __future__ import annotations

import re
from typing import Any

from src.models import ModuleResult, OfficialEvent, RuleResult
from src.scoring.common import cap_module


_AMOUNT_PATTERN = re.compile(
    r"(?:新台幣|台幣|NT\$?|TWD)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(億元|億|萬元|萬|元)"
)
_YEAR_PATTERN = re.compile(r"(?:期間|合約期間|契約期間)?\s*([1-9][0-9]?)\s*年")


def score_official_events(
    events: list[OfficialEvent],
    scoring: dict[str, Any],
    weight: float,
    available: bool,
    trailing_revenue: float | None,
) -> tuple[ModuleResult, list[str]]:
    if not available:
        return cap_module(
            "official_events", weight, 0.0, [], ["官方重大訊息資料取得失敗"]
        ), []

    config = scoring.get("official_event_rules", {})
    negation_terms = list(config.get("negation_terms", []))
    rule_configs = list(config.get("rules", []))
    rules: list[RuleResult] = []
    matched_event_ids: list[str] = []

    for event in events:
        text = event.combined_text()
        event_matched = False
        for rule_config in rule_configs:
            phrase = _matched_phrase(text, rule_config.get("phrases", []), negation_terms)
            if not phrase:
                continue
            event_matched = True
            score = float(rule_config.get("score", 0))
            title = str(rule_config.get("title") or rule_config.get("id"))
            rules.append(
                RuleResult(
                    rule_id=f"event.{rule_config['id']}.{event.event_id}",
                    module="official_events",
                    score=score,
                    message=f"官方公告符合「{title}」規則：{event.title}",
                    values={
                        "event_id": event.event_id,
                        "published_at": event.published_at,
                        "matched_phrase": phrase,
                    },
                    source=event.source_url,
                )
            )
            if score > 0:
                amount = extract_amount_twd(text)
                years = extract_contract_years(text)
                if amount and trailing_revenue and trailing_revenue > 0:
                    annualized_ratio = amount / max(years, 1) / trailing_revenue * 100
                    materiality_score = materiality_points(annualized_ratio)
                    if materiality_score:
                        rules.append(
                            RuleResult(
                                rule_id=f"event.materiality.{event.event_id}",
                                module="official_events",
                                score=materiality_score,
                                message=(
                                    f"可辨識公告金額，估算年化規模約占近十二月營收 "
                                    f"{annualized_ratio:.1f}%"
                                ),
                                values={
                                    "amount_twd": amount,
                                    "contract_years": years,
                                    "annualized_revenue_ratio_pct": annualized_ratio,
                                },
                                source=event.source_url,
                            )
                        )
            break
        if event_matched:
            matched_event_ids.append(event.event_id)

    notes: list[str] = []
    unmatched = len(events) - len(matched_event_ids)
    if events and unmatched:
        notes.append(f"另有 {unmatched} 則官方訊息無法由保守規則可靠分類，未納入分數")
    if not events:
        notes.append("本次沒有查到追蹤股票的當日官方重大訊息")
    return cap_module("official_events", weight, weight, rules, notes), matched_event_ids


def _matched_phrase(text: str, phrases: list[str], negations: list[str]) -> str | None:
    for phrase in phrases:
        position = text.find(phrase)
        if position < 0:
            continue
        context = text[max(0, position - 12) : position + len(phrase) + 12]
        if any(negation in context for negation in negations):
            continue
        return phrase
    return None


def extract_amount_twd(text: str) -> float | None:
    matches = _AMOUNT_PATTERN.findall(text)
    if not matches:
        return None
    values: list[float] = []
    multipliers = {"億元": 100_000_000, "億": 100_000_000, "萬元": 10_000, "萬": 10_000, "元": 1}
    for raw, unit in matches:
        try:
            values.append(float(raw.replace(",", "")) * multipliers[unit])
        except (ValueError, KeyError):
            continue
    return max(values) if values else None


def extract_contract_years(text: str) -> int:
    match = _YEAR_PATTERN.search(text)
    return max(1, int(match.group(1))) if match else 1


def materiality_points(ratio_pct: float) -> float:
    if ratio_pct >= 40:
        return 10
    if ratio_pct >= 20:
        return 7
    if ratio_pct >= 10:
        return 5
    if ratio_pct >= 3:
        return 3
    return 1 if ratio_pct > 0 else 0
