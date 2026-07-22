from __future__ import annotations

from typing import Any

from src.models import ModuleResult, RuleResult


def cap_module(
    module: str,
    weight: float,
    available_weight: float,
    rules: list[RuleResult],
    notes: list[str] | None = None,
) -> ModuleResult:
    raw = sum(rule.score for rule in rules)
    score = max(-weight, min(weight, raw))
    return ModuleResult(
        module=module,
        weight=weight,
        score=round(score, 2),
        available_weight=available_weight,
        rules=rules,
        notes=notes or [],
    )


def number(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None
