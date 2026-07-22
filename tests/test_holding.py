from __future__ import annotations

import pytest

from src.scoring.holding import score_holding


def test_profit_take_adjustment() -> None:
    stock = {
        "holding": {"enabled": True, "average_cost": 100},
        "profit_take": {"alert_at_pct": 50, "strong_alert_at_pct": 75},
    }
    rules, return_pct = score_holding(stock, 155, 50, 100, {}, False)
    assert return_pct == pytest.approx(55)
    assert sum(rule.score for rule in rules) == -30


def test_drawdown_does_not_add_score_when_objective_is_weak() -> None:
    stock = {
        "holding": {"enabled": True, "average_cost": 100},
        "add_on": {
            "enabled": True,
            "alert_at_loss_pct": [10, 20, 30],
            "minimum_analysis_score": 40,
        },
    }
    rules, return_pct = score_holding(stock, 75, -20, 100, {}, False)
    assert return_pct == -25
    assert sum(rule.score for rule in rules) == 0
    assert rules[0].rule_id == "holding.add_on_not_eligible"


def test_drawdown_can_be_add_on_candidate_when_analysis_is_strong() -> None:
    stock = {
        "holding": {"enabled": True, "average_cost": 100},
        "add_on": {
            "enabled": True,
            "alert_at_loss_pct": [10, 20, 30],
            "minimum_analysis_score": 40,
            "maximum_adjustment": 10,
        },
    }
    rules, _ = score_holding(stock, 75, 60, 90, {}, False)
    assert sum(rule.score for rule in rules) > 0
