from __future__ import annotations

from pathlib import Path

from src.models import AnalysisResult
from src.state.manager import StateManager


def _result() -> AnalysisResult:
    return AnalysisResult(
        user_id="u",
        symbol="2330",
        name="台積電",
        market="TWSE",
        analyzed_at="2026-01-01T00:00:00+08:00",
        quote=None,
        objective_score=50,
        operation_score=50,
        opportunity_score=50,
        risk_score=0,
        completeness=100,
        objective_label="偏多",
        operation_label="買入傾向",
        holding_enabled=False,
        price_return_pct=None,
        modules=[],
        holding_rules=[],
        summary="測試。",
        new_event_ids=[],
        errors=[],
    )


def test_first_run_notifies_and_state_persists(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    manager = StateManager(path)
    result = _result()
    decision = manager.decide(result, {"send_on_first_run": True})
    assert decision.should_send
    manager.update(result, True)
    manager.save()
    restored = StateManager(path)
    assert restored.get("u", "2330").operation_score == 50


def test_holding_threshold_not_repeated_when_still_active(tmp_path: Path) -> None:
    from src.models import RuleResult

    path = tmp_path / "state.json"
    manager = StateManager(path)
    result = _result()
    result.holding_enabled = True
    result.operation_label = "續抱觀察"
    result.holding_rules = [
        RuleResult(
            rule_id="holding.profit_take",
            module="holding",
            score=-30,
            message="達到停利門檻",
        )
    ]
    first = manager.decide(result, {"send_on_first_run": True})
    assert first.should_send
    manager.update(result, True)
    second = manager.decide(result, {"send_on_first_run": True})
    assert not second.should_send
