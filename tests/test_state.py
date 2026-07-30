from __future__ import annotations

import json
from pathlib import Path

from src.models import HoldingMonitorResult, InstitutionalNotification
from src.state.manager import StateManager


def _result(active: list[float], new: list[float]) -> HoldingMonitorResult:
    return HoldingMonitorResult(
        user_id="u",
        symbol="2330",
        name="台積電",
        analyzed_at="2026-07-29T18:00:00+08:00",
        quote=None,
        average_cost=100,
        profit_return_pct=50,
        active_profit_thresholds=active,
        new_profit_thresholds=new,
    )


def test_profit_threshold_crossing_can_trigger_again_after_falling_below(tmp_path: Path) -> None:
    state = StateManager(tmp_path / "state.json")
    assert state.new_profit_thresholds("u", "2330", [30]) == [30]
    state.update_holding(_result([30], [30]))
    assert state.new_profit_thresholds("u", "2330", [30]) == []

    state.update_holding(_result([], []))
    assert state.new_profit_thresholds("u", "2330", [30]) == [30]


def test_legacy_state_is_baselined_without_realerting(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "records": {
                    "u:2330": {
                        "operation_score": 10,
                        "operation_label": "續抱",
                        "risk_score": 0,
                        "notified_at": None,
                        "processed_event_ids": [],
                        "triggered_rule_ids": ["holding.profit_take"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    state = StateManager(path)
    assert state.new_profit_thresholds("u", "2330", [30, 50]) == []
    state.update_holding(_result([30, 50], []))
    assert state.new_profit_thresholds("u", "2330", [30, 50]) == []


def test_institutional_notification_state_persists(tmp_path: Path) -> None:
    notification = InstitutionalNotification(
        user_id="u",
        symbol="7828",
        name="創新服務",
        signal_date="2026-07-23",
        event_id="P5E-7828-2026-07-23",
        notification_type="LAYOUT_CONFIRMED_DIRECT",
        notify_mode="HIGH_PRIORITY",
        percentile=84.7,
        event_age_days=0,
        reason="法人布局確認",
        positive_factors="投信近20日買超日比例",
        negative_factors="",
        is_configured_stock=False,
    )
    path = tmp_path / "state.json"
    manager = StateManager(path)
    manager.mark_institutional_notification_sent(notification)
    manager.save()
    assert StateManager(path).was_institutional_notification_sent(notification)
