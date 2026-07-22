from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from dateutil.parser import isoparse

from src.models import AnalysisResult, AppState, NotificationDecision, StateRecord


class StateManager:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.state = self._load()

    def key(self, user_id: str, symbol: str) -> str:
        return f"{user_id}:{symbol}"

    def get(self, user_id: str, symbol: str) -> StateRecord:
        return self.state.records.get(self.key(user_id, symbol), StateRecord.empty())

    def processed_event_ids(self, user_id: str, symbol: str) -> set[str]:
        return set(self.get(user_id, symbol).processed_event_ids)

    def decide(
        self,
        result: AnalysisResult,
        notification_config: dict[str, Any],
    ) -> NotificationDecision:
        previous = self.get(result.user_id, result.symbol)
        reasons: list[str] = []
        first_run = previous.operation_label == ""
        if first_run and notification_config.get("send_on_first_run", True):
            reasons.append("首次分析")
        if result.new_event_ids:
            reasons.append(f"發現 {len(result.new_event_ids)} 則新的官方事件")
        if previous.operation_label and previous.operation_label != result.operation_label:
            reasons.append(f"操作區間由「{previous.operation_label}」變為「{result.operation_label}」")
        score_change = abs(result.operation_score - previous.operation_score)
        if previous.operation_label and score_change >= float(
            notification_config.get("score_change_threshold", 15)
        ):
            reasons.append(f"操作指數變動 {score_change:.1f} 點")
        risk_change = result.risk_score - previous.risk_score
        if previous.operation_label and risk_change >= float(
            notification_config.get("risk_change_threshold", 10)
        ):
            reasons.append(f"風險分數增加 {risk_change:.1f} 點")
        holding_rule_ids = {
            rule.rule_id
            for rule in result.holding_rules
            if rule.rule_id
            in {
                "holding.profit_watch",
                "holding.profit_take",
                "holding.strong_profit_take",
                "holding.conditional_add_on",
            }
        }
        new_holding_triggers = holding_rule_ids - set(previous.triggered_rule_ids)
        if {"holding.profit_take", "holding.strong_profit_take"} & new_holding_triggers:
            reasons.append("達到新的持倉停利門檻")
        elif "holding.profit_watch" in new_holding_triggers:
            reasons.append("進入持倉停利觀察區")
        if "holding.conditional_add_on" in new_holding_triggers:
            reasons.append("進入條件式加碼觀察區")

        should_send = bool(reasons)
        if should_send and not result.new_event_ids and not self._cooldown_passed(
            previous.notified_at,
            float(notification_config.get("cooldown_hours", 12)),
        ):
            material_reason = any(
                reason.startswith("操作區間") or reason.startswith("達到持倉") for reason in reasons
            )
            if not material_reason:
                return NotificationDecision(False, ["仍在通知冷卻期間"])
        return NotificationDecision(should_send, reasons)

    def update(self, result: AnalysisResult, notified: bool) -> None:
        key = self.key(result.user_id, result.symbol)
        previous = self.state.records.get(key, StateRecord.empty())
        event_ids = list(dict.fromkeys(previous.processed_event_ids + result.new_event_ids))[-500:]
        self.state.records[key] = StateRecord(
            operation_score=result.operation_score,
            operation_label=result.operation_label,
            risk_score=result.risk_score,
            notified_at=datetime.now().astimezone().isoformat() if notified else previous.notified_at,
            processed_event_ids=event_ids,
            triggered_rule_ids=[
                rule.rule_id
                for rule in result.holding_rules
                if rule.rule_id
                in {
                    "holding.profit_watch",
                    "holding.profit_take",
                    "holding.strong_profit_take",
                    "holding.conditional_add_on",
                }
            ],
        )
        self.state.updated_at = datetime.now().astimezone().isoformat()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": self.state.updated_at,
            "records": {key: value.to_dict() for key, value in self.state.records.items()},
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load(self) -> AppState:
        if not self.path.exists():
            return AppState()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            records = {
                key: StateRecord(**value) for key, value in payload.get("records", {}).items()
            }
            return AppState(records=records, updated_at=payload.get("updated_at", ""))
        except (ValueError, TypeError, json.JSONDecodeError):
            return AppState()

    @staticmethod
    def _cooldown_passed(notified_at: str | None, hours: float) -> bool:
        if not notified_at:
            return True
        try:
            return datetime.now().astimezone() - isoparse(notified_at) >= timedelta(hours=hours)
        except (ValueError, TypeError):
            return True
