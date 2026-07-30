from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.models import (
    AppState,
    HoldingMonitorResult,
    HoldingStateRecord,
    InstitutionalNotification,
)


class StateManager:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.state = self._load()

    def key(self, user_id: str, symbol: str) -> str:
        return f"{user_id}:{symbol}"

    def get(self, user_id: str, symbol: str) -> HoldingStateRecord:
        return self.state.records.get(self.key(user_id, symbol), HoldingStateRecord())

    def new_profit_thresholds(
        self,
        user_id: str,
        symbol: str,
        active_thresholds: list[float],
    ) -> list[float]:
        previous = self.get(user_id, symbol)
        if previous.migrated_from_legacy:
            return []
        return sorted(set(active_thresholds) - set(previous.active_profit_thresholds))

    def update_holding(self, result: HoldingMonitorResult) -> None:
        self.state.records[self.key(result.user_id, result.symbol)] = HoldingStateRecord(
            active_profit_thresholds=list(result.active_profit_thresholds),
            migrated_from_legacy=False,
        )
        self.state.updated_at = datetime.now().astimezone().isoformat()

    def was_institutional_notification_sent(
        self, notification: InstitutionalNotification
    ) -> bool:
        return notification.state_key in set(
            self.state.institutional_notification_keys.get(notification.user_id, [])
        )

    def mark_institutional_notification_sent(
        self, notification: InstitutionalNotification
    ) -> None:
        values = list(
            self.state.institutional_notification_keys.get(notification.user_id, [])
        )
        if notification.state_key not in values:
            values.append(notification.state_key)
        self.state.institutional_notification_keys[notification.user_id] = values[-5000:]
        self.state.updated_at = datetime.now().astimezone().isoformat()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": self.state.updated_at,
            "records": {key: value.to_dict() for key, value in self.state.records.items()},
            "institutional_notification_keys": self.state.institutional_notification_keys,
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
            records: dict[str, HoldingStateRecord] = {}
            for key, value in payload.get("records", {}).items():
                if not isinstance(value, dict):
                    continue
                if "active_profit_thresholds" in value:
                    records[str(key)] = HoldingStateRecord(
                        active_profit_thresholds=[
                            float(item) for item in value.get("active_profit_thresholds", [])
                        ],
                        migrated_from_legacy=bool(value.get("migrated_from_legacy", False)),
                    )
                else:
                    records[str(key)] = HoldingStateRecord(migrated_from_legacy=True)
            institutional_keys = {
                str(user_id): [str(value) for value in values]
                for user_id, values in payload.get(
                    "institutional_notification_keys", {}
                ).items()
                if isinstance(values, list)
            }
            return AppState(
                records=records,
                institutional_notification_keys=institutional_keys,
                updated_at=str(payload.get("updated_at", "")),
            )
        except (ValueError, TypeError, json.JSONDecodeError):
            return AppState()
