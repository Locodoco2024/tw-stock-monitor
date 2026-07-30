from __future__ import annotations

import csv
from datetime import date, datetime
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

from src.config_loader import institutional_candidates_settings
from src.models import HoldingMonitorResult, InstitutionalNotification
from src.notifications.discord import DiscordNotifier
from src.state.manager import StateManager


LOGGER = logging.getLogger("tw-stock-monitor.institutional")

ACTIVE_NOTIFICATION_TYPES = {"LAYOUT_CONFIRMED_DIRECT", "NEW_CANDIDATE"}
STATE_UPDATE_TYPES = {
    "LAYOUT_CONFIRMED",
    "LAYOUT_CONFIRMED_AND_EXTENDED",
    "DAY20_EXTEND",
    "DAY20_EXTEND_STRONG",
}
NOTIFICATION_PRIORITY = {
    "LAYOUT_CONFIRMED_DIRECT": 50,
    "NEW_CANDIDATE": 40,
    "LAYOUT_CONFIRMED_AND_EXTENDED": 35,
    "LAYOUT_CONFIRMED": 30,
    "DAY20_EXTEND": 20,
    "DAY20_EXTEND_STRONG": 20,
}


@dataclass(frozen=True, slots=True)
class InstitutionalDispatchResult:
    loaded: int = 0
    eligible: int = 0
    sent: int = 0
    skipped_disabled: int = 0
    skipped_user_limit: int = 0
    skipped_state_updates: int = 0
    skipped_already_sent: int = 0
    skipped_no_webhook: int = 0
    skipped_no_discord: int = 0


def load_institutional_plan(path: str | Path) -> list[InstitutionalNotification]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"找不到法人通知計畫：{target}")

    rows: list[InstitutionalNotification] = []
    with target.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "user_id",
            "stock_id",
            "signal_date",
            "event_id",
            "notification_type",
            "notify_mode",
            "eligible_for_future_github",
            "ready_to_send",
            "trade_action",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"法人通知計畫缺少欄位：{sorted(missing)}")
        today = datetime.now(ZoneInfo("Asia/Taipei")).date()
        for raw in reader:
            if not _truthy(raw.get("eligible_for_future_github")):
                continue
            if not _truthy(raw.get("ready_to_send")):
                continue
            valid_until = str(raw.get("plan_valid_until") or "").strip()
            if valid_until:
                try:
                    if date.fromisoformat(valid_until) < today:
                        LOGGER.warning(
                            "略過過期法人通知計畫：%s %s",
                            raw.get("event_id"),
                            valid_until,
                        )
                        continue
                except ValueError as exc:
                    raise ValueError(
                        f"法人通知計畫 plan_valid_until 格式錯誤：{valid_until}"
                    ) from exc
            trade_action = str(raw.get("trade_action") or "")
            if trade_action != "TRACK_ONLY":
                raise ValueError(
                    f"法人通知只能是 TRACK_ONLY：{raw.get('event_id')}={trade_action}"
                )
            rows.append(
                InstitutionalNotification(
                    user_id=str(raw.get("user_id") or ""),
                    symbol=str(raw.get("stock_id") or ""),
                    name=str(raw.get("stock_name") or ""),
                    signal_date=str(raw.get("signal_date") or ""),
                    event_id=str(raw.get("event_id") or ""),
                    notification_type=str(raw.get("notification_type") or ""),
                    notify_mode=str(raw.get("notify_mode") or ""),
                    percentile=_float_or_none(raw.get("percentile")),
                    event_age_days=_int_or_none(raw.get("event_age_days")),
                    reason=str(raw.get("reason") or ""),
                    positive_factors=str(raw.get("positive_factors") or ""),
                    negative_factors=str(raw.get("negative_factors") or ""),
                    is_configured_stock=_truthy(raw.get("is_configured_stock")),
                    trade_action=trade_action,
                )
            )

    keys = [item.state_key for item in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("法人通知計畫含重複 user/event/type/date")
    return sorted(
        rows,
        key=lambda item: (
            item.user_id,
            -NOTIFICATION_PRIORITY.get(item.notification_type, 0),
            -(item.percentile or -1.0),
            item.symbol,
        ),
    )


def dispatch_institutional_notifications(
    *,
    plan_path: str | Path,
    user_configs: Iterable[dict],
    holding_results: Iterable[HoldingMonitorResult],
    state: StateManager,
    notifier: DiscordNotifier,
    no_discord: bool,
) -> InstitutionalDispatchResult:
    notifications = load_institutional_plan(plan_path)
    configs = {str(config["user"]["id"]): config for config in user_configs}
    holding_by_key = {
        (result.user_id, result.symbol): result for result in holding_results
    }
    counters = {
        "loaded": len(notifications),
        "eligible": 0,
        "sent": 0,
        "skipped_disabled": 0,
        "skipped_user_limit": 0,
        "skipped_state_updates": 0,
        "skipped_already_sent": 0,
        "skipped_no_webhook": 0,
        "skipped_no_discord": 0,
    }
    new_candidate_counts: dict[str, int] = {}

    for notification in notifications:
        config = configs.get(notification.user_id)
        if config is None:
            counters["skipped_disabled"] += 1
            continue
        settings = institutional_candidates_settings(config)
        if not settings["enabled"]:
            counters["skipped_disabled"] += 1
            continue
        if (
            notification.notification_type in STATE_UPDATE_TYPES
            and not settings["include_state_updates"]
        ):
            counters["skipped_state_updates"] += 1
            continue
        if state.was_institutional_notification_sent(notification):
            counters["skipped_already_sent"] += 1
            continue
        if (
            notification.notification_type in ACTIVE_NOTIFICATION_TYPES
            and not notification.is_configured_stock
        ):
            current = new_candidate_counts.get(notification.user_id, 0)
            if current >= settings["max_new_candidates"]:
                counters["skipped_user_limit"] += 1
                continue
            new_candidate_counts[notification.user_id] = current + 1

        counters["eligible"] += 1
        if no_discord:
            counters["skipped_no_discord"] += 1
            LOGGER.info(
                "略過法人 Discord：%s/%s %s",
                notification.user_id,
                notification.symbol,
                notification.notification_type,
            )
            continue

        webhook = notifier.resolve_webhook(
            config.get("user", {}).get("discord_webhook_key")
        )
        if not webhook:
            counters["skipped_no_webhook"] += 1
            LOGGER.warning(
                "%s/%s 缺少 Discord Webhook，法人通知未發送",
                notification.user_id,
                notification.symbol,
            )
            continue

        notifier.send_institutional(
            notification,
            webhook,
            holding=holding_by_key.get((notification.user_id, notification.symbol)),
        )
        state.mark_institutional_notification_sent(notification)
        counters["sent"] += 1
        LOGGER.info(
            "已發送法人 Discord：%s/%s %s",
            notification.user_id,
            notification.symbol,
            notification.notification_type,
        )

    return InstitutionalDispatchResult(**counters)


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "1.0", "true", "yes", "y"}


def _float_or_none(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _int_or_none(value: object) -> int | None:
    number = _float_or_none(value)
    return int(number) if number is not None else None
