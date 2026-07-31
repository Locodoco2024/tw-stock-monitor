from __future__ import annotations

import csv
from pathlib import Path

from src.institutional.integration import (
    dispatch_institutional_notifications,
    load_institutional_plan,
)
from src.notifications.discord import DiscordNotifier
from src.state.manager import StateManager


class FakeHttp:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def post_json(self, url: str, *, payload):  # noqa: ANN001, ANN201
        self.payloads.append(payload)
        return {}


def _write_plan(path: Path) -> None:
    columns = [
        "user_id",
        "stock_id",
        "stock_name",
        "signal_date",
        "event_id",
        "notification_type",
        "notify_mode",
        "percentile",
        "event_age_days",
        "reason",
        "positive_factors",
        "negative_factors",
        "estimated_cost_window_days",
        "estimated_cost_low",
        "estimated_cost_mid",
        "estimated_cost_high",
        "estimated_cost_buy_days",
        "signal_close",
        "signal_deviation_pct",
        "is_configured_stock",
        "eligible_for_future_github",
        "ready_to_send",
        "trade_action",
    ]
    rows = [
        {
            "user_id": "u",
            "stock_id": "7828",
            "stock_name": "創新服務",
            "signal_date": "2026-07-23",
            "event_id": "P5E-7828-2026-07-23",
            "notification_type": "LAYOUT_CONFIRMED_DIRECT",
            "notify_mode": "HIGH_PRIORITY",
            "percentile": "84.7",
            "event_age_days": "0",
            "reason": "法人布局確認",
            "positive_factors": "投信近20日買超日比例",
            "negative_factors": "",
            "estimated_cost_window_days": "20",
            "estimated_cost_low": "96",
            "estimated_cost_mid": "100",
            "estimated_cost_high": "104",
            "estimated_cost_buy_days": "7",
            "signal_close": "110",
            "signal_deviation_pct": "10",
            "is_configured_stock": "0",
            "eligible_for_future_github": "1",
            "ready_to_send": "1",
            "trade_action": "TRACK_ONLY",
        },
        {
            "user_id": "u",
            "stock_id": "6121",
            "stock_name": "新普",
            "signal_date": "2026-07-23",
            "event_id": "P5E-6121-2026-07-23",
            "notification_type": "NEW_CANDIDATE",
            "notify_mode": "NEW_CANDIDATE",
            "percentile": "99.0",
            "event_age_days": "0",
            "reason": "法人排名新進前10%",
            "positive_factors": "投信近20日買超日比例",
            "negative_factors": "",
            "estimated_cost_window_days": "",
            "estimated_cost_low": "",
            "estimated_cost_mid": "",
            "estimated_cost_high": "",
            "estimated_cost_buy_days": "",
            "signal_close": "",
            "signal_deviation_pct": "",
            "is_configured_stock": "0",
            "eligible_for_future_github": "1",
            "ready_to_send": "1",
            "trade_action": "TRACK_ONLY",
        },
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_dispatch_prioritizes_direct_confirmation_and_persists_sent_key(
    tmp_path: Path, monkeypatch
) -> None:
    plan = tmp_path / "plan.csv"
    _write_plan(plan)
    state_path = tmp_path / "state.json"
    state = StateManager(state_path)
    http = FakeHttp()
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://example.invalid/webhook")
    configs = [
        {
            "user": {"id": "u", "enabled": True, "discord_webhook_key": "default"},
            "institutional_candidates": {
                "enabled": True,
                "max_new_candidates": 1,
                "include_state_updates": True,
            },
            "stocks": [{"symbol": "2330", "average_cost": 100, "profit_alerts": [30]}],
        }
    ]

    first = dispatch_institutional_notifications(
        plan_path=plan,
        user_configs=configs,
        holding_results=[],
        state=state,
        notifier=DiscordNotifier(http=http),
        no_discord=False,
    )
    assert first.sent == 1
    assert first.skipped_user_limit == 1
    embed = http.payloads[0]["embeds"][0]
    assert "7828" in embed["title"]
    fields = {field["name"]: field["value"] for field in embed["fields"]}
    assert fields["三法人近20日推估成本帶"] == "96 ～ 104"
    assert fields["推估成本中間值"] == "100"
    assert fields["訊號日收盤價"] == "110"
    assert fields["收盤價相對推估成本"] == "+10.0%"
    state.save()

    restored = StateManager(state_path)
    second = dispatch_institutional_notifications(
        plan_path=plan,
        user_configs=configs,
        holding_results=[],
        state=restored,
        notifier=DiscordNotifier(http=http),
        no_discord=False,
    )
    assert second.sent == 1
    assert second.skipped_already_sent == 1
    assert len(http.payloads) == 2


def test_load_legacy_plan_without_estimated_cost_columns(tmp_path: Path) -> None:
    plan = tmp_path / "legacy-plan.csv"
    columns = [
        "user_id",
        "stock_id",
        "stock_name",
        "signal_date",
        "event_id",
        "notification_type",
        "notify_mode",
        "percentile",
        "event_age_days",
        "reason",
        "positive_factors",
        "negative_factors",
        "is_configured_stock",
        "eligible_for_future_github",
        "ready_to_send",
        "trade_action",
    ]
    with plan.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow(
            {
                "user_id": "u",
                "stock_id": "7828",
                "stock_name": "創新服務",
                "signal_date": "2026-07-30",
                "event_id": "P5H-7828-2026-07-30",
                "notification_type": "NEW_CANDIDATE",
                "notify_mode": "NEW_CANDIDATE",
                "percentile": "92",
                "event_age_days": "0",
                "reason": "new",
                "positive_factors": "正向",
                "negative_factors": "",
                "is_configured_stock": "0",
                "eligible_for_future_github": "1",
                "ready_to_send": "1",
                "trade_action": "TRACK_ONLY",
            }
        )

    notifications = load_institutional_plan(plan)

    assert len(notifications) == 1
    assert notifications[0].estimated_cost_mid is None
    assert notifications[0].signal_close is None

