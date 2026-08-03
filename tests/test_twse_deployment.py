from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from research.institutional_model.cli import build_parser
from research.institutional_model.phase6_twse_final import validate_phase6d_gates
from src.institutional.integration import dispatch_institutional_notifications
from src.institutional.twse_daily_pipeline import (
    TwseOfficialClient,
    merge_twse_daily_rows,
    replay_twse_lifecycle,
)
from src.models import InstitutionalNotification
from src.notifications.discord import DiscordNotifier
from src.state.manager import StateManager


class _Response:
    def __init__(self, payload):  # noqa: ANN001
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):  # noqa: ANN201
        return self.payload


class _Session:
    def __init__(self, payloads):  # noqa: ANN001
        self.payloads = list(payloads)
        self.calls = []

    def get(self, url, **kwargs):  # noqa: ANN001, ANN201
        self.calls.append({"url": url, **kwargs})
        return _Response(self.payloads.pop(0))


class _Http:
    def __init__(self) -> None:
        self.payload = None

    def post_json(self, url: str, *, payload):  # noqa: ANN001, ANN201
        self.payload = payload
        return {}


def test_phase6d_cli_command_is_available() -> None:
    args = build_parser().parse_args(["phase6d"])
    assert args.command == "phase6d"


def test_phase6d_gate_requires_selected_phase6c_rule(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {"metric": "phase6b_version", "value": "phase6b-v1"},
            {"metric": "return_rank_40d_validation_pass", "value": "1"},
        ]
    ).to_csv(tmp_path / "phase6b_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"metric": "phase6c_version", "value": "phase6c-v1"},
            {"metric": "lifecycle_validation_pass", "value": "1"},
        ]
    ).to_csv(tmp_path / "phase6c_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "universe_id": "money100m_volume300lots",
                "entry_threshold": 90,
                "confirmation_days": 3,
                "candidate_status": "strong_candidate",
            }
        ]
    ).to_csv(tmp_path / "phase6c_rule_candidates.csv", index=False, encoding="utf-8-sig")

    result = validate_phase6d_gates(tmp_path)

    assert result["phase6b_validation_pass"] == "1"
    assert result["phase6c_validation_pass"] == "1"


def test_twse_official_payload_parses_and_excludes_dealer_hedging() -> None:
    quote_fields = [
        "證券代號", "證券名稱", "成交股數", "成交金額",
        "開盤價", "最高價", "最低價", "收盤價",
    ]
    flow_fields = [
        "證券代號", "證券名稱",
        "外陸資買賣超股數(不含外資自營商)",
        "投信買賣超股數",
        "自營商自行買賣買賣超股數",
        "自營商避險買賣超股數",
    ]
    session = _Session(
        [
            {
                "stat": "OK",
                "fields1": ["說明"],
                "data1": [["x"]],
                "fields9": quote_fields,
                "data9": [["2330", "台積電", "1,000,000", "900,000,000", "900", "910", "895", "905"]],
            },
            {
                "stat": "OK",
                "fields": flow_fields,
                "data": [["2330", "台積電", "100,000", "20,000", "5,000", "999,999"]],
            },
        ]
    )
    client = TwseOfficialClient(session=session)
    quotes, flows = client.fetch_date(date(2026, 7, 31))
    universe = pd.DataFrame([{"stock_id": "2330", "stock_name": "台積電", "listing_date": ""}])

    merged = merge_twse_daily_rows(quotes, flows, universe)

    assert merged.iloc[0]["selected_total_net"] == 125_000
    assert merged.iloc[0]["trading_volume"] == 1_000_000
    assert session.calls[0]["params"]["type"] == "ALLBUT0999"
    assert session.calls[1]["params"]["selectType"] == "ALLBUT0999"


def test_twse_lifecycle_confirms_on_third_top10_day_and_ends_on_day40() -> None:
    start = date(2026, 1, 1)
    rows = []
    for index in range(45):
        signal_date = (start + timedelta(days=index)).isoformat()
        rows.append(
            {
                "stock_id": "2330",
                "stock_name": "台積電",
                "signal_date": signal_date,
                "return_rank_score": 95.0,
                "positive_factors": "foreign (+0.1)",
                "negative_factors": "",
            }
        )
    result = replay_twse_lifecycle(pd.DataFrame(rows))
    notifications = result["notifications"]

    entry = notifications[notifications["notification_type"] == "TWSE_TRACK_CONFIRMED"].iloc[0]
    end = notifications[notifications["notification_type"] == "TWSE_DAY40_END"].iloc[0]
    assert entry["signal_date"] == (start + timedelta(days=2)).isoformat()
    assert int(entry["event_age_days"]) == 0
    assert end["signal_date"] == (start + timedelta(days=42)).isoformat()
    assert int(end["event_age_days"]) == 40


def test_twse_notification_has_market_specific_rank_and_state_key() -> None:
    notification = InstitutionalNotification(
        user_id="u",
        symbol="2330",
        name="台積電",
        signal_date="2026-07-31",
        event_id="P6D-TWSE-2330-2026-07-31",
        notification_type="TWSE_TRACK_CONFIRMED",
        notify_mode="HIGH_PRIORITY_NEW",
        percentile=95.0,
        event_age_days=0,
        reason="上市法人排名連續3日前10%",
        positive_factors="外資近20日流量 (+0.1)",
        negative_factors="",
        is_configured_stock=False,
        market="twse",
    )
    http = _Http()
    DiscordNotifier(http=http).send_institutional(
        notification,
        "https://example.invalid/webhook",
    )
    fields = {item["name"]: item["value"] for item in http.payload["embeds"][0]["fields"]}

    assert fields["市場"] == "TWSE 上市"
    assert fields["同日上市法人排名"] == "95.0 百分位"
    assert "|twse|" in notification.state_key
    assert "TWSE 上市同日相對排序" in http.payload["embeds"][0]["footer"]["text"]


def test_tpex_state_key_remains_backward_compatible() -> None:
    notification = InstitutionalNotification(
        user_id="u",
        symbol="1815",
        name="富喬",
        signal_date="2026-07-30",
        event_id="P5H-1815-2026-07-30",
        notification_type="NEW_CANDIDATE",
        notify_mode="NEW_CANDIDATE",
        percentile=92.0,
        event_age_days=0,
        reason="new",
        positive_factors="",
        negative_factors="",
        is_configured_stock=False,
    )
    assert notification.state_key == "u|P5H-1815-2026-07-30|NEW_CANDIDATE|2026-07-30"


def test_combined_twse_and_tpex_plans_share_user_candidate_limit(tmp_path: Path) -> None:
    columns = [
        "market", "user_id", "stock_id", "stock_name", "signal_date",
        "event_id", "notification_type", "notify_mode", "percentile",
        "event_age_days", "reason", "positive_factors", "negative_factors",
        "is_configured_stock", "eligible_for_future_github", "ready_to_send",
        "trade_action",
    ]
    paths = []
    for market, stock_id, notification_type in (
        ("tpex", "1815", "LAYOUT_CONFIRMED_DIRECT"),
        ("twse", "2330", "TWSE_TRACK_CONFIRMED"),
    ):
        path = tmp_path / f"{market}.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerow(
                {
                    "market": market,
                    "user_id": "u",
                    "stock_id": stock_id,
                    "stock_name": stock_id,
                    "signal_date": "2026-07-31",
                    "event_id": f"{market}-{stock_id}",
                    "notification_type": notification_type,
                    "notify_mode": "HIGH_PRIORITY_NEW",
                    "percentile": "95",
                    "event_age_days": "0",
                    "reason": "track",
                    "positive_factors": "",
                    "negative_factors": "",
                    "is_configured_stock": "0",
                    "eligible_for_future_github": "1",
                    "ready_to_send": "1",
                    "trade_action": "TRACK_ONLY",
                }
            )
        paths.append(path)
    config = {
        "user": {"id": "u", "enabled": True, "discord_webhook_key": "default"},
        "institutional_candidates": {
            "enabled": True,
            "max_new_candidates": 1,
            "include_state_updates": True,
        },
        "stocks": [],
    }

    result = dispatch_institutional_notifications(
        plan_path=paths,
        user_configs=[config],
        holding_results=[],
        state=StateManager(tmp_path / "state.json"),
        notifier=DiscordNotifier(http=_Http()),
        no_discord=True,
    )

    assert result.loaded == 2
    assert result.eligible == 1
    assert result.skipped_user_limit == 1
