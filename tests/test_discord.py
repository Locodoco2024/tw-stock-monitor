from __future__ import annotations

from src.models import HoldingMonitorResult, InstitutionalNotification, Quote
from src.notifications.discord import DiscordNotifier


class FakeHttp:
    def __init__(self) -> None:
        self.payload = None

    def post_json(self, url: str, *, payload):  # noqa: ANN001, ANN201
        self.payload = payload
        return {}


def _holding() -> HoldingMonitorResult:
    return HoldingMonitorResult(
        user_id="test",
        symbol="1815",
        name="富喬",
        analyzed_at="2026-07-29T18:00:00+08:00",
        quote=Quote(
            symbol="1815",
            name="富喬",
            price=150,
            previous_close=148,
            open_price=149,
            high_price=152,
            low_price=147,
            change_percent=1.35,
            trade_volume=1000,
            as_of="2026-07-29T13:30:00+08:00",
        ),
        average_cost=100,
        profit_return_pct=50,
        active_profit_thresholds=[30, 50],
        new_profit_thresholds=[50],
    )


def test_profit_alert_contains_only_cost_return_and_thresholds() -> None:
    http = FakeHttp()
    DiscordNotifier(http=http).send_profit_alert(
        _holding(), "https://example.invalid/webhook"
    )
    embed = http.payload["embeds"][0]
    fields = {field["name"]: field["value"] for field in embed["fields"]}
    assert fields["目前價格"] == "150"
    assert fields["平均成本"] == "100"
    assert fields["目前損益"] == "+50.0%"
    assert "方向分數" not in fields
    assert "分析涵蓋度" not in fields


def test_institutional_message_adds_holding_profit_without_old_score() -> None:
    http = FakeHttp()
    notification = InstitutionalNotification(
        user_id="test",
        symbol="1815",
        name="富喬",
        signal_date="2026-07-23",
        event_id="P5E-1815-2026-07-23",
        notification_type="LAYOUT_CONFIRMED_DIRECT",
        notify_mode="HIGH_PRIORITY",
        percentile=94.7,
        event_age_days=0,
        reason="連續5日維持法人排名前20%",
        positive_factors="投信近20日買超日比例",
        negative_factors="外資近20日流量",
        is_configured_stock=True,
        estimated_cost_window_days=20,
        estimated_cost_low=120.25,
        estimated_cost_mid=125.5,
        estimated_cost_high=130.75,
        estimated_cost_buy_days=9,
        signal_close=134.0,
        signal_deviation_pct=6.7729,
    )
    DiscordNotifier(http=http).send_institutional(
        notification,
        "https://example.invalid/webhook",
        holding=_holding(),
    )
    fields = {
        field["name"]: field["value"]
        for field in http.payload["embeds"][0]["fields"]
    }
    assert fields["法人狀態"] == "法人連續布局確認"
    assert fields["目前損益"] == "+50.0%"
    assert fields["三法人近20日推估成本帶"] == "120.25 ～ 130.75"
    assert fields["推估成本中間值"] == "125.5"
    assert fields["訊號日收盤價"] == "134"
    assert fields["收盤價相對推估成本"] == "+6.8%"
    assert fields["三法人淨買超日數"] == "9／20 個交易日"
    assert "非真實持倉成本" in http.payload["embeds"][0]["footer"]["text"]
    serialized = str(http.payload)
    assert "魚尾" not in serialized
    assert "追價風險" not in serialized
    assert "原模型判斷" not in fields
    assert "原方向分數" not in fields

def test_institutional_message_explains_when_cost_cannot_be_estimated() -> None:
    http = FakeHttp()
    notification = InstitutionalNotification(
        user_id="test",
        symbol="7828",
        name="創新服務",
        signal_date="2026-07-30",
        event_id="P5H-7828-2026-07-30",
        notification_type="NEW_CANDIDATE",
        notify_mode="NEW_CANDIDATE",
        percentile=92.0,
        event_age_days=0,
        reason="法人排名首次進入同日TPEx前10%",
        positive_factors="投信近20日買超日比例",
        negative_factors="",
        is_configured_stock=False,
        estimated_cost_window_days=20,
        estimated_cost_buy_days=0,
        signal_close=88.5,
    )

    DiscordNotifier(http=http).send_institutional(
        notification,
        "https://example.invalid/webhook",
    )
    fields = {
        field["name"]: field["value"]
        for field in http.payload["embeds"][0]["fields"]
    }

    assert fields["三法人近20日推估成本帶"] == (
        "無法估算（期間內沒有三法人合計正淨買超）"
    )
    assert fields["訊號日收盤價"] == "88.5"
    assert fields["三法人淨買超日數"] == "0／20 個交易日"
    assert "收盤價相對推估成本" not in fields

