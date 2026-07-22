from __future__ import annotations

from src.models import AnalysisResult, NotificationDecision
from src.notifications.discord import DiscordNotifier


class FakeHttp:
    def __init__(self) -> None:
        self.payload = None

    def post_json(self, url: str, *, payload):  # noqa: ANN001, ANN201
        self.payload = payload
        return {}


def test_discord_uses_direction_score_instead_of_probability() -> None:
    http = FakeHttp()
    result = AnalysisResult(
        user_id="test",
        symbol="2330",
        name="台積電",
        market="TWSE",
        analyzed_at="2026-07-22T09:00:00+08:00",
        quote=None,
        objective_score=8,
        operation_score=8,
        opportunity_score=20,
        risk_score=12,
        completeness=80,
        objective_label="中性觀察",
        operation_label="續抱觀察",
        holding_enabled=True,
        price_return_pct=None,
        modules=[],
        holding_rules=[],
        summary="目前沒有足夠證據支持明顯加碼或減碼。",
        new_event_ids=[],
        errors=[],
    )
    DiscordNotifier(http=http).send(
        result,
        NotificationDecision(should_send=True, reasons=["命令列強制通知"]),
        "https://example.invalid/webhook",
    )
    embed = http.payload["embeds"][0]
    fields = {field["name"]: field["value"] for field in embed["fields"]}
    assert embed["title"].startswith("⚖️")
    assert fields["持倉判斷"] == "續抱觀察"
    assert fields["方向分數"] == "+8 / 100"
    assert fields["分析涵蓋度"] == "80%"
