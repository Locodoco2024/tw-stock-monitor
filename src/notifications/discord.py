from __future__ import annotations

import json
import os
from typing import Any

from src.models import AnalysisResult, NotificationDecision
from src.providers.http import HttpClient, ProviderError


class DiscordNotifier:
    def __init__(self, http: HttpClient | None = None) -> None:
        self.http = http or HttpClient()

    def resolve_webhook(self, key: str | None) -> str | None:
        mapping_raw = os.getenv("DISCORD_WEBHOOKS_JSON", "").strip()
        if mapping_raw:
            try:
                mapping = json.loads(mapping_raw)
                webhook = mapping.get(key or "default")
                if webhook:
                    return str(webhook)
            except json.JSONDecodeError as exc:
                raise ProviderError("DISCORD_WEBHOOKS_JSON 不是合法 JSON") from exc
        return os.getenv("DISCORD_WEBHOOK_URL") or None

    def send(
        self,
        result: AnalysisResult,
        decision: NotificationDecision,
        webhook_url: str,
    ) -> None:
        score = result.operation_score
        direction = "📈" if score >= 15 else "📉" if score <= -15 else "⚖️"
        operation_name = "持倉判斷" if result.holding_enabled else "觀察判斷"
        fields: list[dict[str, Any]] = [
            {
                "name": operation_name,
                "value": result.operation_label,
                "inline": True,
            },
            {
                "name": "方向分數",
                "value": _score_text(score),
                "inline": True,
            },
            {
                "name": "股票本身",
                "value": f"{result.objective_label}（{_score_text(result.objective_score)}）",
                "inline": True,
            },
            {
                "name": "機會 / 風險",
                "value": f"{result.opportunity_score:.0f} / {result.risk_score:.0f}",
                "inline": True,
            },
            {
                "name": "分析涵蓋度",
                "value": f"{result.completeness:.0f}%",
                "inline": True,
            },
        ]
        if result.price_return_pct is not None:
            fields.insert(
                3,
                {
                    "name": "價格報酬",
                    "value": f"{result.price_return_pct:+.1f}%",
                    "inline": True,
                },
            )
        if result.quote:
            fields.insert(
                0,
                {
                    "name": "目前價格",
                    "value": f"{result.quote.price:g}",
                    "inline": True,
                },
            )
        if decision.reasons:
            fields.append(
                {
                    "name": "本次通知原因",
                    "value": "\n".join(f"• {reason}" for reason in decision.reasons),
                    "inline": False,
                }
            )
        embed: dict[str, Any] = {
            "title": f"{direction} {result.symbol} {result.name}",
            "description": result.summary,
            "fields": fields,
            "footer": {"text": "方向分數是規則分析結果，不代表未來漲跌機率或投資建議"},
            "timestamp": result.analyzed_at,
        }
        if result.report_url:
            embed["url"] = result.report_url
        self.http.post_json(webhook_url, payload={"embeds": [embed]})


def _score_text(score: float) -> str:
    if score == 0:
        return "0 / 100"
    return f"{score:+.0f} / 100"
