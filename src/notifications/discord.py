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
        direction = "📈" if score > 0 else "📉" if score < 0 else "⚖️"
        fields: list[dict[str, Any]] = [
            {
                "name": "操作傾向",
                "value": f"{result.operation_label} {abs(score):.0f}%",
                "inline": True,
            },
            {
                "name": "個股分析",
                "value": f"{result.objective_label} {abs(result.objective_score):.0f}%",
                "inline": True,
            },
            {
                "name": "機會 / 風險",
                "value": f"{result.opportunity_score:.0f} / {result.risk_score:.0f}",
                "inline": True,
            },
            {
                "name": "資料完整度",
                "value": f"{result.completeness:.0f}%",
                "inline": True,
            },
        ]
        if result.price_return_pct is not None:
            fields.insert(
                2,
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
            "footer": {"text": "規則分析結果，不代表未來漲跌機率或投資建議"},
            "timestamp": result.analyzed_at,
        }
        if result.report_url:
            embed["url"] = result.report_url
        self.http.post_json(webhook_url, payload={"embeds": [embed]})
