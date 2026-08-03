from __future__ import annotations

import json
import os
from typing import Any

from src.models import HoldingMonitorResult, InstitutionalNotification
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

    def send_profit_alert(
        self,
        result: HoldingMonitorResult,
        webhook_url: str,
        *,
        force: bool = False,
    ) -> None:
        if result.quote is None or result.profit_return_pct is None:
            raise ValueError("缺少有效報價，無法發送損益通知")
        if result.new_profit_thresholds:
            threshold_text = "、".join(
                f"+{threshold:g}%" for threshold in result.new_profit_thresholds
            )
            description = f"價格報酬首次跨越本輪門檻：{threshold_text}"
            icon = "💰"
        elif force:
            description = "手動要求輸出目前持股損益"
            icon = "📋"
        else:
            raise ValueError("沒有新門檻且未強制通知")

        fields: list[dict[str, Any]] = [
            {"name": "目前價格", "value": f"{result.quote.price:g}", "inline": True},
            {"name": "平均成本", "value": f"{result.average_cost:g}", "inline": True},
            {
                "name": "目前損益",
                "value": f"{result.profit_return_pct:+.1f}%",
                "inline": True,
            },
        ]
        if result.active_profit_thresholds:
            fields.append(
                {
                    "name": "目前已達門檻",
                    "value": "、".join(
                        f"+{threshold:g}%" for threshold in result.active_profit_thresholds
                    ),
                    "inline": False,
                }
            )
        embed = {
            "title": f"{icon} {result.symbol} {result.name}",
            "description": description,
            "fields": fields,
            "footer": {"text": "成本損益提醒；不代表買進、續抱或賣出建議"},
            "timestamp": result.analyzed_at,
        }
        self.http.post_json(webhook_url, payload={"embeds": [embed]})

    def send_institutional(
        self,
        notification: InstitutionalNotification,
        webhook_url: str,
        *,
        holding: HoldingMonitorResult | None = None,
    ) -> None:
        title_prefix, status_text = _institutional_status(notification.notification_type)
        market = str(notification.market or "tpex").lower()
        market_label = "TWSE 上市" if market == "twse" else "TPEx 上櫃"
        rank_label = "同日上市法人排名" if market == "twse" else "同日上櫃法人排名"
        stock_name = f" {notification.name}" if notification.name else ""
        fields: list[dict[str, Any]] = [
            {"name": "市場", "value": market_label, "inline": True},
            {"name": "法人狀態", "value": status_text, "inline": True},
            {
                "name": rank_label,
                "value": (
                    f"{notification.percentile:.1f} 百分位"
                    if notification.percentile is not None
                    else "目前無有效分數"
                ),
                "inline": True,
            },
            {"name": "用途", "value": "值得追蹤（TRACK_ONLY）", "inline": True},
        ]
        if notification.event_age_days is not None:
            fields.append(
                {
                    "name": "事件進度",
                    "value": f"第 {notification.event_age_days} 個交易日",
                    "inline": True,
                }
            )
        if notification.estimated_cost_window_days is not None:
            window = notification.estimated_cost_window_days
            if notification.estimated_cost_mid is None:
                fields.append(
                    {
                        "name": f"三法人近{window}日推估成本帶",
                        "value": "無法估算（期間內沒有三法人合計正淨買超）",
                        "inline": False,
                    }
                )
            else:
                if (
                    notification.estimated_cost_low is not None
                    and notification.estimated_cost_high is not None
                ):
                    fields.append(
                        {
                            "name": f"三法人近{window}日推估成本帶",
                            "value": (
                                f"{_price_text(notification.estimated_cost_low)} ～ "
                                f"{_price_text(notification.estimated_cost_high)}"
                            ),
                            "inline": True,
                        }
                    )
                fields.append(
                    {
                        "name": "推估成本中間值",
                        "value": _price_text(notification.estimated_cost_mid),
                        "inline": True,
                    }
                )
                if notification.signal_deviation_pct is not None:
                    fields.append(
                        {
                            "name": "收盤價相對推估成本",
                            "value": f"{notification.signal_deviation_pct:+.1f}%",
                            "inline": True,
                        }
                    )
            if notification.signal_close is not None:
                fields.append(
                    {
                        "name": "訊號日收盤價",
                        "value": _price_text(notification.signal_close),
                        "inline": True,
                    }
                )
            if notification.estimated_cost_buy_days is not None:
                fields.append(
                    {
                        "name": "三法人淨買超日數",
                        "value": (
                            f"{notification.estimated_cost_buy_days}／{window} 個交易日"
                        ),
                        "inline": True,
                    }
                )
        if holding is not None:
            if holding.quote is not None:
                fields.append(
                    {"name": "目前價格", "value": f"{holding.quote.price:g}", "inline": True}
                )
            fields.append(
                {"name": "平均成本", "value": f"{holding.average_cost:g}", "inline": True}
            )
            if holding.profit_return_pct is not None:
                fields.append(
                    {
                        "name": "目前損益",
                        "value": f"{holding.profit_return_pct:+.1f}%",
                        "inline": True,
                    }
                )
        if notification.positive_factors:
            fields.append(
                {
                    "name": "主要正向因素",
                    "value": _factor_lines(notification.positive_factors),
                    "inline": False,
                }
            )
        if notification.negative_factors:
            fields.append(
                {
                    "name": "主要負向因素",
                    "value": _factor_lines(notification.negative_factors),
                    "inline": False,
                }
            )
        embed: dict[str, Any] = {
            "title": f"{title_prefix} {notification.symbol}{stock_name}",
            "description": notification.reason or status_text,
            "fields": fields,
            "footer": {
                "text": (
                    f"法人分數是{market_label}同日相對排序；推估成本以近20日三法人合計"
                    "正淨買超股數加權日內典型價計算，非真實持倉成本；"
                    "僅供自行判斷，不是買賣建議"
                )
            },
        }
        if holding is not None:
            embed["timestamp"] = holding.analyzed_at
        self.http.post_json(webhook_url, payload={"embeds": [embed]})


def _price_text(value: float) -> str:
    return f"{value:,.2f}".rstrip("0").rstrip(".")

def _institutional_status(notification_type: str) -> tuple[str, str]:
    mapping = {
        "LAYOUT_CONFIRMED_DIRECT": ("🔥", "法人連續布局確認"),
        "NEW_CANDIDATE": ("🔎", "法人排名新進前10%"),
        "LAYOUT_CONFIRMED": ("🧭", "既有候選升級為布局確認"),
        "LAYOUT_CONFIRMED_AND_EXTENDED": ("🧭", "布局確認並延長觀察"),
        "DAY20_EXTEND": ("📌", "第20日仍在前20%，延長觀察"),
        "DAY20_EXTEND_STRONG": ("📌", "第20日排名仍高，延長觀察"),
        "TWSE_TRACK_CONFIRMED": ("🔥", "上市法人排名連續3日前10%，建立40日追蹤"),
        "TWSE_DAY40_END": ("🏁", "上市法人40日追蹤期結束"),
    }
    return mapping.get(notification_type, ("📎", notification_type))


def _factor_lines(value: str) -> str:
    parts = [part.strip() for part in value.split("；") if part.strip()]
    return "\n".join(f"• {part}" for part in parts) or "無"
