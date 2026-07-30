from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import logging
import os
from pathlib import Path

from src.config_loader import (
    enabled_stocks,
    institutional_candidates_settings,
    load_user_configs,
)
from src.institutional.integration import dispatch_institutional_notifications
from src.models import HoldingMonitorResult, Quote
from src.notifications.discord import DiscordNotifier
from src.providers.fixture import load_fixture_quote
from src.providers.fugle import FugleProvider
from src.providers.http import HttpClient
from src.state.manager import StateManager


LOGGER = logging.getLogger("tw-stock-monitor")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="台股持股損益與三大法人候選通知")
    parser.add_argument("--users-config", default="configs/users", help="使用者 YAML 或資料夾")
    parser.add_argument("--state-file", default="runtime/state.json")
    parser.add_argument("--offline-fixture", help="使用離線 JSON，不呼叫 Fugle")
    parser.add_argument("--no-discord", action="store_true", help="不發送 Discord")
    parser.add_argument(
        "--force-notify",
        action="store_true",
        help="忽略獲利門檻是否新跨越，輸出全部配置股目前損益",
    )
    parser.add_argument(
        "--enable-institutional-candidates",
        action="store_true",
        help="啟用法人候選通知",
    )
    parser.add_argument(
        "--institutional-plan",
        default="runtime/institutional/notification_plan.csv",
        help="Phase 5H 法人通知計畫 CSV",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def cli() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        exit_code = run(args)
    except Exception:
        LOGGER.exception("執行失敗")
        exit_code = 1
    raise SystemExit(exit_code)


def run(args: argparse.Namespace) -> int:
    user_configs = load_user_configs(args.users_config)
    institutional_enabled_users = [
        str(config["user"]["id"])
        for config in user_configs
        if institutional_candidates_settings(config)["enabled"]
    ]
    if (
        args.enable_institutional_candidates
        and institutional_enabled_users
        and not Path(args.institutional_plan).exists()
    ):
        raise FileNotFoundError(f"找不到法人通知計畫：{args.institutional_plan}")

    state = StateManager(args.state_file)
    notifier = DiscordNotifier()
    provider = (
        None
        if args.offline_fixture
        else FugleProvider(os.getenv("FUGLE_API_KEY"), HttpClient())
    )
    results: list[HoldingMonitorResult] = []

    for user_config in user_configs:
        user = user_config["user"]
        user_id = str(user["id"])
        webhook = notifier.resolve_webhook(user.get("discord_webhook_key"))

        for stock in enabled_stocks(user_config):
            symbol = str(stock["symbol"])
            LOGGER.info("檢查持股損益 %s/%s", user_id, symbol)
            quote, errors = _load_quote(
                symbol=symbol,
                provider=provider,
                fixture_path=args.offline_fixture,
            )
            average_cost = float(stock["average_cost"])
            profit_return_pct = (
                (quote.price / average_cost - 1.0) * 100.0 if quote is not None else None
            )
            thresholds = [float(value) for value in stock["profit_alerts"]]
            active_thresholds = (
                [threshold for threshold in thresholds if profit_return_pct >= threshold]
                if profit_return_pct is not None
                else []
            )
            new_thresholds = (
                state.new_profit_thresholds(user_id, symbol, active_thresholds)
                if quote is not None
                else []
            )
            result = HoldingMonitorResult(
                user_id=user_id,
                symbol=symbol,
                name=quote.name if quote else symbol,
                analyzed_at=datetime.now().astimezone().isoformat(),
                quote=quote,
                average_cost=average_cost,
                profit_return_pct=profit_return_pct,
                active_profit_thresholds=active_thresholds,
                new_profit_thresholds=new_thresholds,
                errors=errors,
            )

            should_send = bool(new_thresholds) or (args.force_notify and quote is not None)
            if should_send:
                if args.no_discord:
                    LOGGER.info("略過 Discord：%s/%s", user_id, symbol)
                elif not webhook:
                    LOGGER.warning("%s/%s 缺少 Discord Webhook，未發送損益通知", user_id, symbol)
                else:
                    notifier.send_profit_alert(result, webhook, force=args.force_notify)
                    LOGGER.info(
                        "已發送損益通知：%s/%s %s",
                        user_id,
                        symbol,
                        new_thresholds or "force",
                    )
            elif quote is None:
                LOGGER.warning("%s/%s 無有效報價：%s", user_id, symbol, "；".join(errors))
            else:
                LOGGER.info("未跨越新獲利門檻：%s/%s", user_id, symbol)

            if quote is not None:
                state.update_holding(result)
            results.append(result)

    institutional_dispatch = None
    if args.enable_institutional_candidates:
        if institutional_enabled_users:
            institutional_dispatch = dispatch_institutional_notifications(
                plan_path=args.institutional_plan,
                user_configs=user_configs,
                holding_results=results,
                state=state,
                notifier=notifier,
                no_discord=args.no_discord,
            )
        else:
            LOGGER.info("法人候選全域開關已啟用，但沒有啟用中的使用者")

    state.save()
    output_payload = {
        "holdings": [
            {
                "user": item.user_id,
                "symbol": item.symbol,
                "name": item.name,
                "price": item.quote.price if item.quote else None,
                "average_cost": item.average_cost,
                "profit_return_pct": item.profit_return_pct,
                "new_profit_thresholds": item.new_profit_thresholds,
                "errors": item.errors,
            }
            for item in results
        ],
        "institutional": (
            {
                "enabled_users": institutional_enabled_users,
                "dispatch": asdict(institutional_dispatch),
            }
            if institutional_dispatch is not None
            else {
                "enabled_users": institutional_enabled_users,
                "dispatch": None,
            }
        ),
    }
    print(json.dumps(output_payload, ensure_ascii=False, indent=2))
    return 0


def _load_quote(
    *,
    symbol: str,
    provider: FugleProvider | None,
    fixture_path: str | None,
) -> tuple[Quote | None, list[str]]:
    try:
        if fixture_path:
            return load_fixture_quote(fixture_path, symbol), []
        if provider is None:
            raise RuntimeError("Fugle provider 尚未初始化")
        return provider.quote(symbol), []
    except Exception as exc:
        return None, [str(exc)]


if __name__ == "__main__":
    cli()
