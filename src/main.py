from __future__ import annotations

import argparse
import json
import logging
import os

from src.config_loader import enabled_stocks, load_user_configs, load_yaml
from src.models import AnalysisResult
from src.notifications.discord import DiscordNotifier
from src.providers.aggregator import DataAggregator
from src.providers.finmind import FinMindProvider
from src.providers.fixture import load_fixture_bundle
from src.providers.fugle import FugleProvider
from src.providers.http import HttpClient
from src.providers.official_events import OfficialEventProvider
from src.reports.html_report import write_index, write_stock_report
from src.scoring.engine import ScoringEngine
from src.state.manager import StateManager


LOGGER = logging.getLogger("tw-stock-monitor")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="台股規則式監控與 Discord 通知")
    parser.add_argument("--users-config", default="configs/users", help="使用者 YAML 或資料夾")
    parser.add_argument("--scoring-config", default="configs/scoring.yaml")
    parser.add_argument("--state-file", default="runtime/state.json")
    parser.add_argument("--output-dir", default="site")
    parser.add_argument("--offline-fixture", help="使用離線 JSON，不呼叫外部 API")
    parser.add_argument("--no-discord", action="store_true", help="不發送 Discord")
    parser.add_argument("--force-notify", action="store_true", help="忽略通知條件，強制發送")
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
    scoring = load_yaml(args.scoring_config)
    user_configs = load_user_configs(args.users_config)
    state = StateManager(args.state_file)
    engine = ScoringEngine(scoring)
    notifier = DiscordNotifier()
    aggregator = _build_aggregator() if not args.offline_fixture else None
    results: list[AnalysisResult] = []

    for user_config in user_configs:
        user = user_config["user"]
        user_id = str(user["id"])
        app = user_config.get("app", {})
        market_symbol = str(app.get("market_symbol", "TAIEX"))
        report_base_url = (
            str(app.get("report_base_url") or os.getenv("REPORT_BASE_URL") or "").rstrip("/")
        )
        webhook = notifier.resolve_webhook(user.get("discord_webhook_key"))

        for stock in enabled_stocks(user_config):
            symbol = str(stock["symbol"])
            LOGGER.info("分析 %s/%s", user_id, symbol)
            processed_ids = state.processed_event_ids(user_id, symbol)
            if args.offline_fixture:
                bundle = load_fixture_bundle(args.offline_fixture, symbol, stock, market_symbol)
            else:
                assert aggregator is not None
                bundle = aggregator.collect(stock, market_symbol)
            result = engine.analyze(user_id, stock, bundle, processed_ids)
            relative_path = f"{user_id}/{symbol}.html"
            result.report_url = f"{report_base_url}/{relative_path}" if report_base_url else None
            report_path = write_stock_report(result, args.output_dir)
            result.report_path = str(report_path)

            decision = state.decide(result, scoring.get("notification", {}))
            if args.force_notify:
                decision.should_send = True
                decision.reasons.append("命令列強制通知")
            notified = False
            if decision.should_send:
                if args.no_discord:
                    LOGGER.info("略過 Discord：%s", "、".join(decision.reasons))
                elif not webhook:
                    LOGGER.warning("%s/%s 缺少 Discord Webhook，未發送通知", user_id, symbol)
                else:
                    notifier.send(result, decision, webhook)
                    notified = True
                    LOGGER.info("已發送 Discord：%s", "、".join(decision.reasons))
            else:
                LOGGER.info("不通知：%s", "、".join(decision.reasons) or "未跨越通知條件")

            state.update(result, notified)
            results.append(result)

    write_index(results, args.output_dir)
    state.save()
    print(
        json.dumps(
            [
                {
                    "user": item.user_id,
                    "symbol": item.symbol,
                    "operation_label": item.operation_label,
                    "operation_score": item.operation_score,
                    "objective_score": item.objective_score,
                    "completeness": item.completeness,
                    "report": item.report_path,
                }
                for item in results
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _build_aggregator() -> DataAggregator:
    http = HttpClient()
    return DataAggregator(
        fugle=FugleProvider(os.getenv("FUGLE_API_KEY"), http),
        finmind=FinMindProvider(os.getenv("FINMIND_TOKEN"), http),
        official_events=OfficialEventProvider(http),
    )


if __name__ == "__main__":
    cli()
