from __future__ import annotations

import argparse
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from research.institutional_model.config import PROJECT_ROOT, ResearchSettings
from research.institutional_model.database import ResearchDatabase
from research.institutional_model.downloader import (
    download_global_reference_data,
    download_phase1_data,
)
from research.institutional_model.finmind_client import (
    FinMindQuotaExceeded,
    FinMindResearchClient,
)
from research.institutional_model.label_builder import build_labels
from research.institutional_model.official_metadata import (
    OfficialMarketMetadataClient,
    OfficialMetadataError,
    refresh_official_metadata,
)
from research.institutional_model.phase2_downloader import (
    download_phase2_batch,
    get_phase2_progress,
)
from research.institutional_model.phase2_report import export_phase2_reports
from research.institutional_model.phase2_universe import build_phase2_universe
from research.institutional_model.phase3_audit import run_phase3_quality_audit
from research.institutional_model.phase3_dataset import build_phase3_dataset
from research.institutional_model.phase3_label_repair import (
    create_phase3c_validation_archive,
    repair_phase3_labels,
)
from research.institutional_model.phase3_report import export_phase3_reports
from research.institutional_model.phase4_horizon import (
    Phase4DHorizonSettings,
    run_phase4d_horizon_research,
)
from research.institutional_model.phase4_model import (
    Phase4Settings,
    run_phase4a_rolling_baseline,
)
from research.institutional_model.phase4_selection import (
    Phase4CSettings,
    run_phase4c_selection_validation,
)
from research.institutional_model.phase4_stability import (
    run_phase4b_stability_research,
)
from research.institutional_model.phase4_target import (
    Phase4ETargetSettings,
    run_phase4e_target_research,
)
from research.institutional_model.phase4_lifecycle import (
    Phase4FLifecycleSettings,
    run_phase4f_lifecycle_research,
)
from research.institutional_model.phase5_daily_reference import (
    Phase5BSettings,
    run_phase5b_daily_reference,
)
from research.institutional_model.phase5_selection_index import (
    Phase5ASettings,
    run_phase5a_selection_index,
)
from research.institutional_model.phase5_final_model import (
    Phase5DSettings,
    run_phase5d_final_model,
)
from research.institutional_model.phase5_entry_risk import (
    Phase5ISettings,
    run_phase5i_entry_risk_research,
)
from research.institutional_model.self_check import run_self_check
from research.institutional_model.universe import (
    compare_with_current_holdings,
    load_validation_universe,
    sync_universe,
)
from research.institutional_model.validation_report import export_validation_reports


PHASE1_COMMANDS = {"init", "download", "labels", "report", "phase1"}
PHASE2_COMMANDS = {
    "phase2-universe",
    "phase2-download",
    "phase2-report",
    "phase2",
}
PHASE3_COMMANDS = {"phase3", "phase3-audit", "phase3-label-repair"}
PHASE4_COMMANDS = {"phase4a", "phase4b", "phase4c", "phase4d", "phase4e", "phase4f"}
PHASE5_COMMANDS = {"phase5a", "phase5b", "phase5d", "phase5i"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="法人佈局模型本機研究工具")
    parser.add_argument(
        "command",
        choices=(
            "init",
            "download",
            "labels",
            "report",
            "phase1",
            "self-check",
            "phase2-universe",
            "phase2-download",
            "phase2-report",
            "phase2",
            "phase3",
            "phase3-audit",
            "phase3-label-repair",
            "phase4a",
            "phase4b",
            "phase4c",
            "phase4d",
            "phase4e",
            "phase4f",
            "phase5a",
            "phase5b",
            "phase5d",
            "phase5i",
        ),
    )
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--db", type=Path)
    parser.add_argument("--universe", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--symbols", help="逗號分隔，只處理指定股票")
    parser.add_argument("--force", action="store_true", help="忽略下載完成紀錄重新抓取")
    parser.add_argument(
        "--max-stocks",
        type=int,
        default=100,
        help="Phase 2 每個批次最多處理幾檔；0 代表不限制",
    )
    parser.add_argument(
        "--market",
        choices=("all", "twse", "tpex", "unknown"),
        default="all",
        help="Phase 2 限制下載市場",
    )
    parser.add_argument(
        "--include-unclassified",
        action="store_true",
        help="Phase 2 同時下載尚未確認市場別的歷史下市櫃候選",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Phase 2 任一股票下載失敗時立即停止",
    )
    parser.add_argument(
        "--skip-official",
        action="store_true",
        help="不重新抓取 TWSE／TPEx 官方公司基本資料",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Phase 2 自動重複批次，直到完成或遇到無法前進的錯誤",
    )
    parser.add_argument(
        "--quota-wait-minutes",
        type=int,
        default=65,
        help="額度用完後自動等待幾分鐘再繼續；0 代表不等待",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="單次執行最多跑幾個批次；0 代表不限",
    )
    parser.add_argument(
        "--phase3-max-stocks",
        type=int,
        default=0,
        help="Phase 3 本次最多建立幾檔；0 代表全部，重跑可續接",
    )
    parser.add_argument(
        "--audit-chunk-size",
        type=int,
        default=100_000,
        help="Phase 3B 每次讀取訓練檔的列數",
    )
    parser.add_argument(
        "--audit-sample-size",
        type=int,
        default=100_000,
        help="Phase 3B 每個市場用於分位數與相關性的抽樣列數",
    )
    parser.add_argument(
        "--audit-correlation-threshold",
        type=float,
        default=0.995,
        help="Phase 3B 高相關特徵警示門檻",
    )
    parser.add_argument(
        "--phase4-market",
        choices=("all", "twse", "tpex"),
        default="all",
        help="Phase 4A／4B 要執行的市場",
    )
    parser.add_argument(
        "--phase4-first-test-year",
        type=int,
        default=2019,
        help="Phase 4A／4B 第一個樣本外測試年度",
    )
    parser.add_argument(
        "--phase4-epochs",
        type=int,
        default=6,
        help="Phase 4A／4B 每一折最多訓練 epoch 數",
    )
    parser.add_argument(
        "--phase4-batch-size",
        type=int,
        default=65_536,
        help="Phase 4A／4B multinomial logistic mini-batch 大小",
    )
    parser.add_argument(
        "--phase4-chunk-size",
        type=int,
        default=100_000,
        help="Phase 4A／4B 建立快取與讀取資料時的 chunk 大小",
    )
    parser.add_argument(
        "--phase4-quantile-sample-size",
        type=int,
        default=250_000,
        help="每折只從訓練期間抽樣多少列估計截尾分位數",
    )
    parser.add_argument(
        "--phase4c-min-daily-stocks",
        type=int,
        default=50,
        help="Phase 4C 同日橫斷面至少需有幾檔股票",
    )
    parser.add_argument(
        "--phase4c-bootstrap-iterations",
        type=int,
        default=2_000,
        help="Phase 4C 月份移動區塊 bootstrap 次數",
    )
    parser.add_argument(
        "--phase4c-bootstrap-block-months",
        type=int,
        default=3,
        help="Phase 4C bootstrap 每個移動區塊包含幾個月",
    )
    parser.add_argument(
        "--phase4d-min-daily-stocks",
        type=int,
        default=50,
        help="Phase 4D 同日橫斷面至少需有幾檔股票",
    )
    parser.add_argument(
        "--phase4d-label-threshold",
        type=float,
        default=0.05,
        help="Phase 4D 10／20／40 日 UP／DOWN 絕對報酬門檻",
    )
    parser.add_argument(
        "--phase4e-min-daily-stocks",
        type=int,
        default=50,
        help="Phase 4E 同日橫斷面至少需有幾檔股票",
    )
    parser.add_argument(
        "--phase4e-label-threshold",
        type=float,
        default=0.05,
        help="Phase 4E 20 日 UP／DOWN 二分類絕對報酬門檻",
    )
    parser.add_argument(
        "--phase4e-ranking-l2",
        type=float,
        default=0.001,
        help="Phase 4E 20 日同日報酬排名 ridge L2",
    )
    parser.add_argument(
        "--phase4f-min-daily-stocks",
        type=int,
        default=50,
        help="Phase 4F 輸入的每個樣本外日期至少需有幾檔股票",
    )
    parser.add_argument(
        "--phase4f-bootstrap-iterations",
        type=int,
        default=1_000,
        help="Phase 4F 確認期月份移動區塊 bootstrap 次數",
    )
    parser.add_argument(
        "--phase4f-bootstrap-block-months",
        type=int,
        default=3,
        help="Phase 4F bootstrap 每個移動區塊包含幾個月",
    )
    parser.add_argument(
        "--phase5a-signal-date",
        help="Phase 5A 指定要產生指數的訊號日；未指定則使用最新合格日期",
    )
    parser.add_argument(
        "--phase5a-training-epochs",
        type=int,
        default=3,
        help="Phase 5A 最終模型固定 epoch；必須等於 Phase 4B 最佳 epoch 中位數",
    )
    parser.add_argument(
        "--phase5a-min-daily-stocks",
        type=int,
        default=50,
        help="Phase 5A 2,000 萬母體最低同日股票數",
    )
    parser.add_argument(
        "--phase5b-as-of-date",
        help="Phase 5B 新鮮度判定日期；未指定則使用台北當地日期",
    )
    parser.add_argument(
        "--phase5b-reference-market-date",
        help="Phase 5B 明確指定最新應有的市場交易日；預設讀 SQLite market_calendar",
    )
    parser.add_argument(
        "--phase5b-min-daily-stocks",
        type=int,
        default=50,
        help="Phase 5B 2,000 萬母體最低同日股票數",
    )
    parser.add_argument(
        "--phase5b-max-trading-day-lag",
        type=int,
        default=1,
        help="Phase 5B 最多允許訊號落後幾個市場交易日",
    )
    parser.add_argument(
        "--phase5b-max-calendar-age-days",
        type=int,
        default=4,
        help="Phase 5B 最多允許訊號或市場日曆距 as-of 幾個日曆日",
    )
    parser.add_argument(
        "--phase5b-recent-date-count",
        type=int,
        default=10,
        help="Phase 5B 日期完整性診斷保留最近幾個訊號日",
    )
    parser.add_argument(
        "--phase5d-min-daily-stocks",
        type=int,
        default=50,
        help="Phase 5D 樣本外重播及最終模型同日最低股票數",
    )
    parser.add_argument(
        "--phase5d-ranking-l2",
        type=float,
        default=0.001,
        help="Phase 5D 最終 return-rank ridge L2",
    )
    parser.add_argument(
        "--phase5d-bootstrap-iterations",
        type=int,
        default=1_000,
        help="Phase 5D 最終生命週期重播 bootstrap 次數",
    )
    parser.add_argument(
        "--phase5d-bootstrap-block-months",
        type=int,
        default=3,
        help="Phase 5D bootstrap 移動區塊月份",
    )
    parser.add_argument(
        "--phase5i-min-proxy-events",
        type=int,
        default=100,
        help="Phase 5I 每個成本代理進行確認期判斷所需最低事件數",
    )
    parser.add_argument(
        "--phase5i-bootstrap-iterations",
        type=int,
        default=1_000,
        help="Phase 5I 高低成本偏離分組 bootstrap 次數",
    )
    parser.add_argument(
        "--phase5i-bootstrap-block-months",
        type=int,
        default=3,
        help="Phase 5I bootstrap 移動區塊月份",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    default_start = (
        "2015-01-01"
        if args.command in PHASE2_COMMANDS | PHASE3_COMMANDS | PHASE4_COMMANDS | PHASE5_COMMANDS
        else "2019-01-01"
    )
    settings = ResearchSettings(start_date=args.start_date or default_start)
    if args.end_date:
        settings = replace(settings, end_date=args.end_date)
    if args.db:
        settings = replace(settings, database_path=args.db)
    if args.universe:
        settings = replace(settings, universe_path=args.universe)
    if args.output_dir:
        settings = replace(settings, output_dir=args.output_dir)
    settings.ensure_directories()

    if args.command == "self-check":
        run_self_check()
        return

    database = ResearchDatabase(settings.database_path)
    database.initialize()

    if args.command in PHASE2_COMMANDS | PHASE3_COMMANDS | PHASE4_COMMANDS:
        settings = _resolve_phase2_backfill_end_date(args, settings, database)

    if args.command in PHASE1_COMMANDS:
        _run_phase1(args, settings, database)
        return
    if args.command in PHASE2_COMMANDS:
        _run_phase2(args, settings, database)
        return
    if args.command == "phase3-audit":
        _run_phase3_audit(args, settings, database)
        return
    if args.command == "phase3-label-repair":
        _run_phase3_label_repair(args, settings, database)
        return
    if args.command == "phase4a":
        _run_phase4a(args, settings)
        return
    if args.command == "phase4b":
        _run_phase4b(args, settings)
        return
    if args.command == "phase4c":
        _run_phase4c(args, settings)
        return
    if args.command == "phase4d":
        _run_phase4d(args, settings, database)
        return
    if args.command == "phase4e":
        _run_phase4e(args, settings, database)
        return
    if args.command == "phase4f":
        _run_phase4f(args, settings)
        return
    if args.command == "phase5a":
        _run_phase5a(args, settings)
        return
    if args.command == "phase5b":
        _run_phase5b(args, settings, database)
        return
    if args.command == "phase5d":
        _run_phase5d(args, settings)
        return
    if args.command == "phase5i":
        _run_phase5i(args, settings, database)
        return

    _run_phase3(args, settings, database)


PHASE2_END_DATE_METADATA_KEY = "phase2_backfill_end_date"


def _resolve_phase2_backfill_end_date(
    args: argparse.Namespace,
    settings: ResearchSettings,
    database: ResearchDatabase,
) -> ResearchSettings:
    """Keep the original backfill end date stable across restarts."""
    stored_end_date = database.get_metadata(PHASE2_END_DATE_METADATA_KEY)

    if args.end_date:
        database.set_metadata(PHASE2_END_DATE_METADATA_KEY, settings.end_date)
        print(f"Phase 2 回補截止日已指定並保存：{settings.end_date}")
        return settings

    if stored_end_date:
        if stored_end_date != settings.end_date:
            print(
                "Phase 2 沿用首次執行的回補截止日："
                f"{stored_end_date}（避免跨日重跑已完成資料）"
            )
        return replace(settings, end_date=stored_end_date)

    previous_end_date = database.scalar(
        """
        SELECT MAX(requested_end)
        FROM download_status
        WHERE stock_id <> '*' AND status='complete'
        """
    )
    if previous_end_date:
        inferred_end_date = str(previous_end_date)
        database.set_metadata(PHASE2_END_DATE_METADATA_KEY, inferred_end_date)
        print(
            "Phase 2 已從既有下載進度還原回補截止日："
            f"{inferred_end_date}"
        )
        return replace(settings, end_date=inferred_end_date)

    database.set_metadata(PHASE2_END_DATE_METADATA_KEY, settings.end_date)
    print(f"Phase 2 已保存本次回補截止日：{settings.end_date}")
    return settings


def _run_phase1(
    args: argparse.Namespace,
    settings: ResearchSettings,
    database: ResearchDatabase,
) -> None:
    rows = load_validation_universe(settings.universe_path)
    sync_universe(database, rows)
    extra, missing = compare_with_current_holdings(rows, PROJECT_ROOT / "configs/users")
    if extra:
        print(f"提醒：目前配置新增持股但尚未列入 v1 驗證清單：{sorted(extra)}")
    if missing:
        print(f"提醒：v1 持股清單中已有股票不在目前啟用配置：{sorted(missing)}")

    symbols = [row["stock_id"] for row in rows]
    if args.symbols:
        requested = {item.strip() for item in args.symbols.split(",") if item.strip()}
        unknown = requested.difference(symbols)
        if unknown:
            raise SystemExit(f"指定股票不在驗證清單：{sorted(unknown)}")
        symbols = [symbol for symbol in symbols if symbol in requested]

    if args.command == "init":
        print(f"SQLite 已初始化：{settings.database_path}")
        return

    if args.command in {"download", "phase1"}:
        client = _finmind_client(settings)
        download_phase1_data(
            database=database,
            client=client,
            symbols=symbols,
            start_date=settings.start_date,
            end_date=settings.end_date,
            force=args.force,
        )

    if args.command in {"labels", "phase1"}:
        build_labels(
            database=database,
            symbols=symbols,
            horizons=settings.horizons,
            primary_horizon=settings.primary_horizon,
            threshold=settings.label_threshold,
        )

    if args.command in {"report", "phase1"}:
        paths = export_validation_reports(
            database=database,
            output_dir=settings.output_dir,
            primary_horizon=settings.primary_horizon,
        )
        _print_paths(paths)


def _run_phase2(
    args: argparse.Namespace,
    settings: ResearchSettings,
    database: ResearchDatabase,
) -> None:
    if args.command in {"phase2-universe", "phase2"}:
        client = _finmind_client(settings)
        while True:
            try:
                download_global_reference_data(
                    database,
                    client,
                    settings.start_date,
                    settings.end_date,
                    args.force,
                )
                break
            except FinMindQuotaExceeded as exc:
                print(f"FinMind API 額度已用完（全市場參考資料階段）：{exc}")
                if not args.continuous or args.quota_wait_minutes <= 0:
                    print("下次重跑會沿用 SQLite 進度，重新嘗試尚未完成的參考資料。")
                    return
                _wait_for_quota(args.quota_wait_minutes)
                client = _finmind_client(settings)
        if not args.skip_official:
            try:
                counts = refresh_official_metadata(
                    database, OfficialMarketMetadataClient()
                )
                print(f"官方公司資料已更新：{counts}")
            except OfficialMetadataError as exc:
                print(f"提醒：官方公司資料更新失敗，沿用 SQLite 既有資料：{exc}")
        _ensure_official_company_info(database)
        counts = build_phase2_universe(
            database=database,
            start_date=settings.start_date,
            end_date=settings.end_date,
            overrides_path=settings.market_overrides_path,
        )
        print(f"Phase 2 股票母體已建立：{counts}")

    if args.command in {"phase2-download", "phase2"}:
        universe_count = database.scalar(
            "SELECT COUNT(*) FROM model_universe WHERE download_enabled=1"
        )
        if not universe_count:
            raise SystemExit("model_universe 尚未建立，請先執行 phase2-universe")
        _run_phase2_download_loop(args, settings, database)
        _ensure_official_company_info(database)
        counts = build_phase2_universe(
            database=database,
            start_date=settings.start_date,
            end_date=settings.end_date,
            overrides_path=settings.market_overrides_path,
        )
        print(f"Phase 2 回補後母體已重建：{counts}")

    if args.command in {"phase2-report", "phase2"}:
        paths = export_phase2_reports(
            database=database,
            output_dir=settings.output_dir,
            start_date=settings.start_date,
            end_date=settings.end_date,
            include_unclassified=args.include_unclassified,
        )
        _print_paths(paths)



def _ensure_official_company_info(database: ResearchDatabase) -> None:
    counts = {
        market: int(
            database.scalar(
                "SELECT COUNT(*) FROM official_company_info WHERE market_type=?",
                (market,),
            )
            or 0
        )
        for market in ("twse", "tpex")
    }
    missing = [market for market, count in counts.items() if count == 0]
    if missing:
        raise SystemExit(
            "官方公司基本資料不完整，為避免錯誤重建全市場母體已停止："
            f"缺少 {missing}；目前筆數 {counts}"
        )


def _run_phase2_download_loop(
    args: argparse.Namespace,
    settings: ResearchSettings,
    database: ResearchDatabase,
) -> None:
    client = _finmind_client(settings)
    batch_number = 0

    while True:
        progress_before = get_phase2_progress(
            database=database,
            start_date=settings.start_date,
            end_date=settings.end_date,
            include_unclassified=args.include_unclassified,
        )
        _print_phase2_progress(progress_before, client)
        if progress_before.remaining_requests == 0:
            print("Phase 2 全市場回補已完成。")
            return
        if args.max_batches > 0 and batch_number >= args.max_batches:
            print(f"已達本次設定的最大批次數：{args.max_batches}")
            return

        batch_number += 1
        started_at = datetime.now().isoformat(timespec="seconds")
        request_count_before = client.request_count
        print(f"\n========== Phase 2 批次 {batch_number} 開始 ==========")
        result = download_phase2_batch(
            database=database,
            client=client,
            start_date=settings.start_date,
            end_date=settings.end_date,
            max_stocks=args.max_stocks,
            market=args.market,
            include_unclassified=args.include_unclassified,
            force=args.force,
            stop_on_error=args.stop_on_error,
            overall_completed_before=progress_before.completed_stocks,
            overall_total_stocks=progress_before.total_stocks,
            remaining_requests_before=progress_before.remaining_requests,
        )
        finished_at = datetime.now().isoformat(timespec="seconds")
        progress_after = get_phase2_progress(
            database=database,
            start_date=settings.start_date,
            end_date=settings.end_date,
            include_unclassified=args.include_unclassified,
        )
        requests_made = client.request_count - request_count_before
        database.record_phase2_batch(
            started_at=started_at,
            finished_at=finished_at,
            selected_stocks=result.selected_stocks,
            completed_stocks=result.completed_stocks,
            failed_stocks=result.failed_stocks,
            skipped_stocks=result.skipped_stocks,
            quota_exhausted=result.quota_exhausted,
            requests_made=requests_made,
            completed_stocks_after=progress_after.completed_stocks,
            remaining_stocks_after=progress_after.remaining_stocks,
            remaining_requests_after=progress_after.remaining_requests,
        )

        print(
            "本批次完成："
            f"選取 {result.selected_stocks}、完成 {result.completed_stocks}、"
            f"失敗 {result.failed_stocks}、略過 {result.skipped_stocks}、"
            f"API 請求 {requests_made}、額度停止 {result.quota_exhausted}"
        )
        _print_phase2_progress(progress_after, client)
        paths = export_phase2_reports(
            database=database,
            output_dir=settings.output_dir,
            start_date=settings.start_date,
            end_date=settings.end_date,
            include_unclassified=args.include_unclassified,
        )
        _print_paths(paths)

        if not args.continuous:
            return
        if progress_after.remaining_requests == 0:
            print("Phase 2 全市場回補已完成。")
            return
        if result.quota_exhausted:
            if args.quota_wait_minutes <= 0:
                print("額度已滿且未啟用自動等待；下次重跑會從 SQLite 進度續傳。")
                return
            _wait_for_quota(args.quota_wait_minutes)
            client = _finmind_client(settings)
            continue
        if result.completed_stocks == 0 and result.failed_stocks > 0:
            print("本批次沒有任何股票完成，為避免無限重試已停止。")
            return
        if result.selected_stocks == 0:
            print("沒有可繼續下載的股票，請查看 phase2_download_progress.csv。")
            return


def _print_phase2_progress(progress, client: FinMindResearchClient) -> None:
    stock_pct = (
        progress.completed_stocks / progress.total_stocks * 100
        if progress.total_stocks
        else 0.0
    )
    dataset_pct = (
        progress.completed_datasets / progress.total_datasets * 100
        if progress.total_datasets
        else 0.0
    )
    print(
        "Phase 2 總進度："
        f"股票 {progress.completed_stocks}/{progress.total_stocks} "
        f"({stock_pct:.1f}%)；"
        f"資料集 {progress.completed_datasets}/{progress.total_datasets} "
        f"({dataset_pct:.1f}%)；"
        f"待完成股票 {progress.remaining_stocks}；"
        f"預估剩餘 API {progress.remaining_requests}"
    )
    usage = client.get_api_usage()
    if usage and usage.used is not None and usage.limit is not None:
        print(
            f"FinMind API 用量：{usage.used}/{usage.limit}；"
            f"剩餘 {usage.remaining}"
        )


def _wait_for_quota(minutes: int) -> None:
    print(f"FinMind 每小時額度已滿，將自動等待 {minutes} 分鐘後續傳。")
    for remaining in range(minutes, 0, -1):
        print(f"  尚待 {remaining} 分鐘...", flush=True)
        time.sleep(60)


def _finmind_client(settings: ResearchSettings) -> FinMindResearchClient:
    if not settings.finmind_token:
        print("提醒：未設定 FINMIND_TOKEN，匿名額度不足以進行全市場回補。")
    return FinMindResearchClient(
        token=settings.finmind_token,
        min_interval_seconds=settings.request_interval_seconds,
    )


def _print_paths(paths: list[Path]) -> None:
    for path in paths:
        print(f"已產生：{path}")


def _run_phase3(
    args: argparse.Namespace,
    settings: ResearchSettings,
    database: ResearchDatabase,
) -> None:
    _ensure_official_company_info(database)
    counts = build_phase2_universe(
        database=database,
        start_date=settings.start_date,
        end_date=settings.end_date,
        overrides_path=settings.market_overrides_path,
    )
    print(f"Phase 3 前已套用研究截止日邊界：{counts}")

    requested_symbols = None
    if args.symbols:
        requested_symbols = [
            value.strip() for value in args.symbols.split(",") if value.strip()
        ]

    result = build_phase3_dataset(
        database=database,
        start_date=settings.start_date,
        end_date=settings.end_date,
        output_dir=settings.output_dir,
        shard_root=settings.database_path.parent / "phase3_shards",
        symbols=requested_symbols,
        max_stocks=args.phase3_max_stocks,
        force=args.force,
    )
    print(
        "Phase 3A 完成："
        f"母體 {result.total_stocks}、本次完成 {result.completed_stocks}、"
        f"沿用 {result.skipped_stocks}、失敗 {result.failed_stocks}、"
        f"待完成 {result.pending_stocks}；"
        f"TWSE 訓練列 {result.merged_rows_twse}、"
        f"TPEx 訓練列 {result.merged_rows_tpex}"
    )
    paths = export_phase3_reports(
        database=database,
        output_dir=settings.output_dir,
        config_signature=result.config_signature,
        start_date=settings.start_date,
        end_date=settings.end_date,
    )
    _print_paths([*result.output_paths, *paths])


def _run_phase3_label_repair(
    args: argparse.Namespace,
    settings: ResearchSettings,
    database: ResearchDatabase,
) -> None:
    repair = repair_phase3_labels(
        database=database,
        output_dir=settings.output_dir,
        shard_root=settings.database_path.parent / "phase3_shards",
        start_date=settings.start_date,
        end_date=settings.end_date,
    )
    print(
        "Phase 3C 標籤修正完成："
        f"母體 {repair.total_stocks}、本次修正 {repair.repaired_stocks}、"
        f"沿用 {repair.skipped_stocks}、失敗 {repair.failed_stocks}、"
        f"累計變更標籤 {repair.changed_rows}；"
        f"TWSE 訓練列 {repair.merged_rows_twse}、"
        f"TPEx 訓練列 {repair.merged_rows_tpex}"
    )

    phase3_paths = export_phase3_reports(
        database=database,
        output_dir=settings.output_dir,
        config_signature=repair.config_signature,
        start_date=settings.start_date,
        end_date=settings.end_date,
    )
    audit = run_phase3_quality_audit(
        database=database,
        output_dir=settings.output_dir,
        chunk_size=args.audit_chunk_size,
        sample_size=args.audit_sample_size,
        correlation_threshold=args.audit_correlation_threshold,
    )
    print(
        "Phase 3B 已自動重跑："
        f"狀態 {audit.status}、資料列 {audit.total_rows}、"
        f"錯誤 {audit.error_count}、警告 {audit.warning_count}、"
        f"可進模型 {int(audit.ready_for_modeling)}"
    )
    archive = create_phase3c_validation_archive(
        output_dir=settings.output_dir,
        paths=[
            settings.output_dir / "phase3c_label_repair_summary.csv",
            settings.output_dir / "phase3_validation_reports.zip",
            settings.output_dir / "phase3b_validation_reports.zip",
        ],
    )
    _print_paths([*repair.output_paths, *phase3_paths, *audit.output_paths, archive])
    if not audit.ready_for_modeling:
        raise SystemExit(
            "Phase 3C 已完成修正，但 Phase 3B 稽核仍未通過；"
            "請上傳 phase3c_validation_reports.zip。"
        )


def _run_phase4a(
    args: argparse.Namespace,
    settings: ResearchSettings,
) -> None:
    markets = ("twse", "tpex") if args.phase4_market == "all" else (args.phase4_market,)
    phase4_settings = Phase4Settings(
        first_test_year=args.phase4_first_test_year,
        maximum_epochs=args.phase4_epochs,
        batch_size=args.phase4_batch_size,
        cache_chunk_size=args.phase4_chunk_size,
        training_chunk_size=args.phase4_chunk_size,
        quantile_sample_size=args.phase4_quantile_sample_size,
    )
    result = run_phase4a_rolling_baseline(
        output_dir=settings.output_dir,
        cache_root=settings.database_path.parent / "phase4_cache",
        run_root=settings.database_path.parent / "phase4_runs",
        settings=phase4_settings,
        markets=markets,
        force=args.force,
    )
    print(
        "Phase 4A 時間滾動基準模型完成："
        f"狀態 {result.status}、完成折 {result.completed_folds}/"
        f"{result.expected_folds}、失敗 {result.failed_folds}、"
        f"可進 Phase 4B {int(result.ready_for_phase4b)}"
    )
    _print_paths(list(result.output_paths))
    if result.status != "PASS":
        raise SystemExit(
            "Phase 4A 尚有未完成折；重新執行相同腳本會沿用已完成結果。"
        )


def _run_phase4b(
    args: argparse.Namespace,
    settings: ResearchSettings,
) -> None:
    markets = ("twse", "tpex") if args.phase4_market == "all" else (args.phase4_market,)
    phase4_settings = Phase4Settings(
        first_test_year=args.phase4_first_test_year,
        maximum_epochs=args.phase4_epochs,
        batch_size=args.phase4_batch_size,
        cache_chunk_size=args.phase4_chunk_size,
        training_chunk_size=args.phase4_chunk_size,
        quantile_sample_size=args.phase4_quantile_sample_size,
    )
    result = run_phase4b_stability_research(
        output_dir=settings.output_dir,
        cache_root=settings.database_path.parent / "phase4_cache",
        phase4a_run_root=settings.database_path.parent / "phase4_runs",
        run_root=settings.database_path.parent / "phase4b_runs",
        settings=phase4_settings,
        markets=markets,
        force=args.force,
    )
    print(
        "Phase 4B 穩定化研究完成："
        f"狀態 {result.status}、候選折 {result.completed_candidate_folds}/"
        f"{result.expected_candidate_folds}、失敗 {result.failed_candidate_folds}、"
        f"可進 Phase 4C {int(result.ready_for_phase4c)}"
    )
    _print_paths(list(result.output_paths))
    if result.status != "PASS":
        raise SystemExit(
            "Phase 4B 尚有未完成候選折；重新執行相同腳本會沿用已完成結果。"
        )


def _run_phase4c(
    args: argparse.Namespace,
    settings: ResearchSettings,
) -> None:
    if args.phase4_market == "twse":
        raise SystemExit("Phase 4C 目前固定只驗證 TPEx 排序模型。")
    phase4c_settings = Phase4CSettings(
        chunk_size=args.phase4_chunk_size,
        minimum_daily_stocks=args.phase4c_min_daily_stocks,
        bootstrap_iterations=args.phase4c_bootstrap_iterations,
        bootstrap_block_months=args.phase4c_bootstrap_block_months,
    )
    result = run_phase4c_selection_validation(
        output_dir=settings.output_dir,
        shard_root=settings.database_path.parent / "phase3_shards",
        settings=phase4c_settings,
    )
    print(
        "Phase 4C TPEx 同日排序驗證完成："
        f"狀態 {result.status}、樣本 {result.scored_rows:,}、"
        f"訊號日 {result.scored_dates:,}、"
        f"可建立選股指數 {int(result.ready_for_selection_index)}"
    )
    _print_paths(list(result.output_paths))


def _run_phase4d(
    args: argparse.Namespace,
    settings: ResearchSettings,
    database: ResearchDatabase,
) -> None:
    if args.phase4_market == "twse":
        raise SystemExit("Phase 4D 目前固定只研究 TPEx 的 10／20／40 日期限。")
    phase4d_settings = Phase4DHorizonSettings(
        label_threshold=args.phase4d_label_threshold,
        minimum_daily_stocks=args.phase4d_min_daily_stocks,
        first_test_year=args.phase4_first_test_year,
        quantile_sample_size=args.phase4_quantile_sample_size,
        training_chunk_size=args.phase4_chunk_size,
        batch_size=args.phase4_batch_size,
        maximum_epochs=args.phase4_epochs,
    )
    result = run_phase4d_horizon_research(
        database=database,
        output_dir=settings.output_dir,
        shard_root=settings.database_path.parent / "phase3_shards",
        cache_root=settings.database_path.parent / "phase4d_cache",
        settings=phase4d_settings,
        force=args.force,
    )
    print(
        "Phase 4D TPEx 持有期研究完成："
        f"狀態 {result.status}、完成折 {result.completed_folds}/"
        f"{result.expected_folds}、失敗 {result.failed_folds}、"
        f"可進行期限決策 {int(result.ready_for_horizon_decision)}"
    )
    _print_paths(list(result.output_paths))
    if result.status != "PASS":
        raise SystemExit("Phase 4D 尚有未完成折；請先檢查 phase4d_fold_summary.csv。")


def _run_phase4e(
    args: argparse.Namespace,
    settings: ResearchSettings,
    database: ResearchDatabase,
) -> None:
    if args.phase4_market == "twse":
        raise SystemExit("Phase 4E 目前固定研究 TPEx 的 20 日主模型目標。")
    phase4e_settings = Phase4ETargetSettings(
        label_threshold=args.phase4e_label_threshold,
        minimum_daily_stocks=args.phase4e_min_daily_stocks,
        first_test_year=args.phase4_first_test_year,
        quantile_sample_size=args.phase4_quantile_sample_size,
        training_chunk_size=args.phase4_chunk_size,
        batch_size=args.phase4_batch_size,
        maximum_epochs=args.phase4_epochs,
        ranking_l2_penalty=args.phase4e_ranking_l2,
        bootstrap_iterations=args.phase4c_bootstrap_iterations,
        bootstrap_block_months=args.phase4c_bootstrap_block_months,
    )
    result = run_phase4e_target_research(
        database=database,
        output_dir=settings.output_dir,
        shard_root=settings.database_path.parent / "phase3_shards",
        cache_root=settings.database_path.parent / "phase4d_cache",
        settings=phase4e_settings,
        force=args.force,
    )
    print(
        "Phase 4E TPEx 20 日模型目標研究完成："
        f"狀態 {result.status}、完成折 {result.completed_folds}/"
        f"{result.expected_folds}、失敗 {result.failed_folds}、"
        f"可進行模型目標決策 {int(result.ready_for_target_decision)}"
    )
    _print_paths(list(result.output_paths))
    if result.status != "PASS":
        raise SystemExit("Phase 4E 尚有未完成折；請先檢查 phase4e_fold_summary.csv。")


def _run_phase4f(
    args: argparse.Namespace,
    settings: ResearchSettings,
) -> None:
    if args.phase4_market == "twse":
        raise SystemExit("Phase 4F 目前固定研究 TPEx return rank 訊號生命週期。")
    phase4f_settings = Phase4FLifecycleSettings(
        minimum_daily_stocks=args.phase4f_min_daily_stocks,
        bootstrap_iterations=args.phase4f_bootstrap_iterations,
        bootstrap_block_months=args.phase4f_bootstrap_block_months,
    )
    result = run_phase4f_lifecycle_research(
        output_dir=settings.output_dir,
        settings=phase4f_settings,
    )
    print(
        "Phase 4F TPEx 法人布局訊號生命週期研究完成："
        f"狀態 {result.status}、來源樣本 {result.source_rows:,}、"
        f"事件 {result.event_rows:,}、"
        f"可進行通知規則決策 {int(result.ready_for_lifecycle_decision)}"
    )
    _print_paths(list(result.output_paths))


def _run_phase5a(
    args: argparse.Namespace,
    settings: ResearchSettings,
) -> None:
    phase5a_settings = Phase5ASettings(
        chunk_size=args.phase4_chunk_size,
        quantile_sample_size=args.phase4_quantile_sample_size,
        batch_size=args.phase4_batch_size,
        training_epochs=args.phase5a_training_epochs,
        minimum_daily_stocks=args.phase5a_min_daily_stocks,
    )
    result = run_phase5a_selection_index(
        output_dir=settings.output_dir,
        cache_root=settings.database_path.parent / "phase4_cache",
        shard_root=settings.database_path.parent / "phase3_shards",
        model_root=settings.database_path.parent / "phase5_models",
        settings=phase5a_settings,
        signal_date=args.phase5a_signal_date,
        force=args.force,
    )
    print(
        "Phase 5A TPEx 本機選股指數完成："
        f"狀態 {result.status}、訊號日 {result.signal_date}、"
        f"候選股票 {result.selected_rows:,}"
    )
    _print_paths(list(result.output_paths))


def _run_phase5b(
    args: argparse.Namespace,
    settings: ResearchSettings,
    database: ResearchDatabase,
) -> None:
    model_settings = Phase5ASettings(
        chunk_size=args.phase4_chunk_size,
        quantile_sample_size=args.phase4_quantile_sample_size,
        batch_size=args.phase4_batch_size,
        training_epochs=args.phase5a_training_epochs,
        minimum_daily_stocks=args.phase5a_min_daily_stocks,
    )
    phase5b_settings = Phase5BSettings(
        chunk_size=args.phase4_chunk_size,
        minimum_daily_stocks=args.phase5b_min_daily_stocks,
        recent_rows_per_stock=model_settings.recent_rows_per_stock,
        recent_date_count=args.phase5b_recent_date_count,
        maximum_trading_day_lag=args.phase5b_max_trading_day_lag,
        maximum_calendar_age_days=args.phase5b_max_calendar_age_days,
    )
    result = run_phase5b_daily_reference(
        database=database,
        output_dir=settings.output_dir,
        shard_root=settings.database_path.parent / "phase3_shards",
        model_root=settings.database_path.parent / "phase5_models",
        model_settings=model_settings,
        settings=phase5b_settings,
        as_of_date=args.phase5b_as_of_date,
        reference_market_date=args.phase5b_reference_market_date,
    )
    print(
        "Phase 5B TPEx 每日選股參考完成："
        f"狀態 {result.status}、訊號日 {result.signal_date}、"
        f"可用股票 {result.selected_rows:,}"
    )
    _print_paths(list(result.output_paths))


def _run_phase5d(
    args: argparse.Namespace,
    settings: ResearchSettings,
) -> None:
    phase5d_settings = Phase5DSettings(
        minimum_daily_stocks=args.phase5d_min_daily_stocks,
        quantile_sample_size=args.phase4_quantile_sample_size,
        ranking_l2_penalty=args.phase5d_ranking_l2,
        bootstrap_iterations=args.phase5d_bootstrap_iterations,
        bootstrap_block_months=args.phase5d_bootstrap_block_months,
    )
    result = run_phase5d_final_model(
        output_dir=settings.output_dir,
        shard_root=settings.database_path.parent / "phase3_shards",
        model_root=settings.database_path.parent / "phase5_models",
        settings=phase5d_settings,
        force=args.force,
    )
    print(
        "Phase 5D TPEx 最終法人模型與通知重播完成："
        f"狀態 {result.status}、成熟訓練列 {result.training_rows:,}、"
        f"重播事件 {result.replay_events:,}、通知 {result.replay_notifications:,}"
    )
    _print_paths(list(result.output_paths))



def _run_phase5i(
    args: argparse.Namespace,
    settings: ResearchSettings,
    database: ResearchDatabase,
) -> None:
    phase5i_settings = Phase5ISettings(
        minimum_proxy_events=args.phase5i_min_proxy_events,
        bootstrap_iterations=args.phase5i_bootstrap_iterations,
        bootstrap_block_months=args.phase5i_bootstrap_block_months,
    )
    result = run_phase5i_entry_risk_research(
        database=database,
        output_dir=settings.output_dir,
        shard_root=settings.database_path.parent / "phase3_shards",
        settings=phase5i_settings,
    )
    print(
        "Phase 5I 法人推估成本帶與追價風險驗證完成："
        f"狀態 {result.status}、正式事件 {result.source_events:,}、"
        f"完成成本特徵 {result.enriched_events:,}"
    )
    _print_paths(list(result.output_paths))

def _run_phase3_audit(
    args: argparse.Namespace,
    settings: ResearchSettings,
    database: ResearchDatabase,
) -> None:
    result = run_phase3_quality_audit(
        database=database,
        output_dir=settings.output_dir,
        chunk_size=args.audit_chunk_size,
        sample_size=args.audit_sample_size,
        correlation_threshold=args.audit_correlation_threshold,
    )
    print(
        "Phase 3B 特徵品質稽核完成："
        f"狀態 {result.status}、資料列 {result.total_rows}、"
        f"錯誤 {result.error_count}、警告 {result.warning_count}、"
        f"可進模型 {int(result.ready_for_modeling)}"
    )
    _print_paths(list(result.output_paths))

if __name__ == "__main__":
    main()

