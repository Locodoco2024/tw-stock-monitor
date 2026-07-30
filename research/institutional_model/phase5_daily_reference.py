from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from research.institutional_model.database import ResearchDatabase
from research.institutional_model.phase4_selection import resolve_phase3_shard_directory
from research.institutional_model.phase5_selection_index import (
    Phase5ASettings,
    add_live_liquidity_flags,
    attach_historical_behavior,
    build_historical_behavior_lookup,
    final_model_signature,
    finalize_selection_columns,
    latest_source_columns,
    load_final_model,
    rank_liquidity_universes,
    score_latest_cross_section,
    validate_phase5a_inputs,
    write_csv,
)


PHASE5B_VERSION = "phase5b-v1"
TARGET_MARKET = "tpex"
TAIPEI_TIMEZONE = ZoneInfo("Asia/Taipei")


@dataclass(frozen=True)
class Phase5BSettings:
    chunk_size: int = 100_000
    minimum_daily_stocks: int = 50
    recent_rows_per_stock: int = 90
    recent_date_count: int = 10
    maximum_trading_day_lag: int = 1
    maximum_calendar_age_days: int = 4

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("Phase 5B chunk size 必須大於 0")
        if self.minimum_daily_stocks < 20:
            raise ValueError("Phase 5B 同日股票數不可低於 20")
        if self.recent_rows_per_stock < 10:
            raise ValueError("Phase 5B 每檔保留的近期列數不可低於 10")
        if self.recent_date_count <= 0:
            raise ValueError("Phase 5B 近期日期診斷筆數必須大於 0")
        if self.maximum_trading_day_lag < 0:
            raise ValueError("Phase 5B 允許交易日落後不可小於 0")
        if self.maximum_calendar_age_days < 0:
            raise ValueError("Phase 5B 允許日曆日落後不可小於 0")


@dataclass(frozen=True)
class CrossSectionDiagnostics:
    selected: pd.DataFrame
    recent_dates: pd.DataFrame
    signal_date: str
    latest_raw_signal_date: str
    latest_feature_ok_date: str
    latest_eligible_signal_date: str
    eligible_date_count: int
    latest_shard_modified_at: str
    cross_section_source: str
    cross_section_target_date: str


@dataclass(frozen=True)
class FreshnessDecision:
    model_status: str
    data_freshness_status: str
    selection_readiness_status: str
    readiness_reason: str
    as_of_date: str
    reference_market_date: str
    reference_market_date_source: str
    calendar_reference_age_days: int | None
    calendar_age_days: int
    trading_day_lag: int | None
    latest_calendar_date: str
    latest_tpex_price_date: str
    latest_tpex_flow_date: str
    latest_common_tpex_source_date: str


@dataclass(frozen=True)
class Phase5BResult:
    status: str
    signal_date: str
    selected_rows: int
    output_paths: tuple[Path, ...]


def run_phase5b_daily_reference(
    *,
    database: ResearchDatabase,
    output_dir: Path | str,
    shard_root: Path | str,
    live_shard_root: Path | str | None = None,
    model_root: Path | str,
    model_settings: Phase5ASettings | None = None,
    settings: Phase5BSettings | None = None,
    as_of_date: str | None = None,
    reference_market_date: str | None = None,
) -> Phase5BResult:
    config = settings or Phase5BSettings()
    config.validate()
    phase5a_config = model_settings or Phase5ASettings()
    phase5a_config.validate()

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    validation = validate_phase5a_inputs(output, config=phase5a_config)
    bundle = load_existing_phase5a_model(
        model_root=Path(model_root),
        source_sha256=validation["phase3_source_sha256"],
        settings=phase5a_config,
    )
    shard_dir, stock_ids, cross_section_source, cross_section_target_date = (
        resolve_phase5_cross_section_source(
            output_dir=output,
            phase3_shard_root=Path(shard_root),
            live_shard_root=Path(live_shard_root) if live_shard_root else None,
        )
    )
    diagnostics = load_recent_tpex_cross_sections(
        output_dir=output,
        shard_dir=shard_dir,
        feature_columns=bundle.feature_columns,
        settings=config,
        stock_ids=stock_ids,
        cross_section_source=cross_section_source,
        cross_section_target_date=cross_section_target_date,
    )
    if diagnostics.signal_date <= bundle.last_training_date:
        raise RuntimeError(
            "Phase 5B 最新可用訊號日不得早於或等於最終模型訓練截止日；"
            f"訓練截止 {bundle.last_training_date}，訊號日 {diagnostics.signal_date}"
        )

    effective_as_of = resolve_iso_date(
        as_of_date,
        default=datetime.now(TAIPEI_TIMEZONE).date(),
        field_name="Phase 5B as-of date",
    )
    explicit_reference = (
        resolve_iso_date(
            reference_market_date,
            default=effective_as_of,
            field_name="Phase 5B reference market date",
        )
        if reference_market_date
        else None
    )
    freshness = evaluate_freshness(
        database=database,
        diagnostics=diagnostics,
        settings=config,
        as_of_date=effective_as_of,
        explicit_reference_market_date=explicit_reference,
    )

    scored = score_latest_cross_section(diagnostics.selected, bundle=bundle)
    ranked = rank_liquidity_universes(
        scored,
        minimum_daily_stocks=config.minimum_daily_stocks,
    )
    lookup = build_historical_behavior_lookup(output)
    selection = finalize_selection_columns(
        attach_historical_behavior(ranked, lookup=lookup)
    )
    selection = add_phase5b_selection_metadata(selection, freshness=freshness)

    ready = freshness.selection_readiness_status == "READY"
    usable_selection = selection.copy() if ready else selection.iloc[0:0].copy()
    top20_stocks = usable_selection.head(20).copy()
    top20_percent = usable_selection[
        pd.to_numeric(usable_selection["percentile_20m"], errors="coerce") > 80
    ].copy()

    source_freshness = build_source_freshness_report(
        database=database,
        diagnostics=diagnostics,
        freshness=freshness,
        settings=config,
    )
    summary = build_phase5b_summary(
        validation=validation,
        bundle=bundle,
        diagnostics=diagnostics,
        freshness=freshness,
        selection=selection,
        usable_selection=usable_selection,
        top20_percent=top20_percent,
        settings=config,
    )
    human_summary = build_human_summary(
        summary=summary,
        selection=selection,
        usable_selection=usable_selection,
        freshness=freshness,
    )
    paths = export_phase5b_reports(
        output_dir=output,
        usable_selection=usable_selection,
        diagnostic_selection=selection,
        top20_stocks=top20_stocks,
        top20_percent=top20_percent,
        recent_dates=diagnostics.recent_dates,
        source_freshness=source_freshness,
        summary=summary,
        human_summary=human_summary,
    )
    return Phase5BResult(
        status=freshness.selection_readiness_status,
        signal_date=diagnostics.signal_date,
        selected_rows=int(len(usable_selection)),
        output_paths=tuple(paths),
    )


def load_existing_phase5a_model(
    *,
    model_root: Path,
    source_sha256: str,
    settings: Phase5ASettings,
):
    signature = final_model_signature(
        source_sha256=source_sha256,
        settings=settings,
    )
    model_dir = model_root / signature[:16]
    manifest_path = model_dir / "phase5a_model_manifest.json"
    arrays_path = model_dir / "phase5a_model_arrays.npz"
    if not manifest_path.exists() or not arrays_path.exists():
        raise FileNotFoundError(
            "Phase 5B 找不到既有 Phase 5A 最終模型；請先成功執行 Phase 5A。"
            f" 預期目錄：{model_dir}"
        )
    bundle = load_final_model(manifest_path=manifest_path, arrays_path=arrays_path)
    if bundle.signature != signature:
        raise RuntimeError("Phase 5B 模型簽章與目前 Phase 3／Phase 5A 設定不一致")
    return bundle


def resolve_phase5_cross_section_source(
    *,
    output_dir: Path,
    phase3_shard_root: Path,
    live_shard_root: Path | None,
) -> tuple[Path, list[str] | None, str, str]:
    if live_shard_root is not None:
        pointer_path = live_shard_root / "latest.json"
        if pointer_path.exists():
            payload = json.loads(pointer_path.read_text(encoding="utf-8"))
            if payload.get("status") == "PASS":
                shard_name = str(payload.get("shard_directory") or "")
                live_dir = live_shard_root / shard_name
                if live_dir.is_dir():
                    stock_ids = sorted(path.name[:-7] for path in live_dir.glob("*.csv.gz"))
                    if stock_ids:
                        return (
                            live_dir,
                            stock_ids,
                            "phase5c_live_shards",
                            str(payload.get("target_date") or ""),
                        )
    phase3_dir = resolve_phase3_shard_directory(
        output_dir=output_dir,
        shard_root=phase3_shard_root,
    )
    return phase3_dir, None, "phase3_shards", ""


def load_recent_tpex_cross_sections(
    *,
    output_dir: Path,
    shard_dir: Path,
    feature_columns: tuple[str, ...],
    settings: Phase5BSettings,
    stock_ids: list[str] | None = None,
    cross_section_source: str = "phase3_shards",
    cross_section_target_date: str = "",
) -> CrossSectionDiagnostics:
    if stock_ids is None:
        stocks = pd.read_csv(
            output_dir / "phase3_stock_summary.csv",
            encoding="utf-8-sig",
            dtype=str,
        ).fillna("")
        stocks = stocks[
            (stocks["market_type"].str.lower() == TARGET_MARKET)
            & (stocks["phase3_status"] == "complete")
        ]
        stock_ids = sorted(set(stocks["stock_id"].astype(str)))
    required = latest_source_columns(feature_columns)
    frames: list[pd.DataFrame] = []
    latest_modified = 0.0

    for position, stock_id in enumerate(stock_ids, start=1):
        path = shard_dir / f"{stock_id}.csv.gz"
        if not path.exists():
            raise FileNotFoundError(f"Phase 3 分片遺失：{path}")
        latest_modified = max(latest_modified, path.stat().st_mtime)
        recent: pd.DataFrame | None = None
        for chunk in pd.read_csv(
            path,
            compression="gzip",
            usecols=required,
            chunksize=settings.chunk_size,
            dtype={
                "stock_id": str,
                "stock_name": str,
                "market_type": str,
                "signal_date": str,
                "feature_status": str,
            },
            low_memory=False,
        ):
            chunk = chunk[
                chunk["market_type"].astype(str).str.lower() == TARGET_MARKET
            ].copy()
            if chunk.empty:
                continue
            recent = (
                chunk
                if recent is None
                else pd.concat([recent, chunk], ignore_index=True)
            )
            if len(recent) > settings.recent_rows_per_stock:
                recent = (
                    recent.sort_values("signal_date", kind="mergesort")
                    .tail(settings.recent_rows_per_stock)
                    .copy()
                )
        if recent is not None and not recent.empty:
            frames.append(recent)
        if position % 100 == 0 or position == len(stock_ids):
            print(f"  Phase 5B 最新分片讀取進度：{position}/{len(stock_ids)}")

    if not frames:
        raise RuntimeError("Phase 3 分片找不到 TPEx 近期資料")
    recent_rows = pd.concat(frames, ignore_index=True)
    recent_rows["signal_date"] = recent_rows["signal_date"].astype(str)
    duplicate_count = int(
        recent_rows.duplicated(["stock_id", "signal_date"]).sum()
    )
    if duplicate_count:
        raise RuntimeError(f"Phase 5B 近期分片有 {duplicate_count} 筆重複股票日期")

    recent_rows = add_live_liquidity_flags(recent_rows)
    ok_mask = recent_rows["feature_status"].astype(str) == "ok"
    ok_rows = recent_rows[ok_mask].copy()
    if ok_rows.empty:
        raise RuntimeError("Phase 5B 找不到 feature_status=ok 的 TPEx 近期資料")

    daily = build_recent_date_diagnostics(
        recent_rows,
        minimum_daily_stocks=settings.minimum_daily_stocks,
    )
    eligible = daily[daily["eligible_20m"] == 1]
    if eligible.empty:
        raise RuntimeError("Phase 5B 找不到符合最低股票數的 2,000 萬訊號日")
    signal_date = str(eligible["signal_date"].max())
    selected = ok_rows[
        (ok_rows["signal_date"] == signal_date)
        & (ok_rows["liquidity_pass_20m"] == 1)
    ].copy()
    if selected.empty:
        raise RuntimeError("Phase 5B 最新合格訊號日沒有可評分股票")

    recent_dates = (
        daily.sort_values("signal_date", ascending=False, kind="mergesort")
        .head(settings.recent_date_count)
        .reset_index(drop=True)
    )
    latest_raw = str(recent_rows["signal_date"].max())
    latest_ok = str(ok_rows["signal_date"].max())
    latest_modified_at = (
        datetime.fromtimestamp(latest_modified, tz=TAIPEI_TIMEZONE).isoformat(
            timespec="seconds"
        )
        if latest_modified
        else ""
    )
    return CrossSectionDiagnostics(
        selected=selected,
        recent_dates=recent_dates,
        signal_date=signal_date,
        latest_raw_signal_date=latest_raw,
        latest_feature_ok_date=latest_ok,
        latest_eligible_signal_date=signal_date,
        eligible_date_count=int(len(eligible)),
        latest_shard_modified_at=latest_modified_at,
        cross_section_source=cross_section_source,
        cross_section_target_date=cross_section_target_date,
    )


def build_recent_date_diagnostics(
    frame: pd.DataFrame,
    *,
    minimum_daily_stocks: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for signal_date, group in frame.groupby("signal_date", sort=True):
        ok = group[group["feature_status"].astype(str) == "ok"]
        count_20m = int(ok.loc[ok["liquidity_pass_20m"] == 1, "stock_id"].nunique())
        rows.append(
            {
                "signal_date": str(signal_date),
                "raw_stock_count": int(group["stock_id"].nunique()),
                "feature_ok_stock_count": int(ok["stock_id"].nunique()),
                "liquidity_20m_stock_count": count_20m,
                "liquidity_50m_stock_count": int(
                    ok.loc[ok["liquidity_pass_50m"] == 1, "stock_id"].nunique()
                ),
                "liquidity_100m_stock_count": int(
                    ok.loc[ok["liquidity_pass_100m"] == 1, "stock_id"].nunique()
                ),
                "entry_price_available_count": int(
                    ok.loc[ok["entry_price_available"] == 1, "stock_id"].nunique()
                ),
                "eligible_20m": int(count_20m >= minimum_daily_stocks),
            }
        )
    result = pd.DataFrame(rows)
    latest_raw = str(result["signal_date"].max())
    eligible_dates = result.loc[result["eligible_20m"] == 1, "signal_date"]
    latest_eligible = str(eligible_dates.max()) if not eligible_dates.empty else ""
    result["is_latest_raw_date"] = (result["signal_date"] == latest_raw).astype(int)
    result["is_latest_eligible_date"] = (
        result["signal_date"] == latest_eligible
    ).astype(int)
    return result


def evaluate_freshness(
    *,
    database: ResearchDatabase,
    diagnostics: CrossSectionDiagnostics,
    settings: Phase5BSettings,
    as_of_date: date,
    explicit_reference_market_date: date | None,
) -> FreshnessDecision:
    calendar_dates = [
        str(row["date"])
        for row in database.query(
            "SELECT date FROM market_calendar WHERE date <= ? ORDER BY date",
            (as_of_date.isoformat(),),
        )
    ]
    latest_calendar_date = calendar_dates[-1] if calendar_dates else ""
    if explicit_reference_market_date is not None:
        reference = explicit_reference_market_date
        reference_source = "explicit_argument"
    elif latest_calendar_date:
        reference = date.fromisoformat(latest_calendar_date)
        reference_source = "sqlite_market_calendar"
    else:
        reference = as_of_date
        reference_source = "unavailable"

    signal = date.fromisoformat(diagnostics.signal_date)
    if reference > as_of_date:
        raise ValueError("Phase 5B reference market date 不可晚於 as-of date")
    calendar_age_days = (as_of_date - signal).days
    reference_age_days = (as_of_date - reference).days
    trading_day_lag: int | None = None
    if calendar_dates:
        reference_text = reference.isoformat()
        covered_dates = [value for value in calendar_dates if value <= reference_text]
        if covered_dates and diagnostics.signal_date <= covered_dates[-1]:
            trading_day_lag = sum(
                diagnostics.signal_date < value <= reference_text
                for value in covered_dates
            )
    if trading_day_lag is None and explicit_reference_market_date is not None:
        trading_day_lag = max((reference - signal).days, 0)

    latest_price = latest_tpex_source_date(database, table="stock_prices")
    latest_flow = latest_tpex_source_date(database, table="institutional_flows")
    latest_common_source = (
        min(latest_price, latest_flow) if latest_price and latest_flow else ""
    )

    if signal > as_of_date:
        freshness_status = "INVALID_FUTURE_SIGNAL_DATE"
        readiness = "BLOCKED_INVALID_DATA"
        reason = (
            f"訊號日 {signal.isoformat()} 晚於 as-of {as_of_date.isoformat()}"
        )
    elif reference_source == "unavailable":
        freshness_status = "UNKNOWN_NO_MARKET_CALENDAR"
        readiness = "BLOCKED_FRESHNESS_UNKNOWN"
        reason = "SQLite market_calendar 沒有可用日期，無法判定市場交易日落後"
    elif latest_common_source and latest_common_source > diagnostics.latest_raw_signal_date:
        freshness_status = "PHASE3_NOT_UPDATED_TO_LATEST_SOURCE"
        readiness = "BLOCKED_INCOMPLETE_DATA"
        reason = (
            "SQLite 價格與法人資料已有共同較新日期，但 Phase 3 分片尚未更新："
            f"共同來源最新 {latest_common_source}，"
            f"Phase 3 原始最新 {diagnostics.latest_raw_signal_date}"
        )
    elif explicit_reference_market_date is None and reference_age_days > settings.maximum_calendar_age_days:
        freshness_status = "STALE_MARKET_CALENDAR"
        readiness = "BLOCKED_STALE_DATA"
        reason = (
            "SQLite market_calendar 本身已過舊："
            f"最後交易日 {reference.isoformat()}，距 as-of {reference_age_days} 個日曆日"
        )
    elif diagnostics.latest_raw_signal_date > diagnostics.signal_date:
        freshness_status = "INCOMPLETE_LATEST_CROSS_SECTION"
        readiness = "BLOCKED_INCOMPLETE_DATA"
        reason = (
            "Phase 3 已有較新的原始訊號日，但未形成足額 2,000 萬截面："
            f"原始最新 {diagnostics.latest_raw_signal_date}，"
            f"合格最新 {diagnostics.signal_date}"
        )
    elif trading_day_lag is None:
        freshness_status = "UNKNOWN_TRADING_DAY_LAG"
        readiness = "BLOCKED_FRESHNESS_UNKNOWN"
        reason = "訊號日不在可用市場日曆範圍內，無法計算交易日落後"
    elif trading_day_lag > settings.maximum_trading_day_lag:
        freshness_status = "STALE_SELECTION_DATE"
        readiness = "BLOCKED_STALE_DATA"
        reason = (
            f"訊號日落後參考市場日 {trading_day_lag} 個交易日，"
            f"超過允許 {settings.maximum_trading_day_lag} 日"
        )
    elif (
        explicit_reference_market_date is None
        and calendar_age_days > settings.maximum_calendar_age_days
    ):
        freshness_status = "STALE_CALENDAR_AGE"
        readiness = "BLOCKED_STALE_DATA"
        reason = (
            f"訊號日距 as-of 已 {calendar_age_days} 個日曆日，"
            f"超過允許 {settings.maximum_calendar_age_days} 日"
        )
    else:
        freshness_status = "FRESH"
        readiness = "READY"
        reason = "最新合格截面在允許的交易日與日曆日落後範圍內"

    return FreshnessDecision(
        model_status="PASS",
        data_freshness_status=freshness_status,
        selection_readiness_status=readiness,
        readiness_reason=reason,
        as_of_date=as_of_date.isoformat(),
        reference_market_date=reference.isoformat(),
        reference_market_date_source=reference_source,
        calendar_reference_age_days=reference_age_days,
        calendar_age_days=calendar_age_days,
        trading_day_lag=trading_day_lag,
        latest_calendar_date=latest_calendar_date,
        latest_tpex_price_date=latest_price,
        latest_tpex_flow_date=latest_flow,
        latest_common_tpex_source_date=latest_common_source,
    )


def latest_tpex_source_date(database: ResearchDatabase, *, table: str) -> str:
    if table not in {"stock_prices", "institutional_flows"}:
        raise ValueError(f"不支援的來源表：{table}")
    value = database.scalar(
        f"""
        SELECT MAX(source.date)
        FROM {table} source
        JOIN model_universe universe ON universe.stock_id=source.stock_id
        WHERE LOWER(universe.market_type)='tpex'
        """
    )
    return str(value or "")


def add_phase5b_selection_metadata(
    selection: pd.DataFrame,
    *,
    freshness: FreshnessDecision,
) -> pd.DataFrame:
    result = selection.copy()
    result["selection_band"] = np.select(
        [
            result["percentile_20m"] > 90,
            result["percentile_20m"] > 80,
            result["percentile_20m"] <= 10,
            result["percentile_20m"] <= 20,
        ],
        ["前10%", "前10%～20%", "後10%", "後10%～20%"],
        default="中間60%",
    )
    result["model_status"] = freshness.model_status
    result["data_freshness_status"] = freshness.data_freshness_status
    result["selection_readiness_status"] = freshness.selection_readiness_status
    result["freshness_reason"] = freshness.readiness_reason
    result["as_of_date"] = freshness.as_of_date
    result["reference_market_date"] = freshness.reference_market_date
    result["trading_day_lag"] = freshness.trading_day_lag
    result["calendar_age_days"] = freshness.calendar_age_days
    result["data_status"] = (
        "OK" if freshness.selection_readiness_status == "READY" else "STALE_OR_INCOMPLETE"
    )
    metadata_columns = [
        "model_status",
        "data_freshness_status",
        "selection_readiness_status",
        "freshness_reason",
        "as_of_date",
        "reference_market_date",
        "trading_day_lag",
        "calendar_age_days",
    ]
    base_columns = [column for column in result.columns if column not in metadata_columns]
    return result[metadata_columns + base_columns]


def build_source_freshness_report(
    *,
    database: ResearchDatabase,
    diagnostics: CrossSectionDiagnostics,
    freshness: FreshnessDecision,
    settings: Phase5BSettings,
) -> pd.DataFrame:
    calendar_max = str(database.scalar("SELECT MAX(date) FROM market_calendar") or "")
    metrics: list[tuple[str, Any]] = [
        ("phase5b_version", PHASE5B_VERSION),
        ("as_of_date", freshness.as_of_date),
        ("reference_market_date", freshness.reference_market_date),
        ("reference_market_date_source", freshness.reference_market_date_source),
        ("latest_market_calendar_date_at_or_before_as_of", freshness.latest_calendar_date),
        ("market_calendar_max_date", calendar_max),
        ("calendar_reference_age_days", freshness.calendar_reference_age_days),
        ("latest_tpex_price_date", freshness.latest_tpex_price_date),
        ("latest_tpex_institutional_flow_date", freshness.latest_tpex_flow_date),
        ("latest_common_tpex_source_date", freshness.latest_common_tpex_source_date),
        ("cross_section_source", diagnostics.cross_section_source),
        ("cross_section_target_date", diagnostics.cross_section_target_date),
        ("phase3_latest_raw_signal_date", diagnostics.latest_raw_signal_date),
        ("phase3_latest_feature_ok_date", diagnostics.latest_feature_ok_date),
        ("phase3_latest_eligible_signal_date", diagnostics.latest_eligible_signal_date),
        ("phase3_latest_shard_modified_at", diagnostics.latest_shard_modified_at),
        ("trading_day_lag", freshness.trading_day_lag),
        ("calendar_age_days", freshness.calendar_age_days),
        ("maximum_trading_day_lag", settings.maximum_trading_day_lag),
        ("maximum_calendar_age_days", settings.maximum_calendar_age_days),
        ("data_freshness_status", freshness.data_freshness_status),
        ("selection_readiness_status", freshness.selection_readiness_status),
        ("readiness_reason", freshness.readiness_reason),
    ]
    return pd.DataFrame([{"metric": key, "value": value} for key, value in metrics])


def build_phase5b_summary(
    *,
    validation: dict[str, Any],
    bundle,
    diagnostics: CrossSectionDiagnostics,
    freshness: FreshnessDecision,
    selection: pd.DataFrame,
    usable_selection: pd.DataFrame,
    top20_percent: pd.DataFrame,
    settings: Phase5BSettings,
) -> pd.DataFrame:
    metrics: list[tuple[str, Any]] = [
        ("phase5b_version", PHASE5B_VERSION),
        ("pipeline_status", "PASS"),
        ("model_status", freshness.model_status),
        ("data_freshness_status", freshness.data_freshness_status),
        ("selection_readiness_status", freshness.selection_readiness_status),
        ("readiness_reason", freshness.readiness_reason),
        ("target_market", TARGET_MARKET),
        ("cross_section_source", diagnostics.cross_section_source),
        ("cross_section_target_date", diagnostics.cross_section_target_date),
        ("signal_date", diagnostics.signal_date),
        ("as_of_date", freshness.as_of_date),
        ("reference_market_date", freshness.reference_market_date),
        ("reference_market_date_source", freshness.reference_market_date_source),
        ("calendar_age_days", freshness.calendar_age_days),
        ("trading_day_lag", freshness.trading_day_lag),
        ("maximum_trading_day_lag", settings.maximum_trading_day_lag),
        ("maximum_calendar_age_days", settings.maximum_calendar_age_days),
        ("latest_raw_signal_date", diagnostics.latest_raw_signal_date),
        ("latest_feature_ok_date", diagnostics.latest_feature_ok_date),
        ("latest_eligible_signal_date", diagnostics.latest_eligible_signal_date),
        ("eligible_date_count", diagnostics.eligible_date_count),
        ("diagnostic_selection_rows", len(selection)),
        ("usable_selection_rows", len(usable_selection)),
        ("top20_stocks_rows", min(len(usable_selection), 20)),
        ("top20_percent_rows", len(top20_percent)),
        ("model_signature", bundle.signature),
        ("model_last_training_date", bundle.last_training_date),
        ("phase3_source_sha256", bundle.source_sha256),
        ("phase3_config_signature", validation["phase3_config_signature"]),
        ("duplicate_stock_count", int(selection["stock_id"].duplicated().sum())),
        ("rank_duplicate_count", int(selection["rank_20m"].duplicated().sum())),
        ("ready_for_local_reference", int(len(usable_selection) > 0)),
        ("deployment_status", "LOCAL_RESEARCH_ONLY"),
    ]
    return pd.DataFrame([{"metric": key, "value": value} for key, value in metrics])


def build_human_summary(
    *,
    summary: pd.DataFrame,
    selection: pd.DataFrame,
    usable_selection: pd.DataFrame,
    freshness: FreshnessDecision,
) -> str:
    values = {
        str(row["metric"]): str(row["value"])
        for _, row in summary.iterrows()
    }
    lines = [
        "# Phase 5B TPEx 法人選股參考摘要",
        "",
        f"- 模型狀態：`{freshness.model_status}`",
        f"- 資料新鮮度：`{freshness.data_freshness_status}`",
        f"- 選股可用狀態：`{freshness.selection_readiness_status}`",
        f"- 訊號日期：`{values['signal_date']}`",
        f"- as-of 日期：`{freshness.as_of_date}`",
        f"- 判定原因：{freshness.readiness_reason}",
        "",
        "> 法人選股指數是同日 TPEx 相對排名，不是上漲機率、買進指令或獲利保證。",
        "",
    ]
    if usable_selection.empty:
        lines.extend(
            [
                "## 本次沒有產生可使用的選股名單",
                "",
                "完整舊截面只保存在 `phase5b_selection_index_diagnostic.csv` 供診斷；",
                "`phase5b_selection_index.csv` 與兩份 Top 清單會保持空白，避免把過期資料誤當今日名單。",
                "",
                "## 最近可診斷截面",
                "",
                f"- 股票數：{len(selection):,}",
                f"- 最新合格訊號日：{values['latest_eligible_signal_date']}",
                f"- 最新原始訊號日：{values['latest_raw_signal_date']}",
            ]
        )
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "## 可用名單摘要",
            "",
            f"- 合格股票數：{len(usable_selection):,}",
            f"- 前 20% 股票數：{int((usable_selection['percentile_20m'] > 80).sum()):,}",
            "",
            "## 前 20 檔",
            "",
            "| 排名 | 股票 | 指數百分位 | Raw 指數 | 流動性 | 歷史10日等權平均 | 歷史10日下跌5%比例 |",
            "|---:|---|---:|---:|---|---:|---:|",
        ]
    )
    for _, row in usable_selection.head(20).iterrows():
        stock = f"{row['stock_id']} {row['stock_name']}"
        lines.append(
            "| {rank} | {stock} | {percentile:.2f} | {raw:+.3f} | {tier} | {history:+.3%} | {down:.2%} |".format(
                rank=int(row["rank_20m"]),
                stock=stock,
                percentile=float(row["percentile_20m"]),
                raw=float(row["institutional_index_raw"]),
                tier=row["liquidity_tier"],
                history=float(row["history_10d_equal_day_average_return"]),
                down=float(row["history_10d_down_5pct_rate"]),
            )
        )
    return "\n".join(lines) + "\n"


def export_phase5b_reports(
    *,
    output_dir: Path,
    usable_selection: pd.DataFrame,
    diagnostic_selection: pd.DataFrame,
    top20_stocks: pd.DataFrame,
    top20_percent: pd.DataFrame,
    recent_dates: pd.DataFrame,
    source_freshness: pd.DataFrame,
    summary: pd.DataFrame,
    human_summary: str,
) -> list[Path]:
    paths = [
        write_csv(output_dir / "phase5b_selection_index.csv", usable_selection),
        write_csv(
            output_dir / "phase5b_selection_index_diagnostic.csv",
            diagnostic_selection,
        ),
        write_csv(
            output_dir / "phase5b_selection_index_top20_stocks.csv",
            top20_stocks,
        ),
        write_csv(
            output_dir / "phase5b_selection_index_top20_percent.csv",
            top20_percent,
        ),
        write_csv(output_dir / "phase5b_recent_date_diagnostics.csv", recent_dates),
        write_csv(output_dir / "phase5b_data_source_freshness.csv", source_freshness),
        write_csv(output_dir / "phase5b_summary.csv", summary),
    ]
    markdown_path = output_dir / "phase5b_daily_selection_summary.md"
    temporary_markdown = markdown_path.with_suffix(".md.tmp")
    temporary_markdown.write_text(human_summary, encoding="utf-8")
    os.replace(temporary_markdown, markdown_path)
    paths.append(markdown_path)

    archive = output_dir / "phase5b_selection_index_reports.zip"
    temporary = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in paths:
            handle.write(path, arcname=path.name)
    os.replace(temporary, archive)
    paths.append(archive)
    return paths


def resolve_iso_date(value: str | None, *, default: date, field_name: str) -> date:
    if not value:
        return default
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} 必須為 YYYY-MM-DD：{value}") from exc
