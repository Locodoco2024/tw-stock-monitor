from __future__ import annotations

import hashlib
import json
import math
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from research.institutional_model.database import ResearchDatabase
from research.institutional_model.phase4_horizon import _calculate_stock_horizon_outcomes
from research.institutional_model.phase4_selection import (
    moving_block_bootstrap_mean_ci,
    resolve_phase3_shard_directory,
)
from research.institutional_model.phase4_stability import CONFIRMATION_START_YEAR


PHASE6C_VERSION = "phase6c-v1"
TARGET_MARKET = "twse"
RAW_SCORE_COLUMN = "return_rank_40d_score"
BASE_PERCENTILE_COLUMN = "return_rank_40d_score_daily_percentile"
ENTRY_THRESHOLDS = (85.0, 90.0, 95.0)
CONFIRMATION_DAYS = (1, 3, 5, 10)
COOLDOWN_DAYS = (0, 10, 20, 40)
OUTCOME_HORIZONS = (40, 60, 80)

# These universes are fixed before lifecycle rule selection.  The volume screen is
# expressed in board lots (1 lot = 1,000 shares) and is paired with money screens
# so a high-priced, low-share-turnover stock cannot pass only because of price.
LIQUIDITY_UNIVERSES: tuple[tuple[str, int, int], ...] = (
    ("money20m", 20, 0),
    ("money50m", 50, 0),
    ("money100m", 100, 0),
    ("money50m_volume100lots", 50, 100),
    ("money100m_volume100lots", 100, 100),
    ("money100m_volume300lots", 100, 300),
)


@dataclass(frozen=True)
class Phase6CTWSELifecycleSettings:
    minimum_daily_stocks: int = 50
    entry_thresholds: tuple[float, ...] = ENTRY_THRESHOLDS
    confirmation_days: tuple[int, ...] = CONFIRMATION_DAYS
    cooldown_days: tuple[int, ...] = COOLDOWN_DAYS
    minimum_candidate_events: int = 100
    minimum_candidate_stocks: int = 30
    bootstrap_iterations: int = 1_000
    bootstrap_block_months: int = 3
    random_seed: int = 20260731

    def validate(self) -> None:
        if self.minimum_daily_stocks < 20:
            raise ValueError("Phase 6C 每日最少股票數不可小於 20")
        if tuple(sorted(set(self.entry_thresholds))) != self.entry_thresholds:
            raise ValueError("Phase 6C 進榜門檻必須由小到大且不可重複")
        if any(value <= 0 or value >= 100 for value in self.entry_thresholds):
            raise ValueError("Phase 6C 進榜門檻必須介於 0 與 100 之間")
        if tuple(sorted(set(self.confirmation_days))) != self.confirmation_days:
            raise ValueError("Phase 6C 確認日必須由小到大且不可重複")
        if any(value < 1 for value in self.confirmation_days):
            raise ValueError("Phase 6C 確認日必須大於 0")
        if tuple(sorted(set(self.cooldown_days))) != self.cooldown_days:
            raise ValueError("Phase 6C 冷卻日必須由小到大且不可重複")
        if any(value < 0 for value in self.cooldown_days):
            raise ValueError("Phase 6C 冷卻日不可小於 0")
        if self.minimum_candidate_events < 20:
            raise ValueError("Phase 6C 候選規則最低事件數不可小於 20")
        if self.minimum_candidate_stocks < 10:
            raise ValueError("Phase 6C 候選規則最低股票數不可小於 10")
        if self.bootstrap_iterations < 200:
            raise ValueError("Phase 6C bootstrap 次數不可小於 200")
        if self.bootstrap_block_months < 1:
            raise ValueError("Phase 6C bootstrap 區塊月份必須大於 0")


@dataclass(frozen=True)
class Phase6CTWSELifecycleResult:
    status: str
    lifecycle_candidate_found: bool
    source_rows: int
    event_rows: int
    output_paths: tuple[Path, ...]


def run_phase6c_twse_lifecycle_validation(
    *,
    database: ResearchDatabase,
    output_dir: Path | str,
    shard_root: Path | str,
    outcome_cache_root: Path | str,
    settings: Phase6CTWSELifecycleSettings | None = None,
    source_path: Path | str | None = None,
    force: bool = False,
) -> Phase6CTWSELifecycleResult:
    config = settings or Phase6CTWSELifecycleSettings()
    config.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    validate_phase6b_gate(output)
    resolved_source = resolve_phase6b_oos_source(output, source_path)
    source = pd.read_csv(
        resolved_source,
        compression="infer",
        dtype={"stock_id": "string", "signal_date": "string"},
        low_memory=False,
    )
    shard_dir = resolve_phase3_shard_directory(
        output_dir=output,
        shard_root=Path(shard_root),
    )
    enriched = enrich_phase6b_scores(
        source,
        database=database,
        shard_dir=shard_dir,
    )
    reports = evaluate_twse_lifecycle(
        enriched,
        settings=config,
        source_path=resolved_source,
    )
    events = reports["entry_events"]
    if not events.empty:
        event_outcomes = load_event_long_horizon_outcomes(
            database=database,
            events=events,
            cache_root=Path(outcome_cache_root),
            source_path=resolved_source,
            force=force,
        )
        events = merge_event_outcomes(events, event_outcomes)
        reports["entry_events"] = events
        reports.update(
            build_post_outcome_reports(
                enriched=enriched,
                events=events,
                settings=config,
            )
        )
    else:
        reports.update(_empty_post_outcome_reports())
    reports["summary"] = build_phase6c_summary(
        enriched=enriched,
        events=reports["entry_events"],
        candidates=reports["rule_candidates"],
        settings=config,
    )
    paths = export_phase6c_reports(output_dir=output, reports=reports)
    candidate_found = bool(
        not reports["rule_candidates"].empty
        and reports["rule_candidates"]["candidate_status"].eq("strong_candidate").any()
    )
    return Phase6CTWSELifecycleResult(
        status="PASS",
        lifecycle_candidate_found=candidate_found,
        source_rows=len(source),
        event_rows=len(reports["entry_events"]),
        output_paths=tuple(paths),
    )


def validate_phase6b_gate(output_dir: Path) -> None:
    path = output_dir / "phase6b_summary.csv"
    if not path.exists():
        raise FileNotFoundError("Phase 6C 缺少 phase6b_summary.csv")
    summary = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    metrics = dict(
        zip(summary["metric"].astype(str), summary["value"].astype(str), strict=True)
    )
    if metrics.get("market") != TARGET_MARKET:
        raise RuntimeError("Phase 6C 的 Phase 6B summary 不是 TWSE")
    if metrics.get("return_rank_40d_validation_pass") not in {"1", "1.0"}:
        raise RuntimeError("Phase 6B TWSE 40 日模型尚未通過，不可執行 Phase 6C")



def resolve_phase6b_oos_source(
    output_dir: Path,
    source_path: Path | str | None,
) -> Path:
    if source_path is not None:
        candidate = Path(source_path)
        if not candidate.exists():
            raise FileNotFoundError(f"找不到 Phase 6B OOS 分數：{candidate}")
        return candidate
    candidate = output_dir / "phase6b_oos_scores.csv.gz"
    if candidate.exists():
        return candidate
    archive = output_dir / "phase6b_twse_40d_validation_reports.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as handle:
            member = "phase6b_oos_scores.csv.gz"
            if member not in handle.namelist():
                raise RuntimeError("Phase 6B 驗證 ZIP 缺少 phase6b_oos_scores.csv.gz")
            candidate.write_bytes(handle.read(member))
            return candidate
    raise FileNotFoundError("找不到 Phase 6B OOS 分數；請先完成 Phase 6B。")


def enrich_phase6b_scores(
    frame: pd.DataFrame,
    *,
    database: ResearchDatabase,
    shard_dir: Path,
) -> pd.DataFrame:
    required = {
        "stock_id",
        "stock_name",
        "signal_date",
        "signal_year",
        "test_year",
        "adjusted_return_20d",
        "adjusted_return_40d",
        RAW_SCORE_COLUMN,
        BASE_PERCENTILE_COLUMN,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Phase 6C 缺少 Phase 6B 欄位：{missing}")
    base = frame[list(required)].copy()
    base["stock_id"] = base["stock_id"].astype(str)
    base["stock_name"] = base["stock_name"].fillna("").astype(str)
    base["signal_date"] = base["signal_date"].astype(str)
    base["signal_year"] = pd.to_numeric(base["signal_year"], errors="raise").astype(int)
    base["test_year"] = pd.to_numeric(base["test_year"], errors="raise").astype(int)
    for column in (
        "adjusted_return_20d",
        "adjusted_return_40d",
        RAW_SCORE_COLUMN,
        BASE_PERCENTILE_COLUMN,
    ):
        base[column] = pd.to_numeric(base[column], errors="coerce")
    if not np.isfinite(base[RAW_SCORE_COLUMN]).all():
        raise RuntimeError("Phase 6C 原始模型分數包含 NaN／Infinity")
    if int(base.duplicated(["stock_id", "signal_date"]).sum()):
        raise RuntimeError("Phase 6C Phase 6B OOS 股票＋訊號日重複")

    liquidity = load_phase3_liquidity_rows(base, shard_dir=shard_dir)
    enriched = base.merge(
        liquidity,
        on=["stock_id", "signal_date"],
        how="left",
        validate="one_to_one",
    )
    required_liquidity = [
        "median_trading_money_20d",
        "liquidity_pass_20m",
        "liquidity_pass_50m",
        "liquidity_pass_100m",
        "entry_price_available",
    ]
    if enriched[required_liquidity].isna().any().any():
        missing_rows = int(enriched[required_liquidity].isna().any(axis=1).sum())
        raise RuntimeError(f"Phase 6C 有 {missing_rows} 筆 OOS 列缺少 Phase 3 流動性資料")
    for column in (
        "median_trading_money_20d",
        "history_market_days_20",
        "normal_trading_days_20d",
        "max_zero_volume_streak_20d",
        "entry_price_available",
        "liquidity_pass_20m",
        "liquidity_pass_50m",
        "liquidity_pass_100m",
    ):
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce")
    volume = calculate_rolling_volume_metrics(database, enriched)
    enriched = enriched.merge(
        volume,
        on=["stock_id", "signal_date"],
        how="left",
        validate="one_to_one",
    )
    if enriched["median_trading_volume_lots_20d"].isna().any():
        raise RuntimeError("Phase 6C 部分 OOS 列無法重建 20 日成交量中位數")
    ordered_dates = sorted(enriched["signal_date"].unique())
    date_index = {value: index for index, value in enumerate(ordered_dates)}
    enriched["market_day_index"] = enriched["signal_date"].map(date_index).astype(int)
    enriched["signal_month"] = enriched["signal_date"].str[:7]
    return enriched.sort_values(
        ["stock_id", "market_day_index"], kind="stable"
    ).reset_index(drop=True)


def load_phase3_liquidity_rows(
    source: pd.DataFrame,
    *,
    shard_dir: Path,
) -> pd.DataFrame:
    usecols = [
        "stock_id",
        "signal_date",
        "median_trading_money_20d",
        "history_market_days_20",
        "normal_trading_days_20d",
        "max_zero_volume_streak_20d",
        "entry_price_available",
        "liquidity_pass_20m",
        "liquidity_pass_50m",
        "liquidity_pass_100m",
    ]
    frames: list[pd.DataFrame] = []
    date_lookup = {
        stock_id: set(group["signal_date"].astype(str))
        for stock_id, group in source.groupby("stock_id", sort=False)
    }
    for stock_id, dates in date_lookup.items():
        path = shard_dir / f"{stock_id}.csv.gz"
        if not path.exists():
            raise FileNotFoundError(f"Phase 6C 找不到 Phase 3 分片：{path}")
        shard = pd.read_csv(
            path,
            compression="gzip",
            usecols=usecols,
            dtype={"stock_id": "string", "signal_date": "string"},
            low_memory=False,
        )
        selected = shard[shard["signal_date"].astype(str).isin(dates)].copy()
        frames.append(selected)
    if not frames:
        raise RuntimeError("Phase 6C 沒有讀到 Phase 3 流動性資料")
    result = pd.concat(frames, ignore_index=True)
    result["stock_id"] = result["stock_id"].astype(str)
    result["signal_date"] = result["signal_date"].astype(str)
    return result


def calculate_rolling_volume_metrics(
    database: ResearchDatabase,
    source: pd.DataFrame,
) -> pd.DataFrame:
    market_dates = [
        str(row["date"])
        for row in database.query("SELECT date FROM market_calendar ORDER BY date")
    ]
    if not market_dates:
        raise RuntimeError("Phase 6C 市場交易日曆為空")
    date_index = {value: index for index, value in enumerate(market_dates)}
    rows: list[pd.DataFrame] = []
    for stock_id, group in source.groupby("stock_id", sort=False):
        prices = [
            dict(row)
            for row in database.query(
                """
                SELECT date, close, trading_volume, trading_money
                FROM stock_prices
                WHERE stock_id=?
                ORDER BY date
                """,
                (str(stock_id),),
            )
        ]
        price_map = {str(row["date"]): row for row in prices}
        volumes = np.array(
            [float(price_map.get(date, {}).get("trading_volume") or 0.0) for date in market_dates],
            dtype=np.float64,
        )
        rolling = pd.Series(volumes).rolling(20, min_periods=20).median().to_numpy()
        stock_rows: list[dict[str, Any]] = []
        for signal_date in group["signal_date"].astype(str):
            position = date_index.get(signal_date)
            if position is None:
                raise RuntimeError(f"Phase 6C signal_date 不在市場日曆：{signal_date}")
            price = price_map.get(signal_date, {})
            median_shares = float(rolling[position]) if np.isfinite(rolling[position]) else np.nan
            stock_rows.append(
                {
                    "stock_id": str(stock_id),
                    "signal_date": signal_date,
                    "median_trading_volume_shares_20d": median_shares,
                    "median_trading_volume_lots_20d": median_shares / 1000.0,
                    "signal_close": _number_or_nan(price.get("close")),
                    "signal_trading_volume_shares": _number_or_nan(
                        price.get("trading_volume")
                    ),
                    "signal_trading_volume_lots": _number_or_nan(
                        price.get("trading_volume")
                    )
                    / 1000.0,
                    "signal_trading_money": _number_or_nan(price.get("trading_money")),
                }
            )
        rows.append(pd.DataFrame(stock_rows))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def evaluate_twse_lifecycle(
    frame: pd.DataFrame,
    *,
    settings: Phase6CTWSELifecycleSettings,
    source_path: Path | None = None,
) -> dict[str, pd.DataFrame]:
    settings.validate()
    universe_frames: list[pd.DataFrame] = []
    coverage_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    for universe_id, money_million, minimum_lots in LIQUIDITY_UNIVERSES:
        ranked, coverage = build_liquidity_universe(
            frame,
            universe_id=universe_id,
            minimum_money_million=money_million,
            minimum_volume_lots=minimum_lots,
            minimum_daily_stocks=settings.minimum_daily_stocks,
        )
        coverage_frames.append(coverage)
        if ranked.empty:
            continue
        universe_frames.append(ranked)
        event_frames.append(build_universe_entry_events(ranked, settings=settings))
    if not universe_frames:
        raise RuntimeError("Phase 6C 所有流動性母體都低於每日最低股票數")
    universes = pd.concat(universe_frames, ignore_index=True)
    coverage = pd.concat(coverage_frames, ignore_index=True)
    events = (
        pd.concat(event_frames, ignore_index=True)
        if event_frames
        else pd.DataFrame()
    )
    if events.empty:
        raise RuntimeError("Phase 6C 沒有產生任何 TWSE 生命週期事件")
    if int(events.duplicated("event_id").sum()):
        raise RuntimeError("Phase 6C event_id 重複")
    return {
        "input_audit": build_input_audit(
            frame=frame,
            universes=universes,
            events=events,
            coverage=coverage,
            settings=settings,
            source_path=source_path,
        ),
        "universe_coverage": coverage,
        "entry_events": events,
    }


def build_liquidity_universe(
    frame: pd.DataFrame,
    *,
    universe_id: str,
    minimum_money_million: int,
    minimum_volume_lots: int,
    minimum_daily_stocks: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    money_column = f"liquidity_pass_{minimum_money_million}m"
    if money_column not in frame.columns:
        raise KeyError(f"Phase 6C 缺少流動性旗標：{money_column}")
    money_pass = pd.to_numeric(frame[money_column], errors="coerce").fillna(0) == 1
    volume = pd.to_numeric(
        frame["median_trading_volume_lots_20d"], errors="coerce"
    ).fillna(0)
    selected = frame[money_pass & (volume >= minimum_volume_lots)].copy()
    before = selected.groupby("signal_date")["stock_id"].nunique()
    valid_dates = set(before[before >= minimum_daily_stocks].index.astype(str))
    selected = selected[selected["signal_date"].astype(str).isin(valid_dates)].copy()
    if selected.empty:
        coverage = pd.DataFrame(
            [
                {
                    "universe_id": universe_id,
                    "minimum_money_million": minimum_money_million,
                    "minimum_volume_lots": minimum_volume_lots,
                    "signal_dates": 0,
                    "minimum_daily_stocks": 0,
                    "median_daily_stocks": 0,
                    "maximum_daily_stocks": 0,
                    "rows": 0,
                }
            ]
        )
        return selected, coverage
    selected["universe_id"] = universe_id
    selected["minimum_money_million"] = minimum_money_million
    selected["minimum_volume_lots"] = minimum_volume_lots
    selected["universe_stock_count"] = selected.groupby("signal_date")[
        "stock_id"
    ].transform("nunique")
    selected["universe_percentile"] = (
        selected.groupby("signal_date")[RAW_SCORE_COLUMN]
        .rank(method="average", pct=True)
        .mul(100.0)
    )
    counts = selected.groupby("signal_date")["stock_id"].nunique()
    coverage = pd.DataFrame(
        [
            {
                "universe_id": universe_id,
                "minimum_money_million": minimum_money_million,
                "minimum_volume_lots": minimum_volume_lots,
                "signal_dates": int(len(counts)),
                "minimum_daily_stocks": int(counts.min()),
                "median_daily_stocks": float(counts.median()),
                "maximum_daily_stocks": int(counts.max()),
                "rows": int(len(selected)),
            }
        ]
    )
    selected["daily_universe_adjusted_return_40d"] = selected.groupby(
        "signal_date"
    )["adjusted_return_40d"].transform("mean")
    selected["excess_adjusted_return_40d"] = (
        selected["adjusted_return_40d"]
        - selected["daily_universe_adjusted_return_40d"]
    )
    return selected, coverage


def build_universe_entry_events(
    universe: pd.DataFrame,
    *,
    settings: Phase6CTWSELifecycleSettings,
) -> pd.DataFrame:
    events: list[pd.DataFrame] = []
    for threshold in settings.entry_thresholds:
        annotated = annotate_runs(universe, threshold=threshold)
        for confirmation in settings.confirmation_days:
            selected = annotated[
                (annotated["qualifies"] == 1)
                & (annotated["run_length"] == confirmation)
            ].copy()
            if selected.empty:
                continue
            selected["entry_threshold"] = threshold
            selected["confirmation_days"] = confirmation
            selected["entry_rule"] = rule_name(threshold, confirmation)
            selected["event_id"] = (
                selected["universe_id"].astype(str)
                + ":"
                + selected["stock_id"].astype(str)
                + ":"
                + selected["entry_rule"].astype(str)
                + ":"
                + selected["signal_date"].astype(str)
            )
            events.append(selected)
    if not events:
        return pd.DataFrame()
    result = pd.concat(events, ignore_index=True)
    keep = [
        "event_id",
        "universe_id",
        "minimum_money_million",
        "minimum_volume_lots",
        "stock_id",
        "stock_name",
        "signal_date",
        "signal_year",
        "test_year",
        "market_day_index",
        "entry_threshold",
        "confirmation_days",
        "entry_rule",
        "universe_percentile",
        "universe_stock_count",
        RAW_SCORE_COLUMN,
        "adjusted_return_20d",
        "adjusted_return_40d",
        "daily_universe_adjusted_return_40d",
        "excess_adjusted_return_40d",
        "median_trading_money_20d",
        "median_trading_volume_lots_20d",
        "signal_close",
        "signal_trading_volume_lots",
        "signal_trading_money",
        "episode_start_index",
        "run_length",
    ]
    return result[keep].sort_values(
        ["universe_id", "entry_rule", "stock_id", "signal_date"],
        kind="stable",
    ).reset_index(drop=True)


def annotate_runs(frame: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    working = frame.sort_values(
        ["stock_id", "market_day_index"], kind="stable"
    ).copy()
    qualifies = (
        pd.to_numeric(working["universe_percentile"], errors="coerce") >= threshold
    ).to_numpy()
    stock_ids = working["stock_id"].astype(str).to_numpy()
    indexes = working["market_day_index"].to_numpy(dtype=np.int64)
    run_length = np.zeros(len(working), dtype=np.int32)
    episode_start = np.full(len(working), -1, dtype=np.int64)
    previous_stock = ""
    previous_index = -2
    previous_qualified = False
    current_run = 0
    current_start = -1
    for position in range(len(working)):
        consecutive = (
            stock_ids[position] == previous_stock
            and indexes[position] == previous_index + 1
        )
        if qualifies[position]:
            if consecutive and previous_qualified:
                current_run += 1
            else:
                current_run = 1
                current_start = int(indexes[position])
            run_length[position] = current_run
            episode_start[position] = current_start
        else:
            current_run = 0
            current_start = -1
        previous_stock = stock_ids[position]
        previous_index = int(indexes[position])
        previous_qualified = bool(qualifies[position])
    working["qualifies"] = qualifies.astype(np.uint8)
    working["run_length"] = run_length
    working["episode_start_index"] = episode_start
    return working


def load_event_long_horizon_outcomes(
    *,
    database: ResearchDatabase,
    events: pd.DataFrame,
    cache_root: Path,
    source_path: Path,
    force: bool,
) -> pd.DataFrame:
    unique_events = events[["stock_id", "signal_date"]].drop_duplicates().copy()
    cache_dir = resolve_outcome_cache_directory(
        database=database,
        cache_root=cache_root,
        source_path=source_path,
        force=force,
    )
    market_dates = [
        str(row["date"])
        for row in database.query("SELECT date FROM market_calendar ORDER BY date")
    ]
    frames: list[pd.DataFrame] = []
    for stock_id, group in unique_events.groupby("stock_id", sort=True):
        dates = sorted(group["signal_date"].astype(str).unique())
        path = cache_dir / f"{stock_id}_40_60_80d.csv.gz"
        cached: pd.DataFrame | None = None
        if not force and path.exists():
            try:
                candidate = pd.read_csv(
                    path,
                    compression="gzip",
                    dtype={"stock_id": "string", "signal_date": "string"},
                    low_memory=False,
                )
                if set(candidate["signal_date"].astype(str)) == set(dates):
                    cached = candidate
            except (OSError, ValueError, KeyError):
                cached = None
        if cached is None:
            merged: pd.DataFrame | None = None
            for horizon in OUTCOME_HORIZONS:
                outcome = _calculate_stock_horizon_outcomes(
                    database=database,
                    stock_id=str(stock_id),
                    signal_dates=dates,
                    market_dates=market_dates,
                    horizon=horizon,
                    threshold=0.05,
                )
                keep = [
                    "stock_id",
                    "signal_date",
                    f"adjusted_return_{horizon}d",
                    f"max_adjusted_return_{horizon}d",
                    f"min_adjusted_return_{horizon}d",
                    f"label_status_{horizon}d",
                ]
                outcome = outcome[keep]
                merged = outcome if merged is None else merged.merge(
                    outcome,
                    on=["stock_id", "signal_date"],
                    how="outer",
                    validate="one_to_one",
                )
            assert merged is not None
            cached = merged
            _write_gzip_csv(path, cached)
        frames.append(cached)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def resolve_outcome_cache_directory(
    *,
    database: ResearchDatabase,
    cache_root: Path,
    source_path: Path,
    force: bool,
) -> Path:
    stat = database.path.stat()
    payload = {
        "phase6c_version": PHASE6C_VERSION,
        "source_sha256": file_sha256(source_path),
        "database_size": stat.st_size,
        "database_mtime_ns": stat.st_mtime_ns,
        "horizons": OUTCOME_HORIZONS,
    }
    signature = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    directory = cache_root / signature[:16]
    if force and directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(
        json.dumps({"signature": signature, **payload}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return directory


def merge_event_outcomes(events: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    outcome_columns = [
        column
        for column in outcomes.columns
        if column not in {"stock_id", "signal_date", "adjusted_return_40d"}
    ]
    merged = events.merge(
        outcomes[["stock_id", "signal_date", *outcome_columns]],
        on=["stock_id", "signal_date"],
        how="left",
        validate="many_to_one",
    )
    for horizon in OUTCOME_HORIZONS:
        for prefix in ("adjusted_return", "max_adjusted_return", "min_adjusted_return"):
            column = f"{prefix}_{horizon}d"
            if column in merged:
                merged[column] = pd.to_numeric(merged[column], errors="coerce")
    return merged


def build_post_outcome_reports(
    *,
    enriched: pd.DataFrame,
    events: pd.DataFrame,
    settings: Phase6CTWSELifecycleSettings,
) -> dict[str, pd.DataFrame]:
    comparison = build_entry_rule_comparison(events)
    yearly = build_yearly_stability(events)
    bootstrap = build_lifecycle_bootstrap(events, settings=settings)
    cooldown = build_cooldown_analysis(events, settings=settings)
    extension = build_extension_analysis(events)
    candidates = build_rule_candidates(
        comparison=comparison,
        yearly=yearly,
        bootstrap=bootstrap,
        settings=settings,
    )
    return {
        "entry_rule_comparison": comparison,
        "yearly_stability": yearly,
        "bootstrap": bootstrap,
        "cooldown": cooldown,
        "extension": extension,
        "rule_candidates": candidates,
    }


def _empty_post_outcome_reports() -> dict[str, pd.DataFrame]:
    return {
        "entry_rule_comparison": pd.DataFrame(),
        "yearly_stability": pd.DataFrame(),
        "bootstrap": pd.DataFrame(),
        "cooldown": pd.DataFrame(),
        "extension": pd.DataFrame(),
        "rule_candidates": pd.DataFrame(),
    }


def build_entry_rule_comparison(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period_name, period in period_frames(events):
        for keys, group in period.groupby(
            ["universe_id", "entry_threshold", "confirmation_days", "entry_rule"],
            sort=True,
        ):
            universe_id, threshold, confirmation, rule = keys
            row = {
                "period": period_name,
                "universe_id": universe_id,
                "entry_threshold": float(threshold),
                "confirmation_days": int(confirmation),
                "entry_rule": rule,
                "event_count": int(len(group)),
                "unique_stocks": int(group["stock_id"].nunique()),
                "signal_dates": int(group["signal_date"].nunique()),
                "average_percentile": float(group["universe_percentile"].mean()),
                "average_median_trading_money_20d": float(
                    group["median_trading_money_20d"].mean()
                ),
                "average_median_trading_volume_lots_20d": float(
                    group["median_trading_volume_lots_20d"].mean()
                ),
            }
            row.update(return_metrics(group, "adjusted_return_40d", "return_40d"))
            row.update(
                return_metrics(group, "excess_adjusted_return_40d", "excess_return_40d")
            )
            row.update(
                return_metrics(group, "max_adjusted_return_40d", "maximum_gain_40d")
            )
            row.update(
                return_metrics(group, "min_adjusted_return_40d", "maximum_drawdown_40d")
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_yearly_stability(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in events.groupby(
        ["universe_id", "entry_threshold", "confirmation_days", "entry_rule", "test_year"],
        sort=True,
    ):
        universe_id, threshold, confirmation, rule, year = keys
        row = {
            "universe_id": universe_id,
            "entry_threshold": float(threshold),
            "confirmation_days": int(confirmation),
            "entry_rule": rule,
            "test_year": int(year),
            "event_count": int(len(group)),
            "unique_stocks": int(group["stock_id"].nunique()),
        }
        row.update(
            return_metrics(group, "excess_adjusted_return_40d", "excess_return_40d")
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_lifecycle_bootstrap(
    events: pd.DataFrame,
    *,
    settings: Phase6CTWSELifecycleSettings,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period_name, period in period_frames(events):
        for keys, group in period.groupby(
            ["universe_id", "entry_threshold", "confirmation_days", "entry_rule"],
            sort=True,
        ):
            universe_id, threshold, confirmation, rule = keys
            daily = (
                group.groupby("signal_date")["excess_adjusted_return_40d"]
                .mean()
                .dropna()
                .sort_index()
            )
            monthly = daily.groupby(daily.index.astype(str).str[:7]).mean().sort_index()
            if len(monthly) < 2:
                continue
            lower, upper, mean = moving_block_bootstrap_mean_ci(
                monthly.to_numpy(dtype=np.float64),
                iterations=settings.bootstrap_iterations,
                block_length=settings.bootstrap_block_months,
                random_seed=stable_seed(
                    settings.random_seed,
                    f"{period_name}:{universe_id}:{rule}",
                ),
            )
            rows.append(
                {
                    "period": period_name,
                    "universe_id": universe_id,
                    "entry_threshold": float(threshold),
                    "confirmation_days": int(confirmation),
                    "entry_rule": rule,
                    "daily_observations": int(len(daily)),
                    "monthly_blocks": int(len(monthly)),
                    "point_estimate_excess_return_40d": float(daily.mean()),
                    "bootstrap_mean_excess_return_40d": mean,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "ci_excludes_zero_positive": int(lower > 0),
                    "iterations": settings.bootstrap_iterations,
                    "block_months": settings.bootstrap_block_months,
                }
            )
    return pd.DataFrame(rows)


def build_cooldown_analysis(
    events: pd.DataFrame,
    *,
    settings: Phase6CTWSELifecycleSettings,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period_name, period in period_frames(events):
        for keys, group in period.groupby(
            ["universe_id", "entry_threshold", "confirmation_days", "entry_rule"],
            sort=True,
        ):
            universe_id, threshold, confirmation, rule = keys
            for cooldown in settings.cooldown_days:
                selected = apply_cooldown(group, cooldown_days=cooldown)
                row = {
                    "period": period_name,
                    "universe_id": universe_id,
                    "entry_threshold": float(threshold),
                    "confirmation_days": int(confirmation),
                    "entry_rule": rule,
                    "cooldown_days": cooldown,
                    "event_count": int(len(selected)),
                    "unique_stocks": int(selected["stock_id"].nunique()),
                }
                row.update(
                    return_metrics(
                        selected,
                        "excess_adjusted_return_40d",
                        "excess_return_40d",
                    )
                )
                rows.append(row)
    return pd.DataFrame(rows)


def apply_cooldown(events: pd.DataFrame, *, cooldown_days: int) -> pd.DataFrame:
    if cooldown_days <= 0:
        return events.copy()
    keep: list[int] = []
    for _, group in events.sort_values(
        ["stock_id", "market_day_index"], kind="stable"
    ).groupby("stock_id", sort=False):
        last_kept = -10**9
        for index, row in group.iterrows():
            current = int(row["market_day_index"])
            if current - last_kept > cooldown_days:
                keep.append(index)
                last_kept = current
    return events.loc[keep].copy()


def build_extension_analysis(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period_name, period in period_frames(events):
        for keys, group in period.groupby(
            ["universe_id", "entry_threshold", "confirmation_days", "entry_rule"],
            sort=True,
        ):
            universe_id, threshold, confirmation, rule = keys
            row = {
                "period": period_name,
                "universe_id": universe_id,
                "entry_threshold": float(threshold),
                "confirmation_days": int(confirmation),
                "entry_rule": rule,
                "event_count": int(len(group)),
            }
            for horizon in OUTCOME_HORIZONS:
                row.update(
                    return_metrics(
                        group,
                        f"adjusted_return_{horizon}d",
                        f"return_{horizon}d",
                    )
                )
                row.update(
                    return_metrics(
                        group,
                        f"min_adjusted_return_{horizon}d",
                        f"maximum_drawdown_{horizon}d",
                    )
                )
            row["incremental_return_40_to_60_daily_equal_weight"] = incremental_metric(
                group, 40, 60
            )
            row["incremental_return_40_to_80_daily_equal_weight"] = incremental_metric(
                group, 40, 80
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_rule_candidates(
    *,
    comparison: pd.DataFrame,
    yearly: pd.DataFrame,
    bootstrap: pd.DataFrame,
    settings: Phase6CTWSELifecycleSettings,
) -> pd.DataFrame:
    period = comparison[comparison["period"] == "confirmation_ex_latest"].copy()
    confidence = bootstrap[bootstrap["period"] == "confirmation_ex_latest"].copy()
    keys = ["universe_id", "entry_threshold", "confirmation_days", "entry_rule"]
    merged = period.merge(
        confidence[
            [
                *keys,
                "ci_lower",
                "ci_upper",
                "ci_excludes_zero_positive",
            ]
        ],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        mask = np.ones(len(yearly), dtype=bool)
        for key in keys:
            mask &= yearly[key].astype(str).to_numpy() == str(row[key])
        years = yearly[mask]
        if not years.empty:
            latest_year = int(yearly["test_year"].max())
            years = years[
                (years["test_year"] >= CONFIRMATION_START_YEAR)
                & (years["test_year"] < latest_year)
            ]
        total_years = int(years["test_year"].nunique()) if not years.empty else 0
        positive_years = int(
            (
                pd.to_numeric(
                    years["excess_return_40d_daily_equal_weight"], errors="coerce"
                )
                > 0
            ).sum()
        ) if not years.empty else 0
        required_positive = max(2, math.ceil(total_years * 0.60)) if total_years else 2
        strong = (
            int(row["event_count"]) >= settings.minimum_candidate_events
            and int(row["unique_stocks"]) >= settings.minimum_candidate_stocks
            and total_years >= 3
            and positive_years >= required_positive
            and float(row["excess_return_40d_daily_equal_weight"]) > 0
            and pd.notna(row.get("ci_lower"))
            and float(row["ci_lower"]) > 0
        )
        rows.append(
            {
                **{key: row[key] for key in keys},
                "event_count": int(row["event_count"]),
                "unique_stocks": int(row["unique_stocks"]),
                "confirmation_years": total_years,
                "positive_years": positive_years,
                "required_positive_years": required_positive,
                "excess_return_40d_daily_equal_weight": row[
                    "excess_return_40d_daily_equal_weight"
                ],
                "maximum_drawdown_40d_daily_equal_weight": row.get(
                    "maximum_drawdown_40d_daily_equal_weight", np.nan
                ),
                "ci_lower": row.get("ci_lower", np.nan),
                "ci_upper": row.get("ci_upper", np.nan),
                "candidate_status": "strong_candidate" if strong else "not_selected",
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["candidate_priority"] = (
            result["candidate_status"] == "strong_candidate"
        ).astype(int)
        result = result.sort_values(
            [
                "candidate_priority",
                "ci_lower",
                "excess_return_40d_daily_equal_weight",
            ],
            ascending=[False, False, False],
            kind="stable",
        ).drop(columns="candidate_priority").reset_index(drop=True)
    return result


def build_input_audit(
    *,
    frame: pd.DataFrame,
    universes: pd.DataFrame,
    events: pd.DataFrame,
    coverage: pd.DataFrame,
    settings: Phase6CTWSELifecycleSettings,
    source_path: Path | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {"metric": "phase6c_version", "value": PHASE6C_VERSION},
        {"metric": "market", "value": TARGET_MARKET},
        {"metric": "source_rows", "value": len(frame)},
        {"metric": "source_stocks", "value": frame["stock_id"].nunique()},
        {"metric": "source_signal_dates", "value": frame["signal_date"].nunique()},
        {"metric": "universe_rows", "value": len(universes)},
        {"metric": "universe_count", "value": coverage["universe_id"].nunique()},
        {"metric": "event_rows_before_outcomes", "value": len(events)},
        {
            "metric": "entry_thresholds",
            "value": ",".join(str(value) for value in settings.entry_thresholds),
        },
        {
            "metric": "confirmation_days",
            "value": ",".join(str(value) for value in settings.confirmation_days),
        },
        {
            "metric": "cooldown_days",
            "value": ",".join(str(value) for value in settings.cooldown_days),
        },
        {
            "metric": "liquidity_universes",
            "value": ",".join(value[0] for value in LIQUIDITY_UNIVERSES),
        },
        {
            "metric": "leakage_note",
            "value": "母體、百分位、連續日與成交量門檻只使用訊號日及以前資料；40/60/80日結果只供事後驗證。",
        },
    ]
    if source_path and source_path.exists():
        rows.extend(
            [
                {"metric": "source_path", "value": str(source_path)},
                {"metric": "source_size_bytes", "value": source_path.stat().st_size},
                {"metric": "source_sha256", "value": file_sha256(source_path)},
            ]
        )
    return pd.DataFrame(rows)


def build_phase6c_summary(
    *,
    enriched: pd.DataFrame,
    events: pd.DataFrame,
    candidates: pd.DataFrame,
    settings: Phase6CTWSELifecycleSettings,
) -> pd.DataFrame:
    strong = (
        candidates[candidates["candidate_status"] == "strong_candidate"]
        if not candidates.empty
        else pd.DataFrame()
    )
    latest_year = int(enriched["test_year"].max())
    rows = [
        {"metric": "phase6c_version", "value": PHASE6C_VERSION},
        {"metric": "market", "value": TARGET_MARKET},
        {"metric": "model_target", "value": "same_day_future_return_rank_40d"},
        {"metric": "source_rows", "value": len(enriched)},
        {"metric": "event_rows", "value": len(events)},
        {"metric": "latest_test_year", "value": latest_year},
        {"metric": "strong_candidate_count", "value": len(strong)},
        {"metric": "lifecycle_validation_pass", "value": int(len(strong) > 0)},
        {"metric": "deployment_ready", "value": 0},
        {
            "metric": "candidate_rule",
            "value": "確認期排除最新年度至少3年、正向年度>=60%、事件>=最低數、股票>=最低數、40日超額報酬>0、95% bootstrap下緣>0",
        },
        {
            "metric": "liquidity_design",
            "value": "分別以20m/50m/100m成交金額及100/300張成交量中位數建立獨立同日排名母體。",
        },
        {
            "metric": "deployment_note",
            "value": "Phase 6C 只選生命週期與流動性規格；通過後仍須 Phase 6D 訓練最終模型並整合每日 TWSE 管線。",
        },
    ]
    return pd.DataFrame(rows)


def export_phase6c_reports(
    *,
    output_dir: Path,
    reports: dict[str, pd.DataFrame],
) -> list[Path]:
    paths = [
        _write_csv(output_dir / "phase6c_input_audit.csv", reports["input_audit"]),
        _write_csv(
            output_dir / "phase6c_liquidity_universe_coverage.csv",
            reports["universe_coverage"],
        ),
        _write_gzip_csv(
            output_dir / "phase6c_entry_events.csv.gz", reports["entry_events"]
        ),
        _write_csv(
            output_dir / "phase6c_entry_rule_comparison.csv",
            reports["entry_rule_comparison"],
        ),
        _write_csv(
            output_dir / "phase6c_yearly_stability.csv", reports["yearly_stability"]
        ),
        _write_csv(
            output_dir / "phase6c_bootstrap_confidence.csv", reports["bootstrap"]
        ),
        _write_csv(
            output_dir / "phase6c_cooldown_analysis.csv", reports["cooldown"]
        ),
        _write_csv(
            output_dir / "phase6c_horizon_extension.csv", reports["extension"]
        ),
        _write_csv(
            output_dir / "phase6c_rule_candidates.csv", reports["rule_candidates"]
        ),
        _write_csv(output_dir / "phase6c_summary.csv", reports["summary"]),
    ]
    archive = output_dir / "phase6c_twse_lifecycle_validation_reports.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in paths:
            compression = zipfile.ZIP_STORED if path.suffix == ".gz" else zipfile.ZIP_DEFLATED
            handle.write(path, arcname=path.name, compress_type=compression)
    paths.append(archive)
    return paths


def period_frames(frame: pd.DataFrame) -> tuple[tuple[str, pd.DataFrame], ...]:
    if frame.empty:
        return tuple()
    latest_year = int(frame["test_year"].max())
    return (
        ("all", frame),
        ("development", frame[frame["test_year"] < CONFIRMATION_START_YEAR]),
        ("confirmation", frame[frame["test_year"] >= CONFIRMATION_START_YEAR]),
        (
            "confirmation_ex_latest",
            frame[
                (frame["test_year"] >= CONFIRMATION_START_YEAR)
                & (frame["test_year"] < latest_year)
            ],
        ),
    )


def return_metrics(frame: pd.DataFrame, column: str, prefix: str) -> dict[str, Any]:
    if column not in frame.columns:
        return _empty_return_metrics(prefix)
    values = pd.to_numeric(frame[column], errors="coerce")
    valid = frame[np.isfinite(values)].copy()
    if valid.empty:
        return _empty_return_metrics(prefix)
    values = pd.to_numeric(valid[column], errors="coerce")
    daily = valid.groupby("signal_date")[column].mean()
    return {
        f"{prefix}_sample_count": int(len(valid)),
        f"{prefix}_daily_equal_weight": float(daily.mean()),
        f"{prefix}_sample_average": float(values.mean()),
        f"{prefix}_sample_median": float(values.median()),
        f"{prefix}_positive_rate": float((values > 0).mean()),
    }


def _empty_return_metrics(prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_sample_count": 0,
        f"{prefix}_daily_equal_weight": np.nan,
        f"{prefix}_sample_average": np.nan,
        f"{prefix}_sample_median": np.nan,
        f"{prefix}_positive_rate": np.nan,
    }


def incremental_metric(frame: pd.DataFrame, start: int, end: int) -> float:
    left = pd.to_numeric(frame.get(f"adjusted_return_{start}d"), errors="coerce")
    right = pd.to_numeric(frame.get(f"adjusted_return_{end}d"), errors="coerce")
    valid = np.isfinite(left) & np.isfinite(right) & ((1.0 + left) > 0)
    if not valid.any():
        return np.nan
    values = (1.0 + right[valid]) / (1.0 + left[valid]) - 1.0
    work = pd.DataFrame(
        {"signal_date": frame.loc[valid, "signal_date"].astype(str), "value": values}
    )
    return float(work.groupby("signal_date")["value"].mean().mean())


def rule_name(threshold: float, confirmation: int) -> str:
    return f"top{int(round(100 - threshold))}_confirm{confirmation}d"


def stable_seed(base_seed: int, value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int((base_seed + int.from_bytes(digest[:4], "big")) % (2**32 - 1))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _write_gzip_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "compresslevel": 1, "mtime": 0},
        encoding="utf-8",
    )
    return path
