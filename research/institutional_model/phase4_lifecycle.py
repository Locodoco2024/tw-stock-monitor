from __future__ import annotations

import hashlib
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from research.institutional_model.phase4_selection import (
    moving_block_bootstrap_mean_ci,
)
from research.institutional_model.phase4_stability import CONFIRMATION_START_YEAR


PHASE4F_VERSION = "phase4f-v1"
TARGET_MARKET = "tpex"
TARGET_SCORE_COLUMN = "return_rank_score_daily_percentile"
PRIMARY_HORIZON_DAYS = 20
EXTENSION_HORIZON_DAYS = 40
ENTRY_THRESHOLDS = (80.0, 90.0, 95.0)
CONFIRMATION_DAYS = (1, 2, 3, 5)
COOLDOWN_DAYS = (0, 5, 10, 20)
BREAKDOWN_THRESHOLDS = (80.0, 70.0, 60.0)
REQUIRED_COLUMNS = (
    "stock_id",
    "stock_name",
    "signal_date",
    "signal_year",
    "test_year",
    "adjusted_return_20d",
    "adjusted_return_40d",
    TARGET_SCORE_COLUMN,
)


@dataclass(frozen=True)
class Phase4FLifecycleSettings:
    minimum_daily_stocks: int = 50
    entry_thresholds: tuple[float, ...] = ENTRY_THRESHOLDS
    confirmation_days: tuple[int, ...] = CONFIRMATION_DAYS
    cooldown_days: tuple[int, ...] = COOLDOWN_DAYS
    breakdown_thresholds: tuple[float, ...] = BREAKDOWN_THRESHOLDS
    extension_check_days: int = PRIMARY_HORIZON_DAYS
    maximum_tracking_days: int = EXTENSION_HORIZON_DAYS
    momentum_lookback_days: int = 5
    bootstrap_iterations: int = 1_000
    bootstrap_block_months: int = 3
    random_seed: int = 20260729

    def validate(self) -> None:
        if self.minimum_daily_stocks < 10:
            raise ValueError("Phase 4F 每日最少股票數不可小於 10")
        if tuple(sorted(set(self.entry_thresholds))) != self.entry_thresholds:
            raise ValueError("Phase 4F 進榜門檻必須由小到大且不可重複")
        if any(value <= 0 or value >= 100 for value in self.entry_thresholds):
            raise ValueError("Phase 4F 進榜門檻必須介於 0 與 100 之間")
        if tuple(sorted(set(self.confirmation_days))) != self.confirmation_days:
            raise ValueError("Phase 4F 連續確認日必須由小到大且不可重複")
        if any(value < 1 for value in self.confirmation_days):
            raise ValueError("Phase 4F 連續確認日必須大於 0")
        if tuple(sorted(set(self.cooldown_days))) != self.cooldown_days:
            raise ValueError("Phase 4F 冷卻日必須由小到大且不可重複")
        if any(value < 0 for value in self.cooldown_days):
            raise ValueError("Phase 4F 冷卻日不可小於 0")
        if tuple(sorted(set(self.breakdown_thresholds), reverse=True)) != (
            self.breakdown_thresholds
        ):
            raise ValueError("Phase 4F 轉弱門檻必須由高到低且不可重複")
        if any(value <= 0 or value >= 100 for value in self.breakdown_thresholds):
            raise ValueError("Phase 4F 轉弱門檻必須介於 0 與 100 之間")
        if self.extension_check_days < 1:
            raise ValueError("Phase 4F 延伸檢查日必須大於 0")
        if self.maximum_tracking_days <= self.extension_check_days:
            raise ValueError("Phase 4F 最長追蹤日必須大於延伸檢查日")
        if self.momentum_lookback_days < 1:
            raise ValueError("Phase 4F 動能回看日必須大於 0")
        if self.bootstrap_iterations < 200:
            raise ValueError("Phase 4F bootstrap 次數不可小於 200")
        if self.bootstrap_block_months < 1:
            raise ValueError("Phase 4F bootstrap 區塊月份必須大於 0")


@dataclass(frozen=True)
class Phase4FLifecycleResult:
    status: str
    ready_for_lifecycle_decision: bool
    source_rows: int
    event_rows: int
    output_paths: tuple[Path, ...]


def run_phase4f_lifecycle_research(
    *,
    output_dir: Path | str,
    settings: Phase4FLifecycleSettings | None = None,
    source_path: Path | str | None = None,
) -> Phase4FLifecycleResult:
    """Evaluate lifecycle rules from frozen Phase 4E out-of-sample scores."""
    config = settings or Phase4FLifecycleSettings()
    config.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resolved_source = _resolve_phase4e_source(output, source_path)
    source = pd.read_csv(
        resolved_source,
        compression="infer",
        dtype={"stock_id": "string", "signal_date": "string"},
        low_memory=False,
    )
    reports = evaluate_lifecycle_scores(
        source,
        settings=config,
        source_path=resolved_source,
    )
    paths = export_phase4f_reports(output_dir=output, reports=reports)
    return Phase4FLifecycleResult(
        status="PASS",
        ready_for_lifecycle_decision=True,
        source_rows=len(source),
        event_rows=len(reports["entry_events"]),
        output_paths=tuple(paths),
    )


def evaluate_lifecycle_scores(
    frame: pd.DataFrame,
    *,
    settings: Phase4FLifecycleSettings,
    source_path: Path | None = None,
) -> dict[str, pd.DataFrame]:
    settings.validate()
    base = prepare_lifecycle_frame(frame, settings=settings)
    event_frames: list[pd.DataFrame] = []
    signal_age_frames: list[pd.DataFrame] = []
    for threshold in settings.entry_thresholds:
        annotated = annotate_threshold_runs(
            base,
            threshold=threshold,
            momentum_lookback_days=settings.momentum_lookback_days,
        )
        event_frames.append(
            build_entry_events(
                annotated,
                threshold=threshold,
                confirmation_days=settings.confirmation_days,
            )
        )
        signal_age_frames.append(
            build_signal_age_analysis({threshold: annotated})
        )
        del annotated
    entry_events = pd.concat(event_frames, ignore_index=True)
    signal_age = pd.concat(signal_age_frames, ignore_index=True)
    if entry_events.empty:
        raise RuntimeError("Phase 4F 沒有產生任何進榜事件")
    _validate_entry_events(entry_events)

    entry_rule_comparison = build_entry_rule_comparison(entry_events)
    yearly_stability = build_yearly_stability(entry_events)
    bootstrap = build_lifecycle_bootstrap(entry_events, settings=settings)
    momentum = build_momentum_analysis(entry_events)
    extension = build_extension_analysis(
        base=base,
        events=entry_events,
        settings=settings,
    )
    breakdown_events, breakdown = build_breakdown_analysis(
        base=base,
        events=entry_events,
        settings=settings,
    )
    cooldown = build_cooldown_analysis(entry_events, settings=settings)
    rule_candidates = build_rule_candidates(
        entry_rule_comparison=entry_rule_comparison,
        yearly_stability=yearly_stability,
        bootstrap=bootstrap,
        extension=extension,
        breakdown=breakdown,
    )
    audit = build_input_audit(
        base=base,
        entry_events=entry_events,
        source_path=source_path,
        settings=settings,
    )
    summary = build_lifecycle_summary(
        base=base,
        entry_events=entry_events,
        rule_candidates=rule_candidates,
        settings=settings,
    )
    return {
        "input_audit": audit,
        "entry_events": entry_events,
        "entry_rule_comparison": entry_rule_comparison,
        "yearly_stability": yearly_stability,
        "bootstrap": bootstrap,
        "signal_age": signal_age,
        "momentum": momentum,
        "extension": extension,
        "breakdown_events": breakdown_events,
        "breakdown": breakdown,
        "cooldown": cooldown,
        "rule_candidates": rule_candidates,
        "summary": summary,
    }


def prepare_lifecycle_frame(
    frame: pd.DataFrame,
    *,
    settings: Phase4FLifecycleSettings,
) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Phase 4F 缺少 Phase 4E 欄位：{missing}")
    selected = frame[list(REQUIRED_COLUMNS)].copy()
    selected["stock_id"] = selected["stock_id"].astype(str)
    selected["stock_name"] = selected["stock_name"].fillna("").astype(str)
    selected["signal_date"] = selected["signal_date"].astype(str)
    parsed_dates = pd.to_datetime(selected["signal_date"], errors="coerce")
    if parsed_dates.isna().any():
        raise RuntimeError("Phase 4F 包含無效 signal_date")
    selected["signal_year"] = pd.to_numeric(
        selected["signal_year"], errors="coerce"
    ).astype("Int64")
    selected["test_year"] = pd.to_numeric(
        selected["test_year"], errors="coerce"
    ).astype("Int64")
    if selected[["signal_year", "test_year"]].isna().any().any():
        raise RuntimeError("Phase 4F signal_year／test_year 無效")
    selected["signal_year"] = selected["signal_year"].astype(int)
    selected["test_year"] = selected["test_year"].astype(int)
    selected[TARGET_SCORE_COLUMN] = pd.to_numeric(
        selected[TARGET_SCORE_COLUMN], errors="coerce"
    )
    if not np.isfinite(selected[TARGET_SCORE_COLUMN]).all():
        raise RuntimeError("Phase 4F return rank 百分位包含 NaN／Infinity")
    if not selected[TARGET_SCORE_COLUMN].between(0, 100, inclusive="both").all():
        raise RuntimeError("Phase 4F return rank 百分位超出 0～100")
    for column in ("adjusted_return_20d", "adjusted_return_40d"):
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
        finite_or_missing = np.isfinite(selected[column]) | selected[column].isna()
        if not finite_or_missing.all():
            raise RuntimeError(f"Phase 4F {column} 包含 Infinity")
    duplicate_count = int(selected.duplicated(["stock_id", "signal_date"]).sum())
    if duplicate_count:
        raise RuntimeError(f"Phase 4F 股票＋訊號日重複：{duplicate_count}")
    date_counts = selected.groupby("signal_date")["stock_id"].nunique()
    if int(date_counts.min()) < settings.minimum_daily_stocks:
        raise RuntimeError(
            "Phase 4F 存在低於每日最低股票數的樣本外日期："
            f"最低 {int(date_counts.min())}"
        )
    selected["incremental_return_20_to_40"] = _incremental_return(
        selected["adjusted_return_20d"], selected["adjusted_return_40d"]
    )
    for column in (
        "adjusted_return_20d",
        "adjusted_return_40d",
        "incremental_return_20_to_40",
    ):
        benchmark_column = f"daily_universe_{column}"
        excess_column = f"excess_{column}"
        selected[benchmark_column] = selected.groupby("signal_date")[column].transform(
            "mean"
        )
        selected[excess_column] = selected[column] - selected[benchmark_column]
    ordered_dates = sorted(selected["signal_date"].unique())
    date_index = {value: index for index, value in enumerate(ordered_dates)}
    selected["market_day_index"] = selected["signal_date"].map(date_index).astype(int)
    selected["signal_month"] = selected["signal_date"].str[:7]
    selected = selected.sort_values(
        ["stock_id", "market_day_index"], kind="stable"
    ).reset_index(drop=True)
    return selected


def annotate_threshold_runs(
    base: pd.DataFrame,
    *,
    threshold: float,
    momentum_lookback_days: int,
) -> pd.DataFrame:
    working = base.copy()
    score = working[TARGET_SCORE_COLUMN].to_numpy(dtype=np.float64)
    market_index = working["market_day_index"].to_numpy(dtype=np.int64)
    stock_ids = working["stock_id"].to_numpy(dtype=str)
    qualifies = score >= threshold
    run_length = np.zeros(len(working), dtype=np.int32)
    episode_id = np.zeros(len(working), dtype=np.int32)
    episode_start_index = np.full(len(working), -1, dtype=np.int64)
    current_episode = 0
    previous_stock = ""
    previous_index = -2
    previous_qualified = False
    current_run = 0
    current_start = -1
    for position in range(len(working)):
        stock = stock_ids[position]
        consecutive = stock == previous_stock and market_index[position] == previous_index + 1
        if qualifies[position]:
            if consecutive and previous_qualified:
                current_run += 1
            else:
                current_episode += 1
                current_run = 1
                current_start = int(market_index[position])
            run_length[position] = current_run
            episode_id[position] = current_episode
            episode_start_index[position] = current_start
        else:
            current_run = 0
            current_start = -1
        if stock != previous_stock:
            current_episode = int(episode_id[position])
        previous_stock = stock
        previous_index = int(market_index[position])
        previous_qualified = bool(qualifies[position])

    working["entry_threshold"] = threshold
    working["qualifies"] = qualifies.astype(np.uint8)
    working["run_length"] = run_length
    working["episode_id"] = episode_id
    working["episode_start_index"] = episode_start_index
    qualified = working[working["qualifies"] == 1]
    if qualified.empty:
        working["episode_end_index"] = -1
        working["episode_length"] = 0
    else:
        keys = ["stock_id", "episode_id"]
        episode_end = qualified.groupby(keys)["market_day_index"].transform("max")
        episode_length = qualified.groupby(keys)["market_day_index"].transform("size")
        working["episode_end_index"] = -1
        working["episode_length"] = 0
        working.loc[qualified.index, "episode_end_index"] = episode_end.to_numpy(dtype=int)
        working.loc[qualified.index, "episode_length"] = episode_length.to_numpy(dtype=int)
        working["episode_end_index"] = working["episode_end_index"].astype(int)
        working["episode_length"] = working["episode_length"].astype(int)

    prior = working[["stock_id", "market_day_index", TARGET_SCORE_COLUMN]].copy()
    prior["market_day_index"] = prior["market_day_index"] + momentum_lookback_days
    prior = prior.rename(columns={TARGET_SCORE_COLUMN: "score_lookback"})
    working = working.merge(
        prior,
        on=["stock_id", "market_day_index"],
        how="left",
        validate="one_to_one",
    )
    working["score_change_lookback"] = (
        working[TARGET_SCORE_COLUMN] - working["score_lookback"]
    )
    return working


def build_entry_events(
    annotated: pd.DataFrame,
    *,
    threshold: float,
    confirmation_days: Iterable[int],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    base_columns = [
        "stock_id",
        "stock_name",
        "signal_date",
        "signal_year",
        "test_year",
        "market_day_index",
        TARGET_SCORE_COLUMN,
        "score_lookback",
        "score_change_lookback",
        "episode_id",
        "episode_start_index",
        "episode_end_index",
        "episode_length",
        "adjusted_return_20d",
        "adjusted_return_40d",
        "incremental_return_20_to_40",
        "daily_universe_adjusted_return_20d",
        "daily_universe_adjusted_return_40d",
        "daily_universe_incremental_return_20_to_40",
        "excess_adjusted_return_20d",
        "excess_adjusted_return_40d",
        "excess_incremental_return_20_to_40",
    ]
    for confirmation in confirmation_days:
        selected = annotated[
            (annotated["qualifies"] == 1)
            & (annotated["run_length"] == confirmation)
        ][base_columns].copy()
        if selected.empty:
            continue
        selected["entry_threshold"] = threshold
        selected["confirmation_days"] = int(confirmation)
        selected["entry_rule"] = _rule_name(threshold, confirmation)
        selected["event_id"] = (
            selected["stock_id"].astype(str)
            + ":"
            + selected["entry_rule"]
            + ":"
            + selected["signal_date"].astype(str)
        )
        selected["momentum_bucket"] = selected["score_change_lookback"].map(
            _momentum_bucket
        )
        frames.append(selected)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_entry_rule_comparison(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period_name, period in _period_frames(events):
        for (threshold, confirmation, rule), group in period.groupby(
            ["entry_threshold", "confirmation_days", "entry_rule"], sort=True
        ):
            base = {
                "period": period_name,
                "entry_threshold": threshold,
                "confirmation_days": int(confirmation),
                "entry_rule": rule,
                "event_count": int(len(group)),
                "unique_stocks": int(group["stock_id"].nunique()),
                "signal_dates": int(group["signal_date"].nunique()),
                "average_episode_length": float(group["episode_length"].mean()),
                "average_entry_percentile": float(group[TARGET_SCORE_COLUMN].mean()),
            }
            base.update(_return_metrics(group, "adjusted_return_20d", "return_20d"))
            base.update(_return_metrics(group, "adjusted_return_40d", "return_40d"))
            base.update(
                _return_metrics(group, "excess_adjusted_return_20d", "excess_return_20d")
            )
            base.update(
                _return_metrics(group, "excess_adjusted_return_40d", "excess_return_40d")
            )
            rows.append(base)
    return pd.DataFrame(rows)


def build_yearly_stability(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (threshold, confirmation, rule, year), group in events.groupby(
        ["entry_threshold", "confirmation_days", "entry_rule", "test_year"],
        sort=True,
    ):
        row = {
            "entry_threshold": threshold,
            "confirmation_days": int(confirmation),
            "entry_rule": rule,
            "test_year": int(year),
            "event_count": int(len(group)),
            "unique_stocks": int(group["stock_id"].nunique()),
            "signal_dates": int(group["signal_date"].nunique()),
        }
        row.update(_return_metrics(group, "adjusted_return_20d", "return_20d"))
        row.update(_return_metrics(group, "adjusted_return_40d", "return_40d"))
        row.update(
            _return_metrics(group, "excess_adjusted_return_20d", "excess_return_20d")
        )
        row.update(
            _return_metrics(group, "excess_adjusted_return_40d", "excess_return_40d")
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_lifecycle_bootstrap(
    events: pd.DataFrame,
    *,
    settings: Phase4FLifecycleSettings,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period_name, period in _period_frames(events):
        if period_name not in {"confirmation", "confirmation_ex_latest"}:
            continue
        for (threshold, confirmation, rule), group in period.groupby(
            ["entry_threshold", "confirmation_days", "entry_rule"], sort=True
        ):
            for horizon in (PRIMARY_HORIZON_DAYS, EXTENSION_HORIZON_DAYS):
                column = f"excess_adjusted_return_{horizon}d"
                valid = group[np.isfinite(group[column])]
                daily = valid.groupby("signal_date")[column].mean().sort_index()
                monthly = daily.groupby(daily.index.astype(str).str[:7]).mean().sort_index()
                if len(monthly) < 2:
                    continue
                seed = _stable_seed(
                    settings.random_seed,
                    f"{period_name}:{rule}:{horizon}",
                )
                lower, upper, bootstrap_mean = moving_block_bootstrap_mean_ci(
                    monthly.to_numpy(dtype=np.float64),
                    iterations=settings.bootstrap_iterations,
                    block_length=settings.bootstrap_block_months,
                    random_seed=seed,
                )
                rows.append(
                    {
                        "period": period_name,
                        "entry_threshold": threshold,
                        "confirmation_days": int(confirmation),
                        "entry_rule": rule,
                        "horizon_days": horizon,
                        "event_count": int(len(valid)),
                        "daily_observations": int(len(daily)),
                        "monthly_blocks": int(len(monthly)),
                        "point_estimate_daily_equal_weight_excess_return": float(daily.mean()),
                        "bootstrap_mean_excess_return": bootstrap_mean,
                        "confidence_level": 0.95,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "ci_excludes_zero_positive": int(lower > 0),
                        "iterations": settings.bootstrap_iterations,
                        "block_months": min(settings.bootstrap_block_months, len(monthly)),
                    }
                )
    return pd.DataFrame(rows)


def build_signal_age_analysis(
    annotated_by_threshold: dict[float, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for threshold, annotated in annotated_by_threshold.items():
        selected = annotated[annotated["qualifies"] == 1].copy()
        selected["signal_age_bucket"] = selected["run_length"].map(_age_bucket)
        for period_name, period in _period_frames(selected):
            for age_bucket, group in period.groupby("signal_age_bucket", sort=False):
                row = {
                    "period": period_name,
                    "entry_threshold": threshold,
                    "signal_age_bucket": age_bucket,
                    "observation_count": int(len(group)),
                    "unique_stocks": int(group["stock_id"].nunique()),
                    "signal_dates": int(group["signal_date"].nunique()),
                    "note": "每日狀態樣本彼此重疊，僅用於訊號成熟度比較。",
                }
                row.update(_return_metrics(group, "adjusted_return_20d", "return_20d"))
                row.update(_return_metrics(group, "adjusted_return_40d", "return_40d"))
                row.update(
                    _return_metrics(group, "excess_adjusted_return_20d", "excess_return_20d")
                )
                row.update(
                    _return_metrics(group, "excess_adjusted_return_40d", "excess_return_40d")
                )
                rows.append(row)
    return pd.DataFrame(rows)


def build_momentum_analysis(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period_name, period in _period_frames(events):
        for (threshold, confirmation, rule, bucket), group in period.groupby(
            [
                "entry_threshold",
                "confirmation_days",
                "entry_rule",
                "momentum_bucket",
            ],
            sort=True,
        ):
            row = {
                "period": period_name,
                "entry_threshold": threshold,
                "confirmation_days": int(confirmation),
                "entry_rule": rule,
                "momentum_bucket": bucket,
                "event_count": int(len(group)),
                "average_score_change": float(group["score_change_lookback"].mean())
                if group["score_change_lookback"].notna().any()
                else np.nan,
            }
            row.update(_return_metrics(group, "adjusted_return_20d", "return_20d"))
            row.update(_return_metrics(group, "adjusted_return_40d", "return_40d"))
            row.update(
                _return_metrics(group, "excess_adjusted_return_20d", "excess_return_20d")
            )
            row.update(
                _return_metrics(group, "excess_adjusted_return_40d", "excess_return_40d")
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_extension_analysis(
    *,
    base: pd.DataFrame,
    events: pd.DataFrame,
    settings: Phase4FLifecycleSettings,
) -> pd.DataFrame:
    lookup = base[
        ["stock_id", "market_day_index", "signal_date", TARGET_SCORE_COLUMN]
    ].copy()
    lookup["market_day_index"] = (
        lookup["market_day_index"] - settings.extension_check_days
    )
    lookup = lookup.rename(
        columns={
            "signal_date": "extension_signal_date",
            TARGET_SCORE_COLUMN: "extension_percentile",
        }
    )
    extended = events.merge(
        lookup,
        on=["stock_id", "market_day_index"],
        how="left",
        validate="many_to_one",
    )
    extended["extension_state"] = [
        _extension_state(value, threshold)
        for value, threshold in zip(
            extended["extension_percentile"],
            extended["entry_threshold"],
            strict=True,
        )
    ]
    rows: list[dict[str, Any]] = []
    for period_name, period in _period_frames(extended):
        for (threshold, confirmation, rule, state), group in period.groupby(
            [
                "entry_threshold",
                "confirmation_days",
                "entry_rule",
                "extension_state",
            ],
            sort=True,
        ):
            row = {
                "period": period_name,
                "entry_threshold": threshold,
                "confirmation_days": int(confirmation),
                "entry_rule": rule,
                "extension_state": state,
                "event_count": int(len(group)),
                "extension_observed_count": int(group["extension_percentile"].notna().sum()),
                "average_extension_percentile": float(group["extension_percentile"].mean())
                if group["extension_percentile"].notna().any()
                else np.nan,
            }
            row.update(_return_metrics(group, "adjusted_return_20d", "return_20d"))
            row.update(_return_metrics(group, "adjusted_return_40d", "return_40d"))
            row.update(
                _return_metrics(group, "excess_adjusted_return_20d", "excess_return_20d")
            )
            row.update(
                _return_metrics(group, "excess_adjusted_return_40d", "excess_return_40d")
            )
            row.update(
                _return_metrics(
                    group,
                    "incremental_return_20_to_40",
                    "incremental_return_20_to_40",
                )
            )
            row.update(
                _return_metrics(
                    group,
                    "excess_incremental_return_20_to_40",
                    "excess_incremental_return_20_to_40",
                )
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_breakdown_analysis(
    *,
    base: pd.DataFrame,
    events: pd.DataFrame,
    settings: Phase4FLifecycleSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_stock: dict[
        str,
        dict[int, tuple[float, float, float, float, float, str]],
    ] = {}
    for stock_id, group in base.groupby("stock_id", sort=False):
        per_stock[str(stock_id)] = {
            int(row.market_day_index): (
                float(getattr(row, TARGET_SCORE_COLUMN)),
                float(row.adjusted_return_20d)
                if pd.notna(row.adjusted_return_20d)
                else np.nan,
                float(row.adjusted_return_40d)
                if pd.notna(row.adjusted_return_40d)
                else np.nan,
                float(row.excess_adjusted_return_20d)
                if pd.notna(row.excess_adjusted_return_20d)
                else np.nan,
                float(row.excess_adjusted_return_40d)
                if pd.notna(row.excess_adjusted_return_40d)
                else np.nan,
                str(row.signal_date),
            )
            for row in group.itertuples(index=False)
        }
    event_rows: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        stock_lookup = per_stock[str(event.stock_id)]
        entry_index = int(event.market_day_index)
        for threshold in settings.breakdown_thresholds:
            first_break: tuple[int, float, float, float, float, float, str] | None = None
            missing_before_resolution = False
            for offset in range(1, settings.extension_check_days + 1):
                row = stock_lookup.get(entry_index + offset)
                if row is None:
                    missing_before_resolution = True
                    continue
                (
                    score,
                    return_20d,
                    return_40d,
                    excess_return_20d,
                    excess_return_40d,
                    signal_date,
                ) = row
                if score < threshold:
                    first_break = (
                        offset,
                        score,
                        return_20d,
                        return_40d,
                        excess_return_20d,
                        excess_return_40d,
                        signal_date,
                    )
                    break
            if missing_before_resolution and first_break is None:
                state = "incomplete_path"
            elif first_break is None:
                state = "no_breakdown_20d"
            elif missing_before_resolution:
                state = "breakdown_timing_uncertain"
            else:
                state = _breakdown_timing_bucket(first_break[0])
            event_rows.append(
                {
                    "event_id": event.event_id,
                    "stock_id": event.stock_id,
                    "stock_name": event.stock_name,
                    "signal_date": event.signal_date,
                    "test_year": int(event.test_year),
                    "entry_threshold": float(event.entry_threshold),
                    "confirmation_days": int(event.confirmation_days),
                    "entry_rule": event.entry_rule,
                    "breakdown_threshold": threshold,
                    "breakdown_state": state,
                    "breakdown_offset_days": first_break[0] if first_break else np.nan,
                    "breakdown_signal_date": first_break[6] if first_break else "",
                    "breakdown_percentile": first_break[1] if first_break else np.nan,
                    "entry_return_20d": event.adjusted_return_20d,
                    "entry_return_40d": event.adjusted_return_40d,
                    "entry_excess_return_20d": event.excess_adjusted_return_20d,
                    "entry_excess_return_40d": event.excess_adjusted_return_40d,
                    "post_break_return_20d": first_break[2] if first_break else np.nan,
                    "post_break_return_40d": first_break[3] if first_break else np.nan,
                    "post_break_excess_return_20d": first_break[4] if first_break else np.nan,
                    "post_break_excess_return_40d": first_break[5] if first_break else np.nan,
                }
            )
    breakdown_events = pd.DataFrame(event_rows)
    rows: list[dict[str, Any]] = []
    for period_name, period in _period_frames(breakdown_events):
        for (
            threshold,
            confirmation,
            rule,
            breakdown_threshold,
            state,
        ), group in period.groupby(
            [
                "entry_threshold",
                "confirmation_days",
                "entry_rule",
                "breakdown_threshold",
                "breakdown_state",
            ],
            sort=True,
        ):
            row = {
                "period": period_name,
                "entry_threshold": threshold,
                "confirmation_days": int(confirmation),
                "entry_rule": rule,
                "breakdown_threshold": breakdown_threshold,
                "breakdown_state": state,
                "event_count": int(len(group)),
                "average_breakdown_offset_days": float(
                    group["breakdown_offset_days"].mean()
                )
                if group["breakdown_offset_days"].notna().any()
                else np.nan,
            }
            row.update(_return_metrics(group, "entry_return_20d", "entry_return_20d"))
            row.update(_return_metrics(group, "entry_return_40d", "entry_return_40d"))
            row.update(
                _return_metrics(group, "post_break_return_20d", "post_break_return_20d")
            )
            row.update(
                _return_metrics(group, "post_break_return_40d", "post_break_return_40d")
            )
            row.update(
                _return_metrics(group, "entry_excess_return_20d", "entry_excess_return_20d")
            )
            row.update(
                _return_metrics(group, "entry_excess_return_40d", "entry_excess_return_40d")
            )
            row.update(
                _return_metrics(
                    group,
                    "post_break_excess_return_20d",
                    "post_break_excess_return_20d",
                )
            )
            row.update(
                _return_metrics(
                    group,
                    "post_break_excess_return_40d",
                    "post_break_excess_return_40d",
                )
            )
            rows.append(row)
    return breakdown_events, pd.DataFrame(rows)


def build_cooldown_analysis(
    events: pd.DataFrame,
    *,
    settings: Phase4FLifecycleSettings,
) -> pd.DataFrame:
    accepted_frames: list[pd.DataFrame] = []
    for (threshold, confirmation, rule), rule_events in events.groupby(
        ["entry_threshold", "confirmation_days", "entry_rule"], sort=True
    ):
        for cooldown in settings.cooldown_days:
            accepted_indices: list[int] = []
            for _, stock_events in rule_events.groupby("stock_id", sort=False):
                last_end = -10**9
                for index, row in stock_events.sort_values("market_day_index").iterrows():
                    if int(row["market_day_index"]) > last_end + cooldown:
                        accepted_indices.append(index)
                        last_end = int(row["episode_end_index"])
            selected = rule_events.loc[accepted_indices].copy()
            selected["cooldown_days"] = cooldown
            accepted_frames.append(selected)
    accepted = pd.concat(accepted_frames, ignore_index=True)
    rows: list[dict[str, Any]] = []
    for period_name, period in _period_frames(accepted):
        for (threshold, confirmation, rule, cooldown), group in period.groupby(
            [
                "entry_threshold",
                "confirmation_days",
                "entry_rule",
                "cooldown_days",
            ],
            sort=True,
        ):
            row = {
                "period": period_name,
                "entry_threshold": threshold,
                "confirmation_days": int(confirmation),
                "entry_rule": rule,
                "cooldown_days": int(cooldown),
                "accepted_event_count": int(len(group)),
                "unique_stocks": int(group["stock_id"].nunique()),
                "signal_dates": int(group["signal_date"].nunique()),
            }
            row.update(_return_metrics(group, "adjusted_return_20d", "return_20d"))
            row.update(_return_metrics(group, "adjusted_return_40d", "return_40d"))
            row.update(
                _return_metrics(group, "excess_adjusted_return_20d", "excess_return_20d")
            )
            row.update(
                _return_metrics(group, "excess_adjusted_return_40d", "excess_return_40d")
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_rule_candidates(
    *,
    entry_rule_comparison: pd.DataFrame,
    yearly_stability: pd.DataFrame,
    bootstrap: pd.DataFrame,
    extension: pd.DataFrame,
    breakdown: pd.DataFrame,
) -> pd.DataFrame:
    selected = entry_rule_comparison[
        entry_rule_comparison["period"] == "confirmation_ex_latest"
    ].copy()
    if selected.empty:
        return pd.DataFrame()
    year_selected = yearly_stability[
        (yearly_stability["test_year"] >= CONFIRMATION_START_YEAR)
        & (yearly_stability["test_year"] < yearly_stability["test_year"].max())
    ]
    year_summary = (
        year_selected.groupby(["entry_threshold", "confirmation_days", "entry_rule"])
        .agg(
            evaluated_years=("test_year", "nunique"),
            positive_excess_20d_years=(
                "excess_return_20d_daily_equal_weight",
                lambda value: int((value > 0).sum()),
            ),
            positive_excess_40d_years=(
                "excess_return_40d_daily_equal_weight",
                lambda value: int((value > 0).sum()),
            ),
            worst_year_excess_20d=("excess_return_20d_daily_equal_weight", "min"),
            worst_year_excess_40d=("excess_return_40d_daily_equal_weight", "min"),
        )
        .reset_index()
    )
    boot = bootstrap[
        bootstrap["period"] == "confirmation_ex_latest"
    ].pivot_table(
        index=["entry_threshold", "confirmation_days", "entry_rule"],
        columns="horizon_days",
        values=["ci_lower", "ci_upper", "ci_excludes_zero_positive"],
        aggfunc="first",
    )
    boot.columns = [f"bootstrap_{metric}_{int(horizon)}d" for metric, horizon in boot.columns]
    boot = boot.reset_index()
    ext = extension[
        (extension["period"] == "confirmation_ex_latest")
        & (extension["extension_state"] == "maintained_entry_band")
    ][
        [
            "entry_threshold",
            "confirmation_days",
            "entry_rule",
            "event_count",
            "excess_incremental_return_20_to_40_daily_equal_weight",
        ]
    ].rename(
        columns={
            "event_count": "maintained_band_event_count_day20",
            "excess_incremental_return_20_to_40_daily_equal_weight": (
                "maintained_band_excess_incremental_return_20_to_40"
            ),
        }
    )
    no_break = breakdown[
        (breakdown["period"] == "confirmation_ex_latest")
        & (breakdown["breakdown_threshold"] == 80.0)
        & (breakdown["breakdown_state"] == "no_breakdown_20d")
    ][
        [
            "entry_threshold",
            "confirmation_days",
            "entry_rule",
            "event_count",
            "entry_excess_return_20d_daily_equal_weight",
        ]
    ].rename(
        columns={
            "event_count": "no_top20_breakdown_event_count",
            "entry_excess_return_20d_daily_equal_weight": (
                "no_top20_breakdown_excess_return_20d"
            ),
        }
    )
    result = selected.merge(
        year_summary,
        on=["entry_threshold", "confirmation_days", "entry_rule"],
        how="left",
        validate="one_to_one",
    )
    result = result.merge(
        boot,
        on=["entry_threshold", "confirmation_days", "entry_rule"],
        how="left",
        validate="one_to_one",
    )
    result = result.merge(
        ext,
        on=["entry_threshold", "confirmation_days", "entry_rule"],
        how="left",
        validate="one_to_one",
    )
    result = result.merge(
        no_break,
        on=["entry_threshold", "confirmation_days", "entry_rule"],
        how="left",
        validate="one_to_one",
    )
    result["decision_status"] = "REQUIRES_MANUAL_REVIEW"
    result["decision_note"] = (
        "不可只依絕對平均報酬選規則；須核對相對同日母體超額報酬、年度方向、bootstrap、通知頻率與轉弱結果。"
    )
    return result.sort_values(["entry_threshold", "confirmation_days"]).reset_index(drop=True)


def build_input_audit(
    *,
    base: pd.DataFrame,
    entry_events: pd.DataFrame,
    source_path: Path | None,
    settings: Phase4FLifecycleSettings,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {"metric": "phase4f_version", "value": PHASE4F_VERSION},
        {"metric": "market", "value": TARGET_MARKET},
        {"metric": "score_column", "value": TARGET_SCORE_COLUMN},
        {"metric": "source_rows", "value": len(base)},
        {"metric": "source_stocks", "value": base["stock_id"].nunique()},
        {"metric": "source_signal_dates", "value": base["signal_date"].nunique()},
        {"metric": "first_signal_date", "value": base["signal_date"].min()},
        {"metric": "last_signal_date", "value": base["signal_date"].max()},
        {"metric": "duplicate_stock_date", "value": int(base.duplicated(["stock_id", "signal_date"]).sum())},
        {"metric": "minimum_daily_stocks", "value": int(base.groupby("signal_date")["stock_id"].nunique().min())},
        {"metric": "entry_event_rows", "value": len(entry_events)},
        {"metric": "entry_rules", "value": entry_events["entry_rule"].nunique()},
        {"metric": "entry_thresholds", "value": ",".join(str(value) for value in settings.entry_thresholds)},
        {"metric": "confirmation_days", "value": ",".join(str(value) for value in settings.confirmation_days)},
        {"metric": "cooldown_days", "value": ",".join(str(value) for value in settings.cooldown_days)},
        {"metric": "breakdown_thresholds", "value": ",".join(str(value) for value in settings.breakdown_thresholds)},
    ]
    if source_path and source_path.exists():
        rows.extend(
            [
                {"metric": "source_path", "value": str(source_path)},
                {"metric": "source_size_bytes", "value": source_path.stat().st_size},
                {"metric": "source_sha256", "value": _file_sha256(source_path)},
            ]
        )
    return pd.DataFrame(rows)


def build_lifecycle_summary(
    *,
    base: pd.DataFrame,
    entry_events: pd.DataFrame,
    rule_candidates: pd.DataFrame,
    settings: Phase4FLifecycleSettings,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {"metric": "phase4f_version", "value": PHASE4F_VERSION},
        {"metric": "pipeline_status", "value": "PASS"},
        {"metric": "market", "value": TARGET_MARKET},
        {"metric": "primary_horizon_days", "value": PRIMARY_HORIZON_DAYS},
        {"metric": "extension_horizon_days", "value": EXTENSION_HORIZON_DAYS},
        {"metric": "score_column", "value": TARGET_SCORE_COLUMN},
        {"metric": "source_rows", "value": len(base)},
        {"metric": "event_rows", "value": len(entry_events)},
        {"metric": "entry_rule_count", "value": entry_events["entry_rule"].nunique()},
        {"metric": "candidate_rule_rows", "value": len(rule_candidates)},
        {"metric": "latest_test_year", "value": int(base["test_year"].max())},
        {"metric": "ready_for_lifecycle_decision", "value": 1},
        {
            "metric": "leakage_note",
            "value": "進榜、連續確認、動能與冷卻只使用當日及過去分數；第20日狀態與轉弱僅作未來驗證。",
        },
        {
            "metric": "decision_note",
            "value": "PASS 只代表生命週期報告完整，不代表任一通知規則已自動勝出。",
        },
    ]
    return pd.DataFrame(rows)


def export_phase4f_reports(
    *,
    output_dir: Path,
    reports: dict[str, pd.DataFrame],
) -> list[Path]:
    paths = [
        _write_csv(output_dir / "phase4f_input_audit.csv", reports["input_audit"]),
        _write_gzip_csv(output_dir / "phase4f_entry_events.csv.gz", reports["entry_events"]),
        _write_csv(
            output_dir / "phase4f_entry_rule_comparison.csv",
            reports["entry_rule_comparison"],
        ),
        _write_csv(
            output_dir / "phase4f_yearly_stability.csv",
            reports["yearly_stability"],
        ),
        _write_csv(
            output_dir / "phase4f_bootstrap_confidence.csv", reports["bootstrap"]
        ),
        _write_csv(
            output_dir / "phase4f_signal_age_analysis.csv", reports["signal_age"]
        ),
        _write_csv(
            output_dir / "phase4f_momentum_analysis.csv", reports["momentum"]
        ),
        _write_csv(
            output_dir / "phase4f_extension_analysis.csv", reports["extension"]
        ),
        _write_gzip_csv(
            output_dir / "phase4f_breakdown_events.csv.gz",
            reports["breakdown_events"],
        ),
        _write_csv(
            output_dir / "phase4f_breakdown_analysis.csv", reports["breakdown"]
        ),
        _write_csv(
            output_dir / "phase4f_cooldown_analysis.csv", reports["cooldown"]
        ),
        _write_csv(
            output_dir / "phase4f_rule_candidates.csv", reports["rule_candidates"]
        ),
        _write_csv(output_dir / "phase4f_summary.csv", reports["summary"]),
    ]
    archive = output_dir / "phase4f_lifecycle_validation_reports.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in paths:
            compression = zipfile.ZIP_STORED if path.suffix == ".gz" else zipfile.ZIP_DEFLATED
            handle.write(path, arcname=path.name, compress_type=compression)
    paths.append(archive)
    return paths


def _resolve_phase4e_source(output_dir: Path, source_path: Path | str | None) -> Path:
    if source_path is not None:
        candidate = Path(source_path)
        if not candidate.exists():
            raise FileNotFoundError(f"找不到 Phase 4E OOS 分數：{candidate}")
        return candidate
    candidate = output_dir / "phase4e_oos_scores.csv.gz"
    if candidate.exists():
        return candidate
    archive = output_dir / "phase4e_target_validation_reports.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as handle:
            member = "phase4e_oos_scores.csv.gz"
            if member not in handle.namelist():
                raise RuntimeError("Phase 4E 驗證 ZIP 缺少 phase4e_oos_scores.csv.gz")
            extracted = output_dir / member
            extracted.write_bytes(handle.read(member))
            return extracted
    raise FileNotFoundError(
        "找不到 phase4e_oos_scores.csv.gz；請先完成 Phase 4E。"
    )


def _validate_entry_events(events: pd.DataFrame) -> None:
    duplicate_count = int(events.duplicated("event_id").sum())
    if duplicate_count:
        raise RuntimeError(f"Phase 4F event_id 重複：{duplicate_count}")
    invalid_confirmation = events["confirmation_days"] > events["episode_length"]
    if invalid_confirmation.any():
        raise RuntimeError("Phase 4F 確認日超過 episode 長度")
    if not (
        events["market_day_index"]
        == events["episode_start_index"] + events["confirmation_days"] - 1
    ).all():
        raise RuntimeError("Phase 4F 連續確認事件日期計算不一致")


def _return_metrics(
    frame: pd.DataFrame,
    column: str,
    prefix: str,
) -> dict[str, Any]:
    values = pd.to_numeric(frame[column], errors="coerce")
    valid = frame[np.isfinite(values)].copy()
    values = pd.to_numeric(valid[column], errors="coerce")
    if valid.empty:
        return {
            f"{prefix}_sample_count": 0,
            f"{prefix}_daily_equal_weight": np.nan,
            f"{prefix}_sample_average": np.nan,
            f"{prefix}_sample_median": np.nan,
            f"{prefix}_positive_rate": np.nan,
            f"{prefix}_up_5pct_rate": np.nan,
            f"{prefix}_down_5pct_rate": np.nan,
        }
    daily = valid.groupby("signal_date")[column].mean()
    return {
        f"{prefix}_sample_count": int(len(valid)),
        f"{prefix}_daily_equal_weight": float(daily.mean()),
        f"{prefix}_sample_average": float(values.mean()),
        f"{prefix}_sample_median": float(values.median()),
        f"{prefix}_positive_rate": float((values > 0).mean()),
        f"{prefix}_up_5pct_rate": float((values >= 0.05).mean()),
        f"{prefix}_down_5pct_rate": float((values <= -0.05).mean()),
    }


def _period_frames(frame: pd.DataFrame) -> tuple[tuple[str, pd.DataFrame], ...]:
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


def _rule_name(threshold: float, confirmation: int) -> str:
    return f"top{int(round(100 - threshold))}_confirm{confirmation}d"


def _momentum_bucket(value: Any) -> str:
    if pd.isna(value):
        return "unavailable"
    number = float(value)
    if number <= 0:
        return "declining_or_flat"
    if number < 10:
        return "rise_0_to_10"
    if number < 20:
        return "rise_10_to_20"
    if number < 30:
        return "rise_20_to_30"
    return "rise_30_plus"


def _age_bucket(value: Any) -> str:
    number = int(value)
    if number == 1:
        return "day_1_first_entry"
    if number == 2:
        return "day_2"
    if number <= 5:
        return "day_3_to_5"
    if number <= 10:
        return "day_6_to_10"
    if number <= 20:
        return "day_11_to_20"
    return "day_21_plus"


def _extension_state(value: Any, entry_threshold: float) -> str:
    if pd.isna(value):
        return "missing_day20_score"
    number = float(value)
    if number >= entry_threshold:
        return "maintained_entry_band"
    if number >= 80:
        return "weakened_but_still_top20"
    return "below_top20"


def _breakdown_timing_bucket(offset: int) -> str:
    if offset <= 5:
        return "breakdown_day_1_to_5"
    if offset <= 10:
        return "breakdown_day_6_to_10"
    return "breakdown_day_11_to_20"


def _incremental_return(
    return_20d: pd.Series,
    return_40d: pd.Series,
) -> pd.Series:
    left = pd.to_numeric(return_20d, errors="coerce")
    right = pd.to_numeric(return_40d, errors="coerce")
    valid = np.isfinite(left) & np.isfinite(right) & ((1.0 + left) > 0)
    result = pd.Series(np.nan, index=return_20d.index, dtype=float)
    result.loc[valid] = (1.0 + right.loc[valid]) / (1.0 + left.loc[valid]) - 1.0
    return result


def _stable_seed(base_seed: int, value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int((base_seed + int.from_bytes(digest[:4], "big")) % (2**32 - 1))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
