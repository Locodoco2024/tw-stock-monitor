from __future__ import annotations

import hashlib
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from research.institutional_model.database import ResearchDatabase
from research.institutional_model.phase4_selection import (
    moving_block_bootstrap_mean_ci,
    resolve_phase3_shard_directory,
)


PHASE5I_VERSION = "phase5i-v2"
FORMAL_ENTRY_RULES = ("top10_confirm1d", "top20_confirm5d")
ACTORS = {
    "selected_total": "三法人合計",
    "foreign": "外資",
    "investment_trust": "投信",
    "dealer_self": "自營商自行買賣",
}
ACTOR_COLUMNS = {
    "selected_total": "selected_total_net",
    "foreign": "foreign_net",
    "investment_trust": "investment_trust_net",
    "dealer_self": "dealer_self_net",
}
COST_WINDOWS = (5, 10, 20)
FIXED_DEVIATION_BUCKETS = (
    (-math.inf, -0.02, "below_cost_more_than_2pct"),
    (-0.02, 0.02, "near_cost_plus_minus_2pct"),
    (0.02, 0.05, "above_cost_2_to_5pct"),
    (0.05, 0.10, "above_cost_5_to_10pct"),
    (0.10, 0.15, "above_cost_10_to_15pct"),
    (0.15, math.inf, "above_cost_15pct_or_more"),
)


@dataclass(frozen=True)
class Phase5ISettings:
    entry_rules: tuple[str, ...] = FORMAL_ENTRY_RULES
    cost_windows: tuple[int, ...] = COST_WINDOWS
    minimum_proxy_events: int = 100
    bootstrap_iterations: int = 1_000
    bootstrap_block_months: int = 3
    random_seed: int = 20260731

    def validate(self) -> None:
        if not self.entry_rules:
            raise ValueError("Phase 5I 至少需要一種正式進榜規則")
        if tuple(sorted(set(self.cost_windows))) != self.cost_windows:
            raise ValueError("Phase 5I 成本窗口必須由小到大且不可重複")
        if any(value < 2 for value in self.cost_windows):
            raise ValueError("Phase 5I 成本窗口不可小於 2")
        if self.minimum_proxy_events < 20:
            raise ValueError("Phase 5I 每個代理最低事件數不可低於 20")
        if self.bootstrap_iterations < 200:
            raise ValueError("Phase 5I bootstrap 次數不可低於 200")
        if self.bootstrap_block_months < 1:
            raise ValueError("Phase 5I bootstrap 區塊月份不可小於 1")


@dataclass(frozen=True)
class Phase5IResult:
    status: str
    source_events: int
    enriched_events: int
    output_paths: tuple[Path, ...]


def run_phase5i_entry_risk_research(
    *,
    database: ResearchDatabase,
    output_dir: Path | str,
    shard_root: Path | str,
    settings: Phase5ISettings | None = None,
    source_path: Path | str | None = None,
) -> Phase5IResult:
    config = settings or Phase5ISettings()
    config.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resolved_source = resolve_phase4f_events(output, source_path)
    events = pd.read_csv(
        resolved_source,
        compression="infer",
        dtype={"stock_id": "string", "signal_date": "string"},
        low_memory=False,
    )
    events = prepare_phase5i_events(events, settings=config)
    shard_dir = resolve_phase3_shard_directory(
        output_dir=output,
        shard_root=Path(shard_root),
    )
    outcomes = load_phase3_event_outcomes(events, shard_dir=shard_dir)
    history = load_event_market_history(database, events, settings=config)
    features = build_event_cost_features(events, history=history, settings=config)
    enriched = events.merge(
        features,
        on=["stock_id", "signal_date"],
        how="left",
        validate="many_to_one",
    ).merge(
        outcomes,
        on=["stock_id", "signal_date"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_phase3"),
    )
    enriched = finalize_event_features(enriched, settings=config)
    reports = evaluate_entry_risk(enriched, settings=config, source_path=resolved_source)
    paths = export_phase5i_reports(output_dir=output, reports=reports)
    return Phase5IResult(
        status="PASS",
        source_events=len(events),
        enriched_events=len(enriched),
        output_paths=tuple(paths),
    )


def prepare_phase5i_events(
    frame: pd.DataFrame,
    *,
    settings: Phase5ISettings,
) -> pd.DataFrame:
    required = {
        "event_id",
        "entry_rule",
        "stock_id",
        "stock_name",
        "signal_date",
        "test_year",
        "return_rank_score_daily_percentile",
        "adjusted_return_20d",
        "adjusted_return_40d",
        "excess_adjusted_return_20d",
        "excess_adjusted_return_40d",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"Phase 5I 缺少 Phase 4F 事件欄位：{missing}")
    selected = frame[frame["entry_rule"].isin(settings.entry_rules)].copy()
    if selected.empty:
        raise RuntimeError("Phase 5I 找不到正式通知規則事件")
    selected["stock_id"] = selected["stock_id"].astype(str)
    selected["signal_date"] = pd.to_datetime(
        selected["signal_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    if selected["signal_date"].isna().any():
        raise RuntimeError("Phase 5I 事件包含無效 signal_date")
    selected["event_type"] = selected["entry_rule"].map(
        {
            "top10_confirm1d": "NEW_CANDIDATE",
            "top20_confirm5d": "LAYOUT_CONFIRMED_DIRECT",
        }
    ).fillna("OTHER")
    selected["test_year"] = pd.to_numeric(selected["test_year"], errors="coerce")
    if selected["test_year"].isna().any():
        raise RuntimeError("Phase 5I test_year 無效")
    selected["test_year"] = selected["test_year"].astype(int)
    duplicate_count = int(selected.duplicated("event_id").sum())
    if duplicate_count:
        raise RuntimeError(f"Phase 5I event_id 重複：{duplicate_count}")
    return selected.sort_values(["signal_date", "stock_id", "entry_rule"]).reset_index(
        drop=True
    )


def resolve_phase4f_events(output_dir: Path, source_path: Path | str | None) -> Path:
    if source_path is not None:
        candidate = Path(source_path)
        if not candidate.exists():
            raise FileNotFoundError(f"找不到 Phase 4F 事件檔：{candidate}")
        return candidate
    candidate = output_dir / "phase4f_entry_events.csv.gz"
    if candidate.exists():
        return candidate
    archive = output_dir / "phase4f_lifecycle_validation_reports.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as handle:
            member = "phase4f_entry_events.csv.gz"
            if member not in handle.namelist():
                raise RuntimeError("Phase 4F ZIP 缺少 phase4f_entry_events.csv.gz")
            candidate.write_bytes(handle.read(member))
            return candidate
    raise FileNotFoundError("找不到 Phase 4F 事件；請先完成 Phase 4F。")


def load_phase3_event_outcomes(events: pd.DataFrame, *, shard_dir: Path) -> pd.DataFrame:
    columns = [
        "stock_id",
        "signal_date",
        "entry_date_20d",
        "entry_open_20d",
        "max_adjusted_return_20d",
        "min_adjusted_return_20d",
        "max_adjusted_return_40d",
        "min_adjusted_return_40d",
    ]
    phase3_columns = columns[:6]
    key_dates = (
        events.groupby("stock_id")["signal_date"].agg(lambda values: set(values)).to_dict()
    )
    frames: list[pd.DataFrame] = []
    for stock_id, dates in key_dates.items():
        path = shard_dir / f"{stock_id}.csv.gz"
        if not path.exists():
            continue
        shard = pd.read_csv(
            path,
            compression="gzip",
            usecols=lambda column: column in columns,
            dtype={"stock_id": "string", "signal_date": "string"},
            low_memory=False,
        )
        selected = shard[shard["signal_date"].astype(str).isin(dates)].copy()
        if not selected.empty:
            frames.append(selected)
    if not frames:
        raise RuntimeError("Phase 5I 無法從 Phase 3 分片取得事件進場與回撤資料")
    result = pd.concat(frames, ignore_index=True)
    result["stock_id"] = result["stock_id"].astype(str)
    result["signal_date"] = pd.to_datetime(
        result["signal_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    duplicate_count = int(result.duplicated(["stock_id", "signal_date"]).sum())
    if duplicate_count:
        raise RuntimeError(f"Phase 5I Phase 3 結果重複：{duplicate_count}")

    # Phase 3 的固定規格只有 5／10／20 日標籤。40 日最大上漲與最大回撤
    # 是 Phase 4D 額外計算並存放於 research/data/phase4d_cache。
    cache_extrema = _load_phase4d_40d_extrema(
        key_dates=key_dates,
        shard_dir=shard_dir,
    )
    if not cache_extrema.empty:
        result = result.merge(
            cache_extrema,
            on=["stock_id", "signal_date"],
            how="left",
            validate="one_to_one",
            suffixes=("", "_phase4d"),
        )
        for column in ("max_adjusted_return_40d", "min_adjusted_return_40d"):
            cache_column = f"{column}_phase4d"
            if cache_column not in result.columns:
                continue
            if column not in result.columns:
                result[column] = result[cache_column]
            else:
                result[column] = result[column].fillna(result[cache_column])
            result = result.drop(columns=[cache_column])

    # 沒有 Phase 4D cache 時仍可完成成本偏離與 20 日追價風險研究；
    # 40 日最大上漲／回撤欄位保留為空，不可用不存在的資料冒充結果。
    for column in columns:
        if column not in result.columns:
            result[column] = np.nan
    for column in columns[3:]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result[columns]


def _load_phase4d_40d_extrema(
    *,
    key_dates: dict[str, set[str]],
    shard_dir: Path,
) -> pd.DataFrame:
    columns = [
        "stock_id",
        "signal_date",
        "max_adjusted_return_40d",
        "min_adjusted_return_40d",
    ]
    cache_root = shard_dir.parent.parent / "phase4d_cache"
    if not cache_root.is_dir():
        return pd.DataFrame(columns=columns)
    cache_directories = sorted(
        (path for path in cache_root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    frames: list[pd.DataFrame] = []
    for stock_id, dates in key_dates.items():
        unresolved = set(dates)
        stock_frames: list[pd.DataFrame] = []
        for directory in cache_directories:
            if not unresolved:
                break
            path = directory / f"{stock_id}_40d.csv.gz"
            if not path.exists():
                continue
            try:
                cached = pd.read_csv(
                    path,
                    compression="gzip",
                    usecols=lambda column: column in columns,
                    dtype={"stock_id": "string", "signal_date": "string"},
                    low_memory=False,
                )
            except (OSError, ValueError, KeyError):
                continue
            required = set(columns)
            if not required.issubset(cached.columns):
                continue
            cached["stock_id"] = cached["stock_id"].astype(str)
            cached["signal_date"] = pd.to_datetime(
                cached["signal_date"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
            selected = cached[
                (cached["stock_id"] == str(stock_id))
                & cached["signal_date"].isin(unresolved)
            ][columns].copy()
            if selected.empty:
                continue
            stock_frames.append(selected)
            unresolved.difference_update(selected["signal_date"].dropna().astype(str))
        if stock_frames:
            frames.append(
                pd.concat(stock_frames, ignore_index=True).drop_duplicates(
                    ["stock_id", "signal_date"], keep="first"
                )
            )
    if not frames:
        return pd.DataFrame(columns=columns)
    result = pd.concat(frames, ignore_index=True)
    for column in columns[2:]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.drop_duplicates(["stock_id", "signal_date"], keep="first")


def load_event_market_history(
    database: ResearchDatabase,
    events: pd.DataFrame,
    *,
    settings: Phase5ISettings,
) -> pd.DataFrame:
    first_date = pd.Timestamp(events["signal_date"].min()) - pd.Timedelta(days=60)
    last_date = pd.Timestamp(events["signal_date"].max()) + pd.Timedelta(days=10)
    stocks = sorted(events["stock_id"].unique().tolist())
    frames: list[pd.DataFrame] = []
    for chunk in _chunks(stocks, 800):
        placeholders = ",".join("?" for _ in chunk)
        sql = f"""
            SELECT
                p.stock_id,
                substr(p.date, 1, 10) AS date,
                p.open, p.high, p.low, p.close,
                p.trading_volume, p.trading_money,
                COALESCE(f.foreign_net, 0) AS foreign_net,
                COALESCE(f.investment_trust_net, 0) AS investment_trust_net,
                COALESCE(f.dealer_self_net, 0) AS dealer_self_net,
                COALESCE(f.selected_total_net, 0) AS selected_total_net
            FROM stock_prices p
            LEFT JOIN institutional_flows f
              ON f.stock_id=p.stock_id
             AND substr(f.date, 1, 10)=substr(p.date, 1, 10)
            WHERE p.stock_id IN ({placeholders})
              AND substr(p.date, 1, 10) BETWEEN ? AND ?
            ORDER BY p.stock_id, substr(p.date, 1, 10)
        """
        params = [*chunk, first_date.strftime("%Y-%m-%d"), last_date.strftime("%Y-%m-%d")]
        with database.connect() as connection:
            frames.append(pd.read_sql_query(sql, connection, params=params))
    history = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if history.empty:
        raise RuntimeError("Phase 5I 找不到事件期間行情與法人資料")
    history["stock_id"] = history["stock_id"].astype(str)
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history = history[history["date"].notna()].copy()
    number_columns = [
        "open",
        "high",
        "low",
        "close",
        "trading_volume",
        "trading_money",
        *ACTOR_COLUMNS.values(),
    ]
    for column in number_columns:
        history[column] = pd.to_numeric(history[column], errors="coerce")
    valid_price = (
        history[["open", "high", "low", "close"]].notna().all(axis=1)
        & (history[["open", "high", "low", "close"]] > 0).all(axis=1)
    )
    history = history[valid_price].sort_values(["stock_id", "date"]).reset_index(drop=True)
    if history.empty:
        raise RuntimeError("Phase 5I 行情資料沒有有效 OHLC")
    return history


def build_event_cost_features(
    events: pd.DataFrame,
    *,
    history: pd.DataFrame,
    settings: Phase5ISettings,
) -> pd.DataFrame:
    base = history.copy()
    base["typical_price"] = (base["high"] + base["low"] + base["close"]) / 3.0
    grouped = base.groupby("stock_id", sort=False, group_keys=False)
    base["entry_date_computed"] = grouped["date"].shift(-1)
    base["entry_open_computed"] = grouped["open"].shift(-1)
    base["return_5d_before_signal"] = grouped["close"].pct_change(5, fill_method=None)
    base["return_20d_before_signal"] = grouped["close"].pct_change(20, fill_method=None)
    rolling_low = grouped["low"].rolling(20, min_periods=20).min().reset_index(level=0, drop=True)
    rolling_high = grouped["high"].rolling(20, min_periods=20).max().reset_index(level=0, drop=True)
    spread = rolling_high - rolling_low
    base["range_position_20d"] = np.where(
        spread > 0,
        (base["close"] - rolling_low) / spread,
        np.nan,
    )

    generated: list[str] = []
    for actor, flow_column in ACTOR_COLUMNS.items():
        positive = base[flow_column].clip(lower=0).fillna(0.0)
        base[f"_{actor}_positive"] = positive
        for window in settings.cost_windows:
            weight_sum = positive.groupby(base["stock_id"], sort=False).transform(
                lambda values, size=window: values.rolling(size, min_periods=size).sum()
            )
            buy_days = (positive > 0).astype(int).groupby(base["stock_id"], sort=False).transform(
                lambda values, size=window: values.rolling(size, min_periods=size).sum()
            )
            for price_name, source_column in (
                ("mid", "typical_price"),
                ("low", "low"),
                ("high", "high"),
                ("close", "close"),
            ):
                weighted = positive * base[source_column]
                weighted_sum = weighted.groupby(base["stock_id"], sort=False).transform(
                    lambda values, size=window: values.rolling(size, min_periods=size).sum()
                )
                output_column = f"{actor}_cost_{price_name}_{window}d"
                base[output_column] = np.where(weight_sum > 0, weighted_sum / weight_sum, np.nan)
                generated.append(output_column)
            base[f"{actor}_positive_net_shares_{window}d"] = weight_sum
            base[f"{actor}_buy_days_{window}d"] = buy_days
            generated.extend(
                [
                    f"{actor}_positive_net_shares_{window}d",
                    f"{actor}_buy_days_{window}d",
                ]
            )

    base["signal_date"] = base["date"].dt.strftime("%Y-%m-%d")
    event_keys = events[["stock_id", "signal_date"]].drop_duplicates()
    selected_columns = [
        "stock_id",
        "signal_date",
        "close",
        "entry_date_computed",
        "entry_open_computed",
        "return_5d_before_signal",
        "return_20d_before_signal",
        "range_position_20d",
        *generated,
    ]
    result = event_keys.merge(
        base[selected_columns],
        on=["stock_id", "signal_date"],
        how="left",
        validate="one_to_one",
    )
    result = result.rename(columns={"close": "signal_close"})
    result["entry_date_computed"] = pd.to_datetime(
        result["entry_date_computed"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    return result


def finalize_event_features(
    frame: pd.DataFrame,
    *,
    settings: Phase5ISettings,
) -> pd.DataFrame:
    result = frame.copy()
    result["entry_open"] = pd.to_numeric(result["entry_open_20d"], errors="coerce")
    result["entry_open"] = result["entry_open"].fillna(
        pd.to_numeric(result["entry_open_computed"], errors="coerce")
    )
    result["entry_date"] = result["entry_date_20d"].fillna(
        result["entry_date_computed"]
    )
    result["entry_gap_from_signal_close"] = result["entry_open"] / result["signal_close"] - 1.0
    for actor in ACTORS:
        for window in settings.cost_windows:
            mid = f"{actor}_cost_mid_{window}d"
            result[f"entry_deviation_{actor}_{window}d"] = result["entry_open"] / result[mid] - 1.0
            result[f"signal_deviation_{actor}_{window}d"] = result["signal_close"] / result[mid] - 1.0
            result[f"cost_band_width_{actor}_{window}d"] = (
                result[f"{actor}_cost_high_{window}d"]
                / result[f"{actor}_cost_low_{window}d"]
                - 1.0
            )
    return result


def evaluate_entry_risk(
    enriched: pd.DataFrame,
    *,
    settings: Phase5ISettings,
    source_path: Path,
) -> dict[str, pd.DataFrame]:
    long = build_proxy_long_frame(enriched, settings=settings)
    proxy_comparison = build_proxy_comparison(long)
    deviation_buckets = build_deviation_bucket_analysis(long)
    quantiles = build_quantile_analysis(long)
    yearly = build_yearly_stability(long)
    bootstrap = build_deviation_bootstrap(long, settings=settings)
    overheat = build_overheat_rule_comparison(enriched)
    candidates = build_rule_candidates(proxy_comparison, quantiles, bootstrap, overheat)
    audit = build_input_audit(enriched, long, source_path=source_path)
    summary = build_summary(enriched, long, candidates)
    return {
        "input_audit": audit,
        "event_features": enriched,
        "proxy_comparison": proxy_comparison,
        "deviation_buckets": deviation_buckets,
        "quantiles": quantiles,
        "yearly": yearly,
        "bootstrap": bootstrap,
        "overheat": overheat,
        "rule_candidates": candidates,
        "summary": summary,
    }


def build_proxy_long_frame(
    frame: pd.DataFrame,
    *,
    settings: Phase5ISettings,
) -> pd.DataFrame:
    common = [
        "event_id",
        "event_type",
        "entry_rule",
        "stock_id",
        "stock_name",
        "signal_date",
        "test_year",
        "return_rank_score_daily_percentile",
        "signal_close",
        "entry_open",
        "return_5d_before_signal",
        "return_20d_before_signal",
        "range_position_20d",
        "adjusted_return_20d",
        "adjusted_return_40d",
        "excess_adjusted_return_20d",
        "excess_adjusted_return_40d",
        "max_adjusted_return_20d",
        "min_adjusted_return_20d",
        "max_adjusted_return_40d",
        "min_adjusted_return_40d",
    ]
    frames: list[pd.DataFrame] = []
    for actor, actor_label in ACTORS.items():
        for window in settings.cost_windows:
            selected = frame[common].copy()
            selected["actor"] = actor
            selected["actor_label"] = actor_label
            selected["window_days"] = window
            selected["proxy_name"] = f"{actor}_{window}d"
            selected["estimated_cost_mid"] = frame[f"{actor}_cost_mid_{window}d"]
            selected["estimated_cost_low"] = frame[f"{actor}_cost_low_{window}d"]
            selected["estimated_cost_high"] = frame[f"{actor}_cost_high_{window}d"]
            selected["positive_net_shares"] = frame[
                f"{actor}_positive_net_shares_{window}d"
            ]
            selected["buy_days"] = frame[f"{actor}_buy_days_{window}d"]
            selected["entry_deviation"] = frame[f"entry_deviation_{actor}_{window}d"]
            selected["signal_deviation"] = frame[f"signal_deviation_{actor}_{window}d"]
            selected["cost_band_width"] = frame[f"cost_band_width_{actor}_{window}d"]
            selected["proxy_available"] = (
                np.isfinite(selected["estimated_cost_mid"])
                & np.isfinite(selected["entry_open"])
                & (selected["positive_net_shares"] > 0)
            ).astype(int)
            frames.append(selected)
    return pd.concat(frames, ignore_index=True)


def build_proxy_comparison(long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period_name, period in _period_frames(long):
        for (event_type, proxy_name, actor, actor_label, window), group in period.groupby(
            ["event_type", "proxy_name", "actor", "actor_label", "window_days"],
            sort=True,
        ):
            valid = group[group["proxy_available"] == 1].copy()
            row: dict[str, Any] = {
                "period": period_name,
                "event_type": event_type,
                "proxy_name": proxy_name,
                "actor": actor,
                "actor_label": actor_label,
                "window_days": int(window),
                "event_count": int(len(group)),
                "available_count": int(len(valid)),
                "availability_rate": float(len(valid) / len(group)) if len(group) else np.nan,
                "average_entry_deviation": _mean(valid["entry_deviation"]),
                "median_entry_deviation": _median(valid["entry_deviation"]),
                "average_buy_days": _mean(valid["buy_days"]),
                "average_cost_band_width": _mean(valid["cost_band_width"]),
                "spearman_deviation_vs_excess_return_20d": _spearman(
                    valid["entry_deviation"], valid["excess_adjusted_return_20d"]
                ),
                "spearman_deviation_vs_excess_return_40d": _spearman(
                    valid["entry_deviation"], valid["excess_adjusted_return_40d"]
                ),
                "spearman_deviation_vs_min_return_20d": _spearman(
                    valid["entry_deviation"], valid["min_adjusted_return_20d"]
                ),
            }
            row.update(_outcome_metrics(valid))
            rows.append(row)
    return pd.DataFrame(rows)


def build_deviation_bucket_analysis(long: pd.DataFrame) -> pd.DataFrame:
    valid = long[long["proxy_available"] == 1].copy()
    valid["deviation_bucket"] = valid["entry_deviation"].map(_deviation_bucket)
    rows: list[dict[str, Any]] = []
    for period_name, period in _period_frames(valid):
        for (event_type, proxy_name, bucket), group in period.groupby(
            ["event_type", "proxy_name", "deviation_bucket"], sort=True
        ):
            row = {
                "period": period_name,
                "event_type": event_type,
                "proxy_name": proxy_name,
                "deviation_bucket": bucket,
                "event_count": int(len(group)),
                "unique_stocks": int(group["stock_id"].nunique()),
                "average_entry_deviation": _mean(group["entry_deviation"]),
            }
            row.update(_outcome_metrics(group))
            rows.append(row)
    return pd.DataFrame(rows)


def build_quantile_analysis(long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    valid = long[long["proxy_available"] == 1].copy()
    for period_name, period in _period_frames(valid):
        for (event_type, proxy_name), group in period.groupby(
            ["event_type", "proxy_name"], sort=True
        ):
            ranked = group.copy()
            if ranked["entry_deviation"].nunique() < 5:
                continue
            ranked["deviation_quintile"] = pd.qcut(
                ranked["entry_deviation"], 5, labels=False, duplicates="drop"
            )
            for quintile, bucket in ranked.groupby("deviation_quintile", sort=True):
                row = {
                    "period": period_name,
                    "event_type": event_type,
                    "proxy_name": proxy_name,
                    "deviation_quintile": int(quintile) + 1,
                    "event_count": int(len(bucket)),
                    "average_entry_deviation": _mean(bucket["entry_deviation"]),
                }
                row.update(_outcome_metrics(bucket))
                rows.append(row)
    return pd.DataFrame(rows)


def build_yearly_stability(long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    valid = long[long["proxy_available"] == 1].copy()
    for (event_type, proxy_name, year), group in valid.groupby(
        ["event_type", "proxy_name", "test_year"], sort=True
    ):
        if group["entry_deviation"].nunique() < 5:
            continue
        quintile = pd.qcut(group["entry_deviation"], 5, labels=False, duplicates="drop")
        low = group[quintile == quintile.min()]
        high = group[quintile == quintile.max()]
        rows.append(
            {
                "event_type": event_type,
                "proxy_name": proxy_name,
                "test_year": int(year),
                "event_count": int(len(group)),
                "low_deviation_count": int(len(low)),
                "high_deviation_count": int(len(high)),
                "high_minus_low_excess_return_20d": _mean(
                    high["excess_adjusted_return_20d"]
                )
                - _mean(low["excess_adjusted_return_20d"]),
                "high_minus_low_excess_return_40d": _mean(
                    high["excess_adjusted_return_40d"]
                )
                - _mean(low["excess_adjusted_return_40d"]),
                "high_minus_low_min_return_20d": _mean(high["min_adjusted_return_20d"])
                - _mean(low["min_adjusted_return_20d"]),
            }
        )
    return pd.DataFrame(rows)


def build_deviation_bootstrap(
    long: pd.DataFrame,
    *,
    settings: Phase5ISettings,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    valid = long[long["proxy_available"] == 1].copy()
    for period_name, period in _period_frames(valid):
        if period_name not in {"confirmation", "confirmation_ex_latest"}:
            continue
        for (event_type, proxy_name), group in period.groupby(
            ["event_type", "proxy_name"], sort=True
        ):
            if len(group) < settings.minimum_proxy_events or group["entry_deviation"].nunique() < 5:
                continue
            ranked = group.copy()
            ranked["deviation_quintile"] = pd.qcut(
                ranked["entry_deviation"], 5, labels=False, duplicates="drop"
            )
            low_q = ranked["deviation_quintile"].min()
            high_q = ranked["deviation_quintile"].max()
            for horizon in (20, 40):
                column = f"excess_adjusted_return_{horizon}d"
                low = ranked[ranked["deviation_quintile"] == low_q].groupby("signal_date")[
                    column
                ].mean()
                high = ranked[ranked["deviation_quintile"] == high_q].groupby("signal_date")[
                    column
                ].mean()
                aligned = pd.concat([high.rename("high"), low.rename("low")], axis=1).dropna()
                daily = aligned["high"] - aligned["low"]
                monthly = daily.groupby(daily.index.astype(str).str[:7]).mean().sort_index()
                if len(monthly) < 2:
                    continue
                seed = _stable_seed(
                    settings.random_seed,
                    f"{period_name}:{event_type}:{proxy_name}:{horizon}",
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
                        "event_type": event_type,
                        "proxy_name": proxy_name,
                        "horizon_days": horizon,
                        "event_count": int(len(ranked)),
                        "daily_observations": int(len(daily)),
                        "monthly_blocks": int(len(monthly)),
                        "point_estimate_high_minus_low_excess_return": float(daily.mean()),
                        "bootstrap_mean_high_minus_low_excess_return": bootstrap_mean,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "ci_excludes_zero_negative": int(upper < 0),
                        "iterations": settings.bootstrap_iterations,
                    }
                )
    return pd.DataFrame(rows)


def build_overheat_rule_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period_name, period in _period_frames(frame):
        for event_type, group in period.groupby("event_type", sort=True):
            for deviation_threshold in (0.05, 0.10, 0.15):
                for return_threshold in (0.05, 0.10, 0.15):
                    for range_threshold in (0.80, 0.90):
                        available = group[
                            np.isfinite(group["entry_deviation_selected_total_20d"])
                            & np.isfinite(group["return_5d_before_signal"])
                            & np.isfinite(group["range_position_20d"])
                        ].copy()
                        overheated = available[
                            (available["entry_deviation_selected_total_20d"] >= deviation_threshold)
                            & (
                                (available["return_5d_before_signal"] >= return_threshold)
                                | (available["range_position_20d"] >= range_threshold)
                            )
                        ]
                        normal = available.drop(index=overheated.index)
                        if len(overheated) < 20 or len(normal) < 20:
                            continue
                        rows.append(
                            {
                                "period": period_name,
                                "event_type": event_type,
                                "deviation_threshold": deviation_threshold,
                                "return_5d_threshold": return_threshold,
                                "range_position_threshold": range_threshold,
                                "available_count": int(len(available)),
                                "overheated_count": int(len(overheated)),
                                "overheated_rate": float(len(overheated) / len(available)),
                                "overheated_excess_return_20d": _mean(
                                    overheated["excess_adjusted_return_20d"]
                                ),
                                "normal_excess_return_20d": _mean(
                                    normal["excess_adjusted_return_20d"]
                                ),
                                "overheated_minus_normal_excess_return_20d": _mean(
                                    overheated["excess_adjusted_return_20d"]
                                )
                                - _mean(normal["excess_adjusted_return_20d"]),
                                "overheated_min_return_20d": _mean(
                                    overheated["min_adjusted_return_20d"]
                                ),
                                "normal_min_return_20d": _mean(normal["min_adjusted_return_20d"]),
                                "overheated_minus_normal_min_return_20d": _mean(
                                    overheated["min_adjusted_return_20d"]
                                )
                                - _mean(normal["min_adjusted_return_20d"]),
                            }
                        )
    return pd.DataFrame(rows)


def build_rule_candidates(
    comparison: pd.DataFrame,
    quantiles: pd.DataFrame,
    bootstrap: pd.DataFrame,
    overheat: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    periods = comparison[comparison["period"] == "confirmation_ex_latest"]
    for _, row in periods.iterrows():
        proxy = str(row["proxy_name"])
        event_type = str(row["event_type"])
        quintile = quantiles[
            (quantiles["period"] == "confirmation_ex_latest")
            & (quantiles["proxy_name"] == proxy)
            & (quantiles["event_type"] == event_type)
        ]
        low = quintile[quintile["deviation_quintile"] == 1]
        high = quintile[quintile["deviation_quintile"] == 5]
        high_minus_low_20 = (
            _first(high, "average_excess_return_20d")
            - _first(low, "average_excess_return_20d")
        )
        high_minus_low_drawdown = (
            _first(high, "average_min_return_20d")
            - _first(low, "average_min_return_20d")
        )
        boot = bootstrap[
            (bootstrap["period"] == "confirmation_ex_latest")
            & (bootstrap["proxy_name"] == proxy)
            & (bootstrap["event_type"] == event_type)
            & (bootstrap["horizon_days"] == 20)
        ]
        rows.append(
            {
                "event_type": event_type,
                "proxy_name": proxy,
                "availability_rate": row["availability_rate"],
                "spearman_deviation_vs_excess_return_20d": row[
                    "spearman_deviation_vs_excess_return_20d"
                ],
                "high_minus_low_excess_return_20d": high_minus_low_20,
                "high_minus_low_min_return_20d": high_minus_low_drawdown,
                "bootstrap_ci_upper_high_minus_low_20d": _first(boot, "ci_upper"),
                "bootstrap_negative_confirmed": int(
                    _first(boot, "ci_excludes_zero_negative", default=0) == 1
                ),
                "candidate_status": _candidate_status(
                    availability=float(row["availability_rate"]),
                    high_minus_low_return=high_minus_low_20,
                    high_minus_low_drawdown=high_minus_low_drawdown,
                    bootstrap_upper=_first(boot, "ci_upper"),
                ),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    status_rank = {"strong_candidate": 0, "research_candidate": 1, "reject": 2}
    result["_rank"] = result["candidate_status"].map(status_rank).fillna(9)
    result = result.sort_values(
        ["event_type", "_rank", "high_minus_low_excess_return_20d"],
        ascending=[True, True, True],
    ).drop(columns="_rank")
    result["overheat_rule_rows"] = len(overheat)
    return result.reset_index(drop=True)


def build_input_audit(
    events: pd.DataFrame,
    long: pd.DataFrame,
    *,
    source_path: Path,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "phase5i_version", "value": PHASE5I_VERSION},
            {"metric": "source_path", "value": str(source_path)},
            {"metric": "source_sha256", "value": _sha256(source_path)},
            {"metric": "event_rows", "value": len(events)},
            {"metric": "unique_event_ids", "value": events["event_id"].nunique()},
            {"metric": "unique_stocks", "value": events["stock_id"].nunique()},
            {"metric": "first_signal_date", "value": events["signal_date"].min()},
            {"metric": "last_signal_date", "value": events["signal_date"].max()},
            {"metric": "proxy_rows", "value": len(long)},
            {"metric": "available_proxy_rows", "value": int(long["proxy_available"].sum())},
            {
                "metric": "max_min_20d_available_events",
                "value": int(
                    pd.to_numeric(events["min_adjusted_return_20d"], errors="coerce")
                    .notna()
                    .sum()
                ),
            },
            {
                "metric": "max_min_40d_available_events",
                "value": int(
                    pd.to_numeric(events["min_adjusted_return_40d"], errors="coerce")
                    .notna()
                    .sum()
                ),
            },
            {
                "metric": "future_inputs_used_for_cost_features",
                "value": 0,
            },
            {
                "metric": "cost_proxy_definition",
                "value": "positive net-buy shares weighted by same-day typical price; not true inventory cost",
            },
        ]
    )


def build_summary(
    events: pd.DataFrame,
    long: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    strong = int((candidates.get("candidate_status", pd.Series(dtype=str)) == "strong_candidate").sum())
    return pd.DataFrame(
        [
            {"metric": "pipeline_status", "value": "PASS"},
            {"metric": "phase5i_version", "value": PHASE5I_VERSION},
            {"metric": "event_rows", "value": len(events)},
            {"metric": "event_types", "value": events["event_type"].nunique()},
            {"metric": "proxy_rows", "value": len(long)},
            {"metric": "proxy_variants", "value": long["proxy_name"].nunique()},
            {"metric": "strong_candidate_count", "value": strong},
            {
                "metric": "ready_for_notification_change",
                "value": int(strong > 0),
            },
            {
                "metric": "interpretation_limit",
                "value": "estimated cost band is a recent net-buy proxy, not actual institutional inventory cost",
            },
        ]
    )


def export_phase5i_reports(
    *,
    output_dir: Path,
    reports: dict[str, pd.DataFrame],
) -> list[Path]:
    paths = [
        _write_csv(output_dir / "phase5i_input_audit.csv", reports["input_audit"]),
        _write_gzip_csv(
            output_dir / "phase5i_event_cost_features.csv.gz",
            reports["event_features"],
        ),
        _write_csv(
            output_dir / "phase5i_cost_proxy_comparison.csv",
            reports["proxy_comparison"],
        ),
        _write_csv(
            output_dir / "phase5i_deviation_bucket_analysis.csv",
            reports["deviation_buckets"],
        ),
        _write_csv(
            output_dir / "phase5i_deviation_quantile_analysis.csv",
            reports["quantiles"],
        ),
        _write_csv(
            output_dir / "phase5i_yearly_stability.csv",
            reports["yearly"],
        ),
        _write_csv(
            output_dir / "phase5i_bootstrap_confidence.csv",
            reports["bootstrap"],
        ),
        _write_csv(
            output_dir / "phase5i_overheat_rule_comparison.csv",
            reports["overheat"],
        ),
        _write_csv(
            output_dir / "phase5i_rule_candidates.csv",
            reports["rule_candidates"],
        ),
        _write_csv(output_dir / "phase5i_summary.csv", reports["summary"]),
    ]
    archive = output_dir / "phase5i_entry_risk_validation_reports.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in paths:
            compression = zipfile.ZIP_STORED if path.suffix == ".gz" else zipfile.ZIP_DEFLATED
            handle.write(path, arcname=path.name, compress_type=compression)
    paths.append(archive)
    return paths


def _period_frames(frame: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    years = pd.to_numeric(frame["test_year"], errors="coerce")
    latest_year = int(years.max())
    yield "all", frame
    yield "development", frame[years <= 2022]
    yield "confirmation", frame[years >= 2023]
    yield "confirmation_ex_latest", frame[(years >= 2023) & (years < latest_year)]


def _outcome_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "average_return_20d": _mean(frame["adjusted_return_20d"]),
        "average_return_40d": _mean(frame["adjusted_return_40d"]),
        "average_excess_return_20d": _mean(frame["excess_adjusted_return_20d"]),
        "average_excess_return_40d": _mean(frame["excess_adjusted_return_40d"]),
        "average_max_return_20d": _mean(frame["max_adjusted_return_20d"]),
        "average_min_return_20d": _mean(frame["min_adjusted_return_20d"]),
        "average_max_return_40d": _mean(frame["max_adjusted_return_40d"]),
        "average_min_return_40d": _mean(frame["min_adjusted_return_40d"]),
        "positive_return_rate_20d": _rate(frame["adjusted_return_20d"] > 0),
        "down_5pct_rate_20d": _rate(frame["adjusted_return_20d"] <= -0.05),
    }


def _candidate_status(
    *,
    availability: float,
    high_minus_low_return: float,
    high_minus_low_drawdown: float,
    bootstrap_upper: float,
) -> str:
    if not np.isfinite(high_minus_low_return):
        return "reject"
    if (
        availability >= 0.70
        and high_minus_low_return < 0
        and high_minus_low_drawdown < 0
        and np.isfinite(bootstrap_upper)
        and bootstrap_upper < 0
    ):
        return "strong_candidate"
    if availability >= 0.60 and high_minus_low_return < 0:
        return "research_candidate"
    return "reject"


def _deviation_bucket(value: Any) -> str:
    if pd.isna(value):
        return "unavailable"
    number = float(value)
    for lower, upper, label in FIXED_DEVIATION_BUCKETS:
        if lower <= number < upper:
            return label
    return "unavailable"


def _spearman(left: pd.Series, right: pd.Series) -> float:
    pair = pd.concat([left, right], axis=1).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return np.nan

    # Spearman correlation is Pearson correlation of the average ranks.
    # Ranking explicitly avoids pandas importing scipy for method="spearman".
    left_rank = pair.iloc[:, 0].rank(method="average")
    right_rank = pair.iloc[:, 1].rank(method="average")
    return float(left_rank.corr(right_rank))


def _mean(values: pd.Series) -> float:
    numbers = pd.to_numeric(values, errors="coerce")
    finite = numbers[np.isfinite(numbers)]
    return float(finite.mean()) if len(finite) else np.nan


def _median(values: pd.Series) -> float:
    numbers = pd.to_numeric(values, errors="coerce")
    finite = numbers[np.isfinite(numbers)]
    return float(finite.median()) if len(finite) else np.nan


def _rate(values: pd.Series) -> float:
    clean = values.dropna()
    return float(clean.mean()) if len(clean) else np.nan


def _first(frame: pd.DataFrame, column: str, default: float = np.nan) -> float:
    if frame.empty or column not in frame.columns:
        return default
    value = frame.iloc[0][column]
    return float(value) if pd.notna(value) else default


def _stable_seed(base: int, value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return (base + int(digest[:8], 16)) % (2**32 - 1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


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
