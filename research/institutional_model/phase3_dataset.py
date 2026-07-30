from __future__ import annotations

import bisect
import csv
import gzip
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from research.institutional_model.adjusted_returns import calculate_holding_return
from research.institutional_model.corporate_actions import CorporateAction, load_action_map
from research.institutional_model.database import ResearchDatabase


PHASE3_VERSION = "phase3a-v1"
FLOW_WINDOWS = (1, 3, 5, 10, 20)
BUY_DAY_WINDOWS = (5, 10, 20)
HORIZONS = (5, 10, 20)
LIQUIDITY_THRESHOLDS = (10_000_000, 20_000_000, 50_000_000, 100_000_000)
PRIMARY_LIQUIDITY_THRESHOLD = 20_000_000
PRIMARY_HORIZON = 10
LABEL_THRESHOLD = 0.05
RETURN_ROUND_DIGITS = 10
LABEL_RULE_VERSION = "rounded-return-10dp-v1"
ACTORS = (
    ("foreign", "foreign_net"),
    ("investment_trust", "investment_trust_net"),
    ("dealer_self", "dealer_self_net"),
    ("selected_total", "selected_total_net"),
)


@dataclass(frozen=True)
class Phase3BuildResult:
    config_signature: str
    total_stocks: int
    completed_stocks: int
    skipped_stocks: int
    failed_stocks: int
    pending_stocks: int
    merged_rows_twse: int
    merged_rows_tpex: int
    output_paths: tuple[Path, ...]


@dataclass(frozen=True)
class HorizonResult:
    status: str
    error: str | None = None
    entry_date: str | None = None
    target_date: str | None = None
    entry_open: float | None = None
    target_close: float | None = None
    raw_return: float | None = None
    adjusted_return: float | None = None
    max_adjusted_return: float | None = None
    min_adjusted_return: float | None = None
    action_types: tuple[str, ...] = ()
    entry_day_action_ignored: int = 0


@dataclass(frozen=True)
class StockBuildStats:
    total_rows: int
    feature_ready_rows: int
    eligible_5d_rows: int
    eligible_10d_rows: int
    eligible_20d_rows: int
    liquidity_rows: dict[int, int]
    first_signal_date: str | None
    last_signal_date: str | None
    label_distribution: dict[tuple[int, str], int]
    exclusion_reasons: dict[str, int]


BASE_COLUMNS = [
    "stock_id",
    "stock_name",
    "market_type",
    "signal_date",
    "signal_year",
    "listing_date",
    "delisting_date",
    "current_status",
    "history_market_days_20",
    "history_valid_price_days_20",
    "history_flow_rows_20",
    "median_trading_money_20d",
    "max_zero_volume_streak_20d",
    "normal_trading_days_20d",
    "feature_status",
]

FEATURE_COLUMNS: list[str] = []
for actor, _ in ACTORS:
    FEATURE_COLUMNS.extend(f"{actor}_flow_pct_{window}d" for window in FLOW_WINDOWS)
    FEATURE_COLUMNS.extend(
        f"{actor}_buy_day_ratio_{window}d" for window in BUY_DAY_WINDOWS
    )
    FEATURE_COLUMNS.append(f"{actor}_streak")
FEATURE_COLUMNS.extend(
    [
        "institutional_agreement_1d",
        "institutional_agreement_5d",
        "institutional_agreement_20d",
        "selected_total_acceleration_5d_vs_20d",
    ]
)

QUALITY_COLUMNS = [
    "entry_price_available",
    "liquidity_pass_10m",
    "liquidity_pass_20m",
    "liquidity_pass_50m",
    "liquidity_pass_100m",
    "primary_exclusion_reason",
]

LABEL_COLUMNS: list[str] = []
for horizon in HORIZONS:
    LABEL_COLUMNS.extend(
        [
            f"entry_date_{horizon}d",
            f"target_date_{horizon}d",
            f"entry_open_{horizon}d",
            f"target_close_{horizon}d",
            f"raw_return_{horizon}d",
            f"adjusted_return_{horizon}d",
            f"max_adjusted_return_{horizon}d",
            f"min_adjusted_return_{horizon}d",
            f"action_types_{horizon}d",
            f"entry_day_action_ignored_{horizon}d",
            f"label_status_{horizon}d",
            f"label_error_{horizon}d",
            f"sample_eligible_{horizon}d",
        ]
    )
LABEL_COLUMNS.append("label_10d")

ALL_COLUMNS = BASE_COLUMNS + FEATURE_COLUMNS + QUALITY_COLUMNS + LABEL_COLUMNS


def build_phase3_dataset(
    *,
    database: ResearchDatabase,
    start_date: str,
    end_date: str,
    output_dir: Path | str,
    shard_root: Path | str,
    symbols: Iterable[str] | None = None,
    max_stocks: int = 0,
    force: bool = False,
) -> Phase3BuildResult:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    signature = phase3_config_signature(start_date=start_date, end_date=end_date)
    shard_dir = Path(shard_root) / signature[:16]
    shard_dir.mkdir(parents=True, exist_ok=True)

    market_dates = [
        str(row["date"])
        for row in database.query(
            "SELECT date FROM market_calendar WHERE date <= ? ORDER BY date",
            (end_date,),
        )
    ]
    if not market_dates:
        raise RuntimeError("市場交易日曆為空，請先完成 Phase 2。")
    market_index = {value: index for index, value in enumerate(market_dates)}

    all_candidates = _load_candidates(database, start_date, end_date, None)
    candidates = list(all_candidates)
    if symbols is not None:
        selected = {str(value).strip() for value in symbols if str(value).strip()}
        known = {row["stock_id"] for row in all_candidates}
        unknown = selected.difference(known)
        if unknown:
            raise ValueError(f"指定股票不在 Phase 3 可用母體：{sorted(unknown)}")
        candidates = [row for row in candidates if row["stock_id"] in selected]
    if max_stocks > 0:
        candidates = candidates[:max_stocks]

    completed = 0
    skipped = 0
    failed = 0
    for position, stock in enumerate(candidates, start=1):
        stock_id = str(stock["stock_id"])
        shard_path = shard_dir / f"{stock_id}.csv.gz"
        if not force and _stock_is_complete(
            database=database,
            stock_id=stock_id,
            signature=signature,
            shard_path=shard_path,
        ):
            skipped += 1
            print(
                f"[{position}/{len(candidates)}] {stock_id} "
                "已完成，略過並沿用既有分片。"
            )
            continue

        print(
            f"[{position}/{len(candidates)}] 建立 {stock_id} "
            f"{stock.get('stock_name') or ''} {stock['market_type']}"
        )
        try:
            stats = _build_stock_shard(
                database=database,
                stock=stock,
                start_date=start_date,
                end_date=end_date,
                market_dates=market_dates,
                market_index=market_index,
                shard_path=shard_path,
            )
            _save_stock_stats(
                database=database,
                stock_id=stock_id,
                signature=signature,
                shard_name=shard_path.name,
                stats=stats,
            )
            completed += 1
            print(
                "  完成："
                f"訊號 {stats.total_rows}、"
                f"特徵可用 {stats.feature_ready_rows}、"
                f"10 日訓練樣本 {stats.eligible_10d_rows}"
            )
        except Exception as exc:
            failed += 1
            _save_stock_failure(database, stock_id, signature, str(exc))
            print(f"  失敗：{exc}")

    pending = _count_pending(database, all_candidates, signature, shard_dir)
    if pending == 0:
        merged_rows, merged_paths = merge_phase3_training_files(
            database=database,
            candidates=all_candidates,
            signature=signature,
            shard_dir=shard_dir,
            output_dir=output,
        )
    else:
        merged_rows = {"twse": 0, "tpex": 0}
        merged_paths = []
        print(
            f"尚有 {pending} 檔未完成；先保留分片進度，"
            "全部完成後才合併正式訓練檔。"
        )
    database.set_metadata("phase3_config_signature", signature)
    database.set_metadata("phase3_start_date", start_date)
    database.set_metadata("phase3_end_date", end_date)

    return Phase3BuildResult(
        config_signature=signature,
        total_stocks=len(all_candidates),
        completed_stocks=completed,
        skipped_stocks=skipped,
        failed_stocks=failed,
        pending_stocks=pending,
        merged_rows_twse=merged_rows.get("twse", 0),
        merged_rows_tpex=merged_rows.get("tpex", 0),
        output_paths=tuple(merged_paths),
    )


def phase3_config_signature(*, start_date: str, end_date: str) -> str:
    payload = {
        "version": PHASE3_VERSION,
        "start_date": start_date,
        "end_date": end_date,
        "flow_windows": FLOW_WINDOWS,
        "buy_day_windows": BUY_DAY_WINDOWS,
        "horizons": HORIZONS,
        "label_threshold": LABEL_THRESHOLD,
        "liquidity_thresholds": LIQUIDITY_THRESHOLDS,
        "primary_liquidity_threshold": PRIMARY_LIQUIDITY_THRESHOLD,
        "tpex_min_normal_days": 18,
        "tpex_max_zero_volume_streak": 2,
        "entry": "T+1_market_open",
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def merge_phase3_training_files(
    *,
    database: ResearchDatabase,
    candidates: list[dict[str, Any]],
    signature: str,
    shard_dir: Path,
    output_dir: Path,
) -> tuple[dict[str, int], list[Path]]:
    status_rows = {
        row["stock_id"]: dict(row)
        for row in database.query(
            """
            SELECT * FROM phase3_build_status
            WHERE config_signature=? AND status='complete'
            """,
            (signature,),
        )
    }
    counts = {"twse": 0, "tpex": 0}
    outputs: list[Path] = []
    for market in ("twse", "tpex"):
        market_candidates = [
            stock for stock in candidates if stock["market_type"] == market
        ]
        print(
            f"開始合併 {market.upper()} 正式訓練檔："
            f"{len(market_candidates)} 檔股票"
        )
        target = output_dir / f"phase3_training_{market}.csv.gz"
        temporary = target.with_suffix(target.suffix + ".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=ALL_COLUMNS)
            writer.writeheader()
            for position, stock in enumerate(market_candidates, start=1):
                if position % 100 == 0 or position == len(market_candidates):
                    print(
                        f"  {market.upper()} 合併進度："
                        f"{position}/{len(market_candidates)}"
                    )
                status = status_rows.get(stock["stock_id"])
                if not status:
                    continue
                shard_path = shard_dir / str(status.get("shard_name") or "")
                if not shard_path.exists():
                    continue
                with gzip.open(
                    shard_path, "rt", encoding="utf-8-sig", newline=""
                ) as shard_handle:
                    for row in csv.DictReader(shard_handle):
                        if row.get("sample_eligible_10d") != "1":
                            continue
                        writer.writerow(row)
                        counts[market] += 1
        temporary.replace(target)
        outputs.append(target)
        print(f"{market.upper()} 訓練檔完成：{counts[market]} 列")
    return counts, outputs


def feature_dictionary_rows() -> list[dict[str, str]]:
    descriptions: dict[str, tuple[str, str]] = {
        "stock_id": ("metadata", "股票代號"),
        "stock_name": ("metadata", "股票名稱"),
        "market_type": ("metadata", "市場別；TWSE 與 TPEx 分開訓練"),
        "signal_date": ("metadata", "訊號日 T；特徵只使用 T 收盤前可得資料"),
        "signal_year": ("metadata", "供時間切分與報告使用，不進模型"),
        "listing_date": ("metadata", "研究採用的掛牌起日"),
        "delisting_date": ("metadata", "終止上市櫃日"),
        "current_status": ("metadata", "目前掛牌或歷史下市櫃"),
        "history_market_days_20": ("quality", "訊號日前含當日可取得的市場交易日數"),
        "history_valid_price_days_20": ("quality", "最近 20 市場日有效價格天數"),
        "history_flow_rows_20": ("quality", "最近 20 市場日有法人原始列的天數"),
        "median_trading_money_20d": ("filter", "最近 20 市場日成交金額中位數"),
        "max_zero_volume_streak_20d": ("filter", "最近 20 市場日最長零量連續天數"),
        "normal_trading_days_20d": ("filter", "最近 20 市場日正常交易天數"),
        "feature_status": ("quality", "法人特徵是否具備完整 20 市場日歷史"),
        "entry_price_available": ("filter", "T+1 是否有可計算的有效開盤與收盤"),
        "primary_exclusion_reason": ("quality", "未進入 10 日主訓練集的主要原因"),
        "label_10d": ("label", "10 日還原報酬 >=5% 為 UP，<=-5% 為 DOWN"),
    }
    for threshold in (10, 20, 50, 100):
        descriptions[f"liquidity_pass_{threshold}m"] = (
            "filter",
            f"TPEx 流動性門檻新台幣 {threshold} 百萬元的樣本通過旗標",
        )
    for actor, _ in ACTORS:
        actor_name = {
            "foreign": "外資",
            "investment_trust": "投信",
            "dealer_self": "自營商自行買賣",
            "selected_total": "外資、投信、自營商自行買賣合計",
        }[actor]
        for window in FLOW_WINDOWS:
            descriptions[f"{actor}_flow_pct_{window}d"] = (
                "feature",
                f"{actor_name}最近 {window} 市場日淨買賣超／同期成交量，百分比",
            )
        for window in BUY_DAY_WINDOWS:
            descriptions[f"{actor}_buy_day_ratio_{window}d"] = (
                "feature",
                f"{actor_name}最近 {window} 市場日淨買超天數比例",
            )
        descriptions[f"{actor}_streak"] = (
            "feature",
            f"{actor_name}截至 T 的連續淨買超正值或淨賣超負值天數",
        )
    for window in (1, 5, 20):
        descriptions[f"institutional_agreement_{window}d"] = (
            "feature",
            f"三類法人最近 {window} 日淨流向方向一致程度，範圍 -1 至 1",
        )
    descriptions["selected_total_acceleration_5d_vs_20d"] = (
        "feature",
        "法人合計 5 日流量比例減 20 日流量比例",
    )
    for horizon in HORIZONS:
        for key, label in (
            ("entry_date", "T+1 進場日"),
            ("target_date", f"第 {horizon} 市場交易日目標日"),
            ("entry_open", "T+1 開盤價"),
            ("target_close", "目標日收盤價"),
            ("raw_return", "未還原報酬"),
            ("adjusted_return", "公司行動還原後報酬"),
            ("max_adjusted_return", "期間最高還原報酬"),
            ("min_adjusted_return", "期間最低還原報酬"),
            ("action_types", "持有期間公司行動種類"),
            ("entry_day_action_ignored", "進場日公司行動是否依規格忽略"),
            ("label_status", "該持有期標籤計算狀態"),
            ("label_error", "標籤無法計算原因"),
            ("sample_eligible", "該持有期是否可用"),
        ):
            descriptions[f"{key}_{horizon}d"] = ("label", label)

    result = []
    for column in ALL_COLUMNS:
        role, description = descriptions.get(column, ("quality", column))
        result.append(
            {
                "column_name": column,
                "role": role,
                "description": description,
                "model_input": "1" if role == "feature" else "0",
            }
        )
    return result


def _load_candidates(
    database: ResearchDatabase,
    start_date: str,
    end_date: str,
    symbols: Iterable[str] | None,
) -> list[dict[str, Any]]:
    rows = [
        dict(row)
        for row in database.query(
            """
            SELECT * FROM model_universe
            WHERE training_enabled=1
              AND market_type IN ('twse', 'tpex')
              AND listing_date IS NOT NULL
              AND listing_date <= ?
              AND (delisting_date IS NULL OR delisting_date > ?)
            ORDER BY CASE market_type WHEN 'twse' THEN 0 ELSE 1 END, stock_id
            """,
            (end_date, start_date),
        )
    ]
    if symbols is None:
        return rows
    selected = {str(value).strip() for value in symbols if str(value).strip()}
    return [row for row in rows if row["stock_id"] in selected]


def _stock_is_complete(
    *,
    database: ResearchDatabase,
    stock_id: str,
    signature: str,
    shard_path: Path,
) -> bool:
    rows = database.query(
        """
        SELECT status, config_signature
        FROM phase3_build_status
        WHERE stock_id=?
        """,
        (stock_id,),
    )
    return bool(
        rows
        and rows[0]["status"] == "complete"
        and rows[0]["config_signature"] == signature
        and shard_path.exists()
    )


def _count_pending(
    database: ResearchDatabase,
    candidates: list[dict[str, Any]],
    signature: str,
    shard_dir: Path,
) -> int:
    return sum(
        1
        for row in candidates
        if not _stock_is_complete(
            database=database,
            stock_id=row["stock_id"],
            signature=signature,
            shard_path=shard_dir / f"{row['stock_id']}.csv.gz",
        )
    )


def _build_stock_shard(
    *,
    database: ResearchDatabase,
    stock: dict[str, Any],
    start_date: str,
    end_date: str,
    market_dates: list[str],
    market_index: dict[str, int],
    shard_path: Path,
) -> StockBuildStats:
    stock_id = str(stock["stock_id"])
    listing_date = str(stock.get("listing_date") or start_date)
    effective_start = max(start_date, listing_date)
    effective_end = end_date
    delisting_date = str(stock.get("delisting_date") or "")
    if delisting_date:
        effective_end = min(effective_end, _day_before(delisting_date))

    price_rows = [
        dict(row)
        for row in database.query(
            """
            SELECT date, trading_volume, trading_money, open, high, low, close
            FROM stock_prices
            WHERE stock_id=? AND date <= ?
            ORDER BY date
            """,
            (stock_id, end_date),
        )
    ]
    flow_rows = [
        dict(row)
        for row in database.query(
            """
            SELECT date, foreign_net, investment_trust_net,
                   dealer_self_net, selected_total_net
            FROM institutional_flows
            WHERE stock_id=? AND date <= ?
            ORDER BY date
            """,
            (stock_id, end_date),
        )
    ]
    price_by_date = {str(row["date"]): row for row in price_rows}
    flow_by_date = {str(row["date"]): row for row in flow_rows}
    action_map = load_action_map(database, stock_id)

    arrays = _aligned_arrays(market_dates, price_by_date, flow_by_date)
    listing_index = bisect.bisect_left(market_dates, listing_date)
    label_distribution: Counter[tuple[int, str]] = Counter()
    exclusion_reasons: Counter[str] = Counter()
    liquidity_rows: Counter[int] = Counter()
    totals = Counter()
    first_signal: str | None = None
    last_signal: str | None = None

    temporary = shard_path.with_suffix(shard_path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ALL_COLUMNS)
        writer.writeheader()
        for signal_date in sorted(flow_by_date):
            if signal_date < effective_start or signal_date > effective_end:
                continue
            signal_index = market_index.get(signal_date)
            if signal_index is None:
                exclusion_reasons["non_market_flow_date"] += 1
                continue
            current_price = price_by_date.get(signal_date)
            if current_price is None:
                exclusion_reasons["missing_signal_price"] += 1
                continue
            if not _valid_signal_price(current_price):
                exclusion_reasons["invalid_signal_price"] += 1
                continue

            row = _build_sample_row(
                stock=stock,
                signal_date=signal_date,
                signal_index=signal_index,
                listing_index=listing_index,
                market_dates=market_dates,
                arrays=arrays,
                price_by_date=price_by_date,
                action_map=action_map,
                delisting_date=delisting_date,
            )
            writer.writerow(row)
            totals["total"] += 1
            if row["feature_status"] == "ok":
                totals["feature_ready"] += 1
            for horizon in HORIZONS:
                if row[f"sample_eligible_{horizon}d"] == 1:
                    totals[f"eligible_{horizon}"] += 1
            for threshold in LIQUIDITY_THRESHOLDS:
                key = f"liquidity_pass_{threshold // 1_000_000}m"
                if (
                    stock["market_type"] == "tpex"
                    and row["feature_status"] == "ok"
                    and row["label_status_10d"] == "ok"
                    and row[key] == 1
                ):
                    liquidity_rows[threshold] += 1
            reason = str(row["primary_exclusion_reason"])
            if reason != "eligible":
                exclusion_reasons[reason] += 1
            else:
                label = str(row["label_10d"])
                label_distribution[(int(row["signal_year"]), label)] += 1
            first_signal = first_signal or signal_date
            last_signal = signal_date
    temporary.replace(shard_path)

    return StockBuildStats(
        total_rows=totals["total"],
        feature_ready_rows=totals["feature_ready"],
        eligible_5d_rows=totals["eligible_5"],
        eligible_10d_rows=totals["eligible_10"],
        eligible_20d_rows=totals["eligible_20"],
        liquidity_rows=dict(liquidity_rows),
        first_signal_date=first_signal,
        last_signal_date=last_signal,
        label_distribution=dict(label_distribution),
        exclusion_reasons=dict(exclusion_reasons),
    )


def build_recent_stock_feature_rows(
    *,
    database: ResearchDatabase,
    stock: dict[str, Any],
    end_date: str,
    recent_rows: int = 90,
    market_dates: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build recent Phase 3-compatible rows without mutating frozen Phase 3 outputs."""
    if recent_rows <= 0:
        raise ValueError("recent_rows 必須大於 0")

    effective_market_dates = market_dates or [
        str(row["date"])
        for row in database.query(
            "SELECT date FROM market_calendar WHERE date <= ? ORDER BY date",
            (end_date,),
        )
    ]
    if not effective_market_dates:
        raise RuntimeError("市場交易日曆為空，無法建立即時法人特徵")
    market_index = {value: index for index, value in enumerate(effective_market_dates)}

    stock_id = str(stock["stock_id"])
    listing_date = str(stock.get("listing_date") or effective_market_dates[0])
    delisting_date = str(stock.get("delisting_date") or "")
    effective_end = min(end_date, _day_before(delisting_date)) if delisting_date else end_date

    price_rows = [
        dict(row)
        for row in database.query(
            """
            SELECT date, trading_volume, trading_money, open, high, low, close
            FROM stock_prices
            WHERE stock_id=? AND date <= ?
            ORDER BY date
            """,
            (stock_id, effective_end),
        )
    ]
    flow_rows = [
        dict(row)
        for row in database.query(
            """
            SELECT date, foreign_net, investment_trust_net,
                   dealer_self_net, selected_total_net
            FROM institutional_flows
            WHERE stock_id=? AND date <= ?
            ORDER BY date
            """,
            (stock_id, effective_end),
        )
    ]
    price_by_date = {str(row["date"]): row for row in price_rows}
    flow_by_date = {str(row["date"]): row for row in flow_rows}
    if not flow_by_date:
        return []

    candidate_dates = [
        signal_date
        for signal_date in sorted(flow_by_date)
        if listing_date <= signal_date <= effective_end
        and signal_date in market_index
        and _valid_signal_price(price_by_date.get(signal_date))
    ]
    if not candidate_dates:
        return []
    candidate_dates = candidate_dates[-recent_rows:]

    arrays = _aligned_arrays(effective_market_dates, price_by_date, flow_by_date)
    listing_index = bisect.bisect_left(effective_market_dates, listing_date)
    action_map = load_action_map(database, stock_id)
    return [
        _build_sample_row(
            stock=stock,
            signal_date=signal_date,
            signal_index=market_index[signal_date],
            listing_index=listing_index,
            market_dates=effective_market_dates,
            arrays=arrays,
            price_by_date=price_by_date,
            action_map=action_map,
            delisting_date=delisting_date,
        )
        for signal_date in candidate_dates
    ]


def _aligned_arrays(
    market_dates: list[str],
    price_by_date: dict[str, dict[str, Any]],
    flow_by_date: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    volume: list[float] = []
    trading_money: list[float] = []
    valid_price: list[int] = []
    flow_present: list[int] = []
    actor_values: dict[str, list[float]] = {actor: [] for actor, _ in ACTORS}

    for market_date in market_dates:
        price = price_by_date.get(market_date)
        flow = flow_by_date.get(market_date)
        valid = _valid_signal_price(price)
        valid_price.append(1 if valid else 0)
        volume.append(_number(price.get("trading_volume")) if valid and price else 0.0)
        trading_money.append(
            _number(price.get("trading_money")) if valid and price else 0.0
        )
        flow_present.append(1 if flow else 0)
        for actor, field in ACTORS:
            actor_values[actor].append(_number(flow.get(field)) if flow else 0.0)

    prefix: dict[str, list[float]] = {
        "volume": _prefix(volume),
        "valid_price": _prefix(valid_price),
        "flow_present": _prefix(flow_present),
    }
    streaks: dict[str, list[int]] = {}
    positives: dict[str, list[float]] = {}
    for actor, _ in ACTORS:
        prefix[actor] = _prefix(actor_values[actor])
        positives[actor] = _prefix(1 if value > 0 else 0 for value in actor_values[actor])
        streaks[actor] = _signed_streaks(actor_values[actor])
    return {
        "volume": volume,
        "trading_money": trading_money,
        "valid_price": valid_price,
        "flow_present": flow_present,
        "actor_values": actor_values,
        "prefix": prefix,
        "positive_prefix": positives,
        "streaks": streaks,
    }


def _build_sample_row(
    *,
    stock: dict[str, Any],
    signal_date: str,
    signal_index: int,
    listing_index: int,
    market_dates: list[str],
    arrays: dict[str, Any],
    price_by_date: dict[str, dict[str, Any]],
    action_map: dict[str, list[CorporateAction]],
    delisting_date: str,
) -> dict[str, Any]:
    history_start = signal_index - 19
    history_ready = history_start >= listing_index and history_start >= 0
    if history_ready:
        history_slice = slice(history_start, signal_index + 1)
        money_values = arrays["trading_money"][history_slice]
        volume_values = arrays["volume"][history_slice]
        normal_days = int(sum(arrays["valid_price"][history_slice]))
        median_money = float(median(money_values))
        max_zero_streak = _max_zero_streak(volume_values)
        flow_days = int(sum(arrays["flow_present"][history_slice]))
    else:
        available_start = max(listing_index, 0)
        market_days = max(0, signal_index - available_start + 1)
        normal_days = 0
        median_money = 0.0
        max_zero_streak = 0
        flow_days = 0

    row: dict[str, Any] = {
        "stock_id": stock["stock_id"],
        "stock_name": stock.get("stock_name") or "",
        "market_type": stock["market_type"],
        "signal_date": signal_date,
        "signal_year": int(signal_date[:4]),
        "listing_date": stock.get("listing_date") or "",
        "delisting_date": stock.get("delisting_date") or "",
        "current_status": stock.get("current_status") or "",
        "history_market_days_20": 20 if history_ready else market_days,
        "history_valid_price_days_20": normal_days,
        "history_flow_rows_20": flow_days,
        "median_trading_money_20d": round(median_money, 4),
        "max_zero_volume_streak_20d": max_zero_streak,
        "normal_trading_days_20d": normal_days,
        "feature_status": "ok" if history_ready else "insufficient_feature_history",
    }

    if history_ready:
        _add_features(row, signal_index, arrays)
    else:
        for column in FEATURE_COLUMNS:
            row[column] = ""

    horizons = {
        horizon: _calculate_horizon(
            signal_index=signal_index,
            horizon=horizon,
            market_dates=market_dates,
            price_by_date=price_by_date,
            action_map=action_map,
            delisting_date=delisting_date,
        )
        for horizon in HORIZONS
    }
    entry_available = _entry_price_available(signal_index, market_dates, price_by_date)
    row["entry_price_available"] = 1 if entry_available else 0
    for threshold in LIQUIDITY_THRESHOLDS:
        key = f"liquidity_pass_{threshold // 1_000_000}m"
        row[key] = 1 if (
            history_ready
            and normal_days >= 18
            and median_money >= threshold
            and max_zero_streak < 3
            and entry_available
        ) else 0

    screen_pass = (
        True
        if stock["market_type"] == "twse"
        else bool(row["liquidity_pass_20m"])
    )
    for horizon, result in horizons.items():
        _add_horizon_result(row, horizon, result)
        row[f"sample_eligible_{horizon}d"] = 1 if (
            history_ready and screen_pass and result.status == "ok"
        ) else 0

    ten_day = horizons[PRIMARY_HORIZON]
    if ten_day.status == "ok" and ten_day.adjusted_return is not None:
        row["label_10d"] = classify_10d_label(ten_day.adjusted_return)
    else:
        row["label_10d"] = ""
    row["primary_exclusion_reason"] = _primary_exclusion_reason(
        row=row,
        market_type=str(stock["market_type"]),
        ten_day_status=ten_day.status,
    )
    return row


def _add_features(row: dict[str, Any], index: int, arrays: dict[str, Any]) -> None:
    volume_prefix = arrays["prefix"]["volume"]
    for actor, _ in ACTORS:
        actor_prefix = arrays["prefix"][actor]
        positive_prefix = arrays["positive_prefix"][actor]
        for window in FLOW_WINDOWS:
            net = _window_sum(actor_prefix, index, window)
            total_volume = _window_sum(volume_prefix, index, window)
            ratio = net / total_volume * 100 if total_volume > 0 else 0.0
            row[f"{actor}_flow_pct_{window}d"] = round(ratio, 8)
        for window in BUY_DAY_WINDOWS:
            positive_days = _window_sum(positive_prefix, index, window)
            row[f"{actor}_buy_day_ratio_{window}d"] = round(
                positive_days / window, 8
            )
        row[f"{actor}_streak"] = arrays["streaks"][actor][index]

    for window in (1, 5, 20):
        signs = []
        for actor in ("foreign", "investment_trust", "dealer_self"):
            total = _window_sum(arrays["prefix"][actor], index, window)
            signs.append(_sign(total))
        row[f"institutional_agreement_{window}d"] = round(sum(signs) / 3, 8)
    row["selected_total_acceleration_5d_vs_20d"] = round(
        float(row["selected_total_flow_pct_5d"])
        - float(row["selected_total_flow_pct_20d"]),
        8,
    )


def _calculate_horizon(
    *,
    signal_index: int,
    horizon: int,
    market_dates: list[str],
    price_by_date: dict[str, dict[str, Any]],
    action_map: dict[str, list[CorporateAction]],
    delisting_date: str,
) -> HorizonResult:
    entry_index = signal_index + 1
    target_index = entry_index + horizon - 1
    if entry_index >= len(market_dates) or target_index >= len(market_dates):
        return HorizonResult(
            status="insufficient_future_data",
            error="研究截止日前的未來市場交易日不足",
        )
    entry_date = market_dates[entry_index]
    target_date = market_dates[target_index]
    if delisting_date and entry_date >= delisting_date:
        return HorizonResult(
            status="delisted_before_entry",
            error=f"股票已於 {delisting_date} 終止上市櫃",
            entry_date=entry_date,
            target_date=target_date,
        )
    if delisting_date and target_date >= delisting_date:
        return HorizonResult(
            status="delisted_before_target",
            error=f"持有期間跨越 {delisting_date} 終止上市櫃日",
            entry_date=entry_date,
            target_date=target_date,
        )

    entry_row = price_by_date.get(entry_date)
    if not _valid_entry_price(entry_row):
        return HorizonResult(
            status="unavailable_entry_price",
            error="T+1 市場交易日沒有有效開盤與收盤資料",
            entry_date=entry_date,
            target_date=target_date,
        )
    target_row = price_by_date.get(target_date)
    if not _valid_close(target_row):
        return HorizonResult(
            status="unavailable_target_price",
            error=f"第 {horizon} 個市場交易日沒有有效收盤資料",
            entry_date=entry_date,
            target_date=target_date,
            entry_open=float(entry_row["open"]),
        )

    calendar_path = market_dates[entry_index : target_index + 1]
    missing_action_dates = [
        value
        for value in action_map
        if entry_date < value <= target_date
        and not _valid_close(price_by_date.get(value))
    ]
    if missing_action_dates:
        return HorizonResult(
            status="invalid_data",
            error="公司行動日缺少有效價格：" + ",".join(sorted(missing_action_dates)),
            entry_date=entry_date,
            target_date=target_date,
            entry_open=float(entry_row["open"]),
            target_close=float(target_row["close"]),
        )
    path = [
        price_by_date[value]
        for value in calendar_path
        if _valid_close(price_by_date.get(value))
    ]
    try:
        result = calculate_holding_return(path, action_map)
    except ValueError as exc:
        return HorizonResult(
            status="invalid_data",
            error=str(exc),
            entry_date=entry_date,
            target_date=target_date,
            entry_open=float(entry_row["open"]),
            target_close=float(target_row["close"]),
        )
    return HorizonResult(
        status="ok",
        entry_date=entry_date,
        target_date=target_date,
        entry_open=float(entry_row["open"]),
        target_close=float(target_row["close"]),
        raw_return=result.raw_return,
        adjusted_return=result.adjusted_return,
        max_adjusted_return=result.max_adjusted_return,
        min_adjusted_return=result.min_adjusted_return,
        action_types=result.action_types,
        entry_day_action_ignored=int(result.entry_day_action_ignored),
    )


def _add_horizon_result(
    row: dict[str, Any], horizon: int, result: HorizonResult
) -> None:
    row[f"entry_date_{horizon}d"] = result.entry_date or ""
    row[f"target_date_{horizon}d"] = result.target_date or ""
    row[f"entry_open_{horizon}d"] = _round_or_blank(result.entry_open)
    row[f"target_close_{horizon}d"] = _round_or_blank(result.target_close)
    row[f"raw_return_{horizon}d"] = _round_or_blank(result.raw_return)
    row[f"adjusted_return_{horizon}d"] = _round_or_blank(result.adjusted_return)
    row[f"max_adjusted_return_{horizon}d"] = _round_or_blank(
        result.max_adjusted_return
    )
    row[f"min_adjusted_return_{horizon}d"] = _round_or_blank(
        result.min_adjusted_return
    )
    row[f"action_types_{horizon}d"] = json.dumps(
        result.action_types, ensure_ascii=False
    )
    row[f"entry_day_action_ignored_{horizon}d"] = (
        result.entry_day_action_ignored
    )
    row[f"label_status_{horizon}d"] = result.status
    row[f"label_error_{horizon}d"] = result.error or ""


def _primary_exclusion_reason(
    *, row: dict[str, Any], market_type: str, ten_day_status: str
) -> str:
    if row["feature_status"] != "ok":
        return str(row["feature_status"])
    if ten_day_status != "ok":
        return f"label_10d:{ten_day_status}"
    if market_type == "twse":
        return "eligible"
    if int(row["normal_trading_days_20d"]) < 18:
        return "tpex_normal_days_lt_18"
    if int(row["max_zero_volume_streak_20d"]) >= 3:
        return "tpex_zero_volume_streak_ge_3"
    if float(row["median_trading_money_20d"]) < PRIMARY_LIQUIDITY_THRESHOLD:
        return "tpex_median_trading_money_lt_20m"
    if not int(row["entry_price_available"]):
        return "tpex_invalid_t_plus_1_open"
    return "eligible"


def _save_stock_stats(
    *,
    database: ResearchDatabase,
    stock_id: str,
    signature: str,
    shard_name: str,
    stats: StockBuildStats,
) -> None:
    with database.connect() as connection:
        connection.execute(
            "DELETE FROM phase3_label_distribution WHERE stock_id=?",
            (stock_id,),
        )
        connection.execute(
            "DELETE FROM phase3_exclusion_stats WHERE stock_id=?",
            (stock_id,),
        )
        connection.executemany(
            """
            INSERT INTO phase3_label_distribution (
                stock_id, config_signature, signal_year, label, sample_count
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                (stock_id, signature, year, label, count)
                for (year, label), count in stats.label_distribution.items()
            ),
        )
        connection.executemany(
            """
            INSERT INTO phase3_exclusion_stats (
                stock_id, config_signature, reason, sample_count
            ) VALUES (?, ?, ?, ?)
            """,
            (
                (stock_id, signature, reason, count)
                for reason, count in stats.exclusion_reasons.items()
            ),
        )
        connection.execute(
            """
            INSERT INTO phase3_build_status (
                stock_id, config_signature, status, shard_name,
                total_rows, feature_ready_rows,
                eligible_5d_rows, eligible_10d_rows, eligible_20d_rows,
                liquidity_10m_rows, liquidity_20m_rows,
                liquidity_50m_rows, liquidity_100m_rows,
                first_signal_date, last_signal_date, error, updated_at
            ) VALUES (?, ?, 'complete', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL,
                      CURRENT_TIMESTAMP)
            ON CONFLICT(stock_id) DO UPDATE SET
                config_signature=excluded.config_signature,
                status='complete',
                shard_name=excluded.shard_name,
                total_rows=excluded.total_rows,
                feature_ready_rows=excluded.feature_ready_rows,
                eligible_5d_rows=excluded.eligible_5d_rows,
                eligible_10d_rows=excluded.eligible_10d_rows,
                eligible_20d_rows=excluded.eligible_20d_rows,
                liquidity_10m_rows=excluded.liquidity_10m_rows,
                liquidity_20m_rows=excluded.liquidity_20m_rows,
                liquidity_50m_rows=excluded.liquidity_50m_rows,
                liquidity_100m_rows=excluded.liquidity_100m_rows,
                first_signal_date=excluded.first_signal_date,
                last_signal_date=excluded.last_signal_date,
                error=NULL,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                stock_id,
                signature,
                shard_name,
                stats.total_rows,
                stats.feature_ready_rows,
                stats.eligible_5d_rows,
                stats.eligible_10d_rows,
                stats.eligible_20d_rows,
                stats.liquidity_rows.get(10_000_000, 0),
                stats.liquidity_rows.get(20_000_000, 0),
                stats.liquidity_rows.get(50_000_000, 0),
                stats.liquidity_rows.get(100_000_000, 0),
                stats.first_signal_date,
                stats.last_signal_date,
            ),
        )


def _save_stock_failure(
    database: ResearchDatabase, stock_id: str, signature: str, error: str
) -> None:
    database.execute(
        """
        INSERT INTO phase3_build_status (
            stock_id, config_signature, status, error, updated_at
        ) VALUES (?, ?, 'failed', ?, CURRENT_TIMESTAMP)
        ON CONFLICT(stock_id) DO UPDATE SET
            config_signature=excluded.config_signature,
            status='failed',
            error=excluded.error,
            updated_at=CURRENT_TIMESTAMP
        """,
        (stock_id, signature, error[:2000]),
    )


def _prefix(values: Iterable[float]) -> list[float]:
    result = [0.0]
    running = 0.0
    for value in values:
        running += float(value)
        result.append(running)
    return result


def _window_sum(prefix: list[float], index: int, window: int) -> float:
    start = index - window + 1
    return prefix[index + 1] - prefix[start]


def _signed_streaks(values: list[float]) -> list[int]:
    result: list[int] = []
    previous = 0
    for value in values:
        if value > 0:
            previous = previous + 1 if previous > 0 else 1
        elif value < 0:
            previous = previous - 1 if previous < 0 else -1
        else:
            previous = 0
        result.append(previous)
    return result


def _max_zero_streak(values: list[float]) -> int:
    maximum = 0
    running = 0
    for value in values:
        if value <= 0:
            running += 1
            maximum = max(maximum, running)
        else:
            running = 0
    return maximum


def _entry_price_available(
    signal_index: int,
    market_dates: list[str],
    price_by_date: dict[str, dict[str, Any]],
) -> bool:
    entry_index = signal_index + 1
    return bool(
        entry_index < len(market_dates)
        and _valid_entry_price(price_by_date.get(market_dates[entry_index]))
    )


def _valid_signal_price(row: dict[str, Any] | None) -> bool:
    return bool(
        row
        and _number(row.get("open")) > 0
        and _number(row.get("close")) > 0
        and _number(row.get("trading_volume")) > 0
        and _number(row.get("trading_money")) > 0
    )


def _valid_entry_price(row: dict[str, Any] | None) -> bool:
    return bool(
        row
        and _number(row.get("open")) > 0
        and _number(row.get("close")) > 0
        and _number(row.get("trading_volume")) > 0
    )


def _valid_close(row: dict[str, Any] | None) -> bool:
    return bool(row and _number(row.get("close")) > 0)


def _number(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def canonical_adjusted_return(value: float | str) -> float:
    return round(float(value), RETURN_ROUND_DIGITS)


def classify_10d_label(value: float | str) -> str:
    canonical = canonical_adjusted_return(value)
    if canonical >= LABEL_THRESHOLD:
        return "UP"
    if canonical <= -LABEL_THRESHOLD:
        return "DOWN"
    return "FLAT"


def _round_or_blank(value: float | None) -> float | str:
    return "" if value is None else canonical_adjusted_return(value)


def _day_before(value: str) -> str:
    return (date.fromisoformat(value) - timedelta(days=1)).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_shard_for_debug(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
