from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from html import unescape
from html.parser import HTMLParser
import gzip
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import sqlite3
from typing import Any, Sequence
import unicodedata
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from src.config_loader import enabled_stocks, load_user_configs


LOGGER = logging.getLogger("tw-stock-monitor.institutional.daily")
TAIPEI = ZoneInfo("Asia/Taipei")
DEFAULT_STATE_DIR = Path("runtime/institutional")
DEFAULT_MODEL_DIR = Path("models/tpex")
DEFAULT_DATABASE = Path("research/data/institutional_phase1.sqlite")
QUOTE_ENDPOINT = (
    "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"
)
FLOW_ENDPOINT = "https://www.tpex.org.tw/openapi/v1/tpex_3insti_daily_trading"
HISTORICAL_QUOTE_ENDPOINT = (
    "https://www.tpex.org.tw/web/stock/aftertrading/otc_quotes_no1430/"
    "stk_wn1430_result.php"
)
HISTORICAL_FLOW_ENDPOINT = (
    "https://www.tpex.org.tw/web/stock/3insti/daily_trade/"
    "3itrade_hedge_result.php"
)
HISTORICAL_FLOW_REFERER = (
    "https://www.tpex.org.tw/web/stock/3insti/daily_trade/"
    "3itrade_hedge.php?l=zh-tw"
)
ROLLING_COLUMNS = (
    "date",
    "stock_id",
    "stock_name",
    "trading_volume",
    "trading_money",
    "open",
    "high",
    "low",
    "close",
    "foreign_net",
    "investment_trust_net",
    "dealer_self_net",
    "selected_total_net",
)
FEATURE_GROUPS = {
    "foreign": "外資",
    "investment_trust": "投信",
    "dealer_self": "自營商自行買賣",
    "consensus": "三法人一致性／加速度",
}
FEATURE_LABELS = {
    "foreign_flow_pct_1d": "外資單日買賣超占成交量",
    "foreign_flow_pct_5d": "外資近5日買賣超占成交量",
    "foreign_flow_pct_20d": "外資近20日買賣超占成交量",
    "investment_trust_flow_pct_1d": "投信單日買賣超占成交量",
    "investment_trust_flow_pct_5d": "投信近5日買賣超占成交量",
    "investment_trust_flow_pct_20d": "投信近20日買賣超占成交量",
    "dealer_self_flow_pct_1d": "自營商單日買賣超占成交量",
    "dealer_self_flow_pct_5d": "自營商近5日買賣超占成交量",
    "dealer_self_flow_pct_20d": "自營商近20日買賣超占成交量",
    "foreign_buy_day_ratio_5d": "外資近5日買超日比例",
    "foreign_buy_day_ratio_20d": "外資近20日買超日比例",
    "investment_trust_buy_day_ratio_5d": "投信近5日買超日比例",
    "investment_trust_buy_day_ratio_20d": "投信近20日買超日比例",
    "dealer_self_buy_day_ratio_5d": "自營商近5日買超日比例",
    "dealer_self_buy_day_ratio_20d": "自營商近20日買超日比例",
    "foreign_streak": "外資連續買賣超",
    "investment_trust_streak": "投信連續買賣超",
    "dealer_self_streak": "自營商連續買賣超",
    "institutional_agreement_1d": "三法人單日方向一致性",
    "institutional_agreement_5d": "三法人近5日方向一致性",
    "institutional_agreement_20d": "三法人近20日方向一致性",
    "selected_total_acceleration_5d_vs_20d": "三法人近5日相對20日加速度",
}


@dataclass(frozen=True, slots=True)
class DeployModel:
    feature_columns: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    intercept: float
    signature: str

    def transform(self, values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, self.lower, self.upper)
        return (clipped - self.mean) / self.std

    def score(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        transformed = self.transform(values)
        contributions = transformed * self.weights
        return contributions.sum(axis=1) + self.intercept, contributions


@dataclass(frozen=True, slots=True)
class PipelineResult:
    status: str
    signal_date: str
    source_rows: int
    eligible_stocks: int
    notification_rows: int
    state_dir: str


class TpexOpenApiClient:
    def __init__(
        self,
        *,
        quote_endpoint: str = QUOTE_ENDPOINT,
        flow_endpoint: str = FLOW_ENDPOINT,
        historical_quote_endpoint: str = HISTORICAL_QUOTE_ENDPOINT,
        historical_flow_endpoint: str = HISTORICAL_FLOW_ENDPOINT,
        timeout_seconds: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self.quote_endpoint = quote_endpoint
        self.flow_endpoint = flow_endpoint
        self.historical_quote_endpoint = historical_quote_endpoint
        self.historical_flow_endpoint = historical_flow_endpoint
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()

    def fetch_latest(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return self._get(self.quote_endpoint), self._get(self.flow_endpoint)

    def fetch_date(
        self,
        trade_date: date,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        roc_year = trade_date.year - 1911
        quote_payload = self._get_payload(
            self.historical_quote_endpoint,
            params={
                "l": "zh-tw",
                "o": "json",
                "d": f"{roc_year:03d}/{trade_date:%m/%d}",
                "se": "EW",
                "s": "0,asc,0",
            },
        )
        flow_params = {
            "l": "zh-tw",
            "o": "json",
            "se": "EW",
            "t": "D",
            "d": f"{roc_year:03d}/{trade_date:%m/%d}",
            "s": "0,asc",
        }
        flow_payload = self._get_payload(
            self.historical_flow_endpoint,
            params=flow_params,
            headers={"Referer": HISTORICAL_FLOW_REFERER},
        )
        quote_rows = _historical_quote_payload_rows(quote_payload, trade_date)
        flow_rows = _historical_flow_payload_rows(flow_payload, trade_date)
        if quote_rows and not flow_rows:
            LOGGER.warning("TPEx %s 法人 JSON 為空，改讀官方 HTML", trade_date)
            html_params = dict(flow_params)
            html_params["o"] = "htm"
            flow_html = self._get_text(
                self.historical_flow_endpoint,
                params=html_params,
                headers={"Referer": HISTORICAL_FLOW_REFERER},
            )
            flow_rows = _historical_flow_html_rows(flow_html, trade_date)
        if bool(quote_rows) != bool(flow_rows):
            raise RuntimeError(
                f"TPEx {trade_date} 行情與法人歷史資料只有一方有值："
                f"行情={len(quote_rows)}、法人={len(flow_rows)}，拒絕補資料"
            )
        return quote_rows, flow_rows

    def _get(self, url: str) -> list[dict[str, Any]]:
        payload = self._get_payload(url)
        if not isinstance(payload, list):
            raise RuntimeError(f"TPEx OpenAPI 回傳格式不是陣列：{url}")
        return [row for row in payload if isinstance(row, dict)]

    def _get_payload(
        self,
        url: str,
        *,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        request_headers = {
            "Accept": "application/json",
            "Accept-Language": "zh-TW,zh;q=0.9",
            "User-Agent": "tw-stock-monitor/phase5h",
        }
        request_headers.update(headers or {})
        response = self.session.get(
            url,
            params=params,
            timeout=self.timeout_seconds,
            headers=request_headers,
        )
        response.raise_for_status()
        return response.json()

    def _get_text(
        self,
        url: str,
        *,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        request_headers = {
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "zh-TW,zh;q=0.9",
            "User-Agent": "tw-stock-monitor/phase5h",
        }
        request_headers.update(headers or {})
        response = self.session.get(
            url,
            params=params,
            timeout=self.timeout_seconds,
            headers=request_headers,
        )
        response.raise_for_status()
        return str(response.text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TPEx 法人模型每日部署管線")
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed = subparsers.add_parser("seed", help="由本機歷史 SQLite 建立部署 seed")
    seed.add_argument("--db", type=Path, default=DEFAULT_DATABASE)
    seed.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    seed.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    seed.add_argument("--users-config", default="configs/users")
    seed.add_argument("--lookback-market-days", type=int, default=100)
    seed.add_argument("--as-of-date")

    update = subparsers.add_parser("update", help="抓取最新 TPEx 日資料並更新通知計畫")
    update.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    update.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    update.add_argument("--users-config", default="configs/users")
    update.add_argument("--lookback-market-days", type=int, default=100)
    update.add_argument("--minimum-daily-stocks", type=int, default=50)
    update.add_argument("--minimum-source-coverage-ratio", type=float, default=0.85)
    update.add_argument("--maximum-catchup-calendar-days", type=int, default=31)
    update.add_argument("--quote-endpoint", default=QUOTE_ENDPOINT)
    update.add_argument("--flow-endpoint", default=FLOW_ENDPOINT)
    update.add_argument("--historical-quote-endpoint", default=HISTORICAL_QUOTE_ENDPOINT)
    update.add_argument("--historical-flow-endpoint", default=HISTORICAL_FLOW_ENDPOINT)
    update.add_argument("--as-of-date")

    parser.add_argument("--log-level", default="INFO")
    return parser


def cli() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        if args.command == "seed":
            result = create_seed(
                database_path=args.db,
                state_dir=args.state_dir,
                model_dir=args.model_dir,
                users_config=args.users_config,
                lookback_market_days=args.lookback_market_days,
                as_of_date=args.as_of_date,
            )
        else:
            result = update_daily(
                state_dir=args.state_dir,
                model_dir=args.model_dir,
                users_config=args.users_config,
                lookback_market_days=args.lookback_market_days,
                minimum_daily_stocks=args.minimum_daily_stocks,
                minimum_source_coverage_ratio=args.minimum_source_coverage_ratio,
                maximum_catchup_calendar_days=args.maximum_catchup_calendar_days,
                as_of_date=args.as_of_date,
                client=TpexOpenApiClient(
                    quote_endpoint=args.quote_endpoint,
                    flow_endpoint=args.flow_endpoint,
                    historical_quote_endpoint=args.historical_quote_endpoint,
                    historical_flow_endpoint=args.historical_flow_endpoint,
                ),
            )
    except Exception:
        LOGGER.exception("Phase 5H 執行失敗")
        raise SystemExit(1) from None
    LOGGER.info(
        "Phase 5H %s：訊號日=%s、來源列=%s、合格股票=%s、通知=%s",
        result.status,
        result.signal_date,
        result.source_rows,
        result.eligible_stocks,
        result.notification_rows,
    )


def create_seed(
    *,
    database_path: Path | str,
    state_dir: Path | str,
    model_dir: Path | str,
    users_config: Path | str,
    lookback_market_days: int = 100,
    as_of_date: str | None = None,
) -> PipelineResult:
    if lookback_market_days < 60:
        raise ValueError("seed 至少需要 60 個市場交易日")
    db_path = Path(database_path)
    if not db_path.exists():
        raise FileNotFoundError(f"找不到歷史 SQLite：{db_path}")
    state = Path(state_dir)
    state.mkdir(parents=True, exist_ok=True)
    model = load_deploy_model(model_dir)
    resolved_as_of = _parse_iso_date(as_of_date) if as_of_date else None

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        universe = pd.read_sql_query(
            """
            SELECT stock_id, COALESCE(stock_name, '') AS stock_name,
                   COALESCE(listing_date, '') AS listing_date
            FROM model_universe
            WHERE LOWER(market_type)='tpex'
              AND current_status='active'
              AND training_enabled=1
            ORDER BY stock_id
            """,
            connection,
            dtype={"stock_id": "string"},
        )
        if universe.empty:
            raise RuntimeError("歷史 SQLite 找不到現行 TPEx 普通股模型母體")

        normalized_price_date = (
            "CASE "
            "WHEN LENGTH(TRIM(p.date)) >= 10 "
            "THEN SUBSTR(REPLACE(TRIM(p.date), '/', '-'), 1, 10) "
            "ELSE TRIM(p.date) END"
        )
        expected_stocks = int(universe["stock_id"].nunique())
        required_stocks = max(50, int(math.floor(expected_stocks * 0.70)))
        date_sql = (
            "SELECT " + normalized_price_date + " AS date, "
            "COUNT(DISTINCT p.stock_id) AS stock_count "
            "FROM stock_prices p "
            "JOIN model_universe u ON u.stock_id=p.stock_id "
            "WHERE TRIM(p.date) <> '' "
            "AND LOWER(u.market_type)='tpex' "
            "AND u.current_status='active' "
            "AND u.training_enabled=1"
        )
        params: tuple[Any, ...] = ()
        if resolved_as_of:
            date_sql += " AND " + normalized_price_date + " <= ?"
            params = (resolved_as_of.isoformat(),)
        date_sql += (
            " GROUP BY " + normalized_price_date +
            " HAVING COUNT(DISTINCT p.stock_id) >= ?"
            " ORDER BY date DESC LIMIT ?"
        )
        recent_date_rows = list(
            connection.execute(
                date_sql,
                (*params, required_stocks, lookback_market_days),
            )
        )
        recent_dates = sorted(
            value
            for value in (_normalize_date_text(row["date"]) for row in recent_date_rows)
            if value
        )
        if len(recent_dates) < 60:
            latest_partial = connection.execute(
                (
                    "SELECT " + normalized_price_date + " AS date, "
                    "COUNT(DISTINCT p.stock_id) AS stock_count "
                    "FROM stock_prices p "
                    "JOIN model_universe u ON u.stock_id=p.stock_id "
                    "WHERE TRIM(p.date) <> '' "
                    "AND LOWER(u.market_type)='tpex' "
                    "AND u.current_status='active' "
                    "AND u.training_enabled=1 "
                    "GROUP BY " + normalized_price_date +
                    " ORDER BY date DESC LIMIT 1"
                )
            ).fetchone()
            latest_detail = (
                f"；資料庫最晚日={latest_partial['date']}、檔數={latest_partial['stock_count']}"
                if latest_partial is not None
                else ""
            )
            raise RuntimeError(
                f"歷史 SQLite 只有 {len(recent_dates)} 個完整市場日，至少需要 60 日"
                f"（完整日門檻={required_stocks}/{expected_stocks}）{latest_detail}"
            )
        placeholders = ",".join("?" for _ in recent_dates)
        raw = pd.read_sql_query(
            f"""
            SELECT
                   CASE
                     WHEN LENGTH(TRIM(p.date)) >= 10
                     THEN SUBSTR(REPLACE(TRIM(p.date), '/', '-'), 1, 10)
                     ELSE TRIM(p.date)
                   END AS date,
                   p.stock_id, COALESCE(u.stock_name, '') AS stock_name,
                   p.trading_volume, p.trading_money, p.open, p.high, p.low, p.close,
                   COALESCE(f.foreign_net, 0) AS foreign_net,
                   COALESCE(f.investment_trust_net, 0) AS investment_trust_net,
                   COALESCE(f.dealer_self_net, 0) AS dealer_self_net,
                   COALESCE(f.selected_total_net, 0) AS selected_total_net
            FROM stock_prices p
            JOIN model_universe u ON u.stock_id=p.stock_id
            LEFT JOIN institutional_flows f
              ON f.stock_id=p.stock_id
             AND (
                   CASE
                     WHEN LENGTH(TRIM(f.date)) >= 10
                     THEN SUBSTR(REPLACE(TRIM(f.date), '/', '-'), 1, 10)
                     ELSE TRIM(f.date)
                   END
                 ) = (
                   CASE
                     WHEN LENGTH(TRIM(p.date)) >= 10
                     THEN SUBSTR(REPLACE(TRIM(p.date), '/', '-'), 1, 10)
                     ELSE TRIM(p.date)
                   END
                 )
            WHERE LOWER(u.market_type)='tpex'
              AND u.current_status='active'
              AND u.training_enabled=1
              AND (
                    CASE
                      WHEN LENGTH(TRIM(p.date)) >= 10
                      THEN SUBSTR(REPLACE(TRIM(p.date), '/', '-'), 1, 10)
                      ELSE TRIM(p.date)
                    END
                  ) IN ({placeholders})
            ORDER BY date, p.stock_id
            """,
            connection,
            params=recent_dates,
            dtype={"stock_id": "string", "date": "string"},
        )

    raw = normalize_rolling_frame(raw)
    _validate_source_coverage(raw, universe, minimum_ratio=0.70, minimum_stocks=50)
    write_rolling_state(state, raw, universe)
    outputs = rebuild_outputs(
        rolling=raw,
        universe=universe,
        model=model,
        user_configs=load_user_configs(users_config),
        state_dir=state,
        minimum_daily_stocks=50,
        ready_to_send=False,
        generated_reason="LOCAL_SEED",
    )
    write_manifest(
        state,
        {
            "phase5h_version": "phase5h-v1",
            "status": "SEEDED",
            "signal_date": outputs["signal_date"],
            "source": "local_sqlite",
            "model_signature": model.signature,
            "rolling_rows": len(raw),
            "universe_stocks": int(universe["stock_id"].nunique()),
            "eligible_stocks": outputs["eligible_stocks"],
            "notification_rows": outputs["notification_rows"],
            "generated_at": datetime.now(TAIPEI).isoformat(timespec="seconds"),
            "ready_to_send": 0,
        },
    )
    return PipelineResult(
        status="SEEDED",
        signal_date=outputs["signal_date"],
        source_rows=len(raw),
        eligible_stocks=outputs["eligible_stocks"],
        notification_rows=outputs["notification_rows"],
        state_dir=str(state),
    )


def update_daily(
    *,
    state_dir: Path | str,
    model_dir: Path | str,
    users_config: Path | str,
    lookback_market_days: int = 100,
    minimum_daily_stocks: int = 50,
    minimum_source_coverage_ratio: float = 0.85,
    maximum_catchup_calendar_days: int = 31,
    as_of_date: str | None = None,
    client: TpexOpenApiClient | Any | None = None,
) -> PipelineResult:
    if lookback_market_days < 60:
        raise ValueError("滾動資料至少保留 60 個市場交易日")
    if not 0.5 <= minimum_source_coverage_ratio <= 1.0:
        raise ValueError("minimum_source_coverage_ratio 必須介於 0.5～1.0")
    if maximum_catchup_calendar_days < 1:
        raise ValueError("maximum_catchup_calendar_days 必須大於 0")
    state = Path(state_dir)
    rolling_path = state / "rolling_market_data.csv.gz"
    universe_path = state / "universe.csv"
    if not rolling_path.exists() or not universe_path.exists():
        raise FileNotFoundError(
            "找不到 Phase 5H seed；請先在本機執行 seed 並發佈到 state branch"
        )
    model = load_deploy_model(model_dir)
    rolling = pd.read_csv(
        rolling_path,
        compression="gzip",
        dtype={"date": "string", "stock_id": "string"},
        low_memory=False,
    )
    universe = pd.read_csv(
        universe_path,
        dtype={"stock_id": "string", "listing_date": "string"},
    ).fillna("")
    api = client or TpexOpenApiClient()
    effective_today = _parse_iso_date(as_of_date) if as_of_date else datetime.now(TAIPEI).date()
    previous_dates = sorted(rolling["date"].astype(str).unique())
    previous_latest = previous_dates[-1] if previous_dates else ""
    if not previous_latest:
        raise RuntimeError("Phase 5H seed 沒有任何市場日期")
    previous_day = date.fromisoformat(previous_latest)
    if previous_day > effective_today:
        raise RuntimeError(
            f"Phase 5H seed 最新日 {previous_latest} 晚於執行基準日 {effective_today}"
        )
    gap_days = (effective_today - previous_day).days
    if gap_days > maximum_catchup_calendar_days:
        raise RuntimeError(
            "Phase 5H seed 與執行基準日相差 "
            f"{gap_days} 個日曆日，超過允許的 {maximum_catchup_calendar_days} 日；"
            "請先在本機重新建立 seed"
        )

    prior_counts = rolling.groupby("date")["stock_id"].nunique()
    prior_reference = float(prior_counts.tail(20).median()) if len(prior_counts) else 0.0
    required = max(minimum_daily_stocks, int(math.floor(prior_reference * minimum_source_coverage_ratio)))
    appended_frames: list[pd.DataFrame] = []
    fetched_dates: list[str] = []
    cursor = previous_day + timedelta(days=1)
    while cursor <= effective_today:
        historical_quotes, historical_flows = api.fetch_date(cursor)
        if historical_quotes and historical_flows:
            historical = merge_tpex_daily_rows(
                historical_quotes,
                historical_flows,
                universe,
            )
            historical_dates = sorted(historical["date"].astype(str).unique())
            if historical_dates != [cursor.isoformat()]:
                raise RuntimeError(
                    f"TPEx 依日期查詢 {cursor} 卻回傳 {historical_dates}，拒絕更新"
                )
            historical_count = int(historical["stock_id"].nunique())
            if historical_count < required:
                raise RuntimeError(
                    f"TPEx {cursor} 共同資料只有 {historical_count} 檔，"
                    f"低於要求 {required} 檔"
                )
            appended_frames.append(historical)
            fetched_dates.append(cursor.isoformat())
        cursor += timedelta(days=1)

    signal_date = fetched_dates[-1] if fetched_dates else previous_latest
    signal_day = date.fromisoformat(signal_date)
    if (effective_today - signal_day).days > 7:
        raise RuntimeError(
            f"TPEx 可用資料過舊：{signal_date}，距基準日超過 7 日"
        )
    is_new_market_date = bool(fetched_dates)
    latest_count = (
        int(appended_frames[-1]["stock_id"].nunique())
        if appended_frames
        else int(rolling.loc[rolling["date"] == previous_latest, "stock_id"].nunique())
    )

    combined_parts = [rolling, *appended_frames]
    combined = normalize_rolling_frame(pd.concat(combined_parts, ignore_index=True))
    dates = sorted(combined["date"].unique())[-lookback_market_days:]
    combined = combined[combined["date"].isin(dates)].copy()
    write_rolling_state(state, combined, universe)

    outputs = rebuild_outputs(
        rolling=combined,
        universe=universe,
        model=model,
        user_configs=load_user_configs(users_config),
        state_dir=state,
        minimum_daily_stocks=minimum_daily_stocks,
        ready_to_send=is_new_market_date,
        generated_reason="TPEX_OFFICIAL_DAILY_REPORT",
    )
    status = "UPDATED" if is_new_market_date else "ALREADY_CURRENT"
    write_manifest(
        state,
        {
            "phase5h_version": "phase5h-v1",
            "status": status,
            "signal_date": signal_date,
            "previous_signal_date": previous_latest,
            "source": "tpex_official_daily_report",
            "quote_endpoint": getattr(
                api,
                "historical_quote_endpoint",
                HISTORICAL_QUOTE_ENDPOINT,
            ),
            "flow_endpoint": getattr(
                api,
                "historical_flow_endpoint",
                HISTORICAL_FLOW_ENDPOINT,
            ),
            "model_signature": model.signature,
            "rolling_rows": len(combined),
            "universe_stocks": int(universe["stock_id"].nunique()),
            "latest_source_stocks": latest_count,
            "required_source_stocks": required,
            "catchup_market_dates": fetched_dates,
            "catchup_market_date_count": len(fetched_dates),
            "eligible_stocks": outputs["eligible_stocks"],
            "notification_rows": outputs["notification_rows"],
            "generated_at": datetime.now(TAIPEI).isoformat(timespec="seconds"),
            "ready_to_send": int(is_new_market_date),
        },
    )
    return PipelineResult(
        status=status,
        signal_date=signal_date,
        source_rows=sum(len(frame) for frame in appended_frames),
        eligible_stocks=outputs["eligible_stocks"],
        notification_rows=outputs["notification_rows"],
        state_dir=str(state),
    )


def load_deploy_model(model_dir: Path | str) -> DeployModel:
    root = Path(model_dir)
    manifest_path = root / "phase5d_final_rank_model.json"
    arrays_path = root / "phase5d_final_rank_model.npz"
    if not manifest_path.exists() or not arrays_path.exists():
        raise FileNotFoundError(f"找不到 Phase 5D 部署模型：{root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrays_sha = hashlib.sha256(arrays_path.read_bytes()).hexdigest()
    if arrays_sha != str(manifest.get("arrays_sha256") or ""):
        raise RuntimeError("Phase 5D 模型陣列 SHA-256 不一致")
    arrays = np.load(arrays_path)
    features = tuple(str(value) for value in manifest.get("feature_columns", []))
    required_arrays = {"lower", "upper", "mean", "std", "weights", "intercept"}
    missing = required_arrays.difference(arrays.files)
    if missing:
        raise RuntimeError(f"Phase 5D 模型陣列缺少：{sorted(missing)}")
    lengths = {len(arrays[name]) for name in ("lower", "upper", "mean", "std", "weights")}
    if lengths != {len(features)}:
        raise RuntimeError("Phase 5D 模型特徵與陣列長度不一致")
    if np.any(np.asarray(arrays["std"], dtype=float) <= 0):
        raise RuntimeError("Phase 5D 模型標準差包含非正數")
    return DeployModel(
        feature_columns=features,
        lower=np.asarray(arrays["lower"], dtype=float),
        upper=np.asarray(arrays["upper"], dtype=float),
        mean=np.asarray(arrays["mean"], dtype=float),
        std=np.asarray(arrays["std"], dtype=float),
        weights=np.asarray(arrays["weights"], dtype=float),
        intercept=float(np.asarray(arrays["intercept"]).ravel()[0]),
        signature=str(manifest.get("signature") or ""),
    )


def merge_tpex_daily_rows(
    quote_rows: Sequence[dict[str, Any]],
    flow_rows: Sequence[dict[str, Any]],
    universe: pd.DataFrame,
) -> pd.DataFrame:
    quotes = pd.DataFrame([parse_quote_row(row) for row in quote_rows])
    flows = pd.DataFrame([parse_flow_row(row) for row in flow_rows])
    if quotes.empty or flows.empty:
        raise RuntimeError("TPEx OpenAPI 行情或法人資料為空")
    quote_dates = sorted(value for value in quotes["date"].astype(str).unique() if value)
    flow_dates = sorted(value for value in flows["date"].astype(str).unique() if value)
    if len(quote_dates) == 1 and not flow_dates:
        flows["date"] = quote_dates[0]
    if len(flow_dates) == 1 and not quote_dates:
        quotes["date"] = flow_dates[0]
    quotes = quotes[quotes["stock_id"].astype(str).str.fullmatch(r"\d{4}")].copy()
    flows = flows[flows["stock_id"].astype(str).str.fullmatch(r"\d{4}")].copy()
    allowed = set(universe["stock_id"].astype(str))
    quotes = quotes[quotes["stock_id"].isin(allowed)]
    flows = flows[flows["stock_id"].isin(allowed)]
    if quotes.duplicated(["date", "stock_id"]).any():
        raise RuntimeError("TPEx 行情資料有重複股票日期")
    if flows.duplicated(["date", "stock_id"]).any():
        raise RuntimeError("TPEx 法人資料有重複股票日期")
    merged = quotes.merge(
        flows.drop(columns=["stock_name"], errors="ignore"),
        on=["date", "stock_id"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        quote_sample = quotes["stock_id"].astype(str).head(5).tolist()
        flow_sample = flows["stock_id"].astype(str).head(5).tolist()
        universe_sample = sorted(allowed)[:5]
        raise RuntimeError(
            "TPEx 行情與法人資料合併後為0檔；"
            f"行情代號樣本={quote_sample}、法人代號樣本={flow_sample}、"
            f"母體代號樣本={universe_sample}"
        )
    return normalize_rolling_frame(merged)


def _historical_quote_payload_rows(
    payload: Any,
    trade_date: date,
) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = [dict(row) for row in payload if isinstance(row, dict)]
    elif isinstance(payload, dict):
        rows = []
        tables = payload.get("tables")
        if isinstance(tables, list):
            for table in tables:
                if not isinstance(table, dict):
                    continue
                fields = table.get("fields") or table.get("columns")
                data = table.get("data") or table.get("aaData")
                rows.extend(_table_rows(fields, data))
        if not rows:
            fields = payload.get("fields") or payload.get("columns")
            data = payload.get("data") or payload.get("aaData")
            rows.extend(_table_rows(fields, data))
        if not rows:
            raw_rows = payload.get("aaData") or payload.get("data") or []
            if isinstance(raw_rows, list):
                for raw in raw_rows:
                    if isinstance(raw, dict):
                        rows.append(dict(raw))
                        continue
                    if not isinstance(raw, list) or len(raw) < 10:
                        continue
                    # TPEx historical OTC quote result columns:
                    # code, name, close, change, open, high, low,
                    # volume, amount, transaction count, ...
                    rows.append(
                        {
                            "Date": trade_date.isoformat(),
                            "SecuritiesCompanyCode": raw[0],
                            "CompanyName": raw[1],
                            "Close": raw[2],
                            "Open": raw[4],
                            "High": raw[5],
                            "Low": raw[6],
                            "TradingShares": raw[7],
                            "TransactionAmount": raw[8],
                        }
                    )
    else:
        rows = []
    for row in rows:
        if not _date_value(row, "Date", "date", "資料日期"):
            row["Date"] = trade_date.isoformat()
    return rows


def _historical_flow_payload_rows(
    payload: Any,
    trade_date: date,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(payload, list):
        rows.extend(_historical_flow_array_rows(payload, trade_date))
    elif isinstance(payload, dict):
        # TPEx's legacy response may include both field metadata and aaData.
        # aaData is the authoritative positional layout for this report; parsing
        # it as a generic field table can create non-empty dictionaries whose
        # stock-code key is unusable. Always prefer the raw array first.
        rows.extend(_historical_flow_array_rows(payload.get("aaData"), trade_date))

        tables = payload.get("tables")
        if not rows and isinstance(tables, list):
            for table in tables:
                if not isinstance(table, dict):
                    continue
                rows.extend(
                    _historical_flow_array_rows(table.get("aaData"), trade_date)
                )
                if rows:
                    break
                data = table.get("data")
                rows.extend(_historical_flow_array_rows(data, trade_date))
                if rows:
                    break
                fields = table.get("fields") or table.get("columns")
                rows.extend(_table_rows(fields, data))
                if rows:
                    break

        if not rows:
            data = payload.get("data")
            rows.extend(_historical_flow_array_rows(data, trade_date))
            if not rows:
                fields = payload.get("fields") or payload.get("columns")
                rows.extend(_table_rows(fields, data))
    for row in rows:
        if not _date_value(row, "Date", "date", "資料日期"):
            row["Date"] = trade_date.isoformat()
    return rows


def _historical_flow_array_rows(
    raw_rows: Any,
    trade_date: date,
) -> list[dict[str, Any]]:
    if not isinstance(raw_rows, list):
        return []
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        if isinstance(raw, dict):
            row = dict(raw)
            row.setdefault("Date", trade_date.isoformat())
            rows.append(row)
            continue
        row = _historical_flow_array_row(raw, trade_date)
        if row is not None:
            rows.append(row)
    return rows


def _historical_flow_array_row(
    raw: Any,
    trade_date: date,
) -> dict[str, Any] | None:
    if not isinstance(raw, list) or len(raw) < 17:
        return None
    stock_id = _clean_html_cell(raw[0])
    if not re.fullmatch(r"[0-9A-Za-z]{4,6}", stock_id):
        return None
    # TPEx daily detail keeps proprietary dealing at columns 14～16.
    # Hedging columns 17～19 are intentionally not used by this model.
    return {
        "Date": trade_date.isoformat(),
        "SecuritiesCompanyCode": stock_id,
        "CompanyName": _clean_html_cell(raw[1]),
        "ForeignInvestorsBuy": raw[8],
        "ForeignInvestorsSell": raw[9],
        "ForeignInvestorsDifference": raw[10],
        "SecuritiesInvestmentTrustCompaniesBuy": raw[11],
        "SecuritiesInvestmentTrustCompaniesSell": raw[12],
        "SecuritiesInvestmentTrustCompaniesDifference": raw[13],
        "DealersSelfBuy": raw[14],
        "DealersSelfSell": raw[15],
        "DealersSelfDifference": raw[16],
    }


class _TpexTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.lower() == "tr":
            self._row = []
        elif tag.lower() in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in {"td", "th"} and self._row is not None and self._cell is not None:
            self._row.append(_clean_html_cell("".join(self._cell)))
            self._cell = None
        elif lowered == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None
            self._cell = None


def _historical_flow_html_rows(
    html_text: str,
    trade_date: date,
) -> list[dict[str, Any]]:
    parser = _TpexTableParser()
    parser.feed(html_text)
    rows: list[dict[str, Any]] = []
    for raw in parser.rows:
        row = _historical_flow_array_row(raw, trade_date)
        if row is not None:
            rows.append(row)
    return rows


def _clean_html_cell(value: Any) -> str:
    text = unescape(str(value or "").replace("\xa0", " "))
    text = re.sub(r"<[^>]*>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _table_rows(fields: Any, data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, list):
        return []
    if all(isinstance(row, dict) for row in data):
        return [dict(row) for row in data]
    if not isinstance(fields, list):
        return []
    names = [_field_name(field) for field in fields]
    return [
        dict(zip(names, row, strict=False))
        for row in data
        if isinstance(row, list)
    ]


def _field_name(field: Any) -> str:
    if isinstance(field, str):
        return field
    if isinstance(field, dict):
        for key in ("name", "field", "title", "label", "key", "data"):
            value = field.get(key)
            if value not in (None, ""):
                return _clean_html_cell(value)
    return _clean_html_cell(field)


def parse_quote_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "date": _date_value(row, "Date", "date", "資料日期"),
        "stock_id": _text_value(
            row,
            "SecuritiesCompanyCode",
            "SecuritiesCode",
            "Code",
            "stock_id",
            "證券代號",
            "代號",
        ),
        "stock_name": _text_value(
            row,
            "CompanyName",
            "SecuritiesCompanyName",
            "Name",
            "stock_name",
            "證券名稱",
            "名稱",
        ),
        "trading_volume": _scaled_number_value(
            row,
            scaled_keys=("成交股數(千股)", "成交股數（千股）", "成交量(千股)"),
            scale=1000.0,
            normal_keys=(
                "TradingShares",
                "TradingVolume",
                "Trading_Volume",
                "成交股數",
                "成交量",
            ),
        ),
        "trading_money": _scaled_number_value(
            row,
            scaled_keys=("成交金額(千元)", "成交金額（千元）"),
            scale=1000.0,
            normal_keys=(
                "TransactionAmount",
                "TradingAmount",
                "Trading_money",
                "成交金額",
                "成交金額(元)",
                "成交金額（元）",
            ),
        ),
        "open": _number_value(row, "Open", "OpeningPrice", "open", "開盤", "開盤價"),
        "high": _number_value(row, "High", "HighestPrice", "max", "最高", "最高價"),
        "low": _number_value(row, "Low", "LowestPrice", "min", "最低", "最低價"),
        "close": _number_value(row, "Close", "ClosingPrice", "close", "收盤", "收盤價"),
    }


def parse_flow_row(row: dict[str, Any]) -> dict[str, Any]:
    foreign = _difference_value(
        row,
        difference=(
            "ForeignInvestorsDifference",
            "ForeignInvestorsNetBuySell",
            "ForeignInvestorsBuySellDifference",
            "Foreign Investors include Mainland Area Investors (Foreign Dealers excluded)-Difference",
            "外資及陸資買賣超股數",
        ),
        buy=(
            "ForeignInvestorsBuy",
            "Foreign Investors include Mainland Area Investors (Foreign Dealers excluded)-Total Buy",
            "外資及陸資買進股數",
        ),
        sell=(
            "ForeignInvestorsSell",
            "Foreign Investors include Mainland Area Investors (Foreign Dealers excluded)-Total Sell",
            "外資及陸資賣出股數",
        ),
    )
    trust = _difference_value(
        row,
        difference=(
            "SecuritiesInvestmentTrustCompaniesDifference",
            "InvestmentTrustDifference",
            "SecuritiesInvestmentTrustCompanies-Difference",
            "投信買賣超股數",
        ),
        buy=(
            "SecuritiesInvestmentTrustCompaniesBuy",
            "SecuritiesInvestmentTrustCompanies-TotalBuy",
            "InvestmentTrustBuy",
            "投信買進股數",
        ),
        sell=(
            "SecuritiesInvestmentTrustCompaniesSell",
            "SecuritiesInvestmentTrustCompanies-TotalSell",
            "InvestmentTrustSell",
            "投信賣出股數",
        ),
    )
    dealer_self = _difference_value(
        row,
        difference=(
            "DealersSelfDifference",
            "DealerSelfDifference",
            "Dealers(Proprietary)-Difference",
            "Dealers-Self-Difference",
            "Dealers(Self)-Difference",
            "自營商自行買賣買賣超股數",
        ),
        buy=(
            "DealersSelfBuy",
            "DealerSelfBuy",
            "Dealers(Proprietary)-TotalBuy",
            "Dealers-Self-TotalBuy",
            "Dealers(Self)-TotalBuy",
            "自營商自行買賣買進股數",
        ),
        sell=(
            "DealersSelfSell",
            "DealerSelfSell",
            "Dealers(Proprietary)-TotalSell",
            "Dealers-Self-TotalSell",
            "Dealers(Self)-TotalSell",
            "自營商自行買賣賣出股數",
        ),
    )
    return {
        "date": _date_value(row, "Date", "date", "資料日期"),
        "stock_id": _text_value(
            row,
            "SecuritiesCompanyCode",
            "SecuritiesCode",
            "Code",
            "Symbol",
            "symbol",
            "stock_id",
            "證券代號",
            "代號",
        ),
        "stock_name": _text_value(
            row,
            "CompanyName",
            "SecuritiesCompanyName",
            "Name",
            "name",
            "stock_name",
            "證券名稱",
            "名稱",
        ),
        "foreign_net": foreign,
        "investment_trust_net": trust,
        "dealer_self_net": dealer_self,
        "selected_total_net": foreign + trust + dealer_self,
    }


def normalize_rolling_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ROLLING_COLUMNS:
        if column not in result:
            result[column] = "" if column in {"date", "stock_id", "stock_name"} else 0
    result = result[list(ROLLING_COLUMNS)]
    result["date"] = result["date"].astype(str).map(_normalize_date_text)
    result["stock_id"] = result["stock_id"].astype(str).str.strip()
    result["stock_name"] = result["stock_name"].fillna("").astype(str).str.strip()
    numeric = [column for column in ROLLING_COLUMNS if column not in {"date", "stock_id", "stock_name"}]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
    result = result[(result["date"] != "") & (result["stock_id"] != "")]
    result = result.sort_values(["date", "stock_id"], kind="stable")
    result = result.drop_duplicates(["date", "stock_id"], keep="last").reset_index(drop=True)
    return result


def write_rolling_state(state_dir: Path, rolling: pd.DataFrame, universe: pd.DataFrame) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(state_dir / "rolling_market_data.csv.gz", rolling)
    _write_csv(state_dir / "universe.csv", universe)


def rebuild_outputs(
    *,
    rolling: pd.DataFrame,
    universe: pd.DataFrame,
    model: DeployModel,
    user_configs: list[dict[str, Any]],
    state_dir: Path,
    minimum_daily_stocks: int,
    ready_to_send: bool,
    generated_reason: str,
) -> dict[str, Any]:
    features = build_feature_history(rolling, universe, model.feature_columns)
    scores = score_feature_history(features, model, minimum_daily_stocks=minimum_daily_stocks)
    if scores.empty:
        raise RuntimeError("Phase 5H 沒有可評分的 TPEx 股票日期")
    lifecycle = replay_lifecycle(scores)
    latest_date = str(scores["signal_date"].max())
    latest_scores = scores[scores["signal_date"] == latest_date].copy()
    latest_notifications = lifecycle["notifications"]
    if not latest_notifications.empty:
        latest_notifications = latest_notifications[
            latest_notifications["signal_date"] == latest_date
        ].copy()
    plan = build_notification_plan(
        notifications=latest_notifications,
        latest_scores=latest_scores,
        user_configs=user_configs,
        ready_to_send=ready_to_send,
        generated_reason=generated_reason,
    )
    _write_gzip_csv(state_dir / "score_history.csv.gz", scores)
    _write_gzip_csv(state_dir / "latest_scores.csv.gz", latest_scores)
    _write_csv(state_dir / "lifecycle_events.csv", lifecycle["events"])
    _write_csv(state_dir / "lifecycle_notifications.csv", lifecycle["notifications"])
    _write_csv(state_dir / "notification_plan.csv", plan)
    return {
        "signal_date": latest_date,
        "eligible_stocks": int(latest_scores["stock_id"].nunique()),
        "notification_rows": len(plan),
    }


def build_feature_history(
    rolling: pd.DataFrame,
    universe: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    dates = sorted(rolling["date"].unique())
    if len(dates) < 20:
        raise RuntimeError("滾動資料不足 20 個市場交易日")
    names = universe.set_index("stock_id")["stock_name"].astype(str).to_dict()
    rows: list[dict[str, Any]] = []
    for stock_id, stock_frame in rolling.groupby("stock_id", sort=True):
        aligned = stock_frame.set_index("date").reindex(dates)
        volume = pd.to_numeric(aligned["trading_volume"], errors="coerce").fillna(0.0).to_numpy()
        money = pd.to_numeric(aligned["trading_money"], errors="coerce").fillna(0.0).to_numpy()
        close = pd.to_numeric(aligned["close"], errors="coerce").fillna(0.0).to_numpy()
        valid_price = (close > 0).astype(int)
        actor_values = {
            actor: pd.to_numeric(aligned[column], errors="coerce").fillna(0.0).to_numpy()
            for actor, column in (
                ("foreign", "foreign_net"),
                ("investment_trust", "investment_trust_net"),
                ("dealer_self", "dealer_self_net"),
                ("selected_total", "selected_total_net"),
            )
        }
        streaks = {actor: _signed_streaks(values) for actor, values in actor_values.items()}
        for index in range(19, len(dates)):
            money_window = money[index - 19 : index + 1]
            volume_window = volume[index - 19 : index + 1]
            normal_days = int(valid_price[index - 19 : index + 1].sum())
            row: dict[str, Any] = {
                "stock_id": str(stock_id),
                "stock_name": _safe_name(
                    aligned.iloc[index].get("stock_name"),
                    names.get(str(stock_id), ""),
                ),
                "signal_date": dates[index],
                "history_market_days_20": 20,
                "normal_trading_days_20d": normal_days,
                "median_trading_money_20d": float(np.median(money_window)),
                "max_zero_volume_streak_20d": _max_zero_streak(volume_window),
            }
            for actor in ("foreign", "investment_trust", "dealer_self", "selected_total"):
                values = actor_values[actor]
                for window in (1, 5, 20):
                    net = float(values[index - window + 1 : index + 1].sum())
                    total_volume = float(volume[index - window + 1 : index + 1].sum())
                    row[f"{actor}_flow_pct_{window}d"] = net / total_volume * 100 if total_volume > 0 else 0.0
                for window in (5, 20):
                    row[f"{actor}_buy_day_ratio_{window}d"] = float(
                        (values[index - window + 1 : index + 1] > 0).sum() / window
                    )
                row[f"{actor}_streak"] = int(streaks[actor][index])
            for window in (1, 5, 20):
                signs = []
                for actor in ("foreign", "investment_trust", "dealer_self"):
                    total = float(actor_values[actor][index - window + 1 : index + 1].sum())
                    signs.append(1 if total > 0 else -1 if total < 0 else 0)
                row[f"institutional_agreement_{window}d"] = sum(signs) / 3
            row["selected_total_acceleration_5d_vs_20d"] = (
                row["selected_total_flow_pct_5d"] - row["selected_total_flow_pct_20d"]
            )
            if all(column in row for column in feature_columns):
                rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["liquidity_pass_20m"] = (
        (frame["normal_trading_days_20d"] >= 18)
        & (frame["median_trading_money_20d"] >= 20_000_000)
        & (frame["max_zero_volume_streak_20d"] < 3)
    ).astype(int)
    return frame.sort_values(["signal_date", "stock_id"], kind="stable").reset_index(drop=True)


def score_feature_history(
    features: pd.DataFrame,
    model: DeployModel,
    *,
    minimum_daily_stocks: int,
) -> pd.DataFrame:
    eligible = features[features["liquidity_pass_20m"] == 1].copy()
    counts = eligible.groupby("signal_date")["stock_id"].transform("nunique")
    eligible = eligible[counts >= minimum_daily_stocks].copy()
    if eligible.empty:
        return eligible
    matrix = eligible[list(model.feature_columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(matrix).all():
        raise RuntimeError("Phase 5H 模型特徵包含 NaN／Infinity")
    raw_scores, contributions = model.score(matrix)
    eligible["return_rank_raw_score"] = raw_scores
    eligible["return_rank_score"] = eligible.groupby("signal_date")[
        "return_rank_raw_score"
    ].rank(method="average", pct=True) * 100.0
    for index, feature in enumerate(model.feature_columns):
        eligible[f"contribution__{feature}"] = contributions[:, index]
    for group in FEATURE_GROUPS:
        columns = [
            f"contribution__{feature}"
            for feature in model.feature_columns
            if _feature_group(feature) == group
        ]
        eligible[f"{group}_contribution"] = eligible[columns].sum(axis=1)
    factors = [
        _factor_strings(row, model.feature_columns)
        for _, row in eligible.iterrows()
    ]
    eligible["positive_factors"] = [item[0] for item in factors]
    eligible["negative_factors"] = [item[1] for item in factors]
    keep = [
        "stock_id",
        "stock_name",
        "signal_date",
        "return_rank_raw_score",
        "return_rank_score",
        "foreign_contribution",
        "investment_trust_contribution",
        "dealer_self_contribution",
        "consensus_contribution",
        "positive_factors",
        "negative_factors",
    ]
    return eligible[keep].sort_values(["signal_date", "stock_id"], kind="stable").reset_index(drop=True)


def replay_lifecycle(scores: pd.DataFrame) -> dict[str, pd.DataFrame]:
    dates = sorted(scores["signal_date"].unique())
    date_index = {value: index for index, value in enumerate(dates)}
    day_map = {
        value: group.set_index("stock_id", drop=False)
        for value, group in scores.groupby("signal_date", sort=True)
    }
    active: dict[str, dict[str, Any]] = {}
    cooldown_until: dict[str, int] = {}
    confirmation_run: dict[str, int] = {}
    last_seen: dict[str, int] = {}
    previous_score: dict[str, float] = {}
    events: list[dict[str, Any]] = []
    notifications: list[dict[str, Any]] = []

    for current_date in dates:
        current_index = date_index[current_date]
        day = day_map[current_date]
        stocks = set(day.index.astype(str))
        for stock_id in stocks:
            score = float(day.loc[stock_id, "return_rank_score"])
            consecutive = last_seen.get(stock_id) == current_index - 1
            if score >= 80:
                confirmation_run[stock_id] = confirmation_run.get(stock_id, 0) + 1 if consecutive else 1
            else:
                confirmation_run[stock_id] = 0

        for stock_id in list(active):
            event = active[stock_id]
            age = current_index - int(event["start_index"])
            row = day.loc[stock_id] if stock_id in day.index else None
            percentile = float(row["return_rank_score"]) if row is not None else math.nan
            day_notifications: list[dict[str, Any]] = []
            if not event["confirmed"] and row is not None and confirmation_run.get(stock_id, 0) >= 5:
                event["confirmed"] = True
                event["status"] = "LAYOUT_CONFIRMED"
                day_notifications.append(
                    _notification(event, current_date, "LAYOUT_CONFIRMED", percentile, age,
                                  "連續5個交易日法人排名維持前20%")
                )
            if age == 20:
                if row is None:
                    day_notifications.append(
                        _notification(event, current_date, "DAY20_END_NO_SCORE", percentile, age,
                                      "第20日無可用排名，結束本次追蹤")
                    )
                    _close_event(event, current_date, "DAY20_END_NO_SCORE")
                    cooldown_until[stock_id] = current_index + 20
                    del active[stock_id]
                elif percentile >= 90:
                    event["status"] = "EXTENDED_STRONG"
                    event["extended"] = True
                    day_notifications.append(
                        _notification(event, current_date, "DAY20_EXTEND_STRONG", percentile, age,
                                      "第20日仍在法人排名前10%，延長追蹤至第40日")
                    )
                elif percentile >= 80:
                    event["status"] = "EXTENDED"
                    event["extended"] = True
                    day_notifications.append(
                        _notification(event, current_date, "DAY20_EXTEND", percentile, age,
                                      "第20日仍在法人排名前20%，延長觀察至第40日")
                    )
                else:
                    day_notifications.append(
                        _notification(event, current_date, "DAY20_END_WEAKENED", percentile, age,
                                      "第20日已跌出前20%，結束積極追蹤；不代表賣出")
                    )
                    _close_event(event, current_date, "DAY20_END_WEAKENED")
                    cooldown_until[stock_id] = current_index + 20
                    del active[stock_id]
            elif age >= 40:
                day_notifications.append(
                    _notification(event, current_date, "DAY40_END", percentile, age,
                                  "已達第40日，結束本次法人布局事件")
                )
                _close_event(event, current_date, "DAY40_END")
                cooldown_until[stock_id] = current_index + 20
                del active[stock_id]
            notifications.extend(_merge_same_day_notifications(day_notifications))

        for stock_id in sorted(stocks):
            row = day.loc[stock_id]
            score = float(row["return_rank_score"])
            previous = previous_score.get(stock_id)
            first_crossing = score >= 90 and (
                previous is None
                or last_seen.get(stock_id) != current_index - 1
                or previous < 90
            )
            direct = confirmation_run.get(stock_id, 0) >= 5
            cooling = current_index <= cooldown_until.get(stock_id, -10**9)
            if stock_id not in active and not cooling and (direct or first_crossing):
                event = {
                    "event_id": f"P5H-{stock_id}-{current_date}",
                    "stock_id": stock_id,
                    "stock_name": str(row["stock_name"]),
                    "signal_date": current_date,
                    "start_index": current_index,
                    "entry_percentile": score,
                    "entry_trigger": "DIRECT_LAYOUT_CONFIRMATION" if direct else "FIRST_TOP10_ENTRY",
                    "confirmed": bool(direct),
                    "extended": False,
                    "status": "LAYOUT_CONFIRMED" if direct else "NEW_CANDIDATE",
                    "end_signal_date": "",
                    "end_reason": "",
                }
                active[stock_id] = event
                events.append(event)
                notifications.append(
                    _notification(
                        event,
                        current_date,
                        "LAYOUT_CONFIRMED_DIRECT" if direct else "NEW_CANDIDATE",
                        score,
                        0,
                        "連續5日維持法人排名前20%，直接列為布局確認"
                        if direct
                        else "法人排名首次進入同日TPEx前10%",
                    )
                )
            previous_score[stock_id] = score
            last_seen[stock_id] = current_index

    event_frame = pd.DataFrame(events)
    if not event_frame.empty:
        event_frame = event_frame.drop(columns=["start_index"], errors="ignore")
    notification_frame = pd.DataFrame(notifications)
    if not notification_frame.empty:
        notification_frame = notification_frame.drop_duplicates(
            ["event_id", "signal_date", "notification_type"], keep="last"
        ).sort_values(["signal_date", "stock_id", "notification_type"], kind="stable")
    return {"events": event_frame, "notifications": notification_frame}


def build_notification_plan(
    *,
    notifications: pd.DataFrame,
    latest_scores: pd.DataFrame,
    user_configs: list[dict[str, Any]],
    ready_to_send: bool,
    generated_reason: str,
) -> pd.DataFrame:
    columns = [
        "user_id", "stock_id", "stock_name", "signal_date", "event_id",
        "notification_type", "notify_mode", "eligible_for_future_github",
        "ready_to_send", "trade_action", "percentile", "event_age_days",
        "reason", "positive_factors", "negative_factors", "is_configured_stock",
        "generated_at", "generated_reason", "plan_valid_until",
    ]
    if notifications.empty:
        return pd.DataFrame(columns=columns)
    latest_lookup = latest_scores.set_index("stock_id", drop=False)
    rows: list[dict[str, Any]] = []
    generated_at = datetime.now(TAIPEI).isoformat(timespec="seconds")
    for config in user_configs:
        user_id = str(config["user"]["id"])
        configured = {str(stock["symbol"]) for stock in enabled_stocks(config)}
        for _, notification in notifications.iterrows():
            stock_id = str(notification["stock_id"])
            score_row = latest_lookup.loc[stock_id] if stock_id in latest_lookup.index else None
            signal_date = str(notification["signal_date"])
            rows.append(
                {
                    "user_id": user_id,
                    "stock_id": stock_id,
                    "stock_name": str(notification["stock_name"]),
                    "signal_date": signal_date,
                    "event_id": str(notification["event_id"]),
                    "notification_type": str(notification["notification_type"]),
                    "notify_mode": _notify_mode(str(notification["notification_type"])),
                    "eligible_for_future_github": int(
                        str(notification["notification_type"])
                        in {
                            "LAYOUT_CONFIRMED_DIRECT",
                            "NEW_CANDIDATE",
                            "LAYOUT_CONFIRMED",
                            "LAYOUT_CONFIRMED_AND_EXTENDED",
                            "DAY20_EXTEND",
                            "DAY20_EXTEND_STRONG",
                        }
                    ),
                    "ready_to_send": int(ready_to_send),
                    "trade_action": "TRACK_ONLY",
                    "percentile": notification["percentile"],
                    "event_age_days": notification["event_age_days"],
                    "reason": str(notification["reason"]),
                    "positive_factors": str(score_row["positive_factors"]) if score_row is not None else "",
                    "negative_factors": str(score_row["negative_factors"]) if score_row is not None else "",
                    "is_configured_stock": int(stock_id in configured),
                    "generated_at": generated_at,
                    "generated_reason": generated_reason,
                    "plan_valid_until": (date.fromisoformat(signal_date) + timedelta(days=7)).isoformat(),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def write_manifest(state_dir: Path, payload: dict[str, Any]) -> None:
    target = state_dir / "update_manifest.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)


def _notification(
    event: dict[str, Any],
    signal_date: str,
    notification_type: str,
    percentile: float,
    age: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "event_id": event["event_id"],
        "stock_id": event["stock_id"],
        "stock_name": event["stock_name"],
        "signal_date": signal_date,
        "notification_type": notification_type,
        "status": event["status"],
        "percentile": percentile,
        "event_age_days": age,
        "reason": reason,
        "trade_action": "TRACK_ONLY",
    }


def _merge_same_day_notifications(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    confirm = next((row for row in rows if row["notification_type"] == "LAYOUT_CONFIRMED"), None)
    extend = next(
        (row for row in rows if row["notification_type"] in {"DAY20_EXTEND", "DAY20_EXTEND_STRONG"}),
        None,
    )
    if confirm is not None and extend is not None:
        merged = dict(extend)
        merged["notification_type"] = "LAYOUT_CONFIRMED_AND_EXTENDED"
        merged["reason"] = "法人布局確認，且第20日仍維持高排名，延長觀察至第40日"
        return [merged]
    return rows


def _close_event(event: dict[str, Any], signal_date: str, reason: str) -> None:
    event["status"] = "ENDED"
    event["end_signal_date"] = signal_date
    event["end_reason"] = reason


def _notify_mode(notification_type: str) -> str:
    if notification_type == "LAYOUT_CONFIRMED_DIRECT":
        return "HIGH_PRIORITY_NEW"
    if notification_type == "NEW_CANDIDATE":
        return "NEW_CANDIDATE"
    if notification_type in {
        "LAYOUT_CONFIRMED",
        "LAYOUT_CONFIRMED_AND_EXTENDED",
        "DAY20_EXTEND",
        "DAY20_EXTEND_STRONG",
    }:
        return "STATE_UPDATE"
    return "SILENT_STATE"


def _factor_strings(row: pd.Series, features: Sequence[str]) -> tuple[str, str]:
    values = [(feature, float(row[f"contribution__{feature}"])) for feature in features]
    positive = sorted((item for item in values if item[1] > 0), key=lambda item: item[1], reverse=True)[:3]
    negative = sorted((item for item in values if item[1] < 0), key=lambda item: item[1])[:3]
    return (
        "；".join(f"{FEATURE_LABELS.get(name, name)}({value:+.4f})" for name, value in positive),
        "；".join(f"{FEATURE_LABELS.get(name, name)}({value:+.4f})" for name, value in negative),
    )


def _feature_group(feature: str) -> str:
    if feature.startswith("foreign_"):
        return "foreign"
    if feature.startswith("investment_trust_"):
        return "investment_trust"
    if feature.startswith("dealer_self_"):
        return "dealer_self"
    return "consensus"


def _signed_streaks(values: np.ndarray) -> np.ndarray:
    result = np.zeros(len(values), dtype=int)
    previous = 0
    for index, value in enumerate(values):
        if value > 0:
            previous = previous + 1 if previous > 0 else 1
        elif value < 0:
            previous = previous - 1 if previous < 0 else -1
        else:
            previous = 0
        result[index] = previous
    return result


def _max_zero_streak(values: np.ndarray) -> int:
    maximum = 0
    running = 0
    for value in values:
        if value <= 0:
            running += 1
            maximum = max(maximum, running)
        else:
            running = 0
    return maximum


def _validate_source_coverage(
    raw: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    minimum_ratio: float,
    minimum_stocks: int,
) -> None:
    expected = int(universe["stock_id"].nunique())
    latest_date = str(raw["date"].max())
    latest_count = int(raw.loc[raw["date"] == latest_date, "stock_id"].nunique())
    required = max(minimum_stocks, int(math.floor(expected * minimum_ratio)))
    if latest_count < required:
        raise RuntimeError(
            f"seed 最新日只有 {latest_count} 檔，低於母體覆蓋要求 {required} 檔"
        )


def _difference_value(
    row: dict[str, Any],
    *,
    difference: Sequence[str],
    buy: Sequence[str],
    sell: Sequence[str],
) -> float:
    value = _first_value(row, difference)
    if value is not None and str(value or "").strip() not in {"", "--", "---"}:
        return _parse_number(value)
    return _number_value(row, *buy) - _number_value(row, *sell)


def _text_value(row: dict[str, Any], *keys: str) -> str:
    value = _first_value(row, keys)
    if value is not None:
        text = _clean_html_cell(value)
        if text:
            return text
    return ""


def _number_value(row: dict[str, Any], *keys: str) -> float:
    value = _first_value(row, keys)
    if value is not None and str(value or "").strip() not in {"", "--", "---", "-"}:
        return _parse_number(value)
    return 0.0


def _scaled_number_value(
    row: dict[str, Any],
    *,
    scaled_keys: Sequence[str],
    scale: float,
    normal_keys: Sequence[str],
) -> float:
    scaled = _first_value(row, scaled_keys)
    if scaled is not None and str(scaled or "").strip() not in {"", "--", "---", "-"}:
        return _parse_number(scaled) * scale
    return _number_value(row, *normal_keys)


def _first_value(row: dict[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        if key in row:
            return row.get(key)
    normalized = {_normalize_key(str(key)): value for key, value in row.items()}
    for key in keys:
        candidate = normalized.get(_normalize_key(key))
        if candidate is not None:
            return candidate
    return None


def _normalize_key(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).lower()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def _date_value(row: dict[str, Any], *keys: str) -> str:
    return _normalize_date_text(_text_value(row, *keys))


def _parse_number(value: Any) -> float:
    text = str(value or "0").replace(",", "").replace("+", "").strip()
    try:
        return float(text)
    except ValueError:
        return 0.0


def _normalize_date_text(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.date().isoformat() if isinstance(value, datetime) else value.isoformat()
    text = str(value or "").strip()
    if not text:
        return ""
    compact = re.match(r"^(\d{4})(\d{2})(\d{2})$", text)
    if compact:
        try:
            return date(*(int(part) for part in compact.groups())).isoformat()
        except ValueError:
            return ""
    roc_compact = re.match(r"^(\d{3})(\d{2})(\d{2})$", text)
    if roc_compact:
        try:
            year, month, day = (int(part) for part in roc_compact.groups())
            return date(year + 1911, month, day).isoformat()
        except ValueError:
            return ""
    separated = re.match(r"^(\d{3,4})[-/](\d{1,2})[-/](\d{1,2})", text)
    if separated:
        try:
            year, month, day = (int(part) for part in separated.groups())
            if year < 1911:
                year += 1911
            return date(year, month, day).isoformat()
        except ValueError:
            return ""
    return ""



def _safe_name(value: Any, fallback: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return str(fallback or "")
    text = str(value).strip()
    return text if text and text.lower() != "nan" else str(fallback or "")

def _parse_iso_date(value: str | None) -> date:
    if not value:
        raise ValueError("日期不可為空")
    return date.fromisoformat(value)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    temporary.replace(path)


def _write_gzip_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8-sig", newline="") as handle:
        frame.to_csv(handle, index=False)
    temporary.replace(path)


if __name__ == "__main__":
    cli()
