from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import logging
import math
from pathlib import Path
import sqlite3
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from src.config_loader import load_user_configs
from src.institutional.daily_pipeline import (
    ESTIMATED_COST_WINDOW_DAYS,
    PipelineResult,
    _clean_html_cell,
    _normalize_date_text,
    _parse_iso_date,
    _parse_number,
    _table_rows,
    _validate_source_coverage,
    _write_csv,
    _write_gzip_csv,
    build_estimated_cost_reference,
    build_feature_history,
    build_notification_plan,
    load_deploy_model,
    normalize_rolling_frame,
    parse_quote_row,
    score_feature_history,
    write_manifest,
    write_rolling_state,
)


LOGGER = logging.getLogger("tw-stock-monitor.institutional.twse.daily")
TAIPEI = ZoneInfo("Asia/Taipei")
DEFAULT_STATE_DIR = Path("runtime/institutional_twse")
DEFAULT_MODEL_DIR = Path("models/twse")
DEFAULT_DATABASE = Path("research/data/institutional_phase1.sqlite")
QUOTE_ENDPOINT = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
FLOW_ENDPOINT = "https://www.twse.com.tw/rwd/zh/fund/T86"
MINIMUM_TRADING_MONEY = 100_000_000.0
MINIMUM_TRADING_VOLUME_LOTS = 300.0
ENTRY_PERCENTILE = 90.0
CONFIRMATION_DAYS = 3
TRACKING_DAYS = 40
COOLDOWN_DAYS = 20


class TwseOfficialClient:
    def __init__(
        self,
        *,
        quote_endpoint: str = QUOTE_ENDPOINT,
        flow_endpoint: str = FLOW_ENDPOINT,
        timeout_seconds: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self.quote_endpoint = quote_endpoint
        self.flow_endpoint = flow_endpoint
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()

    def fetch_date(self, trade_date: date) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        common = {"date": trade_date.strftime("%Y%m%d"), "response": "json"}
        quote_payload = self._get(
            self.quote_endpoint,
            params={**common, "type": "ALLBUT0999"},
        )
        flow_payload = self._get(
            self.flow_endpoint,
            params={**common, "selectType": "ALLBUT0999"},
        )
        quotes = _extract_twse_quote_rows(quote_payload, trade_date)
        flows = _extract_twse_flow_rows(flow_payload, trade_date)
        return quotes, flows

    def _get(self, endpoint: str, *, params: dict[str, str]) -> Any:
        response = self.session.get(
            endpoint,
            params=params,
            headers={"User-Agent": "Mozilla/5.0 tw-stock-monitor/phase6d"},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            status = str(payload.get("stat") or payload.get("status") or "").strip()
            if status and status.upper() not in {"OK", "SUCCESS"}:
                return {}
        return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TWSE institutional daily deployment pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    seed = subparsers.add_parser("seed", help="Build TWSE deployment seed from local SQLite")
    seed.add_argument("--db", type=Path, default=DEFAULT_DATABASE)
    seed.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    seed.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    seed.add_argument("--users-config", default="configs/users")
    seed.add_argument("--lookback-market-days", type=int, default=120)
    seed.add_argument("--as-of-date")
    update = subparsers.add_parser("update", help="Update TWSE daily data and notification plan")
    update.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    update.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    update.add_argument("--users-config", default="configs/users")
    update.add_argument("--lookback-market-days", type=int, default=120)
    update.add_argument("--minimum-daily-stocks", type=int, default=50)
    update.add_argument("--minimum-source-coverage-ratio", type=float, default=0.85)
    update.add_argument("--maximum-catchup-calendar-days", type=int, default=31)
    update.add_argument("--quote-endpoint", default=QUOTE_ENDPOINT)
    update.add_argument("--flow-endpoint", default=FLOW_ENDPOINT)
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
                client=TwseOfficialClient(
                    quote_endpoint=args.quote_endpoint,
                    flow_endpoint=args.flow_endpoint,
                ),
            )
    except Exception:
        LOGGER.exception("Phase 6D TWSE deployment failed")
        raise SystemExit(1) from None
    LOGGER.info(
        "Phase 6D TWSE %s: signal_date=%s, source_rows=%s, eligible=%s, notifications=%s",
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
    lookback_market_days: int = 120,
    as_of_date: str | None = None,
) -> PipelineResult:
    if lookback_market_days < 60:
        raise ValueError("TWSE seed requires at least 60 market days")
    db_path = Path(database_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Historical SQLite not found: {db_path}")
    state = Path(state_dir)
    state.mkdir(parents=True, exist_ok=True)
    model = load_deploy_model(model_dir)
    resolved_as_of = _parse_iso_date(as_of_date) if as_of_date else None
    universe, raw = _load_seed_data(
        db_path,
        lookback_market_days=lookback_market_days,
        as_of_date=resolved_as_of,
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
        generated_reason="LOCAL_TWSE_SEED",
    )
    write_manifest(
        state,
        {
            "phase6d_version": "phase6d-v1",
            "status": "SEEDED",
            "market": "twse",
            "signal_date": outputs["signal_date"],
            "model_signature": model.signature,
            "rolling_rows": len(raw),
            "universe_stocks": int(universe["stock_id"].nunique()),
            "eligible_stocks": outputs["eligible_stocks"],
            "notification_rows": outputs["notification_rows"],
            "ready_to_send": 0,
            "generated_at": datetime.now(TAIPEI).isoformat(timespec="seconds"),
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
    lookback_market_days: int = 120,
    minimum_daily_stocks: int = 50,
    minimum_source_coverage_ratio: float = 0.85,
    maximum_catchup_calendar_days: int = 31,
    as_of_date: str | None = None,
    client: TwseOfficialClient | Any | None = None,
) -> PipelineResult:
    if lookback_market_days < 60:
        raise ValueError("TWSE rolling data must keep at least 60 market days")
    if not 0.5 <= minimum_source_coverage_ratio <= 1.0:
        raise ValueError("minimum_source_coverage_ratio must be between 0.5 and 1.0")
    state = Path(state_dir)
    rolling_path = state / "rolling_market_data.csv.gz"
    universe_path = state / "universe.csv"
    if not rolling_path.exists() or not universe_path.exists():
        raise FileNotFoundError("TWSE seed not found; publish runtime/institutional_twse first")
    model = load_deploy_model(model_dir)
    rolling = pd.read_csv(
        rolling_path,
        compression="gzip",
        dtype={"date": "string", "stock_id": "string"},
        low_memory=False,
    )
    universe = pd.read_csv(universe_path, dtype={"stock_id": "string"}).fillna("")
    effective_today = _parse_iso_date(as_of_date) if as_of_date else datetime.now(TAIPEI).date()
    previous_dates = sorted(rolling["date"].astype(str).unique())
    previous_latest = previous_dates[-1] if previous_dates else ""
    if not previous_latest:
        raise RuntimeError("TWSE seed has no market date")
    previous_day = date.fromisoformat(previous_latest)
    gap_days = (effective_today - previous_day).days
    if gap_days < 0:
        raise RuntimeError(f"TWSE seed date {previous_latest} is after {effective_today}")
    if gap_days > maximum_catchup_calendar_days:
        raise RuntimeError(
            f"TWSE seed is {gap_days} calendar days behind; rebuild local seed first"
        )
    prior_counts = rolling.groupby("date")["stock_id"].nunique()
    prior_reference = float(prior_counts.tail(20).median()) if len(prior_counts) else 0.0
    required = max(
        minimum_daily_stocks,
        int(math.floor(prior_reference * minimum_source_coverage_ratio)),
    )
    api = client or TwseOfficialClient()
    appended: list[pd.DataFrame] = []
    fetched_dates: list[str] = []
    cursor = previous_day + timedelta(days=1)
    while cursor <= effective_today:
        quotes, flows = api.fetch_date(cursor)
        if quotes and flows:
            daily = merge_twse_daily_rows(quotes, flows, universe)
            observed_dates = sorted(daily["date"].astype(str).unique())
            if observed_dates != [cursor.isoformat()]:
                raise RuntimeError(f"TWSE query {cursor} returned dates {observed_dates}")
            count = int(daily["stock_id"].nunique())
            if count < required:
                raise RuntimeError(f"TWSE {cursor} common rows {count} below required {required}")
            appended.append(daily)
            fetched_dates.append(cursor.isoformat())
        cursor += timedelta(days=1)
    signal_date = fetched_dates[-1] if fetched_dates else previous_latest
    if (effective_today - date.fromisoformat(signal_date)).days > 7:
        raise RuntimeError(f"TWSE available data is stale: {signal_date}")
    is_new_market_date = bool(fetched_dates)
    combined = normalize_rolling_frame(pd.concat([rolling, *appended], ignore_index=True))
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
        generated_reason="TWSE_OFFICIAL_DAILY_REPORT",
    )
    status = "UPDATED" if is_new_market_date else "ALREADY_CURRENT"
    write_manifest(
        state,
        {
            "phase6d_version": "phase6d-v1",
            "status": status,
            "market": "twse",
            "signal_date": signal_date,
            "previous_signal_date": previous_latest,
            "source": "twse_official_daily_report",
            "quote_endpoint": getattr(api, "quote_endpoint", QUOTE_ENDPOINT),
            "flow_endpoint": getattr(api, "flow_endpoint", FLOW_ENDPOINT),
            "model_signature": model.signature,
            "rolling_rows": len(combined),
            "universe_stocks": int(universe["stock_id"].nunique()),
            "required_source_stocks": required,
            "catchup_market_dates": fetched_dates,
            "eligible_stocks": outputs["eligible_stocks"],
            "notification_rows": outputs["notification_rows"],
            "ready_to_send": int(is_new_market_date),
            "generated_at": datetime.now(TAIPEI).isoformat(timespec="seconds"),
        },
    )
    return PipelineResult(
        status=status,
        signal_date=signal_date,
        source_rows=sum(len(frame) for frame in appended),
        eligible_stocks=outputs["eligible_stocks"],
        notification_rows=outputs["notification_rows"],
        state_dir=str(state),
    )


def rebuild_outputs(
    *,
    rolling: pd.DataFrame,
    universe: pd.DataFrame,
    model: Any,
    user_configs: list[dict[str, Any]],
    state_dir: Path,
    minimum_daily_stocks: int,
    ready_to_send: bool,
    generated_reason: str,
) -> dict[str, Any]:
    features = build_feature_history(rolling, universe, model.feature_columns)
    scores = score_feature_history(
        features,
        model,
        minimum_daily_stocks=minimum_daily_stocks,
        minimum_trading_money=MINIMUM_TRADING_MONEY,
        minimum_trading_volume_lots=MINIMUM_TRADING_VOLUME_LOTS,
    )
    if scores.empty:
        raise RuntimeError("No TWSE stocks passed the Phase 6D scoring universe")
    lifecycle = replay_twse_lifecycle(scores)
    latest_date = str(scores["signal_date"].max())
    latest_scores = scores[scores["signal_date"] == latest_date].copy()
    latest_notifications = lifecycle["notifications"]
    if not latest_notifications.empty:
        latest_notifications = latest_notifications[
            latest_notifications["signal_date"] == latest_date
        ].copy()
    cost_reference = build_estimated_cost_reference(
        rolling,
        window=ESTIMATED_COST_WINDOW_DAYS,
    )
    plan = build_notification_plan(
        notifications=latest_notifications,
        latest_scores=latest_scores,
        cost_reference=cost_reference,
        user_configs=user_configs,
        ready_to_send=ready_to_send,
        generated_reason=generated_reason,
        market="twse",
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


def replay_twse_lifecycle(scores: pd.DataFrame) -> dict[str, pd.DataFrame]:
    dates = sorted(scores["signal_date"].astype(str).unique())
    date_index = {value: index for index, value in enumerate(dates)}
    day_map = {
        value: group.set_index("stock_id", drop=False)
        for value, group in scores.groupby("signal_date", sort=True)
    }
    active: dict[str, dict[str, Any]] = {}
    cooldown_until: dict[str, int] = {}
    run_length: dict[str, int] = {}
    last_seen: dict[str, int] = {}
    events: list[dict[str, Any]] = []
    notifications: list[dict[str, Any]] = []
    for current_date in dates:
        current_index = date_index[current_date]
        day = day_map[current_date]
        stocks = set(day.index.astype(str))
        for stock_id in stocks:
            score = float(day.loc[stock_id, "return_rank_score"])
            consecutive = last_seen.get(stock_id) == current_index - 1
            if score >= ENTRY_PERCENTILE:
                run_length[stock_id] = run_length.get(stock_id, 0) + 1 if consecutive else 1
            else:
                run_length[stock_id] = 0
        for stock_id in list(active):
            event = active[stock_id]
            age = current_index - int(event["start_index"])
            if age >= TRACKING_DAYS:
                row = day.loc[stock_id] if stock_id in day.index else None
                percentile = float(row["return_rank_score"]) if row is not None else float("nan")
                notifications.append(
                    _twse_notification(
                        event,
                        current_date,
                        "TWSE_DAY40_END",
                        percentile,
                        age,
                        "上市法人40日追蹤期結束；不代表賣出",
                    )
                )
                event["status"] = "ENDED"
                event["end_signal_date"] = current_date
                event["end_reason"] = "TWSE_DAY40_END"
                cooldown_until[stock_id] = current_index + COOLDOWN_DAYS
                del active[stock_id]
        for stock_id in sorted(stocks):
            if stock_id in active:
                last_seen[stock_id] = current_index
                continue
            if current_index <= cooldown_until.get(stock_id, -10**9):
                last_seen[stock_id] = current_index
                continue
            if run_length.get(stock_id, 0) == CONFIRMATION_DAYS:
                row = day.loc[stock_id]
                event = {
                    "event_id": f"P6D-TWSE-{stock_id}-{current_date}",
                    "stock_id": stock_id,
                    "stock_name": str(row["stock_name"]),
                    "signal_date": current_date,
                    "start_index": current_index,
                    "entry_percentile": float(row["return_rank_score"]),
                    "entry_trigger": "TOP10_CONFIRM_3D",
                    "status": "TWSE_TRACK_CONFIRMED",
                    "end_signal_date": "",
                    "end_reason": "",
                }
                active[stock_id] = event
                events.append(event)
                notifications.append(
                    _twse_notification(
                        event,
                        current_date,
                        "TWSE_TRACK_CONFIRMED",
                        float(row["return_rank_score"]),
                        0,
                        "上市法人排名連續3個交易日位於同日TWSE前10%，建立40日追蹤",
                    )
                )
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


def merge_twse_daily_rows(
    quote_rows: Sequence[dict[str, Any]],
    flow_rows: Sequence[dict[str, Any]],
    universe: pd.DataFrame,
) -> pd.DataFrame:
    quotes = pd.DataFrame([parse_quote_row(row) for row in quote_rows])
    flows = pd.DataFrame([_parse_twse_flow_row(row) for row in flow_rows])
    if quotes.empty or flows.empty:
        raise RuntimeError("TWSE quote or institutional report is empty")
    quotes = quotes[quotes["stock_id"].astype(str).str.fullmatch(r"\d{4}")].copy()
    flows = flows[flows["stock_id"].astype(str).str.fullmatch(r"\d{4}")].copy()
    allowed = set(universe["stock_id"].astype(str))
    quotes = quotes[quotes["stock_id"].isin(allowed)]
    flows = flows[flows["stock_id"].isin(allowed)]
    if quotes.duplicated(["date", "stock_id"]).any():
        raise RuntimeError("TWSE quote report has duplicate stock/date rows")
    if flows.duplicated(["date", "stock_id"]).any():
        raise RuntimeError("TWSE institutional report has duplicate stock/date rows")
    merged = quotes.merge(
        flows.drop(columns=["stock_name"], errors="ignore"),
        on=["date", "stock_id"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise RuntimeError("TWSE quote and institutional reports have no common stocks")
    return normalize_rolling_frame(merged)


def _load_seed_data(
    database_path: Path,
    *,
    lookback_market_days: int,
    as_of_date: date | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(database_path) as connection:
        universe = pd.read_sql_query(
            """
            SELECT stock_id, COALESCE(stock_name, '') AS stock_name,
                   COALESCE(listing_date, '') AS listing_date
            FROM model_universe
            WHERE LOWER(market_type)='twse'
              AND current_status='active'
              AND training_enabled=1
            ORDER BY stock_id
            """,
            connection,
            dtype={"stock_id": "string"},
        )
        if universe.empty:
            raise RuntimeError("Historical SQLite has no active TWSE model universe")
        normalized_date = (
            "CASE WHEN LENGTH(TRIM(p.date)) >= 10 "
            "THEN SUBSTR(REPLACE(TRIM(p.date), '/', '-'), 1, 10) "
            "ELSE TRIM(p.date) END"
        )
        expected = int(universe["stock_id"].nunique())
        required = max(50, int(math.floor(expected * 0.70)))
        sql = (
            f"SELECT {normalized_date} AS date, COUNT(DISTINCT p.stock_id) AS stock_count "
            "FROM stock_prices p JOIN model_universe u ON u.stock_id=p.stock_id "
            "WHERE LOWER(u.market_type)='twse' AND u.current_status='active' "
            "AND u.training_enabled=1"
        )
        params: list[Any] = []
        if as_of_date is not None:
            sql += f" AND {normalized_date} <= ?"
            params.append(as_of_date.isoformat())
        sql += f" GROUP BY {normalized_date} HAVING COUNT(DISTINCT p.stock_id) >= ? ORDER BY date DESC LIMIT ?"
        params.extend([required, lookback_market_days])
        date_rows = connection.execute(sql, params).fetchall()
        recent_dates = sorted(_normalize_date_text(row[0]) for row in date_rows if _normalize_date_text(row[0]))
        if len(recent_dates) < 60:
            raise RuntimeError(f"Historical SQLite has only {len(recent_dates)} complete TWSE market days")
        placeholders = ",".join("?" for _ in recent_dates)
        raw = pd.read_sql_query(
            f"""
            SELECT {normalized_date} AS date,
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
             AND (CASE WHEN LENGTH(TRIM(f.date)) >= 10
                       THEN SUBSTR(REPLACE(TRIM(f.date), '/', '-'), 1, 10)
                       ELSE TRIM(f.date) END) = {normalized_date}
            WHERE LOWER(u.market_type)='twse'
              AND u.current_status='active'
              AND u.training_enabled=1
              AND {normalized_date} IN ({placeholders})
            ORDER BY date, p.stock_id
            """,
            connection,
            params=recent_dates,
            dtype={"stock_id": "string", "date": "string"},
        )
    return universe, raw


def _extract_twse_quote_rows(payload: Any, trade_date: date) -> list[dict[str, Any]]:
    candidates = _payload_tables(payload)
    rows: list[dict[str, Any]] = []
    for fields, data in candidates:
        names = [_clean_html_cell(value) for value in fields]
        normalized = "|".join(names)
        if "證券代號" in normalized and "收盤價" in normalized and "成交股數" in normalized:
            rows = _table_rows(fields, data)
            break
    for row in rows:
        row.setdefault("Date", trade_date.isoformat())
    return rows


def _extract_twse_flow_rows(payload: Any, trade_date: date) -> list[dict[str, Any]]:
    candidates = _payload_tables(payload)
    rows: list[dict[str, Any]] = []
    for fields, data in candidates:
        names = [_clean_html_cell(value) for value in fields]
        normalized = "|".join(names)
        if "證券代號" in normalized and "投信" in normalized and "自行買賣" in normalized:
            rows = _table_rows(fields, data)
            break
    for row in rows:
        row.setdefault("Date", trade_date.isoformat())
    return rows


def _payload_tables(payload: Any) -> list[tuple[list[Any], list[Any]]]:
    tables: list[tuple[list[Any], list[Any]]] = []
    if isinstance(payload, dict):
        raw_tables = payload.get("tables")
        if isinstance(raw_tables, list):
            for table in raw_tables:
                if isinstance(table, dict):
                    fields = table.get("fields") or table.get("columns") or []
                    data = table.get("data") or table.get("aaData") or []
                    if isinstance(fields, list) and isinstance(data, list):
                        tables.append((fields, data))
        fields = payload.get("fields") or payload.get("columns") or []
        data = payload.get("data") or payload.get("aaData") or []
        if isinstance(fields, list) and isinstance(data, list):
            tables.append((fields, data))
        # Legacy MI_INDEX JSON uses fields1/data1 ... fields9/data9.
        for key, numbered_fields in payload.items():
            if not str(key).startswith("fields") or str(key) == "fields":
                continue
            suffix = str(key)[len("fields") :]
            numbered_data = payload.get(f"data{suffix}")
            if isinstance(numbered_fields, list) and isinstance(numbered_data, list):
                tables.append((numbered_fields, numbered_data))
    return tables


def _parse_twse_flow_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {_normalize_field_name(key): value for key, value in row.items()}
    foreign = _find_foreign_net(normalized)
    trust = _find_number(normalized, include=("投信", "買賣超"), exclude=("總計",))
    dealer_self = _find_number(
        normalized,
        include=("自營商自行買賣", "買賣超"),
        exclude=("避險", "總計"),
    )
    stock_id = _find_text(normalized, ("證券代號", "代號"))
    stock_name = _find_text(normalized, ("證券名稱", "名稱"))
    signal_date = _normalize_date_text(_find_text(normalized, ("資料日期", "日期", "date")))
    return {
        "date": signal_date,
        "stock_id": stock_id,
        "stock_name": stock_name,
        "foreign_net": foreign,
        "investment_trust_net": trust,
        "dealer_self_net": dealer_self,
        "selected_total_net": foreign + trust + dealer_self,
    }


def _normalize_field_name(value: Any) -> str:
    text = _clean_html_cell(value).replace("（", "(").replace("）", ")")
    return "".join(text.split()).lower()


def _find_foreign_net(row: dict[str, Any]) -> float:
    for key, value in row.items():
        if "外陸資" not in key or "買賣超" not in key or "總計" in key:
            continue
        if "外資自營商" in key and "不含外資自營商" not in key:
            continue
        return _parse_number(value)
    return 0.0


def _find_number(
    row: dict[str, Any],
    *,
    include: tuple[str, ...],
    exclude: tuple[str, ...] = (),
) -> float:
    for key, value in row.items():
        if all(part.lower() in key for part in include) and not any(part.lower() in key for part in exclude):
            return _parse_number(value)
    return 0.0


def _find_text(row: dict[str, Any], candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        normalized = _normalize_field_name(candidate)
        for key, value in row.items():
            if key == normalized or normalized in key:
                text = _clean_html_cell(value)
                if text:
                    return text
    return ""


def _twse_notification(
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


if __name__ == "__main__":
    cli()
