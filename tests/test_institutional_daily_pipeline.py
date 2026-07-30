from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from src.institutional.daily_pipeline import (
    build_notification_plan,
    create_seed,
    load_deploy_model,
    merge_tpex_daily_rows,
    normalize_rolling_frame,
    parse_flow_row,
    parse_quote_row,
    replay_lifecycle,
    TpexOpenApiClient,
    update_daily,
    write_rolling_state,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models/tpex"


def test_deploy_model_is_self_consistent() -> None:
    model = load_deploy_model(MODEL_DIR)
    values = np.vstack([model.mean, model.mean])
    scores, contributions = model.score(values)

    assert len(model.feature_columns) == 22
    assert contributions.shape == (2, 22)
    assert np.allclose(scores, model.intercept)


def test_tpex_openapi_parsers_keep_dealer_hedging_out() -> None:
    quote = parse_quote_row(
        {
            "Date": "115/07/29",
            "SecuritiesCompanyCode": "7828",
            "CompanyName": "創新服務",
            "TradingShares": "1,000,000",
            "TransactionAmount": "50,000,000",
            "Open": "100",
            "High": "105",
            "Low": "99",
            "Close": "103",
        }
    )
    flow = parse_flow_row(
        {
            "Date": "115/07/29",
            "SecuritiesCompanyCode": "7828",
            "CompanyName": "創新服務",
            "ForeignInvestorsDifference": "100,000",
            "SecuritiesInvestmentTrustCompaniesDifference": "20,000",
            "DealersSelfDifference": "5,000",
            "DealersHedgingDifference": "999,999",
        }
    )

    assert quote["date"] == "2026-07-29"
    assert quote["trading_money"] == 50_000_000
    assert flow["selected_total_net"] == 125_000


def test_normalize_rolling_frame_accepts_sqlite_timestamps() -> None:
    frame = pd.DataFrame(
        [
            {
                "date": "2026-07-23 00:00:00",
                "stock_id": "1815",
                "stock_name": "富喬",
                "trading_volume": 1,
                "trading_money": 1,
                "open": 1,
                "high": 1,
                "low": 1,
                "close": 1,
                "foreign_net": 0,
                "investment_trust_net": 0,
                "dealer_self_net": 0,
                "selected_total_net": 0,
            }
        ]
    )

    normalized = normalize_rolling_frame(frame)

    assert normalized["date"].tolist() == ["2026-07-23"]


def test_historical_tpex_payload_is_converted_to_canonical_rows() -> None:
    quote_payload = {
        "aaData": [
            [
                "7828", "創新服務", "103", "+3", "100", "105", "99",
                "1,000,000", "50,000,000", "500", "", "", "", "",
                "100,000,000", "113", "93",
            ]
        ]
    }
    flow_values = [
        "7828", "創新服務",
        "80", "10", "70",  # foreign excluding foreign dealer
        "0", "0", "0",  # foreign dealer
        "100", "20", "80",  # total foreign used by the model
        "30", "10", "20",  # investment trust
        "8", "3", "5",  # dealer proprietary used by the model
        "999", "1", "998",  # dealer hedge must be ignored
        "1007", "4", "1003", "105", "EE",
    ]
    session = _PayloadSession([quote_payload, {"aaData": [flow_values]}])
    client = TpexOpenApiClient(session=session)

    quotes, flows = client.fetch_date(date(2026, 7, 29))
    quote = parse_quote_row(quotes[0])
    flow = parse_flow_row(flows[0])

    assert quote["date"] == "2026-07-29"
    assert quote["trading_volume"] == 1_000_000
    assert quote["trading_money"] == 50_000_000
    quote_call = session.calls[0]
    assert quote_call["params"]["d"] == "115/07/29"
    assert quote_call["params"]["se"] == "EW"
    assert flow["foreign_net"] == 80
    assert flow["investment_trust_net"] == 20
    assert flow["dealer_self_net"] == 5
    assert flow["selected_total_net"] == 105


def test_merge_tpex_rows_filters_to_seed_universe() -> None:
    universe = pd.DataFrame(
        [
            {"stock_id": "1001", "stock_name": "甲", "listing_date": ""},
            {"stock_id": "1002", "stock_name": "乙", "listing_date": ""},
        ]
    )
    quotes = [
        _quote_row("1001", "甲", "20260729"),
        _quote_row("9999", "非母體", "20260729"),
    ]
    flows = [
        _flow_row("1001", "甲", "20260729", 10),
        _flow_row("9999", "非母體", "20260729", 20),
    ]

    result = merge_tpex_daily_rows(quotes, flows, universe)

    assert result["stock_id"].tolist() == ["1001"]
    assert result.iloc[0]["selected_total_net"] == 130


def test_lifecycle_merges_confirmation_and_day20_extension() -> None:
    start = date(2026, 1, 1)
    rows = []
    for index in range(25):
        signal_date = (start + timedelta(days=index)).isoformat()
        # C starts as a candidate on day 0, drops below 80, then reaches five
        # consecutive >=80 observations exactly on event day 20.
        c_score = 95 if index == 0 else 70 if index < 16 else 85
        for stock_id, score in (("C", c_score), ("F", 50 + index * 0.1)):
            rows.append(
                {
                    "stock_id": stock_id,
                    "stock_name": stock_id,
                    "signal_date": signal_date,
                    "return_rank_score": score,
                }
            )
    result = replay_lifecycle(pd.DataFrame(rows))
    notifications = result["notifications"]
    day20 = notifications[
        (notifications["stock_id"] == "C")
        & (notifications["event_age_days"] == 20)
    ]

    assert day20["notification_type"].tolist() == [
        "LAYOUT_CONFIRMED_AND_EXTENDED"
    ]


def test_notification_plan_marks_end_states_as_silent() -> None:
    notifications = pd.DataFrame(
        [
            {
                "event_id": "P5H-1001-2026-07-29",
                "stock_id": "1001",
                "stock_name": "甲",
                "signal_date": "2026-07-29",
                "notification_type": "NEW_CANDIDATE",
                "percentile": 95.0,
                "event_age_days": 0,
                "reason": "new",
            },
            {
                "event_id": "P5H-1002-2026-06-01",
                "stock_id": "1002",
                "stock_name": "乙",
                "signal_date": "2026-07-29",
                "notification_type": "DAY40_END",
                "percentile": 70.0,
                "event_age_days": 40,
                "reason": "end",
            },
        ]
    )
    latest_scores = pd.DataFrame(
        [
            {
                "stock_id": "1001",
                "positive_factors": "正向",
                "negative_factors": "負向",
            },
            {
                "stock_id": "1002",
                "positive_factors": "",
                "negative_factors": "",
            },
        ]
    )
    configs = [
        {
            "user": {"id": "u1"},
            "stocks": [
                {"symbol": "1001", "average_cost": 10, "profit_alerts": [30]}
            ],
        }
    ]

    plan = build_notification_plan(
        notifications=notifications,
        latest_scores=latest_scores,
        user_configs=configs,
        ready_to_send=True,
        generated_reason="test",
    )

    assert plan.loc[plan["notification_type"] == "NEW_CANDIDATE", "eligible_for_future_github"].item() == 1
    assert plan.loc[plan["notification_type"] == "DAY40_END", "eligible_for_future_github"].item() == 0
    assert plan.loc[plan["stock_id"] == "1001", "is_configured_stock"].item() == 1


def test_daily_update_is_idempotent_for_same_market_date(tmp_path: Path) -> None:
    state_dir = tmp_path / "institutional"
    universe = pd.DataFrame(
        [
            {"stock_id": f"{1000 + index}", "stock_name": f"S{index}", "listing_date": "2020-01-01"}
            for index in range(60)
        ]
    )
    rolling = _rolling_frame(universe, start=date(2026, 4, 1), market_days=70)
    write_rolling_state(state_dir, rolling, universe)
    users = tmp_path / "users"
    users.mkdir()
    (users / "u.yaml").write_text(
        """
user:
  id: u
  enabled: true
  discord_webhook_key: default
stocks:
  - symbol: "1000"
    average_cost: 10
    profit_alerts: [30]
""".strip(),
        encoding="utf-8",
    )
    latest_date = (date(2026, 4, 1) + timedelta(days=70)).isoformat()
    client = _FakeClient(universe, latest_date)

    first = update_daily(
        state_dir=state_dir,
        model_dir=MODEL_DIR,
        users_config=users,
        as_of_date=latest_date,
        client=client,
    )
    second = update_daily(
        state_dir=state_dir,
        model_dir=MODEL_DIR,
        users_config=users,
        as_of_date=latest_date,
        client=client,
    )
    manifest = json.loads((state_dir / "update_manifest.json").read_text(encoding="utf-8"))

    assert first.status == "UPDATED"
    assert second.status == "ALREADY_CURRENT"
    assert manifest["ready_to_send"] == 0


def test_daily_update_catches_up_missing_market_dates(tmp_path: Path) -> None:
    state_dir = tmp_path / "institutional"
    universe = pd.DataFrame(
        [
            {"stock_id": f"{1000 + index}", "stock_name": f"S{index}", "listing_date": "2020-01-01"}
            for index in range(60)
        ]
    )
    rolling = _rolling_frame(universe, start=date(2026, 4, 1), market_days=70)
    write_rolling_state(state_dir, rolling, universe)
    users = tmp_path / "users"
    users.mkdir()
    (users / "u.yaml").write_text(
        """
user:
  id: u
  enabled: true
stocks:
  - symbol: "1000"
    average_cost: 10
    profit_alerts: [30]
""".strip(),
        encoding="utf-8",
    )
    previous_latest = date.fromisoformat(str(rolling["date"].max()))
    intermediate = previous_latest + timedelta(days=1)
    latest = previous_latest + timedelta(days=3)
    client = _CatchupClient(universe, latest=latest, trading_dates={intermediate})

    result = update_daily(
        state_dir=state_dir,
        model_dir=MODEL_DIR,
        users_config=users,
        as_of_date=latest.isoformat(),
        client=client,
    )
    updated = pd.read_csv(state_dir / "rolling_market_data.csv.gz", dtype={"date": "string"})
    manifest = json.loads((state_dir / "update_manifest.json").read_text(encoding="utf-8"))

    assert result.status == "UPDATED"
    assert intermediate.isoformat() in set(updated["date"].astype(str))
    assert latest.isoformat() in set(updated["date"].astype(str))
    assert manifest["catchup_market_dates"] == [
        intermediate.isoformat(),
        latest.isoformat(),
    ]


def test_create_seed_reads_existing_research_sqlite(tmp_path: Path) -> None:
    database = tmp_path / "research.sqlite"
    universe = pd.DataFrame(
        [
            {"stock_id": f"{1000 + index}", "stock_name": f"S{index}", "listing_date": "2020-01-01"}
            for index in range(60)
        ]
    )
    rolling = _rolling_frame(universe, start=date(2026, 1, 1), market_days=65)
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE model_universe (
                stock_id TEXT, stock_name TEXT, market_type TEXT, listing_date TEXT,
                current_status TEXT, training_enabled INTEGER
            );
            CREATE TABLE market_calendar (date TEXT);
            CREATE TABLE stock_prices (
                stock_id TEXT, date TEXT, trading_volume REAL, trading_money REAL,
                open REAL, high REAL, low REAL, close REAL
            );
            CREATE TABLE institutional_flows (
                stock_id TEXT, date TEXT, foreign_net REAL, investment_trust_net REAL,
                dealer_self_net REAL, selected_total_net REAL
            );
            """
        )
        universe_rows = [
            (row.stock_id, row.stock_name, "tpex", row.listing_date, "active", 1)
            for row in universe.itertuples(index=False)
        ]
        connection.executemany(
            "INSERT INTO model_universe VALUES (?, ?, ?, ?, ?, ?)", universe_rows
        )
        connection.executemany(
            "INSERT INTO market_calendar VALUES (?)",
            [(f"{value} 00:00:00",) for value in sorted(rolling["date"].unique())],
        )
        connection.executemany(
            "INSERT INTO stock_prices VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row.stock_id, f"{row.date} 00:00:00", row.trading_volume, row.trading_money,
                    row.open, row.high, row.low, row.close,
                )
                for row in rolling.itertuples(index=False)
            ],
        )
        connection.executemany(
            "INSERT INTO institutional_flows VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    row.stock_id, f"{row.date} 00:00:00", row.foreign_net,
                    row.investment_trust_net, row.dealer_self_net,
                    row.selected_total_net,
                )
                for row in rolling.itertuples(index=False)
            ],
        )
        partial_date = (
            date.fromisoformat(str(rolling["date"].max())) + timedelta(days=1)
        ).isoformat()
        partial = rolling[rolling["date"] == rolling["date"].max()].head(10)
        connection.executemany(
            "INSERT INTO stock_prices VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row.stock_id, f"{partial_date} 00:00:00", row.trading_volume,
                    row.trading_money, row.open, row.high, row.low, row.close,
                )
                for row in partial.itertuples(index=False)
            ],
        )
        connection.executemany(
            "INSERT INTO institutional_flows VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    row.stock_id, f"{partial_date} 00:00:00", row.foreign_net,
                    row.investment_trust_net, row.dealer_self_net,
                    row.selected_total_net,
                )
                for row in partial.itertuples(index=False)
            ],
        )
    users = tmp_path / "users"
    users.mkdir()
    (users / "u.yaml").write_text(
        """
user:
  id: u
  enabled: true
stocks:
  - symbol: "1000"
    average_cost: 10
    profit_alerts: [30]
""".strip(),
        encoding="utf-8",
    )

    result = create_seed(
        database_path=database,
        state_dir=tmp_path / "state",
        model_dir=MODEL_DIR,
        users_config=users,
        lookback_market_days=60,
    )

    assert result.status == "SEEDED"
    assert result.eligible_stocks == 60
    rolling_state = pd.read_csv(
        tmp_path / "state/rolling_market_data.csv.gz",
        dtype={"date": "string"},
    )
    assert str(rolling_state["date"].max()) == str(rolling["date"].max())



class _FakeClient:
    quote_endpoint = "fixture://quotes"
    flow_endpoint = "fixture://flows"

    def __init__(self, universe: pd.DataFrame, signal_date: str) -> None:
        self.universe = universe
        self.signal_date = signal_date.replace("-", "")

    def fetch_latest(self):
        quotes = []
        flows = []
        for index, row in self.universe.reset_index(drop=True).iterrows():
            stock_id = str(row["stock_id"])
            name = str(row["stock_name"])
            quotes.append(_quote_row(stock_id, name, self.signal_date, index=index))
            flows.append(_flow_row(stock_id, name, self.signal_date, index + 1))
        return quotes, flows

    def fetch_date(self, trade_date: date):
        if trade_date.strftime("%Y%m%d") != self.signal_date:
            return [], []
        return self.fetch_latest()


class _CatchupClient(_FakeClient):
    def __init__(
        self,
        universe: pd.DataFrame,
        *,
        latest: date,
        trading_dates: set[date],
    ) -> None:
        super().__init__(universe, latest.isoformat())
        self.latest = latest
        self.trading_dates = trading_dates

    def fetch_date(self, trade_date: date):
        if trade_date == self.latest:
            return super().fetch_date(trade_date)
        if trade_date not in self.trading_dates:
            return [], []
        compact = trade_date.strftime("%Y%m%d")
        quotes = []
        flows = []
        for index, row in self.universe.reset_index(drop=True).iterrows():
            stock_id = str(row["stock_id"])
            name = str(row["stock_name"])
            quotes.append(_quote_row(stock_id, name, compact, index=index))
            flows.append(_flow_row(stock_id, name, compact, index + 1))
        return quotes, flows


class _PayloadResponse:
    def __init__(self, payload) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self.payload


class _PayloadSession:
    def __init__(self, payloads: list[object]) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict[str, object]] = []

    def get(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return _PayloadResponse(self.payloads.pop(0))


def _quote_row(stock_id: str, name: str, signal_date: str, index: int = 0):
    return {
        "Date": signal_date,
        "SecuritiesCompanyCode": stock_id,
        "CompanyName": name,
        "TradingShares": str(1_000_000 + index * 1000),
        "TransactionAmount": str(30_000_000 + index * 100_000),
        "Open": "10",
        "High": "11",
        "Low": "9",
        "Close": "10.5",
    }


def _flow_row(stock_id: str, name: str, signal_date: str, value: int):
    return {
        "Date": signal_date,
        "SecuritiesCompanyCode": stock_id,
        "CompanyName": name,
        "ForeignInvestorsDifference": str(value * 10),
        "SecuritiesInvestmentTrustCompaniesDifference": str(value * 2),
        "DealersSelfDifference": str(value),
    }


def _rolling_frame(universe: pd.DataFrame, *, start: date, market_days: int) -> pd.DataFrame:
    rows = []
    for day in range(market_days):
        signal_date = (start + timedelta(days=day)).isoformat()
        for index, stock in universe.reset_index(drop=True).iterrows():
            flow = (index + 1) * ((day % 7) - 3)
            rows.append(
                {
                    "date": signal_date,
                    "stock_id": str(stock["stock_id"]),
                    "stock_name": str(stock["stock_name"]),
                    "trading_volume": 1_000_000 + index * 1000,
                    "trading_money": 30_000_000 + index * 100_000,
                    "open": 10,
                    "high": 11,
                    "low": 9,
                    "close": 10.5,
                    "foreign_net": flow * 10,
                    "investment_trust_net": flow * 2,
                    "dealer_self_net": flow,
                    "selected_total_net": flow * 13,
                }
            )
    return normalize_rolling_frame(pd.DataFrame(rows))
