from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS stock_universe (
    stock_id TEXT PRIMARY KEY,
    stock_name TEXT NOT NULL,
    market_hint TEXT,
    selection_group TEXT NOT NULL,
    selection_reason TEXT NOT NULL,
    expected_cases TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS stock_info (
    stock_id TEXT PRIMARY KEY,
    stock_name TEXT,
    industry_category TEXT,
    market_type TEXT,
    info_date TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE IF NOT EXISTS official_company_info (
    stock_id TEXT NOT NULL,
    market_type TEXT NOT NULL,
    stock_name TEXT,
    listing_date TEXT,
    report_date TEXT,
    source_dataset TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (stock_id, market_type)
);

CREATE TABLE IF NOT EXISTS official_twse_delisted (
    stock_id TEXT PRIMARY KEY,
    stock_name TEXT,
    delisting_date TEXT NOT NULL,
    source_dataset TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_universe (
    stock_id TEXT PRIMARY KEY,
    stock_name TEXT,
    market_type TEXT NOT NULL,
    industry_category TEXT,
    listing_date TEXT,
    delisting_date TEXT,
    current_status TEXT NOT NULL,
    download_enabled INTEGER NOT NULL DEFAULT 1,
    training_enabled INTEGER NOT NULL DEFAULT 0,
    inclusion_status TEXT NOT NULL,
    inclusion_reason TEXT NOT NULL,
    source TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS market_calendar (
    date TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS stock_prices (
    stock_id TEXT NOT NULL,
    date TEXT NOT NULL,
    trading_volume INTEGER,
    trading_money INTEGER,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    spread REAL,
    trading_turnover INTEGER,
    PRIMARY KEY (stock_id, date)
);

CREATE TABLE IF NOT EXISTS institutional_flows (
    stock_id TEXT NOT NULL,
    date TEXT NOT NULL,
    foreign_net INTEGER NOT NULL DEFAULT 0,
    investment_trust_net INTEGER NOT NULL DEFAULT 0,
    dealer_self_net INTEGER NOT NULL DEFAULT 0,
    dealer_hedging_net INTEGER NOT NULL DEFAULT 0,
    foreign_dealer_self_net INTEGER NOT NULL DEFAULT 0,
    selected_total_net INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (stock_id, date)
);

CREATE TABLE IF NOT EXISTS corporate_actions (
    stock_id TEXT NOT NULL,
    date TEXT NOT NULL,
    action_type TEXT NOT NULL,
    before_price REAL,
    reference_price REAL,
    description TEXT,
    source_dataset TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    PRIMARY KEY (stock_id, date, action_type, source_dataset)
);

CREATE TABLE IF NOT EXISTS delistings (
    stock_id TEXT NOT NULL,
    date TEXT NOT NULL,
    stock_name TEXT,
    PRIMARY KEY (stock_id, date)
);

CREATE TABLE IF NOT EXISTS label_results (
    stock_id TEXT NOT NULL,
    signal_date TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    entry_date TEXT,
    target_date TEXT,
    entry_open REAL,
    target_close REAL,
    raw_return REAL,
    adjusted_return REAL,
    max_adjusted_return REAL,
    min_adjusted_return REAL,
    label TEXT,
    action_types TEXT,
    entry_day_action_ignored INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    error TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (stock_id, signal_date, horizon)
);

CREATE TABLE IF NOT EXISTS download_status (
    dataset TEXT NOT NULL,
    stock_id TEXT NOT NULL,
    requested_start TEXT NOT NULL,
    requested_end TEXT NOT NULL,
    status TEXT NOT NULL,
    row_count INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (dataset, stock_id)
);

CREATE TABLE IF NOT EXISTS phase2_batch_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    selected_stocks INTEGER NOT NULL DEFAULT 0,
    completed_stocks INTEGER NOT NULL DEFAULT 0,
    failed_stocks INTEGER NOT NULL DEFAULT 0,
    skipped_stocks INTEGER NOT NULL DEFAULT 0,
    quota_exhausted INTEGER NOT NULL DEFAULT 0,
    requests_made INTEGER NOT NULL DEFAULT 0,
    completed_stocks_after INTEGER NOT NULL DEFAULT 0,
    remaining_stocks_after INTEGER NOT NULL DEFAULT 0,
    remaining_requests_after INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS research_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS phase3_build_status (
    stock_id TEXT PRIMARY KEY,
    config_signature TEXT NOT NULL,
    status TEXT NOT NULL,
    shard_name TEXT,
    total_rows INTEGER NOT NULL DEFAULT 0,
    feature_ready_rows INTEGER NOT NULL DEFAULT 0,
    eligible_5d_rows INTEGER NOT NULL DEFAULT 0,
    eligible_10d_rows INTEGER NOT NULL DEFAULT 0,
    eligible_20d_rows INTEGER NOT NULL DEFAULT 0,
    liquidity_10m_rows INTEGER NOT NULL DEFAULT 0,
    liquidity_20m_rows INTEGER NOT NULL DEFAULT 0,
    liquidity_50m_rows INTEGER NOT NULL DEFAULT 0,
    liquidity_100m_rows INTEGER NOT NULL DEFAULT 0,
    first_signal_date TEXT,
    last_signal_date TEXT,
    error TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS phase3_label_distribution (
    stock_id TEXT NOT NULL,
    config_signature TEXT NOT NULL,
    signal_year INTEGER NOT NULL,
    label TEXT NOT NULL,
    sample_count INTEGER NOT NULL,
    PRIMARY KEY (stock_id, config_signature, signal_year, label)
);

CREATE TABLE IF NOT EXISTS phase3_exclusion_stats (
    stock_id TEXT NOT NULL,
    config_signature TEXT NOT NULL,
    reason TEXT NOT NULL,
    sample_count INTEGER NOT NULL,
    PRIMARY KEY (stock_id, config_signature, reason)
);

CREATE TABLE IF NOT EXISTS phase3_label_repair_status (
    stock_id TEXT NOT NULL,
    config_signature TEXT NOT NULL,
    repair_version TEXT NOT NULL,
    status TEXT NOT NULL,
    changed_rows INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (stock_id, config_signature, repair_version)
);

CREATE TABLE IF NOT EXISTS phase5c_update_status (
    target_date TEXT NOT NULL,
    stock_id TEXT NOT NULL,
    dataset TEXT NOT NULL,
    requested_start TEXT NOT NULL,
    requested_end TEXT NOT NULL,
    status TEXT NOT NULL,
    row_count INTEGER NOT NULL DEFAULT 0,
    latest_before TEXT,
    latest_after TEXT,
    error TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (target_date, stock_id, dataset)
);

CREATE INDEX IF NOT EXISTS idx_model_universe_market ON model_universe(market_type, current_status);
CREATE INDEX IF NOT EXISTS idx_official_company_listing ON official_company_info(market_type, listing_date);
CREATE INDEX IF NOT EXISTS idx_calendar_date ON market_calendar(date);
CREATE INDEX IF NOT EXISTS idx_prices_date ON stock_prices(date);
CREATE INDEX IF NOT EXISTS idx_flows_date ON institutional_flows(date);
CREATE INDEX IF NOT EXISTS idx_actions_date ON corporate_actions(date);
CREATE INDEX IF NOT EXISTS idx_labels_horizon_status ON label_results(horizon, status);
CREATE INDEX IF NOT EXISTS idx_phase3_status_signature ON phase3_build_status(config_signature, status);
CREATE INDEX IF NOT EXISTS idx_phase3_label_signature ON phase3_label_distribution(config_signature, signal_year, label);
CREATE INDEX IF NOT EXISTS idx_phase3_exclusion_signature ON phase3_exclusion_stats(config_signature, reason);
CREATE INDEX IF NOT EXISTS idx_phase3_repair_signature ON phase3_label_repair_status(config_signature, repair_version, status);
CREATE INDEX IF NOT EXISTS idx_phase5c_status_target ON phase5c_update_status(target_date, status);
"""


class ResearchDatabase:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> int:
        materialized = list(rows)
        if not materialized:
            return 0
        with self.connect() as connection:
            connection.executemany(sql, materialized)
        return len(materialized)

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        with self.connect() as connection:
            connection.execute(sql, params)

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return list(connection.execute(sql, params).fetchall())

    def scalar(self, sql: str, params: Sequence[Any] = ()) -> Any:
        rows = self.query(sql, params)
        return rows[0][0] if rows else None


    def record_phase2_batch(
        self,
        *,
        started_at: str,
        finished_at: str,
        selected_stocks: int,
        completed_stocks: int,
        failed_stocks: int,
        skipped_stocks: int,
        quota_exhausted: bool,
        requests_made: int,
        completed_stocks_after: int,
        remaining_stocks_after: int,
        remaining_requests_after: int,
    ) -> None:
        self.execute(
            """
            INSERT INTO phase2_batch_history (
                started_at, finished_at, selected_stocks, completed_stocks,
                failed_stocks, skipped_stocks, quota_exhausted, requests_made,
                completed_stocks_after, remaining_stocks_after,
                remaining_requests_after
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                started_at,
                finished_at,
                selected_stocks,
                completed_stocks,
                failed_stocks,
                skipped_stocks,
                1 if quota_exhausted else 0,
                requests_made,
                completed_stocks_after,
                remaining_stocks_after,
                remaining_requests_after,
            ),
        )

    def get_metadata(self, key: str) -> str | None:
        rows = self.query(
            "SELECT value FROM research_metadata WHERE key=?",
            (key,),
        )
        if not rows:
            return None
        return str(rows[0]["value"])

    def set_metadata(self, key: str, value: str) -> None:
        self.execute(
            """
            INSERT INTO research_metadata (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value,
                updated_at=CURRENT_TIMESTAMP
            """,
            (key, value),
        )

    def mark_download(
        self,
        *,
        dataset: str,
        stock_id: str,
        requested_start: str,
        requested_end: str,
        status: str,
        row_count: int = 0,
        error: str | None = None,
    ) -> None:
        self.execute(
            """
            INSERT INTO download_status (
                dataset, stock_id, requested_start, requested_end,
                status, row_count, error, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(dataset, stock_id) DO UPDATE SET
                requested_start=excluded.requested_start,
                requested_end=excluded.requested_end,
                status=excluded.status,
                row_count=excluded.row_count,
                error=excluded.error,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                dataset,
                stock_id,
                requested_start,
                requested_end,
                status,
                row_count,
                error,
            ),
        )

    def download_is_complete(
        self,
        *,
        dataset: str,
        stock_id: str,
        requested_start: str,
        requested_end: str,
    ) -> bool:
        rows = self.query(
            """
            SELECT requested_start, requested_end, status
            FROM download_status
            WHERE dataset=? AND stock_id=?
            """,
            (dataset, stock_id),
        )
        if not rows:
            return False
        row = rows[0]
        return (
            row["status"] == "complete"
            and row["requested_start"] <= requested_start
            and row["requested_end"] >= requested_end
        )
