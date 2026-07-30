from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from research.institutional_model.corporate_actions import (
    cleanup_duplicate_corporate_actions,
    normalize_capital_reduction_rows,
    normalize_dividend_rows,
    normalize_par_value_rows,
    normalize_split_rows,
    replace_corporate_actions,
)
from research.institutional_model.database import ResearchDatabase
from research.institutional_model.finmind_client import FinMindResearchClient


PER_STOCK_DATASETS = (
    "TaiwanStockPrice",
    "TaiwanStockInstitutionalInvestorsBuySell",
    "TaiwanStockDividendResult",
    "TaiwanStockCapitalReductionReferencePrice",
)


def download_phase1_data(
    *,
    database: ResearchDatabase,
    client: FinMindResearchClient,
    symbols: list[str],
    start_date: str,
    end_date: str,
    force: bool = False,
) -> None:
    download_global_reference_data(database, client, start_date, end_date, force)
    for index, symbol in enumerate(symbols, start=1):
        print(f"[{index}/{len(symbols)}] 下載 {symbol}")
        for dataset in PER_STOCK_DATASETS:
            _download_stock_dataset(
                database=database,
                client=client,
                dataset=dataset,
                stock_id=symbol,
                start_date=start_date,
                end_date=end_date,
                force=force,
            )

    removed = cleanup_duplicate_corporate_actions(database, symbols)
    if removed:
        print(f"已清除重複或失效的公司行動日期：{removed} 筆")


def download_global_reference_data(
    database: ResearchDatabase,
    client: FinMindResearchClient,
    start_date: str,
    end_date: str,
    force: bool,
) -> None:
    global_specs = (
        ("TaiwanStockInfo", None, None),
        ("TaiwanStockTradingDate", None, None),
        ("TaiwanStockSplitPrice", start_date, end_date),
        ("TaiwanStockParValueChange", start_date, end_date),
        ("TaiwanStockDelisting", None, None),
    )
    for dataset, dataset_start, dataset_end in global_specs:
        key = "*"
        requested_start = dataset_start or "0001-01-01"
        requested_end = dataset_end or "9999-12-31"
        if not force and database.download_is_complete(
            dataset=dataset,
            stock_id=key,
            requested_start=requested_start,
            requested_end=requested_end,
        ):
            continue
        print(f"下載全市場資料：{dataset}")
        try:
            rows = client.fetch(
                dataset,
                start_date=dataset_start,
                end_date=dataset_end,
            )
            if dataset == "TaiwanStockInfo":
                count = _upsert_stock_info(database, rows)
            elif dataset == "TaiwanStockTradingDate":
                count = _upsert_market_calendar(database, rows)
            elif dataset == "TaiwanStockSplitPrice":
                count = replace_corporate_actions(
                    database=database,
                    rows=normalize_split_rows(rows),
                    source_dataset=dataset,
                    start_date=requested_start,
                    end_date=requested_end,
                )
            elif dataset == "TaiwanStockParValueChange":
                count = replace_corporate_actions(
                    database=database,
                    rows=normalize_par_value_rows(rows),
                    source_dataset=dataset,
                    start_date=requested_start,
                    end_date=requested_end,
                )
            else:
                count = _upsert_delistings(database, rows)
            database.mark_download(
                dataset=dataset,
                stock_id=key,
                requested_start=requested_start,
                requested_end=requested_end,
                status="complete",
                row_count=count,
            )
        except Exception as exc:
            database.mark_download(
                dataset=dataset,
                stock_id=key,
                requested_start=requested_start,
                requested_end=requested_end,
                status="failed",
                error=str(exc),
            )
            raise


def _download_stock_dataset(
    *,
    database: ResearchDatabase,
    client: FinMindResearchClient,
    dataset: str,
    stock_id: str,
    start_date: str,
    end_date: str,
    force: bool,
) -> None:
    if not force and database.download_is_complete(
        dataset=dataset,
        stock_id=stock_id,
        requested_start=start_date,
        requested_end=end_date,
    ):
        return
    try:
        rows = client.fetch(
            dataset,
            data_id=stock_id,
            start_date=start_date,
            end_date=end_date,
        )
        if dataset == "TaiwanStockPrice":
            count = _upsert_prices(database, rows)
        elif dataset == "TaiwanStockInstitutionalInvestorsBuySell":
            count = _upsert_institutional(database, rows)
        elif dataset == "TaiwanStockDividendResult":
            count = replace_corporate_actions(
                database=database,
                rows=normalize_dividend_rows(rows),
                source_dataset=dataset,
                stock_id=stock_id,
                start_date=start_date,
                end_date=end_date,
            )
        else:
            count = replace_corporate_actions(
                database=database,
                rows=normalize_capital_reduction_rows(rows),
                source_dataset=dataset,
                stock_id=stock_id,
                start_date=start_date,
                end_date=end_date,
            )
        database.mark_download(
            dataset=dataset,
            stock_id=stock_id,
            requested_start=start_date,
            requested_end=end_date,
            status="complete",
            row_count=count,
        )
    except Exception as exc:
        database.mark_download(
            dataset=dataset,
            stock_id=stock_id,
            requested_start=start_date,
            requested_end=end_date,
            status="failed",
            error=str(exc),
        )
        raise


def _upsert_market_calendar(
    database: ResearchDatabase, rows: Iterable[dict[str, Any]]
) -> int:
    return database.executemany(
        """
        INSERT INTO market_calendar (date)
        VALUES (?)
        ON CONFLICT(date) DO NOTHING
        """,
        (
            (str(row.get("date") or ""),)
            for row in rows
            if row.get("date")
        ),
    )


def _upsert_stock_info(database: ResearchDatabase, rows: Iterable[dict[str, Any]]) -> int:
    return database.executemany(
        """
        INSERT INTO stock_info (
            stock_id, stock_name, industry_category, market_type, info_date, updated_at
        ) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(stock_id) DO UPDATE SET
            stock_name=excluded.stock_name,
            industry_category=excluded.industry_category,
            market_type=excluded.market_type,
            info_date=excluded.info_date,
            updated_at=CURRENT_TIMESTAMP
        """,
        (
            (
                str(row.get("stock_id") or ""),
                str(row.get("stock_name") or ""),
                str(row.get("industry_category") or ""),
                str(row.get("type") or ""),
                str(row.get("date") or ""),
            )
            for row in rows
            if row.get("stock_id")
        ),
    )


def _upsert_prices(database: ResearchDatabase, rows: Iterable[dict[str, Any]]) -> int:
    return database.executemany(
        """
        INSERT INTO stock_prices (
            stock_id, date, trading_volume, trading_money, open, high,
            low, close, spread, trading_turnover
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(stock_id, date) DO UPDATE SET
            trading_volume=excluded.trading_volume,
            trading_money=excluded.trading_money,
            open=excluded.open,
            high=excluded.high,
            low=excluded.low,
            close=excluded.close,
            spread=excluded.spread,
            trading_turnover=excluded.trading_turnover
        """,
        (
            (
                str(row.get("stock_id") or ""),
                str(row.get("date") or ""),
                _integer(row.get("Trading_Volume")),
                _integer(row.get("Trading_money")),
                _float(row.get("open")),
                _float(row.get("max")),
                _float(row.get("min")),
                _float(row.get("close")),
                _float(row.get("spread")),
                _integer(row.get("Trading_turnover")),
            )
            for row in rows
            if row.get("stock_id") and row.get("date")
        ),
    )


def _upsert_institutional(database: ResearchDatabase, rows: Iterable[dict[str, Any]]) -> int:
    grouped: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        stock_id = str(row.get("stock_id") or "")
        event_date = str(row.get("date") or "")
        name = str(row.get("name") or "")
        if not stock_id or not event_date or not name:
            continue
        grouped[(stock_id, event_date)][name] += _integer(row.get("buy")) - _integer(
            row.get("sell")
        )

    materialized = []
    for (stock_id, event_date), values in grouped.items():
        foreign = values.get("Foreign_Investor", 0)
        trust = values.get("Investment_Trust", 0)
        dealer_self = values.get("Dealer_self", 0)
        dealer_hedging = values.get("Dealer_Hedging", 0)
        foreign_dealer_self = values.get("Foreign_Dealer_Self", 0)
        materialized.append(
            (
                stock_id,
                event_date,
                foreign,
                trust,
                dealer_self,
                dealer_hedging,
                foreign_dealer_self,
                foreign + trust + dealer_self,
            )
        )

    return database.executemany(
        """
        INSERT INTO institutional_flows (
            stock_id, date, foreign_net, investment_trust_net,
            dealer_self_net, dealer_hedging_net, foreign_dealer_self_net,
            selected_total_net
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(stock_id, date) DO UPDATE SET
            foreign_net=excluded.foreign_net,
            investment_trust_net=excluded.investment_trust_net,
            dealer_self_net=excluded.dealer_self_net,
            dealer_hedging_net=excluded.dealer_hedging_net,
            foreign_dealer_self_net=excluded.foreign_dealer_self_net,
            selected_total_net=excluded.selected_total_net
        """,
        materialized,
    )


def _upsert_delistings(database: ResearchDatabase, rows: Iterable[dict[str, Any]]) -> int:
    return database.executemany(
        """
        INSERT INTO delistings (stock_id, date, stock_name)
        VALUES (?, ?, ?)
        ON CONFLICT(stock_id, date) DO UPDATE SET stock_name=excluded.stock_name
        """,
        (
            (
                str(row.get("stock_id") or ""),
                str(row.get("date") or ""),
                str(row.get("stock_name") or ""),
            )
            for row in rows
            if row.get("stock_id") and row.get("date")
        ),
    )


def _integer(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def upsert_market_calendar(
    database: ResearchDatabase, rows: Iterable[dict[str, Any]]
) -> int:
    """Public incremental-update wrapper for the existing calendar upsert."""
    return _upsert_market_calendar(database, rows)


def upsert_prices(database: ResearchDatabase, rows: Iterable[dict[str, Any]]) -> int:
    """Public incremental-update wrapper for the existing price upsert."""
    return _upsert_prices(database, rows)


def upsert_institutional(
    database: ResearchDatabase, rows: Iterable[dict[str, Any]]
) -> int:
    """Public incremental-update wrapper for the existing institutional upsert."""
    return _upsert_institutional(database, rows)
