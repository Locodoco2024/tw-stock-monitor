from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.institutional_model.database import ResearchDatabase


def export_validation_reports(
    *,
    database: ResearchDatabase,
    output_dir: Path | str,
    primary_horizon: int = 10,
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with database.connect() as connection:
        labels = pd.read_sql_query(
            """
            SELECT
                l.stock_id,
                COALESCE(i.stock_name, u.stock_name) AS stock_name,
                COALESCE(i.market_type, u.market_hint) AS market,
                l.signal_date,
                l.entry_date,
                l.entry_open,
                l.target_date,
                l.target_close,
                l.raw_return,
                l.adjusted_return,
                l.max_adjusted_return,
                l.min_adjusted_return,
                l.action_types,
                l.entry_day_action_ignored,
                l.label,
                l.status,
                l.error
            FROM label_results l
            JOIN stock_universe u ON u.stock_id=l.stock_id AND u.enabled=1
            LEFT JOIN stock_info i ON i.stock_id=l.stock_id
            WHERE l.horizon=?
            ORDER BY l.stock_id, l.signal_date
            """,
            connection,
            params=(primary_horizon,),
        )
        universe = pd.read_sql_query(
            """
            SELECT
                u.stock_id,
                COALESCE(i.stock_name, u.stock_name) AS stock_name,
                COALESCE(i.market_type, u.market_hint) AS market,
                u.selection_group,
                u.selection_reason,
                u.expected_cases
            FROM stock_universe u
            LEFT JOIN stock_info i ON i.stock_id=u.stock_id
            WHERE u.enabled=1
            ORDER BY u.selection_group, u.stock_id
            """,
            connection,
        )
        prices = pd.read_sql_query(
            """
            WITH delisting_dates AS (
                SELECT stock_id, MAX(date) AS delisting_date
                FROM delistings
                GROUP BY stock_id
            )
            SELECT p.stock_id, p.date, p.trading_volume, p.trading_money,
                   p.open, p.high, p.low, p.close
            FROM stock_prices p
            JOIN stock_universe u ON u.stock_id=p.stock_id AND u.enabled=1
            LEFT JOIN delisting_dates d ON d.stock_id=p.stock_id
            WHERE d.delisting_date IS NULL OR p.date < d.delisting_date
            ORDER BY p.stock_id, p.date
            """,
            connection,
        )
        flow_counts = pd.read_sql_query(
            """
            WITH delisting_dates AS (
                SELECT stock_id, MAX(date) AS delisting_date
                FROM delistings
                GROUP BY stock_id
            )
            SELECT f.stock_id, COUNT(*) AS institutional_days
            FROM institutional_flows f
            JOIN stock_universe u ON u.stock_id=f.stock_id AND u.enabled=1
            LEFT JOIN delisting_dates d ON d.stock_id=f.stock_id
            WHERE d.delisting_date IS NULL OR f.date < d.delisting_date
            GROUP BY f.stock_id
            """,
            connection,
        )
        action_counts = pd.read_sql_query(
            """
            SELECT a.stock_id, COUNT(*) AS action_count
            FROM corporate_actions a
            JOIN stock_universe u ON u.stock_id=a.stock_id AND u.enabled=1
            GROUP BY a.stock_id
            """,
            connection,
        )
        label_counts = pd.read_sql_query(
            """
            SELECT
                l.stock_id,
                SUM(CASE WHEN l.status='ok' THEN 1 ELSE 0 END)
                    AS valid_10d_labels,
                SUM(CASE WHEN l.status='invalid_data' THEN 1 ELSE 0 END)
                    AS invalid_10d_labels,
                SUM(CASE WHEN l.status='insufficient_future_data' THEN 1 ELSE 0 END)
                    AS pending_10d_labels,
                SUM(CASE WHEN l.status='unavailable_entry_price' THEN 1 ELSE 0 END)
                    AS unavailable_entry_10d_labels,
                SUM(CASE WHEN l.status='unavailable_target_price' THEN 1 ELSE 0 END)
                    AS unavailable_target_10d_labels,
                SUM(CASE WHEN l.status LIKE 'delisted_before_%' THEN 1 ELSE 0 END)
                    AS delisted_10d_labels
            FROM label_results l
            JOIN stock_universe u ON u.stock_id=l.stock_id AND u.enabled=1
            WHERE l.horizon=?
            GROUP BY l.stock_id
            """,
            connection,
            params=(primary_horizon,),
        )
        delistings = pd.read_sql_query(
            """
            SELECT d.stock_id, MAX(d.date) AS delisting_date
            FROM delistings d
            JOIN stock_universe u ON u.stock_id=d.stock_id AND u.enabled=1
            GROUP BY d.stock_id
            """,
            connection,
        )
        actions = pd.read_sql_query(
            """
            SELECT
                a.stock_id,
                COALESCE(i.stock_name, u.stock_name) AS stock_name,
                a.date,
                a.action_type,
                a.before_price,
                a.reference_price,
                a.description,
                a.source_dataset
            FROM corporate_actions a
            JOIN stock_universe u ON u.stock_id=a.stock_id AND u.enabled=1
            LEFT JOIN stock_info i ON i.stock_id=a.stock_id
            ORDER BY a.stock_id, a.date, a.action_type
            """,
            connection,
        )
        signal_audit = pd.read_sql_query(
            """
            WITH delisting_dates AS (
                SELECT stock_id, MAX(date) AS delisting_date
                FROM delistings
                GROUP BY stock_id
            ), classified AS (
                SELECT
                    f.stock_id,
                    COALESCE(i.stock_name, u.stock_name) AS stock_name,
                    COALESCE(i.market_type, u.market_hint) AS market,
                    f.date,
                    f.foreign_net,
                    f.investment_trust_net,
                    f.dealer_self_net,
                    f.dealer_hedging_net,
                    f.selected_total_net,
                    p.trading_volume,
                    p.trading_money,
                    p.open,
                    p.close,
                    d.delisting_date,
                    CASE
                        WHEN c.date IS NULL THEN 'non_market_trading_date'
                        WHEN p.date IS NULL THEN 'missing_stock_price'
                        WHEN COALESCE(p.open, 0) <= 0
                          OR COALESCE(p.close, 0) <= 0
                          OR COALESCE(p.trading_volume, 0) <= 0
                            THEN 'invalid_stock_price'
                        WHEN d.delisting_date IS NOT NULL AND f.date >= d.delisting_date
                            THEN 'on_or_after_delisting'
                        ELSE 'valid'
                    END AS exclusion_reason
                FROM institutional_flows f
                JOIN stock_universe u ON u.stock_id=f.stock_id AND u.enabled=1
                LEFT JOIN stock_info i ON i.stock_id=f.stock_id
                LEFT JOIN market_calendar c ON c.date=f.date
                LEFT JOIN stock_prices p ON p.stock_id=f.stock_id AND p.date=f.date
                LEFT JOIN delisting_dates d ON d.stock_id=f.stock_id
            )
            SELECT *
            FROM classified
            WHERE exclusion_reason <> 'valid'
            ORDER BY stock_id, date
            """,
            connection,
        )
        zero_price_audit = pd.read_sql_query(
            """
            WITH delisting_dates AS (
                SELECT stock_id, MAX(date) AS delisting_date
                FROM delistings
                GROUP BY stock_id
            )
            SELECT
                p.stock_id,
                COALESCE(i.stock_name, u.stock_name) AS stock_name,
                COALESCE(i.market_type, u.market_hint) AS market,
                p.date,
                p.trading_volume,
                p.trading_money,
                p.open,
                p.high,
                p.low,
                p.close,
                CASE WHEN c.date IS NOT NULL THEN 1 ELSE 0 END AS is_market_trading_date,
                CASE WHEN f.date IS NOT NULL THEN 1 ELSE 0 END AS has_institutional_flow,
                d.delisting_date,
                CASE
                    WHEN d.delisting_date IS NOT NULL AND p.date >= d.delisting_date
                        THEN 'post_delisting_price'
                    WHEN p.open IS NULL OR p.close IS NULL THEN 'missing_price_value'
                    WHEN p.open <= 0 AND p.close <= 0 THEN 'zero_open_and_close'
                    WHEN p.open <= 0 THEN 'zero_open'
                    WHEN p.close <= 0 THEN 'zero_close'
                    WHEN COALESCE(p.trading_volume, 0) <= 0 THEN 'zero_volume'
                    ELSE 'other'
                END AS issue_type
            FROM stock_prices p
            JOIN stock_universe u ON u.stock_id=p.stock_id AND u.enabled=1
            LEFT JOIN stock_info i ON i.stock_id=p.stock_id
            LEFT JOIN market_calendar c ON c.date=p.date
            LEFT JOIN institutional_flows f ON f.stock_id=p.stock_id AND f.date=p.date
            LEFT JOIN delisting_dates d ON d.stock_id=p.stock_id
            WHERE (d.delisting_date IS NOT NULL AND p.date >= d.delisting_date)
               OR p.open IS NULL
               OR p.close IS NULL
               OR p.open <= 0
               OR p.close <= 0
               OR COALESCE(p.trading_volume, 0) <= 0
            ORDER BY p.stock_id, p.date
            """,
            connection,
        )
        price_exclusion_counts = pd.read_sql_query(
            """
            WITH delisting_dates AS (
                SELECT stock_id, MAX(date) AS delisting_date
                FROM delistings
                GROUP BY stock_id
            )
            SELECT
                p.stock_id,
                COUNT(*) AS excluded_post_delisting_price_days
            FROM stock_prices p
            JOIN stock_universe u ON u.stock_id=p.stock_id AND u.enabled=1
            JOIN delisting_dates d ON d.stock_id=p.stock_id
            WHERE p.date >= d.delisting_date
            GROUP BY p.stock_id
            """,
            connection,
        )
        signal_exclusion_counts = pd.read_sql_query(
            """
            WITH delisting_dates AS (
                SELECT stock_id, MAX(date) AS delisting_date
                FROM delistings
                GROUP BY stock_id
            ), classified AS (
                SELECT
                    f.stock_id,
                    CASE
                        WHEN c.date IS NULL THEN 'non_market_trading_date'
                        WHEN p.date IS NULL THEN 'missing_stock_price'
                        WHEN COALESCE(p.open, 0) <= 0
                          OR COALESCE(p.close, 0) <= 0
                          OR COALESCE(p.trading_volume, 0) <= 0
                            THEN 'invalid_stock_price'
                        WHEN d.delisting_date IS NOT NULL AND f.date >= d.delisting_date
                            THEN 'on_or_after_delisting'
                        ELSE 'valid'
                    END AS result
                FROM institutional_flows f
                JOIN stock_universe u ON u.stock_id=f.stock_id AND u.enabled=1
                LEFT JOIN market_calendar c ON c.date=f.date
                LEFT JOIN stock_prices p ON p.stock_id=f.stock_id AND p.date=f.date
                LEFT JOIN delisting_dates d ON d.stock_id=f.stock_id
            )
            SELECT
                stock_id,
                SUM(CASE WHEN result='valid' THEN 1 ELSE 0 END) AS valid_signal_days,
                SUM(CASE WHEN result='non_market_trading_date' THEN 1 ELSE 0 END)
                    AS excluded_non_market_flow_days,
                SUM(CASE WHEN result='missing_stock_price' THEN 1 ELSE 0 END)
                    AS excluded_missing_price_signal_days,
                SUM(CASE WHEN result='invalid_stock_price' THEN 1 ELSE 0 END)
                    AS excluded_invalid_price_signal_days,
                SUM(CASE WHEN result='on_or_after_delisting' THEN 1 ELSE 0 END)
                    AS excluded_post_delisting_signal_days
            FROM classified
            GROUP BY stock_id
            """,
            connection,
        )

    price_summary = _summarize_prices(prices)
    summary = universe
    for frame in (
        price_summary,
        flow_counts,
        action_counts,
        label_counts,
        delistings,
        signal_exclusion_counts,
        price_exclusion_counts,
    ):
        summary = summary.merge(frame, how="left", on="stock_id")

    integer_columns = (
        "price_days",
        "institutional_days",
        "action_count",
        "valid_10d_labels",
        "invalid_10d_labels",
        "pending_10d_labels",
        "unavailable_entry_10d_labels",
        "unavailable_target_10d_labels",
        "delisted_10d_labels",
        "recent_20_normal_trading_days",
        "recent_20_max_zero_volume_streak",
        "valid_signal_days",
        "excluded_non_market_flow_days",
        "excluded_missing_price_signal_days",
        "excluded_invalid_price_signal_days",
        "excluded_post_delisting_signal_days",
        "excluded_post_delisting_price_days",
    )
    for column in integer_columns:
        if column in summary:
            summary[column] = summary[column].fillna(0).astype(int)

    labels_path = output_path / "phase1_validation_10d.csv"
    summary_path = output_path / "phase1_stock_summary.csv"
    actions_path = output_path / "phase1_corporate_actions.csv"
    signal_audit_path = output_path / "phase1_signal_date_audit.csv"
    zero_price_path = output_path / "phase1_zero_price_audit.csv"
    labels.to_csv(labels_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    actions.to_csv(actions_path, index=False, encoding="utf-8-sig")
    signal_audit.to_csv(signal_audit_path, index=False, encoding="utf-8-sig")
    zero_price_audit.to_csv(zero_price_path, index=False, encoding="utf-8-sig")
    return [
        labels_path,
        summary_path,
        actions_path,
        signal_audit_path,
        zero_price_path,
    ]


def _summarize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "stock_id",
        "price_start",
        "price_end",
        "price_days",
        "recent_20_median_trading_money",
        "recent_20_normal_trading_days",
        "recent_20_max_zero_volume_streak",
    ]
    if prices.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for stock_id, group in prices.groupby("stock_id", sort=True):
        ordered = group.sort_values("date")
        recent = ordered.tail(20).copy()
        volume = pd.to_numeric(recent["trading_volume"], errors="coerce").fillna(0)
        money = pd.to_numeric(recent["trading_money"], errors="coerce")
        open_price = pd.to_numeric(recent["open"], errors="coerce").fillna(0)
        close_price = pd.to_numeric(recent["close"], errors="coerce").fillna(0)
        normal = (volume > 0) & (open_price > 0) & (close_price > 0)
        rows.append(
            {
                "stock_id": stock_id,
                "price_start": ordered["date"].min(),
                "price_end": ordered["date"].max(),
                "price_days": len(ordered),
                "recent_20_median_trading_money": money.median(),
                "recent_20_normal_trading_days": int(normal.sum()),
                "recent_20_max_zero_volume_streak": _max_true_streak(volume <= 0),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _max_true_streak(series: pd.Series) -> int:
    best = 0
    current = 0
    for value in series.astype(bool):
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best
