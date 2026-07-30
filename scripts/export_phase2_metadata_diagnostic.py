from __future__ import annotations

import csv
import json
import sqlite3
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "research/data/institutional_phase1.sqlite"
OUTPUT_DIR = PROJECT_ROOT / "research/output/phase2_metadata_diagnostic"
OUTPUT_ZIP = PROJECT_ROOT / "research/output/phase2_metadata_diagnostic.zip"


def main() -> None:
    database_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_DB
    if not database_path.exists():
        raise SystemExit(f"找不到 SQLite：{database_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        _export_official_key_summary(connection)
        _export_official_samples(connection)
        _export_delistings(connection)
        _export_overlap_diagnostics(connection)
        _export_non_common_candidates(connection)
        _export_summary(connection, database_path)

    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUTPUT_DIR.glob("*.csv")):
            archive.write(path, arcname=path.name)

    print("Phase 2 母體診斷已完成。")
    print(f"已產生：{OUTPUT_ZIP}")


def _export_official_key_summary(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        """
        SELECT stock_id, market_type, listing_date, raw_json
        FROM official_company_info
        ORDER BY market_type, stock_id
        """
    ).fetchall()

    key_counts: dict[str, Counter[str]] = defaultdict(Counter)
    date_like_values: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    market_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"rows": 0, "listing_date_present": 0, "listing_date_missing": 0}
    )

    for row in rows:
        market = str(row["market_type"] or "")
        market_counts[market]["rows"] += 1
        if row["listing_date"]:
            market_counts[market]["listing_date_present"] += 1
        else:
            market_counts[market]["listing_date_missing"] += 1
        payload = _load_json(row["raw_json"])
        for key, value in payload.items():
            key_text = str(key)
            key_counts[market][key_text] += 1
            if _looks_like_date_key(key_text):
                value_text = str(value or "").strip()
                if value_text:
                    date_like_values[(market, key_text)][value_text] += 1

    output: list[dict[str, Any]] = []
    for market in sorted(market_counts):
        stats = market_counts[market]
        for key, count in key_counts[market].most_common():
            samples = [
                value
                for value, _ in date_like_values.get((market, key), Counter()).most_common(5)
            ]
            output.append(
                {
                    "market_type": market,
                    "official_rows": stats["rows"],
                    "listing_date_present": stats["listing_date_present"],
                    "listing_date_missing": stats["listing_date_missing"],
                    "raw_key": key,
                    "key_row_count": count,
                    "date_like_key": 1 if _looks_like_date_key(key) else 0,
                    "sample_values": " | ".join(samples),
                }
            )
    _write_csv(OUTPUT_DIR / "official_company_info_key_summary.csv", output)


def _export_official_samples(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        """
        SELECT stock_id, market_type, stock_name, listing_date, report_date,
               source_dataset, raw_json, updated_at
        FROM official_company_info
        ORDER BY market_type, stock_id
        """
    ).fetchall()

    selected: list[dict[str, Any]] = []
    per_market = Counter()
    for row in rows:
        market = str(row["market_type"] or "")
        if per_market[market] >= 10:
            continue
        payload = _load_json(row["raw_json"])
        selected.append(
            {
                "stock_id": row["stock_id"],
                "market_type": market,
                "stock_name": row["stock_name"],
                "parsed_listing_date": row["listing_date"],
                "parsed_report_date": row["report_date"],
                "source_dataset": row["source_dataset"],
                "raw_keys": " | ".join(str(key) for key in payload.keys()),
                "date_like_fields": json.dumps(
                    {
                        str(key): value
                        for key, value in payload.items()
                        if _looks_like_date_key(str(key))
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "raw_json": row["raw_json"],
                "updated_at": row["updated_at"],
            }
        )
        per_market[market] += 1
    _write_csv(OUTPUT_DIR / "official_company_info_sample.csv", selected)


def _export_delistings(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        """
        SELECT d.stock_id, d.stock_name, d.date AS delisting_date,
               s.stock_name AS stock_info_name,
               s.market_type AS stock_info_market,
               s.industry_category,
               o.market_type AS current_official_market,
               o.stock_name AS current_official_name,
               o.listing_date AS current_official_listing_date,
               CASE WHEN t.stock_id IS NOT NULL THEN 1 ELSE 0 END AS in_twse_official_delisted
        FROM delistings d
        LEFT JOIN stock_info s ON s.stock_id=d.stock_id
        LEFT JOIN official_company_info o ON o.stock_id=d.stock_id
        LEFT JOIN official_twse_delisted t ON t.stock_id=d.stock_id
        WHERE d.date >= '2015-01-01'
        ORDER BY d.date, d.stock_id
        """
    ).fetchall()
    _write_csv(OUTPUT_DIR / "delistings_since_2015.csv", [dict(row) for row in rows])


def _export_overlap_diagnostics(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        """
        WITH delisted AS (
            SELECT stock_id, MIN(date) AS first_delisting_date,
                   MAX(date) AS last_delisting_date, COUNT(*) AS delisting_count
            FROM delistings
            WHERE date >= '2015-01-01'
            GROUP BY stock_id
        ), price_range AS (
            SELECT stock_id, MIN(date) AS first_price_date, MAX(date) AS last_price_date,
                   COUNT(*) AS price_rows
            FROM stock_prices
            GROUP BY stock_id
        ), flow_range AS (
            SELECT stock_id, MIN(date) AS first_flow_date, MAX(date) AS last_flow_date,
                   COUNT(*) AS flow_rows
            FROM institutional_flows
            GROUP BY stock_id
        )
        SELECT d.stock_id, s.stock_name AS stock_info_name,
               s.market_type AS stock_info_market, s.industry_category,
               d.first_delisting_date, d.last_delisting_date, d.delisting_count,
               o.market_type AS current_official_market,
               o.stock_name AS current_official_name,
               o.listing_date AS current_official_listing_date,
               p.first_price_date, p.last_price_date, COALESCE(p.price_rows, 0) AS price_rows,
               f.first_flow_date, f.last_flow_date, COALESCE(f.flow_rows, 0) AS flow_rows,
               m.current_status AS current_model_status,
               m.download_enabled, m.training_enabled, m.inclusion_reason
        FROM delisted d
        LEFT JOIN stock_info s ON s.stock_id=d.stock_id
        LEFT JOIN official_company_info o ON o.stock_id=d.stock_id
        LEFT JOIN price_range p ON p.stock_id=d.stock_id
        LEFT JOIN flow_range f ON f.stock_id=d.stock_id
        LEFT JOIN model_universe m ON m.stock_id=d.stock_id
        ORDER BY d.stock_id
        """
    ).fetchall()
    _write_csv(OUTPUT_DIR / "delisting_overlap_diagnostic.csv", [dict(row) for row in rows])


def _export_non_common_candidates(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        """
        SELECT m.stock_id, m.stock_name, m.market_type, m.industry_category,
               m.current_status, m.download_enabled, m.training_enabled,
               m.inclusion_status, m.inclusion_reason
        FROM model_universe m
        WHERE m.industry_category IN ('存託憑證', 'TDR', '特別股', 'ETF', 'ETN')
           OR UPPER(m.stock_name) LIKE '%-DR%'
           OR m.stock_name LIKE '%特別股%'
        ORDER BY m.market_type, m.stock_id
        """
    ).fetchall()
    _write_csv(OUTPUT_DIR / "non_common_security_candidates.csv", [dict(row) for row in rows])


def _export_summary(connection: sqlite3.Connection, database_path: Path) -> None:
    metrics = [
        ("generated_at", datetime.now().isoformat(timespec="seconds")),
        ("database_path", str(database_path)),
        ("official_company_info_total", _scalar(connection, "SELECT COUNT(*) FROM official_company_info")),
        ("official_twse_total", _scalar(connection, "SELECT COUNT(*) FROM official_company_info WHERE market_type='twse'")),
        ("official_tpex_total", _scalar(connection, "SELECT COUNT(*) FROM official_company_info WHERE market_type='tpex'")),
        ("official_twse_missing_listing_date", _scalar(connection, "SELECT COUNT(*) FROM official_company_info WHERE market_type='twse' AND (listing_date IS NULL OR listing_date='')")),
        ("official_tpex_missing_listing_date", _scalar(connection, "SELECT COUNT(*) FROM official_company_info WHERE market_type='tpex' AND (listing_date IS NULL OR listing_date='')")),
        ("delistings_since_2015", _scalar(connection, "SELECT COUNT(*) FROM delistings WHERE date >= '2015-01-01'")),
        ("delisted_codes_since_2015", _scalar(connection, "SELECT COUNT(DISTINCT stock_id) FROM delistings WHERE date >= '2015-01-01'")),
        ("delisted_codes_also_in_current_official", _scalar(connection, "SELECT COUNT(DISTINCT d.stock_id) FROM delistings d JOIN official_company_info o ON o.stock_id=d.stock_id WHERE d.date >= '2015-01-01'")),
        ("model_review_required", _scalar(connection, "SELECT COUNT(*) FROM model_universe WHERE inclusion_status='review_required'")),
        ("model_training_enabled", _scalar(connection, "SELECT COUNT(*) FROM model_universe WHERE training_enabled=1")),
    ]
    _write_csv(
        OUTPUT_DIR / "diagnostic_summary.csv",
        [{"metric": metric, "value": value} for metric, value in metrics],
    )


def _scalar(connection: sqlite3.Connection, sql: str) -> Any:
    row = connection.execute(sql).fetchone()
    return row[0] if row else None


def _load_json(value: Any) -> dict[str, Any]:
    try:
        payload = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _looks_like_date_key(key: str) -> bool:
    normalized = key.lower().replace("_", "").replace(" ", "")
    return any(token in normalized for token in ("日期", "date", "掛牌", "上市", "上櫃"))


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    fieldnames: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["empty"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


if __name__ == "__main__":
    main()
