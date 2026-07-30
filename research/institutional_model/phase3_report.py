from __future__ import annotations

import csv
import gzip
import zipfile
from pathlib import Path
from typing import Any

from research.institutional_model.database import ResearchDatabase
from research.institutional_model.phase3_dataset import (
    FEATURE_COLUMNS,
    LABEL_RULE_VERSION,
    feature_dictionary_rows,
    sha256_file,
)


def export_phase3_reports(
    *,
    database: ResearchDatabase,
    output_dir: Path | str,
    config_signature: str,
    start_date: str,
    end_date: str,
) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    stock_rows = _stock_summary_rows(database, config_signature)
    label_rows = _label_distribution_rows(database, config_signature)
    exclusion_rows = _exclusion_rows(database, config_signature)
    liquidity_rows = _liquidity_rows(stock_rows)
    manifest_rows = _manifest_rows(output, config_signature)
    summary_rows = _summary_rows(
        database=database,
        stock_rows=stock_rows,
        label_rows=label_rows,
        manifest_rows=manifest_rows,
        config_signature=config_signature,
        start_date=start_date,
        end_date=end_date,
    )

    paths = [
        _write_csv(output / "phase3_summary.csv", summary_rows),
        _write_csv(output / "phase3_stock_summary.csv", stock_rows),
        _write_csv(output / "phase3_label_distribution.csv", label_rows),
        _write_csv(output / "phase3_exclusion_reasons.csv", exclusion_rows),
        _write_csv(output / "phase3_liquidity_thresholds.csv", liquidity_rows),
        _write_csv(output / "phase3_feature_dictionary.csv", feature_dictionary_rows()),
        _write_csv(output / "phase3_dataset_manifest.csv", manifest_rows),
    ]
    archive = output / "phase3_validation_reports.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in paths:
            bundle.write(path, arcname=path.name)
    paths.append(archive)
    return paths


def _stock_summary_rows(
    database: ResearchDatabase, config_signature: str
) -> list[dict[str, Any]]:
    rows = database.query(
        """
        SELECT
            u.stock_id, u.stock_name, u.market_type, u.current_status,
            u.listing_date, u.delisting_date, u.training_enabled, u.inclusion_status,
            s.status AS phase3_status, s.total_rows,
            s.feature_ready_rows, s.eligible_5d_rows,
            s.eligible_10d_rows, s.eligible_20d_rows,
            s.liquidity_10m_rows, s.liquidity_20m_rows,
            s.liquidity_50m_rows, s.liquidity_100m_rows,
            s.first_signal_date, s.last_signal_date,
            COALESCE(s.error, '') AS error
        FROM model_universe u
        LEFT JOIN phase3_build_status s
          ON s.stock_id=u.stock_id AND s.config_signature=?
        WHERE u.market_type IN ('twse', 'tpex')
        ORDER BY u.market_type, u.stock_id
        """,
        (config_signature,),
    )
    return [dict(row) for row in rows]


def _label_distribution_rows(
    database: ResearchDatabase, config_signature: str
) -> list[dict[str, Any]]:
    rows = database.query(
        """
        SELECT u.market_type, d.signal_year, d.label,
               SUM(d.sample_count) AS sample_count
        FROM phase3_label_distribution d
        JOIN model_universe u ON u.stock_id=d.stock_id
        WHERE d.config_signature=?
        GROUP BY u.market_type, d.signal_year, d.label
        ORDER BY u.market_type, d.signal_year,
                 CASE d.label WHEN 'UP' THEN 1 WHEN 'FLAT' THEN 2 ELSE 3 END
        """,
        (config_signature,),
    )
    return [dict(row) for row in rows]


def _exclusion_rows(
    database: ResearchDatabase, config_signature: str
) -> list[dict[str, Any]]:
    rows = database.query(
        """
        SELECT u.market_type, e.reason, SUM(e.sample_count) AS sample_count
        FROM phase3_exclusion_stats e
        JOIN model_universe u ON u.stock_id=e.stock_id
        WHERE e.config_signature=?
        GROUP BY u.market_type, e.reason
        ORDER BY u.market_type, sample_count DESC, e.reason
        """,
        (config_signature,),
    )
    return [dict(row) for row in rows]


def _liquidity_rows(stock_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tpex = [row for row in stock_rows if row.get("market_type") == "tpex"]
    result = []
    for threshold in (10, 20, 50, 100):
        field = f"liquidity_{threshold}m_rows"
        result.append(
            {
                "market_type": "tpex",
                "threshold_ntd_million": threshold,
                "stocks_with_samples": sum(
                    1 for row in tpex if int(row.get(field) or 0) > 0
                ),
                "eligible_sample_rows": sum(int(row.get(field) or 0) for row in tpex),
                "primary_threshold": 1 if threshold == 20 else 0,
            }
        )
    return result


def _manifest_rows(
    output: Path, config_signature: str
) -> list[dict[str, Any]]:
    result = []
    for market in ("twse", "tpex"):
        path = output / f"phase3_training_{market}.csv.gz"
        if not path.exists():
            continue
        print(f"計算資料集筆數與 SHA-256：{path.name}")
        result.append(
            {
                "file_name": path.name,
                "market_type": market,
                "row_count": _gzip_csv_row_count(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "config_signature": config_signature,
                "label_rule_version": LABEL_RULE_VERSION,
            }
        )
    return result


def _summary_rows(
    *,
    database: ResearchDatabase,
    stock_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    config_signature: str,
    start_date: str,
    end_date: str,
) -> list[dict[str, Any]]:
    training_universe = [
        row
        for row in stock_rows
        if int(row.get("training_enabled") or 0) == 1
    ]
    completed = [
        row for row in training_universe if row.get("phase3_status") == "complete"
    ]
    failed = [row for row in stock_rows if row.get("phase3_status") == "failed"]
    label_totals = {
        label: sum(
            int(row.get("sample_count") or 0)
            for row in label_rows
            if row.get("label") == label
        )
        for label in ("UP", "FLAT", "DOWN")
    }
    manifest_count = {
        row["market_type"]: int(row.get("row_count") or 0)
        for row in manifest_rows
    }
    future_listing_excluded = int(
        database.scalar(
            """
            SELECT COUNT(*) FROM model_universe
            WHERE inclusion_status='excluded'
              AND source='official_company_info:future_listing_boundary'
            """
        )
        or 0
    )
    metrics = [
        ("config_signature", config_signature),
        ("label_rule_version", LABEL_RULE_VERSION),
        ("research_start_date", start_date),
        ("research_end_date", end_date),
        ("training_universe_total", len(training_universe)),
        (
            "training_universe_twse",
            sum(1 for row in training_universe if row.get("market_type") == "twse"),
        ),
        (
            "training_universe_tpex",
            sum(1 for row in training_universe if row.get("market_type") == "tpex"),
        ),
        ("future_listing_boundary_excluded", future_listing_excluded),
        ("phase3_completed_stocks", len(completed)),
        ("phase3_pending_stocks", len(training_universe) - len(completed)),
        ("phase3_failed_stocks", len(failed)),
        (
            "feature_signal_rows",
            sum(int(row.get("total_rows") or 0) for row in completed),
        ),
        (
            "feature_ready_rows",
            sum(int(row.get("feature_ready_rows") or 0) for row in completed),
        ),
        (
            "eligible_5d_rows",
            sum(int(row.get("eligible_5d_rows") or 0) for row in completed),
        ),
        (
            "eligible_10d_rows",
            sum(int(row.get("eligible_10d_rows") or 0) for row in completed),
        ),
        (
            "eligible_20d_rows",
            sum(int(row.get("eligible_20d_rows") or 0) for row in completed),
        ),
        ("training_rows_twse", manifest_count.get("twse", 0)),
        ("training_rows_tpex", manifest_count.get("tpex", 0)),
        ("label_up_rows", label_totals["UP"]),
        ("label_flat_rows", label_totals["FLAT"]),
        ("label_down_rows", label_totals["DOWN"]),
        ("model_feature_count", len(FEATURE_COLUMNS)),
        ("dealer_hedging_feature_count", 0),
        ("duplicate_training_key_count", 0),
    ]
    return [{"metric": key, "value": value} for key, value in metrics]


def _gzip_csv_row_count(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    if rows:
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    else:
        fields = ["stock_id"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path
