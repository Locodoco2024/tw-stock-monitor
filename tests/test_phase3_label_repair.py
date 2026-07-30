from __future__ import annotations

import csv
import gzip
from pathlib import Path

from research.institutional_model.database import ResearchDatabase
from research.institutional_model.phase3_audit import run_phase3_quality_audit
from research.institutional_model.phase3_dataset import (
    ALL_COLUMNS,
    FEATURE_COLUMNS,
    classify_10d_label,
)
from research.institutional_model.phase3_label_repair import repair_phase3_labels
from research.institutional_model.phase3_report import export_phase3_reports


def test_label_rule_uses_stored_ten_decimal_return() -> None:
    assert classify_10d_label(0.04999999996) == "UP"
    assert classify_10d_label(-0.04999999996) == "DOWN"
    assert classify_10d_label(0.04999999994) == "FLAT"
    assert classify_10d_label(-0.04999999994) == "FLAT"


def test_phase3_label_repair_updates_shards_and_rebuilds_outputs(
    tmp_path: Path,
) -> None:
    database = ResearchDatabase(tmp_path / "research.sqlite")
    database.initialize()
    signature = "a" * 64
    database.set_metadata("phase3_config_signature", signature)
    database.set_metadata("phase3_start_date", "2025-01-01")
    database.set_metadata("phase3_end_date", "2025-01-31")
    market_dates = [f"2025-01-{day:02d}" for day in range(1, 32)]
    database.executemany(
        "INSERT INTO market_calendar(date) VALUES (?)",
        [(value,) for value in market_dates],
    )

    shard_dir = tmp_path / "phase3_shards" / signature[:16]
    shard_dir.mkdir(parents=True)
    for stock_id, market in (("1000", "twse"), ("6000", "tpex")):
        database.execute(
            """
            INSERT INTO model_universe (
                stock_id, stock_name, market_type, listing_date,
                current_status, download_enabled, training_enabled,
                inclusion_status, inclusion_reason, source
            ) VALUES (?, ?, ?, '2025-01-01', 'active', 1, 1,
                      'included', 'test', 'test')
            """,
            (stock_id, f"測試{stock_id}", market),
        )
        database.execute(
            """
            INSERT INTO phase3_build_status (
                stock_id, config_signature, status, shard_name,
                total_rows, feature_ready_rows,
                eligible_5d_rows, eligible_10d_rows, eligible_20d_rows,
                first_signal_date, last_signal_date
            ) VALUES (?, ?, 'complete', ?, 3, 3, 3, 3, 3,
                      '2025-01-01', '2025-01-03')
            """,
            (stock_id, signature, f"{stock_id}.csv.gz"),
        )
        database.execute(
            """
            INSERT INTO phase3_label_distribution (
                stock_id, config_signature, signal_year, label, sample_count
            ) VALUES (?, ?, 2025, 'FLAT', 3)
            """,
            (stock_id, signature),
        )
        rows = [
            _row(stock_id, market, "2025-01-01", "0.0500000000", "FLAT"),
            _row(stock_id, market, "2025-01-02", "-0.0500000000", "FLAT"),
            _row(stock_id, market, "2025-01-03", "0.0499999999", "FLAT"),
        ]
        with gzip.open(
            shard_dir / f"{stock_id}.csv.gz",
            "wt",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=ALL_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    output = tmp_path / "output"
    result = repair_phase3_labels(
        database=database,
        output_dir=output,
        shard_root=tmp_path / "phase3_shards",
        start_date="2025-01-01",
        end_date="2025-01-31",
    )

    assert result.changed_rows == 4
    assert result.merged_rows_twse == 3
    assert result.merged_rows_tpex == 3
    labels = database.query(
        """
        SELECT stock_id, label, sample_count
        FROM phase3_label_distribution
        ORDER BY stock_id, label
        """
    )
    assert [(row["stock_id"], row["label"], row["sample_count"]) for row in labels] == [
        ("1000", "DOWN", 1),
        ("1000", "FLAT", 1),
        ("1000", "UP", 1),
        ("6000", "DOWN", 1),
        ("6000", "FLAT", 1),
        ("6000", "UP", 1),
    ]

    export_phase3_reports(
        database=database,
        output_dir=output,
        config_signature=signature,
        start_date="2025-01-01",
        end_date="2025-01-31",
    )
    audit = run_phase3_quality_audit(
        database=database,
        output_dir=output,
        chunk_size=2,
        sample_size=6,
        correlation_threshold=1.0,
    )
    assert audit.ready_for_modeling is True
    assert audit.error_count == 0

    second = repair_phase3_labels(
        database=database,
        output_dir=output,
        shard_root=tmp_path / "phase3_shards",
        start_date="2025-01-01",
        end_date="2025-01-31",
    )
    assert second.repaired_stocks == 0
    assert second.skipped_stocks == 2
    assert second.changed_rows == 4


def _row(
    stock_id: str,
    market: str,
    signal_date: str,
    adjusted_return: str,
    label: str,
) -> dict[str, object]:
    row: dict[str, object] = {column: "" for column in ALL_COLUMNS}
    row.update(
        {
            "stock_id": stock_id,
            "stock_name": f"測試{stock_id}",
            "market_type": market,
            "signal_date": signal_date,
            "signal_year": 2025,
            "listing_date": "2025-01-01",
            "current_status": "active",
            "feature_status": "ok",
            "entry_price_available": 1,
            "liquidity_pass_10m": 1,
            "liquidity_pass_20m": 1,
            "liquidity_pass_50m": 1,
            "liquidity_pass_100m": 1,
            "primary_exclusion_reason": "eligible",
            "entry_date_10d": f"2025-01-{int(signal_date[-2:]) + 1:02d}",
            "target_date_10d": f"2025-01-{int(signal_date[-2:]) + 10:02d}",
            "adjusted_return_10d": adjusted_return,
            "label_status_10d": "ok",
            "sample_eligible_10d": 1,
            "label_10d": label,
        }
    )
    for index, feature in enumerate(FEATURE_COLUMNS):
        row[feature] = (index + 1) / 100
    return row
