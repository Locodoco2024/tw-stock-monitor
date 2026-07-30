from __future__ import annotations

import csv
import gzip
from datetime import date, timedelta
from pathlib import Path

from research.institutional_model.database import ResearchDatabase
from research.institutional_model.phase3_audit import run_phase3_quality_audit
from research.institutional_model.phase3_dataset import (
    ALL_COLUMNS,
    FEATURE_COLUMNS,
    sha256_file,
)


def test_phase3_audit_accepts_valid_training_files(tmp_path: Path) -> None:
    database, output = _prepare_files(tmp_path, duplicate_twse=False)

    result = run_phase3_quality_audit(
        database=database,
        output_dir=output,
        chunk_size=37,
        sample_size=120,
        correlation_threshold=0.9999,
    )

    assert result.ready_for_modeling is True
    assert result.error_count == 0
    assert result.total_rows == 260
    assert (output / "phase3b_validation_reports.zip").exists()


def test_phase3_audit_rejects_duplicate_training_key(tmp_path: Path) -> None:
    database, output = _prepare_files(tmp_path, duplicate_twse=True)

    result = run_phase3_quality_audit(
        database=database,
        output_dir=output,
        chunk_size=37,
        sample_size=120,
        correlation_threshold=0.9999,
    )

    assert result.ready_for_modeling is False
    assert result.error_count >= 1
    issues = (output / "phase3b_feature_issues.csv").read_text(
        encoding="utf-8-sig"
    )
    assert "duplicate_key_count" in issues


def _prepare_files(
    tmp_path: Path,
    *,
    duplicate_twse: bool,
) -> tuple[ResearchDatabase, Path]:
    output = tmp_path / "output"
    output.mkdir()
    database = ResearchDatabase(tmp_path / "research.sqlite")
    database.initialize()
    market_dates = [
        (date(2025, 1, 1) + timedelta(days=index)).isoformat()
        for index in range(150)
    ]
    database.executemany(
        "INSERT INTO market_calendar(date) VALUES (?)",
        [(value,) for value in market_dates],
    )

    manifest_rows = []
    for market in ("twse", "tpex"):
        path = output / f"phase3_training_{market}.csv.gz"
        rows = [
            _training_row(market, market_dates, index)
            for index in range(130)
        ]
        if market == "twse" and duplicate_twse:
            rows.insert(51, dict(rows[50]))
        with gzip.open(path, "wt", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=ALL_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        manifest_rows.append(
            {
                "file_name": path.name,
                "market_type": market,
                "row_count": len(rows),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "config_signature": "test",
            }
        )

    with (output / "phase3_dataset_manifest.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0])
        writer.writeheader()
        writer.writerows(manifest_rows)
    return database, output


def _training_row(
    market: str,
    market_dates: list[str],
    index: int,
) -> dict[str, object]:
    adjusted_return = (-0.06, 0.0, 0.06)[index % 3]
    label = "DOWN" if adjusted_return < 0 else "UP" if adjusted_return > 0 else "FLAT"
    row: dict[str, object] = {column: "" for column in ALL_COLUMNS}
    row.update(
        {
            "stock_id": "1000" if market == "twse" else "6000",
            "stock_name": "測試股票",
            "market_type": market,
            "signal_date": market_dates[index],
            "signal_year": int(market_dates[index][:4]),
            "entry_date_10d": market_dates[index + 1],
            "target_date_10d": market_dates[index + 10],
            "adjusted_return_10d": adjusted_return,
            "label_10d": label,
            "sample_eligible_10d": 1,
            "label_status_10d": "ok",
        }
    )
    for feature_index, feature in enumerate(FEATURE_COLUMNS):
        if "buy_day_ratio" in feature:
            value = ((index + feature_index) % 6) / 5
        elif "institutional_agreement" in feature:
            value = (-1, -1 / 3, 1 / 3, 1)[(index + feature_index) % 4]
        elif feature.endswith("_streak"):
            value = ((index * 3 + feature_index) % 21) - 10
        else:
            value = (((index + 1) * (feature_index + 3)) % 101 - 50) / 10
        row[feature] = value
    return row
