from __future__ import annotations

import csv
import gzip
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research.institutional_model.database import ResearchDatabase
from research.institutional_model.phase3_dataset import (
    ALL_COLUMNS,
    LABEL_RULE_VERSION,
    _load_candidates,
    classify_10d_label,
    merge_phase3_training_files,
)


REPAIR_VERSION = "phase3c-v1"


@dataclass(frozen=True)
class Phase3LabelRepairResult:
    config_signature: str
    total_stocks: int
    repaired_stocks: int
    skipped_stocks: int
    failed_stocks: int
    changed_rows: int
    merged_rows_twse: int
    merged_rows_tpex: int
    output_paths: tuple[Path, ...]


def repair_phase3_labels(
    *,
    database: ResearchDatabase,
    output_dir: Path | str,
    shard_root: Path | str,
    start_date: str,
    end_date: str,
) -> Phase3LabelRepairResult:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    signature = database.get_metadata("phase3_config_signature")
    if not signature:
        raise RuntimeError("找不到 Phase 3 設定簽章，請先完成 Phase 3A。")

    shard_dir = Path(shard_root) / signature[:16]
    status_rows = [
        dict(row)
        for row in database.query(
            """
            SELECT stock_id, shard_name
            FROM phase3_build_status
            WHERE config_signature=? AND status='complete'
            ORDER BY stock_id
            """,
            (signature,),
        )
    ]
    if not status_rows:
        raise RuntimeError("找不到已完成的 Phase 3 分片，請先完成 Phase 3A。")

    repaired = 0
    skipped = 0
    failed = 0
    changed_this_run = 0
    for position, status in enumerate(status_rows, start=1):
        stock_id = str(status["stock_id"])
        shard_name = str(status.get("shard_name") or f"{stock_id}.csv.gz")
        shard_path = shard_dir / shard_name
        if _repair_is_complete(database, stock_id, signature, shard_path):
            skipped += 1
            print(f"[{position}/{len(status_rows)}] {stock_id} 已修正，略過。")
            continue

        print(f"[{position}/{len(status_rows)}] 修正 {stock_id} 標籤邊界")
        try:
            changed_rows, distribution = _repair_shard(shard_path, stock_id)
            _save_repair_result(
                database=database,
                stock_id=stock_id,
                signature=signature,
                changed_rows=changed_rows,
                distribution=distribution,
            )
            repaired += 1
            changed_this_run += changed_rows
            print(f"  完成：修正 {changed_rows} 列")
        except Exception as exc:
            failed += 1
            _save_repair_failure(database, stock_id, signature, str(exc))
            print(f"  失敗：{exc}")

    incomplete = int(
        database.scalar(
            """
            SELECT COUNT(*)
            FROM phase3_build_status b
            WHERE b.config_signature=? AND b.status='complete'
              AND NOT EXISTS (
                  SELECT 1
                  FROM phase3_label_repair_status r
                  WHERE r.stock_id=b.stock_id
                    AND r.config_signature=b.config_signature
                    AND r.repair_version=?
                    AND r.status='complete'
              )
            """,
            (signature, REPAIR_VERSION),
        )
        or 0
    )
    if incomplete or failed:
        raise RuntimeError(
            "Phase 3C 尚有分片未修正完成："
            f"未完成 {incomplete}、本次失敗 {failed}。重新執行可續接。"
        )

    changed_total = int(
        database.scalar(
            """
            SELECT COALESCE(SUM(changed_rows), 0)
            FROM phase3_label_repair_status
            WHERE config_signature=? AND repair_version=? AND status='complete'
            """,
            (signature, REPAIR_VERSION),
        )
        or 0
    )

    candidates = _load_candidates(database, start_date, end_date, None)
    merged_rows, merged_paths = merge_phase3_training_files(
        database=database,
        candidates=candidates,
        signature=signature,
        shard_dir=shard_dir,
        output_dir=output,
    )
    database.set_metadata("phase3_label_rule_version", LABEL_RULE_VERSION)

    summary_path = _write_repair_summary(
        output_dir=output,
        signature=signature,
        total_stocks=len(status_rows),
        repaired_stocks=repaired,
        skipped_stocks=skipped,
        failed_stocks=failed,
        changed_rows_this_run=changed_this_run,
        changed_rows_total=changed_total,
        merged_rows=merged_rows,
    )
    return Phase3LabelRepairResult(
        config_signature=signature,
        total_stocks=len(status_rows),
        repaired_stocks=repaired,
        skipped_stocks=skipped,
        failed_stocks=failed,
        changed_rows=changed_total,
        merged_rows_twse=merged_rows.get("twse", 0),
        merged_rows_tpex=merged_rows.get("tpex", 0),
        output_paths=tuple([*merged_paths, summary_path]),
    )


def create_phase3c_validation_archive(
    *,
    output_dir: Path | str,
    paths: list[Path],
) -> Path:
    output = Path(output_dir)
    archive = output / "phase3c_validation_reports.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in paths:
            if path.exists():
                bundle.write(path, arcname=path.name)
    return archive


def _repair_is_complete(
    database: ResearchDatabase,
    stock_id: str,
    signature: str,
    shard_path: Path,
) -> bool:
    if not shard_path.exists():
        return False
    return bool(
        database.scalar(
            """
            SELECT COUNT(*)
            FROM phase3_label_repair_status
            WHERE stock_id=? AND config_signature=?
              AND repair_version=? AND status='complete'
            """,
            (stock_id, signature, REPAIR_VERSION),
        )
    )


def _repair_shard(
    shard_path: Path,
    stock_id: str,
) -> tuple[int, Counter[tuple[int, str]]]:
    if not shard_path.exists():
        raise FileNotFoundError(f"找不到分片：{shard_path}")

    temporary = shard_path.with_suffix(shard_path.suffix + ".repair.tmp")
    changed_rows = 0
    distribution: Counter[tuple[int, str]] = Counter()
    try:
        with gzip.open(
            shard_path, "rt", encoding="utf-8-sig", newline=""
        ) as source, gzip.open(
            temporary, "wt", encoding="utf-8-sig", newline=""
        ) as target:
            reader = csv.DictReader(source)
            if reader.fieldnames != ALL_COLUMNS:
                raise ValueError(
                    f"分片欄位與 Phase 3A 不一致：{shard_path.name}"
                )
            writer = csv.DictWriter(target, fieldnames=ALL_COLUMNS)
            writer.writeheader()
            for row in reader:
                expected = _expected_label(row, stock_id)
                if row.get("label_10d", "") != expected:
                    row["label_10d"] = expected
                    changed_rows += 1
                if row.get("primary_exclusion_reason") == "eligible":
                    distribution[(int(row["signal_year"]), expected)] += 1
                writer.writerow(row)
        temporary.replace(shard_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return changed_rows, distribution


def _expected_label(row: dict[str, str], stock_id: str) -> str:
    if row.get("label_status_10d") != "ok":
        return ""
    value = row.get("adjusted_return_10d", "").strip()
    if not value:
        raise ValueError(
            f"{stock_id} {row.get('signal_date', '')} 10 日還原報酬為空"
        )
    return classify_10d_label(value)


def _save_repair_result(
    *,
    database: ResearchDatabase,
    stock_id: str,
    signature: str,
    changed_rows: int,
    distribution: Counter[tuple[int, str]],
) -> None:
    with database.connect() as connection:
        connection.execute(
            """
            DELETE FROM phase3_label_distribution
            WHERE stock_id=? AND config_signature=?
            """,
            (stock_id, signature),
        )
        connection.executemany(
            """
            INSERT INTO phase3_label_distribution (
                stock_id, config_signature, signal_year, label, sample_count
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                (stock_id, signature, year, label, count)
                for (year, label), count in sorted(distribution.items())
            ],
        )
        connection.execute(
            """
            INSERT INTO phase3_label_repair_status (
                stock_id, config_signature, repair_version,
                status, changed_rows, error, updated_at
            ) VALUES (?, ?, ?, 'complete', ?, NULL, CURRENT_TIMESTAMP)
            ON CONFLICT(stock_id, config_signature, repair_version) DO UPDATE SET
                status='complete',
                changed_rows=excluded.changed_rows,
                error=NULL,
                updated_at=CURRENT_TIMESTAMP
            """,
            (stock_id, signature, REPAIR_VERSION, changed_rows),
        )


def _save_repair_failure(
    database: ResearchDatabase,
    stock_id: str,
    signature: str,
    error: str,
) -> None:
    database.execute(
        """
        INSERT INTO phase3_label_repair_status (
            stock_id, config_signature, repair_version,
            status, changed_rows, error, updated_at
        ) VALUES (?, ?, ?, 'failed', 0, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(stock_id, config_signature, repair_version) DO UPDATE SET
            status='failed',
            changed_rows=0,
            error=excluded.error,
            updated_at=CURRENT_TIMESTAMP
        """,
        (stock_id, signature, REPAIR_VERSION, error[:2000]),
    )


def _write_repair_summary(
    *,
    output_dir: Path,
    signature: str,
    total_stocks: int,
    repaired_stocks: int,
    skipped_stocks: int,
    failed_stocks: int,
    changed_rows_this_run: int,
    changed_rows_total: int,
    merged_rows: dict[str, int],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "phase3c_label_repair_summary.csv"
    rows: list[dict[str, Any]] = [
        {"metric": "repair_version", "value": REPAIR_VERSION},
        {"metric": "label_rule_version", "value": LABEL_RULE_VERSION},
        {"metric": "config_signature", "value": signature},
        {"metric": "total_stocks", "value": total_stocks},
        {"metric": "repaired_stocks_this_run", "value": repaired_stocks},
        {"metric": "skipped_stocks_this_run", "value": skipped_stocks},
        {"metric": "failed_stocks_this_run", "value": failed_stocks},
        {
            "metric": "changed_label_rows_this_run",
            "value": changed_rows_this_run,
        },
        {"metric": "changed_label_rows_total", "value": changed_rows_total},
        {"metric": "training_rows_twse", "value": merged_rows.get("twse", 0)},
        {"metric": "training_rows_tpex", "value": merged_rows.get("tpex", 0)},
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("metric", "value"))
        writer.writeheader()
        writer.writerows(rows)
    return path
