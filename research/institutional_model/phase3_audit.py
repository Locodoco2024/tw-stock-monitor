from __future__ import annotations

import csv
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.institutional_model.database import ResearchDatabase
from research.institutional_model.phase3_dataset import (
    ALL_COLUMNS,
    FEATURE_COLUMNS,
    LABEL_THRESHOLD,
    RETURN_ROUND_DIGITS,
    _add_features,
    _aligned_arrays,
    sha256_file,
)


AUDIT_VERSION = "phase3b-v1"
MARKETS = ("twse", "tpex")
VALID_LABELS = {"UP", "FLAT", "DOWN"}
FORBIDDEN_FEATURE_TERMS = (
    "label",
    "target",
    "entry",
    "return",
    "future",
    "price",
    "open",
    "close",
    "signal_year",
    "market_type",
    "stock_id",
)


@dataclass(frozen=True)
class Phase3AuditResult:
    status: str
    ready_for_modeling: bool
    total_rows: int
    error_count: int
    warning_count: int
    output_paths: tuple[Path, ...]


@dataclass
class NumericAccumulator:
    valid: np.ndarray
    missing: np.ndarray
    non_numeric: np.ndarray
    pos_inf: np.ndarray
    neg_inf: np.ndarray
    zero: np.ndarray
    total_sum: np.ndarray
    total_sumsq: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    abs_gt_100: np.ndarray
    abs_gt_500: np.ndarray
    abs_gt_1000: np.ndarray

    @classmethod
    def create(cls, width: int) -> "NumericAccumulator":
        zeros = lambda: np.zeros(width, dtype=np.int64)
        return cls(
            valid=zeros(),
            missing=zeros(),
            non_numeric=zeros(),
            pos_inf=zeros(),
            neg_inf=zeros(),
            zero=zeros(),
            total_sum=np.zeros(width, dtype=np.float64),
            total_sumsq=np.zeros(width, dtype=np.float64),
            minimum=np.full(width, np.inf, dtype=np.float64),
            maximum=np.full(width, -np.inf, dtype=np.float64),
            abs_gt_100=zeros(),
            abs_gt_500=zeros(),
            abs_gt_1000=zeros(),
        )

    def add(self, original: pd.DataFrame, numeric: pd.DataFrame) -> None:
        values = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
        original_missing = original.isna().to_numpy()
        numeric_nan = np.isnan(values)
        positive_infinity = np.isposinf(values)
        negative_infinity = np.isneginf(values)
        finite = np.isfinite(values)
        safe = np.where(finite, values, 0.0)

        self.valid += finite.sum(axis=0)
        self.missing += original_missing.sum(axis=0)
        self.non_numeric += ((~original_missing) & numeric_nan).sum(axis=0)
        self.pos_inf += positive_infinity.sum(axis=0)
        self.neg_inf += negative_infinity.sum(axis=0)
        self.zero += (finite & (values == 0)).sum(axis=0)
        self.total_sum += safe.sum(axis=0)
        self.total_sumsq += np.square(safe).sum(axis=0)
        self.abs_gt_100 += (finite & (np.abs(values) > 100)).sum(axis=0)
        self.abs_gt_500 += (finite & (np.abs(values) > 500)).sum(axis=0)
        self.abs_gt_1000 += (finite & (np.abs(values) > 1000)).sum(axis=0)

        chunk_min = np.min(np.where(finite, values, np.inf), axis=0)
        chunk_max = np.max(np.where(finite, values, -np.inf), axis=0)
        valid_min = np.isfinite(chunk_min)
        valid_max = np.isfinite(chunk_max)
        self.minimum[valid_min] = np.minimum(
            self.minimum[valid_min], chunk_min[valid_min]
        )
        self.maximum[valid_max] = np.maximum(
            self.maximum[valid_max], chunk_max[valid_max]
        )


@dataclass
class GroupAccumulator:
    count: np.ndarray
    total_sum: np.ndarray
    total_sumsq: np.ndarray

    @classmethod
    def create(cls, width: int) -> "GroupAccumulator":
        return cls(
            count=np.zeros(width, dtype=np.int64),
            total_sum=np.zeros(width, dtype=np.float64),
            total_sumsq=np.zeros(width, dtype=np.float64),
        )

    def add(self, values: np.ndarray) -> None:
        finite = np.isfinite(values)
        safe = np.where(finite, values, 0.0)
        self.count += finite.sum(axis=0)
        self.total_sum += safe.sum(axis=0)
        self.total_sumsq += np.square(safe).sum(axis=0)


def run_phase3_quality_audit(
    *,
    database: ResearchDatabase,
    output_dir: Path | str,
    chunk_size: int = 100_000,
    sample_size: int = 100_000,
    correlation_threshold: float = 0.995,
) -> Phase3AuditResult:
    if chunk_size <= 0:
        raise ValueError("audit chunk size 必須大於 0")
    if sample_size <= 0:
        raise ValueError("audit sample size 必須大於 0")
    if not 0 < correlation_threshold <= 1:
        raise ValueError("audit correlation threshold 必須介於 0 與 1")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(output / "phase3_dataset_manifest.csv")
    market_calendar = _load_market_calendar(database)

    feature_profiles: list[dict[str, Any]] = []
    key_audits: list[dict[str, Any]] = []
    all_samples: dict[str, pd.DataFrame] = {}
    yearly_groups: dict[tuple[str, int], GroupAccumulator] = {}
    label_groups: dict[tuple[str, str], GroupAccumulator] = {}
    issues: list[dict[str, Any]] = []
    total_rows = 0

    for market in MARKETS:
        path = output / f"phase3_training_{market}.csv.gz"
        if not path.exists():
            issues.append(
                _issue("ERROR", market, "missing_training_file", path.name, 1)
            )
            continue
        expected = manifest.get(path.name, {})
        print(f"Phase 3B 稽核 {market.upper()}：{path.name}")
        market_result = _audit_market_file(
            path=path,
            market=market,
            expected=expected,
            market_calendar=market_calendar,
            chunk_size=chunk_size,
            sample_size=sample_size,
        )
        total_rows += market_result["row_count"]
        feature_profiles.extend(market_result["feature_profiles"])
        key_audits.append(market_result["key_audit"])
        all_samples[market] = market_result["sample"]
        yearly_groups.update(market_result["yearly_groups"])
        label_groups.update(market_result["label_groups"])
        issues.extend(market_result["issues"])

    correlation_rows = _correlation_rows(all_samples, correlation_threshold)
    yearly_rows = _yearly_drift_rows(yearly_groups, feature_profiles)
    label_rows = _label_direction_rows(label_groups, all_samples)
    leakage_rows, leakage_issues = _leakage_audit()
    issues.extend(leakage_issues)
    issues.extend(_profile_issues(feature_profiles))
    issues.extend(_correlation_issues(correlation_rows))
    issues.extend(_drift_issues(yearly_rows))

    error_count = sum(row["severity"] == "ERROR" for row in issues)
    warning_count = sum(row["severity"] == "WARN" for row in issues)
    status = "FAIL" if error_count else ("PASS_WITH_WARNINGS" if warning_count else "PASS")
    ready = error_count == 0

    summary_rows = _summary_rows(
        status=status,
        ready=ready,
        total_rows=total_rows,
        feature_profiles=feature_profiles,
        key_audits=key_audits,
        correlation_rows=correlation_rows,
        yearly_rows=yearly_rows,
        leakage_rows=leakage_rows,
        issues=issues,
        all_samples=all_samples,
    )

    paths = [
        _write_csv(output / "phase3b_summary.csv", summary_rows),
        _write_csv(output / "phase3b_feature_profile.csv", feature_profiles),
        _write_csv(output / "phase3b_feature_issues.csv", issues),
        _write_csv(output / "phase3b_high_correlation.csv", correlation_rows),
        _write_csv(output / "phase3b_yearly_drift.csv", yearly_rows),
        _write_csv(output / "phase3b_label_feature_direction.csv", label_rows),
        _write_csv(output / "phase3b_key_audit.csv", key_audits),
        _write_csv(output / "phase3b_leakage_audit.csv", leakage_rows),
    ]
    archive = output / "phase3b_validation_reports.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in paths:
            handle.write(path, arcname=path.name)
    paths.append(archive)
    return Phase3AuditResult(
        status=status,
        ready_for_modeling=ready,
        total_rows=total_rows,
        error_count=error_count,
        warning_count=warning_count,
        output_paths=tuple(paths),
    )


def _audit_market_file(
    *,
    path: Path,
    market: str,
    expected: dict[str, str],
    market_calendar: dict[str, int],
    chunk_size: int,
    sample_size: int,
) -> dict[str, Any]:
    header = pd.read_csv(path, compression="gzip", nrows=0).columns.tolist()
    missing_columns = [column for column in ALL_COLUMNS if column not in header]
    issues: list[dict[str, Any]] = []
    if missing_columns:
        issues.append(
            _issue(
                "ERROR",
                market,
                "missing_columns",
                ",".join(missing_columns),
                len(missing_columns),
            )
        )
        return {
            "row_count": 0,
            "feature_profiles": [],
            "key_audit": _empty_key_audit(market, path.name),
            "sample": pd.DataFrame(),
            "yearly_groups": {},
            "label_groups": {},
            "issues": issues,
        }

    required = [
        "stock_id",
        "market_type",
        "signal_date",
        "signal_year",
        "label_10d",
        "adjusted_return_10d",
        "sample_eligible_10d",
        "label_status_10d",
        "entry_date_10d",
        "target_date_10d",
        *FEATURE_COLUMNS,
    ]
    accumulator = NumericAccumulator.create(len(FEATURE_COLUMNS))
    yearly_groups: dict[tuple[str, int], GroupAccumulator] = {}
    label_groups: dict[tuple[str, str], GroupAccumulator] = {}
    samples: list[pd.DataFrame] = []
    expected_rows = _int_or_zero(expected.get("row_count"))
    if not expected:
        issues.append(
            _issue("ERROR", market, "missing_manifest_entry", path.name, 1)
        )
    sampling_probability = (
        min(1.0, sample_size * 1.25 / expected_rows)
        if expected_rows > 0
        else 0.01
    )
    sample_threshold = int(np.iinfo(np.uint64).max * sampling_probability)

    row_count = 0
    duplicate_count = 0
    order_violation_count = 0
    invalid_market_count = 0
    invalid_label_count = 0
    invalid_year_count = 0
    ineligible_row_count = 0
    invalid_label_status_count = 0
    label_mismatch_count = 0
    signal_calendar_missing_count = 0
    entry_calendar_mismatch_count = 0
    target_calendar_mismatch_count = 0
    previous_key: tuple[str, str] | None = None
    next_progress = 500_000

    dtype = {
        "stock_id": "string",
        "market_type": "string",
        "signal_date": "string",
        "label_10d": "string",
        "entry_date_10d": "string",
        "target_date_10d": "string",
    }
    for chunk in pd.read_csv(
        path,
        compression="gzip",
        usecols=required,
        dtype=dtype,
        chunksize=chunk_size,
        low_memory=False,
    ):
        chunk = chunk.reset_index(drop=True)
        row_count += len(chunk)
        if row_count >= next_progress:
            if expected_rows:
                progress = min(100.0, row_count / expected_rows * 100)
                print(f"  已掃描 {row_count:,}/{expected_rows:,} 列（{progress:.1f}%）")
            else:
                print(f"  已掃描 {row_count:,} 列")
            while next_progress <= row_count:
                next_progress += 500_000
        original_features = chunk[FEATURE_COLUMNS]
        numeric_features = original_features.apply(pd.to_numeric, errors="coerce")
        accumulator.add(original_features, numeric_features)
        values = numeric_features.to_numpy(dtype=np.float64, na_value=np.nan)

        chunk_years = pd.to_numeric(chunk["signal_year"], errors="coerce")
        for year, positions in chunk_years.groupby(chunk_years).groups.items():
            if pd.isna(year):
                continue
            key = (market, int(year))
            yearly_groups.setdefault(
                key, GroupAccumulator.create(len(FEATURE_COLUMNS))
            ).add(values[np.asarray(list(positions), dtype=np.int64)])

        for label, positions in chunk["label_10d"].groupby(chunk["label_10d"]).groups.items():
            if pd.isna(label):
                continue
            key = (market, str(label))
            label_groups.setdefault(
                key, GroupAccumulator.create(len(FEATURE_COLUMNS))
            ).add(values[np.asarray(list(positions), dtype=np.int64)])

        stocks = chunk["stock_id"].fillna("").astype(str)
        dates = chunk["signal_date"].fillna("").astype(str)
        previous_stocks = stocks.shift(1)
        previous_dates = dates.shift(1)
        same_as_previous = stocks.eq(previous_stocks) & dates.eq(previous_dates)
        lower_than_previous = (stocks < previous_stocks) | (
            stocks.eq(previous_stocks) & (dates < previous_dates)
        )
        if previous_key is not None and len(chunk):
            first_key = (stocks.iloc[0], dates.iloc[0])
            if first_key == previous_key:
                duplicate_count += 1
            if first_key < previous_key:
                order_violation_count += 1
        duplicate_count += int(same_as_previous.iloc[1:].sum())
        order_violation_count += int(lower_than_previous.iloc[1:].sum())
        if len(chunk):
            previous_key = (stocks.iloc[-1], dates.iloc[-1])

        invalid_market_count += int(
            chunk["market_type"].fillna("").ne(market).sum()
        )
        invalid_label_count += int((~chunk["label_10d"].isin(VALID_LABELS)).sum())
        derived_year = pd.to_numeric(dates.str.slice(0, 4), errors="coerce")
        invalid_year_count += int((chunk_years != derived_year).fillna(True).sum())
        eligible = pd.to_numeric(chunk["sample_eligible_10d"], errors="coerce")
        ineligible_row_count += int((eligible != 1).fillna(True).sum())
        invalid_label_status_count += int((chunk["label_status_10d"] != "ok").sum())

        adjusted = pd.to_numeric(chunk["adjusted_return_10d"], errors="coerce")
        canonical_adjusted = adjusted.round(RETURN_ROUND_DIGITS)
        expected_label = np.where(
            canonical_adjusted >= LABEL_THRESHOLD,
            "UP",
            np.where(
                canonical_adjusted <= -LABEL_THRESHOLD,
                "DOWN",
                "FLAT",
            ),
        )
        label_mismatch_count += int(
            (adjusted.isna() | (chunk["label_10d"].to_numpy() != expected_label)).sum()
        )

        signal_indexes = dates.map(market_calendar)
        entry_indexes = chunk["entry_date_10d"].map(market_calendar)
        target_indexes = chunk["target_date_10d"].map(market_calendar)
        signal_calendar_missing_count += int(signal_indexes.isna().sum())
        entry_invalid = (
            signal_indexes.isna()
            | entry_indexes.isna()
            | (entry_indexes != signal_indexes + 1)
        )
        target_invalid = (
            entry_indexes.isna()
            | target_indexes.isna()
            | (target_indexes != entry_indexes + 9)
        )
        entry_calendar_mismatch_count += int(entry_invalid.sum())
        target_calendar_mismatch_count += int(target_invalid.sum())

        hashes = pd.util.hash_pandas_object(
            chunk[["stock_id", "signal_date"]], index=False
        ).to_numpy(dtype=np.uint64)
        selected = hashes <= sample_threshold
        if selected.any():
            sample = chunk.loc[selected, [
                "stock_id",
                "signal_date",
                "signal_year",
                "label_10d",
                *FEATURE_COLUMNS,
            ]].copy()
            sample["__audit_hash"] = hashes[selected]
            samples.append(sample)

    sample_frame = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
    if len(sample_frame) > sample_size:
        sample_frame = sample_frame.nsmallest(sample_size, "__audit_hash")
    if "__audit_hash" in sample_frame:
        sample_frame = sample_frame.drop(columns=["__audit_hash"])

    feature_profiles = _feature_profile_rows(
        market=market,
        row_count=row_count,
        accumulator=accumulator,
        sample=sample_frame,
    )
    print(f"  數值掃描完成：{row_count:,} 列；驗證 SHA-256。")
    actual_hash = sha256_file(path)
    manifest_hash = str(expected.get("sha256") or "")
    row_count_match = int(not expected_rows or expected_rows == row_count)
    hash_match = int(not manifest_hash or manifest_hash == actual_hash)

    key_audit = {
        "market_type": market,
        "file_name": path.name,
        "row_count": row_count,
        "manifest_row_count": expected_rows,
        "row_count_match": row_count_match,
        "manifest_sha256": manifest_hash,
        "actual_sha256": actual_hash,
        "sha256_match": hash_match,
        "duplicate_key_count": duplicate_count,
        "key_order_violation_count": order_violation_count,
        "invalid_market_count": invalid_market_count,
        "invalid_label_count": invalid_label_count,
        "invalid_signal_year_count": invalid_year_count,
        "ineligible_row_count": ineligible_row_count,
        "invalid_label_status_count": invalid_label_status_count,
        "label_mismatch_count": label_mismatch_count,
        "signal_calendar_missing_count": signal_calendar_missing_count,
        "entry_calendar_mismatch_count": entry_calendar_mismatch_count,
        "target_calendar_mismatch_count": target_calendar_mismatch_count,
        "sample_rows": len(sample_frame),
    }
    for field in (
        "row_count_match",
        "sha256_match",
    ):
        if not key_audit[field]:
            issues.append(_issue("ERROR", market, field, path.name, 1))
    for field in (
        "duplicate_key_count",
        "key_order_violation_count",
        "invalid_market_count",
        "invalid_label_count",
        "invalid_signal_year_count",
        "ineligible_row_count",
        "invalid_label_status_count",
        "label_mismatch_count",
        "signal_calendar_missing_count",
        "entry_calendar_mismatch_count",
        "target_calendar_mismatch_count",
    ):
        count = int(key_audit[field])
        if count:
            issues.append(_issue("ERROR", market, field, path.name, count))

    print(
        f"  {market.upper()} 稽核完成：抽樣 {len(sample_frame):,} 列、"
        f"重複鍵 {duplicate_count}、標籤錯置 {label_mismatch_count}。"
    )
    return {
        "row_count": row_count,
        "feature_profiles": feature_profiles,
        "key_audit": key_audit,
        "sample": sample_frame,
        "yearly_groups": yearly_groups,
        "label_groups": label_groups,
        "issues": issues,
    }


def _feature_profile_rows(
    *,
    market: str,
    row_count: int,
    accumulator: NumericAccumulator,
    sample: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_numeric = (
        sample[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        if not sample.empty
        else pd.DataFrame(columns=FEATURE_COLUMNS)
    )
    for index, feature in enumerate(FEATURE_COLUMNS):
        count = int(accumulator.valid[index])
        mean = accumulator.total_sum[index] / count if count else math.nan
        variance = (
            accumulator.total_sumsq[index] / count - mean * mean
            if count
            else math.nan
        )
        std = math.sqrt(max(variance, 0.0)) if count else math.nan
        series = sample_numeric[feature].replace([np.inf, -np.inf], np.nan).dropna()
        quantiles = (
            series.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
            if len(series)
            else pd.Series(dtype=float)
        )
        value_counts = series.value_counts(dropna=False) if len(series) else pd.Series(dtype=int)
        top_ratio = float(value_counts.iloc[0] / len(series)) if len(series) else math.nan
        rows.append(
            {
                "market_type": market,
                "feature": feature,
                "row_count": row_count,
                "valid_count": count,
                "missing_count": int(accumulator.missing[index]),
                "non_numeric_count": int(accumulator.non_numeric[index]),
                "positive_infinity_count": int(accumulator.pos_inf[index]),
                "negative_infinity_count": int(accumulator.neg_inf[index]),
                "zero_count": int(accumulator.zero[index]),
                "zero_ratio": _safe_ratio(accumulator.zero[index], count),
                "mean": _finite_or_blank(mean),
                "std": _finite_or_blank(std),
                "min": _finite_or_blank(accumulator.minimum[index]),
                "p01": _series_value(quantiles, 0.01),
                "p05": _series_value(quantiles, 0.05),
                "p25": _series_value(quantiles, 0.25),
                "p50": _series_value(quantiles, 0.5),
                "p75": _series_value(quantiles, 0.75),
                "p95": _series_value(quantiles, 0.95),
                "p99": _series_value(quantiles, 0.99),
                "max": _finite_or_blank(accumulator.maximum[index]),
                "sample_count": len(series),
                "sample_unique_count": int(series.nunique()) if len(series) else 0,
                "sample_top_value_ratio": _finite_or_blank(top_ratio),
                "abs_gt_100_count": int(accumulator.abs_gt_100[index]),
                "abs_gt_500_count": int(accumulator.abs_gt_500[index]),
                "abs_gt_1000_count": int(accumulator.abs_gt_1000[index]),
            }
        )
    return rows


def _correlation_rows(
    samples: dict[str, pd.DataFrame], threshold: float
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for market, sample in samples.items():
        if sample.empty:
            continue
        numeric = sample[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        correlation = numeric.corr(method="pearson", min_periods=100)
        for left_index, left in enumerate(FEATURE_COLUMNS):
            for right in FEATURE_COLUMNS[left_index + 1 :]:
                value = correlation.at[left, right]
                if pd.isna(value) or abs(float(value)) < threshold:
                    continue
                valid = numeric[[left, right]].dropna()
                equal_ratio = (
                    float(np.isclose(valid[left], valid[right], rtol=0, atol=1e-12).mean())
                    if len(valid)
                    else 0.0
                )
                rows.append(
                    {
                        "market_type": market,
                        "feature_a": left,
                        "feature_b": right,
                        "pearson_correlation": round(float(value), 8),
                        "absolute_correlation": round(abs(float(value)), 8),
                        "sample_count": len(valid),
                        "exact_value_match_ratio": round(equal_ratio, 8),
                    }
                )
    return sorted(
        rows,
        key=lambda row: (
            row["market_type"],
            -float(row["absolute_correlation"]),
            row["feature_a"],
            row["feature_b"],
        ),
    )


def _yearly_drift_rows(
    groups: dict[tuple[str, int], GroupAccumulator],
    profiles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    profile_map = {
        (str(row["market_type"]), str(row["feature"])): row for row in profiles
    }
    rows: list[dict[str, Any]] = []
    for (market, year), accumulator in sorted(groups.items()):
        for index, feature in enumerate(FEATURE_COLUMNS):
            count = int(accumulator.count[index])
            if not count:
                continue
            mean = accumulator.total_sum[index] / count
            variance = accumulator.total_sumsq[index] / count - mean * mean
            std = math.sqrt(max(variance, 0.0))
            profile = profile_map[(market, feature)]
            overall_mean = _float_or_nan(profile.get("mean"))
            overall_std = _float_or_nan(profile.get("std"))
            drift_z = (
                (mean - overall_mean) / overall_std
                if overall_std and math.isfinite(overall_std)
                else 0.0
            )
            rows.append(
                {
                    "market_type": market,
                    "signal_year": year,
                    "feature": feature,
                    "sample_count": count,
                    "year_mean": round(mean, 10),
                    "year_std": round(std, 10),
                    "overall_mean": profile.get("mean", ""),
                    "overall_std": profile.get("std", ""),
                    "mean_shift_z": round(drift_z, 8),
                    "absolute_mean_shift_z": round(abs(drift_z), 8),
                }
            )
    return rows


def _label_direction_rows(
    groups: dict[tuple[str, str], GroupAccumulator],
    samples: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    sample_medians: dict[tuple[str, str, str], float] = {}
    for market, sample in samples.items():
        if sample.empty:
            continue
        numeric = sample[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        labels = sample["label_10d"]
        for label in sorted(VALID_LABELS):
            positions = labels == label
            if not positions.any():
                continue
            medians = numeric.loc[positions].median(numeric_only=True)
            for feature in FEATURE_COLUMNS:
                sample_medians[(market, label, feature)] = float(medians[feature])

    rows: list[dict[str, Any]] = []
    for (market, label), accumulator in sorted(groups.items()):
        for index, feature in enumerate(FEATURE_COLUMNS):
            count = int(accumulator.count[index])
            if not count:
                continue
            mean = accumulator.total_sum[index] / count
            variance = accumulator.total_sumsq[index] / count - mean * mean
            rows.append(
                {
                    "market_type": market,
                    "label": label,
                    "feature": feature,
                    "sample_count": count,
                    "mean": round(mean, 10),
                    "std": round(math.sqrt(max(variance, 0.0)), 10),
                    "sample_median": round(
                        sample_medians.get((market, label, feature), math.nan), 10
                    ),
                }
            )
    return rows


def _leakage_audit() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    forbidden = [
        feature
        for feature in FEATURE_COLUMNS
        if any(term in feature.lower() for term in FORBIDDEN_FEATURE_TERMS)
    ]
    rows.append(
        {
            "check_name": "forbidden_future_or_metadata_feature_name",
            "status": "PASS" if not forbidden else "FAIL",
            "observed": ",".join(forbidden),
            "expected": "0 個禁止名稱",
        }
    )
    if forbidden:
        issues.append(
            _issue("ERROR", "all", "forbidden_feature_name", ",".join(forbidden), len(forbidden))
        )

    unexpected_count = len(FEATURE_COLUMNS) - len(set(FEATURE_COLUMNS))
    rows.append(
        {
            "check_name": "unique_model_feature_names",
            "status": "PASS" if unexpected_count == 0 else "FAIL",
            "observed": unexpected_count,
            "expected": 0,
        }
    )
    if unexpected_count:
        issues.append(_issue("ERROR", "all", "duplicate_feature_name", "", unexpected_count))

    future_invariance = _future_invariance_check()
    rows.append(
        {
            "check_name": "future_data_invariance",
            "status": "PASS" if future_invariance else "FAIL",
            "observed": int(future_invariance),
            "expected": 1,
        }
    )
    if not future_invariance:
        issues.append(_issue("ERROR", "all", "future_data_invariance", "", 1))

    invalid_windows = []
    for feature in FEATURE_COLUMNS:
        for token in feature.split("_"):
            if token.endswith("d") and token[:-1].isdigit() and int(token[:-1]) > 20:
                invalid_windows.append(feature)
    rows.append(
        {
            "check_name": "feature_lookback_not_over_20_market_days",
            "status": "PASS" if not invalid_windows else "FAIL",
            "observed": ",".join(invalid_windows),
            "expected": "最大回看 20 個市場交易日",
        }
    )
    if invalid_windows:
        issues.append(
            _issue(
                "ERROR",
                "all",
                "feature_lookback_over_20d",
                ",".join(invalid_windows),
                len(invalid_windows),
            )
        )

    rows.append(
        {
            "check_name": "dealer_hedging_excluded",
            "status": (
                "PASS"
                if not any("hedg" in feature.lower() for feature in FEATURE_COLUMNS)
                else "FAIL"
            ),
            "observed": sum("hedg" in feature.lower() for feature in FEATURE_COLUMNS),
            "expected": 0,
        }
    )
    if any("hedg" in feature.lower() for feature in FEATURE_COLUMNS):
        issues.append(_issue("ERROR", "all", "dealer_hedging_feature", "", 1))

    rows.append(
        {
            "check_name": "model_feature_count",
            "status": "PASS" if len(FEATURE_COLUMNS) == 40 else "FAIL",
            "observed": len(FEATURE_COLUMNS),
            "expected": 40,
        }
    )
    if len(FEATURE_COLUMNS) != 40:
        issues.append(_issue("ERROR", "all", "feature_count", "", len(FEATURE_COLUMNS)))
    return rows, issues


def _future_invariance_check() -> bool:
    market_dates = [f"2026-01-{day:02d}" for day in range(1, 61)]
    base_prices: dict[str, dict[str, Any]] = {}
    base_flows: dict[str, dict[str, Any]] = {}
    changed_prices: dict[str, dict[str, Any]] = {}
    changed_flows: dict[str, dict[str, Any]] = {}
    signal_index = 29
    for index, market_date in enumerate(market_dates):
        price = {
            "trading_volume": 1_000_000 + index,
            "trading_money": 50_000_000 + index,
            "open": 100 + index / 10,
            "high": 101 + index / 10,
            "low": 99 + index / 10,
            "close": 100 + index / 10,
        }
        flow = {
            "foreign_net": index * 100 - 500,
            "investment_trust_net": index * 10 - 50,
            "dealer_self_net": 100 - index * 5,
            "selected_total_net": index * 105 - 450,
        }
        base_prices[market_date] = dict(price)
        base_flows[market_date] = dict(flow)
        changed_prices[market_date] = dict(price)
        changed_flows[market_date] = dict(flow)
        if index > signal_index:
            changed_prices[market_date]["trading_volume"] = 999_999_999
            changed_prices[market_date]["trading_money"] = 9_999_999_999
            for key in changed_flows[market_date]:
                changed_flows[market_date][key] = 9_999_999_999

    first_arrays = _aligned_arrays(market_dates, base_prices, base_flows)
    second_arrays = _aligned_arrays(market_dates, changed_prices, changed_flows)
    first: dict[str, Any] = {}
    second: dict[str, Any] = {}
    _add_features(first, signal_index, first_arrays)
    _add_features(second, signal_index, second_arrays)
    return all(first[feature] == second[feature] for feature in FEATURE_COLUMNS)


def _profile_issues(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for row in profiles:
        market = str(row["market_type"])
        feature = str(row["feature"])
        for field in (
            "missing_count",
            "non_numeric_count",
            "positive_infinity_count",
            "negative_infinity_count",
        ):
            count = _int_or_zero(row.get(field))
            if count:
                issues.append(_issue("ERROR", market, field, feature, count))

        minimum = _float_or_nan(row.get("min"))
        maximum = _float_or_nan(row.get("max"))
        if "buy_day_ratio" in feature and (
            (math.isfinite(minimum) and minimum < 0)
            or (math.isfinite(maximum) and maximum > 1)
        ):
            issues.append(_issue("ERROR", market, "ratio_out_of_range", feature, 1))
        if "institutional_agreement" in feature and (
            (math.isfinite(minimum) and minimum < -1)
            or (math.isfinite(maximum) and maximum > 1)
        ):
            issues.append(_issue("ERROR", market, "agreement_out_of_range", feature, 1))

        unique_count = _int_or_zero(row.get("sample_unique_count"))
        top_ratio = _float_or_nan(row.get("sample_top_value_ratio"))
        if unique_count <= 1:
            issues.append(_issue("WARN", market, "constant_feature", feature, unique_count))
        elif math.isfinite(top_ratio) and top_ratio >= 0.995:
            issues.append(
                _issue(
                    "WARN",
                    market,
                    "near_constant_feature",
                    feature,
                    round(top_ratio, 8),
                )
            )

        extreme_count = _int_or_zero(row.get("abs_gt_1000_count"))
        valid_count = max(1, _int_or_zero(row.get("valid_count")))
        if "flow_pct" in feature and extreme_count / valid_count >= 0.0001:
            issues.append(
                _issue(
                    "WARN",
                    market,
                    "extreme_flow_pct_gt_1000",
                    feature,
                    extreme_count,
                )
            )
    return issues


def _correlation_issues(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _issue(
            "WARN",
            str(row["market_type"]),
            "high_feature_correlation",
            f"{row['feature_a']}|{row['feature_b']}",
            row["absolute_correlation"],
        )
        for row in rows
    ]


def _drift_issues(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for row in rows:
        shift = abs(float(row["mean_shift_z"]))
        if int(row["sample_count"]) >= 1_000 and shift >= 2.0:
            issues.append(
                _issue(
                    "WARN",
                    str(row["market_type"]),
                    "yearly_mean_shift_ge_2std",
                    f"{row['signal_year']}|{row['feature']}",
                    row["mean_shift_z"],
                )
            )
    return issues


def _summary_rows(
    *,
    status: str,
    ready: bool,
    total_rows: int,
    feature_profiles: list[dict[str, Any]],
    key_audits: list[dict[str, Any]],
    correlation_rows: list[dict[str, Any]],
    yearly_rows: list[dict[str, Any]],
    leakage_rows: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    all_samples: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    errors = sum(row["severity"] == "ERROR" for row in issues)
    warnings = sum(row["severity"] == "WARN" for row in issues)
    metrics = [
        ("audit_version", AUDIT_VERSION),
        ("status", status),
        ("ready_for_modeling", int(ready)),
        ("training_rows_total", total_rows),
        ("market_files_audited", len(key_audits)),
        ("model_feature_count", len(FEATURE_COLUMNS)),
        ("error_count", errors),
        ("warning_count", warnings),
        (
            "missing_or_non_numeric_feature_values",
            sum(
                _int_or_zero(row.get("missing_count"))
                + _int_or_zero(row.get("non_numeric_count"))
                for row in feature_profiles
            ),
        ),
        (
            "infinite_feature_values",
            sum(
                _int_or_zero(row.get("positive_infinity_count"))
                + _int_or_zero(row.get("negative_infinity_count"))
                for row in feature_profiles
            ),
        ),
        (
            "duplicate_training_keys",
            sum(_int_or_zero(row.get("duplicate_key_count")) for row in key_audits),
        ),
        (
            "label_mismatches",
            sum(_int_or_zero(row.get("label_mismatch_count")) for row in key_audits),
        ),
        (
            "calendar_alignment_mismatches",
            sum(
                _int_or_zero(row.get("entry_calendar_mismatch_count"))
                + _int_or_zero(row.get("target_calendar_mismatch_count"))
                for row in key_audits
            ),
        ),
        ("high_correlation_pairs", len(correlation_rows)),
        (
            "yearly_drift_rows_ge_2std",
            sum(abs(float(row["mean_shift_z"])) >= 2.0 for row in yearly_rows),
        ),
        (
            "leakage_checks_failed",
            sum(row["status"] != "PASS" for row in leakage_rows),
        ),
        ("audit_sample_rows_twse", len(all_samples.get("twse", []))),
        ("audit_sample_rows_tpex", len(all_samples.get("tpex", []))),
    ]
    return [{"metric": key, "value": value} for key, value in metrics]


def _load_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            str(row.get("file_name") or ""): row for row in csv.DictReader(handle)
        }


def _load_market_calendar(database: ResearchDatabase) -> dict[str, int]:
    rows = database.query("SELECT date FROM market_calendar ORDER BY date")
    if not rows:
        raise RuntimeError("市場交易日曆為空，無法驗證 T+1 與第 10 市場交易日。")
    return {str(row["date"]): index for index, row in enumerate(rows)}


def _empty_key_audit(market: str, file_name: str) -> dict[str, Any]:
    return {
        "market_type": market,
        "file_name": file_name,
        "row_count": 0,
        "manifest_row_count": 0,
        "row_count_match": 0,
        "manifest_sha256": "",
        "actual_sha256": "",
        "sha256_match": 0,
        "duplicate_key_count": 0,
        "key_order_violation_count": 0,
        "invalid_market_count": 0,
        "invalid_label_count": 0,
        "invalid_signal_year_count": 0,
        "ineligible_row_count": 0,
        "invalid_label_status_count": 0,
        "label_mismatch_count": 0,
        "signal_calendar_missing_count": 0,
        "entry_calendar_mismatch_count": 0,
        "target_calendar_mismatch_count": 0,
        "sample_rows": 0,
    }


def _issue(
    severity: str,
    market: str,
    issue_type: str,
    subject: str,
    observed: Any,
) -> dict[str, Any]:
    return {
        "severity": severity,
        "market_type": market,
        "issue_type": issue_type,
        "subject": subject,
        "observed": observed,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["status"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _safe_ratio(numerator: int | np.integer, denominator: int) -> float:
    return round(float(numerator) / denominator, 10) if denominator else 0.0


def _series_value(series: pd.Series, key: float) -> float | str:
    if key not in series:
        return ""
    return _finite_or_blank(float(series.loc[key]))


def _finite_or_blank(value: float) -> float | str:
    if not math.isfinite(float(value)):
        return ""
    return round(float(value), 10)


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
