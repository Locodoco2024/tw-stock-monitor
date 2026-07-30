from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from research.institutional_model.phase3_dataset import FEATURE_COLUMNS


PHASE4_VERSION = "phase4a-v1"
LABELS = ("DOWN", "FLAT", "UP")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
MARKETS = ("twse", "tpex")
EPSILON = 1e-12


@dataclass(frozen=True)
class Phase4Settings:
    first_test_year: int = 2019
    minimum_training_years: int = 3
    calibration_years: int = 1
    clip_lower_quantile: float = 0.005
    clip_upper_quantile: float = 0.995
    quantile_sample_size: int = 250_000
    cache_chunk_size: int = 100_000
    training_chunk_size: int = 100_000
    batch_size: int = 65_536
    maximum_epochs: int = 6
    minimum_epochs: int = 3
    early_stopping_patience: int = 2
    learning_rate: float = 0.02
    l2_penalty: float = 0.0001
    random_seed: int = 20260728

    def validate(self) -> None:
        if self.minimum_training_years < 2:
            raise ValueError("Phase 4 最少訓練年度必須至少為 2")
        if self.calibration_years < 1:
            raise ValueError("Phase 4 校準年度必須至少為 1")
        if not 0 <= self.clip_lower_quantile < self.clip_upper_quantile <= 1:
            raise ValueError("Phase 4 截尾分位數設定無效")
        if self.quantile_sample_size <= 0:
            raise ValueError("Phase 4 分位數抽樣筆數必須大於 0")
        if self.cache_chunk_size <= 0 or self.training_chunk_size <= 0:
            raise ValueError("Phase 4 chunk size 必須大於 0")
        if self.batch_size <= 0:
            raise ValueError("Phase 4 batch size 必須大於 0")
        if self.maximum_epochs < self.minimum_epochs or self.minimum_epochs <= 0:
            raise ValueError("Phase 4 epoch 設定無效")
        if self.early_stopping_patience < 1:
            raise ValueError("Phase 4 early stopping patience 必須大於 0")
        if self.learning_rate <= 0:
            raise ValueError("Phase 4 learning rate 必須大於 0")
        if self.l2_penalty < 0:
            raise ValueError("Phase 4 L2 penalty 不可小於 0")


@dataclass(frozen=True)
class FoldSpec:
    market: str
    train_start_year: int
    train_end_year: int
    calibration_start_year: int
    calibration_end_year: int
    test_year: int

    @property
    def fold_id(self) -> str:
        return f"{self.market}_{self.test_year}"


@dataclass(frozen=True)
class Preprocessor:
    lower: np.ndarray
    upper: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    sampled_rows: int
    training_rows: int

    def transform(self, values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, self.lower, self.upper)
        return (clipped - self.mean) / self.std


@dataclass(frozen=True)
class SoftmaxModel:
    weights: np.ndarray
    intercept: np.ndarray

    def logits(self, features: np.ndarray) -> np.ndarray:
        return features @ self.weights + self.intercept


@dataclass(frozen=True)
class MarketCache:
    market: str
    signature: str
    row_count: int
    source_sha256: str
    directory: Path
    features: np.ndarray
    labels: np.ndarray
    years: np.ndarray
    returns: np.ndarray


@dataclass(frozen=True)
class Phase4Result:
    status: str
    ready_for_phase4b: bool
    completed_folds: int
    expected_folds: int
    failed_folds: int
    output_paths: tuple[Path, ...]


def run_phase4a_rolling_baseline(
    *,
    output_dir: Path | str,
    cache_root: Path | str,
    run_root: Path | str,
    settings: Phase4Settings | None = None,
    markets: Iterable[str] = MARKETS,
    force: bool = False,
) -> Phase4Result:
    config = settings or Phase4Settings()
    config.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache_base = Path(cache_root)
    run_base = Path(run_root)
    manifest = _load_phase3_manifest(output / "phase3_dataset_manifest.csv")
    _ensure_phase3_ready(output / "phase3b_summary.csv")

    selected_markets = tuple(dict.fromkeys(str(value).lower() for value in markets))
    unknown = set(selected_markets).difference(MARKETS)
    if unknown:
        raise ValueError(f"不支援的 Phase 4 市場：{sorted(unknown)}")

    caches: dict[str, MarketCache] = {}
    expected_specs: list[FoldSpec] = []
    for market in selected_markets:
        cache = build_or_load_market_cache(
            output_dir=output,
            cache_root=cache_base,
            manifest=manifest,
            market=market,
            chunk_size=config.cache_chunk_size,
            force=force,
        )
        caches[market] = cache
        specs = build_rolling_folds(
            market=market,
            years=np.asarray(cache.years),
            first_test_year=config.first_test_year,
            minimum_training_years=config.minimum_training_years,
            calibration_years=config.calibration_years,
        )
        if not specs:
            raise RuntimeError(f"{market.upper()} 沒有足夠年度建立 Phase 4 滾動折")
        expected_specs.extend(specs)

    run_signature = phase4_config_signature(config, caches)
    run_dir = run_base / run_signature[:16]
    fold_dir = run_dir / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(
        run_dir / "run_config.json",
        {
            "phase4_version": PHASE4_VERSION,
            "run_signature": run_signature,
            "settings": asdict(config),
            "markets": list(selected_markets),
            "cache_signatures": {
                market: cache.signature for market, cache in caches.items()
            },
            "expected_folds": [asdict(spec) for spec in expected_specs],
        },
    )

    failed_folds = 0
    for position, spec in enumerate(expected_specs, start=1):
        result_path = fold_dir / f"{spec.fold_id}.json.gz"
        if result_path.exists() and not force:
            print(
                f"[{position}/{len(expected_specs)}] {spec.fold_id} 已完成，"
                "略過並沿用既有折結果。"
            )
            continue
        print(
            f"[{position}/{len(expected_specs)}] Phase 4A {spec.market.upper()} "
            f"測試 {spec.test_year}：訓練 {spec.train_start_year}-{spec.train_end_year}、"
            f"校準 {spec.calibration_start_year}-{spec.calibration_end_year}"
        )
        try:
            fold_result = run_single_fold(
                cache=caches[spec.market],
                spec=spec,
                settings=config,
            )
            _write_gzip_json_atomic(result_path, fold_result)
        except Exception as exc:
            failed_folds += 1
            print(f"  折執行失敗：{type(exc).__name__}: {exc}")
            _write_gzip_json_atomic(
                fold_dir / f"{spec.fold_id}.error.json.gz",
                {
                    "fold": asdict(spec),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "failed_at": datetime.now().isoformat(timespec="seconds"),
                },
            )

    fold_results = []
    for spec in expected_specs:
        result_path = fold_dir / f"{spec.fold_id}.json.gz"
        if result_path.exists():
            fold_results.append(_read_gzip_json(result_path))

    paths, ready = export_phase4a_reports(
        output_dir=output,
        run_signature=run_signature,
        settings=config,
        expected_specs=expected_specs,
        fold_results=fold_results,
        caches=caches,
    )
    completed = len(fold_results)
    status = "PASS" if completed == len(expected_specs) and failed_folds == 0 else "FAIL"
    return Phase4Result(
        status=status,
        ready_for_phase4b=ready and status == "PASS",
        completed_folds=completed,
        expected_folds=len(expected_specs),
        failed_folds=failed_folds,
        output_paths=tuple(paths),
    )


def build_or_load_market_cache(
    *,
    output_dir: Path,
    cache_root: Path,
    manifest: dict[str, dict[str, str]],
    market: str,
    chunk_size: int,
    force: bool,
) -> MarketCache:
    file_name = f"phase3_training_{market}.csv.gz"
    source_path = output_dir / file_name
    if not source_path.exists():
        raise FileNotFoundError(f"找不到 Phase 3 訓練檔：{source_path}")
    manifest_row = manifest.get(file_name)
    if not manifest_row:
        raise RuntimeError(f"Phase 3 manifest 缺少 {file_name}")
    row_count = int(manifest_row["row_count"])
    source_sha256 = str(manifest_row["sha256"])
    signature = hashlib.sha256(
        json.dumps(
            {
                "phase4_version": PHASE4_VERSION,
                "market": market,
                "source_sha256": source_sha256,
                "row_count": row_count,
                "features": FEATURE_COLUMNS,
                "labels": LABELS,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    directory = cache_root / signature[:16] / market
    metadata_path = directory / "metadata.json"
    paths = {
        "features": directory / "features.npy",
        "labels": directory / "labels.npy",
        "years": directory / "years.npy",
        "returns": directory / "returns.npy",
    }
    if not force and _cache_is_complete(
        metadata_path=metadata_path,
        paths=paths,
        signature=signature,
        row_count=row_count,
    ):
        return _load_market_cache(
            market=market,
            signature=signature,
            row_count=row_count,
            source_sha256=source_sha256,
            directory=directory,
            paths=paths,
        )

    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = {name: path.with_suffix(".partial.npy") for name, path in paths.items()}
    feature_map = np.lib.format.open_memmap(
        temporary["features"],
        mode="w+",
        dtype=np.float32,
        shape=(row_count, len(FEATURE_COLUMNS)),
    )
    label_map = np.lib.format.open_memmap(
        temporary["labels"], mode="w+", dtype=np.uint8, shape=(row_count,)
    )
    year_map = np.lib.format.open_memmap(
        temporary["years"], mode="w+", dtype=np.int16, shape=(row_count,)
    )
    return_map = np.lib.format.open_memmap(
        temporary["returns"], mode="w+", dtype=np.float32, shape=(row_count,)
    )

    required = ["signal_year", "label_10d", "adjusted_return_10d", *FEATURE_COLUMNS]
    offset = 0
    print(f"建立 Phase 4 快取 {market.upper()}：{row_count:,} 列")
    for chunk in pd.read_csv(
        source_path,
        compression="gzip",
        usecols=required,
        chunksize=chunk_size,
        low_memory=False,
    ):
        size = len(chunk)
        end = offset + size
        if end > row_count:
            raise RuntimeError(f"{file_name} 實際筆數超過 manifest")
        feature_values = chunk[FEATURE_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float32)
        if not np.isfinite(feature_values).all():
            raise RuntimeError(f"{file_name} 包含無效模型特徵，請先重跑 Phase 3B")
        labels = chunk["label_10d"].map(LABEL_TO_INDEX)
        if labels.isna().any():
            invalid = sorted(set(chunk.loc[labels.isna(), "label_10d"].astype(str)))
            raise RuntimeError(f"{file_name} 包含無效標籤：{invalid}")
        years = pd.to_numeric(chunk["signal_year"], errors="coerce")
        returns = pd.to_numeric(chunk["adjusted_return_10d"], errors="coerce")
        if years.isna().any() or returns.isna().any():
            raise RuntimeError(f"{file_name} 包含無效年度或 10 日報酬")
        feature_map[offset:end] = feature_values
        label_map[offset:end] = labels.to_numpy(dtype=np.uint8)
        year_map[offset:end] = years.to_numpy(dtype=np.int16)
        return_map[offset:end] = returns.to_numpy(dtype=np.float32)
        offset = end
        if offset % 500_000 < size:
            print(f"  已快取 {offset:,}/{row_count:,} 列")

    if offset != row_count:
        raise RuntimeError(
            f"{file_name} 實際筆數 {offset:,} 與 manifest {row_count:,} 不一致"
        )
    del feature_map, label_map, year_map, return_map
    for name, final_path in paths.items():
        os.replace(temporary[name], final_path)
    _write_json_atomic(
        metadata_path,
        {
            "phase4_version": PHASE4_VERSION,
            "signature": signature,
            "market": market,
            "source_file": file_name,
            "source_sha256": source_sha256,
            "row_count": row_count,
            "feature_count": len(FEATURE_COLUMNS),
            "feature_columns": FEATURE_COLUMNS,
            "labels": LABELS,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    return _load_market_cache(
        market=market,
        signature=signature,
        row_count=row_count,
        source_sha256=source_sha256,
        directory=directory,
        paths=paths,
    )


def _cache_is_complete(
    *,
    metadata_path: Path,
    paths: dict[str, Path],
    signature: str,
    row_count: int,
) -> bool:
    if not metadata_path.exists() or not all(path.exists() for path in paths.values()):
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") != signature:
            return False
        if int(metadata.get("row_count") or 0) != row_count:
            return False
        features = np.load(paths["features"], mmap_mode="r")
        labels = np.load(paths["labels"], mmap_mode="r")
        years = np.load(paths["years"], mmap_mode="r")
        returns = np.load(paths["returns"], mmap_mode="r")
        return (
            features.shape == (row_count, len(FEATURE_COLUMNS))
            and labels.shape == (row_count,)
            and years.shape == (row_count,)
            and returns.shape == (row_count,)
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _load_market_cache(
    *,
    market: str,
    signature: str,
    row_count: int,
    source_sha256: str,
    directory: Path,
    paths: dict[str, Path],
) -> MarketCache:
    return MarketCache(
        market=market,
        signature=signature,
        row_count=row_count,
        source_sha256=source_sha256,
        directory=directory,
        features=np.load(paths["features"], mmap_mode="r"),
        labels=np.load(paths["labels"], mmap_mode="r"),
        years=np.load(paths["years"], mmap_mode="r"),
        returns=np.load(paths["returns"], mmap_mode="r"),
    )


def build_rolling_folds(
    *,
    market: str,
    years: np.ndarray,
    first_test_year: int,
    minimum_training_years: int,
    calibration_years: int,
) -> list[FoldSpec]:
    available = sorted(int(value) for value in np.unique(years))
    if not available:
        return []
    first_year = available[0]
    earliest_test = first_year + minimum_training_years + calibration_years
    start = max(first_test_year, earliest_test)
    folds: list[FoldSpec] = []
    available_set = set(available)
    for test_year in range(start, available[-1] + 1):
        calibration_start = test_year - calibration_years
        calibration_year_set = set(range(calibration_start, test_year))
        train_end = calibration_start - 1
        train_year_set = set(range(first_year, train_end + 1))
        if test_year not in available_set:
            continue
        if not calibration_year_set.issubset(available_set):
            continue
        if len(train_year_set.intersection(available_set)) < minimum_training_years:
            continue
        folds.append(
            FoldSpec(
                market=market,
                train_start_year=first_year,
                train_end_year=train_end,
                calibration_start_year=calibration_start,
                calibration_end_year=test_year - 1,
                test_year=test_year,
            )
        )
    return folds


def run_single_fold(
    *,
    cache: MarketCache,
    spec: FoldSpec,
    settings: Phase4Settings,
) -> dict[str, Any]:
    years = np.asarray(cache.years)
    labels = np.asarray(cache.labels)
    train_mask = (years >= spec.train_start_year) & (years <= spec.train_end_year)
    calibration_mask = (years >= spec.calibration_start_year) & (
        years <= spec.calibration_end_year
    )
    test_mask = years == spec.test_year
    train_count = int(train_mask.sum())
    calibration_count = int(calibration_mask.sum())
    test_count = int(test_mask.sum())
    if min(train_count, calibration_count, test_count) == 0:
        raise RuntimeError(
            f"{spec.fold_id} 切分為空：train={train_count}, "
            f"calibration={calibration_count}, test={test_count}"
        )
    _ensure_all_classes(labels[train_mask], spec.fold_id, "train")
    _ensure_all_classes(labels[calibration_mask], spec.fold_id, "calibration")
    _ensure_all_classes(labels[test_mask], spec.fold_id, "test")

    seed = settings.random_seed + spec.test_year + (0 if spec.market == "twse" else 10_000)
    preprocessor = fit_preprocessor(
        features=cache.features,
        years=cache.years,
        train_start_year=spec.train_start_year,
        train_end_year=spec.train_end_year,
        sample_size=settings.quantile_sample_size,
        lower_quantile=settings.clip_lower_quantile,
        upper_quantile=settings.clip_upper_quantile,
        chunk_size=settings.training_chunk_size,
        seed=seed,
    )
    model, history, train_priors = train_softmax_model(
        features=cache.features,
        labels=cache.labels,
        years=cache.years,
        preprocessor=preprocessor,
        spec=spec,
        settings=settings,
        seed=seed,
    )

    calibration_logits, calibration_labels, _ = collect_split_predictions(
        cache=cache,
        mask=calibration_mask,
        preprocessor=preprocessor,
        model=model,
        chunk_size=settings.training_chunk_size,
    )
    temperature, calibration_raw_loss, calibration_scaled_loss = fit_temperature(
        calibration_logits, calibration_labels
    )
    test_logits, test_labels, test_returns = collect_split_predictions(
        cache=cache,
        mask=test_mask,
        preprocessor=preprocessor,
        model=model,
        chunk_size=settings.training_chunk_size,
    )
    raw_probabilities = softmax(test_logits)
    calibrated_probabilities = softmax(test_logits / temperature)
    historical_probabilities = np.tile(train_priors, (test_count, 1))
    majority_index = int(np.argmax(train_priors))
    majority_probabilities = np.full((test_count, len(LABELS)), 1e-6, dtype=np.float64)
    majority_probabilities[:, majority_index] = 1 - 2e-6

    metric_rows = []
    for model_name, probabilities in (
        ("multinomial_logistic_raw", raw_probabilities),
        ("multinomial_logistic_calibrated", calibrated_probabilities),
        ("historical_probability", historical_probabilities),
        ("majority_class", majority_probabilities),
    ):
        metrics = classification_metrics(test_labels, probabilities)
        metric_rows.append(
            {
                "market_type": spec.market,
                "test_year": spec.test_year,
                "model": model_name,
                **metrics,
            }
        )

    decile_rows, monotonicity = institutional_index_deciles(
        market=spec.market,
        test_year=spec.test_year,
        probabilities=calibrated_probabilities,
        labels=test_labels,
        adjusted_returns=test_returns,
    )
    calibration_rows = calibration_report_rows(
        market=spec.market,
        test_year=spec.test_year,
        raw_probabilities=raw_probabilities,
        calibrated_probabilities=calibrated_probabilities,
        labels=test_labels,
    )
    coefficient_rows = coefficient_report_rows(
        market=spec.market,
        test_year=spec.test_year,
        model=model,
    )
    preprocessing_rows = preprocessing_report_rows(
        market=spec.market,
        test_year=spec.test_year,
        preprocessor=preprocessor,
    )
    for row in history:
        row.update({"market_type": spec.market, "test_year": spec.test_year})

    calibrated_metrics = next(
        row for row in metric_rows if row["model"] == "multinomial_logistic_calibrated"
    )
    historical_metrics = next(
        row for row in metric_rows if row["model"] == "historical_probability"
    )
    majority_metrics = next(row for row in metric_rows if row["model"] == "majority_class")
    fold_summary = {
        **asdict(spec),
        "train_rows": train_count,
        "calibration_rows": calibration_count,
        "test_rows": test_count,
        "train_down_rate": float(train_priors[0]),
        "train_flat_rate": float(train_priors[1]),
        "train_up_rate": float(train_priors[2]),
        "temperature": float(temperature),
        "calibration_raw_log_loss": float(calibration_raw_loss),
        "calibration_scaled_log_loss": float(calibration_scaled_loss),
        "epochs_completed": len(history),
        "test_log_loss": calibrated_metrics["log_loss"],
        "historical_log_loss": historical_metrics["log_loss"],
        "log_loss_improvement": historical_metrics["log_loss"]
        - calibrated_metrics["log_loss"],
        "test_macro_f1": calibrated_metrics["macro_f1"],
        "majority_macro_f1": majority_metrics["macro_f1"],
        "test_balanced_accuracy": calibrated_metrics["balanced_accuracy"],
        "test_ece": calibrated_metrics["ece"],
        **monotonicity,
    }
    return {
        "phase4_version": PHASE4_VERSION,
        "fold": asdict(spec),
        "fold_summary": fold_summary,
        "metrics": metric_rows,
        "deciles": decile_rows,
        "calibration": calibration_rows,
        "coefficients": coefficient_rows,
        "preprocessing": preprocessing_rows,
        "training_history": history,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }


def fit_preprocessor(
    *,
    features: np.ndarray,
    years: np.ndarray,
    train_start_year: int,
    train_end_year: int,
    sample_size: int,
    lower_quantile: float,
    upper_quantile: float,
    chunk_size: int,
    seed: int,
) -> Preprocessor:
    year_values = np.asarray(years)
    train_positions = np.flatnonzero(
        (year_values >= train_start_year) & (year_values <= train_end_year)
    )
    if train_positions.size == 0:
        raise RuntimeError("Phase 4 訓練折沒有資料")
    rng = np.random.default_rng(seed)
    sample_count = min(sample_size, train_positions.size)
    selected = np.sort(rng.choice(train_positions, size=sample_count, replace=False))
    sample = np.asarray(features[selected], dtype=np.float64)
    lower = np.quantile(sample, lower_quantile, axis=0)
    upper = np.quantile(sample, upper_quantile, axis=0)
    invalid_bounds = upper < lower
    if invalid_bounds.any():
        upper[invalid_bounds] = lower[invalid_bounds]

    total_sum = np.zeros(len(FEATURE_COLUMNS), dtype=np.float64)
    total_sumsq = np.zeros(len(FEATURE_COLUMNS), dtype=np.float64)
    count = 0
    for start in range(0, len(year_values), chunk_size):
        end = min(len(year_values), start + chunk_size)
        mask = (year_values[start:end] >= train_start_year) & (
            year_values[start:end] <= train_end_year
        )
        if not mask.any():
            continue
        values = np.asarray(features[start:end][mask], dtype=np.float64)
        values = np.clip(values, lower, upper)
        total_sum += values.sum(axis=0)
        total_sumsq += np.square(values).sum(axis=0)
        count += len(values)
    mean = total_sum / count
    variance = np.maximum(total_sumsq / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1e-8] = 1.0
    return Preprocessor(
        lower=lower,
        upper=upper,
        mean=mean,
        std=std,
        sampled_rows=sample_count,
        training_rows=count,
    )


def train_softmax_model(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    years: np.ndarray,
    preprocessor: Preprocessor,
    spec: FoldSpec,
    settings: Phase4Settings,
    seed: int,
) -> tuple[SoftmaxModel, list[dict[str, Any]], np.ndarray]:
    year_values = np.asarray(years)
    label_values = np.asarray(labels)
    train_mask = (year_values >= spec.train_start_year) & (
        year_values <= spec.train_end_year
    )
    class_counts = np.bincount(label_values[train_mask], minlength=len(LABELS)).astype(
        np.float64
    )
    priors = class_counts / class_counts.sum()
    weights = np.zeros((len(FEATURE_COLUMNS), len(LABELS)), dtype=np.float64)
    intercept = np.log(np.maximum(priors, EPSILON))
    intercept -= intercept.mean()

    first_moment_w = np.zeros_like(weights)
    second_moment_w = np.zeros_like(weights)
    first_moment_b = np.zeros_like(intercept)
    second_moment_b = np.zeros_like(intercept)
    beta1 = 0.9
    beta2 = 0.999
    adam_epsilon = 1e-8
    update_count = 0
    rng = np.random.default_rng(seed)
    chunk_ranges = [
        (start, min(len(year_values), start + settings.training_chunk_size))
        for start in range(0, len(year_values), settings.training_chunk_size)
    ]
    history: list[dict[str, Any]] = []
    best_loss = math.inf
    best_weights = weights.copy()
    best_intercept = intercept.copy()
    stale_epochs = 0

    for epoch in range(1, settings.maximum_epochs + 1):
        rng.shuffle(chunk_ranges)
        loss_total = 0.0
        example_total = 0
        batch_total = 0
        for start, end in chunk_ranges:
            mask = (year_values[start:end] >= spec.train_start_year) & (
                year_values[start:end] <= spec.train_end_year
            )
            if not mask.any():
                continue
            x = np.asarray(features[start:end][mask], dtype=np.float64)
            y = label_values[start:end][mask].astype(np.int64, copy=False)
            x = preprocessor.transform(x)
            order = rng.permutation(len(y))
            x = x[order]
            y = y[order]
            for batch_start in range(0, len(y), settings.batch_size):
                batch_end = min(len(y), batch_start + settings.batch_size)
                xb = x[batch_start:batch_end]
                yb = y[batch_start:batch_end]
                logits = xb @ weights + intercept
                probabilities = softmax(logits)
                loss = multiclass_log_loss(yb, probabilities)
                loss_total += loss * len(yb)
                example_total += len(yb)
                batch_total += 1

                gradient = probabilities
                gradient[np.arange(len(yb)), yb] -= 1.0
                gradient /= len(yb)
                gradient_w = xb.T @ gradient + settings.l2_penalty * weights
                gradient_b = gradient.sum(axis=0)

                update_count += 1
                first_moment_w = beta1 * first_moment_w + (1 - beta1) * gradient_w
                second_moment_w = beta2 * second_moment_w + (1 - beta2) * np.square(
                    gradient_w
                )
                first_moment_b = beta1 * first_moment_b + (1 - beta1) * gradient_b
                second_moment_b = beta2 * second_moment_b + (1 - beta2) * np.square(
                    gradient_b
                )
                correction1 = 1 - beta1**update_count
                correction2 = 1 - beta2**update_count
                weights -= settings.learning_rate * (
                    first_moment_w / correction1
                ) / (np.sqrt(second_moment_w / correction2) + adam_epsilon)
                intercept -= settings.learning_rate * (
                    first_moment_b / correction1
                ) / (np.sqrt(second_moment_b / correction2) + adam_epsilon)
                intercept -= intercept.mean()

        model = SoftmaxModel(weights=weights, intercept=intercept)
        calibration_loss = split_log_loss(
            features=features,
            labels=labels,
            years=years,
            start_year=spec.calibration_start_year,
            end_year=spec.calibration_end_year,
            preprocessor=preprocessor,
            model=model,
            chunk_size=settings.training_chunk_size,
        )
        average_training_loss = loss_total / max(example_total, 1)
        history.append(
            {
                "epoch": epoch,
                "training_log_loss": float(average_training_loss),
                "calibration_log_loss": float(calibration_loss),
                "batches": batch_total,
                "examples": example_total,
            }
        )
        print(
            f"  epoch {epoch}: train log loss={average_training_loss:.6f}, "
            f"calibration log loss={calibration_loss:.6f}"
        )
        if calibration_loss < best_loss - 1e-6:
            best_loss = calibration_loss
            best_weights = weights.copy()
            best_intercept = intercept.copy()
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch >= settings.minimum_epochs and stale_epochs >= settings.early_stopping_patience:
            break

    return (
        SoftmaxModel(weights=best_weights, intercept=best_intercept),
        history,
        priors,
    )


def split_log_loss(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    years: np.ndarray,
    start_year: int,
    end_year: int,
    preprocessor: Preprocessor,
    model: SoftmaxModel,
    chunk_size: int,
) -> float:
    year_values = np.asarray(years)
    label_values = np.asarray(labels)
    total_loss = 0.0
    total_count = 0
    for start in range(0, len(year_values), chunk_size):
        end = min(len(year_values), start + chunk_size)
        mask = (year_values[start:end] >= start_year) & (year_values[start:end] <= end_year)
        if not mask.any():
            continue
        x = preprocessor.transform(
            np.asarray(features[start:end][mask], dtype=np.float64)
        )
        y = label_values[start:end][mask].astype(np.int64, copy=False)
        probabilities = softmax(model.logits(x))
        total_loss += multiclass_log_loss(y, probabilities) * len(y)
        total_count += len(y)
    return total_loss / max(total_count, 1)


def collect_split_predictions(
    *,
    cache: MarketCache,
    mask: np.ndarray,
    preprocessor: Preprocessor,
    model: SoftmaxModel,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.flatnonzero(mask)
    logits = np.empty((len(positions), len(LABELS)), dtype=np.float64)
    labels = np.empty(len(positions), dtype=np.uint8)
    returns = np.empty(len(positions), dtype=np.float64)
    for output_start in range(0, len(positions), chunk_size):
        output_end = min(len(positions), output_start + chunk_size)
        selected = positions[output_start:output_end]
        x = preprocessor.transform(np.asarray(cache.features[selected], dtype=np.float64))
        logits[output_start:output_end] = model.logits(x)
        labels[output_start:output_end] = cache.labels[selected]
        returns[output_start:output_end] = cache.returns[selected]
    return logits, labels, returns


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    raw_loss = multiclass_log_loss(labels, softmax(logits))

    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        return multiclass_log_loss(labels, softmax(logits / temperature))

    left = math.log(0.25)
    right = math.log(4.0)
    ratio = (math.sqrt(5) - 1) / 2
    x1 = right - ratio * (right - left)
    x2 = left + ratio * (right - left)
    f1 = objective(x1)
    f2 = objective(x2)
    for _ in range(36):
        if f1 <= f2:
            right = x2
            x2 = x1
            f2 = f1
            x1 = right - ratio * (right - left)
            f1 = objective(x1)
        else:
            left = x1
            x1 = x2
            f1 = f2
            x2 = left + ratio * (right - left)
            f2 = objective(x2)
    temperature = math.exp((left + right) / 2)
    scaled_loss = objective(math.log(temperature))
    if raw_loss <= scaled_loss:
        return 1.0, raw_loss, raw_loss
    return temperature, raw_loss, scaled_loss


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def multiclass_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    selected = probabilities[np.arange(len(labels)), labels.astype(np.int64)]
    return float(-np.log(np.clip(selected, EPSILON, 1.0)).mean())


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = np.argmax(probabilities, axis=1)
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    np.add.at(confusion, (labels.astype(np.int64), predictions), 1)
    recalls = []
    f1_scores = []
    for index in range(len(LABELS)):
        true_positive = confusion[index, index]
        false_negative = confusion[index, :].sum() - true_positive
        false_positive = confusion[:, index].sum() - true_positive
        recall = true_positive / max(true_positive + false_negative, 1)
        precision = true_positive / max(true_positive + false_positive, 1)
        f1 = 2 * precision * recall / max(precision + recall, EPSILON)
        recalls.append(recall)
        f1_scores.append(f1)
    one_hot = np.eye(len(LABELS), dtype=np.float64)[labels.astype(np.int64)]
    confidence = probabilities.max(axis=1)
    correctness = predictions == labels
    return {
        "row_count": int(len(labels)),
        "accuracy": float(np.trace(confusion) / max(confusion.sum(), 1)),
        "macro_f1": float(np.mean(f1_scores)),
        "balanced_accuracy": float(np.mean(recalls)),
        "log_loss": multiclass_log_loss(labels, probabilities),
        "brier_score": float(np.mean(np.sum(np.square(probabilities - one_hot), axis=1))),
        "ece": expected_calibration_error(confidence, correctness, bins=10),
    }


def expected_calibration_error(
    confidence: np.ndarray,
    correctness: np.ndarray,
    *,
    bins: int,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        mask = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        if not mask.any():
            continue
        accuracy_gap = abs(
            float(correctness[mask].mean()) - float(confidence[mask].mean())
        )
        result += mask.mean() * accuracy_gap
    return float(result)


def institutional_index_deciles(
    *,
    market: str,
    test_year: int,
    probabilities: np.ndarray,
    labels: np.ndarray,
    adjusted_returns: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index_values = 100.0 * (
        probabilities[:, LABEL_TO_INDEX["UP"]]
        - probabilities[:, LABEL_TO_INDEX["DOWN"]]
    )
    order = np.argsort(index_values, kind="mergesort")
    buckets = np.empty(len(order), dtype=np.uint8)
    for decile, selected in enumerate(np.array_split(order, 10), start=1):
        buckets[selected] = decile
    rows: list[dict[str, Any]] = []
    average_returns = []
    up_rates = []
    down_rates = []
    for decile in range(1, 11):
        mask = buckets == decile
        count = int(mask.sum())
        average_return = float(adjusted_returns[mask].mean())
        up_rate = float((labels[mask] == LABEL_TO_INDEX["UP"]).mean())
        flat_rate = float((labels[mask] == LABEL_TO_INDEX["FLAT"]).mean())
        down_rate = float((labels[mask] == LABEL_TO_INDEX["DOWN"]).mean())
        average_returns.append(average_return)
        up_rates.append(up_rate)
        down_rates.append(down_rate)
        rows.append(
            {
                "market_type": market,
                "test_year": test_year,
                "decile": decile,
                "row_count": count,
                "min_layout_index": float(index_values[mask].min()),
                "max_layout_index": float(index_values[mask].max()),
                "mean_layout_index": float(index_values[mask].mean()),
                "mean_adjusted_return_10d": average_return,
                "up_rate": up_rate,
                "flat_rate": flat_rate,
                "down_rate": down_rate,
            }
        )
    decile_numbers = np.arange(1, 11, dtype=np.float64)
    return rows, {
        "return_monotonic_violations": int(np.sum(np.diff(average_returns) < 0)),
        "up_rate_monotonic_violations": int(np.sum(np.diff(up_rates) < 0)),
        "down_rate_monotonic_violations": int(np.sum(np.diff(down_rates) > 0)),
        "return_decile_correlation": _safe_correlation(decile_numbers, np.asarray(average_returns)),
        "high_minus_low_return": float(average_returns[-1] - average_returns[0]),
        "high_minus_low_up_rate": float(up_rates[-1] - up_rates[0]),
        "low_minus_high_down_rate": float(down_rates[0] - down_rates[-1]),
    }


def calibration_report_rows(
    *,
    market: str,
    test_year: int,
    raw_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
    labels: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    edges = np.linspace(0.0, 1.0, 11)
    for model_name, probabilities in (
        ("raw", raw_probabilities),
        ("calibrated", calibrated_probabilities),
    ):
        predictions = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)
        correct = predictions == labels
        for bin_index in range(10):
            mask = _probability_bin_mask(confidence, edges, bin_index)
            if not mask.any():
                continue
            rows.append(
                {
                    "market_type": market,
                    "test_year": test_year,
                    "model": model_name,
                    "calibration_type": "confidence",
                    "class_label": "",
                    "bin": bin_index + 1,
                    "lower_bound": edges[bin_index],
                    "upper_bound": edges[bin_index + 1],
                    "row_count": int(mask.sum()),
                    "mean_predicted_probability": float(confidence[mask].mean()),
                    "observed_rate": float(correct[mask].mean()),
                }
            )
        for class_index, class_label in enumerate(LABELS):
            values = probabilities[:, class_index]
            actual = labels == class_index
            for bin_index in range(10):
                mask = _probability_bin_mask(values, edges, bin_index)
                if not mask.any():
                    continue
                rows.append(
                    {
                        "market_type": market,
                        "test_year": test_year,
                        "model": model_name,
                        "calibration_type": "class_probability",
                        "class_label": class_label,
                        "bin": bin_index + 1,
                        "lower_bound": edges[bin_index],
                        "upper_bound": edges[bin_index + 1],
                        "row_count": int(mask.sum()),
                        "mean_predicted_probability": float(values[mask].mean()),
                        "observed_rate": float(actual[mask].mean()),
                    }
                )
    return rows


def _probability_bin_mask(values: np.ndarray, edges: np.ndarray, index: int) -> np.ndarray:
    return (values >= edges[index]) & (
        values <= edges[index + 1] if index == len(edges) - 2 else values < edges[index + 1]
    )


def coefficient_report_rows(
    *,
    market: str,
    test_year: int,
    model: SoftmaxModel,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for class_index, class_label in enumerate(LABELS):
        rows.append(
            {
                "market_type": market,
                "test_year": test_year,
                "class_label": class_label,
                "feature": "(intercept)",
                "standardized_coefficient": float(model.intercept[class_index]),
            }
        )
        for feature_index, feature in enumerate(FEATURE_COLUMNS):
            rows.append(
                {
                    "market_type": market,
                    "test_year": test_year,
                    "class_label": class_label,
                    "feature": feature,
                    "standardized_coefficient": float(
                        model.weights[feature_index, class_index]
                    ),
                }
            )
    up_index = LABEL_TO_INDEX["UP"]
    down_index = LABEL_TO_INDEX["DOWN"]
    rows.append(
        {
            "market_type": market,
            "test_year": test_year,
            "class_label": "UP_MINUS_DOWN",
            "feature": "(intercept)",
            "standardized_coefficient": float(
                model.intercept[up_index] - model.intercept[down_index]
            ),
        }
    )
    for feature_index, feature in enumerate(FEATURE_COLUMNS):
        rows.append(
            {
                "market_type": market,
                "test_year": test_year,
                "class_label": "UP_MINUS_DOWN",
                "feature": feature,
                "standardized_coefficient": float(
                    model.weights[feature_index, up_index]
                    - model.weights[feature_index, down_index]
                ),
            }
        )
    return rows


def preprocessing_report_rows(
    *,
    market: str,
    test_year: int,
    preprocessor: Preprocessor,
) -> list[dict[str, Any]]:
    return [
        {
            "market_type": market,
            "test_year": test_year,
            "feature": feature,
            "clip_lower": float(preprocessor.lower[index]),
            "clip_upper": float(preprocessor.upper[index]),
            "training_mean_after_clip": float(preprocessor.mean[index]),
            "training_std_after_clip": float(preprocessor.std[index]),
            "quantile_sample_rows": preprocessor.sampled_rows,
            "training_rows": preprocessor.training_rows,
        }
        for index, feature in enumerate(FEATURE_COLUMNS)
    ]


def export_phase4a_reports(
    *,
    output_dir: Path,
    run_signature: str,
    settings: Phase4Settings,
    expected_specs: list[FoldSpec],
    fold_results: list[dict[str, Any]],
    caches: dict[str, MarketCache],
) -> tuple[list[Path], bool]:
    fold_summaries = [row["fold_summary"] for row in fold_results]
    metrics = [item for row in fold_results for item in row["metrics"]]
    deciles = [item for row in fold_results for item in row["deciles"]]
    calibration = [item for row in fold_results for item in row["calibration"]]
    coefficients = [item for row in fold_results for item in row["coefficients"]]
    preprocessing = [item for row in fold_results for item in row["preprocessing"]]
    training_history = [item for row in fold_results for item in row["training_history"]]
    coefficient_stability = coefficient_stability_rows(coefficients)
    summary_rows, ready = phase4_summary_rows(
        run_signature=run_signature,
        settings=settings,
        expected_specs=expected_specs,
        fold_summaries=fold_summaries,
        metrics=metrics,
        caches=caches,
    )
    paths = [
        _write_csv(output_dir / "phase4a_summary.csv", summary_rows),
        _write_csv(output_dir / "phase4a_fold_summary.csv", fold_summaries),
        _write_csv(output_dir / "phase4a_metrics.csv", metrics),
        _write_csv(output_dir / "phase4a_index_deciles.csv", deciles),
        _write_csv(output_dir / "phase4a_calibration.csv", calibration),
        _write_csv(output_dir / "phase4a_coefficients.csv", coefficients),
        _write_csv(
            output_dir / "phase4a_coefficient_stability.csv", coefficient_stability
        ),
        _write_csv(output_dir / "phase4a_preprocessing.csv", preprocessing),
        _write_csv(output_dir / "phase4a_training_history.csv", training_history),
    ]
    archive = output_dir / "phase4a_validation_reports.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in paths:
            handle.write(path, arcname=path.name)
    paths.append(archive)
    return paths, ready


def coefficient_stability_rows(coefficients: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not coefficients:
        return []
    frame = pd.DataFrame(coefficients)
    rows = []
    for (market, class_label, feature), group in frame.groupby(
        ["market_type", "class_label", "feature"], sort=True
    ):
        values = group["standardized_coefficient"].to_numpy(dtype=np.float64)
        positive_ratio = float((values > 0).mean())
        negative_ratio = float((values < 0).mean())
        rows.append(
            {
                "market_type": market,
                "class_label": class_label,
                "feature": feature,
                "fold_count": len(values),
                "mean_coefficient": float(values.mean()),
                "median_coefficient": float(np.median(values)),
                "std_coefficient": float(values.std()),
                "minimum_coefficient": float(values.min()),
                "maximum_coefficient": float(values.max()),
                "positive_fold_ratio": positive_ratio,
                "negative_fold_ratio": negative_ratio,
                "sign_consistency": max(positive_ratio, negative_ratio),
            }
        )
    return rows


def phase4_summary_rows(
    *,
    run_signature: str,
    settings: Phase4Settings,
    expected_specs: list[FoldSpec],
    fold_summaries: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    caches: dict[str, MarketCache],
) -> tuple[list[dict[str, Any]], bool]:
    completed = len(fold_summaries)
    expected = len(expected_specs)
    metric_frame = pd.DataFrame(metrics)
    fold_frame = pd.DataFrame(fold_summaries)
    checks: dict[str, bool] = {
        "all_folds_completed": completed == expected,
        "minimum_folds_each_market": False,
        "calibrated_log_loss_beats_history_each_market": False,
        "balanced_accuracy_above_random_each_market": False,
        "positive_high_low_return_each_market": False,
    }
    aggregate_rows: list[tuple[str, Any]] = []
    market_ready = []
    if not metric_frame.empty and not fold_frame.empty:
        minimum_folds = True
        beats_history = True
        balanced_above_random = True
        positive_spread = True
        for market in sorted(caches):
            market_folds = fold_frame[fold_frame["market"] == market]
            minimum_folds &= len(market_folds) >= 5
            model_rows = metric_frame[
                (metric_frame["market_type"] == market)
                & (metric_frame["model"] == "multinomial_logistic_calibrated")
            ]
            history_rows = metric_frame[
                (metric_frame["market_type"] == market)
                & (metric_frame["model"] == "historical_probability")
            ]
            model_loss = _weighted_metric(model_rows, "log_loss")
            history_loss = _weighted_metric(history_rows, "log_loss")
            model_balanced = _weighted_metric(model_rows, "balanced_accuracy")
            high_low_return = _weighted_fold_metric(
                market_folds, "high_minus_low_return"
            )
            market_pass = (
                len(market_folds) >= 5
                and model_loss < history_loss
                and model_balanced > 1 / 3
                and high_low_return > 0
            )
            market_ready.append(market_pass)
            beats_history &= model_loss < history_loss
            balanced_above_random &= model_balanced > 1 / 3
            positive_spread &= high_low_return > 0
            aggregate_rows.extend(
                [
                    (f"{market}_fold_count", len(market_folds)),
                    (f"{market}_test_rows", int(market_folds["test_rows"].sum())),
                    (f"{market}_calibrated_log_loss", model_loss),
                    (f"{market}_historical_log_loss", history_loss),
                    (f"{market}_log_loss_improvement", history_loss - model_loss),
                    (f"{market}_balanced_accuracy", model_balanced),
                    (
                        f"{market}_macro_f1",
                        _weighted_metric(model_rows, "macro_f1"),
                    ),
                    (f"{market}_high_minus_low_return", high_low_return),
                    (
                        f"{market}_return_monotonic_violations",
                        _weighted_fold_metric(
                            market_folds, "return_monotonic_violations"
                        ),
                    ),
                ]
            )
        checks["minimum_folds_each_market"] = minimum_folds
        checks["calibrated_log_loss_beats_history_each_market"] = beats_history
        checks["balanced_accuracy_above_random_each_market"] = balanced_above_random
        checks["positive_high_low_return_each_market"] = positive_spread
    ready = all(checks.values()) and all(market_ready)
    status = "PASS" if completed == expected else "FAIL"
    rows = [
        {"metric": "phase4_version", "value": PHASE4_VERSION},
        {"metric": "run_signature", "value": run_signature},
        {"metric": "pipeline_status", "value": status},
        {"metric": "ready_for_phase4b", "value": int(ready)},
        {"metric": "expected_folds", "value": expected},
        {"metric": "completed_folds", "value": completed},
        {"metric": "model_feature_count", "value": len(FEATURE_COLUMNS)},
        {"metric": "first_test_year", "value": settings.first_test_year},
        {"metric": "maximum_epochs", "value": settings.maximum_epochs},
        {"metric": "batch_size", "value": settings.batch_size},
        {"metric": "clip_lower_quantile", "value": settings.clip_lower_quantile},
        {"metric": "clip_upper_quantile", "value": settings.clip_upper_quantile},
    ]
    rows.extend({"metric": key, "value": value} for key, value in aggregate_rows)
    rows.extend(
        {"metric": f"check_{key}", "value": int(value)}
        for key, value in checks.items()
    )
    for market, cache in sorted(caches.items()):
        rows.extend(
            [
                {"metric": f"{market}_source_rows", "value": cache.row_count},
                {"metric": f"{market}_source_sha256", "value": cache.source_sha256},
                {"metric": f"{market}_cache_signature", "value": cache.signature},
            ]
        )
    return rows, ready


def phase4_config_signature(
    settings: Phase4Settings,
    caches: dict[str, MarketCache],
) -> str:
    payload = {
        "phase4_version": PHASE4_VERSION,
        "settings": asdict(settings),
        "cache_signatures": {
            market: cache.signature for market, cache in sorted(caches.items())
        },
        "features": FEATURE_COLUMNS,
        "labels": LABELS,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _weighted_metric(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        return math.nan
    weights = frame["row_count"].to_numpy(dtype=np.float64)
    values = frame[column].to_numpy(dtype=np.float64)
    return float(np.average(values, weights=weights))


def _weighted_fold_metric(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        return math.nan
    weights = frame["test_rows"].to_numpy(dtype=np.float64)
    values = frame[column].to_numpy(dtype=np.float64)
    return float(np.average(values, weights=weights))


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _ensure_all_classes(labels: np.ndarray, fold_id: str, split: str) -> None:
    observed = set(int(value) for value in np.unique(labels))
    missing = set(range(len(LABELS))).difference(observed)
    if missing:
        names = [LABELS[index] for index in sorted(missing)]
        raise RuntimeError(f"{fold_id} {split} 缺少類別：{names}")


def _load_phase3_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"找不到 Phase 3 manifest：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["file_name"]: row for row in csv.DictReader(handle)}


def _ensure_phase3_ready(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"找不到 Phase 3B 摘要：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        values = {row["metric"]: row["value"] for row in csv.DictReader(handle)}
    if values.get("ready_for_modeling") != "1" or values.get("status") != "PASS":
        raise RuntimeError("Phase 3B 尚未通過，不可執行 Phase 4A")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("status\n", encoding="utf-8-sig")
        return path
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_gzip_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
    os.replace(temporary, path)


def _read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)
