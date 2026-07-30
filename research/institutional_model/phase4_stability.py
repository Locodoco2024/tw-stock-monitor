from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import zipfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from research.institutional_model.phase3_dataset import FEATURE_COLUMNS
from research.institutional_model.phase4_model import (
    EPSILON,
    LABELS,
    LABEL_TO_INDEX,
    MARKETS,
    FoldSpec,
    MarketCache,
    Phase4Settings,
    Preprocessor,
    SoftmaxModel,
    build_or_load_market_cache,
    build_rolling_folds,
    calibration_report_rows,
    classification_metrics,
    fit_temperature,
    institutional_index_deciles,
    multiclass_log_loss,
    softmax,
)


PHASE4B_VERSION = "phase4b-v1"
DEVELOPMENT_END_YEAR = 2022
CONFIRMATION_START_YEAR = 2023
PRIOR_BLEND_ALPHAS = tuple(index / 20 for index in range(21))

CORE_ACTORS = ("foreign", "investment_trust", "dealer_self")
CORE_FEATURE_COLUMNS = tuple(
    [
        f"{actor}_flow_pct_{window}d"
        for actor in CORE_ACTORS
        for window in (1, 5, 20)
    ]
    + [
        f"{actor}_buy_day_ratio_{window}d"
        for actor in CORE_ACTORS
        for window in (5, 20)
    ]
    + [f"{actor}_streak" for actor in CORE_ACTORS]
    + [
        "institutional_agreement_1d",
        "institutional_agreement_5d",
        "institutional_agreement_20d",
        "selected_total_acceleration_5d_vs_20d",
    ]
)


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    feature_set: str
    feature_columns: tuple[str, ...]
    l2_penalty: float

    @property
    def feature_indices(self) -> np.ndarray:
        positions = [FEATURE_COLUMNS.index(name) for name in self.feature_columns]
        return np.asarray(positions, dtype=np.int64)


CANDIDATES = (
    CandidateSpec(
        candidate_id="full40_l2_1e-4",
        feature_set="full40",
        feature_columns=tuple(FEATURE_COLUMNS),
        l2_penalty=0.0001,
    ),
    CandidateSpec(
        candidate_id="full40_l2_1e-3",
        feature_set="full40",
        feature_columns=tuple(FEATURE_COLUMNS),
        l2_penalty=0.001,
    ),
    CandidateSpec(
        candidate_id="core22_l2_1e-3",
        feature_set="core22",
        feature_columns=CORE_FEATURE_COLUMNS,
        l2_penalty=0.001,
    ),
)


@dataclass(frozen=True)
class Phase4BResult:
    status: str
    ready_for_phase4c: bool
    completed_candidate_folds: int
    expected_candidate_folds: int
    failed_candidate_folds: int
    output_paths: tuple[Path, ...]


def run_phase4b_stability_research(
    *,
    output_dir: Path | str,
    cache_root: Path | str,
    phase4a_run_root: Path | str,
    run_root: Path | str,
    settings: Phase4Settings | None = None,
    markets: Iterable[str] = MARKETS,
    force: bool = False,
) -> Phase4BResult:
    config = settings or Phase4Settings()
    config.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache_base = Path(cache_root)
    phase4a_runs = Path(phase4a_run_root)
    run_base = Path(run_root)
    manifest = _load_phase3_manifest(output / "phase3_dataset_manifest.csv")
    _ensure_phase3_ready(output / "phase3b_summary.csv")

    selected_markets = tuple(dict.fromkeys(str(value).lower() for value in markets))
    unknown = set(selected_markets).difference(MARKETS)
    if unknown:
        raise ValueError(f"不支援的 Phase 4B 市場：{sorted(unknown)}")

    caches: dict[str, MarketCache] = {}
    folds_by_market: dict[str, list[FoldSpec]] = {}
    for market in selected_markets:
        cache = build_or_load_market_cache(
            output_dir=output,
            cache_root=cache_base,
            manifest=manifest,
            market=market,
            chunk_size=config.cache_chunk_size,
            force=False,
        )
        caches[market] = cache
        folds = build_rolling_folds(
            market=market,
            years=np.asarray(cache.years),
            first_test_year=config.first_test_year,
            minimum_training_years=config.minimum_training_years,
            calibration_years=config.calibration_years,
        )
        folds_by_market[market] = folds

    signature = phase4b_config_signature(config, caches, selected_markets)
    run_dir = run_base / signature[:16]
    fold_dir = run_dir / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(
        run_dir / "run_config.json",
        {
            "phase4b_version": PHASE4B_VERSION,
            "run_signature": signature,
            "settings": asdict(config),
            "markets": list(selected_markets),
            "development_end_year": DEVELOPMENT_END_YEAR,
            "confirmation_start_year": CONFIRMATION_START_YEAR,
            "candidates": [candidate_to_dict(value) for value in CANDIDATES],
            "cache_signatures": {
                market: cache.signature for market, cache in caches.items()
            },
        },
    )

    failed = 0
    expected = 0
    completed = 0
    development_results: list[dict[str, Any]] = []
    for market in selected_markets:
        dev_folds = [
            fold
            for fold in folds_by_market[market]
            if fold.test_year <= DEVELOPMENT_END_YEAR
        ]
        if len(dev_folds) < 3:
            raise RuntimeError(f"{market.upper()} Phase 4B 開發期折數不足")
        for candidate in CANDIDATES:
            for fold in dev_folds:
                expected += 1
                result, was_failed = _load_or_run_candidate_fold(
                    cache=caches[market],
                    fold=fold,
                    candidate=candidate,
                    settings=config,
                    phase4a_run_root=phase4a_runs,
                    result_path=fold_dir / _result_file_name(fold, candidate),
                    force=force,
                    position=f"開發期 {market.upper()} {candidate.candidate_id} {fold.test_year}",
                )
                failed += int(was_failed)
                if result is not None:
                    completed += 1
                    development_results.append(result)

    selections = select_development_candidates(development_results)
    confirmation_results: list[dict[str, Any]] = []
    for market in selected_markets:
        selected = selections.get(market)
        if not selected:
            continue
        candidate = candidate_by_id(str(selected["selected_candidate_id"]))
        confirmation_folds = [
            fold
            for fold in folds_by_market[market]
            if fold.test_year >= CONFIRMATION_START_YEAR
        ]
        if len(confirmation_folds) < 3:
            raise RuntimeError(f"{market.upper()} Phase 4B 確認期折數不足")
        for fold in confirmation_folds:
            expected += 1
            result, was_failed = _load_or_run_candidate_fold(
                cache=caches[market],
                fold=fold,
                candidate=candidate,
                settings=config,
                phase4a_run_root=phase4a_runs,
                result_path=fold_dir / _result_file_name(fold, candidate),
                force=force,
                position=f"確認期 {market.upper()} {candidate.candidate_id} {fold.test_year}",
            )
            failed += int(was_failed)
            if result is not None:
                completed += 1
                confirmation_results.append(result)

    paths, ready = export_phase4b_reports(
        output_dir=output,
        run_signature=signature,
        settings=config,
        caches=caches,
        development_results=development_results,
        selections=selections,
        confirmation_results=confirmation_results,
        completed=completed,
        expected=expected,
        failed=failed,
    )
    status = "PASS" if completed == expected and failed == 0 else "FAIL"
    return Phase4BResult(
        status=status,
        ready_for_phase4c=ready and status == "PASS",
        completed_candidate_folds=completed,
        expected_candidate_folds=expected,
        failed_candidate_folds=failed,
        output_paths=tuple(paths),
    )


def _load_or_run_candidate_fold(
    *,
    cache: MarketCache,
    fold: FoldSpec,
    candidate: CandidateSpec,
    settings: Phase4Settings,
    phase4a_run_root: Path,
    result_path: Path,
    force: bool,
    position: str,
) -> tuple[dict[str, Any] | None, bool]:
    if result_path.exists() and not force:
        print(f"{position} 已完成，略過並沿用既有結果。")
        return _read_gzip_json(result_path), False

    print(f"Phase 4B {position}")
    try:
        result = None
        if candidate.candidate_id == "full40_l2_1e-4" and not force:
            result = reuse_phase4a_fold(
                cache=cache,
                fold=fold,
                candidate=candidate,
                settings=settings,
                phase4a_run_root=phase4a_run_root,
            )
            if result is not None:
                print("  已沿用 Phase 4A 原始模型，不重新訓練。")
        if result is None:
            result = run_candidate_fold(
                cache=cache,
                fold=fold,
                candidate=candidate,
                settings=settings,
            )
        _write_gzip_json_atomic(result_path, result)
        return result, False
    except Exception as exc:
        print(f"  候選折執行失敗：{type(exc).__name__}: {exc}")
        _write_gzip_json_atomic(
            result_path.with_name(result_path.name.replace(".json.gz", ".error.json.gz")),
            {
                "fold": asdict(fold),
                "candidate": candidate_to_dict(candidate),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failed_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        return None, True


def run_candidate_fold(
    *,
    cache: MarketCache,
    fold: FoldSpec,
    candidate: CandidateSpec,
    settings: Phase4Settings,
) -> dict[str, Any]:
    years = np.asarray(cache.years)
    labels = np.asarray(cache.labels)
    train_mask = (years >= fold.train_start_year) & (years <= fold.train_end_year)
    calibration_mask = (years >= fold.calibration_start_year) & (
        years <= fold.calibration_end_year
    )
    test_mask = years == fold.test_year
    _ensure_split(train_mask, labels, fold, "train")
    _ensure_split(calibration_mask, labels, fold, "calibration")
    _ensure_split(test_mask, labels, fold, "test")

    seed = settings.random_seed + fold.test_year
    if fold.market == "tpex":
        seed += 10_000
    seed += int(candidate.l2_penalty * 10_000_000)
    feature_indices = candidate.feature_indices
    preprocessor = fit_candidate_preprocessor(
        cache=cache,
        fold=fold,
        feature_indices=feature_indices,
        settings=settings,
        seed=seed,
    )
    model_settings = replace(settings, l2_penalty=candidate.l2_penalty)
    model, history, priors = train_candidate_model(
        cache=cache,
        fold=fold,
        candidate=candidate,
        feature_indices=feature_indices,
        preprocessor=preprocessor,
        settings=model_settings,
        seed=seed,
    )
    return evaluate_candidate_fold(
        cache=cache,
        fold=fold,
        candidate=candidate,
        feature_indices=feature_indices,
        preprocessor=preprocessor,
        model=model,
        train_priors=priors,
        history=history,
        settings=settings,
        reused_phase4a=False,
    )


def fit_candidate_preprocessor(
    *,
    cache: MarketCache,
    fold: FoldSpec,
    feature_indices: np.ndarray,
    settings: Phase4Settings,
    seed: int,
) -> Preprocessor:
    years = np.asarray(cache.years)
    positions = np.flatnonzero(
        (years >= fold.train_start_year) & (years <= fold.train_end_year)
    )
    rng = np.random.default_rng(seed)
    sample_count = min(settings.quantile_sample_size, len(positions))
    selected = np.sort(rng.choice(positions, size=sample_count, replace=False))
    sample = np.asarray(cache.features[selected], dtype=np.float64)[:, feature_indices]
    lower = np.quantile(sample, settings.clip_lower_quantile, axis=0)
    upper = np.quantile(sample, settings.clip_upper_quantile, axis=0)
    upper = np.maximum(upper, lower)

    total_sum = np.zeros(len(feature_indices), dtype=np.float64)
    total_sumsq = np.zeros(len(feature_indices), dtype=np.float64)
    count = 0
    for start in range(0, len(years), settings.training_chunk_size):
        end = min(len(years), start + settings.training_chunk_size)
        mask = (years[start:end] >= fold.train_start_year) & (
            years[start:end] <= fold.train_end_year
        )
        if not mask.any():
            continue
        values = np.asarray(cache.features[start:end][mask], dtype=np.float64)
        values = values[:, feature_indices]
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


def train_candidate_model(
    *,
    cache: MarketCache,
    fold: FoldSpec,
    candidate: CandidateSpec,
    feature_indices: np.ndarray,
    preprocessor: Preprocessor,
    settings: Phase4Settings,
    seed: int,
) -> tuple[SoftmaxModel, list[dict[str, Any]], np.ndarray]:
    years = np.asarray(cache.years)
    labels = np.asarray(cache.labels)
    train_mask = (years >= fold.train_start_year) & (years <= fold.train_end_year)
    counts = np.bincount(labels[train_mask], minlength=len(LABELS)).astype(np.float64)
    priors = counts / counts.sum()
    weights = np.zeros((len(feature_indices), len(LABELS)), dtype=np.float64)
    intercept = np.log(np.maximum(priors, EPSILON))
    intercept -= intercept.mean()

    first_w = np.zeros_like(weights)
    second_w = np.zeros_like(weights)
    first_b = np.zeros_like(intercept)
    second_b = np.zeros_like(intercept)
    beta1 = 0.9
    beta2 = 0.999
    adam_epsilon = 1e-8
    updates = 0
    rng = np.random.default_rng(seed)
    ranges = [
        (start, min(len(years), start + settings.training_chunk_size))
        for start in range(0, len(years), settings.training_chunk_size)
    ]
    history: list[dict[str, Any]] = []
    best_loss = math.inf
    best_weights = weights.copy()
    best_intercept = intercept.copy()
    stale = 0

    for epoch in range(1, settings.maximum_epochs + 1):
        rng.shuffle(ranges)
        loss_total = 0.0
        examples = 0
        batches = 0
        for start, end in ranges:
            mask = (years[start:end] >= fold.train_start_year) & (
                years[start:end] <= fold.train_end_year
            )
            if not mask.any():
                continue
            x = np.asarray(cache.features[start:end][mask], dtype=np.float64)
            x = preprocessor.transform(x[:, feature_indices])
            y = labels[start:end][mask].astype(np.int64, copy=False)
            order = rng.permutation(len(y))
            x = x[order]
            y = y[order]
            for batch_start in range(0, len(y), settings.batch_size):
                batch_end = min(len(y), batch_start + settings.batch_size)
                xb = x[batch_start:batch_end]
                yb = y[batch_start:batch_end]
                probabilities = softmax(xb @ weights + intercept)
                loss_total += multiclass_log_loss(yb, probabilities) * len(yb)
                examples += len(yb)
                batches += 1

                gradient = probabilities
                gradient[np.arange(len(yb)), yb] -= 1.0
                gradient /= len(yb)
                gradient_w = xb.T @ gradient + settings.l2_penalty * weights
                gradient_b = gradient.sum(axis=0)
                updates += 1
                first_w = beta1 * first_w + (1 - beta1) * gradient_w
                second_w = beta2 * second_w + (1 - beta2) * np.square(gradient_w)
                first_b = beta1 * first_b + (1 - beta1) * gradient_b
                second_b = beta2 * second_b + (1 - beta2) * np.square(gradient_b)
                correction1 = 1 - beta1**updates
                correction2 = 1 - beta2**updates
                weights -= settings.learning_rate * (first_w / correction1) / (
                    np.sqrt(second_w / correction2) + adam_epsilon
                )
                intercept -= settings.learning_rate * (first_b / correction1) / (
                    np.sqrt(second_b / correction2) + adam_epsilon
                )
                intercept -= intercept.mean()

        model = SoftmaxModel(weights=weights, intercept=intercept)
        calibration_loss = candidate_split_log_loss(
            cache=cache,
            fold=fold,
            feature_indices=feature_indices,
            preprocessor=preprocessor,
            model=model,
            chunk_size=settings.training_chunk_size,
        )
        training_loss = loss_total / max(examples, 1)
        history.append(
            {
                "epoch": epoch,
                "training_log_loss": float(training_loss),
                "calibration_log_loss": float(calibration_loss),
                "batches": batches,
                "examples": examples,
            }
        )
        print(
            f"  {candidate.candidate_id} epoch {epoch}: "
            f"train={training_loss:.6f}, calibration={calibration_loss:.6f}"
        )
        if calibration_loss < best_loss - 1e-6:
            best_loss = calibration_loss
            best_weights = weights.copy()
            best_intercept = intercept.copy()
            stale = 0
        else:
            stale += 1
        if epoch >= settings.minimum_epochs and stale >= settings.early_stopping_patience:
            break

    return SoftmaxModel(best_weights, best_intercept), history, priors


def candidate_split_log_loss(
    *,
    cache: MarketCache,
    fold: FoldSpec,
    feature_indices: np.ndarray,
    preprocessor: Preprocessor,
    model: SoftmaxModel,
    chunk_size: int,
) -> float:
    years = np.asarray(cache.years)
    labels = np.asarray(cache.labels)
    total_loss = 0.0
    total_count = 0
    for start in range(0, len(years), chunk_size):
        end = min(len(years), start + chunk_size)
        mask = (years[start:end] >= fold.calibration_start_year) & (
            years[start:end] <= fold.calibration_end_year
        )
        if not mask.any():
            continue
        x = np.asarray(cache.features[start:end][mask], dtype=np.float64)
        x = preprocessor.transform(x[:, feature_indices])
        y = labels[start:end][mask].astype(np.int64, copy=False)
        probabilities = softmax(model.logits(x))
        total_loss += multiclass_log_loss(y, probabilities) * len(y)
        total_count += len(y)
    return total_loss / max(total_count, 1)


def evaluate_candidate_fold(
    *,
    cache: MarketCache,
    fold: FoldSpec,
    candidate: CandidateSpec,
    feature_indices: np.ndarray,
    preprocessor: Preprocessor,
    model: SoftmaxModel,
    train_priors: np.ndarray,
    history: list[dict[str, Any]],
    settings: Phase4Settings,
    reused_phase4a: bool,
) -> dict[str, Any]:
    years = np.asarray(cache.years)
    calibration_mask = (years >= fold.calibration_start_year) & (
        years <= fold.calibration_end_year
    )
    test_mask = years == fold.test_year
    calibration_logits, calibration_labels, _ = collect_candidate_predictions(
        cache=cache,
        mask=calibration_mask,
        feature_indices=feature_indices,
        preprocessor=preprocessor,
        model=model,
        chunk_size=settings.training_chunk_size,
    )
    test_logits, test_labels, test_returns = collect_candidate_predictions(
        cache=cache,
        mask=test_mask,
        feature_indices=feature_indices,
        preprocessor=preprocessor,
        model=model,
        chunk_size=settings.training_chunk_size,
    )
    calibration_raw = softmax(calibration_logits)
    test_raw = softmax(test_logits)
    temperature, raw_calibration_loss, scaled_calibration_loss = fit_temperature(
        calibration_logits, calibration_labels
    )
    test_temperature = softmax(test_logits / temperature)
    blend_alpha, blend_calibration_loss = fit_prior_blend(
        probabilities=calibration_raw,
        labels=calibration_labels,
        priors=train_priors,
    )
    test_blend = blend_probabilities(test_raw, train_priors, blend_alpha)
    history_probabilities = np.tile(train_priors, (len(test_labels), 1))

    metric_rows: list[dict[str, Any]] = []
    for variant, probabilities in (
        ("raw", test_raw),
        ("temperature", test_temperature),
        ("prior_blend", test_blend),
        ("historical_probability", history_probabilities),
    ):
        metric_rows.append(
            {
                "market_type": fold.market,
                "test_year": fold.test_year,
                "candidate_id": candidate.candidate_id,
                "probability_variant": variant,
                **classification_metrics(test_labels, probabilities),
            }
        )

    deciles, monotonicity = institutional_index_deciles(
        market=fold.market,
        test_year=fold.test_year,
        probabilities=test_raw,
        labels=test_labels,
        adjusted_returns=test_returns,
    )
    for row in deciles:
        row["candidate_id"] = candidate.candidate_id
        row["index_probability_variant"] = "raw"
    calibration_rows = calibration_report_rows(
        market=fold.market,
        test_year=fold.test_year,
        raw_probabilities=test_raw,
        calibrated_probabilities=test_temperature,
        labels=test_labels,
    )
    for row in calibration_rows:
        row["candidate_id"] = candidate.candidate_id
    coefficients = candidate_coefficient_rows(
        market=fold.market,
        test_year=fold.test_year,
        candidate=candidate,
        model=model,
    )
    preprocessing = candidate_preprocessing_rows(
        market=fold.market,
        test_year=fold.test_year,
        candidate=candidate,
        preprocessor=preprocessor,
    )
    for row in history:
        row.update(
            {
                "market_type": fold.market,
                "test_year": fold.test_year,
                "candidate_id": candidate.candidate_id,
            }
        )
    metric_map = {row["probability_variant"]: row for row in metric_rows}
    fold_summary = {
        **asdict(fold),
        "candidate_id": candidate.candidate_id,
        "feature_set": candidate.feature_set,
        "feature_count": len(candidate.feature_columns),
        "l2_penalty": candidate.l2_penalty,
        "reused_phase4a": int(reused_phase4a),
        "train_rows": preprocessor.training_rows,
        "calibration_rows": int(calibration_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "train_down_rate": float(train_priors[0]),
        "train_flat_rate": float(train_priors[1]),
        "train_up_rate": float(train_priors[2]),
        "temperature": float(temperature),
        "prior_blend_alpha": float(blend_alpha),
        "calibration_raw_log_loss": float(raw_calibration_loss),
        "calibration_temperature_log_loss": float(scaled_calibration_loss),
        "calibration_prior_blend_log_loss": float(blend_calibration_loss),
        "raw_test_log_loss": metric_map["raw"]["log_loss"],
        "temperature_test_log_loss": metric_map["temperature"]["log_loss"],
        "prior_blend_test_log_loss": metric_map["prior_blend"]["log_loss"],
        "historical_test_log_loss": metric_map["historical_probability"]["log_loss"],
        "raw_test_ece": metric_map["raw"]["ece"],
        "temperature_test_ece": metric_map["temperature"]["ece"],
        "prior_blend_test_ece": metric_map["prior_blend"]["ece"],
        "historical_test_ece": metric_map["historical_probability"]["ece"],
        **monotonicity,
    }
    return {
        "phase4b_version": PHASE4B_VERSION,
        "candidate": candidate_to_dict(candidate),
        "fold": asdict(fold),
        "fold_summary": fold_summary,
        "metrics": metric_rows,
        "deciles": deciles,
        "calibration": calibration_rows,
        "coefficients": coefficients,
        "preprocessing": preprocessing,
        "training_history": history,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }


def collect_candidate_predictions(
    *,
    cache: MarketCache,
    mask: np.ndarray,
    feature_indices: np.ndarray,
    preprocessor: Preprocessor,
    model: SoftmaxModel,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.flatnonzero(mask)
    logits = np.empty((len(positions), len(LABELS)), dtype=np.float64)
    labels = np.empty(len(positions), dtype=np.uint8)
    returns = np.empty(len(positions), dtype=np.float64)
    for start in range(0, len(positions), chunk_size):
        end = min(len(positions), start + chunk_size)
        selected = positions[start:end]
        x = np.asarray(cache.features[selected], dtype=np.float64)
        x = preprocessor.transform(x[:, feature_indices])
        logits[start:end] = model.logits(x)
        labels[start:end] = cache.labels[selected]
        returns[start:end] = cache.returns[selected]
    return logits, labels, returns


def fit_prior_blend(
    *,
    probabilities: np.ndarray,
    labels: np.ndarray,
    priors: np.ndarray,
) -> tuple[float, float]:
    best_alpha = 0.0
    best_loss = math.inf
    for alpha in PRIOR_BLEND_ALPHAS:
        blended = blend_probabilities(probabilities, priors, alpha)
        loss = multiclass_log_loss(labels, blended)
        if loss < best_loss - 1e-12:
            best_alpha = alpha
            best_loss = loss
    return float(best_alpha), float(best_loss)


def blend_probabilities(
    probabilities: np.ndarray,
    priors: np.ndarray,
    alpha: float,
) -> np.ndarray:
    return (1.0 - alpha) * probabilities + alpha * priors.reshape(1, -1)


def select_development_candidates(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    summaries = [row["fold_summary"] for row in results]
    frame = pd.DataFrame(summaries)
    selections: dict[str, dict[str, Any]] = {}
    if frame.empty:
        return selections
    for market, market_frame in frame.groupby("market", sort=True):
        aggregates = []
        for candidate_id, group in market_frame.groupby("candidate_id", sort=True):
            aggregates.append(candidate_development_aggregate(group))
        eligible = [row for row in aggregates if row["development_ranking_supported"]]
        pool = eligible or aggregates
        selected = sorted(
            pool,
            key=lambda row: (
                -int(row["positive_spread_years"]),
                -float(row["median_high_minus_low_return"]),
                -float(row["weighted_high_minus_low_return"]),
                -float(row["raw_log_loss_improvement"]),
                str(row["candidate_id"]),
            ),
        )[0]
        selections[str(market)] = {
            "market_type": str(market),
            "selected_candidate_id": selected["candidate_id"],
            "selected_probability_variant": selected["development_probability_variant"],
            "development_ranking_supported": int(
                selected["development_ranking_supported"]
            ),
            **selected,
        }
    return selections


def candidate_development_aggregate(group: pd.DataFrame) -> dict[str, Any]:
    weights = group["test_rows"].to_numpy(dtype=np.float64)
    spreads = group["high_minus_low_return"].to_numpy(dtype=np.float64)
    variant_losses = {
        "raw": _weighted_values(group, "raw_test_log_loss", weights),
        "temperature": _weighted_values(
            group, "temperature_test_log_loss", weights
        ),
        "prior_blend": _weighted_values(group, "prior_blend_test_log_loss", weights),
    }
    probability_variant = min(
        variant_losses,
        key=lambda name: (variant_losses[name], name),
    )
    history_loss = _weighted_values(group, "historical_test_log_loss", weights)
    required_positive = max(1, math.ceil(len(group) * 0.75))
    positive_years = int((spreads > 0).sum())
    median_spread = float(np.median(spreads))
    weighted_spread = float(np.average(spreads, weights=weights))
    return {
        "candidate_id": str(group.iloc[0]["candidate_id"]),
        "feature_set": str(group.iloc[0]["feature_set"]),
        "feature_count": int(group.iloc[0]["feature_count"]),
        "l2_penalty": float(group.iloc[0]["l2_penalty"]),
        "development_fold_count": int(len(group)),
        "development_test_rows": int(group["test_rows"].sum()),
        "positive_spread_years": positive_years,
        "required_positive_spread_years": required_positive,
        "weighted_high_minus_low_return": weighted_spread,
        "median_high_minus_low_return": median_spread,
        "minimum_high_minus_low_return": float(spreads.min()),
        "maximum_high_minus_low_return": float(spreads.max()),
        "raw_log_loss": variant_losses["raw"],
        "temperature_log_loss": variant_losses["temperature"],
        "prior_blend_log_loss": variant_losses["prior_blend"],
        "historical_log_loss": history_loss,
        "raw_log_loss_improvement": history_loss - variant_losses["raw"],
        "development_probability_variant": probability_variant,
        "development_probability_log_loss": variant_losses[probability_variant],
        "development_ranking_supported": int(
            positive_years >= required_positive
            and median_spread > 0
            and weighted_spread > 0
        ),
    }


def build_confirmation_decisions(
    *,
    selections: dict[str, dict[str, Any]],
    confirmation_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    frame = pd.DataFrame([row["fold_summary"] for row in confirmation_results])
    decisions: list[dict[str, Any]] = []
    for market, selection in sorted(selections.items()):
        group = frame[frame["market"] == market]
        if group.empty:
            decisions.append(
                {
                    "market_type": market,
                    "selected_candidate_id": selection["selected_candidate_id"],
                    "decision": "INCOMPLETE",
                    "ranking_supported": 0,
                    "probability_supported": 0,
                    "reason": "沒有完整確認期結果",
                }
            )
            continue
        weights = group["test_rows"].to_numpy(dtype=np.float64)
        spreads = group["high_minus_low_return"].to_numpy(dtype=np.float64)
        required_positive = max(1, math.ceil(len(group) * 0.75))
        positive_years = int((spreads > 0).sum())
        weighted_spread = float(np.average(spreads, weights=weights))
        median_spread = float(np.median(spreads))
        ranking_supported = (
            positive_years >= required_positive
            and weighted_spread > 0
            and median_spread > 0
        )
        variant = str(selection["selected_probability_variant"])
        loss_column = {
            "raw": "raw_test_log_loss",
            "temperature": "temperature_test_log_loss",
            "prior_blend": "prior_blend_test_log_loss",
        }[variant]
        ece_column = {
            "raw": "raw_test_ece",
            "temperature": "temperature_test_ece",
            "prior_blend": "prior_blend_test_ece",
        }[variant]
        model_loss = _weighted_values(group, loss_column, weights)
        history_loss = _weighted_values(group, "historical_test_log_loss", weights)
        model_ece = _weighted_values(group, ece_column, weights)
        history_ece = _weighted_values(group, "historical_test_ece", weights)
        probability_supported = model_loss < history_loss and model_ece <= history_ece + 0.005
        if ranking_supported and probability_supported:
            decision = "PROBABILITY_AND_RANKING"
            reason = "確認期排序穩定，固定機率方案亦優於歷史機率基準"
        elif ranking_supported:
            decision = "RANKING_ONLY"
            reason = "確認期排序穩定，但機率品質尚未勝過歷史機率基準"
        else:
            decision = "REJECT"
            reason = "確認期高低組報酬方向不夠穩定"
        decisions.append(
            {
                "market_type": market,
                "selected_candidate_id": selection["selected_candidate_id"],
                "selected_probability_variant": variant,
                "decision": decision,
                "ranking_supported": int(ranking_supported),
                "probability_supported": int(probability_supported),
                "confirmation_fold_count": int(len(group)),
                "confirmation_test_rows": int(group["test_rows"].sum()),
                "positive_spread_years": positive_years,
                "required_positive_spread_years": required_positive,
                "weighted_high_minus_low_return": weighted_spread,
                "median_high_minus_low_return": median_spread,
                "minimum_high_minus_low_return": float(spreads.min()),
                "maximum_high_minus_low_return": float(spreads.max()),
                "model_log_loss": model_loss,
                "historical_log_loss": history_loss,
                "log_loss_improvement": history_loss - model_loss,
                "model_ece": model_ece,
                "historical_ece": history_ece,
                "reason": reason,
            }
        )
    return decisions


def export_phase4b_reports(
    *,
    output_dir: Path,
    run_signature: str,
    settings: Phase4Settings,
    caches: dict[str, MarketCache],
    development_results: list[dict[str, Any]],
    selections: dict[str, dict[str, Any]],
    confirmation_results: list[dict[str, Any]],
    completed: int,
    expected: int,
    failed: int,
) -> tuple[list[Path], bool]:
    all_results = [*development_results, *confirmation_results]
    fold_summaries = [row["fold_summary"] for row in all_results]
    metrics = [item for row in all_results for item in row["metrics"]]
    deciles = [item for row in all_results for item in row["deciles"]]
    calibration = [item for row in all_results for item in row["calibration"]]
    coefficients = [item for row in all_results for item in row["coefficients"]]
    preprocessing = [item for row in all_results for item in row["preprocessing"]]
    training_history = [item for row in all_results for item in row["training_history"]]
    development = _all_development_aggregates(development_results)
    selected_rows = list(selections.values())
    decisions = build_confirmation_decisions(
        selections=selections,
        confirmation_results=confirmation_results,
    )
    ready = any(
        row.get("decision") in {"RANKING_ONLY", "PROBABILITY_AND_RANKING"}
        for row in decisions
    )
    pipeline_status = "PASS" if completed == expected and failed == 0 else "FAIL"
    summary = [
        {"metric": "phase4b_version", "value": PHASE4B_VERSION},
        {"metric": "run_signature", "value": run_signature},
        {"metric": "pipeline_status", "value": pipeline_status},
        {"metric": "ready_for_phase4c", "value": int(ready)},
        {"metric": "completed_candidate_folds", "value": completed},
        {"metric": "expected_candidate_folds", "value": expected},
        {"metric": "failed_candidate_folds", "value": failed},
        {"metric": "candidate_count", "value": len(CANDIDATES)},
        {"metric": "development_end_year", "value": DEVELOPMENT_END_YEAR},
        {"metric": "confirmation_start_year", "value": CONFIRMATION_START_YEAR},
        {"metric": "maximum_epochs", "value": settings.maximum_epochs},
        {"metric": "batch_size", "value": settings.batch_size},
    ]
    for decision in decisions:
        market = decision["market_type"]
        summary.extend(
            [
                {"metric": f"{market}_decision", "value": decision["decision"]},
                {
                    "metric": f"{market}_selected_candidate",
                    "value": decision["selected_candidate_id"],
                },
                {
                    "metric": f"{market}_selected_probability_variant",
                    "value": decision.get("selected_probability_variant", ""),
                },
                {
                    "metric": f"{market}_confirmation_high_low_return",
                    "value": decision.get("weighted_high_minus_low_return", ""),
                },
                {
                    "metric": f"{market}_confirmation_log_loss_improvement",
                    "value": decision.get("log_loss_improvement", ""),
                },
            ]
        )
    for market, cache in sorted(caches.items()):
        summary.extend(
            [
                {"metric": f"{market}_source_rows", "value": cache.row_count},
                {"metric": f"{market}_source_sha256", "value": cache.source_sha256},
                {"metric": f"{market}_cache_signature", "value": cache.signature},
            ]
        )

    feature_sets = []
    for candidate in CANDIDATES:
        for order, feature in enumerate(candidate.feature_columns, start=1):
            feature_sets.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "feature_set": candidate.feature_set,
                    "l2_penalty": candidate.l2_penalty,
                    "feature_order": order,
                    "feature": feature,
                }
            )
    paths = [
        _write_csv(output_dir / "phase4b_summary.csv", summary),
        _write_csv(output_dir / "phase4b_candidate_development.csv", development),
        _write_csv(output_dir / "phase4b_selected_candidates.csv", selected_rows),
        _write_csv(output_dir / "phase4b_market_decisions.csv", decisions),
        _write_csv(output_dir / "phase4b_fold_summary.csv", fold_summaries),
        _write_csv(output_dir / "phase4b_metrics.csv", metrics),
        _write_csv(output_dir / "phase4b_index_deciles.csv", deciles),
        _write_csv(output_dir / "phase4b_calibration.csv", calibration),
        _write_csv(output_dir / "phase4b_coefficients.csv", coefficients),
        _write_csv(output_dir / "phase4b_preprocessing.csv", preprocessing),
        _write_csv(output_dir / "phase4b_training_history.csv", training_history),
        _write_csv(output_dir / "phase4b_feature_sets.csv", feature_sets),
    ]
    archive = output_dir / "phase4b_validation_reports.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in paths:
            handle.write(path, arcname=path.name)
    paths.append(archive)
    return paths, ready


def _all_development_aggregates(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    frame = pd.DataFrame([row["fold_summary"] for row in results])
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return rows
    for (market, candidate_id), group in frame.groupby(
        ["market", "candidate_id"], sort=True
    ):
        rows.append(
            {
                "market_type": market,
                **candidate_development_aggregate(group),
            }
        )
    return rows


def reuse_phase4a_fold(
    *,
    cache: MarketCache,
    fold: FoldSpec,
    candidate: CandidateSpec,
    settings: Phase4Settings,
    phase4a_run_root: Path,
) -> dict[str, Any] | None:
    candidates = sorted(
        phase4a_run_root.glob(f"*/folds/{fold.fold_id}.json.gz"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        config_path = path.parent.parent / "run_config.json"
        if not config_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            source = _read_gzip_json(path)
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            continue
        if config.get("phase4_version") != "phase4a-v1":
            continue
        source_settings = config.get("settings") or {}
        if not _phase4a_settings_match(source_settings, settings):
            continue
        if source.get("fold") != asdict(fold):
            continue
        preprocessor = reconstruct_phase4a_preprocessor(source)
        model = reconstruct_phase4a_model(source)
        summary = source["fold_summary"]
        priors = np.asarray(
            [
                summary["train_down_rate"],
                summary["train_flat_rate"],
                summary["train_up_rate"],
            ],
            dtype=np.float64,
        )
        return evaluate_candidate_fold(
            cache=cache,
            fold=fold,
            candidate=candidate,
            feature_indices=candidate.feature_indices,
            preprocessor=preprocessor,
            model=model,
            train_priors=priors,
            history=list(source.get("training_history") or []),
            settings=settings,
            reused_phase4a=True,
        )
    return None


def _phase4a_settings_match(source: dict[str, Any], target: Phase4Settings) -> bool:
    keys = (
        "clip_lower_quantile",
        "clip_upper_quantile",
        "quantile_sample_size",
        "training_chunk_size",
        "batch_size",
        "maximum_epochs",
        "minimum_epochs",
        "early_stopping_patience",
        "learning_rate",
        "l2_penalty",
        "random_seed",
    )
    target_values = asdict(target)
    return all(source.get(key) == target_values.get(key) for key in keys)


def reconstruct_phase4a_preprocessor(source: dict[str, Any]) -> Preprocessor:
    rows = source["preprocessing"]
    by_feature = {str(row["feature"]): row for row in rows}
    ordered = [by_feature[name] for name in FEATURE_COLUMNS]
    return Preprocessor(
        lower=np.asarray([row["clip_lower"] for row in ordered], dtype=np.float64),
        upper=np.asarray([row["clip_upper"] for row in ordered], dtype=np.float64),
        mean=np.asarray(
            [row["training_mean_after_clip"] for row in ordered], dtype=np.float64
        ),
        std=np.asarray(
            [row["training_std_after_clip"] for row in ordered], dtype=np.float64
        ),
        sampled_rows=int(ordered[0]["quantile_sample_rows"]),
        training_rows=int(ordered[0]["training_rows"]),
    )


def reconstruct_phase4a_model(source: dict[str, Any]) -> SoftmaxModel:
    rows = source["coefficients"]
    lookup = {
        (str(row["class_label"]), str(row["feature"])): float(
            row["standardized_coefficient"]
        )
        for row in rows
        if row["class_label"] in LABELS
    }
    weights = np.empty((len(FEATURE_COLUMNS), len(LABELS)), dtype=np.float64)
    intercept = np.empty(len(LABELS), dtype=np.float64)
    for class_index, label in enumerate(LABELS):
        intercept[class_index] = lookup[(label, "(intercept)")]
        for feature_index, feature in enumerate(FEATURE_COLUMNS):
            weights[feature_index, class_index] = lookup[(label, feature)]
    return SoftmaxModel(weights=weights, intercept=intercept)


def candidate_coefficient_rows(
    *,
    market: str,
    test_year: int,
    candidate: CandidateSpec,
    model: SoftmaxModel,
) -> list[dict[str, Any]]:
    rows = []
    for class_index, label in enumerate(LABELS):
        rows.append(
            {
                "market_type": market,
                "test_year": test_year,
                "candidate_id": candidate.candidate_id,
                "class_label": label,
                "feature": "(intercept)",
                "standardized_coefficient": float(model.intercept[class_index]),
            }
        )
        for feature_index, feature in enumerate(candidate.feature_columns):
            rows.append(
                {
                    "market_type": market,
                    "test_year": test_year,
                    "candidate_id": candidate.candidate_id,
                    "class_label": label,
                    "feature": feature,
                    "standardized_coefficient": float(
                        model.weights[feature_index, class_index]
                    ),
                }
            )
    up = LABEL_TO_INDEX["UP"]
    down = LABEL_TO_INDEX["DOWN"]
    for feature_index, feature in enumerate(candidate.feature_columns):
        rows.append(
            {
                "market_type": market,
                "test_year": test_year,
                "candidate_id": candidate.candidate_id,
                "class_label": "UP_MINUS_DOWN",
                "feature": feature,
                "standardized_coefficient": float(
                    model.weights[feature_index, up] - model.weights[feature_index, down]
                ),
            }
        )
    return rows


def candidate_preprocessing_rows(
    *,
    market: str,
    test_year: int,
    candidate: CandidateSpec,
    preprocessor: Preprocessor,
) -> list[dict[str, Any]]:
    return [
        {
            "market_type": market,
            "test_year": test_year,
            "candidate_id": candidate.candidate_id,
            "feature": feature,
            "clip_lower": float(preprocessor.lower[index]),
            "clip_upper": float(preprocessor.upper[index]),
            "training_mean_after_clip": float(preprocessor.mean[index]),
            "training_std_after_clip": float(preprocessor.std[index]),
            "quantile_sample_rows": preprocessor.sampled_rows,
            "training_rows": preprocessor.training_rows,
        }
        for index, feature in enumerate(candidate.feature_columns)
    ]


def candidate_to_dict(candidate: CandidateSpec) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "feature_set": candidate.feature_set,
        "feature_columns": list(candidate.feature_columns),
        "feature_count": len(candidate.feature_columns),
        "l2_penalty": candidate.l2_penalty,
    }


def candidate_by_id(candidate_id: str) -> CandidateSpec:
    for candidate in CANDIDATES:
        if candidate.candidate_id == candidate_id:
            return candidate
    raise KeyError(f"找不到 Phase 4B 候選：{candidate_id}")


def phase4b_config_signature(
    settings: Phase4Settings,
    caches: dict[str, MarketCache],
    markets: tuple[str, ...],
) -> str:
    payload = {
        "phase4b_version": PHASE4B_VERSION,
        "settings": asdict(settings),
        "markets": list(markets),
        "development_end_year": DEVELOPMENT_END_YEAR,
        "confirmation_start_year": CONFIRMATION_START_YEAR,
        "candidates": [candidate_to_dict(value) for value in CANDIDATES],
        "cache_signatures": {
            market: cache.signature for market, cache in sorted(caches.items())
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _result_file_name(fold: FoldSpec, candidate: CandidateSpec) -> str:
    return f"{fold.fold_id}__{candidate.candidate_id}.json.gz"


def _ensure_split(
    mask: np.ndarray,
    labels: np.ndarray,
    fold: FoldSpec,
    split: str,
) -> None:
    if not mask.any():
        raise RuntimeError(f"{fold.fold_id} {split} 無資料")
    observed = set(int(value) for value in np.unique(labels[mask]))
    required = set(range(len(LABELS)))
    if observed != required:
        raise RuntimeError(f"{fold.fold_id} {split} 缺少類別：{sorted(required - observed)}")


def _weighted_values(frame: pd.DataFrame, column: str, weights: np.ndarray) -> float:
    return float(np.average(frame[column].to_numpy(dtype=np.float64), weights=weights))


def _load_phase3_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"找不到 Phase 3 manifest：{path}")
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    return {str(row["file_name"]): row.to_dict() for _, row in frame.iterrows()}


def _ensure_phase3_ready(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"找不到 Phase 3B 稽核摘要：{path}")
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    values = {str(row["metric"]): str(row["value"]) for _, row in frame.iterrows()}
    if values.get("ready_for_modeling") != "1" or values.get("status") != "PASS":
        raise RuntimeError("Phase 3B 尚未通過，不可執行 Phase 4B")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return path
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = pd.DataFrame(rows, columns=columns)
        writer.to_csv(handle, index=False)
    os.replace(temporary, path)
    return path


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_gzip_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
    os.replace(temporary, path)


def _read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)
