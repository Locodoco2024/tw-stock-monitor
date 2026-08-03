from __future__ import annotations

import bisect
import gzip
import hashlib
import json
import math
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.institutional_model.adjusted_returns import calculate_holding_return
from research.institutional_model.corporate_actions import load_action_map
from research.institutional_model.database import ResearchDatabase
from research.institutional_model.phase4_model import (
    LABELS,
    LABEL_TO_INDEX,
    Preprocessor,
    SoftmaxModel,
    classification_metrics,
    fit_temperature,
    multiclass_log_loss,
    softmax,
)
from research.institutional_model.phase4_selection import (
    moving_block_bootstrap_mean_ci,
    resolve_phase3_shard_directory,
)
from research.institutional_model.market_model_spec import market_model_spec


PHASE4D_VERSION = "phase4d-v1"
DEFAULT_HORIZONS = (10, 20, 40)
LABEL_ROUND_DIGITS = 10
EPSILON = 1e-12


@dataclass(frozen=True)
class Phase4DHorizonSettings:
    target_market: str = "tpex"
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    label_threshold: float = 0.05
    minimum_daily_stocks: int = 50
    first_test_year: int = 2019
    minimum_training_years: int = 3
    calibration_years: int = 1
    clip_lower_quantile: float = 0.005
    clip_upper_quantile: float = 0.995
    quantile_sample_size: int = 250_000
    training_chunk_size: int = 100_000
    batch_size: int = 65_536
    maximum_epochs: int = 6
    minimum_epochs: int = 3
    early_stopping_patience: int = 2
    learning_rate: float = 0.02
    l2_penalty: float = 0.001
    bootstrap_iterations: int = 2_000
    bootstrap_block_months: int = 3
    random_seed: int = 20260728

    def validate(self) -> None:
        if self.target_market not in {"twse", "tpex"}:
            raise ValueError("Phase 4D target_market 只支援 twse 或 tpex")
        horizons = tuple(sorted(set(int(value) for value in self.horizons)))
        if horizons != self.horizons:
            raise ValueError("Phase 4D horizons 必須由小到大且不可重複")
        if 10 not in horizons or 20 not in horizons or 40 not in horizons:
            raise ValueError("Phase 4D 固定比較 10、20、40 個市場交易日")
        if self.label_threshold <= 0:
            raise ValueError("Phase 4D 標籤門檻必須大於 0")
        if self.minimum_daily_stocks < 10:
            raise ValueError("Phase 4D 每日最少股票數不可小於 10")
        if self.minimum_training_years < 2:
            raise ValueError("Phase 4D 最少訓練年度必須至少為 2")
        if self.calibration_years != 1:
            raise ValueError("Phase 4D 目前固定使用前一年度校準")
        if not 0 <= self.clip_lower_quantile < self.clip_upper_quantile <= 1:
            raise ValueError("Phase 4D 截尾分位數設定無效")
        if self.quantile_sample_size <= 0 or self.training_chunk_size <= 0:
            raise ValueError("Phase 4D chunk／sample size 必須大於 0")
        if self.batch_size <= 0:
            raise ValueError("Phase 4D batch size 必須大於 0")
        if self.maximum_epochs < self.minimum_epochs or self.minimum_epochs <= 0:
            raise ValueError("Phase 4D epoch 設定無效")
        if self.early_stopping_patience < 1:
            raise ValueError("Phase 4D early stopping patience 必須大於 0")
        if self.learning_rate <= 0 or self.l2_penalty < 0:
            raise ValueError("Phase 4D learning rate／L2 設定無效")
        if self.bootstrap_iterations < 200:
            raise ValueError("Phase 4D bootstrap 次數不可小於 200")
        if self.bootstrap_block_months < 1:
            raise ValueError("Phase 4D bootstrap 區塊月份必須大於 0")


@dataclass(frozen=True)
class HorizonFold:
    horizon_days: int
    train_start_year: int
    train_end_year: int
    calibration_year: int
    test_year: int
    market: str = "tpex"

    @property
    def fold_id(self) -> str:
        return f"{self.market}_{self.horizon_days}d_{self.test_year}"


@dataclass(frozen=True)
class Phase4DHorizonResult:
    status: str
    ready_for_horizon_decision: bool
    completed_folds: int
    expected_folds: int
    failed_folds: int
    output_paths: tuple[Path, ...]


def run_phase4d_horizon_research(
    *,
    database: ResearchDatabase,
    output_dir: Path | str,
    shard_root: Path | str,
    cache_root: Path | str,
    settings: Phase4DHorizonSettings | None = None,
    force: bool = False,
) -> Phase4DHorizonResult:
    """Compare 10/20/40-day labels for one market without rebuilding Phase 1-3."""
    config = settings or Phase4DHorizonSettings()
    config.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    shard_dir = resolve_phase3_shard_directory(
        output_dir=output,
        shard_root=Path(shard_root),
    )
    cache_dir = _resolve_cache_directory(
        database=database,
        output_dir=output,
        cache_root=Path(cache_root),
        settings=config,
        force=force,
    )
    frame = build_horizon_research_frame(
        database=database,
        shard_dir=shard_dir,
        cache_dir=cache_dir,
        settings=config,
        force=force,
    )
    reports = evaluate_horizon_frame(frame=frame, settings=config)
    paths = export_phase4d_reports(output_dir=output, reports=reports)
    failed = int(reports["fold_summary"]["status"].ne("complete").sum())
    completed = int(reports["fold_summary"]["status"].eq("complete").sum())
    expected = len(reports["fold_summary"])
    status = "PASS" if expected > 0 and failed == 0 else "FAIL"
    ready = status == "PASS" and bool(len(reports["summary"]))
    return Phase4DHorizonResult(
        status=status,
        ready_for_horizon_decision=ready,
        completed_folds=completed,
        expected_folds=expected,
        failed_folds=failed,
        output_paths=tuple(paths),
    )


def build_horizon_research_frame(
    *,
    database: ResearchDatabase,
    shard_dir: Path,
    cache_dir: Path,
    settings: Phase4DHorizonSettings,
    force: bool,
) -> pd.DataFrame:
    stocks = [
        dict(row)
        for row in database.query(
            """
            SELECT stock_id, stock_name, delisting_date
            FROM model_universe
            WHERE training_enabled=1 AND market_type=?
            ORDER BY stock_id
            """,
            (settings.target_market,),
        )
    ]
    if not stocks:
        raise RuntimeError(
            f"SQLite model_universe 沒有可訓練的 {settings.target_market.upper()} 股票"
        )
    source_columns = _phase3_source_columns(
        market_model_spec(settings.target_market).feature_columns
    )
    market_dates = [
        str(row["date"])
        for row in database.query("SELECT date FROM market_calendar ORDER BY date")
    ]
    if not market_dates:
        raise RuntimeError("市場交易日曆為空，無法計算 40 日結果")

    frames: list[pd.DataFrame] = []
    for position, stock in enumerate(stocks, start=1):
        stock_id = str(stock["stock_id"])
        shard_path = shard_dir / f"{stock_id}.csv.gz"
        if not shard_path.exists():
            continue
        shard = pd.read_csv(
            shard_path,
            compression="gzip",
            usecols=source_columns,
            dtype={"stock_id": "string", "signal_date": "string"},
            low_memory=False,
        )
        if shard.empty:
            continue
        shard = shard[
            (shard["market_type"].astype(str).str.lower() == settings.target_market)
            & (shard["feature_status"].astype(str) == "ok")
            & (pd.to_numeric(shard["liquidity_pass_20m"], errors="coerce") == 1)
        ].copy()
        if shard.empty:
            continue
        outcome_path = cache_dir / f"{stock_id}_40d.csv.gz"
        signal_dates = shard["signal_date"].astype(str).tolist()
        outcome: pd.DataFrame | None = None
        if not force and outcome_path.exists():
            try:
                cached = pd.read_csv(
                    outcome_path,
                    compression="gzip",
                    dtype={"stock_id": "string", "signal_date": "string"},
                    low_memory=False,
                )
                expected_keys = set(zip([stock_id] * len(signal_dates), signal_dates, strict=True))
                cached_keys = set(
                    zip(
                        cached["stock_id"].astype(str),
                        cached["signal_date"].astype(str),
                        strict=True,
                    )
                )
                if len(cached) == len(signal_dates) and cached_keys == expected_keys:
                    outcome = cached
            except (OSError, ValueError, KeyError):
                outcome = None
        if outcome is None:
            outcome = _calculate_stock_horizon_outcomes(
                database=database,
                stock_id=stock_id,
                signal_dates=signal_dates,
                market_dates=market_dates,
                horizon=40,
                threshold=settings.label_threshold,
            )
            _write_gzip_csv(outcome_path, outcome)
        shard = shard.merge(
            outcome,
            on=["stock_id", "signal_date"],
            how="left",
            validate="one_to_one",
        )
        frames.append(shard)
        if position % 100 == 0 or position == len(stocks):
            print(f"Phase 4D 資料準備：{position}/{len(stocks)}")

    if not frames:
        raise RuntimeError(
            f"Phase 4D 沒有可用的 {settings.target_market.upper()} Phase 3 分片"
        )
    result = pd.concat(frames, ignore_index=True)
    for horizon in (10, 20):
        returns = pd.to_numeric(
            result[f"adjusted_return_{horizon}d"], errors="coerce"
        )
        statuses = result[f"label_status_{horizon}d"].astype(str)
        result[f"label_{horizon}d"] = [
            classify_horizon_return(value, threshold=settings.label_threshold)
            if status == "ok" and np.isfinite(value)
            else ""
            for value, status in zip(returns.to_numpy(), statuses, strict=True)
        ]
    _validate_research_frame(result, settings)
    return result


def evaluate_horizon_frame(
    *,
    frame: pd.DataFrame,
    settings: Phase4DHorizonSettings,
) -> dict[str, pd.DataFrame]:
    settings.validate()
    label_distribution = horizon_label_distribution(frame, settings=settings)
    fold_summaries: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    daily_group_rows: list[dict[str, Any]] = []
    daily_spread_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    purge_rows: list[dict[str, Any]] = []

    for horizon in settings.horizons:
        horizon_frame = _eligible_horizon_frame(
            frame,
            horizon,
            feature_columns=market_model_spec(settings.target_market).feature_columns,
        )
        folds = build_horizon_folds(horizon_frame, horizon=horizon, settings=settings)
        if not folds:
            raise RuntimeError(f"{horizon} 日沒有足夠年度建立時間滾動折")
        for fold in folds:
            try:
                result = evaluate_horizon_fold(
                    frame=horizon_frame,
                    fold=fold,
                    settings=settings,
                )
                fold_summaries.append(result["fold_summary"])
                metric_rows.extend(result["metrics"])
                daily_group_rows.extend(result["daily_groups"])
                daily_spread_rows.extend(result["daily_spreads"])
                yearly_rows.append(result["yearly_ranking"])
                coefficient_rows.extend(result["coefficients"])
                training_rows.extend(result["training_history"])
                purge_rows.append(result["boundary_purge"])
            except Exception as exc:
                fold_summaries.append(
                    {
                        **asdict(fold),
                        "fold_id": fold.fold_id,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    fold_summary = pd.DataFrame(fold_summaries)
    metrics = pd.DataFrame(metric_rows)
    daily_groups = pd.DataFrame(daily_group_rows)
    daily_spreads = pd.DataFrame(daily_spread_rows)
    yearly = pd.DataFrame(yearly_rows)
    bootstrap = horizon_bootstrap_report(daily_spreads, settings=settings)
    coefficients = pd.DataFrame(coefficient_rows)
    coefficient_stability = coefficient_stability_report(coefficients)
    training_history = pd.DataFrame(training_rows)
    boundary_purge = pd.DataFrame(purge_rows)
    summary = build_horizon_summary(
        frame=frame,
        fold_summary=fold_summary,
        bootstrap=bootstrap,
        settings=settings,
    )
    return {
        "label_distribution": label_distribution,
        "fold_summary": fold_summary,
        "metrics": metrics,
        "daily_groups": daily_groups,
        "daily_spreads": daily_spreads,
        "yearly": yearly,
        "bootstrap": bootstrap,
        "coefficients": coefficients,
        "coefficient_stability": coefficient_stability,
        "training_history": training_history,
        "boundary_purge": boundary_purge,
        "summary": summary,
    }


def build_horizon_folds(
    frame: pd.DataFrame,
    *,
    horizon: int,
    settings: Phase4DHorizonSettings,
) -> list[HorizonFold]:
    available = sorted(int(value) for value in frame["signal_year"].unique())
    if not available:
        return []
    first_year = available[0]
    earliest = first_year + settings.minimum_training_years + settings.calibration_years
    start = max(settings.first_test_year, earliest)
    available_set = set(available)
    result: list[HorizonFold] = []
    for test_year in range(start, available[-1] + 1):
        calibration_year = test_year - 1
        train_end = calibration_year - 1
        if test_year not in available_set or calibration_year not in available_set:
            continue
        training_years = set(range(first_year, train_end + 1)).intersection(available_set)
        if len(training_years) < settings.minimum_training_years:
            continue
        result.append(
            HorizonFold(
                horizon_days=horizon,
                market=settings.target_market,
                train_start_year=first_year,
                train_end_year=train_end,
                calibration_year=calibration_year,
                test_year=test_year,
            )
        )
    return result


def evaluate_horizon_fold(
    *,
    frame: pd.DataFrame,
    fold: HorizonFold,
    settings: Phase4DHorizonSettings,
) -> dict[str, Any]:
    train_mask, calibration_mask, test_mask, purge = build_purged_masks(frame, fold)
    labels = frame["label_index"].to_numpy(dtype=np.uint8)
    _ensure_all_classes(labels[train_mask], fold.fold_id, "train")
    _ensure_all_classes(labels[calibration_mask], fold.fold_id, "calibration")
    _ensure_all_classes(labels[test_mask], fold.fold_id, "test")
    feature_columns = market_model_spec(settings.target_market).feature_columns
    features = frame[list(feature_columns)].to_numpy(dtype=np.float64)
    seed = settings.random_seed + fold.horizon_days * 100 + fold.test_year
    preprocessor = fit_mask_preprocessor(
        features=features,
        train_mask=train_mask,
        settings=settings,
        seed=seed,
    )
    model, history, priors = train_mask_softmax_model(
        features=features,
        labels=labels,
        train_mask=train_mask,
        calibration_mask=calibration_mask,
        preprocessor=preprocessor,
        settings=settings,
        seed=seed,
    )
    calibration_logits = model.logits(
        preprocessor.transform(features[calibration_mask])
    )
    calibration_labels = labels[calibration_mask]
    temperature, calibration_raw_loss, calibration_scaled_loss = fit_temperature(
        calibration_logits,
        calibration_labels,
    )
    test_logits = model.logits(preprocessor.transform(features[test_mask]))
    test_labels = labels[test_mask]
    raw_probabilities = softmax(test_logits)
    temperature_probabilities = softmax(test_logits / temperature)
    historical_probabilities = np.tile(priors, (int(test_mask.sum()), 1))

    metrics: list[dict[str, Any]] = []
    for variant, probabilities in (
        ("raw", raw_probabilities),
        ("temperature", temperature_probabilities),
        ("historical_probability", historical_probabilities),
    ):
        metrics.append(
            {
                "fold_id": fold.fold_id,
                "horizon_days": fold.horizon_days,
                "test_year": fold.test_year,
                "probability_variant": variant,
                **classification_metrics(test_labels, probabilities),
            }
        )

    test_frame = frame.loc[
        test_mask, ["stock_id", "signal_date", "adjusted_return", "label"]
    ].copy()
    test_frame["p_down_raw"] = raw_probabilities[:, LABEL_TO_INDEX["DOWN"]]
    test_frame["p_flat_raw"] = raw_probabilities[:, LABEL_TO_INDEX["FLAT"]]
    test_frame["p_up_raw"] = raw_probabilities[:, LABEL_TO_INDEX["UP"]]
    test_frame["institutional_index_raw"] = 100.0 * (
        test_frame["p_up_raw"] - test_frame["p_down_raw"]
    )
    test_frame = assign_same_day_ranks(
        test_frame,
        minimum_daily_stocks=settings.minimum_daily_stocks,
    )
    if test_frame.empty:
        raise RuntimeError(f"{fold.fold_id} 沒有符合每日最低股票數的測試日")
    daily_groups = daily_rank_group_rows(
        test_frame,
        horizon=fold.horizon_days,
        test_year=fold.test_year,
    )
    daily_spreads = daily_spread_rows(
        test_frame,
        horizon=fold.horizon_days,
        test_year=fold.test_year,
    )
    yearly_ranking = yearly_ranking_row(
        test_frame,
        horizon=fold.horizon_days,
        test_year=fold.test_year,
    )
    raw_metric = next(row for row in metrics if row["probability_variant"] == "raw")
    history_metric = next(
        row for row in metrics if row["probability_variant"] == "historical_probability"
    )
    predicted = np.argmax(raw_probabilities, axis=1)
    fold_summary = {
        **asdict(fold),
        "fold_id": fold.fold_id,
        "status": "complete",
        "error": "",
        "train_rows": int(train_mask.sum()),
        "calibration_rows": int(calibration_mask.sum()),
        "test_rows_before_daily_filter": int(test_mask.sum()),
        "test_rows": int(len(test_frame)),
        "test_signal_dates": int(test_frame["signal_date"].nunique()),
        "train_down_rate": float(priors[LABEL_TO_INDEX["DOWN"]]),
        "train_flat_rate": float(priors[LABEL_TO_INDEX["FLAT"]]),
        "train_up_rate": float(priors[LABEL_TO_INDEX["UP"]]),
        "test_down_rate": float((test_labels == LABEL_TO_INDEX["DOWN"]).mean()),
        "test_flat_rate": float((test_labels == LABEL_TO_INDEX["FLAT"]).mean()),
        "test_up_rate": float((test_labels == LABEL_TO_INDEX["UP"]).mean()),
        "predicted_down_rate": float((predicted == LABEL_TO_INDEX["DOWN"]).mean()),
        "predicted_flat_rate": float((predicted == LABEL_TO_INDEX["FLAT"]).mean()),
        "predicted_up_rate": float((predicted == LABEL_TO_INDEX["UP"]).mean()),
        "temperature": float(temperature),
        "calibration_raw_log_loss": float(calibration_raw_loss),
        "calibration_temperature_log_loss": float(calibration_scaled_loss),
        "raw_test_log_loss": float(raw_metric["log_loss"]),
        "historical_test_log_loss": float(history_metric["log_loss"]),
        "raw_log_loss_improvement": float(
            history_metric["log_loss"] - raw_metric["log_loss"]
        ),
        "raw_test_brier": float(raw_metric["brier_score"]),
        "historical_test_brier": float(history_metric["brier_score"]),
        "raw_test_ece": float(raw_metric["ece"]),
        "historical_test_ece": float(history_metric["ece"]),
        "raw_test_macro_f1": float(raw_metric["macro_f1"]),
        "raw_test_balanced_accuracy": float(raw_metric["balanced_accuracy"]),
        "epochs_completed": len(history),
        **yearly_ranking,
    }
    coefficients = coefficient_report_rows(
        model=model,
        fold=fold,
        feature_columns=market_model_spec(settings.target_market).feature_columns,
    )
    for row in history:
        row.update(
            {
                "fold_id": fold.fold_id,
                "horizon_days": fold.horizon_days,
                "test_year": fold.test_year,
            }
        )
    return {
        "fold_summary": fold_summary,
        "metrics": metrics,
        "daily_groups": daily_groups,
        "daily_spreads": daily_spreads,
        "yearly_ranking": yearly_ranking,
        "coefficients": coefficients,
        "training_history": history,
        "boundary_purge": purge,
    }


def build_purged_masks(
    frame: pd.DataFrame,
    fold: HorizonFold,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    years = frame["signal_year"].to_numpy(dtype=np.int16)
    target_dates = pd.to_datetime(frame["target_date"], errors="coerce")
    calibration_start = pd.Timestamp(f"{fold.calibration_year}-01-01")
    test_start = pd.Timestamp(f"{fold.test_year}-01-01")
    base_train = (years >= fold.train_start_year) & (years <= fold.train_end_year)
    base_calibration = years == fold.calibration_year
    test_mask = years == fold.test_year
    train_mask = base_train & (target_dates < calibration_start).to_numpy()
    calibration_mask = base_calibration & (target_dates < test_start).to_numpy()
    return (
        train_mask,
        calibration_mask,
        test_mask,
        {
            "fold_id": fold.fold_id,
            "horizon_days": fold.horizon_days,
            "test_year": fold.test_year,
            "train_rows_before_purge": int(base_train.sum()),
            "train_rows_after_purge": int(train_mask.sum()),
            "train_rows_purged": int(base_train.sum() - train_mask.sum()),
            "calibration_rows_before_purge": int(base_calibration.sum()),
            "calibration_rows_after_purge": int(calibration_mask.sum()),
            "calibration_rows_purged": int(
                base_calibration.sum() - calibration_mask.sum()
            ),
            "test_rows": int(test_mask.sum()),
            "calibration_start_date": str(calibration_start.date()),
            "test_start_date": str(test_start.date()),
        },
    )


def fit_mask_preprocessor(
    *,
    features: np.ndarray,
    train_mask: np.ndarray,
    settings: Phase4DHorizonSettings,
    seed: int,
) -> Preprocessor:
    positions = np.flatnonzero(train_mask)
    if positions.size == 0:
        raise RuntimeError("Phase 4D 訓練折沒有資料")
    rng = np.random.default_rng(seed)
    sample_count = min(settings.quantile_sample_size, len(positions))
    selected = np.sort(rng.choice(positions, size=sample_count, replace=False))
    sample = features[selected]
    lower = np.quantile(sample, settings.clip_lower_quantile, axis=0)
    upper = np.quantile(sample, settings.clip_upper_quantile, axis=0)
    upper = np.maximum(upper, lower)
    clipped = np.clip(features[train_mask], lower, upper)
    mean = clipped.mean(axis=0)
    std = clipped.std(axis=0)
    std[std < 1e-8] = 1.0
    return Preprocessor(
        lower=lower,
        upper=upper,
        mean=mean,
        std=std,
        sampled_rows=sample_count,
        training_rows=int(train_mask.sum()),
    )


def train_mask_softmax_model(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    calibration_mask: np.ndarray,
    preprocessor: Preprocessor,
    settings: Phase4DHorizonSettings,
    seed: int,
) -> tuple[SoftmaxModel, list[dict[str, Any]], np.ndarray]:
    train_positions = np.flatnonzero(train_mask)
    calibration_positions = np.flatnonzero(calibration_mask)
    counts = np.bincount(labels[train_positions], minlength=len(LABELS)).astype(np.float64)
    priors = counts / counts.sum()
    feature_count = features.shape[1]
    weights = np.zeros((feature_count, len(LABELS)), dtype=np.float64)
    intercept = np.log(np.maximum(priors, EPSILON))
    intercept -= intercept.mean()
    first_w = np.zeros_like(weights)
    second_w = np.zeros_like(weights)
    first_b = np.zeros_like(intercept)
    second_b = np.zeros_like(intercept)
    beta1 = 0.9
    beta2 = 0.999
    adam_epsilon = 1e-8
    update_count = 0
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    best_loss = math.inf
    best_weights = weights.copy()
    best_intercept = intercept.copy()
    stale_epochs = 0

    for epoch in range(1, settings.maximum_epochs + 1):
        positions = rng.permutation(train_positions)
        loss_total = 0.0
        example_total = 0
        batches = 0
        for start in range(0, len(positions), settings.batch_size):
            selected = positions[start : start + settings.batch_size]
            x = preprocessor.transform(features[selected])
            y = labels[selected].astype(np.int64, copy=False)
            probabilities = softmax(x @ weights + intercept)
            loss = multiclass_log_loss(y, probabilities)
            loss_total += loss * len(y)
            example_total += len(y)
            batches += 1
            gradient = probabilities
            gradient[np.arange(len(y)), y] -= 1.0
            gradient /= len(y)
            gradient_w = x.T @ gradient + settings.l2_penalty * weights
            gradient_b = gradient.sum(axis=0)
            update_count += 1
            first_w = beta1 * first_w + (1 - beta1) * gradient_w
            second_w = beta2 * second_w + (1 - beta2) * np.square(gradient_w)
            first_b = beta1 * first_b + (1 - beta1) * gradient_b
            second_b = beta2 * second_b + (1 - beta2) * np.square(gradient_b)
            correction1 = 1 - beta1**update_count
            correction2 = 1 - beta2**update_count
            weights -= settings.learning_rate * (first_w / correction1) / (
                np.sqrt(second_w / correction2) + adam_epsilon
            )
            intercept -= settings.learning_rate * (first_b / correction1) / (
                np.sqrt(second_b / correction2) + adam_epsilon
            )
            intercept -= intercept.mean()

        model = SoftmaxModel(weights=weights, intercept=intercept)
        calibration_probabilities = softmax(
            model.logits(preprocessor.transform(features[calibration_positions]))
        )
        calibration_loss = multiclass_log_loss(
            labels[calibration_positions], calibration_probabilities
        )
        training_loss = loss_total / max(example_total, 1)
        history.append(
            {
                "epoch": epoch,
                "training_log_loss": float(training_loss),
                "calibration_log_loss": float(calibration_loss),
                "batches": batches,
                "examples": example_total,
            }
        )
        if calibration_loss < best_loss - 1e-6:
            best_loss = calibration_loss
            best_weights = weights.copy()
            best_intercept = intercept.copy()
            stale_epochs = 0
        else:
            stale_epochs += 1
        if (
            epoch >= settings.minimum_epochs
            and stale_epochs >= settings.early_stopping_patience
        ):
            break
    return SoftmaxModel(best_weights, best_intercept), history, priors


def assign_same_day_ranks(
    frame: pd.DataFrame,
    *,
    minimum_daily_stocks: int,
) -> pd.DataFrame:
    result = frame.copy()
    counts = result.groupby("signal_date")["stock_id"].transform("size")
    result = result[counts >= minimum_daily_stocks].copy()
    if result.empty:
        return result
    result["daily_rank"] = result.groupby("signal_date")[
        "institutional_index_raw"
    ].rank(method="first", ascending=True)
    result["daily_count"] = result.groupby("signal_date")["stock_id"].transform("size")
    result["daily_percentile"] = np.where(
        result["daily_count"] <= 1,
        100.0,
        100.0 * (result["daily_rank"] - 1.0) / (result["daily_count"] - 1.0),
    )
    result["daily_decile"] = np.minimum(
        10,
        np.floor(result["daily_percentile"] / 10.0).astype(int) + 1,
    )
    return result


def daily_rank_group_rows(
    frame: pd.DataFrame,
    *,
    horizon: int,
    test_year: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (signal_date, decile), group in frame.groupby(
        ["signal_date", "daily_decile"], sort=True
    ):
        returns = group["adjusted_return"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "horizon_days": horizon,
                "test_year": test_year,
                "signal_date": str(signal_date),
                "daily_decile": int(decile),
                "sample_count": int(len(group)),
                "average_return": float(np.mean(returns)),
                "median_return": float(np.median(returns)),
                "up_5pct_rate": float((returns >= 0.05).mean()),
                "down_5pct_rate": float((returns <= -0.05).mean()),
                "positive_return_rate": float((returns > 0).mean()),
                "average_index": float(group["institutional_index_raw"].mean()),
            }
        )
    return rows


def daily_spread_rows(
    frame: pd.DataFrame,
    *,
    horizon: int,
    test_year: int,
) -> list[dict[str, Any]]:
    top = frame[frame["daily_percentile"] > 80]
    bottom = frame[frame["daily_percentile"] <= 20]
    top_daily = top.groupby("signal_date")["adjusted_return"].mean()
    bottom_daily = bottom.groupby("signal_date")["adjusted_return"].mean()
    aligned = pd.concat(
        [top_daily.rename("top"), bottom_daily.rename("bottom")],
        axis=1,
        join="inner",
    ).dropna()
    return [
        {
            "horizon_days": horizon,
            "test_year": test_year,
            "signal_date": str(signal_date),
            "top20_average_return": float(row["top"]),
            "bottom20_average_return": float(row["bottom"]),
            "top20_minus_bottom20": float(row["top"] - row["bottom"]),
        }
        for signal_date, row in aligned.iterrows()
    ]


def horizon_bootstrap_report(
    daily_spreads: pd.DataFrame,
    *,
    settings: Phase4DHorizonSettings,
) -> pd.DataFrame:
    if daily_spreads.empty:
        return pd.DataFrame()
    latest_year = int(daily_spreads["test_year"].max())
    periods = (
        ("all", daily_spreads),
        ("development", daily_spreads[daily_spreads["test_year"] <= 2022]),
        ("confirmation", daily_spreads[daily_spreads["test_year"] >= 2023]),
        (
            "confirmation_ex_latest",
            daily_spreads[
                (daily_spreads["test_year"] >= 2023)
                & (daily_spreads["test_year"] < latest_year)
            ],
        ),
    )
    rows: list[dict[str, Any]] = []
    for period_name, period in periods:
        for horizon, group in period.groupby("horizon_days", sort=True):
            if group.empty:
                continue
            daily = group.set_index("signal_date")["top20_minus_bottom20"].sort_index()
            monthly = daily.groupby(daily.index.astype(str).str[:7]).mean().sort_index()
            if len(monthly) < 2:
                continue
            seed = _stable_seed(
                settings.random_seed,
                f"{period_name}:{int(horizon)}",
            )
            lower, upper, bootstrap_mean = moving_block_bootstrap_mean_ci(
                monthly.to_numpy(dtype=np.float64),
                iterations=settings.bootstrap_iterations,
                block_length=settings.bootstrap_block_months,
                random_seed=seed,
            )
            rows.append(
                {
                    "period": period_name,
                    "horizon_days": int(horizon),
                    "daily_observations": int(len(daily)),
                    "monthly_blocks": int(len(monthly)),
                    "point_estimate_daily_spread": float(daily.mean()),
                    "monthly_mean_spread": float(monthly.mean()),
                    "bootstrap_mean_spread": bootstrap_mean,
                    "confidence_level": 0.95,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "ci_excludes_zero_positive": int(lower > 0),
                    "iterations": settings.bootstrap_iterations,
                    "block_months": min(
                        settings.bootstrap_block_months, len(monthly)
                    ),
                }
            )
    return pd.DataFrame(rows)


def yearly_ranking_row(
    frame: pd.DataFrame,
    *,
    horizon: int,
    test_year: int,
) -> dict[str, Any]:
    top = frame[frame["daily_percentile"] > 80]
    bottom = frame[frame["daily_percentile"] <= 20]
    top_daily = top.groupby("signal_date")["adjusted_return"].mean()
    bottom_daily = bottom.groupby("signal_date")["adjusted_return"].mean()
    aligned = pd.concat(
        [top_daily.rename("top"), bottom_daily.rename("bottom")],
        axis=1,
        join="inner",
    ).dropna()
    spread = aligned["top"] - aligned["bottom"]
    deciles = (
        frame.groupby("daily_decile", as_index=False)
        .agg(average_return=("adjusted_return", "mean"))
        .sort_values("daily_decile")
    )
    correlation = _safe_correlation(
        deciles["daily_decile"].to_numpy(dtype=np.float64),
        deciles["average_return"].to_numpy(dtype=np.float64),
    )
    violations = int(
        (np.diff(deciles["average_return"].to_numpy(dtype=np.float64)) < 0).sum()
    )
    return {
        "horizon_days": horizon,
        "test_year": test_year,
        "signal_dates": int(frame["signal_date"].nunique()),
        "sample_count": int(len(frame)),
        "top20_average_return": float(top_daily.mean()),
        "bottom20_average_return": float(bottom_daily.mean()),
        "top20_minus_bottom20": float(spread.mean()),
        "positive_spread_day_rate": float((spread > 0).mean()),
        "decile_return_correlation": correlation,
        "decile_monotonic_violations": violations,
    }


def horizon_label_distribution(
    frame: pd.DataFrame,
    *,
    settings: Phase4DHorizonSettings,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    common_mask = np.ones(len(frame), dtype=bool)
    for horizon in settings.horizons:
        common_mask &= frame[f"label_{horizon}d"].isin(LABELS).to_numpy()
    for horizon in settings.horizons:
        for cohort, selected in (
            ("horizon_available", frame[frame[f"label_{horizon}d"].isin(LABELS)]),
            ("common_10_20_40", frame[common_mask]),
        ):
            labels = selected[f"label_{horizon}d"]
            total = len(labels)
            for label in LABELS:
                count = int((labels == label).sum())
                rows.append(
                    {
                        "horizon_days": horizon,
                        "cohort": cohort,
                        "label": label,
                        "sample_count": count,
                        "sample_rate": count / max(total, 1),
                        "total_rows": total,
                        "first_signal_date": str(selected["signal_date"].min())
                        if total
                        else "",
                        "last_signal_date": str(selected["signal_date"].max())
                        if total
                        else "",
                    }
                )
    return pd.DataFrame(rows)


def build_horizon_summary(
    *,
    frame: pd.DataFrame,
    fold_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    settings: Phase4DHorizonSettings,
) -> pd.DataFrame:
    complete = fold_summary[fold_summary["status"] == "complete"].copy()
    failed_folds = int(fold_summary["status"].ne("complete").sum())
    pipeline_status = (
        "PASS" if len(fold_summary) > 0 and failed_folds == 0 else "FAIL"
    )
    rows: list[dict[str, Any]] = [
        {"metric": "pipeline_status", "value": pipeline_status},
        {"metric": "phase4d_version", "value": PHASE4D_VERSION},
        {"metric": "market_type", "value": settings.target_market},
        {
            "metric": "candidate_id",
            "value": market_model_spec(settings.target_market).candidate_id,
        },
        {"metric": "horizons", "value": ",".join(map(str, settings.horizons))},
        {"metric": "label_threshold", "value": settings.label_threshold},
        {"metric": "source_rows", "value": len(frame)},
        {"metric": "source_stocks", "value": frame["stock_id"].nunique()},
        {"metric": "automatic_horizon_promotion", "value": 0},
        {"metric": "failed_folds", "value": failed_folds},
        {
            "metric": "ready_for_horizon_decision",
            "value": int(pipeline_status == "PASS"),
        },
    ]
    latest_year = int(complete["test_year"].max()) if not complete.empty else 0
    for horizon in settings.horizons:
        horizon_rows = complete[complete["horizon_days"] == horizon]
        for period_name, selected in (
            ("confirmation", horizon_rows[horizon_rows["test_year"] >= 2023]),
            (
                "confirmation_ex_latest",
                horizon_rows[
                    (horizon_rows["test_year"] >= 2023)
                    & (horizon_rows["test_year"] < latest_year)
                ],
            ),
        ):
            if selected.empty:
                continue
            weights = selected["test_rows"].to_numpy(dtype=np.float64)
            prefix = f"{horizon}d_{period_name}"
            values = {
                "test_rows": int(selected["test_rows"].sum()),
                "flat_rate": float(np.average(selected["test_flat_rate"], weights=weights)),
                "up_rate": float(np.average(selected["test_up_rate"], weights=weights)),
                "down_rate": float(np.average(selected["test_down_rate"], weights=weights)),
                "raw_log_loss": float(
                    np.average(selected["raw_test_log_loss"], weights=weights)
                ),
                "historical_log_loss": float(
                    np.average(selected["historical_test_log_loss"], weights=weights)
                ),
                "raw_log_loss_improvement": float(
                    np.average(selected["raw_log_loss_improvement"], weights=weights)
                ),
                "balanced_accuracy": float(
                    np.average(selected["raw_test_balanced_accuracy"], weights=weights)
                ),
                "macro_f1": float(
                    np.average(selected["raw_test_macro_f1"], weights=weights)
                ),
                "top20_minus_bottom20": float(
                    np.average(selected["top20_minus_bottom20"], weights=weights)
                ),
                "positive_spread_years": int(
                    (selected["top20_minus_bottom20"] > 0).sum()
                ),
                "total_years": int(len(selected)),
                "decile_return_correlation": float(
                    np.average(selected["decile_return_correlation"], weights=weights)
                ),
            }
            rows.extend(
                {"metric": f"{prefix}_{key}", "value": value}
                for key, value in values.items()
            )
    if not bootstrap.empty:
        for _, row in bootstrap.iterrows():
            if row["period"] not in {"confirmation", "confirmation_ex_latest"}:
                continue
            prefix = f"{int(row['horizon_days'])}d_{row['period']}_bootstrap"
            rows.extend(
                [
                    {"metric": f"{prefix}_ci_lower", "value": row["ci_lower"]},
                    {"metric": f"{prefix}_ci_upper", "value": row["ci_upper"]},
                    {
                        "metric": f"{prefix}_ci_excludes_zero_positive",
                        "value": row["ci_excludes_zero_positive"],
                    },
                ]
            )
    return pd.DataFrame(rows)


def coefficient_stability_report(coefficients: pd.DataFrame) -> pd.DataFrame:
    if coefficients.empty:
        return pd.DataFrame()
    feature_rows = coefficients[coefficients["feature_name"] != "__INTERCEPT__"]
    records: list[dict[str, Any]] = []
    for horizon, horizon_frame in feature_rows.groupby("horizon_days", sort=True):
        pivot = horizon_frame.pivot_table(
            index=["test_year", "feature_name"],
            columns="class_label",
            values="coefficient",
            aggfunc="first",
        ).reset_index()
        pivot["up_minus_down"] = pivot["UP"] - pivot["DOWN"]
        matrix = pivot.pivot(
            index="test_year", columns="feature_name", values="up_minus_down"
        ).sort_index()
        years = list(matrix.index.astype(int))
        pairwise: list[float] = []
        adjacent: list[float] = []
        values = matrix.to_numpy(dtype=np.float64)
        for left in range(len(years)):
            for right in range(left + 1, len(years)):
                pairwise.append(_safe_correlation(values[left], values[right]))
                if right == left + 1:
                    adjacent.append(pairwise[-1])
        consistent_features = 0
        high_consistency_features = 0
        required = max(1, math.ceil(len(years) * 0.875))
        for feature in matrix.columns:
            series = matrix[feature].dropna().to_numpy(dtype=np.float64)
            positive = int((series > 0).sum())
            negative = int((series < 0).sum())
            zero = int((series == 0).sum())
            dominant = max(positive, negative)
            sign_rate = dominant / max(len(series), 1)
            if dominant == len(series):
                consistent_features += 1
            if dominant >= required:
                high_consistency_features += 1
            records.append(
                {
                    "row_type": "feature",
                    "horizon_days": int(horizon),
                    "feature_name": str(feature),
                    "test_years": len(series),
                    "mean_up_minus_down": float(series.mean()),
                    "std_up_minus_down": float(series.std(ddof=0)),
                    "positive_years": positive,
                    "negative_years": negative,
                    "zero_years": zero,
                    "dominant_sign_rate": sign_rate,
                    "average_pairwise_correlation": "",
                    "average_adjacent_correlation": "",
                    "minimum_pairwise_correlation": "",
                    "fully_consistent_features": "",
                    "high_consistency_features": "",
                }
            )
        records.append(
            {
                "row_type": "summary",
                "horizon_days": int(horizon),
                "feature_name": "__SUMMARY__",
                "test_years": len(years),
                "mean_up_minus_down": "",
                "std_up_minus_down": "",
                "positive_years": "",
                "negative_years": "",
                "zero_years": "",
                "dominant_sign_rate": "",
                "average_pairwise_correlation": float(np.mean(pairwise))
                if pairwise
                else 0.0,
                "average_adjacent_correlation": float(np.mean(adjacent))
                if adjacent
                else 0.0,
                "minimum_pairwise_correlation": float(np.min(pairwise))
                if pairwise
                else 0.0,
                "fully_consistent_features": consistent_features,
                "high_consistency_features": high_consistency_features,
            }
        )
    return pd.DataFrame(records)


def coefficient_report_rows(
    *,
    model: SoftmaxModel,
    fold: HorizonFold,
    feature_columns: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for class_index, class_label in enumerate(LABELS):
        rows.append(
            {
                "fold_id": fold.fold_id,
                "horizon_days": fold.horizon_days,
                "test_year": fold.test_year,
                "class_label": class_label,
                "feature_name": "__INTERCEPT__",
                "coefficient": float(model.intercept[class_index]),
            }
        )
        for feature_index, feature_name in enumerate(feature_columns):
            rows.append(
                {
                    "fold_id": fold.fold_id,
                    "horizon_days": fold.horizon_days,
                    "test_year": fold.test_year,
                    "class_label": class_label,
                    "feature_name": feature_name,
                    "coefficient": float(model.weights[feature_index, class_index]),
                }
            )
    return rows


def classify_horizon_return(value: Any, *, threshold: float) -> str:
    try:
        rounded = round(float(value), LABEL_ROUND_DIGITS)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(rounded):
        return ""
    rounded_threshold = round(float(threshold), LABEL_ROUND_DIGITS)
    if rounded >= rounded_threshold:
        return "UP"
    if rounded <= -rounded_threshold:
        return "DOWN"
    return "FLAT"


def export_phase4d_reports(
    *,
    output_dir: Path,
    reports: dict[str, pd.DataFrame],
) -> list[Path]:
    paths = [
        _write_csv(
            output_dir / "phase4d_horizon_label_distribution.csv",
            reports["label_distribution"],
        ),
        _write_csv(output_dir / "phase4d_fold_summary.csv", reports["fold_summary"]),
        _write_csv(output_dir / "phase4d_fold_metrics.csv", reports["metrics"]),
        _write_gzip_csv(
            output_dir / "phase4d_daily_rank_groups.csv.gz",
            reports["daily_groups"],
        ),
        _write_gzip_csv(
            output_dir / "phase4d_daily_spreads.csv.gz",
            reports["daily_spreads"],
        ),
        _write_csv(output_dir / "phase4d_yearly_ranking.csv", reports["yearly"]),
        _write_csv(
            output_dir / "phase4d_bootstrap_confidence.csv",
            reports["bootstrap"],
        ),
        _write_csv(output_dir / "phase4d_coefficients.csv", reports["coefficients"]),
        _write_csv(
            output_dir / "phase4d_coefficient_stability.csv",
            reports["coefficient_stability"],
        ),
        _write_csv(
            output_dir / "phase4d_training_history.csv",
            reports["training_history"],
        ),
        _write_csv(
            output_dir / "phase4d_boundary_purge.csv",
            reports["boundary_purge"],
        ),
        _write_csv(output_dir / "phase4d_summary.csv", reports["summary"]),
    ]
    archive = output_dir / "phase4d_horizon_validation_reports.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in paths:
            handle.write(path, arcname=path.name)
    paths.append(archive)
    return paths


def _eligible_horizon_frame(
    frame: pd.DataFrame,
    horizon: int,
    *,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    selected = frame[frame[f"label_{horizon}d"].isin(LABELS)].copy()
    selected["label"] = selected[f"label_{horizon}d"].astype(str)
    selected["label_index"] = selected["label"].map(LABEL_TO_INDEX).astype(np.uint8)
    selected["adjusted_return"] = pd.to_numeric(
        selected[f"adjusted_return_{horizon}d"], errors="coerce"
    )
    selected["target_date"] = selected[f"target_date_{horizon}d"].astype(str)
    selected["signal_year"] = pd.to_numeric(
        selected["signal_year"], errors="raise"
    ).astype(np.int16)
    for feature in feature_columns:
        selected[feature] = pd.to_numeric(selected[feature], errors="coerce")
    required = ["adjusted_return", *feature_columns]
    if selected[required].isna().any().any():
        raise RuntimeError(f"{horizon} 日研究資料包含 NaN")
    values = selected[required].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{horizon} 日研究資料包含 Infinity")
    return selected.reset_index(drop=True)


def _calculate_stock_horizon_outcomes(
    *,
    database: ResearchDatabase,
    stock_id: str,
    signal_dates: list[str],
    market_dates: list[str],
    horizon: int,
    threshold: float,
) -> pd.DataFrame:
    delisting_date = database.scalar(
        "SELECT MAX(date) FROM delistings WHERE stock_id=?",
        (stock_id,),
    )
    price_rows = [
        dict(row)
        for row in database.query(
            """
            SELECT date, open, close, high, low, trading_volume, trading_money
            FROM stock_prices
            WHERE stock_id=? AND (? IS NULL OR date < ?)
            ORDER BY date
            """,
            (stock_id, delisting_date, delisting_date),
        )
    ]
    price_by_date = {str(row["date"]): row for row in price_rows}
    action_map = load_action_map(database, stock_id)
    rows: list[dict[str, Any]] = []
    for signal_date in signal_dates:
        entry_index = bisect.bisect_right(market_dates, signal_date)
        target_index = entry_index + horizon - 1
        base = {
            "stock_id": stock_id,
            "signal_date": signal_date,
            f"entry_date_{horizon}d": "",
            f"target_date_{horizon}d": "",
            f"adjusted_return_{horizon}d": np.nan,
            f"max_adjusted_return_{horizon}d": np.nan,
            f"min_adjusted_return_{horizon}d": np.nan,
            f"label_status_{horizon}d": "insufficient_future_data",
            f"label_{horizon}d": "",
        }
        if entry_index >= len(market_dates) or target_index >= len(market_dates):
            rows.append(base)
            continue
        entry_date = market_dates[entry_index]
        target_date = market_dates[target_index]
        base[f"entry_date_{horizon}d"] = entry_date
        base[f"target_date_{horizon}d"] = target_date
        if delisting_date and target_date >= str(delisting_date):
            base[f"label_status_{horizon}d"] = "delisted_before_target"
            rows.append(base)
            continue
        entry = price_by_date.get(entry_date)
        target = price_by_date.get(target_date)
        if not _valid_entry_row(entry):
            base[f"label_status_{horizon}d"] = "unavailable_entry_price"
            rows.append(base)
            continue
        if not _valid_close_row(target):
            base[f"label_status_{horizon}d"] = "unavailable_target_price"
            rows.append(base)
            continue
        calendar_path = market_dates[entry_index : target_index + 1]
        path = [
            price_by_date[value]
            for value in calendar_path
            if _valid_close_row(price_by_date.get(value))
        ]
        missing_action_dates = [
            value
            for value in action_map
            if entry_date < value <= target_date
            and not _valid_close_row(price_by_date.get(value))
        ]
        if missing_action_dates:
            base[f"label_status_{horizon}d"] = "invalid_data"
            rows.append(base)
            continue
        try:
            result = calculate_holding_return(path, action_map)
        except ValueError:
            base[f"label_status_{horizon}d"] = "invalid_data"
            rows.append(base)
            continue
        base.update(
            {
                f"adjusted_return_{horizon}d": result.adjusted_return,
                f"max_adjusted_return_{horizon}d": result.max_adjusted_return,
                f"min_adjusted_return_{horizon}d": result.min_adjusted_return,
                f"label_status_{horizon}d": "ok",
                f"label_{horizon}d": classify_horizon_return(
                    result.adjusted_return,
                    threshold=threshold,
                ),
            }
        )
        rows.append(base)
    return pd.DataFrame(rows)


def _resolve_cache_directory(
    *,
    database: ResearchDatabase,
    output_dir: Path,
    cache_root: Path,
    settings: Phase4DHorizonSettings,
    force: bool,
) -> Path:
    summary = pd.read_csv(
        output_dir / "phase3_summary.csv",
        encoding="utf-8-sig",
        dtype=str,
    ).fillna("")
    metric_map = {str(row["metric"]): str(row["value"]) for _, row in summary.iterrows()}
    phase3_signature = metric_map.get("config_signature", "")
    stat = database.path.stat()
    wal_path = Path(str(database.path) + "-wal")
    wal_stat = wal_path.stat() if wal_path.exists() else None
    source_markers = {
        "maximum_market_date": database.scalar("SELECT MAX(date) FROM market_calendar"),
        "maximum_target_price_date": database.scalar(
            """
            SELECT MAX(p.date)
            FROM stock_prices p
            JOIN model_universe u ON u.stock_id=p.stock_id
            WHERE u.training_enabled=1 AND u.market_type=?
            """,
            (settings.target_market,),
        ),
        "target_price_rows": database.scalar(
            """
            SELECT COUNT(*)
            FROM stock_prices p
            JOIN model_universe u ON u.stock_id=p.stock_id
            WHERE u.training_enabled=1 AND u.market_type=?
            """,
            (settings.target_market,),
        ),
        "target_corporate_action_rows": database.scalar(
            """
            SELECT COUNT(*)
            FROM corporate_actions a
            JOIN model_universe u ON u.stock_id=a.stock_id
            WHERE u.training_enabled=1 AND u.market_type=?
            """,
            (settings.target_market,),
        ),
    }
    payload = {
        "version": PHASE4D_VERSION,
        "target_market": settings.target_market,
        "phase3_signature": phase3_signature,
        "database_size": stat.st_size,
        "database_mtime_ns": stat.st_mtime_ns,
        "database_wal_size": wal_stat.st_size if wal_stat else 0,
        "database_wal_mtime_ns": wal_stat.st_mtime_ns if wal_stat else 0,
        "source_markers": source_markers,
        "horizons": settings.horizons,
        "threshold": settings.label_threshold,
        "features": market_model_spec(settings.target_market).feature_columns,
    }
    signature = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    directory = cache_root / signature[:16]
    if force and directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "manifest.json"
    manifest.write_text(
        json.dumps({"signature": signature, **payload}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return directory


def _phase3_source_columns(feature_columns: tuple[str, ...]) -> list[str]:
    columns = [
        "stock_id",
        "stock_name",
        "market_type",
        "signal_date",
        "signal_year",
        "feature_status",
        "liquidity_pass_20m",
    ]
    for horizon in (10, 20):
        columns.extend(
            [
                f"target_date_{horizon}d",
                f"adjusted_return_{horizon}d",
                f"max_adjusted_return_{horizon}d",
                f"min_adjusted_return_{horizon}d",
                f"label_status_{horizon}d",
            ]
        )
    columns.extend(feature_columns)
    return columns


def _validate_research_frame(
    frame: pd.DataFrame,
    settings: Phase4DHorizonSettings,
) -> None:
    duplicate_count = int(frame.duplicated(["stock_id", "signal_date"]).sum())
    if duplicate_count:
        raise RuntimeError(f"Phase 4D 股票＋訊號日重複：{duplicate_count}")
    for horizon in settings.horizons:
        status = frame[f"label_status_{horizon}d"].astype(str)
        returns = pd.to_numeric(frame[f"adjusted_return_{horizon}d"], errors="coerce")
        invalid = (status == "ok") & (~np.isfinite(returns))
        if invalid.any():
            raise RuntimeError(f"Phase 4D {horizon} 日 status=ok 但報酬無效")


def _ensure_all_classes(values: np.ndarray, fold_id: str, split: str) -> None:
    observed = set(int(value) for value in np.unique(values))
    missing = set(range(len(LABELS))).difference(observed)
    if missing:
        labels = [LABELS[index] for index in sorted(missing)]
        raise RuntimeError(f"{fold_id} {split} 缺少類別：{labels}")


def _valid_entry_row(row: dict[str, Any] | None) -> bool:
    return bool(
        row
        and _positive(row.get("open"))
        and _positive(row.get("close"))
        and _positive(row.get("trading_volume"))
    )


def _valid_close_row(row: dict[str, Any] | None) -> bool:
    return bool(row and _positive(row.get("close")))


def _positive(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 2:
        return 0.0
    left_values = left[mask]
    right_values = right[mask]
    if np.std(left_values) <= EPSILON or np.std(right_values) <= EPSILON:
        return 0.0
    return float(np.corrcoef(left_values, right_values)[0, 1])


def _stable_seed(base_seed: int, value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], byteorder="big", signed=False)
    return (base_seed + offset) % (2**32 - 1)


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _write_gzip_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8-sig", newline="") as handle:
        frame.to_csv(handle, index=False)
    temporary.replace(path)
    return path
