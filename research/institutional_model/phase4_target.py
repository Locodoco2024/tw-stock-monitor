from __future__ import annotations

import hashlib
import math
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.institutional_model.database import ResearchDatabase
from research.institutional_model.phase4_horizon import (
    EPSILON,
    HorizonFold,
    Phase4DHorizonSettings,
    build_horizon_folds,
    build_horizon_research_frame,
    build_purged_masks,
    fit_mask_preprocessor,
    train_mask_softmax_model,
)
from research.institutional_model.phase4_model import (
    LABELS,
    LABEL_TO_INDEX,
    Preprocessor,
    softmax,
)
from research.institutional_model.phase4_selection import (
    moving_block_bootstrap_mean_ci,
    resolve_phase3_shard_directory,
)
from research.institutional_model.market_model_spec import market_model_spec


PHASE4E_VERSION = "phase4e-v1"
PRIMARY_HORIZON_DAYS = 20
EXTENSION_HORIZON_DAYS = 40
FEATURE_GROUPS = (
    "foreign",
    "investment_trust",
    "dealer_self",
    "institutional_consensus",
)


@dataclass(frozen=True)
class Phase4ETargetSettings:
    target_market: str = "tpex"
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
    ranking_l2_penalty: float = 0.001
    bootstrap_iterations: int = 2_000
    bootstrap_block_months: int = 3
    random_seed: int = 20260729

    def validate(self) -> None:
        if self.target_market not in {"twse", "tpex"}:
            raise ValueError("Phase 4E target_market 只支援 twse 或 tpex")
        if self.label_threshold <= 0:
            raise ValueError("Phase 4E 標籤門檻必須大於 0")
        if self.minimum_daily_stocks < 10:
            raise ValueError("Phase 4E 每日最少股票數不可小於 10")
        if self.minimum_training_years < 2:
            raise ValueError("Phase 4E 最少訓練年度必須至少為 2")
        if self.calibration_years != 1:
            raise ValueError("Phase 4E 目前固定使用前一年度校準")
        if not 0 <= self.clip_lower_quantile < self.clip_upper_quantile <= 1:
            raise ValueError("Phase 4E 截尾分位數設定無效")
        if self.quantile_sample_size <= 0 or self.training_chunk_size <= 0:
            raise ValueError("Phase 4E chunk／sample size 必須大於 0")
        if self.batch_size <= 0:
            raise ValueError("Phase 4E batch size 必須大於 0")
        if self.maximum_epochs < self.minimum_epochs or self.minimum_epochs <= 0:
            raise ValueError("Phase 4E epoch 設定無效")
        if self.early_stopping_patience < 1:
            raise ValueError("Phase 4E early stopping patience 必須大於 0")
        if self.learning_rate <= 0:
            raise ValueError("Phase 4E learning rate 必須大於 0")
        if self.l2_penalty < 0 or self.ranking_l2_penalty < 0:
            raise ValueError("Phase 4E L2 不可小於 0")
        if self.bootstrap_iterations < 200:
            raise ValueError("Phase 4E bootstrap 次數不可小於 200")
        if self.bootstrap_block_months < 1:
            raise ValueError("Phase 4E bootstrap 區塊月份必須大於 0")


@dataclass(frozen=True)
class BinaryLogisticModel:
    weights: np.ndarray
    intercept: float

    def logits(self, features: np.ndarray) -> np.ndarray:
        return features @ self.weights + self.intercept

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        return sigmoid(self.logits(features))


@dataclass(frozen=True)
class LinearRankModel:
    weights: np.ndarray
    intercept: float

    def scores(self, features: np.ndarray) -> np.ndarray:
        return features @ self.weights + self.intercept


@dataclass(frozen=True)
class BinaryCalibrator:
    slope: float
    intercept: float

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        return sigmoid(self.slope * logits + self.intercept)


@dataclass(frozen=True)
class Phase4ETargetResult:
    status: str
    ready_for_target_decision: bool
    completed_folds: int
    expected_folds: int
    failed_folds: int
    output_paths: tuple[Path, ...]


def run_phase4e_target_research(
    *,
    database: ResearchDatabase,
    output_dir: Path | str,
    shard_root: Path | str,
    cache_root: Path | str,
    settings: Phase4ETargetSettings | None = None,
    force: bool = False,
) -> Phase4ETargetResult:
    """Compare 20-day UP/DOWN probability targets and same-day return ranking."""
    config = settings or Phase4ETargetSettings()
    config.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    shard_dir = resolve_phase3_shard_directory(
        output_dir=output,
        shard_root=Path(shard_root),
    )
    horizon_settings = _to_horizon_settings(config)
    frame = build_horizon_research_frame(
        database=database,
        shard_dir=shard_dir,
        cache_dir=Path(cache_root),
        settings=horizon_settings,
        force=force,
    )
    reports = evaluate_target_frame(frame=frame, settings=config)
    paths = export_phase4e_reports(output_dir=output, reports=reports)
    fold_summary = reports["fold_summary"]
    failed = int(fold_summary["status"].ne("complete").sum())
    completed = int(fold_summary["status"].eq("complete").sum())
    expected = len(fold_summary)
    status = "PASS" if expected > 0 and failed == 0 else "FAIL"
    return Phase4ETargetResult(
        status=status,
        ready_for_target_decision=status == "PASS" and bool(len(reports["summary"])),
        completed_folds=completed,
        expected_folds=expected,
        failed_folds=failed,
        output_paths=tuple(paths),
    )


def evaluate_target_frame(
    *,
    frame: pd.DataFrame,
    settings: Phase4ETargetSettings,
) -> dict[str, pd.DataFrame]:
    settings.validate()
    selected = prepare_target_frame(frame, settings=settings)
    folds = build_horizon_folds(
        selected,
        horizon=PRIMARY_HORIZON_DAYS,
        settings=_to_horizon_settings(settings),
    )
    if not folds:
        raise RuntimeError("Phase 4E 沒有足夠年度建立滾動驗證")

    fold_rows: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    daily_spread_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    contribution_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    purge_rows: list[dict[str, Any]] = []
    oos_frames: list[pd.DataFrame] = []

    for fold in folds:
        try:
            result = evaluate_target_fold(
                frame=selected,
                fold=fold,
                settings=settings,
            )
            fold_rows.append(result["fold_summary"])
            probability_rows.extend(result["probability_metrics"])
            threshold_rows.extend(result["threshold_metrics"])
            calibration_rows.extend(result["calibration"])
            yearly_rows.extend(result["yearly_ranking"])
            daily_spread_rows.extend(result["daily_spreads"])
            coefficient_rows.extend(result["coefficients"])
            contribution_rows.extend(result["group_contributions"])
            history_rows.extend(result["training_history"])
            purge_rows.append(result["boundary_purge"])
            oos_frames.append(result["oos_scores"])
        except Exception as exc:
            fold_rows.append(
                {
                    **asdict(fold),
                    "fold_id": fold.fold_id,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    fold_summary = pd.DataFrame(fold_rows)
    probability_metrics = pd.DataFrame(probability_rows)
    threshold_metrics = pd.DataFrame(threshold_rows)
    calibration = pd.DataFrame(calibration_rows)
    yearly_ranking = pd.DataFrame(yearly_rows)
    daily_spreads = pd.DataFrame(daily_spread_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    group_contributions = pd.DataFrame(contribution_rows)
    training_history = pd.DataFrame(history_rows)
    boundary_purge = pd.DataFrame(purge_rows)
    oos_scores = pd.concat(oos_frames, ignore_index=True) if oos_frames else pd.DataFrame()
    bootstrap = target_bootstrap_report(daily_spreads, settings=settings)
    coefficient_stability = target_coefficient_stability(coefficients)
    comparison = build_target_comparison(
        probability_metrics=probability_metrics,
        threshold_metrics=threshold_metrics,
        yearly_ranking=yearly_ranking,
        daily_spreads=daily_spreads,
        oos_scores=oos_scores,
    )
    summary = build_target_summary(
        frame=selected,
        fold_summary=fold_summary,
        comparison=comparison,
        bootstrap=bootstrap,
        settings=settings,
    )
    return {
        "fold_summary": fold_summary,
        "probability_metrics": probability_metrics,
        "threshold_metrics": threshold_metrics,
        "calibration": calibration,
        "yearly_ranking": yearly_ranking,
        "daily_spreads": daily_spreads,
        "bootstrap": bootstrap,
        "coefficients": coefficients,
        "coefficient_stability": coefficient_stability,
        "group_contributions": group_contributions,
        "training_history": training_history,
        "boundary_purge": boundary_purge,
        "oos_scores": oos_scores,
        "comparison": comparison,
        "summary": summary,
    }


def prepare_target_frame(
    frame: pd.DataFrame,
    *,
    settings: Phase4ETargetSettings,
) -> pd.DataFrame:
    selected = frame[
        (frame["label_status_20d"].astype(str) == "ok")
        & pd.to_numeric(frame["adjusted_return_20d"], errors="coerce").notna()
    ].copy()
    selected["adjusted_return_20d"] = pd.to_numeric(
        selected["adjusted_return_20d"], errors="coerce"
    )
    selected["adjusted_return_40d"] = pd.to_numeric(
        selected["adjusted_return_40d"], errors="coerce"
    )
    selected["target_date"] = selected["target_date_20d"].astype(str)
    selected["adjusted_return"] = selected["adjusted_return_20d"]
    selected["label"] = selected["label_20d"].astype(str)
    selected["label_index"] = selected["label"].map(LABEL_TO_INDEX)
    if selected["label_index"].isna().any():
        raise RuntimeError("Phase 4E 包含無效 20 日三分類標籤")
    selected["label_index"] = selected["label_index"].astype(np.uint8)
    rounded = selected["adjusted_return_20d"].round(10)
    threshold = round(settings.label_threshold, 10)
    selected["target_up"] = (rounded >= threshold).astype(np.uint8)
    selected["target_down"] = (rounded <= -threshold).astype(np.uint8)
    counts = selected.groupby("signal_date")["stock_id"].transform("size")
    selected = selected[counts >= settings.minimum_daily_stocks].copy()
    if selected.empty:
        raise RuntimeError("Phase 4E 沒有符合每日最低股票數的資料")
    selected["future_return_rank_pct_20d"] = selected.groupby("signal_date")[
        "adjusted_return_20d"
    ].rank(method="average", pct=True)
    selected["actual_top20_20d"] = (
        selected["future_return_rank_pct_20d"] > 0.80
    ).astype(np.uint8)
    selected["actual_bottom20_20d"] = (
        selected["future_return_rank_pct_20d"] <= 0.20
    ).astype(np.uint8)
    return selected.reset_index(drop=True)


def evaluate_target_fold(
    *,
    frame: pd.DataFrame,
    fold: HorizonFold,
    settings: Phase4ETargetSettings,
) -> dict[str, Any]:
    train_mask, calibration_mask, test_mask, purge = build_purged_masks(frame, fold)
    feature_columns = market_model_spec(settings.target_market).feature_columns
    features = frame[list(feature_columns)].to_numpy(dtype=np.float64)
    seed = settings.random_seed + fold.test_year
    preprocessor = fit_mask_preprocessor(
        features=features,
        train_mask=train_mask,
        settings=_to_horizon_settings(settings),
        seed=seed,
    )

    multiclass_model, multiclass_history, multiclass_priors = train_mask_softmax_model(
        features=features,
        labels=frame["label_index"].to_numpy(dtype=np.uint8),
        train_mask=train_mask,
        calibration_mask=calibration_mask,
        preprocessor=preprocessor,
        settings=_to_horizon_settings(settings),
        seed=seed + 1,
    )
    up_model, up_history, up_prevalence = train_binary_logistic_model(
        features=features,
        labels=frame["target_up"].to_numpy(dtype=np.uint8),
        train_mask=train_mask,
        calibration_mask=calibration_mask,
        preprocessor=preprocessor,
        settings=settings,
        seed=seed + 2,
        target_name="up",
    )
    down_model, down_history, down_prevalence = train_binary_logistic_model(
        features=features,
        labels=frame["target_down"].to_numpy(dtype=np.uint8),
        train_mask=train_mask,
        calibration_mask=calibration_mask,
        preprocessor=preprocessor,
        settings=settings,
        seed=seed + 3,
        target_name="down",
    )
    rank_model = fit_linear_rank_model(
        features=features,
        targets=frame["future_return_rank_pct_20d"].to_numpy(dtype=np.float64),
        train_mask=train_mask,
        preprocessor=preprocessor,
        settings=settings,
    )

    calibration_positions = np.flatnonzero(calibration_mask)
    test_positions = np.flatnonzero(test_mask)
    x_calibration = preprocessor.transform(features[calibration_positions])
    x_test = preprocessor.transform(features[test_positions])
    up_calibration_logits = up_model.logits(x_calibration)
    down_calibration_logits = down_model.logits(x_calibration)
    up_calibrator = fit_binary_calibrator(
        up_calibration_logits,
        frame.loc[calibration_mask, "target_up"].to_numpy(dtype=np.uint8),
    )
    down_calibrator = fit_binary_calibrator(
        down_calibration_logits,
        frame.loc[calibration_mask, "target_down"].to_numpy(dtype=np.uint8),
    )

    up_raw = up_model.probabilities(x_test)
    down_raw = down_model.probabilities(x_test)
    up_platt = up_calibrator.transform_logits(up_model.logits(x_test))
    down_platt = down_calibrator.transform_logits(down_model.logits(x_test))
    up_historical = np.full(len(test_positions), up_prevalence, dtype=np.float64)
    down_historical = np.full(len(test_positions), down_prevalence, dtype=np.float64)
    multiclass_probabilities = softmax(multiclass_model.logits(x_test))
    multinomial_index = 100.0 * (
        multiclass_probabilities[:, LABEL_TO_INDEX["UP"]]
        - multiclass_probabilities[:, LABEL_TO_INDEX["DOWN"]]
    )
    rank_score = rank_model.scores(x_test)
    binary_net = up_platt - down_platt

    probability_metrics: list[dict[str, Any]] = []
    threshold_metrics: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for target_name, y_column, raw, calibrated, historical, calibration_logits, calibrator in (
        (
            "up_20d",
            "target_up",
            up_raw,
            up_platt,
            up_historical,
            up_calibration_logits,
            up_calibrator,
        ),
        (
            "down_20d",
            "target_down",
            down_raw,
            down_platt,
            down_historical,
            down_calibration_logits,
            down_calibrator,
        ),
    ):
        y_test = frame.loc[test_mask, y_column].to_numpy(dtype=np.uint8)
        y_calibration = frame.loc[calibration_mask, y_column].to_numpy(dtype=np.uint8)
        for variant, probabilities in (
            ("raw", raw),
            ("platt", calibrated),
            ("historical_probability", historical),
        ):
            probability_metrics.append(
                {
                    "fold_id": fold.fold_id,
                    "test_year": fold.test_year,
                    "target_name": target_name,
                    "probability_variant": variant,
                    **binary_probability_metrics(y_test, probabilities),
                    "platt_slope": calibrator.slope if variant == "platt" else "",
                    "platt_intercept": calibrator.intercept if variant == "platt" else "",
                }
            )
            calibration_rows.extend(
                binary_calibration_rows(
                    y_test,
                    probabilities,
                    fold_id=fold.fold_id,
                    test_year=fold.test_year,
                    target_name=target_name,
                    probability_variant=variant,
                )
            )
        calibration_probabilities = calibrator.transform_logits(calibration_logits)
        threshold = select_balanced_accuracy_threshold(y_calibration, calibration_probabilities)
        threshold_metrics.append(
            {
                "fold_id": fold.fold_id,
                "test_year": fold.test_year,
                "target_name": target_name,
                "selected_threshold": threshold,
                **binary_threshold_metrics(y_test, calibrated, threshold),
            }
        )

    test_frame = frame.loc[
        test_mask,
        [
            "stock_id",
            "stock_name",
            "signal_date",
            "signal_year",
            "adjusted_return_20d",
            "adjusted_return_40d",
            "label_status_40d",
            "target_up",
            "target_down",
            "future_return_rank_pct_20d",
            "actual_top20_20d",
            "actual_bottom20_20d",
        ],
    ].copy()
    test_frame["test_year"] = fold.test_year
    test_frame["p_up_raw"] = up_raw
    test_frame["p_up_platt"] = up_platt
    test_frame["p_down_raw"] = down_raw
    test_frame["p_down_platt"] = down_platt
    test_frame["binary_net_score"] = binary_net
    test_frame["return_rank_score"] = rank_score
    test_frame["multinomial_index"] = multinomial_index

    score_variants = (
        "multinomial_index",
        "p_up_platt",
        "binary_net_score",
        "return_rank_score",
    )
    yearly_ranking: list[dict[str, Any]] = []
    daily_spreads: list[dict[str, Any]] = []
    ranked_frames: dict[str, pd.DataFrame] = {}
    for score_variant in score_variants:
        ranked = assign_score_ranks(
            test_frame,
            score_column=score_variant,
            minimum_daily_stocks=settings.minimum_daily_stocks,
        )
        ranked_frames[score_variant] = ranked
        for evaluation_horizon in (PRIMARY_HORIZON_DAYS, EXTENSION_HORIZON_DAYS):
            return_column = f"adjusted_return_{evaluation_horizon}d"
            valid = ranked[np.isfinite(pd.to_numeric(ranked[return_column], errors="coerce"))].copy()
            if evaluation_horizon == EXTENSION_HORIZON_DAYS:
                valid = valid[valid["label_status_40d"].astype(str) == "ok"].copy()
            if valid.empty:
                continue
            yearly_ranking.append(
                ranking_summary_row(
                    valid,
                    score_variant=score_variant,
                    return_column=return_column,
                    evaluation_horizon=evaluation_horizon,
                    test_year=fold.test_year,
                )
            )
            daily_spreads.extend(
                ranking_daily_spread_rows(
                    valid,
                    score_variant=score_variant,
                    return_column=return_column,
                    evaluation_horizon=evaluation_horizon,
                    test_year=fold.test_year,
                )
            )

    coefficients = []
    coefficients.extend(
        binary_coefficient_rows(
            up_model,
            fold=fold,
            model_target="up_20d",
            feature_columns=feature_columns,
        )
    )
    coefficients.extend(
        binary_coefficient_rows(
            down_model,
            fold=fold,
            model_target="down_20d",
            feature_columns=feature_columns,
        )
    )
    coefficients.extend(
        rank_coefficient_rows(
            rank_model, fold=fold, feature_columns=feature_columns
        )
    )
    group_contributions = group_contribution_rows(
        x_test=x_test,
        test_frame=test_frame,
        models={
            "up_20d": up_model.weights,
            "down_20d": down_model.weights,
            "return_rank_20d": rank_model.weights,
        },
        fold=fold,
        settings=settings,
        feature_columns=feature_columns,
    )

    training_history: list[dict[str, Any]] = []
    for model_target, rows in (
        ("multinomial_20d", multiclass_history),
        ("up_20d", up_history),
        ("down_20d", down_history),
    ):
        for row in rows:
            training_history.append(
                {
                    **row,
                    "fold_id": fold.fold_id,
                    "test_year": fold.test_year,
                    "model_target": model_target,
                }
            )

    primary_ranked = ranked_frames["binary_net_score"].copy()
    oos_columns = [
        "stock_id",
        "stock_name",
        "signal_date",
        "signal_year",
        "test_year",
        "adjusted_return_20d",
        "adjusted_return_40d",
        "target_up",
        "target_down",
        "future_return_rank_pct_20d",
        "p_up_raw",
        "p_up_platt",
        "p_down_raw",
        "p_down_platt",
        "binary_net_score",
        "return_rank_score",
        "multinomial_index",
    ]
    oos_scores = primary_ranked[oos_columns].copy()
    for score_variant in score_variants:
        ranked = ranked_frames[score_variant].set_index(["stock_id", "signal_date"])
        keys = pd.MultiIndex.from_frame(oos_scores[["stock_id", "signal_date"]])
        oos_scores[f"{score_variant}_daily_percentile"] = ranked.loc[
            keys, "daily_percentile"
        ].to_numpy()
        oos_scores[f"{score_variant}_daily_decile"] = ranked.loc[
            keys, "daily_decile"
        ].to_numpy(dtype=np.int8)

    fold_summary = {
        **asdict(fold),
        "fold_id": fold.fold_id,
        "status": "complete",
        "error": "",
        "train_rows": int(train_mask.sum()),
        "calibration_rows": int(calibration_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "test_signal_dates": int(test_frame["signal_date"].nunique()),
        "train_up_rate": up_prevalence,
        "train_down_rate": down_prevalence,
        "train_flat_rate": float(multiclass_priors[LABEL_TO_INDEX["FLAT"]]),
        "up_epochs_completed": len(up_history),
        "down_epochs_completed": len(down_history),
        "multinomial_epochs_completed": len(multiclass_history),
        "up_platt_slope": up_calibrator.slope,
        "up_platt_intercept": up_calibrator.intercept,
        "down_platt_slope": down_calibrator.slope,
        "down_platt_intercept": down_calibrator.intercept,
    }
    return {
        "fold_summary": fold_summary,
        "probability_metrics": probability_metrics,
        "threshold_metrics": threshold_metrics,
        "calibration": calibration_rows,
        "yearly_ranking": yearly_ranking,
        "daily_spreads": daily_spreads,
        "coefficients": coefficients,
        "group_contributions": group_contributions,
        "training_history": training_history,
        "boundary_purge": purge,
        "oos_scores": oos_scores,
    }


def train_binary_logistic_model(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    calibration_mask: np.ndarray,
    preprocessor: Preprocessor,
    settings: Phase4ETargetSettings,
    seed: int,
    target_name: str,
) -> tuple[BinaryLogisticModel, list[dict[str, Any]], float]:
    train_positions = np.flatnonzero(train_mask)
    calibration_positions = np.flatnonzero(calibration_mask)
    prevalence = float(labels[train_positions].mean())
    if prevalence <= 0 or prevalence >= 1:
        raise RuntimeError(f"Phase 4E {target_name} 訓練資料只有單一類別")
    weights = np.zeros(features.shape[1], dtype=np.float64)
    intercept = float(math.log(prevalence / (1.0 - prevalence)))
    first_w = np.zeros_like(weights)
    second_w = np.zeros_like(weights)
    first_b = 0.0
    second_b = 0.0
    beta1 = 0.9
    beta2 = 0.999
    adam_epsilon = 1e-8
    update_count = 0
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    best_loss = math.inf
    best_weights = weights.copy()
    best_intercept = intercept
    stale_epochs = 0

    for epoch in range(1, settings.maximum_epochs + 1):
        positions = rng.permutation(train_positions)
        total_loss = 0.0
        total_examples = 0
        batches = 0
        for start in range(0, len(positions), settings.batch_size):
            selected = positions[start : start + settings.batch_size]
            x = preprocessor.transform(features[selected])
            y = labels[selected].astype(np.float64, copy=False)
            logits = x @ weights + intercept
            probabilities = sigmoid(logits)
            loss = binary_log_loss(y, probabilities)
            total_loss += loss * len(y)
            total_examples += len(y)
            batches += 1
            residual = (probabilities - y) / len(y)
            gradient_w = x.T @ residual + settings.l2_penalty * weights
            gradient_b = float(residual.sum())
            update_count += 1
            first_w = beta1 * first_w + (1 - beta1) * gradient_w
            second_w = beta2 * second_w + (1 - beta2) * np.square(gradient_w)
            first_b = beta1 * first_b + (1 - beta1) * gradient_b
            second_b = beta2 * second_b + (1 - beta2) * (gradient_b**2)
            correction1 = 1 - beta1**update_count
            correction2 = 1 - beta2**update_count
            weights -= settings.learning_rate * (first_w / correction1) / (
                np.sqrt(second_w / correction2) + adam_epsilon
            )
            intercept -= settings.learning_rate * (first_b / correction1) / (
                math.sqrt(second_b / correction2) + adam_epsilon
            )

        model = BinaryLogisticModel(weights=weights, intercept=intercept)
        calibration_probabilities = model.probabilities(
            preprocessor.transform(features[calibration_positions])
        )
        calibration_loss = binary_log_loss(
            labels[calibration_positions], calibration_probabilities
        )
        history.append(
            {
                "epoch": epoch,
                "training_log_loss": float(total_loss / max(total_examples, 1)),
                "calibration_log_loss": float(calibration_loss),
                "batches": batches,
                "examples": total_examples,
            }
        )
        if calibration_loss < best_loss - 1e-6:
            best_loss = calibration_loss
            best_weights = weights.copy()
            best_intercept = intercept
            stale_epochs = 0
        else:
            stale_epochs += 1
        if (
            epoch >= settings.minimum_epochs
            and stale_epochs >= settings.early_stopping_patience
        ):
            break
    return BinaryLogisticModel(best_weights, best_intercept), history, prevalence


def fit_linear_rank_model(
    *,
    features: np.ndarray,
    targets: np.ndarray,
    train_mask: np.ndarray,
    preprocessor: Preprocessor,
    settings: Phase4ETargetSettings,
) -> LinearRankModel:
    positions = np.flatnonzero(train_mask)
    feature_count = features.shape[1]
    gram = np.zeros((feature_count + 1, feature_count + 1), dtype=np.float64)
    rhs = np.zeros(feature_count + 1, dtype=np.float64)
    for start in range(0, len(positions), settings.training_chunk_size):
        selected = positions[start : start + settings.training_chunk_size]
        x = preprocessor.transform(features[selected])
        y = targets[selected]
        gram[0, 0] += len(selected)
        sums = x.sum(axis=0)
        gram[0, 1:] += sums
        gram[1:, 0] += sums
        gram[1:, 1:] += x.T @ x
        rhs[0] += y.sum()
        rhs[1:] += x.T @ y
    penalty = np.eye(feature_count + 1, dtype=np.float64) * settings.ranking_l2_penalty
    penalty[0, 0] = 0.0
    parameters = np.linalg.solve(gram + penalty, rhs)
    return LinearRankModel(weights=parameters[1:], intercept=float(parameters[0]))


def fit_binary_calibrator(logits: np.ndarray, labels: np.ndarray) -> BinaryCalibrator:
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    slope = 1.0
    intercept = 0.0
    regularization = 1e-6
    previous = math.inf
    for _ in range(100):
        eta = np.clip(slope * z + intercept, -40.0, 40.0)
        probabilities = sigmoid(eta)
        loss = binary_log_loss(y, probabilities) + 0.5 * regularization * (slope - 1.0) ** 2
        residual = probabilities - y
        variance = np.maximum(probabilities * (1.0 - probabilities), 1e-9)
        gradient = np.array(
            [
                float(np.sum(residual * z) + regularization * (slope - 1.0)),
                float(np.sum(residual)),
            ]
        )
        hessian = np.array(
            [
                [float(np.sum(variance * z * z) + regularization), float(np.sum(variance * z))],
                [float(np.sum(variance * z)), float(np.sum(variance))],
            ]
        )
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        accepted = False
        for scale in (1.0, 0.5, 0.25, 0.1, 0.05, 0.01):
            candidate_slope = max(0.01, slope - scale * float(step[0]))
            candidate_intercept = intercept - scale * float(step[1])
            candidate_probabilities = sigmoid(
                candidate_slope * z + candidate_intercept
            )
            candidate_loss = binary_log_loss(y, candidate_probabilities) + (
                0.5 * regularization * (candidate_slope - 1.0) ** 2
            )
            if candidate_loss <= loss + 1e-12:
                slope = candidate_slope
                intercept = candidate_intercept
                previous = candidate_loss
                accepted = True
                break
        if not accepted or np.linalg.norm(step) < 1e-7 or abs(previous - loss) < 1e-10:
            break
    return BinaryCalibrator(slope=float(slope), intercept=float(intercept))


def sigmoid(values: np.ndarray | float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    clipped = np.clip(array, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def binary_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.float64)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def binary_probability_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.uint8)
    p = np.asarray(probabilities, dtype=np.float64)
    return {
        "rows": int(len(y)),
        "base_rate": float(y.mean()),
        "mean_probability": float(p.mean()),
        "log_loss": binary_log_loss(y, p),
        "brier_score": float(np.mean(np.square(p - y))),
        "ece": binary_ece(y, p),
        "roc_auc": binary_roc_auc(y, p),
        "average_precision": binary_average_precision(y, p),
    }


def binary_ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    rows = binary_calibration_rows(
        labels,
        probabilities,
        fold_id="",
        test_year=0,
        target_name="",
        probability_variant="",
        bins=bins,
    )
    total = max(sum(int(row["sample_count"]) for row in rows), 1)
    return float(
        sum(
            abs(float(row["mean_probability"]) - float(row["actual_rate"]))
            * int(row["sample_count"])
            / total
            for row in rows
        )
    )


def binary_calibration_rows(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    fold_id: str,
    test_year: int,
    target_name: str,
    probability_variant: str,
    bins: int = 10,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(
        {
            "label": np.asarray(labels, dtype=np.uint8),
            "probability": np.asarray(probabilities, dtype=np.float64),
        }
    )
    frame["bin"] = np.minimum(
        bins,
        np.floor(np.clip(frame["probability"], 0.0, 1.0 - 1e-15) * bins).astype(int) + 1,
    )
    rows = []
    for bin_number, group in frame.groupby("bin", sort=True):
        rows.append(
            {
                "fold_id": fold_id,
                "test_year": test_year,
                "target_name": target_name,
                "probability_variant": probability_variant,
                "probability_bin": int(bin_number),
                "sample_count": int(len(group)),
                "minimum_probability": float(group["probability"].min()),
                "maximum_probability": float(group["probability"].max()),
                "mean_probability": float(group["probability"].mean()),
                "actual_rate": float(group["label"].mean()),
            }
        )
    return rows


def binary_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.uint8)
    s = pd.Series(np.asarray(scores, dtype=np.float64))
    positive = int(y.sum())
    negative = int(len(y) - positive)
    if positive == 0 or negative == 0:
        return math.nan
    ranks = s.rank(method="average").to_numpy(dtype=np.float64)
    rank_sum = float(ranks[y == 1].sum())
    return float((rank_sum - positive * (positive + 1) / 2) / (positive * negative))


def binary_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.uint8)
    positive = int(y.sum())
    if positive == 0:
        return math.nan
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    sorted_y = y[order]
    cumulative = np.cumsum(sorted_y)
    positions = np.arange(1, len(sorted_y) + 1)
    return float(np.sum((cumulative / positions) * sorted_y) / positive)


def select_balanced_accuracy_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.uint8)
    p = np.asarray(probabilities, dtype=np.float64)
    candidates = np.unique(
        np.concatenate(
            [
                np.quantile(p, np.linspace(0.01, 0.99, 99)),
                np.array([0.5], dtype=np.float64),
            ]
        )
    )
    best = (float("-inf"), float("-inf"), 0.5)
    for threshold in candidates:
        metrics = binary_threshold_metrics(y, p, float(threshold))
        candidate = (
            float(metrics["balanced_accuracy"]),
            float(metrics["f1"]),
            float(threshold),
        )
        if candidate > best:
            best = candidate
    return float(best[2])


def binary_threshold_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.uint8)
    predicted = (np.asarray(probabilities, dtype=np.float64) >= threshold).astype(np.uint8)
    tp = int(((predicted == 1) & (y == 1)).sum())
    tn = int(((predicted == 0) & (y == 0)).sum())
    fp = int(((predicted == 1) & (y == 0)).sum())
    fn = int(((predicted == 0) & (y == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "rows": int(len(y)),
        "predicted_positive_rate": float(predicted.mean()),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(2 * precision * recall / max(precision + recall, EPSILON)),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def assign_score_ranks(
    frame: pd.DataFrame,
    *,
    score_column: str,
    minimum_daily_stocks: int,
) -> pd.DataFrame:
    result = frame.copy()
    counts = result.groupby("signal_date")["stock_id"].transform("size")
    result = result[counts >= minimum_daily_stocks].copy()
    result["daily_rank"] = result.groupby("signal_date")[score_column].rank(
        method="first", ascending=True
    )
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


def ranking_summary_row(
    frame: pd.DataFrame,
    *,
    score_variant: str,
    return_column: str,
    evaluation_horizon: int,
    test_year: int,
) -> dict[str, Any]:
    returns = pd.to_numeric(frame[return_column], errors="coerce")
    top = frame[frame["daily_percentile"] > 80]
    bottom = frame[frame["daily_percentile"] <= 20]
    top_daily = top.groupby("signal_date")[return_column].mean()
    bottom_daily = bottom.groupby("signal_date")[return_column].mean()
    aligned = pd.concat([top_daily.rename("top"), bottom_daily.rename("bottom")], axis=1).dropna()
    correlations = []
    hit_rates = []
    for _, group in frame.groupby("signal_date", sort=True):
        if len(group) < 3:
            continue
        score_rank = group[score_variant].rank(method="average").to_numpy(dtype=np.float64)
        return_rank = group[return_column].rank(method="average").to_numpy(dtype=np.float64)
        correlation = np.corrcoef(score_rank, return_rank)[0, 1]
        if np.isfinite(correlation):
            correlations.append(float(correlation))
        predicted_top = group["daily_percentile"] > 80
        actual_top = group[return_column].rank(method="average", pct=True) > 0.80
        if predicted_top.any():
            hit_rates.append(float(actual_top[predicted_top].mean()))
    deciles = frame.groupby("daily_decile")[return_column].mean().reindex(range(1, 11))
    decile_correlation = float(
        np.corrcoef(np.arange(1, 11), deciles.to_numpy(dtype=np.float64))[0, 1]
    )
    return {
        "score_variant": score_variant,
        "evaluation_horizon_days": evaluation_horizon,
        "test_year": test_year,
        "test_rows": int(len(frame)),
        "signal_dates": int(frame["signal_date"].nunique()),
        "average_return": float(returns.mean()),
        "top20_average_return": float(top[return_column].mean()),
        "bottom20_average_return": float(bottom[return_column].mean()),
        "top20_minus_bottom20": float(aligned["top"].sub(aligned["bottom"]).mean()),
        "positive_spread_days": int((aligned["top"] > aligned["bottom"]).sum()),
        "total_spread_days": int(len(aligned)),
        "positive_spread_day_rate": float((aligned["top"] > aligned["bottom"]).mean()),
        "average_daily_spearman": float(np.mean(correlations)) if correlations else math.nan,
        "median_daily_spearman": float(np.median(correlations)) if correlations else math.nan,
        "top20_actual_top20_hit_rate": float(np.mean(hit_rates)) if hit_rates else math.nan,
        "decile_return_correlation": decile_correlation,
        "top20_up_5pct_rate": float((top[return_column] >= 0.05).mean()),
        "bottom20_up_5pct_rate": float((bottom[return_column] >= 0.05).mean()),
        "top20_down_5pct_rate": float((top[return_column] <= -0.05).mean()),
        "bottom20_down_5pct_rate": float((bottom[return_column] <= -0.05).mean()),
    }


def ranking_daily_spread_rows(
    frame: pd.DataFrame,
    *,
    score_variant: str,
    return_column: str,
    evaluation_horizon: int,
    test_year: int,
) -> list[dict[str, Any]]:
    top = frame[frame["daily_percentile"] > 80]
    bottom = frame[frame["daily_percentile"] <= 20]
    top_daily = top.groupby("signal_date")[return_column].mean()
    bottom_daily = bottom.groupby("signal_date")[return_column].mean()
    aligned = pd.concat([top_daily.rename("top"), bottom_daily.rename("bottom")], axis=1).dropna()
    return [
        {
            "score_variant": score_variant,
            "evaluation_horizon_days": evaluation_horizon,
            "test_year": test_year,
            "signal_date": str(signal_date),
            "top20_average_return": float(row["top"]),
            "bottom20_average_return": float(row["bottom"]),
            "top20_minus_bottom20": float(row["top"] - row["bottom"]),
        }
        for signal_date, row in aligned.iterrows()
    ]


def target_bootstrap_report(
    daily_spreads: pd.DataFrame,
    *,
    settings: Phase4ETargetSettings,
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
        for (score_variant, horizon), group in period.groupby(
            ["score_variant", "evaluation_horizon_days"], sort=True
        ):
            daily = group.set_index("signal_date")["top20_minus_bottom20"].sort_index()
            monthly = daily.groupby(daily.index.astype(str).str[:7]).mean().sort_index()
            if len(monthly) < 2:
                continue
            seed = _stable_seed(settings.random_seed, f"{period_name}:{score_variant}:{horizon}")
            lower, upper, bootstrap_mean = moving_block_bootstrap_mean_ci(
                monthly.to_numpy(dtype=np.float64),
                iterations=settings.bootstrap_iterations,
                block_length=settings.bootstrap_block_months,
                random_seed=seed,
            )
            rows.append(
                {
                    "period": period_name,
                    "score_variant": score_variant,
                    "evaluation_horizon_days": int(horizon),
                    "daily_observations": int(len(daily)),
                    "monthly_blocks": int(len(monthly)),
                    "point_estimate_daily_spread": float(daily.mean()),
                    "bootstrap_mean_spread": bootstrap_mean,
                    "confidence_level": 0.95,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "ci_excludes_zero_positive": int(lower > 0),
                    "iterations": settings.bootstrap_iterations,
                    "block_months": settings.bootstrap_block_months,
                }
            )
    return pd.DataFrame(rows)


def binary_coefficient_rows(
    model: BinaryLogisticModel,
    *,
    fold: HorizonFold,
    model_target: str,
    feature_columns: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows = [
        {
            "fold_id": fold.fold_id,
            "test_year": fold.test_year,
            "model_target": model_target,
            "feature_group": "intercept",
            "feature_name": "__INTERCEPT__",
            "coefficient": float(model.intercept),
        }
    ]
    rows.extend(
        {
            "fold_id": fold.fold_id,
            "test_year": fold.test_year,
            "model_target": model_target,
            "feature_group": feature_group(feature),
            "feature_name": feature,
            "coefficient": float(model.weights[index]),
        }
        for index, feature in enumerate(feature_columns)
    )
    return rows


def rank_coefficient_rows(
    model: LinearRankModel,
    *,
    fold: HorizonFold,
    feature_columns: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows = [
        {
            "fold_id": fold.fold_id,
            "test_year": fold.test_year,
            "model_target": "return_rank_20d",
            "feature_group": "intercept",
            "feature_name": "__INTERCEPT__",
            "coefficient": float(model.intercept),
        }
    ]
    rows.extend(
        {
            "fold_id": fold.fold_id,
            "test_year": fold.test_year,
            "model_target": "return_rank_20d",
            "feature_group": feature_group(feature),
            "feature_name": feature,
            "coefficient": float(model.weights[index]),
        }
        for index, feature in enumerate(feature_columns)
    )
    return rows


def group_contribution_rows(
    *,
    x_test: np.ndarray,
    test_frame: pd.DataFrame,
    models: dict[str, np.ndarray],
    fold: HorizonFold,
    settings: Phase4ETargetSettings,
    feature_columns: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_target, weights in models.items():
        contributions = x_test * weights
        score = contributions.sum(axis=1)
        working = test_frame[["stock_id", "signal_date"]].copy()
        working["model_score_without_intercept"] = score
        for group_name in FEATURE_GROUPS:
            indices = [
                index
                for index, feature in enumerate(feature_columns)
                if feature_group(feature) == group_name
            ]
            working[f"contribution_{group_name}"] = contributions[:, indices].sum(axis=1)
        working = assign_score_ranks(
            working.rename(columns={"model_score_without_intercept": "score"}),
            score_column="score",
            minimum_daily_stocks=settings.minimum_daily_stocks,
        )
        for decile, group in working.groupby("daily_decile", sort=True):
            row: dict[str, Any] = {
                "fold_id": fold.fold_id,
                "test_year": fold.test_year,
                "model_target": model_target,
                "score_decile": int(decile),
                "sample_count": int(len(group)),
            }
            for group_name in FEATURE_GROUPS:
                row[f"average_{group_name}_contribution"] = float(
                    group[f"contribution_{group_name}"].mean()
                )
            rows.append(row)
    return rows


def feature_group(feature_name: str) -> str:
    if feature_name.startswith("foreign_"):
        return "foreign"
    if feature_name.startswith("investment_trust_"):
        return "investment_trust"
    if feature_name.startswith("dealer_self_"):
        return "dealer_self"
    return "institutional_consensus"


def target_coefficient_stability(coefficients: pd.DataFrame) -> pd.DataFrame:
    if coefficients.empty:
        return pd.DataFrame()
    feature_rows = coefficients[coefficients["feature_name"] != "__INTERCEPT__"]
    records: list[dict[str, Any]] = []
    for model_target, target_frame in feature_rows.groupby("model_target", sort=True):
        matrix = target_frame.pivot(
            index="test_year", columns="feature_name", values="coefficient"
        ).sort_index()
        values = matrix.to_numpy(dtype=np.float64)
        years = list(matrix.index.astype(int))
        pairwise: list[float] = []
        adjacent: list[float] = []
        for left in range(len(years)):
            for right in range(left + 1, len(years)):
                correlation = _safe_correlation(values[left], values[right])
                pairwise.append(correlation)
                if right == left + 1:
                    adjacent.append(correlation)
        fully_consistent = 0
        high_consistency = 0
        required = max(1, math.ceil(len(years) * 0.875))
        for feature_name in matrix.columns:
            series = matrix[feature_name].dropna().to_numpy(dtype=np.float64)
            positive = int((series > 0).sum())
            negative = int((series < 0).sum())
            dominant = max(positive, negative)
            if dominant == len(series):
                fully_consistent += 1
            if dominant >= required:
                high_consistency += 1
            records.append(
                {
                    "row_type": "feature",
                    "model_target": model_target,
                    "feature_name": feature_name,
                    "test_years": len(series),
                    "mean_coefficient": float(series.mean()),
                    "std_coefficient": float(series.std(ddof=0)),
                    "positive_years": positive,
                    "negative_years": negative,
                    "dominant_sign_rate": dominant / max(len(series), 1),
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
                "model_target": model_target,
                "feature_name": "__SUMMARY__",
                "test_years": len(years),
                "mean_coefficient": "",
                "std_coefficient": "",
                "positive_years": "",
                "negative_years": "",
                "dominant_sign_rate": "",
                "average_pairwise_correlation": float(np.mean(pairwise)) if pairwise else 0.0,
                "average_adjacent_correlation": float(np.mean(adjacent)) if adjacent else 0.0,
                "minimum_pairwise_correlation": float(np.min(pairwise)) if pairwise else 0.0,
                "fully_consistent_features": fully_consistent,
                "high_consistency_features": high_consistency,
            }
        )
    return pd.DataFrame(records)


def build_target_comparison(
    *,
    probability_metrics: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    yearly_ranking: pd.DataFrame,
    daily_spreads: pd.DataFrame,
    oos_scores: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if probability_metrics.empty:
        return pd.DataFrame()
    latest_year = int(probability_metrics["test_year"].max())
    periods = {
        "development": lambda values: values <= 2022,
        "confirmation": lambda values: values >= 2023,
        "confirmation_ex_latest": lambda values: (values >= 2023) & (values < latest_year),
        "all": lambda values: np.ones(len(values), dtype=bool),
    }
    for period_name, selector in periods.items():
        period_metrics = probability_metrics[selector(probability_metrics["test_year"].to_numpy())]
        for (target_name, variant), group in period_metrics.groupby(
            ["target_name", "probability_variant"], sort=True
        ):
            weights = group["rows"].to_numpy(dtype=np.float64)
            row = {
                "comparison_type": "probability",
                "period": period_name,
                "target_or_score": target_name,
                "variant_or_horizon": variant,
                "rows": int(group["rows"].sum()),
            }
            for metric in (
                "base_rate",
                "mean_probability",
                "log_loss",
                "brier_score",
                "ece",
                "roc_auc",
                "average_precision",
            ):
                row[metric] = float(np.average(group[metric], weights=weights))
            rows.append(row)
        period_threshold = threshold_metrics[
            selector(threshold_metrics["test_year"].to_numpy())
        ]
        for target_name, group in period_threshold.groupby("target_name", sort=True):
            weights = group["rows"].to_numpy(dtype=np.float64)
            row = {
                "comparison_type": "threshold",
                "period": period_name,
                "target_or_score": target_name,
                "variant_or_horizon": "calibration_selected",
                "rows": int(group["rows"].sum()),
            }
            for metric in (
                "selected_threshold",
                "predicted_positive_rate",
                "precision",
                "recall",
                "specificity",
                "f1",
                "balanced_accuracy",
            ):
                row[metric] = float(np.average(group[metric], weights=weights))
            rows.append(row)
        period_yearly = yearly_ranking[selector(yearly_ranking["test_year"].to_numpy())]
        for (score_variant, horizon), group in period_yearly.groupby(
            ["score_variant", "evaluation_horizon_days"], sort=True
        ):
            weights = group["test_rows"].to_numpy(dtype=np.float64)
            row = {
                "comparison_type": "ranking",
                "period": period_name,
                "target_or_score": score_variant,
                "variant_or_horizon": int(horizon),
                "rows": int(group["test_rows"].sum()),
                "top20_minus_bottom20": float(
                    np.average(group["top20_minus_bottom20"], weights=weights)
                ),
                "positive_years": int((group["top20_minus_bottom20"] > 0).sum()),
                "total_years": int(len(group)),
                "average_daily_spearman": float(
                    np.average(group["average_daily_spearman"], weights=weights)
                ),
                "top20_actual_top20_hit_rate": float(
                    np.average(group["top20_actual_top20_hit_rate"], weights=weights)
                ),
                "decile_return_correlation": float(
                    np.average(group["decile_return_correlation"], weights=weights)
                ),
                "up_5pct_rate_lift": float(
                    np.average(
                        group["top20_up_5pct_rate"] - group["bottom20_up_5pct_rate"],
                        weights=weights,
                    )
                ),
                "down_5pct_rate_reduction": float(
                    np.average(
                        group["bottom20_down_5pct_rate"] - group["top20_down_5pct_rate"],
                        weights=weights,
                    )
                ),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def build_target_summary(
    *,
    frame: pd.DataFrame,
    fold_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    bootstrap: pd.DataFrame,
    settings: Phase4ETargetSettings,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {"metric": "phase4e_version", "value": PHASE4E_VERSION},
        {"metric": "market", "value": settings.target_market},
        {
            "metric": "candidate",
            "value": market_model_spec(settings.target_market).candidate_id,
        },
        {"metric": "primary_horizon_days", "value": PRIMARY_HORIZON_DAYS},
        {"metric": "extension_horizon_days", "value": EXTENSION_HORIZON_DAYS},
        {"metric": "label_threshold", "value": settings.label_threshold},
        {
            "metric": "model_features",
            "value": len(market_model_spec(settings.target_market).feature_columns),
        },
        {"metric": "research_rows", "value": len(frame)},
        {"metric": "signal_dates", "value": frame["signal_date"].nunique()},
        {"metric": "completed_folds", "value": int(fold_summary["status"].eq("complete").sum())},
        {"metric": "failed_folds", "value": int(fold_summary["status"].ne("complete").sum())},
        {"metric": "ready_for_target_decision", "value": int(fold_summary["status"].eq("complete").all())},
        {
            "metric": "decision_note",
            "value": "報告完成不代表任一模型已勝出，須自行核對確認期與排除最新年度結果。",
        },
    ]
    if not comparison.empty:
        selected_metrics = comparison[
            (comparison["period"].isin(["confirmation", "confirmation_ex_latest"]))
            & (
                (
                    (comparison["comparison_type"] == "probability")
                    & (comparison["variant_or_horizon"].isin(["platt", "historical_probability"]))
                )
                | (
                    (comparison["comparison_type"] == "ranking")
                    & (comparison["variant_or_horizon"].astype(str).isin(["20", "40"]))
                )
            )
        ]
        for _, row in selected_metrics.iterrows():
            prefix = f"{row['period']}_{row['comparison_type']}_{row['target_or_score']}_{row['variant_or_horizon']}"
            for column in (
                "log_loss",
                "brier_score",
                "ece",
                "roc_auc",
                "average_precision",
                "top20_minus_bottom20",
                "average_daily_spearman",
                "top20_actual_top20_hit_rate",
                "up_5pct_rate_lift",
                "down_5pct_rate_reduction",
            ):
                if column in row and pd.notna(row[column]):
                    rows.append({"metric": f"{prefix}_{column}", "value": row[column]})
    if not bootstrap.empty:
        selected = bootstrap[
            bootstrap["period"].isin(["confirmation", "confirmation_ex_latest"])
        ]
        for _, row in selected.iterrows():
            prefix = (
                f"{row['period']}_bootstrap_{row['score_variant']}_"
                f"{int(row['evaluation_horizon_days'])}d"
            )
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

    rank_row = comparison[
        (comparison["comparison_type"] == "ranking")
        & (comparison["period"] == "confirmation_ex_latest")
        & (comparison["target_or_score"] == "return_rank_score")
        & (
            pd.to_numeric(comparison["variant_or_horizon"], errors="coerce")
            == PRIMARY_HORIZON_DAYS
        )
    ]
    rank_bootstrap = bootstrap[
        (bootstrap["period"] == "confirmation_ex_latest")
        & (bootstrap["score_variant"] == "return_rank_score")
        & (bootstrap["evaluation_horizon_days"] == PRIMARY_HORIZON_DAYS)
    ]
    rank_pass = 0
    if len(rank_row) == 1 and len(rank_bootstrap) == 1:
        rank_metric = rank_row.iloc[0]
        rank_ci = rank_bootstrap.iloc[0]
        total_years = int(rank_metric.get("total_years", 0) or 0)
        positive_years = int(rank_metric.get("positive_years", 0) or 0)
        required_positive_years = max(2, math.ceil(total_years * 0.60))
        rank_pass = int(
            total_years >= 3
            and positive_years >= required_positive_years
            and float(rank_metric.get("top20_minus_bottom20", 0.0) or 0.0) > 0
            and float(rank_metric.get("average_daily_spearman", 0.0) or 0.0) > 0
            and float(rank_ci.get("ci_lower", 0.0) or 0.0) > 0
        )
        rows.extend(
            [
                {"metric": "return_rank_confirmation_years", "value": total_years},
                {"metric": "return_rank_positive_years", "value": positive_years},
                {"metric": "return_rank_required_positive_years", "value": required_positive_years},
                {"metric": "return_rank_confirmation_spread", "value": rank_metric.get("top20_minus_bottom20", "")},
                {"metric": "return_rank_confirmation_daily_spearman", "value": rank_metric.get("average_daily_spearman", "")},
                {"metric": "return_rank_confirmation_bootstrap_ci_lower", "value": rank_ci.get("ci_lower", "")},
            ]
        )
    rows.extend(
        [
            {"metric": "return_rank_validation_pass", "value": rank_pass},
            {
                "metric": "return_rank_validation_rule",
                "value": "確認期排除最新年度至少3年、正向年度>=60%、top20-bottom20>0、daily Spearman>0、95% bootstrap下緣>0",
            },
        ]
    )
    return pd.DataFrame(rows)


def export_phase4e_reports(
    *,
    output_dir: Path,
    reports: dict[str, pd.DataFrame],
) -> list[Path]:
    paths = [
        _write_csv(output_dir / "phase4e_fold_summary.csv", reports["fold_summary"]),
        _write_csv(
            output_dir / "phase4e_probability_metrics.csv",
            reports["probability_metrics"],
        ),
        _write_csv(
            output_dir / "phase4e_threshold_metrics.csv",
            reports["threshold_metrics"],
        ),
        _write_csv(output_dir / "phase4e_calibration.csv", reports["calibration"]),
        _write_csv(
            output_dir / "phase4e_yearly_ranking.csv", reports["yearly_ranking"]
        ),
        _write_gzip_csv(
            output_dir / "phase4e_daily_spreads.csv.gz", reports["daily_spreads"]
        ),
        _write_csv(
            output_dir / "phase4e_bootstrap_confidence.csv", reports["bootstrap"]
        ),
        _write_csv(output_dir / "phase4e_coefficients.csv", reports["coefficients"]),
        _write_csv(
            output_dir / "phase4e_coefficient_stability.csv",
            reports["coefficient_stability"],
        ),
        _write_csv(
            output_dir / "phase4e_group_contributions.csv",
            reports["group_contributions"],
        ),
        _write_csv(
            output_dir / "phase4e_training_history.csv",
            reports["training_history"],
        ),
        _write_csv(
            output_dir / "phase4e_boundary_purge.csv", reports["boundary_purge"]
        ),
        _write_gzip_csv(
            output_dir / "phase4e_oos_scores.csv.gz", reports["oos_scores"]
        ),
        _write_csv(
            output_dir / "phase4e_model_comparison.csv", reports["comparison"]
        ),
        _write_csv(output_dir / "phase4e_summary.csv", reports["summary"]),
    ]
    archive = output_dir / "phase4e_target_validation_reports.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in paths:
            handle.write(path, arcname=path.name)
    paths.append(archive)
    return paths


def _to_horizon_settings(settings: Phase4ETargetSettings) -> Phase4DHorizonSettings:
    return Phase4DHorizonSettings(
        target_market=settings.target_market,
        horizons=(10, 20, 40),
        label_threshold=settings.label_threshold,
        minimum_daily_stocks=settings.minimum_daily_stocks,
        first_test_year=settings.first_test_year,
        minimum_training_years=settings.minimum_training_years,
        calibration_years=settings.calibration_years,
        clip_lower_quantile=settings.clip_lower_quantile,
        clip_upper_quantile=settings.clip_upper_quantile,
        quantile_sample_size=settings.quantile_sample_size,
        training_chunk_size=settings.training_chunk_size,
        batch_size=settings.batch_size,
        maximum_epochs=settings.maximum_epochs,
        minimum_epochs=settings.minimum_epochs,
        early_stopping_patience=settings.early_stopping_patience,
        learning_rate=settings.learning_rate,
        l2_penalty=market_model_spec(settings.target_market).l2_penalty,
        bootstrap_iterations=settings.bootstrap_iterations,
        bootstrap_block_months=settings.bootstrap_block_months,
        random_seed=settings.random_seed,
    )


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or right.size < 2:
        return 0.0
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    value = np.corrcoef(left, right)[0, 1]
    return float(value) if np.isfinite(value) else 0.0


def _stable_seed(base_seed: int, value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int((base_seed + int.from_bytes(digest[:4], "big")) % (2**32 - 1))


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _write_gzip_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip", encoding="utf-8")
    return path
