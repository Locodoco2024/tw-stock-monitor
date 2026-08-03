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
from research.institutional_model.market_model_spec import market_model_spec
from research.institutional_model.phase4_horizon import (
    HorizonFold,
    Phase4DHorizonSettings,
    build_horizon_folds,
    build_horizon_research_frame,
    build_purged_masks,
    fit_mask_preprocessor,
)
from research.institutional_model.phase4_selection import (
    moving_block_bootstrap_mean_ci,
    resolve_phase3_shard_directory,
)
from research.institutional_model.phase4_target import (
    LinearRankModel,
    assign_score_ranks,
    ranking_daily_spread_rows,
    ranking_summary_row,
    target_coefficient_stability,
)


PHASE6B_VERSION = "phase6b-v1"
TARGET_MARKET = "twse"
TARGET_HORIZON_DAYS = 40
SECONDARY_HORIZON_DAYS = 20
SCORE_COLUMN = "return_rank_40d_score"


@dataclass(frozen=True)
class Phase6BTWSE40DSettings:
    minimum_daily_stocks: int = 50
    first_test_year: int = 2019
    minimum_training_years: int = 3
    calibration_years: int = 1
    clip_lower_quantile: float = 0.005
    clip_upper_quantile: float = 0.995
    quantile_sample_size: int = 250_000
    training_chunk_size: int = 100_000
    ranking_l2_penalty: float = 0.001
    bootstrap_iterations: int = 2_000
    bootstrap_block_months: int = 3
    random_seed: int = 20260731

    def validate(self) -> None:
        if self.minimum_daily_stocks < 10:
            raise ValueError("Phase 6B 每日最少股票數不可小於 10")
        if self.minimum_training_years < 2:
            raise ValueError("Phase 6B 最少訓練年度必須至少為 2")
        if self.calibration_years != 1:
            raise ValueError("Phase 6B 固定使用前一年度作為 purge 邊界")
        if not 0 <= self.clip_lower_quantile < self.clip_upper_quantile <= 1:
            raise ValueError("Phase 6B 截尾分位數設定無效")
        if self.quantile_sample_size <= 0 or self.training_chunk_size <= 0:
            raise ValueError("Phase 6B chunk／sample size 必須大於 0")
        if self.ranking_l2_penalty < 0:
            raise ValueError("Phase 6B ranking L2 不可小於 0")
        if self.bootstrap_iterations < 200:
            raise ValueError("Phase 6B bootstrap 次數不可小於 200")
        if self.bootstrap_block_months < 1:
            raise ValueError("Phase 6B bootstrap 區塊月份必須大於 0")


@dataclass(frozen=True)
class Phase6BTWSE40DResult:
    status: str
    validation_passed: bool
    completed_folds: int
    expected_folds: int
    failed_folds: int
    output_paths: tuple[Path, ...]


def run_phase6b_twse_40d_validation(
    *,
    database: ResearchDatabase,
    output_dir: Path | str,
    shard_root: Path | str,
    cache_root: Path | str,
    settings: Phase6BTWSE40DSettings | None = None,
    force: bool = False,
) -> Phase6BTWSE40DResult:
    config = settings or Phase6BTWSE40DSettings()
    config.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    shard_dir = resolve_phase3_shard_directory(
        output_dir=output,
        shard_root=Path(shard_root),
    )
    frame = build_horizon_research_frame(
        database=database,
        shard_dir=shard_dir,
        cache_dir=Path(cache_root),
        settings=_to_horizon_settings(config),
        force=force,
    )
    reports = evaluate_twse_40d_frame(frame=frame, settings=config)
    paths = export_phase6b_reports(output_dir=output, reports=reports)
    fold_summary = reports["fold_summary"]
    completed = int(fold_summary["status"].eq("complete").sum())
    failed = int(fold_summary["status"].ne("complete").sum())
    expected = len(fold_summary)
    status = "PASS" if expected > 0 and failed == 0 else "FAIL"
    metrics = _metric_map(reports["summary"])
    validation_passed = metrics.get("return_rank_40d_validation_pass") in {"1", "1.0"}
    return Phase6BTWSE40DResult(
        status=status,
        validation_passed=validation_passed,
        completed_folds=completed,
        expected_folds=expected,
        failed_folds=failed,
        output_paths=tuple(paths),
    )


def prepare_twse_40d_frame(
    frame: pd.DataFrame,
    *,
    settings: Phase6BTWSE40DSettings,
) -> pd.DataFrame:
    required = {
        "stock_id",
        "stock_name",
        "signal_date",
        "signal_year",
        "target_date_40d",
        "label_status_40d",
        "adjusted_return_20d",
        "adjusted_return_40d",
        *market_model_spec(TARGET_MARKET).feature_columns,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Phase 6B 缺少欄位：{missing}")
    selected = frame[
        (frame["label_status_40d"].astype(str) == "ok")
        & pd.to_numeric(frame["adjusted_return_40d"], errors="coerce").notna()
    ].copy()
    selected["adjusted_return_20d"] = pd.to_numeric(
        selected["adjusted_return_20d"], errors="coerce"
    )
    selected["adjusted_return_40d"] = pd.to_numeric(
        selected["adjusted_return_40d"], errors="coerce"
    )
    selected["target_date"] = selected["target_date_40d"].astype(str)
    counts = selected.groupby("signal_date")["stock_id"].transform("size")
    selected = selected[counts >= settings.minimum_daily_stocks].copy()
    if selected.empty:
        raise RuntimeError("Phase 6B 沒有符合每日最低股票數的 40 日成熟資料")
    selected["future_return_rank_pct_40d"] = selected.groupby("signal_date")[
        "adjusted_return_40d"
    ].rank(method="average", pct=True)
    return selected.reset_index(drop=True)


def evaluate_twse_40d_frame(
    *,
    frame: pd.DataFrame,
    settings: Phase6BTWSE40DSettings,
) -> dict[str, pd.DataFrame]:
    settings.validate()
    selected = prepare_twse_40d_frame(frame, settings=settings)
    horizon_settings = _to_horizon_settings(settings)
    folds = build_horizon_folds(
        selected,
        horizon=TARGET_HORIZON_DAYS,
        settings=horizon_settings,
    )
    if not folds:
        raise RuntimeError("Phase 6B 沒有足夠年度建立 40 日滾動驗證")

    fold_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    spread_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    purge_rows: list[dict[str, Any]] = []
    oos_frames: list[pd.DataFrame] = []

    for fold in folds:
        try:
            fold_result = evaluate_twse_40d_fold(
                frame=selected,
                fold=fold,
                settings=settings,
            )
            fold_rows.append(fold_result["fold_summary"])
            yearly_rows.extend(fold_result["yearly_ranking"])
            spread_rows.extend(fold_result["daily_spreads"])
            coefficient_rows.extend(fold_result["coefficients"])
            purge_rows.append(fold_result["boundary_purge"])
            oos_frames.append(fold_result["oos_scores"])
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
    yearly_ranking = pd.DataFrame(yearly_rows)
    daily_spreads = pd.DataFrame(spread_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    boundary_purge = pd.DataFrame(purge_rows)
    oos_scores = pd.concat(oos_frames, ignore_index=True) if oos_frames else pd.DataFrame()
    bootstrap = build_phase6b_bootstrap(daily_spreads, settings=settings)
    coefficient_stability = target_coefficient_stability(coefficients)
    comparison = build_phase6b_comparison(yearly_ranking)
    summary = build_phase6b_summary(
        frame=selected,
        fold_summary=fold_summary,
        comparison=comparison,
        bootstrap=bootstrap,
        settings=settings,
    )
    return {
        "fold_summary": fold_summary,
        "yearly_ranking": yearly_ranking,
        "daily_spreads": daily_spreads,
        "bootstrap": bootstrap,
        "coefficients": coefficients,
        "coefficient_stability": coefficient_stability,
        "boundary_purge": boundary_purge,
        "oos_scores": oos_scores,
        "comparison": comparison,
        "summary": summary,
    }


def evaluate_twse_40d_fold(
    *,
    frame: pd.DataFrame,
    fold: HorizonFold,
    settings: Phase6BTWSE40DSettings,
) -> dict[str, Any]:
    train_mask, calibration_mask, test_mask, purge = build_purged_masks(frame, fold)
    if int(train_mask.sum()) == 0 or int(test_mask.sum()) == 0:
        raise RuntimeError(f"{fold.fold_id} train/test 無可用資料")
    feature_columns = market_model_spec(TARGET_MARKET).feature_columns
    features = frame[list(feature_columns)].to_numpy(dtype=np.float64)
    targets = frame["future_return_rank_pct_40d"].to_numpy(dtype=np.float64)
    preprocessor = fit_mask_preprocessor(
        features=features,
        train_mask=train_mask,
        settings=_to_horizon_settings(settings),
        seed=settings.random_seed + fold.test_year,
    )
    model = fit_rank_model(
        features=features,
        targets=targets,
        train_mask=train_mask,
        preprocessor=preprocessor,
        settings=settings,
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
            "future_return_rank_pct_40d",
        ],
    ].copy()
    test_frame["test_year"] = fold.test_year
    test_frame[SCORE_COLUMN] = model.scores(
        preprocessor.transform(features[np.flatnonzero(test_mask)])
    )
    ranked = assign_score_ranks(
        test_frame,
        score_column=SCORE_COLUMN,
        minimum_daily_stocks=settings.minimum_daily_stocks,
    )
    if ranked.empty:
        raise RuntimeError(f"{fold.fold_id} 排名後沒有資料")

    yearly_ranking: list[dict[str, Any]] = []
    daily_spreads: list[dict[str, Any]] = []
    for horizon in (SECONDARY_HORIZON_DAYS, TARGET_HORIZON_DAYS):
        column = f"adjusted_return_{horizon}d"
        valid = ranked[np.isfinite(pd.to_numeric(ranked[column], errors="coerce"))].copy()
        if valid.empty:
            continue
        yearly_ranking.append(
            ranking_summary_row(
                valid,
                score_variant=SCORE_COLUMN,
                return_column=column,
                evaluation_horizon=horizon,
                test_year=fold.test_year,
            )
        )
        daily_spreads.extend(
            ranking_daily_spread_rows(
                valid,
                score_variant=SCORE_COLUMN,
                return_column=column,
                evaluation_horizon=horizon,
                test_year=fold.test_year,
            )
        )

    coefficients = [
        {
            "fold_id": fold.fold_id,
            "test_year": fold.test_year,
            "model_target": "return_rank_40d",
            "feature_group": "intercept",
            "feature_name": "__INTERCEPT__",
            "coefficient": float(model.intercept),
        }
    ]
    coefficients.extend(
        {
            "fold_id": fold.fold_id,
            "test_year": fold.test_year,
            "model_target": "return_rank_40d",
            "feature_group": feature_group_name(feature),
            "feature_name": feature,
            "coefficient": float(model.weights[index]),
        }
        for index, feature in enumerate(feature_columns)
    )

    oos_scores = ranked[
        [
            "stock_id",
            "stock_name",
            "signal_date",
            "signal_year",
            "test_year",
            "adjusted_return_20d",
            "adjusted_return_40d",
            "future_return_rank_pct_40d",
            SCORE_COLUMN,
            "daily_percentile",
            "daily_decile",
        ]
    ].copy()
    oos_scores = oos_scores.rename(
        columns={
            "daily_percentile": "return_rank_40d_score_daily_percentile",
            "daily_decile": "return_rank_40d_score_daily_decile",
        }
    )
    fold_summary = {
        **asdict(fold),
        "fold_id": fold.fold_id,
        "status": "complete",
        "error": "",
        "train_rows": int(train_mask.sum()),
        "calibration_rows_after_purge": int(calibration_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "test_signal_dates": int(ranked["signal_date"].nunique()),
        "target_mean_train": float(targets[train_mask].mean()),
        "target_std_train": float(targets[train_mask].std()),
    }
    return {
        "fold_summary": fold_summary,
        "yearly_ranking": yearly_ranking,
        "daily_spreads": daily_spreads,
        "coefficients": coefficients,
        "boundary_purge": purge,
        "oos_scores": oos_scores,
    }


def fit_rank_model(
    *,
    features: np.ndarray,
    targets: np.ndarray,
    train_mask: np.ndarray,
    preprocessor: Any,
    settings: Phase6BTWSE40DSettings,
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
    try:
        parameters = np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError:
        parameters = np.linalg.lstsq(gram + penalty, rhs, rcond=None)[0]
    return LinearRankModel(weights=parameters[1:], intercept=float(parameters[0]))


def build_phase6b_comparison(yearly_ranking: pd.DataFrame) -> pd.DataFrame:
    if yearly_ranking.empty:
        return pd.DataFrame()
    latest_year = int(yearly_ranking["test_year"].max())
    periods = {
        "development": lambda values: values <= 2022,
        "confirmation": lambda values: values >= 2023,
        "confirmation_ex_latest": lambda values: (values >= 2023) & (values < latest_year),
        "all": lambda values: np.ones(len(values), dtype=bool),
    }
    rows: list[dict[str, Any]] = []
    for period_name, selector in periods.items():
        period = yearly_ranking[selector(yearly_ranking["test_year"].to_numpy())]
        for (score_variant, horizon), group in period.groupby(
            ["score_variant", "evaluation_horizon_days"], sort=True
        ):
            weights = group["test_rows"].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "period": period_name,
                    "score_variant": score_variant,
                    "evaluation_horizon_days": int(horizon),
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
                }
            )
    return pd.DataFrame(rows)


def build_phase6b_bootstrap(
    daily_spreads: pd.DataFrame,
    *,
    settings: Phase6BTWSE40DSettings,
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
            lower, upper, mean = moving_block_bootstrap_mean_ci(
                monthly.to_numpy(dtype=np.float64),
                iterations=settings.bootstrap_iterations,
                block_length=settings.bootstrap_block_months,
                random_seed=_stable_seed(
                    settings.random_seed,
                    f"{period_name}:{score_variant}:{horizon}",
                ),
            )
            rows.append(
                {
                    "period": period_name,
                    "score_variant": score_variant,
                    "evaluation_horizon_days": int(horizon),
                    "daily_observations": int(len(daily)),
                    "monthly_blocks": int(len(monthly)),
                    "point_estimate_daily_spread": float(daily.mean()),
                    "bootstrap_mean_spread": mean,
                    "confidence_level": 0.95,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "ci_excludes_zero_positive": int(lower > 0),
                    "iterations": settings.bootstrap_iterations,
                    "block_months": settings.bootstrap_block_months,
                }
            )
    return pd.DataFrame(rows)


def build_phase6b_summary(
    *,
    frame: pd.DataFrame,
    fold_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    bootstrap: pd.DataFrame,
    settings: Phase6BTWSE40DSettings,
) -> pd.DataFrame:
    spec = market_model_spec(TARGET_MARKET)
    rows: list[dict[str, Any]] = [
        {"metric": "phase6b_version", "value": PHASE6B_VERSION},
        {"metric": "market", "value": TARGET_MARKET},
        {"metric": "candidate", "value": spec.candidate_id},
        {"metric": "model_features", "value": len(spec.feature_columns)},
        {"metric": "model_target", "value": "same_day_future_return_rank_40d"},
        {"metric": "primary_horizon_days", "value": TARGET_HORIZON_DAYS},
        {"metric": "research_rows", "value": len(frame)},
        {"metric": "signal_dates", "value": frame["signal_date"].nunique()},
        {
            "metric": "completed_folds",
            "value": int(fold_summary["status"].eq("complete").sum()),
        },
        {
            "metric": "failed_folds",
            "value": int(fold_summary["status"].ne("complete").sum()),
        },
    ]
    primary = comparison[
        (comparison["period"] == "confirmation_ex_latest")
        & (comparison["score_variant"] == SCORE_COLUMN)
        & (comparison["evaluation_horizon_days"] == TARGET_HORIZON_DAYS)
    ] if not comparison.empty else pd.DataFrame()
    primary_bootstrap = bootstrap[
        (bootstrap["period"] == "confirmation_ex_latest")
        & (bootstrap["score_variant"] == SCORE_COLUMN)
        & (bootstrap["evaluation_horizon_days"] == TARGET_HORIZON_DAYS)
    ] if not bootstrap.empty else pd.DataFrame()
    passed = 0
    if len(primary) == 1 and len(primary_bootstrap) == 1:
        metric = primary.iloc[0]
        confidence = primary_bootstrap.iloc[0]
        years = int(metric["total_years"])
        positive_years = int(metric["positive_years"])
        required_positive_years = max(2, math.ceil(years * 0.60))
        passed = int(
            years >= 3
            and positive_years >= required_positive_years
            and float(metric["top20_minus_bottom20"]) > 0
            and float(metric["average_daily_spearman"]) > 0
            and float(confidence["ci_lower"]) > 0
        )
        rows.extend(
            [
                {"metric": "return_rank_40d_confirmation_years", "value": years},
                {
                    "metric": "return_rank_40d_positive_years",
                    "value": positive_years,
                },
                {
                    "metric": "return_rank_40d_required_positive_years",
                    "value": required_positive_years,
                },
                {
                    "metric": "return_rank_40d_confirmation_spread",
                    "value": metric["top20_minus_bottom20"],
                },
                {
                    "metric": "return_rank_40d_confirmation_daily_spearman",
                    "value": metric["average_daily_spearman"],
                },
                {
                    "metric": "return_rank_40d_confirmation_bootstrap_ci_lower",
                    "value": confidence["ci_lower"],
                },
                {
                    "metric": "return_rank_40d_confirmation_bootstrap_ci_upper",
                    "value": confidence["ci_upper"],
                },
            ]
        )
    rows.extend(
        [
            {"metric": "return_rank_40d_validation_pass", "value": passed},
            {
                "metric": "return_rank_40d_validation_rule",
                "value": "確認期排除最新年度至少3年、正向年度>=60%、40日top20-bottom20>0、daily Spearman>0、95% bootstrap下緣>0",
            },
            {
                "metric": "deployment_ready",
                "value": 0,
            },
            {
                "metric": "deployment_note",
                "value": "Phase 6B 僅驗證40日目標；通過後仍須另做TWSE生命週期與最終模型訓練。",
            },
        ]
    )
    return pd.DataFrame(rows)


def export_phase6b_reports(
    *,
    output_dir: Path,
    reports: dict[str, pd.DataFrame],
) -> list[Path]:
    paths = [
        _write_csv(output_dir / "phase6b_fold_summary.csv", reports["fold_summary"]),
        _write_csv(
            output_dir / "phase6b_yearly_ranking.csv", reports["yearly_ranking"]
        ),
        _write_gzip_csv(
            output_dir / "phase6b_daily_spreads.csv.gz", reports["daily_spreads"]
        ),
        _write_csv(
            output_dir / "phase6b_bootstrap_confidence.csv", reports["bootstrap"]
        ),
        _write_csv(output_dir / "phase6b_coefficients.csv", reports["coefficients"]),
        _write_csv(
            output_dir / "phase6b_coefficient_stability.csv",
            reports["coefficient_stability"],
        ),
        _write_csv(
            output_dir / "phase6b_boundary_purge.csv", reports["boundary_purge"]
        ),
        _write_gzip_csv(
            output_dir / "phase6b_oos_scores.csv.gz", reports["oos_scores"]
        ),
        _write_csv(
            output_dir / "phase6b_model_comparison.csv", reports["comparison"]
        ),
        _write_csv(output_dir / "phase6b_summary.csv", reports["summary"]),
    ]
    archive = output_dir / "phase6b_twse_40d_validation_reports.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in paths:
            handle.write(path, arcname=path.name)
    paths.append(archive)
    return paths


def feature_group_name(feature: str) -> str:
    if feature.startswith("foreign_"):
        return "foreign"
    if feature.startswith("investment_trust_"):
        return "investment_trust"
    if feature.startswith("dealer_self_"):
        return "dealer_self"
    return "institutional_consensus"


def _to_horizon_settings(
    settings: Phase6BTWSE40DSettings,
) -> Phase4DHorizonSettings:
    return Phase4DHorizonSettings(
        target_market=TARGET_MARKET,
        horizons=(10, 20, 40),
        minimum_daily_stocks=settings.minimum_daily_stocks,
        first_test_year=settings.first_test_year,
        minimum_training_years=settings.minimum_training_years,
        calibration_years=settings.calibration_years,
        clip_lower_quantile=settings.clip_lower_quantile,
        clip_upper_quantile=settings.clip_upper_quantile,
        quantile_sample_size=settings.quantile_sample_size,
        training_chunk_size=settings.training_chunk_size,
        l2_penalty=market_model_spec(TARGET_MARKET).l2_penalty,
        bootstrap_iterations=settings.bootstrap_iterations,
        bootstrap_block_months=settings.bootstrap_block_months,
        random_seed=settings.random_seed,
    )


def _stable_seed(base_seed: int, value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int((base_seed + int.from_bytes(digest[:4], "big")) % (2**32 - 1))


def _metric_map(frame: pd.DataFrame) -> dict[str, str]:
    return dict(
        zip(
            frame["metric"].astype(str),
            frame["value"].astype(str),
            strict=True,
        )
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _write_gzip_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip", encoding="utf-8")
    return path
