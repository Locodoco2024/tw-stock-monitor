from __future__ import annotations

import gzip
import hashlib
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from research.institutional_model.phase3_dataset import sha256_file
from research.institutional_model.phase4_model import (
    LABELS,
    LABEL_TO_INDEX,
    Preprocessor,
    SoftmaxModel,
    softmax,
)
from research.institutional_model.phase4_stability import (
    CONFIRMATION_START_YEAR,
    CORE_FEATURE_COLUMNS,
    DEVELOPMENT_END_YEAR,
)


PHASE4C_VERSION = "phase4c-v1"
TARGET_MARKET = "tpex"
TARGET_CANDIDATE = "core22_l2_1e-3"
INDEX_PROBABILITY_VARIANT = "raw"
HORIZONS = (5, 10, 20)
LIQUIDITY_THRESHOLDS_MILLION = (10, 20, 50, 100)
TOP_N_VALUES = (5, 10, 20)
EPSILON = 1e-12


@dataclass(frozen=True)
class Phase4CSettings:
    chunk_size: int = 100_000
    minimum_daily_stocks: int = 50
    bootstrap_iterations: int = 2_000
    bootstrap_block_months: int = 3
    random_seed: int = 20260728

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("Phase 4C chunk size 必須大於 0")
        if self.minimum_daily_stocks < 20:
            raise ValueError("Phase 4C 每日最少股票數不可小於 20")
        if self.bootstrap_iterations < 200:
            raise ValueError("Phase 4C bootstrap 次數不可小於 200")
        if self.bootstrap_block_months < 1:
            raise ValueError("Phase 4C bootstrap 區塊月份必須大於 0")


@dataclass(frozen=True)
class FoldScoringModel:
    test_year: int
    feature_columns: tuple[str, ...]
    preprocessor: Preprocessor
    model: SoftmaxModel


@dataclass(frozen=True)
class Phase4CResult:
    status: str
    ready_for_selection_index: bool
    scored_rows: int
    scored_dates: int
    output_paths: tuple[Path, ...]


def run_phase4c_selection_validation(
    *,
    output_dir: Path | str,
    shard_root: Path | str,
    settings: Phase4CSettings | None = None,
) -> Phase4CResult:
    """Validate whether TPEx institutional behavior supports same-day stock ranking."""
    config = settings or Phase4CSettings()
    config.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    validation = validate_phase4c_inputs(output)
    models = load_fold_scoring_models(output)
    training_path = output / "phase3_training_tpex.csv.gz"

    print("Phase 4C：重建 TPEx 逐筆樣本外原始法人佈局指數")
    scores = score_phase3_training_file(
        source_path=training_path,
        models=models,
        chunk_size=config.chunk_size,
        expected_source_rows=int(validation["phase3_source_rows"]),
    )
    validation.update(validate_scored_frame(scores, models=models))
    scores, dropped_dates = assign_same_day_ranks(
        scores,
        minimum_daily_stocks=config.minimum_daily_stocks,
    )
    if scores.empty:
        raise RuntimeError("Phase 4C 沒有符合每日橫斷面最低股票數的樣本")

    latest_year = int(scores["signal_year"].max())
    scores["research_period"] = np.where(
        scores["signal_year"] <= DEVELOPMENT_END_YEAR,
        "development",
        "confirmation",
    )

    print("Phase 4C：彙總同日十分位、五分位與持有期行為")
    daily_groups = build_daily_rank_group_report(scores)
    horizon_behavior = build_horizon_behavior_report(scores, latest_year=latest_year)
    yearly_stability = build_stability_report(scores, frequency="year")
    monthly_stability = build_stability_report(scores, frequency="month")
    top_n = build_top_n_report(scores, latest_year=latest_year)
    concentration = build_stock_concentration_report(scores)

    print("Phase 4C：執行 2,000 萬、5,000 萬、1 億元流動性敏感度")
    liquidity_reports: list[pd.DataFrame] = []
    for threshold in (20, 50, 100):
        if threshold == 20:
            subset = scores
        else:
            column = f"liquidity_pass_{threshold}m"
            subset = scores.loc[
                scores[column] == 1, _liquidity_analysis_columns()
            ].copy()
            subset, _ = assign_same_day_ranks(
                subset,
                minimum_daily_stocks=config.minimum_daily_stocks,
            )
        liquidity_reports.append(
            build_liquidity_sensitivity_report(
                {threshold: subset},
                latest_year=latest_year,
            )
        )
        if threshold != 20:
            del subset

    print("Phase 4C：從既有 Phase 3 分片補做 1,000 萬門檻敏感度")
    shard_dir = resolve_phase3_shard_directory(
        output_dir=output,
        shard_root=Path(shard_root),
    )
    ten_million_scores = score_tpex_shards_for_threshold(
        output_dir=output,
        shard_dir=shard_dir,
        models=models,
        threshold_million=10,
        chunk_size=config.chunk_size,
    )
    ten_million_scores = ten_million_scores[_liquidity_analysis_columns()].copy()
    ten_million_scores, _ = assign_same_day_ranks(
        ten_million_scores,
        minimum_daily_stocks=config.minimum_daily_stocks,
    )
    liquidity_reports.append(
        build_liquidity_sensitivity_report(
            {10: ten_million_scores},
            latest_year=latest_year,
        )
    )
    del ten_million_scores
    liquidity_sensitivity = pd.concat(liquidity_reports, ignore_index=True)

    print("Phase 4C：以月份移動區塊 bootstrap 檢查報酬差不確定性")
    bootstrap = build_bootstrap_report(
        scores,
        latest_year=latest_year,
        iterations=config.bootstrap_iterations,
        block_months=config.bootstrap_block_months,
        random_seed=config.random_seed,
    )

    summary, ready = build_phase4c_summary(
        validation=validation,
        scores=scores,
        dropped_dates=dropped_dates,
        yearly_stability=yearly_stability,
        liquidity_sensitivity=liquidity_sensitivity,
        bootstrap=bootstrap,
        concentration=concentration,
        settings=config,
        latest_year=latest_year,
    )

    paths = export_phase4c_reports(
        output_dir=output,
        scores=scores,
        daily_groups=daily_groups,
        horizon_behavior=horizon_behavior,
        yearly_stability=yearly_stability,
        monthly_stability=monthly_stability,
        liquidity_sensitivity=liquidity_sensitivity,
        top_n=top_n,
        bootstrap=bootstrap,
        concentration=concentration,
        summary=summary,
    )
    return Phase4CResult(
        status="PASS",
        ready_for_selection_index=ready,
        scored_rows=int(len(scores)),
        scored_dates=int(scores["signal_date"].nunique()),
        output_paths=tuple(paths),
    )


def validate_phase4c_inputs(output_dir: Path) -> dict[str, Any]:
    required = (
        "phase3_training_tpex.csv.gz",
        "phase3_dataset_manifest.csv",
        "phase3_summary.csv",
        "phase3_stock_summary.csv",
        "phase3b_summary.csv",
        "phase4b_summary.csv",
        "phase4b_selected_candidates.csv",
        "phase4b_market_decisions.csv",
        "phase4b_feature_sets.csv",
        "phase4b_coefficients.csv",
        "phase4b_preprocessing.csv",
    )
    missing = [name for name in required if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Phase 4C 缺少必要產物：{missing}")

    phase3b = _metric_map(output_dir / "phase3b_summary.csv")
    if phase3b.get("status") != "PASS" or phase3b.get("ready_for_modeling") != "1":
        raise RuntimeError("Phase 3B 尚未完整通過，不可執行 Phase 4C")

    phase4b = _metric_map(output_dir / "phase4b_summary.csv")
    if phase4b.get("pipeline_status") != "PASS":
        raise RuntimeError("Phase 4B 執行流程未完整通過，不可執行 Phase 4C")

    selected = pd.read_csv(
        output_dir / "phase4b_selected_candidates.csv",
        encoding="utf-8-sig",
        dtype=str,
    ).fillna("")
    selected = selected[selected["market_type"].str.lower() == TARGET_MARKET]
    if len(selected) != 1:
        raise RuntimeError("Phase 4B 必須只有一筆 TPEx 選定候選")
    selected_candidate = str(selected.iloc[0]["selected_candidate_id"])
    if selected_candidate != TARGET_CANDIDATE:
        raise RuntimeError(
            "Phase 4C 固定驗證 TPEx core22_l2_1e-3；"
            f"目前 Phase 4B 選定的是 {selected_candidate}"
        )

    decisions = pd.read_csv(
        output_dir / "phase4b_market_decisions.csv",
        encoding="utf-8-sig",
        dtype=str,
    ).fillna("")
    decision = decisions[decisions["market_type"].str.lower() == TARGET_MARKET]
    if len(decision) != 1:
        raise RuntimeError("Phase 4B 必須只有一筆 TPEx 市場決策")
    decision_name = str(decision.iloc[0]["decision"])
    if decision_name not in {"RANKING_ONLY", "PROBABILITY_AND_RANKING"}:
        raise RuntimeError(f"TPEx Phase 4B 決策為 {decision_name}，不可進 Phase 4C")

    manifest = pd.read_csv(
        output_dir / "phase3_dataset_manifest.csv",
        encoding="utf-8-sig",
        dtype=str,
    ).fillna("")
    manifest = manifest[manifest["file_name"] == "phase3_training_tpex.csv.gz"]
    if len(manifest) != 1:
        raise RuntimeError("Phase 3 manifest 缺少唯一 TPEx 訓練檔紀錄")
    source_sha256 = str(manifest.iloc[0]["sha256"])
    actual_sha256 = sha256_file(output_dir / "phase3_training_tpex.csv.gz")
    if actual_sha256 != source_sha256:
        raise RuntimeError("目前 TPEx 訓練檔 SHA-256 與 Phase 3 manifest 不一致")
    reported_sha256 = phase4b.get("tpex_source_sha256", "")
    if reported_sha256 and reported_sha256 != source_sha256:
        raise RuntimeError("Phase 4B 使用的 TPEx 訓練檔與目前 Phase 3 manifest 不一致")

    return {
        "phase3_source_sha256": source_sha256,
        "phase3_source_rows": int(manifest.iloc[0]["row_count"]),
        "phase3_config_signature": str(manifest.iloc[0]["config_signature"]),
        "phase4b_run_signature": phase4b.get("run_signature", ""),
        "phase4b_decision": decision_name,
        "selected_candidate": selected_candidate,
    }


def load_fold_scoring_models(output_dir: Path) -> dict[int, FoldScoringModel]:
    feature_frame = pd.read_csv(
        output_dir / "phase4b_feature_sets.csv",
        encoding="utf-8-sig",
        dtype=str,
    ).fillna("")
    feature_frame = feature_frame[
        feature_frame["candidate_id"] == TARGET_CANDIDATE
    ].copy()
    feature_frame["feature_order"] = pd.to_numeric(
        feature_frame["feature_order"], errors="raise"
    )
    feature_frame = feature_frame.sort_values("feature_order")
    features = tuple(feature_frame["feature"].astype(str))
    if features != tuple(CORE_FEATURE_COLUMNS):
        raise RuntimeError("Phase 4B core22 特徵順序與目前程式定義不一致")

    coefficient_frame = pd.read_csv(
        output_dir / "phase4b_coefficients.csv",
        encoding="utf-8-sig",
    )
    preprocessing_frame = pd.read_csv(
        output_dir / "phase4b_preprocessing.csv",
        encoding="utf-8-sig",
    )
    coefficient_frame = coefficient_frame[
        (coefficient_frame["market_type"].str.lower() == TARGET_MARKET)
        & (coefficient_frame["candidate_id"] == TARGET_CANDIDATE)
        & (coefficient_frame["class_label"].isin(LABELS))
    ].copy()
    preprocessing_frame = preprocessing_frame[
        (preprocessing_frame["market_type"].str.lower() == TARGET_MARKET)
        & (preprocessing_frame["candidate_id"] == TARGET_CANDIDATE)
    ].copy()

    coefficient_years = set(
        pd.to_numeric(coefficient_frame["test_year"], errors="raise").astype(int)
    )
    preprocessing_years = set(
        pd.to_numeric(preprocessing_frame["test_year"], errors="raise").astype(int)
    )
    if coefficient_years != preprocessing_years or not coefficient_years:
        raise RuntimeError("Phase 4B TPEx 模型係數與前處理年度不完整")
    ordered_years = sorted(coefficient_years)
    if ordered_years[0] != 2019 or ordered_years != list(
        range(ordered_years[0], ordered_years[-1] + 1)
    ):
        raise RuntimeError(f"Phase 4B TPEx 測試年度不連續：{ordered_years}")
    if sum(year >= CONFIRMATION_START_YEAR for year in ordered_years) < 3:
        raise RuntimeError("Phase 4B TPEx 確認期模型少於 3 年")

    models: dict[int, FoldScoringModel] = {}
    for year in sorted(coefficient_years):
        coefficients = coefficient_frame[
            pd.to_numeric(coefficient_frame["test_year"], errors="coerce") == year
        ]
        preprocessing = preprocessing_frame[
            pd.to_numeric(preprocessing_frame["test_year"], errors="coerce") == year
        ].set_index("feature")
        missing_preprocessing = set(features).difference(preprocessing.index)
        if missing_preprocessing:
            raise RuntimeError(
                f"Phase 4B {year} 缺少前處理特徵：{sorted(missing_preprocessing)}"
            )

        weights = np.empty((len(features), len(LABELS)), dtype=np.float64)
        intercept = np.empty(len(LABELS), dtype=np.float64)
        for class_index, label in enumerate(LABELS):
            class_rows = coefficients[coefficients["class_label"] == label]
            lookup = {
                str(row["feature"]): float(row["standardized_coefficient"])
                for _, row in class_rows.iterrows()
            }
            expected = {"(intercept)", *features}
            if set(lookup) != expected:
                missing = sorted(expected.difference(lookup))
                extra = sorted(set(lookup).difference(expected))
                raise RuntimeError(
                    f"Phase 4B {year} {label} 係數不完整；缺少 {missing}，多出 {extra}"
                )
            intercept[class_index] = lookup["(intercept)"]
            for feature_index, feature in enumerate(features):
                weights[feature_index, class_index] = lookup[feature]

        lower = preprocessing.loc[list(features), "clip_lower"].to_numpy(dtype=np.float64)
        upper = preprocessing.loc[list(features), "clip_upper"].to_numpy(dtype=np.float64)
        mean = preprocessing.loc[
            list(features), "training_mean_after_clip"
        ].to_numpy(dtype=np.float64)
        std = preprocessing.loc[
            list(features), "training_std_after_clip"
        ].to_numpy(dtype=np.float64)
        if not all(np.isfinite(value).all() for value in (lower, upper, mean, std)):
            raise RuntimeError(f"Phase 4B {year} 前處理參數包含非有限值")
        if np.any(std <= 0) or np.any(upper < lower):
            raise RuntimeError(f"Phase 4B {year} 前處理參數無效")
        first = preprocessing.iloc[0]
        models[year] = FoldScoringModel(
            test_year=year,
            feature_columns=features,
            preprocessor=Preprocessor(
                lower=lower,
                upper=upper,
                mean=mean,
                std=std,
                sampled_rows=int(first["quantile_sample_rows"]),
                training_rows=int(first["training_rows"]),
            ),
            model=SoftmaxModel(weights=weights, intercept=intercept),
        )
    return models


def score_phase3_training_file(
    *,
    source_path: Path,
    models: dict[int, FoldScoringModel],
    chunk_size: int,
    expected_source_rows: int | None = None,
) -> pd.DataFrame:
    required = _score_source_columns(next(iter(models.values())).feature_columns)
    frames: list[pd.DataFrame] = []
    processed = 0
    source_rows = 0
    for chunk in pd.read_csv(
        source_path,
        compression="gzip",
        usecols=required,
        chunksize=chunk_size,
        low_memory=False,
    ):
        source_rows += len(chunk)
        chunk["signal_year"] = pd.to_numeric(chunk["signal_year"], errors="coerce")
        chunk = chunk[chunk["signal_year"].isin(models)].copy()
        if chunk.empty:
            continue
        frames.append(score_source_frame(chunk, models=models))
        processed += len(chunk)
        if processed % 250_000 < len(chunk):
            print(f"  已產生 {processed:,} 筆樣本外指數")
    if expected_source_rows is not None and source_rows != expected_source_rows:
        raise RuntimeError(
            f"TPEx 訓練檔實際筆數 {source_rows:,} 與 manifest "
            f"{expected_source_rows:,} 不一致"
        )
    if not frames:
        raise RuntimeError("TPEx 訓練檔沒有 Phase 4B 測試年度資料")
    return _optimize_score_frame(pd.concat(frames, ignore_index=True))


def score_tpex_shards_for_threshold(
    *,
    output_dir: Path,
    shard_dir: Path,
    models: dict[int, FoldScoringModel],
    threshold_million: int,
    chunk_size: int,
) -> pd.DataFrame:
    stocks = pd.read_csv(
        output_dir / "phase3_stock_summary.csv",
        encoding="utf-8-sig",
        dtype=str,
    ).fillna("")
    stocks = stocks[
        (stocks["market_type"].str.lower() == TARGET_MARKET)
        & (stocks["phase3_status"] == "complete")
    ]
    stock_ids = sorted(set(stocks["stock_id"].astype(str)))
    required = _score_source_columns(next(iter(models.values())).feature_columns)
    liquidity_column = f"liquidity_pass_{threshold_million}m"
    frames: list[pd.DataFrame] = []
    for position, stock_id in enumerate(stock_ids, start=1):
        path = shard_dir / f"{stock_id}.csv.gz"
        if not path.exists():
            raise FileNotFoundError(f"Phase 3 分片遺失：{path}")
        for chunk in pd.read_csv(
            path,
            compression="gzip",
            usecols=required,
            chunksize=chunk_size,
            low_memory=False,
        ):
            years = pd.to_numeric(chunk["signal_year"], errors="coerce")
            liquidity = pd.to_numeric(chunk[liquidity_column], errors="coerce")
            chunk = chunk[
                years.isin(models)
                & (chunk["feature_status"] == "ok")
                & (chunk["label_status_10d"] == "ok")
                & (liquidity == 1)
            ].copy()
            if chunk.empty:
                continue
            chunk["signal_year"] = pd.to_numeric(
                chunk["signal_year"], errors="raise"
            ).astype(int)
            frames.append(score_source_frame(chunk, models=models))
        if position % 100 == 0 or position == len(stock_ids):
            print(f"  1,000 萬門檻分片進度：{position}/{len(stock_ids)}")
    if not frames:
        raise RuntimeError("Phase 3 分片沒有符合 1,000 萬門檻的 TPEx 樣本")
    return _optimize_score_frame(pd.concat(frames, ignore_index=True))


def score_source_frame(
    frame: pd.DataFrame,
    *,
    models: dict[int, FoldScoringModel],
) -> pd.DataFrame:
    frame = frame.copy()
    frame["signal_year"] = pd.to_numeric(frame["signal_year"], errors="raise").astype(int)
    result_frames: list[pd.DataFrame] = []
    for year, year_frame in frame.groupby("signal_year", sort=False):
        model_bundle = models.get(int(year))
        if model_bundle is None:
            continue
        features = year_frame[list(model_bundle.feature_columns)].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float64)
        if not np.isfinite(features).all():
            raise RuntimeError(f"TPEx {year} 含無效 Phase 4C 模型特徵")
        transformed = model_bundle.preprocessor.transform(features)
        probabilities = softmax(model_bundle.model.logits(transformed))
        scored = year_frame[_score_output_source_columns()].copy()
        scored["p_down_raw"] = probabilities[:, LABEL_TO_INDEX["DOWN"]]
        scored["p_flat_raw"] = probabilities[:, LABEL_TO_INDEX["FLAT"]]
        scored["p_up_raw"] = probabilities[:, LABEL_TO_INDEX["UP"]]
        scored["institutional_index_raw"] = 100.0 * (
            scored["p_up_raw"] - scored["p_down_raw"]
        )
        for horizon in HORIZONS:
            scored[f"horizon_valid_{horizon}d"] = (
                scored[f"label_status_{horizon}d"] == "ok"
            ).astype(np.uint8)
            for column in (
                f"adjusted_return_{horizon}d",
                f"max_adjusted_return_{horizon}d",
                f"min_adjusted_return_{horizon}d",
            ):
                scored[column] = pd.to_numeric(scored[column], errors="coerce")
        for threshold in LIQUIDITY_THRESHOLDS_MILLION:
            column = f"liquidity_pass_{threshold}m"
            scored[column] = pd.to_numeric(scored[column], errors="coerce").fillna(0).astype(
                np.uint8
            )
        result_frames.append(scored)
    if not result_frames:
        return pd.DataFrame(columns=_score_output_columns())
    return pd.concat(result_frames, ignore_index=True)


def validate_scored_frame(
    frame: pd.DataFrame,
    *,
    models: dict[int, FoldScoringModel],
) -> dict[str, Any]:
    duplicate_count = int(frame.duplicated(["stock_id", "signal_date"]).sum())
    if duplicate_count:
        raise RuntimeError(f"Phase 4C 樣本出現 {duplicate_count} 筆股票＋訊號日重複")
    markets = set(frame["market_type"].astype(str).str.lower())
    if markets != {TARGET_MARKET}:
        raise RuntimeError(f"Phase 4C 出現非 TPEx 市場資料：{sorted(markets)}")
    observed_years = sorted(int(value) for value in frame["signal_year"].unique())
    expected_years = sorted(models)
    if observed_years != expected_years:
        raise RuntimeError(
            f"Phase 4C 樣本外年度 {observed_years} 與模型年度 {expected_years} 不一致"
        )
    probability_columns = ["p_down_raw", "p_flat_raw", "p_up_raw"]
    probabilities = frame[probability_columns].to_numpy(dtype=np.float64)
    indices = frame["institutional_index_raw"].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all() or not np.isfinite(indices).all():
        raise RuntimeError("Phase 4C 樣本外機率或法人佈局指數包含非有限值")
    probability_sum_deviation = float(
        np.max(np.abs(probabilities.sum(axis=1) - 1.0))
    )
    reconstructed = 100.0 * (
        probabilities[:, LABEL_TO_INDEX["UP"]]
        - probabilities[:, LABEL_TO_INDEX["DOWN"]]
    )
    index_mismatch = float(np.max(np.abs(indices - reconstructed)))
    if probability_sum_deviation > 1e-6 or index_mismatch > 1e-5:
        raise RuntimeError(
            "Phase 4C 樣本外機率加總或法人佈局指數重建檢查失敗"
        )
    return {
        "oos_duplicate_stock_date_count": duplicate_count,
        "oos_probability_sum_max_deviation": probability_sum_deviation,
        "oos_index_reconstruction_max_mismatch": index_mismatch,
        "oos_model_years": ";".join(str(value) for value in observed_years),
    }


def assign_same_day_ranks(
    frame: pd.DataFrame,
    *,
    minimum_daily_stocks: int,
) -> tuple[pd.DataFrame, int]:
    if frame.empty:
        return frame.copy(), 0
    ranked = frame.copy()
    ranked["signal_date"] = ranked["signal_date"].astype(str)
    daily_counts = ranked.groupby("signal_date")["stock_id"].transform("size")
    dropped_dates = int(ranked.loc[daily_counts < minimum_daily_stocks, "signal_date"].nunique())
    ranked = ranked[daily_counts >= minimum_daily_stocks].copy()
    if ranked.empty:
        return ranked, dropped_dates

    ranked = ranked.sort_values(
        ["signal_date", "institutional_index_raw", "stock_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    grouped = ranked.groupby("signal_date", sort=False)
    ranked["daily_rank"] = grouped.cumcount() + 1
    ranked["daily_stock_count"] = grouped["stock_id"].transform("size").astype(int)
    ascending_rank = grouped["institutional_index_raw"].rank(
        method="average", ascending=True
    )
    denominator = (ranked["daily_stock_count"] - 1).replace(0, np.nan)
    percentile = 100.0 * (ascending_rank - 1.0) / denominator
    ranked["daily_percentile"] = percentile.fillna(50.0).clip(0.0, 100.0)
    ranked["daily_decile"] = np.ceil(ranked["daily_percentile"] / 10.0).clip(1, 10).astype(int)
    ranked["daily_quintile"] = np.ceil(ranked["daily_percentile"] / 20.0).clip(1, 5).astype(int)
    return ranked, dropped_dates


def build_daily_rank_group_report(scores: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        return_column = f"adjusted_return_{horizon}d"
        valid = scores[
            (scores[f"horizon_valid_{horizon}d"] == 1)
            & scores[return_column].notna()
        ].copy()
        if valid.empty:
            continue
        valid["_positive"] = (valid[return_column] > 0).astype(np.uint8)
        valid["_up_5pct"] = (valid[return_column] >= 0.05).astype(np.uint8)
        valid["_down_5pct"] = (valid[return_column] <= -0.05).astype(np.uint8)
        valid["_label_up"] = (valid["label_10d"] == "UP").astype(np.uint8)
        valid["_label_flat"] = (valid["label_10d"] == "FLAT").astype(np.uint8)
        valid["_label_down"] = (valid["label_10d"] == "DOWN").astype(np.uint8)
        for scheme, column in (
            ("decile", "daily_decile"),
            ("quintile", "daily_quintile"),
        ):
            keys = ["signal_date", column]
            grouped = valid.groupby(keys, sort=True, observed=True)
            aggregate = grouped.agg(
                sample_count=(return_column, "size"),
                unique_stocks=("stock_id", "nunique"),
                average_return=(return_column, "mean"),
                median_return=(return_column, "median"),
                positive_return_rate=("_positive", "mean"),
                up_5pct_rate=("_up_5pct", "mean"),
                down_5pct_rate=("_down_5pct", "mean"),
                average_max_adjusted_return=(
                    f"max_adjusted_return_{horizon}d",
                    "mean",
                ),
                average_min_adjusted_return=(
                    f"min_adjusted_return_{horizon}d",
                    "mean",
                ),
                label_up_rate=("_label_up", "mean"),
                label_flat_rate=("_label_flat", "mean"),
                label_down_rate=("_label_down", "mean"),
            )
            quantiles = grouped[return_column].quantile(
                [0.10, 0.25, 0.75, 0.90]
            ).unstack(level=-1)
            quantiles.columns = [
                "return_p10",
                "return_p25",
                "return_p75",
                "return_p90",
            ]
            aggregate = aggregate.join(quantiles).reset_index()
            aggregate = aggregate.rename(columns={column: "group_value"})
            aggregate.insert(
                1,
                "signal_year",
                aggregate["signal_date"].astype(str).str[:4].astype(int),
            )
            aggregate.insert(2, "signal_month", aggregate["signal_date"].astype(str).str[:7])
            aggregate.insert(3, "horizon_days", horizon)
            aggregate.insert(4, "group_scheme", scheme)
            aggregate["signal_dates"] = 1
            if horizon != 10:
                aggregate[["label_up_rate", "label_flat_rate", "label_down_rate"]] = ""
            frames.append(aggregate)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_horizon_behavior_report(
    scores: pd.DataFrame,
    *,
    latest_year: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period_name, period in _period_frames(scores, latest_year=latest_year):
        if period.empty:
            continue
        for horizon in HORIZONS:
            valid = period[
                (period[f"horizon_valid_{horizon}d"] == 1)
                & period[f"adjusted_return_{horizon}d"].notna()
            ]
            for scheme, column in (("decile", "daily_decile"), ("quintile", "daily_quintile")):
                for group_value, group in valid.groupby(column, sort=True):
                    daily_mean = group.groupby("signal_date")[
                        f"adjusted_return_{horizon}d"
                    ].mean()
                    rows.append(
                        {
                            "period": period_name,
                            "horizon_days": horizon,
                            "group_scheme": scheme,
                            "group_value": int(group_value),
                            "equal_day_average_return": float(daily_mean.mean()),
                            "equal_day_median_return": float(daily_mean.median()),
                            **_return_metrics(group, horizon),
                        }
                    )
    return pd.DataFrame(rows)


def build_stability_report(scores: pd.DataFrame, *, frequency: str) -> pd.DataFrame:
    if frequency not in {"year", "month"}:
        raise ValueError("stability frequency 僅支援 year 或 month")
    period_column = "signal_year" if frequency == "year" else "signal_month"
    work = scores.copy()
    work["signal_month"] = work["signal_date"].astype(str).str[:7]
    rows: list[dict[str, Any]] = []
    for period_value, period in work.groupby(period_column, sort=True):
        for horizon in HORIZONS:
            valid = period[
                (period[f"horizon_valid_{horizon}d"] == 1)
                & period[f"adjusted_return_{horizon}d"].notna()
            ]
            if valid.empty:
                continue
            decile_returns = (
                valid.groupby(["signal_date", "daily_decile"])[
                    f"adjusted_return_{horizon}d"
                ]
                .mean()
                .groupby("daily_decile")
                .mean()
                .reindex(range(1, 11))
            )
            correlation = _safe_correlation(
                np.arange(1, 11, dtype=np.float64),
                decile_returns.to_numpy(dtype=np.float64),
            )
            violations = int(
                np.sum(np.diff(decile_returns.to_numpy(dtype=np.float64)) < 0)
            )
            for band in (10, 20):
                spreads = daily_top_bottom_spreads(valid, horizon=horizon, band_percent=band)
                rows.append(
                    {
                        "frequency": frequency,
                        "period": period_value,
                        "research_period": (
                            "development"
                            if int(str(period_value)[:4]) <= DEVELOPMENT_END_YEAR
                            else "confirmation"
                        ),
                        "horizon_days": horizon,
                        "band_percent": band,
                        "daily_spread_days": int(len(spreads)),
                        "average_daily_high_minus_low_return": float(spreads.mean()),
                        "median_daily_high_minus_low_return": float(spreads.median()),
                        "positive_daily_spread_rate": float((spreads > 0).mean()),
                        "decile_return_correlation": correlation,
                        "decile_adjacent_violations": violations,
                        "positive_period_direction": int(float(spreads.mean()) > 0),
                    }
                )
    return pd.DataFrame(rows)


def build_liquidity_sensitivity_report(
    frames: dict[int, pd.DataFrame],
    *,
    latest_year: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for threshold, scores in sorted(frames.items()):
        for period_name, period in _period_frames(scores, latest_year=latest_year):
            if period.empty:
                continue
            for horizon in HORIZONS:
                valid = period[
                    (period[f"horizon_valid_{horizon}d"] == 1)
                    & period[f"adjusted_return_{horizon}d"].notna()
                ]
                if valid.empty:
                    continue
                decile_returns = (
                    valid.groupby(["signal_date", "daily_decile"])[
                        f"adjusted_return_{horizon}d"
                    ]
                    .mean()
                    .groupby("daily_decile")
                    .mean()
                    .reindex(range(1, 11))
                )
                for band in (10, 20):
                    spreads = daily_top_bottom_spreads(
                        valid, horizon=horizon, band_percent=band
                    )
                    rows.append(
                        {
                            "threshold_ntd_million": threshold,
                            "period": period_name,
                            "horizon_days": horizon,
                            "band_percent": band,
                            "sample_rows": int(len(valid)),
                            "signal_dates": int(valid["signal_date"].nunique()),
                            "average_daily_stocks": float(
                                valid.groupby("signal_date")["stock_id"].size().mean()
                            ),
                            "average_daily_high_minus_low_return": float(spreads.mean()),
                            "median_daily_high_minus_low_return": float(spreads.median()),
                            "positive_daily_spread_rate": float((spreads > 0).mean()),
                            "decile_return_correlation": _safe_correlation(
                                np.arange(1, 11, dtype=np.float64),
                                decile_returns.to_numpy(dtype=np.float64),
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def build_top_n_report(scores: pd.DataFrame, *, latest_year: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period_name, period in _period_frames(scores, latest_year=latest_year):
        for horizon in HORIZONS:
            valid = period[
                (period[f"horizon_valid_{horizon}d"] == 1)
                & period[f"adjusted_return_{horizon}d"].notna()
            ].copy()
            if valid.empty:
                continue
            daily_universe = valid.groupby("signal_date")[
                f"adjusted_return_{horizon}d"
            ].mean()
            for top_n in TOP_N_VALUES:
                selected = valid[valid["daily_rank"] <= top_n]
                daily_selected = selected.groupby("signal_date")[
                    f"adjusted_return_{horizon}d"
                ].mean()
                aligned = pd.concat(
                    [daily_selected.rename("selected"), daily_universe.rename("universe")],
                    axis=1,
                    join="inner",
                ).dropna()
                rows.append(
                    {
                        "period": period_name,
                        "horizon_days": horizon,
                        "top_n": top_n,
                        "selected_rows": int(len(selected)),
                        "signal_dates": int(len(aligned)),
                        "average_selected_return": float(daily_selected.mean()),
                        "median_selected_return": float(daily_selected.median()),
                        "average_daily_excess_return": float(
                            (aligned["selected"] - aligned["universe"]).mean()
                        ),
                        "positive_selected_day_rate": float((daily_selected > 0).mean()),
                        **_return_metrics(selected, horizon),
                    }
                )
    return pd.DataFrame(rows)


def build_bootstrap_report(
    scores: pd.DataFrame,
    *,
    latest_year: int,
    iterations: int,
    block_months: int,
    random_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period_name, period in _period_frames(scores, latest_year=latest_year):
        for horizon in HORIZONS:
            valid = period[
                (period[f"horizon_valid_{horizon}d"] == 1)
                & period[f"adjusted_return_{horizon}d"].notna()
            ]
            for band in (10, 20):
                daily = daily_top_bottom_spreads(
                    valid, horizon=horizon, band_percent=band
                )
                if daily.empty:
                    continue
                monthly = daily.groupby(daily.index.astype(str).str[:7]).mean().sort_index()
                seed = _stable_seed(
                    random_seed,
                    f"{period_name}:{horizon}:{band}",
                )
                lower, upper, bootstrap_mean = moving_block_bootstrap_mean_ci(
                    monthly.to_numpy(dtype=np.float64),
                    iterations=iterations,
                    block_length=block_months,
                    random_seed=seed,
                )
                rows.append(
                    {
                        "period": period_name,
                        "horizon_days": horizon,
                        "band_percent": band,
                        "daily_observations": int(len(daily)),
                        "monthly_blocks": int(len(monthly)),
                        "point_estimate_daily_spread": float(daily.mean()),
                        "monthly_mean_spread": float(monthly.mean()),
                        "bootstrap_mean_spread": bootstrap_mean,
                        "confidence_level": 0.95,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "ci_excludes_zero_positive": int(lower > 0),
                        "iterations": iterations,
                        "block_months": min(block_months, len(monthly)),
                    }
                )
    return pd.DataFrame(rows)


def moving_block_bootstrap_mean_ci(
    values: np.ndarray,
    *,
    iterations: int,
    block_length: int,
    random_seed: int,
) -> tuple[float, float, float]:
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if len(clean) < 2:
        raise ValueError("moving block bootstrap 至少需要兩個有效區塊")
    length = len(clean)
    block = min(max(1, block_length), length)
    blocks_needed = math.ceil(length / block)
    rng = np.random.default_rng(random_seed)
    means = np.empty(iterations, dtype=np.float64)
    offsets = np.arange(block, dtype=np.int64)
    for iteration in range(iterations):
        starts = rng.integers(0, length, size=blocks_needed)
        indices = np.concatenate(
            [((start + offsets) % length) for start in starts]
        )[:length]
        means[iteration] = clean[indices].mean()
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper), float(means.mean())


def build_stock_concentration_report(scores: pd.DataFrame) -> pd.DataFrame:
    confirmation = scores[scores["signal_year"] >= CONFIRMATION_START_YEAR]
    rows: list[dict[str, Any]] = []
    for band in (10, 20):
        selected = confirmation[confirmation["daily_percentile"] > 100 - band].copy()
        total_rows = len(selected)
        counts = selected.groupby(
            ["stock_id", "stock_name"], sort=False, observed=True
        ).size()
        shares = counts / max(total_rows, 1)
        sorted_shares = shares.sort_values(ascending=False)
        summary = {
            "row_type": "summary",
            "band_percent": band,
            "stock_id": "(summary)",
            "stock_name": "",
            "selected_rows": total_rows,
            "selected_days": int(selected["signal_date"].nunique()),
            "selection_share": 1.0,
            "unique_stocks": int(len(counts)),
            "top1_stock_share": float(sorted_shares.iloc[:1].sum()),
            "top5_stock_share": float(sorted_shares.iloc[:5].sum()),
            "top10_stock_share": float(sorted_shares.iloc[:10].sum()),
            "selection_hhi": float(np.square(sorted_shares.to_numpy()).sum()),
            "average_return_10d": float(selected["adjusted_return_10d"].mean()),
            "median_return_10d": float(selected["adjusted_return_10d"].median()),
        }
        rows.append(summary)
        grouped = selected.groupby(
            ["stock_id", "stock_name"], sort=False, observed=True
        )
        detail = grouped.agg(
            selected_rows=("signal_date", "size"),
            selected_days=("signal_date", "nunique"),
            average_return_10d=("adjusted_return_10d", "mean"),
            median_return_10d=("adjusted_return_10d", "median"),
        ).reset_index()
        detail["selection_share"] = detail["selected_rows"] / max(total_rows, 1)
        detail = detail.sort_values(
            ["selected_rows", "stock_id"], ascending=[False, True]
        )
        for _, row in detail.iterrows():
            rows.append(
                {
                    "row_type": "stock",
                    "band_percent": band,
                    "stock_id": row["stock_id"],
                    "stock_name": row["stock_name"],
                    "selected_rows": int(row["selected_rows"]),
                    "selected_days": int(row["selected_days"]),
                    "selection_share": float(row["selection_share"]),
                    "unique_stocks": "",
                    "top1_stock_share": "",
                    "top5_stock_share": "",
                    "top10_stock_share": "",
                    "selection_hhi": "",
                    "average_return_10d": float(row["average_return_10d"]),
                    "median_return_10d": float(row["median_return_10d"]),
                }
            )
    return pd.DataFrame(rows)


def build_phase4c_summary(
    *,
    validation: dict[str, Any],
    scores: pd.DataFrame,
    dropped_dates: int,
    yearly_stability: pd.DataFrame,
    liquidity_sensitivity: pd.DataFrame,
    bootstrap: pd.DataFrame,
    concentration: pd.DataFrame,
    settings: Phase4CSettings,
    latest_year: int,
) -> tuple[pd.DataFrame, bool]:
    confirmation_yearly = yearly_stability[
        (yearly_stability["research_period"] == "confirmation")
        & (yearly_stability["horizon_days"] == 10)
        & (yearly_stability["band_percent"] == 20)
    ]
    positive_years = int(confirmation_yearly["positive_period_direction"].sum())
    confirmation_years = int(len(confirmation_yearly))
    required_positive_years = max(1, math.ceil(confirmation_years * 0.75))
    positive_monotonic_years = int(
        (confirmation_yearly["decile_return_correlation"] > 0).sum()
    )

    confirmation_bootstrap = _single_report_row(
        bootstrap,
        period="confirmation",
        horizon_days=10,
        band_percent=20,
    )
    excluding_latest_bootstrap = _single_report_row(
        bootstrap,
        period="confirmation_ex_latest_year",
        horizon_days=10,
        band_percent=20,
    )
    liquidity_50m = _single_report_row(
        liquidity_sensitivity,
        threshold_ntd_million=50,
        period="confirmation",
        horizon_days=10,
        band_percent=20,
    )
    concentration_summary = _single_report_row(
        concentration,
        row_type="summary",
        band_percent=20,
    )

    checks = {
        "confirmation_spread_positive": float(
            confirmation_bootstrap["point_estimate_daily_spread"]
        ) > 0,
        "confirmation_year_direction_stable": positive_years
        >= required_positive_years,
        "confirmation_monotonic_years_stable": positive_monotonic_years
        >= required_positive_years,
        "excluding_latest_year_spread_positive": float(
            excluding_latest_bootstrap["point_estimate_daily_spread"]
        ) > 0,
        "bootstrap_ci_lower_positive": float(confirmation_bootstrap["ci_lower"]) > 0,
        "liquidity_50m_spread_positive": float(
            liquidity_50m["average_daily_high_minus_low_return"]
        ) > 0,
    }
    ready = all(checks.values())
    decision = "PROCEED_TO_SELECTION_INDEX_BUILD" if ready else "KEEP_RESEARCH_ONLY"
    failed_checks = [name for name, passed in checks.items() if not passed]

    metrics: list[tuple[str, Any]] = [
        ("phase4c_version", PHASE4C_VERSION),
        ("pipeline_status", "PASS"),
        ("target_market", TARGET_MARKET),
        ("selected_candidate", TARGET_CANDIDATE),
        ("index_probability_variant", INDEX_PROBABILITY_VARIANT),
        ("phase4b_decision", validation["phase4b_decision"]),
        ("phase4b_run_signature", validation["phase4b_run_signature"]),
        ("phase3_config_signature", validation["phase3_config_signature"]),
        ("phase3_source_sha256", validation["phase3_source_sha256"]),
        ("phase3_source_rows", validation["phase3_source_rows"]),
        (
            "oos_duplicate_stock_date_count",
            validation["oos_duplicate_stock_date_count"],
        ),
        (
            "oos_probability_sum_max_deviation",
            validation["oos_probability_sum_max_deviation"],
        ),
        (
            "oos_index_reconstruction_max_mismatch",
            validation["oos_index_reconstruction_max_mismatch"],
        ),
        ("oos_model_years", validation["oos_model_years"]),
        ("scored_rows", len(scores)),
        ("scored_dates", scores["signal_date"].nunique()),
        ("first_signal_date", scores["signal_date"].min()),
        ("last_signal_date", scores["signal_date"].max()),
        ("latest_available_year", latest_year),
        ("minimum_daily_stocks", settings.minimum_daily_stocks),
        ("dropped_low_cross_section_dates", dropped_dates),
        (
            "daily_rank_duplicate_count",
            int(scores.duplicated(["signal_date", "daily_rank"]).sum()),
        ),
        (
            "minimum_observed_daily_stocks",
            int(scores.groupby("signal_date")["stock_id"].size().min()),
        ),
        (
            "maximum_observed_daily_stocks",
            int(scores.groupby("signal_date")["stock_id"].size().max()),
        ),
        ("bootstrap_iterations", settings.bootstrap_iterations),
        ("bootstrap_block_months", settings.bootstrap_block_months),
        ("confirmation_years", confirmation_years),
        ("confirmation_positive_spread_years", positive_years),
        ("required_positive_spread_years", required_positive_years),
        ("confirmation_positive_monotonic_years", positive_monotonic_years),
        (
            "confirmation_10d_top20_minus_bottom20",
            confirmation_bootstrap["point_estimate_daily_spread"],
        ),
        (
            "confirmation_ex_latest_10d_top20_minus_bottom20",
            excluding_latest_bootstrap["point_estimate_daily_spread"],
        ),
        ("confirmation_10d_bootstrap_ci_lower", confirmation_bootstrap["ci_lower"]),
        ("confirmation_10d_bootstrap_ci_upper", confirmation_bootstrap["ci_upper"]),
        (
            "confirmation_50m_10d_top20_minus_bottom20",
            liquidity_50m["average_daily_high_minus_low_return"],
        ),
        ("confirmation_top20_unique_stocks", concentration_summary["unique_stocks"]),
        ("confirmation_top20_top10_stock_share", concentration_summary["top10_stock_share"]),
    ]
    metrics.extend((f"check_{name}", int(value)) for name, value in checks.items())
    metrics.extend(
        [
            ("ready_for_selection_index", int(ready)),
            ("decision", decision),
            ("failed_checks", ";".join(failed_checks)),
        ]
    )
    return pd.DataFrame([{"metric": key, "value": value} for key, value in metrics]), ready


def export_phase4c_reports(
    *,
    output_dir: Path,
    scores: pd.DataFrame,
    daily_groups: pd.DataFrame,
    horizon_behavior: pd.DataFrame,
    yearly_stability: pd.DataFrame,
    monthly_stability: pd.DataFrame,
    liquidity_sensitivity: pd.DataFrame,
    top_n: pd.DataFrame,
    bootstrap: pd.DataFrame,
    concentration: pd.DataFrame,
    summary: pd.DataFrame,
) -> list[Path]:
    paths = [
        _write_gzip_csv(output_dir / "phase4c_oos_scores.csv.gz", scores),
        _write_csv(output_dir / "phase4c_daily_rank_groups.csv", daily_groups),
        _write_csv(output_dir / "phase4c_horizon_behavior.csv", horizon_behavior),
        _write_csv(output_dir / "phase4c_yearly_stability.csv", yearly_stability),
        _write_csv(output_dir / "phase4c_monthly_stability.csv", monthly_stability),
        _write_csv(output_dir / "phase4c_liquidity_sensitivity.csv", liquidity_sensitivity),
        _write_csv(output_dir / "phase4c_top_n_analysis.csv", top_n),
        _write_csv(output_dir / "phase4c_bootstrap_confidence.csv", bootstrap),
        _write_csv(output_dir / "phase4c_stock_concentration.csv", concentration),
        _write_csv(output_dir / "phase4c_summary.csv", summary),
    ]
    archive = output_dir / "phase4c_validation_reports.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in paths:
            handle.write(path, arcname=path.name)
    paths.append(archive)
    return paths


def resolve_phase3_shard_directory(*, output_dir: Path, shard_root: Path) -> Path:
    summary = _metric_map(output_dir / "phase3_summary.csv")
    signature = summary.get("config_signature", "")
    if not signature:
        raise RuntimeError("phase3_summary.csv 缺少 config_signature")
    shard_dir = shard_root / signature[:16]
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"找不到 Phase 3 分片目錄：{shard_dir}")
    return shard_dir


def daily_top_bottom_spreads(
    scores: pd.DataFrame,
    *,
    horizon: int,
    band_percent: int,
) -> pd.Series:
    if band_percent not in {10, 20}:
        raise ValueError("band_percent 僅支援 10 或 20")
    return_column = f"adjusted_return_{horizon}d"
    top = scores[scores["daily_percentile"] > 100 - band_percent]
    bottom = scores[scores["daily_percentile"] <= band_percent]
    top_daily = top.groupby("signal_date")[return_column].mean()
    bottom_daily = bottom.groupby("signal_date")[return_column].mean()
    aligned = pd.concat(
        [top_daily.rename("top"), bottom_daily.rename("bottom")],
        axis=1,
        join="inner",
    ).dropna()
    spread = aligned["top"] - aligned["bottom"]
    spread.index = spread.index.astype(str)
    return spread


def _return_metrics(frame: pd.DataFrame, horizon: int) -> dict[str, Any]:
    return_column = f"adjusted_return_{horizon}d"
    max_column = f"max_adjusted_return_{horizon}d"
    min_column = f"min_adjusted_return_{horizon}d"
    returns = pd.to_numeric(frame[return_column], errors="coerce").dropna()
    result: dict[str, Any] = {
        "sample_count": int(len(returns)),
        "signal_dates": int(frame.loc[returns.index, "signal_date"].nunique()),
        "unique_stocks": int(frame.loc[returns.index, "stock_id"].nunique()),
        "average_return": float(returns.mean()),
        "median_return": float(returns.median()),
        "return_p10": float(returns.quantile(0.10)),
        "return_p25": float(returns.quantile(0.25)),
        "return_p75": float(returns.quantile(0.75)),
        "return_p90": float(returns.quantile(0.90)),
        "positive_return_rate": float((returns > 0).mean()),
        "up_5pct_rate": float((returns >= 0.05).mean()),
        "down_5pct_rate": float((returns <= -0.05).mean()),
        "average_max_adjusted_return": float(
            pd.to_numeric(frame.loc[returns.index, max_column], errors="coerce").mean()
        ),
        "average_min_adjusted_return": float(
            pd.to_numeric(frame.loc[returns.index, min_column], errors="coerce").mean()
        ),
    }
    if horizon == 10:
        labels = frame.loc[returns.index, "label_10d"]
        result.update(
            {
                "label_up_rate": float((labels == "UP").mean()),
                "label_flat_rate": float((labels == "FLAT").mean()),
                "label_down_rate": float((labels == "DOWN").mean()),
            }
        )
    else:
        result.update(
            {"label_up_rate": "", "label_flat_rate": "", "label_down_rate": ""}
        )
    return result


def _period_frames(
    scores: pd.DataFrame,
    *,
    latest_year: int,
) -> Iterable[tuple[str, pd.DataFrame]]:
    yield "all", scores
    yield "development", scores[scores["signal_year"] <= DEVELOPMENT_END_YEAR]
    confirmation = scores[scores["signal_year"] >= CONFIRMATION_START_YEAR]
    yield "confirmation", confirmation
    yield "confirmation_ex_latest_year", confirmation[
        confirmation["signal_year"] < latest_year
    ]


def _liquidity_analysis_columns() -> list[str]:
    columns = [
        "stock_id",
        "stock_name",
        "signal_date",
        "signal_year",
        "label_10d",
        "institutional_index_raw",
    ]
    for horizon in HORIZONS:
        columns.extend(
            [
                f"horizon_valid_{horizon}d",
                f"adjusted_return_{horizon}d",
                f"max_adjusted_return_{horizon}d",
                f"min_adjusted_return_{horizon}d",
            ]
        )
    return columns


def _optimize_score_frame(frame: pd.DataFrame) -> pd.DataFrame:
    optimized = frame.copy()
    for column in (
        "stock_id",
        "stock_name",
        "market_type",
        "feature_status",
        "label_10d",
        *(f"label_status_{horizon}d" for horizon in HORIZONS),
    ):
        optimized[column] = optimized[column].astype("category")
    optimized["signal_year"] = optimized["signal_year"].astype(np.int16)
    for column in (
        *(
            value
            for horizon in HORIZONS
            for value in (
                f"adjusted_return_{horizon}d",
                f"max_adjusted_return_{horizon}d",
                f"min_adjusted_return_{horizon}d",
            )
        ),
    ):
        optimized[column] = pd.to_numeric(optimized[column], errors="coerce").astype(
            np.float32
        )
    return optimized


def _score_source_columns(features: tuple[str, ...]) -> list[str]:
    return [*_score_output_source_columns(), *features]


def _score_output_source_columns() -> list[str]:
    columns = [
        "stock_id",
        "stock_name",
        "market_type",
        "signal_date",
        "signal_year",
        "feature_status",
        "label_10d",
    ]
    columns.extend(f"liquidity_pass_{value}m" for value in LIQUIDITY_THRESHOLDS_MILLION)
    for horizon in HORIZONS:
        columns.extend(
            [
                f"label_status_{horizon}d",
                f"adjusted_return_{horizon}d",
                f"max_adjusted_return_{horizon}d",
                f"min_adjusted_return_{horizon}d",
            ]
        )
    return columns


def _score_output_columns() -> list[str]:
    return [
        *_score_output_source_columns(),
        "p_down_raw",
        "p_flat_raw",
        "p_up_raw",
        "institutional_index_raw",
        *(f"horizon_valid_{horizon}d" for horizon in HORIZONS),
    ]


def _metric_map(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    return {str(row["metric"]): str(row["value"]) for _, row in frame.iterrows()}


def _single_report_row(frame: pd.DataFrame, **conditions: Any) -> pd.Series:
    selected = frame
    for column, value in conditions.items():
        selected = selected[selected[column] == value]
    if len(selected) != 1:
        raise RuntimeError(f"Phase 4C 報告條件 {conditions} 預期一筆，實際 {len(selected)} 筆")
    return selected.iloc[0]


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
    with gzip.open(path, "wt", encoding="utf-8-sig", newline="") as handle:
        frame.to_csv(handle, index=False)
    return path
