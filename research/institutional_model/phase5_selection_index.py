from __future__ import annotations

import hashlib
import json
import os
import zipfile
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.institutional_model.phase3_dataset import FEATURE_COLUMNS, sha256_file
from research.institutional_model.phase4_model import (
    EPSILON,
    LABELS,
    LABEL_TO_INDEX,
    MarketCache,
    Preprocessor,
    SoftmaxModel,
    build_or_load_market_cache,
    multiclass_log_loss,
    softmax,
)
from research.institutional_model.phase4_selection import (
    HORIZONS,
    assign_same_day_ranks,
    resolve_phase3_shard_directory,
)
from research.institutional_model.phase4_stability import CORE_FEATURE_COLUMNS


PHASE5A_VERSION = "phase5a-v1"
TARGET_MARKET = "tpex"
TARGET_CANDIDATE = "core22_l2_1e-3"
DEFAULT_FIXED_EPOCHS = 3


@dataclass(frozen=True)
class Phase5ASettings:
    chunk_size: int = 100_000
    quantile_sample_size: int = 250_000
    batch_size: int = 65_536
    training_epochs: int = DEFAULT_FIXED_EPOCHS
    learning_rate: float = 0.02
    l2_penalty: float = 0.001
    minimum_daily_stocks: int = 50
    recent_rows_per_stock: int = 90
    random_seed: int = 20260728

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("Phase 5A chunk size 必須大於 0")
        if self.quantile_sample_size <= 0:
            raise ValueError("Phase 5A 分位數抽樣筆數必須大於 0")
        if self.batch_size <= 0:
            raise ValueError("Phase 5A batch size 必須大於 0")
        if self.training_epochs <= 0:
            raise ValueError("Phase 5A 固定訓練 epoch 必須大於 0")
        if self.learning_rate <= 0:
            raise ValueError("Phase 5A learning rate 必須大於 0")
        if self.l2_penalty < 0:
            raise ValueError("Phase 5A L2 penalty 不可小於 0")
        if self.minimum_daily_stocks < 20:
            raise ValueError("Phase 5A 同日股票數不可低於 20")
        if self.recent_rows_per_stock < 10:
            raise ValueError("Phase 5A 每檔保留的近期列數不可低於 10")


@dataclass(frozen=True)
class FinalModelBundle:
    signature: str
    feature_columns: tuple[str, ...]
    preprocessor: Preprocessor
    model: SoftmaxModel
    priors: np.ndarray
    training_rows: int
    first_training_year: int
    last_training_year: int
    first_training_date: str
    last_training_date: str
    source_sha256: str
    training_epochs: int


@dataclass(frozen=True)
class Phase5AResult:
    status: str
    signal_date: str
    selected_rows: int
    output_paths: tuple[Path, ...]


def run_phase5a_selection_index(
    *,
    output_dir: Path | str,
    cache_root: Path | str,
    shard_root: Path | str,
    model_root: Path | str,
    settings: Phase5ASettings | None = None,
    signal_date: str | None = None,
    force: bool = False,
) -> Phase5AResult:
    config = settings or Phase5ASettings()
    config.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    validation = validate_phase5a_inputs(output, config=config)
    manifest = load_phase3_manifest(output / "phase3_dataset_manifest.csv")
    cache = build_or_load_market_cache(
        output_dir=output,
        cache_root=Path(cache_root),
        manifest=manifest,
        market=TARGET_MARKET,
        chunk_size=config.chunk_size,
        force=False,
    )
    model_base = Path(model_root)
    bundle = build_or_load_final_model(
        cache=cache,
        source_sha256=validation["phase3_source_sha256"],
        first_training_date=validation["first_training_date"],
        last_training_date=validation["last_training_date"],
        model_root=model_base,
        settings=config,
        force=force,
    )

    shard_dir = resolve_phase3_shard_directory(
        output_dir=output,
        shard_root=Path(shard_root),
    )
    cross_section, resolved_date, available_dates = load_latest_tpex_cross_section(
        output_dir=output,
        shard_dir=shard_dir,
        feature_columns=bundle.feature_columns,
        settings=config,
        requested_signal_date=signal_date,
    )
    if resolved_date <= bundle.last_training_date:
        raise RuntimeError(
            "Phase 5A 最終模型不可回看訓練截止日以前的歷史訊號；"
            "歷史樣本外結果請使用 Phase 4C。"
            f" 訓練截止 {bundle.last_training_date}，訊號日 {resolved_date}"
        )
    scored = score_latest_cross_section(cross_section, bundle=bundle)
    ranked = rank_liquidity_universes(
        scored,
        minimum_daily_stocks=config.minimum_daily_stocks,
    )
    lookup = build_historical_behavior_lookup(output)
    selection = attach_historical_behavior(ranked, lookup=lookup)
    selection = finalize_selection_columns(selection)
    summary = build_phase5a_summary(
        validation=validation,
        bundle=bundle,
        selection=selection,
        resolved_date=resolved_date,
        available_dates=available_dates,
        settings=config,
    )
    model_reports = export_model_reports(
        bundle=bundle,
        settings=config,
        model_root=model_base,
    )
    paths = export_phase5a_reports(
        output_dir=output,
        selection=selection,
        lookup=lookup,
        summary=summary,
        model_reports=model_reports,
    )
    return Phase5AResult(
        status="PASS",
        signal_date=resolved_date,
        selected_rows=int(len(selection)),
        output_paths=tuple(paths),
    )


def validate_phase5a_inputs(output_dir: Path, *, config: Phase5ASettings) -> dict[str, Any]:
    required = (
        "phase3_training_tpex.csv.gz",
        "phase3_dataset_manifest.csv",
        "phase3_stock_summary.csv",
        "phase3b_summary.csv",
        "phase4b_selected_candidates.csv",
        "phase4b_feature_sets.csv",
        "phase4b_training_history.csv",
        "phase4c_summary.csv",
        "phase4c_horizon_behavior.csv",
    )
    missing = [name for name in required if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Phase 5A 缺少必要產物：{missing}")

    phase3b = metric_map(output_dir / "phase3b_summary.csv")
    if phase3b.get("status") != "PASS" or phase3b.get("ready_for_modeling") not in {"1", "1.0"}:
        raise RuntimeError("Phase 3B 尚未完整通過，不可執行 Phase 5A")

    selected = pd.read_csv(
        output_dir / "phase4b_selected_candidates.csv",
        encoding="utf-8-sig",
        dtype=str,
    ).fillna("")
    selected = selected[selected["market_type"].str.lower() == TARGET_MARKET]
    if len(selected) != 1:
        raise RuntimeError("Phase 4B TPEx 選定候選預期一筆")
    selected_column = (
        "selected_candidate_id"
        if "selected_candidate_id" in selected.columns
        else "candidate_id"
    )
    candidate = str(selected.iloc[0][selected_column])
    if candidate != TARGET_CANDIDATE:
        raise RuntimeError(
            f"Phase 5A 固定候選應為 {TARGET_CANDIDATE}，目前為 {candidate}"
        )

    feature_sets = pd.read_csv(
        output_dir / "phase4b_feature_sets.csv",
        encoding="utf-8-sig",
        dtype=str,
    ).fillna("")
    features = feature_sets[feature_sets["candidate_id"] == TARGET_CANDIDATE].copy()
    features["feature_order"] = pd.to_numeric(features["feature_order"], errors="raise")
    features = features.sort_values("feature_order")
    observed_features = tuple(features["feature"].astype(str))
    if observed_features != tuple(CORE_FEATURE_COLUMNS):
        raise RuntimeError("Phase 5A core22 特徵順序與 Phase 4B 報告不一致")

    phase4c = metric_map(output_dir / "phase4c_summary.csv")
    if phase4c.get("pipeline_status") != "PASS":
        raise RuntimeError("Phase 4C pipeline 尚未通過")
    if phase4c.get("ready_for_selection_index") not in {"1", "1.0"}:
        raise RuntimeError("Phase 4C 尚未通過選股指數驗證")
    if phase4c.get("decision") != "PROCEED_TO_SELECTION_INDEX_BUILD":
        raise RuntimeError("Phase 4C 決策不是建立選股指數")

    manifest = load_phase3_manifest(output_dir / "phase3_dataset_manifest.csv")
    tpex = manifest.get("phase3_training_tpex.csv.gz")
    if tpex is None:
        raise RuntimeError("Phase 3 manifest 缺少 TPEx 訓練檔")
    source_path = output_dir / str(tpex["file_name"])
    source_sha = sha256_file(source_path)
    if source_sha != str(tpex["sha256"]):
        raise RuntimeError("Phase 3 TPEx 訓練檔 SHA 與 manifest 不一致")
    source_rows = int(tpex["row_count"])
    first_training_date, last_training_date = training_signal_date_range(
        source_path, chunk_size=config.chunk_size
    )

    training_history = pd.read_csv(
        output_dir / "phase4b_training_history.csv",
        encoding="utf-8-sig",
    )
    selected_history = training_history[
        (training_history["market_type"].astype(str).str.lower() == TARGET_MARKET)
        & (training_history["candidate_id"].astype(str) == TARGET_CANDIDATE)
    ].copy()
    if selected_history.empty:
        raise RuntimeError("Phase 4B 缺少 TPEx core22 訓練歷程")
    best_epochs = selected_history.loc[
        selected_history.groupby("test_year")["calibration_log_loss"].idxmin(),
        "epoch",
    ]
    recommended_epochs = int(round(float(best_epochs.median())))
    if config.training_epochs != recommended_epochs:
        raise RuntimeError(
            "Phase 5A 固定 epoch 必須沿用 Phase 4B 各年度最佳 epoch 中位數："
            f"預期 {recommended_epochs}，目前 {config.training_epochs}"
        )

    return {
        "phase3_config_signature": str(tpex["config_signature"]),
        "phase3_source_sha256": source_sha,
        "phase3_source_rows": source_rows,
        "first_training_date": first_training_date,
        "last_training_date": last_training_date,
        "phase4c_last_signal_date": phase4c.get("last_signal_date", ""),
        "phase4c_confirmation_10d_spread": phase4c.get(
            "confirmation_10d_top20_minus_bottom20", ""
        ),
        "phase4c_confirmation_ex_latest_10d_spread": phase4c.get(
            "confirmation_ex_latest_10d_top20_minus_bottom20", ""
        ),
        "recommended_training_epochs": recommended_epochs,
    }


def load_phase3_manifest(path: Path) -> dict[str, dict[str, str]]:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    return {
        str(row["file_name"]): {str(key): str(value) for key, value in row.items()}
        for _, row in frame.iterrows()
    }


def training_signal_date_range(
    source_path: Path,
    *,
    chunk_size: int,
) -> tuple[str, str]:
    first_date = ""
    last_date = ""
    for chunk in pd.read_csv(
        source_path,
        compression="gzip",
        usecols=["signal_date"],
        dtype={"signal_date": str},
        chunksize=chunk_size,
    ):
        dates = chunk["signal_date"].dropna().astype(str)
        if dates.empty:
            continue
        chunk_first = str(dates.min())
        chunk_last = str(dates.max())
        first_date = chunk_first if not first_date else min(first_date, chunk_first)
        last_date = chunk_last if not last_date else max(last_date, chunk_last)
    if not first_date or not last_date:
        raise RuntimeError("Phase 3 TPEx 訓練檔缺少 signal_date")
    return first_date, last_date


def build_or_load_final_model(
    *,
    cache: MarketCache,
    source_sha256: str,
    first_training_date: str,
    last_training_date: str,
    model_root: Path,
    settings: Phase5ASettings,
    force: bool,
) -> FinalModelBundle:
    signature = final_model_signature(source_sha256=source_sha256, settings=settings)
    model_dir = model_root / signature[:16]
    manifest_path = model_dir / "phase5a_model_manifest.json"
    arrays_path = model_dir / "phase5a_model_arrays.npz"
    if manifest_path.exists() and arrays_path.exists() and not force:
        bundle = load_final_model(manifest_path=manifest_path, arrays_path=arrays_path)
        if bundle.signature == signature:
            print(f"Phase 5A：沿用既有最終模型 {signature[:16]}")
            return bundle

    print(
        "Phase 5A：以全部成熟 TPEx 標籤訓練固定 core22 模型，"
        f"共 {cache.row_count:,} 筆"
    )
    preprocessor = fit_all_data_preprocessor(cache=cache, settings=settings)
    model, priors, history = train_all_data_model(
        cache=cache,
        preprocessor=preprocessor,
        settings=settings,
    )
    years = np.asarray(cache.years, dtype=np.int64)
    bundle = FinalModelBundle(
        signature=signature,
        feature_columns=tuple(CORE_FEATURE_COLUMNS),
        preprocessor=preprocessor,
        model=model,
        priors=priors,
        training_rows=int(cache.row_count),
        first_training_year=int(years.min()),
        last_training_year=int(years.max()),
        first_training_date=first_training_date,
        last_training_date=last_training_date,
        source_sha256=source_sha256,
        training_epochs=settings.training_epochs,
    )
    save_final_model(
        bundle=bundle,
        history=history,
        model_dir=model_dir,
        settings=settings,
    )
    return bundle


def fit_all_data_preprocessor(
    *,
    cache: MarketCache,
    settings: Phase5ASettings,
) -> Preprocessor:
    feature_indices = np.asarray(
        [FEATURE_COLUMNS.index(name) for name in CORE_FEATURE_COLUMNS],
        dtype=np.int64,
    )
    row_count = len(cache.years)
    rng = np.random.default_rng(settings.random_seed)
    sample_count = min(settings.quantile_sample_size, row_count)
    positions = np.sort(rng.choice(row_count, size=sample_count, replace=False))
    sample = np.asarray(cache.features[positions], dtype=np.float64)[:, feature_indices]
    lower = np.quantile(sample, 0.005, axis=0)
    upper = np.quantile(sample, 0.995, axis=0)
    upper = np.maximum(upper, lower)

    total_sum = np.zeros(len(feature_indices), dtype=np.float64)
    total_sumsq = np.zeros(len(feature_indices), dtype=np.float64)
    count = 0
    for start in range(0, row_count, settings.chunk_size):
        end = min(row_count, start + settings.chunk_size)
        values = np.asarray(cache.features[start:end], dtype=np.float64)[:, feature_indices]
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


def train_all_data_model(
    *,
    cache: MarketCache,
    preprocessor: Preprocessor,
    settings: Phase5ASettings,
) -> tuple[SoftmaxModel, np.ndarray, list[dict[str, Any]]]:
    labels = np.asarray(cache.labels, dtype=np.int64)
    features = np.asarray(cache.features)
    feature_indices = np.asarray(
        [FEATURE_COLUMNS.index(name) for name in CORE_FEATURE_COLUMNS],
        dtype=np.int64,
    )
    feature_count = len(CORE_FEATURE_COLUMNS)
    counts = np.bincount(labels, minlength=len(LABELS)).astype(np.float64)
    priors = counts / counts.sum()
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
    updates = 0
    rng = np.random.default_rng(settings.random_seed + 50_000)
    ranges = [
        (start, min(len(labels), start + settings.chunk_size))
        for start in range(0, len(labels), settings.chunk_size)
    ]
    history: list[dict[str, Any]] = []

    for epoch in range(1, settings.training_epochs + 1):
        rng.shuffle(ranges)
        loss_total = 0.0
        examples = 0
        batches = 0
        for start, end in ranges:
            x = np.asarray(features[start:end], dtype=np.float64)[:, feature_indices]
            x = preprocessor.transform(x)
            y = labels[start:end]
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
        training_loss = loss_total / max(examples, 1)
        history.append(
            {
                "epoch": epoch,
                "training_log_loss": float(training_loss),
                "batches": batches,
                "examples": examples,
            }
        )
        print(f"  Phase 5A epoch {epoch}: train={training_loss:.6f}")
    return SoftmaxModel(weights=weights, intercept=intercept), priors, history


def load_latest_tpex_cross_section(
    *,
    output_dir: Path,
    shard_dir: Path,
    feature_columns: tuple[str, ...],
    settings: Phase5ASettings,
    requested_signal_date: str | None,
) -> tuple[pd.DataFrame, str, list[str]]:
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
    required = latest_source_columns(feature_columns)
    frames: list[pd.DataFrame] = []

    for position, stock_id in enumerate(stock_ids, start=1):
        path = shard_dir / f"{stock_id}.csv.gz"
        if not path.exists():
            raise FileNotFoundError(f"Phase 3 分片遺失：{path}")
        recent: pd.DataFrame | None = None
        for chunk in pd.read_csv(
            path,
            compression="gzip",
            usecols=required,
            chunksize=settings.chunk_size,
            dtype={
                "stock_id": str,
                "stock_name": str,
                "market_type": str,
                "signal_date": str,
                "feature_status": str,
            },
            low_memory=False,
        ):
            chunk = chunk[
                (chunk["market_type"].astype(str).str.lower() == TARGET_MARKET)
                & (chunk["feature_status"].astype(str) == "ok")
            ].copy()
            if requested_signal_date:
                chunk = chunk[chunk["signal_date"].astype(str) == requested_signal_date]
            if chunk.empty:
                continue
            recent = (
                chunk
                if recent is None
                else pd.concat([recent, chunk], ignore_index=True)
            )
            if not requested_signal_date and len(recent) > settings.recent_rows_per_stock:
                recent = recent.tail(settings.recent_rows_per_stock).copy()
        if recent is not None and not recent.empty:
            frames.append(recent)
        if position % 100 == 0 or position == len(stock_ids):
            print(f"  Phase 5A 最新分片讀取進度：{position}/{len(stock_ids)}")

    if not frames:
        target = requested_signal_date or "最新日期"
        raise RuntimeError(f"Phase 3 分片找不到 {target} 的 TPEx 特徵資料")
    recent_rows = pd.concat(frames, ignore_index=True)
    recent_rows["signal_date"] = recent_rows["signal_date"].astype(str)
    recent_rows = add_live_liquidity_flags(recent_rows)
    counts = (
        recent_rows[recent_rows["liquidity_pass_20m"] == 1]
        .groupby("signal_date")["stock_id"]
        .nunique()
        .sort_index()
    )
    eligible_dates = counts[counts >= settings.minimum_daily_stocks]
    if eligible_dates.empty:
        raise RuntimeError(
            "Phase 5A 找不到符合最低同日股票數的 2,000 萬流動性訊號日"
        )
    if requested_signal_date:
        if requested_signal_date not in eligible_dates.index:
            observed = int(counts.get(requested_signal_date, 0))
            raise RuntimeError(
                f"指定訊號日 {requested_signal_date} 只有 {observed} 檔符合 2,000 萬門檻，"
                f"低於最低 {settings.minimum_daily_stocks} 檔"
            )
        resolved_date = requested_signal_date
    else:
        resolved_date = str(eligible_dates.index.max())

    selected = recent_rows[
        (recent_rows["signal_date"] == resolved_date)
        & (recent_rows["liquidity_pass_20m"] == 1)
    ].copy()
    duplicates = int(selected.duplicated(["stock_id", "signal_date"]).sum())
    if duplicates:
        raise RuntimeError(f"Phase 5A 最新訊號截面有 {duplicates} 筆重複股票")
    return selected, resolved_date, [str(value) for value in eligible_dates.index]


def add_live_liquidity_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the TPEx screen using only information known at the signal close."""
    result = frame.copy()
    history_days = pd.to_numeric(
        result["history_market_days_20"], errors="coerce"
    ).fillna(0)
    normal_days = pd.to_numeric(
        result["normal_trading_days_20d"], errors="coerce"
    ).fillna(0)
    median_money = pd.to_numeric(
        result["median_trading_money_20d"], errors="coerce"
    ).fillna(0)
    zero_streak = pd.to_numeric(
        result["max_zero_volume_streak_20d"], errors="coerce"
    ).fillna(99)
    common = (history_days >= 20) & (normal_days >= 18) & (zero_streak < 3)
    for threshold in (20, 50, 100):
        result[f"liquidity_pass_{threshold}m"] = (
            common & (median_money >= threshold * 1_000_000)
        ).astype(np.uint8)
    entry_available = pd.to_numeric(
        result["entry_price_available"], errors="coerce"
    ).fillna(0).astype(np.uint8)
    result["entry_price_available"] = entry_available
    result["entry_feasibility_status"] = np.where(
        entry_available == 1,
        "已確認T+1有有效開盤價",
        "待T+1確認或無有效開盤價",
    )
    return result


def score_latest_cross_section(
    frame: pd.DataFrame,
    *,
    bundle: FinalModelBundle,
) -> pd.DataFrame:
    features = frame[list(bundle.feature_columns)].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(features).all():
        raise RuntimeError("Phase 5A 最新截面含無效模型特徵")
    transformed = bundle.preprocessor.transform(features)
    probabilities = softmax(bundle.model.logits(transformed))
    scored = frame[latest_output_source_columns()].copy()
    scored["p_down_raw"] = probabilities[:, LABEL_TO_INDEX["DOWN"]]
    scored["p_flat_raw"] = probabilities[:, LABEL_TO_INDEX["FLAT"]]
    scored["p_up_raw"] = probabilities[:, LABEL_TO_INDEX["UP"]]
    scored["institutional_index_raw"] = 100.0 * (
        scored["p_up_raw"] - scored["p_down_raw"]
    )
    probability_deviation = np.abs(probabilities.sum(axis=1) - 1.0).max()
    if probability_deviation > 1e-6:
        raise RuntimeError("Phase 5A 最新截面機率加總檢查失敗")
    return scored


def rank_liquidity_universes(
    scored: pd.DataFrame,
    *,
    minimum_daily_stocks: int,
) -> pd.DataFrame:
    base, dropped = assign_same_day_ranks(
        scored,
        minimum_daily_stocks=minimum_daily_stocks,
    )
    if dropped or base.empty:
        raise RuntimeError("Phase 5A 最新截面無法建立 2,000 萬同日排名")
    result = base.rename(
        columns={
            "daily_rank": "rank_20m",
            "daily_stock_count": "stock_count_20m",
            "daily_percentile": "percentile_20m",
            "daily_decile": "decile_20m",
            "daily_quintile": "quintile_20m",
        }
    )
    for threshold in (50, 100):
        liquidity = pd.to_numeric(
            result[f"liquidity_pass_{threshold}m"], errors="coerce"
        ).fillna(0)
        subset = result[liquidity == 1][
            ["stock_id", "signal_date", "institutional_index_raw"]
        ].copy()
        if subset.empty:
            continue
        subset, _ = assign_same_day_ranks(subset, minimum_daily_stocks=1)
        subset = subset.rename(
            columns={
                "daily_rank": f"rank_{threshold}m",
                "daily_stock_count": f"stock_count_{threshold}m",
                "daily_percentile": f"percentile_{threshold}m",
                "daily_decile": f"decile_{threshold}m",
                "daily_quintile": f"quintile_{threshold}m",
            }
        )
        result = result.merge(
            subset[
                [
                    "stock_id",
                    "signal_date",
                    f"rank_{threshold}m",
                    f"stock_count_{threshold}m",
                    f"percentile_{threshold}m",
                    f"decile_{threshold}m",
                    f"quintile_{threshold}m",
                ]
            ],
            on=["stock_id", "signal_date"],
            how="left",
            validate="one_to_one",
        )
    for threshold in (50, 100):
        for prefix in ("rank", "stock_count", "decile", "quintile"):
            column = f"{prefix}_{threshold}m"
            if column not in result:
                result[column] = np.nan
        percentile_column = f"percentile_{threshold}m"
        if percentile_column not in result:
            result[percentile_column] = np.nan
    result["daily_rank"] = result["rank_20m"]
    result["daily_percentile"] = result["percentile_20m"]
    result["daily_decile"] = result["decile_20m"]
    result["daily_quintile"] = result["quintile_20m"]
    result["liquidity_tier"] = np.select(
        [
            pd.to_numeric(result["liquidity_pass_100m"], errors="coerce").fillna(0) == 1,
            pd.to_numeric(result["liquidity_pass_50m"], errors="coerce").fillna(0) == 1,
        ],
        ["1億元以上", "5,000萬元以上"],
        default="2,000萬元以上",
    )
    return result


def build_historical_behavior_lookup(output_dir: Path) -> pd.DataFrame:
    behavior = pd.read_csv(
        output_dir / "phase4c_horizon_behavior.csv",
        encoding="utf-8-sig",
    )
    selected = behavior[
        (behavior["period"] == "confirmation_ex_latest_year")
        & (behavior["group_scheme"] == "decile")
    ].copy()
    if selected.empty:
        raise RuntimeError("Phase 4C 缺少排除最新年度的十分位歷史行為")
    required_horizons = set(HORIZONS)
    observed_horizons = set(
        pd.to_numeric(selected["horizon_days"], errors="raise").astype(int)
    )
    if observed_horizons != required_horizons:
        raise RuntimeError("Phase 4C 歷史行為缺少 5／10／20 日資料")
    selected["horizon_days"] = pd.to_numeric(selected["horizon_days"], errors="raise").astype(int)
    selected["group_value"] = pd.to_numeric(selected["group_value"], errors="raise").astype(int)
    keep = [
        "horizon_days",
        "group_value",
        "sample_count",
        "signal_dates",
        "unique_stocks",
        "equal_day_average_return",
        "average_return",
        "median_return",
        "return_p10",
        "return_p25",
        "return_p75",
        "return_p90",
        "positive_return_rate",
        "up_5pct_rate",
        "down_5pct_rate",
        "average_max_adjusted_return",
        "average_min_adjusted_return",
        "label_up_rate",
        "label_flat_rate",
        "label_down_rate",
    ]
    return selected[keep].sort_values(["horizon_days", "group_value"]).reset_index(drop=True)


def attach_historical_behavior(
    selection: pd.DataFrame,
    *,
    lookup: pd.DataFrame,
) -> pd.DataFrame:
    result = selection.copy()
    for horizon in HORIZONS:
        horizon_lookup = lookup[lookup["horizon_days"] == horizon].copy()
        rename = {
            column: f"history_{horizon}d_{column}"
            for column in horizon_lookup.columns
            if column not in {"horizon_days", "group_value"}
        }
        horizon_lookup = horizon_lookup.rename(columns=rename)
        result = result.merge(
            horizon_lookup.drop(columns=["horizon_days"]),
            left_on="decile_20m",
            right_on="group_value",
            how="left",
            validate="many_to_one",
        ).drop(columns=["group_value"])
    return result


def finalize_selection_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["institutional_selection_index"] = result["percentile_20m"]
    result["selection_band"] = np.select(
        [
            result["percentile_20m"] > 90,
            result["percentile_20m"] > 80,
            result["percentile_20m"] <= 10,
            result["percentile_20m"] <= 20,
        ],
        ["前10%", "前20%", "後10%", "後20%"],
        default="中間60%",
    )
    result["index_interpretation"] = (
        "同日TPEx法人行為相對排名；不是上漲機率或買進指令"
    )
    result["data_status"] = "OK"
    result = result.sort_values(
        ["rank_20m", "stock_id"],
        ascending=[True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return result[selection_output_columns()]


def build_phase5a_summary(
    *,
    validation: dict[str, Any],
    bundle: FinalModelBundle,
    selection: pd.DataFrame,
    resolved_date: str,
    available_dates: list[str],
    settings: Phase5ASettings,
) -> pd.DataFrame:
    today = date.today()
    signal = date.fromisoformat(resolved_date)
    metrics: list[tuple[str, Any]] = [
        ("phase5a_version", PHASE5A_VERSION),
        ("pipeline_status", "PASS"),
        ("target_market", TARGET_MARKET),
        ("selected_candidate", TARGET_CANDIDATE),
        ("index_probability_variant", "raw"),
        ("index_interpretation", "same_day_relative_ranking_not_probability"),
        ("signal_date", resolved_date),
        ("available_eligible_dates", len(available_dates)),
        ("selection_rows", len(selection)),
        ("minimum_daily_stocks", settings.minimum_daily_stocks),
        ("calendar_age_days", (today - signal).days),
        ("training_rows", bundle.training_rows),
        ("first_training_year", bundle.first_training_year),
        ("last_training_year", bundle.last_training_year),
        ("first_training_date", bundle.first_training_date),
        ("last_training_date", bundle.last_training_date),
        ("training_epochs", bundle.training_epochs),
        ("model_signature", bundle.signature),
        ("phase3_source_sha256", bundle.source_sha256),
        ("phase3_config_signature", validation["phase3_config_signature"]),
        ("phase4c_last_signal_date", validation["phase4c_last_signal_date"]),
        (
            "phase4c_confirmation_10d_top20_minus_bottom20",
            validation["phase4c_confirmation_10d_spread"],
        ),
        (
            "phase4c_confirmation_ex_latest_10d_top20_minus_bottom20",
            validation["phase4c_confirmation_ex_latest_10d_spread"],
        ),
        ("duplicate_stock_count", int(selection["stock_id"].duplicated().sum())),
        ("rank_duplicate_count", int(selection["rank_20m"].duplicated().sum())),
        (
            "probability_sum_max_deviation",
            float(
                np.max(
                    np.abs(
                        selection[["p_down_raw", "p_flat_raw", "p_up_raw"]]
                        .to_numpy(dtype=float)
                        .sum(axis=1)
                        - 1.0
                    )
                )
            ),
        ),
        ("top10_count", int((selection["percentile_20m"] > 90).sum())),
        ("top20_count", int((selection["percentile_20m"] > 80).sum())),
        ("liquidity_50m_count", int((selection["liquidity_pass_50m"] == 1).sum())),
        ("liquidity_100m_count", int((selection["liquidity_pass_100m"] == 1).sum())),
        ("ready_for_local_reference", 1),
        ("deployment_status", "LOCAL_RESEARCH_ONLY"),
    ]
    return pd.DataFrame([{"metric": key, "value": value} for key, value in metrics])


def export_model_reports(
    *,
    bundle: FinalModelBundle,
    settings: Phase5ASettings,
    model_root: Path,
) -> dict[str, pd.DataFrame]:
    coefficient_rows: list[dict[str, Any]] = []
    for class_index, class_label in enumerate(LABELS):
        coefficient_rows.append(
            {
                "class_label": class_label,
                "feature": "(intercept)",
                "standardized_coefficient": float(bundle.model.intercept[class_index]),
            }
        )
        for feature_index, feature in enumerate(bundle.feature_columns):
            coefficient_rows.append(
                {
                    "class_label": class_label,
                    "feature": feature,
                    "standardized_coefficient": float(
                        bundle.model.weights[feature_index, class_index]
                    ),
                }
            )
    preprocessing = pd.DataFrame(
        [
            {
                "feature_order": index,
                "feature": feature,
                "clip_lower": float(bundle.preprocessor.lower[index - 1]),
                "clip_upper": float(bundle.preprocessor.upper[index - 1]),
                "training_mean": float(bundle.preprocessor.mean[index - 1]),
                "training_std": float(bundle.preprocessor.std[index - 1]),
            }
            for index, feature in enumerate(bundle.feature_columns, start=1)
        ]
    )
    manifest = pd.DataFrame(
        [
            {"metric": "phase5a_version", "value": PHASE5A_VERSION},
            {"metric": "model_signature", "value": bundle.signature},
            {"metric": "candidate_id", "value": TARGET_CANDIDATE},
            {"metric": "feature_count", "value": len(bundle.feature_columns)},
            {"metric": "training_rows", "value": bundle.training_rows},
            {"metric": "first_training_year", "value": bundle.first_training_year},
            {"metric": "last_training_year", "value": bundle.last_training_year},
            {"metric": "first_training_date", "value": bundle.first_training_date},
            {"metric": "last_training_date", "value": bundle.last_training_date},
            {"metric": "training_epochs", "value": bundle.training_epochs},
            {"metric": "l2_penalty", "value": settings.l2_penalty},
            {"metric": "learning_rate", "value": settings.learning_rate},
            {"metric": "source_sha256", "value": bundle.source_sha256},
            {"metric": "p_down_prior", "value": float(bundle.priors[LABEL_TO_INDEX["DOWN"]])},
            {"metric": "p_flat_prior", "value": float(bundle.priors[LABEL_TO_INDEX["FLAT"]])},
            {"metric": "p_up_prior", "value": float(bundle.priors[LABEL_TO_INDEX["UP"]])},
        ]
    )
    history_path = model_root / bundle.signature[:16] / "phase5a_training_history.csv"
    if not history_path.exists():
        raise FileNotFoundError(f"Phase 5A 訓練歷程遺失：{history_path}")
    history = pd.read_csv(history_path, encoding="utf-8-sig")
    return {
        "phase5a_model_coefficients.csv": pd.DataFrame(coefficient_rows),
        "phase5a_model_preprocessing.csv": preprocessing,
        "phase5a_model_manifest.csv": manifest,
        "phase5a_training_history.csv": history,
    }


def export_phase5a_reports(
    *,
    output_dir: Path,
    selection: pd.DataFrame,
    lookup: pd.DataFrame,
    summary: pd.DataFrame,
    model_reports: dict[str, pd.DataFrame],
) -> list[Path]:
    paths = [
        write_csv(output_dir / "phase5a_selection_index.csv", selection),
        write_csv(output_dir / "phase5a_selection_index_top20.csv", selection.head(20)),
        write_csv(output_dir / "phase5a_historical_behavior_lookup.csv", lookup),
        write_csv(output_dir / "phase5a_summary.csv", summary),
    ]
    for name, frame in model_reports.items():
        paths.append(write_csv(output_dir / name, frame))
    archive = output_dir / "phase5a_selection_index_reports.zip"
    temporary = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in paths:
            handle.write(path, arcname=path.name)
    os.replace(temporary, archive)
    paths.append(archive)
    return paths


def save_final_model(
    *,
    bundle: FinalModelBundle,
    history: list[dict[str, Any]],
    model_dir: Path,
    settings: Phase5ASettings,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = model_dir / "phase5a_model_arrays.npz"
    temporary_arrays = arrays_path.with_suffix(".npz.tmp")
    with temporary_arrays.open("wb") as handle:
        np.savez_compressed(
            handle,
            weights=bundle.model.weights,
            intercept=bundle.model.intercept,
            lower=bundle.preprocessor.lower,
            upper=bundle.preprocessor.upper,
            mean=bundle.preprocessor.mean,
            std=bundle.preprocessor.std,
            priors=bundle.priors,
        )
    os.replace(temporary_arrays, arrays_path)
    manifest = {
        "phase5a_version": PHASE5A_VERSION,
        "signature": bundle.signature,
        "feature_columns": list(bundle.feature_columns),
        "training_rows": bundle.training_rows,
        "first_training_year": bundle.first_training_year,
        "last_training_year": bundle.last_training_year,
        "first_training_date": bundle.first_training_date,
        "last_training_date": bundle.last_training_date,
        "source_sha256": bundle.source_sha256,
        "training_epochs": bundle.training_epochs,
        "preprocessor_sampled_rows": bundle.preprocessor.sampled_rows,
        "settings": asdict(settings),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json_atomic(model_dir / "phase5a_model_manifest.json", manifest)
    write_csv(model_dir / "phase5a_training_history.csv", pd.DataFrame(history))


def load_final_model(*, manifest_path: Path, arrays_path: Path) -> FinalModelBundle:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(arrays_path) as values:
        preprocessor = Preprocessor(
            lower=np.asarray(values["lower"], dtype=np.float64),
            upper=np.asarray(values["upper"], dtype=np.float64),
            mean=np.asarray(values["mean"], dtype=np.float64),
            std=np.asarray(values["std"], dtype=np.float64),
            sampled_rows=int(manifest["preprocessor_sampled_rows"]),
            training_rows=int(manifest["training_rows"]),
        )
        model = SoftmaxModel(
            weights=np.asarray(values["weights"], dtype=np.float64),
            intercept=np.asarray(values["intercept"], dtype=np.float64),
        )
        priors = np.asarray(values["priors"], dtype=np.float64)
    return FinalModelBundle(
        signature=str(manifest["signature"]),
        feature_columns=tuple(str(value) for value in manifest["feature_columns"]),
        preprocessor=preprocessor,
        model=model,
        priors=priors,
        training_rows=int(manifest["training_rows"]),
        first_training_year=int(manifest["first_training_year"]),
        last_training_year=int(manifest["last_training_year"]),
        first_training_date=str(manifest["first_training_date"]),
        last_training_date=str(manifest["last_training_date"]),
        source_sha256=str(manifest["source_sha256"]),
        training_epochs=int(manifest["training_epochs"]),
    )


def final_model_signature(*, source_sha256: str, settings: Phase5ASettings) -> str:
    payload = {
        "phase5a_version": PHASE5A_VERSION,
        "candidate_id": TARGET_CANDIDATE,
        "feature_columns": list(CORE_FEATURE_COLUMNS),
        "source_sha256": source_sha256,
        "training_epochs": settings.training_epochs,
        "learning_rate": settings.learning_rate,
        "l2_penalty": settings.l2_penalty,
        "quantile_sample_size": settings.quantile_sample_size,
        "chunk_size": settings.chunk_size,
        "batch_size": settings.batch_size,
        "random_seed": settings.random_seed,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def latest_source_columns(feature_columns: tuple[str, ...]) -> list[str]:
    return [*latest_input_source_columns(), *feature_columns]


def latest_input_source_columns() -> list[str]:
    return [
        column
        for column in latest_output_source_columns()
        if column != "entry_feasibility_status"
    ]


def latest_output_source_columns() -> list[str]:
    return [
        "stock_id",
        "stock_name",
        "market_type",
        "signal_date",
        "signal_year",
        "feature_status",
        "history_market_days_20",
        "history_valid_price_days_20",
        "history_flow_rows_20",
        "median_trading_money_20d",
        "max_zero_volume_streak_20d",
        "normal_trading_days_20d",
        "entry_price_available",
        "entry_feasibility_status",
        "liquidity_pass_20m",
        "liquidity_pass_50m",
        "liquidity_pass_100m",
    ]


def selection_output_columns() -> list[str]:
    columns = [
        "rank_20m",
        "stock_id",
        "stock_name",
        "signal_date",
        "institutional_selection_index",
        "institutional_index_raw",
        "percentile_20m",
        "decile_20m",
        "quintile_20m",
        "selection_band",
        "liquidity_tier",
        "rank_50m",
        "stock_count_50m",
        "percentile_50m",
        "rank_100m",
        "stock_count_100m",
        "percentile_100m",
        "stock_count_20m",
        "median_trading_money_20d",
        "normal_trading_days_20d",
        "max_zero_volume_streak_20d",
        "entry_price_available",
        "entry_feasibility_status",
        "p_down_raw",
        "p_flat_raw",
        "p_up_raw",
    ]
    for horizon in HORIZONS:
        columns.extend(
            [
                f"history_{horizon}d_equal_day_average_return",
                f"history_{horizon}d_average_return",
                f"history_{horizon}d_median_return",
                f"history_{horizon}d_return_p10",
                f"history_{horizon}d_return_p25",
                f"history_{horizon}d_return_p75",
                f"history_{horizon}d_return_p90",
                f"history_{horizon}d_positive_return_rate",
                f"history_{horizon}d_up_5pct_rate",
                f"history_{horizon}d_down_5pct_rate",
                f"history_{horizon}d_average_max_adjusted_return",
                f"history_{horizon}d_average_min_adjusted_return",
                f"history_{horizon}d_sample_count",
                f"history_{horizon}d_signal_dates",
                f"history_{horizon}d_unique_stocks",
            ]
        )
    columns.extend(
        [
            "history_10d_label_up_rate",
            "history_10d_label_flat_rate",
            "history_10d_label_down_rate",
            "history_market_days_20",
            "history_valid_price_days_20",
            "history_flow_rows_20",
            "liquidity_pass_20m",
            "liquidity_pass_50m",
            "liquidity_pass_100m",
            "data_status",
            "index_interpretation",
        ]
    )
    return columns


def metric_map(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    return {str(row["metric"]): str(row["value"]) for _, row in frame.iterrows()}


def write_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)
