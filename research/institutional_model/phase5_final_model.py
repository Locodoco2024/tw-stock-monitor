from __future__ import annotations

import hashlib
import json
import math
import os
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from research.institutional_model.phase3_dataset import sha256_file
from research.institutional_model.phase4_model import Preprocessor
from research.institutional_model.phase4_selection import (
    moving_block_bootstrap_mean_ci,
    resolve_phase3_shard_directory,
)
from research.institutional_model.market_model_spec import market_model_spec
from research.institutional_model.phase4_target import LinearRankModel, feature_group


PHASE5D_VERSION = "phase5d-v1"
TARGET_SCORE_COLUMN = "return_rank_score_daily_percentile"
PRIMARY_HORIZON_DAYS = 20
EXTENSION_HORIZON_DAYS = 40
DEFAULT_ENTRY_THRESHOLD = 90.0
DEFAULT_HIGH_INTENSITY_THRESHOLD = 95.0
DEFAULT_CONFIRMATION_THRESHOLD = 80.0
DEFAULT_CONFIRMATION_DAYS = 5
DEFAULT_PRIMARY_TRACKING_DAYS = 20
DEFAULT_MAXIMUM_TRACKING_DAYS = 40
DEFAULT_COOLDOWN_DAYS = 20

TRAINING_BASE_COLUMNS = (
    "stock_id",
    "stock_name",
    "market_type",
    "signal_date",
    "feature_status",
    "liquidity_pass_20m",
    "label_status_20d",
    "adjusted_return_20d",
)

OOS_REQUIRED_COLUMNS = (
    "stock_id",
    "stock_name",
    "signal_date",
    "test_year",
    "adjusted_return_20d",
    "adjusted_return_40d",
    TARGET_SCORE_COLUMN,
)

FEATURE_LABELS = {
    "foreign_flow_pct_1d": "外資單日流量",
    "foreign_flow_pct_3d": "外資近3日流量",
    "foreign_flow_pct_5d": "外資近5日流量",
    "foreign_flow_pct_10d": "外資近10日流量",
    "foreign_flow_pct_20d": "外資近20日流量",
    "foreign_buy_day_ratio_5d": "外資近5日買超日比例",
    "foreign_buy_day_ratio_10d": "外資近10日買超日比例",
    "foreign_buy_day_ratio_20d": "外資近20日買超日比例",
    "foreign_streak": "外資連續買賣超",
    "investment_trust_flow_pct_1d": "投信單日流量",
    "investment_trust_flow_pct_3d": "投信近3日流量",
    "investment_trust_flow_pct_5d": "投信近5日流量",
    "investment_trust_flow_pct_10d": "投信近10日流量",
    "investment_trust_flow_pct_20d": "投信近20日流量",
    "investment_trust_buy_day_ratio_5d": "投信近5日買超日比例",
    "investment_trust_buy_day_ratio_10d": "投信近10日買超日比例",
    "investment_trust_buy_day_ratio_20d": "投信近20日買超日比例",
    "investment_trust_streak": "投信連續買賣超",
    "dealer_self_flow_pct_1d": "自營商自行買賣單日流量",
    "dealer_self_flow_pct_3d": "自營商自行買賣近3日流量",
    "dealer_self_flow_pct_5d": "自營商自行買賣近5日流量",
    "dealer_self_flow_pct_10d": "自營商自行買賣近10日流量",
    "dealer_self_flow_pct_20d": "自營商自行買賣近20日流量",
    "dealer_self_buy_day_ratio_5d": "自營商自行買賣近5日買超日比例",
    "dealer_self_buy_day_ratio_10d": "自營商自行買賣近10日買超日比例",
    "dealer_self_buy_day_ratio_20d": "自營商自行買賣近20日買超日比例",
    "dealer_self_streak": "自營商自行買賣連續買賣超",
    "selected_total_flow_pct_1d": "三法人合計單日流量",
    "selected_total_flow_pct_3d": "三法人合計近3日流量",
    "selected_total_flow_pct_5d": "三法人合計近5日流量",
    "selected_total_flow_pct_10d": "三法人合計近10日流量",
    "selected_total_flow_pct_20d": "三法人合計近20日流量",
    "selected_total_buy_day_ratio_5d": "三法人合計近5日買超日比例",
    "selected_total_buy_day_ratio_10d": "三法人合計近10日買超日比例",
    "selected_total_buy_day_ratio_20d": "三法人合計近20日買超日比例",
    "selected_total_streak": "三法人合計連續買賣超",
    "institutional_agreement_1d": "三法人單日方向一致性",
    "institutional_agreement_5d": "三法人近5日方向一致性",
    "institutional_agreement_20d": "三法人近20日方向一致性",
    "selected_total_acceleration_5d_vs_20d": "三法人合計近5日相對20日加速度",
}


@dataclass(frozen=True)
class Phase5DSettings:
    target_market: str = "tpex"
    minimum_daily_stocks: int = 50
    clip_lower_quantile: float = 0.005
    clip_upper_quantile: float = 0.995
    quantile_sample_size: int = 250_000
    ranking_l2_penalty: float = 0.001
    entry_threshold: float = DEFAULT_ENTRY_THRESHOLD
    high_intensity_threshold: float = DEFAULT_HIGH_INTENSITY_THRESHOLD
    confirmation_threshold: float = DEFAULT_CONFIRMATION_THRESHOLD
    confirmation_days: int = DEFAULT_CONFIRMATION_DAYS
    primary_tracking_days: int = DEFAULT_PRIMARY_TRACKING_DAYS
    maximum_tracking_days: int = DEFAULT_MAXIMUM_TRACKING_DAYS
    cooldown_days: int = DEFAULT_COOLDOWN_DAYS
    bootstrap_iterations: int = 1_000
    bootstrap_block_months: int = 3
    replay_latest_year: bool = True
    random_seed: int = 20260729

    def validate(self) -> None:
        if self.target_market not in {"twse", "tpex"}:
            raise ValueError("Phase 5D target_market 只支援 twse 或 tpex")
        if self.minimum_daily_stocks < 20:
            raise ValueError("Phase 5D 每日最低股票數不可低於 20")
        if not 0 <= self.clip_lower_quantile < self.clip_upper_quantile <= 1:
            raise ValueError("Phase 5D 截尾分位數設定無效")
        if self.quantile_sample_size <= 0:
            raise ValueError("Phase 5D 分位數抽樣筆數必須大於 0")
        if self.ranking_l2_penalty < 0:
            raise ValueError("Phase 5D ranking L2 不可小於 0")
        if not 0 < self.confirmation_threshold < self.entry_threshold <= 100:
            raise ValueError("Phase 5D 確認與進榜門檻設定無效")
        if not self.entry_threshold <= self.high_intensity_threshold <= 100:
            raise ValueError("Phase 5D 高強度門檻不可低於進榜門檻")
        if self.confirmation_days < 2:
            raise ValueError("Phase 5D 連續確認日至少為 2")
        if self.primary_tracking_days < 1:
            raise ValueError("Phase 5D 主要追蹤日數必須大於 0")
        if self.maximum_tracking_days <= self.primary_tracking_days:
            raise ValueError("Phase 5D 最長追蹤日數必須大於主要追蹤日數")
        if self.cooldown_days < 0:
            raise ValueError("Phase 5D 冷卻日數不可小於 0")
        if self.bootstrap_iterations < 200:
            raise ValueError("Phase 5D bootstrap 次數不可低於 200")
        if self.bootstrap_block_months < 1:
            raise ValueError("Phase 5D bootstrap 區塊月份必須大於 0")


@dataclass(frozen=True)
class FinalRankModelBundle:
    signature: str
    feature_columns: tuple[str, ...]
    preprocessor: Preprocessor
    model: LinearRankModel
    training_rows: int
    signal_dates: int
    first_training_date: str
    last_training_date: str
    source_sha256: str
    target_mean: float
    target_std: float


@dataclass(frozen=True)
class Phase5DResult:
    status: str
    training_rows: int
    replay_events: int
    replay_notifications: int
    output_paths: tuple[Path, ...]


def run_phase5d_final_model(
    *,
    output_dir: Path | str,
    shard_root: Path | str,
    model_root: Path | str,
    settings: Phase5DSettings | None = None,
    force: bool = False,
) -> Phase5DResult:
    config = settings or Phase5DSettings()
    config.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    validation = validate_phase5d_inputs(output, target_market=config.target_market)
    model_base = Path(model_root)
    model_base.mkdir(parents=True, exist_ok=True)
    shard_dir = resolve_phase3_shard_directory(
        output_dir=output,
        shard_root=Path(shard_root),
    )
    training_frame, training_meta = build_or_load_rank_training_frame(
        shard_dir=shard_dir,
        model_root=model_base,
        settings=config,
        force=force,
    )
    bundle = build_or_load_final_rank_model(
        training_frame=training_frame,
        training_meta=training_meta,
        model_root=model_base,
        settings=config,
        force=force,
    )
    oos = load_phase4e_oos_scores(output / "phase4e_oos_scores.csv.gz", config)
    replay = replay_lifecycle_engine(oos, settings=config)
    event_contributions = build_event_contributions(
        events=replay["events"],
        training_frame=training_frame,
        bundle=bundle,
    )
    replay["events"] = replay["events"].merge(
        event_contributions,
        on=["event_id", "stock_id", "signal_date"],
        how="left",
        validate="one_to_one",
    )
    model_reports = build_final_model_reports(
        bundle=bundle,
        training_frame=training_frame,
        validation=validation,
        settings=config,
    )
    reports = {
        **replay,
        **model_reports,
        "summary": build_phase5d_summary(
            validation=validation,
            bundle=bundle,
            replay=replay,
            settings=config,
        ),
    }
    paths = export_phase5d_reports(output_dir=output, reports=reports)
    return Phase5DResult(
        status="PASS",
        training_rows=bundle.training_rows,
        replay_events=len(replay["events"]),
        replay_notifications=len(replay["notifications"]),
        output_paths=tuple(paths),
    )


def validate_phase5d_inputs(
    output_dir: Path, *, target_market: str = "tpex"
) -> dict[str, Any]:
    required = (
        "phase3b_summary.csv",
        "phase4e_summary.csv",
        "phase4e_model_comparison.csv",
        "phase4e_coefficient_stability.csv",
        "phase4e_oos_scores.csv.gz",
        "phase4f_summary.csv",
        "phase4f_rule_candidates.csv",
    )
    missing = [name for name in required if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Phase 5D 缺少必要產物：{missing}")
    phase3b = _metric_map(output_dir / "phase3b_summary.csv")
    if phase3b.get("status") != "PASS" or phase3b.get("ready_for_modeling") not in {"1", "1.0"}:
        raise RuntimeError("Phase 3B 尚未完整通過，不可執行 Phase 5D")
    phase4e = _metric_map(output_dir / "phase4e_summary.csv")
    if phase4e.get("ready_for_target_decision") not in {"1", "1.0"}:
        raise RuntimeError("Phase 4E 報告尚未完整通過")
    if phase4e.get("primary_horizon_days") != "20":
        raise RuntimeError("Phase 5D 固定使用 Phase 4E 的 20 日主期限")
    if phase4e.get("market", target_market) != target_market:
        raise RuntimeError(
            f"Phase 4E 市場 {phase4e.get('market')} 與 Phase 5D {target_market} 不一致"
        )
    if target_market == "twse" and phase4e.get(
        "return_rank_validation_pass"
    ) not in {"1", "1.0"}:
        raise RuntimeError("TWSE return-rank 樣本外驗證未通過，不產生最終模型")
    phase4f = _metric_map(output_dir / "phase4f_summary.csv")
    if phase4f.get("pipeline_status") != "PASS" or phase4f.get(
        "ready_for_lifecycle_decision"
    ) not in {"1", "1.0"}:
        raise RuntimeError("Phase 4F 生命週期報告尚未完整通過")
    if phase4f.get("market", target_market) != target_market:
        raise RuntimeError(
            f"Phase 4F 市場 {phase4f.get('market')} 與 Phase 5D {target_market} 不一致"
        )
    if phase4f.get("score_column") != TARGET_SCORE_COLUMN:
        raise RuntimeError("Phase 5D 固定使用 return_rank_score 同日百分位")
    return {
        "phase4e_version": phase4e.get("phase4e_version", ""),
        "phase4e_research_rows": phase4e.get("research_rows", ""),
        "phase4f_version": phase4f.get("phase4f_version", ""),
        "phase4f_source_rows": phase4f.get("source_rows", ""),
        "phase4e_oos_sha256": sha256_file(output_dir / "phase4e_oos_scores.csv.gz"),
    }


def build_or_load_rank_training_frame(
    *,
    shard_dir: Path,
    model_root: Path,
    settings: Phase5DSettings,
    force: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache_path = model_root / "phase5d_rank_training_frame.csv.gz"
    metadata_path = model_root / "phase5d_rank_training_frame.json"
    fingerprint = shard_directory_fingerprint(shard_dir)
    if not force and cache_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("source_fingerprint") == fingerprint:
            frame = pd.read_csv(
                cache_path,
                compression="gzip",
                dtype={"stock_id": "string", "signal_date": "string"},
                low_memory=False,
            )
            _validate_training_frame(frame, settings)
            observed_sha = sha256_file(cache_path)
            if observed_sha == metadata.get("cache_sha256"):
                return frame, metadata

    frames: list[pd.DataFrame] = []
    shard_paths = sorted(shard_dir.glob("*.csv.gz"))
    if not shard_paths:
        raise RuntimeError(f"Phase 5D 找不到 Phase 3 分片：{shard_dir}")
    for position, path in enumerate(shard_paths, start=1):
        shard = pd.read_csv(
            path,
            compression="gzip",
            usecols=[
                *TRAINING_BASE_COLUMNS,
                *market_model_spec(settings.target_market).feature_columns,
            ],
            dtype={"stock_id": "string", "signal_date": "string"},
            low_memory=False,
        )
        if shard.empty:
            continue
        selected = shard[
            (shard["market_type"].astype(str).str.lower() == settings.target_market)
            & (shard["feature_status"].astype(str) == "ok")
            & (pd.to_numeric(shard["liquidity_pass_20m"], errors="coerce") == 1)
            & (shard["label_status_20d"].astype(str) == "ok")
        ].copy()
        if selected.empty:
            continue
        frames.append(selected)
        if position % 100 == 0 or position == len(shard_paths):
            print(f"Phase 5D 最終模型資料準備：{position}/{len(shard_paths)}")
    if not frames:
        raise RuntimeError(
            f"Phase 5D 沒有可用的 {settings.target_market.upper()} 20 日成熟標籤"
        )
    frame = pd.concat(frames, ignore_index=True)
    frame["adjusted_return_20d"] = pd.to_numeric(
        frame["adjusted_return_20d"], errors="coerce"
    )
    feature_columns = market_model_spec(settings.target_market).feature_columns
    for column in feature_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    finite = np.isfinite(frame[["adjusted_return_20d", *feature_columns]]).all(axis=1)
    frame = frame[finite].copy()
    counts = frame.groupby("signal_date")["stock_id"].transform("nunique")
    frame = frame[counts >= settings.minimum_daily_stocks].copy()
    frame["future_return_rank_pct_20d"] = frame.groupby("signal_date")[
        "adjusted_return_20d"
    ].rank(method="average", pct=True)
    frame = frame[
        [
            "stock_id",
            "stock_name",
            "signal_date",
            "adjusted_return_20d",
            "future_return_rank_pct_20d",
            *feature_columns,
        ]
    ].sort_values(["signal_date", "stock_id"], kind="stable")
    frame = frame.reset_index(drop=True)
    _validate_training_frame(frame, settings)
    _write_gzip_csv(cache_path, frame)
    metadata = {
        "phase5d_version": PHASE5D_VERSION,
        "source_fingerprint": fingerprint,
        "cache_sha256": sha256_file(cache_path),
        "row_count": int(len(frame)),
        "signal_dates": int(frame["signal_date"].nunique()),
        "first_signal_date": str(frame["signal_date"].min()),
        "last_signal_date": str(frame["signal_date"].max()),
    }
    _write_json_atomic(metadata_path, metadata)
    return frame, metadata


def shard_directory_fingerprint(shard_dir: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(shard_dir.glob("*.csv.gz"))
    for path in paths:
        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _validate_training_frame(frame: pd.DataFrame, settings: Phase5DSettings) -> None:
    required = {
        "stock_id",
        "signal_date",
        "adjusted_return_20d",
        "future_return_rank_pct_20d",
        *market_model_spec(settings.target_market).feature_columns,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"Phase 5D 最終模型資料缺少欄位：{missing}")
    if frame.empty:
        raise RuntimeError("Phase 5D 最終模型資料為空")
    duplicates = int(frame.duplicated(["stock_id", "signal_date"]).sum())
    if duplicates:
        raise RuntimeError(f"Phase 5D 最終模型資料重複鍵：{duplicates}")
    numeric = frame[
        [
            "adjusted_return_20d",
            "future_return_rank_pct_20d",
            *market_model_spec(settings.target_market).feature_columns,
        ]
    ]
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Phase 5D 最終模型資料包含 NaN／Infinity")
    if not frame["future_return_rank_pct_20d"].between(0, 1, inclusive="both").all():
        raise RuntimeError("Phase 5D 報酬排名 target 超出 0～1")
    date_counts = frame.groupby("signal_date")["stock_id"].nunique()
    if int(date_counts.min()) < settings.minimum_daily_stocks:
        raise RuntimeError("Phase 5D 最終模型資料存在過小同日母體")


def build_or_load_final_rank_model(
    *,
    training_frame: pd.DataFrame,
    training_meta: dict[str, Any],
    model_root: Path,
    settings: Phase5DSettings,
    force: bool,
) -> FinalRankModelBundle:
    source_sha = str(training_meta["cache_sha256"])
    signature = final_rank_model_signature(source_sha256=source_sha, settings=settings)
    manifest_path = model_root / "phase5d_final_rank_model.json"
    arrays_path = model_root / "phase5d_final_rank_model.npz"
    if not force and manifest_path.exists() and arrays_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("signature") == signature:
            return load_final_rank_model(manifest_path=manifest_path, arrays_path=arrays_path)

    feature_columns = market_model_spec(settings.target_market).feature_columns
    features = training_frame[list(feature_columns)].to_numpy(dtype=np.float64)
    targets = training_frame["future_return_rank_pct_20d"].to_numpy(dtype=np.float64)
    preprocessor = fit_all_data_preprocessor(features, settings=settings)
    model = fit_all_data_rank_model(
        features=features,
        targets=targets,
        preprocessor=preprocessor,
        l2_penalty=settings.ranking_l2_penalty,
    )
    bundle = FinalRankModelBundle(
        signature=signature,
        feature_columns=tuple(feature_columns),
        preprocessor=preprocessor,
        model=model,
        training_rows=len(training_frame),
        signal_dates=int(training_frame["signal_date"].nunique()),
        first_training_date=str(training_frame["signal_date"].min()),
        last_training_date=str(training_frame["signal_date"].max()),
        source_sha256=source_sha,
        target_mean=float(targets.mean()),
        target_std=float(targets.std()),
    )
    save_final_rank_model(bundle=bundle, manifest_path=manifest_path, arrays_path=arrays_path)
    return bundle


def fit_all_data_preprocessor(
    features: np.ndarray,
    *,
    settings: Phase5DSettings,
) -> Preprocessor:
    rng = np.random.default_rng(settings.random_seed)
    sample_count = min(settings.quantile_sample_size, len(features))
    selected = np.sort(rng.choice(len(features), size=sample_count, replace=False))
    sample = features[selected]
    lower = np.quantile(sample, settings.clip_lower_quantile, axis=0)
    upper = np.quantile(sample, settings.clip_upper_quantile, axis=0)
    upper = np.maximum(upper, lower)
    clipped = np.clip(features, lower, upper)
    mean = clipped.mean(axis=0)
    std = clipped.std(axis=0)
    std[std < 1e-8] = 1.0
    return Preprocessor(
        lower=lower,
        upper=upper,
        mean=mean,
        std=std,
        sampled_rows=sample_count,
        training_rows=len(features),
    )


def fit_all_data_rank_model(
    *,
    features: np.ndarray,
    targets: np.ndarray,
    preprocessor: Preprocessor,
    l2_penalty: float,
) -> LinearRankModel:
    x = preprocessor.transform(features)
    feature_count = x.shape[1]
    gram = np.zeros((feature_count + 1, feature_count + 1), dtype=np.float64)
    rhs = np.zeros(feature_count + 1, dtype=np.float64)
    gram[0, 0] = len(x)
    sums = x.sum(axis=0)
    gram[0, 1:] = sums
    gram[1:, 0] = sums
    gram[1:, 1:] = x.T @ x
    rhs[0] = targets.sum()
    rhs[1:] = x.T @ targets
    penalty = np.eye(feature_count + 1, dtype=np.float64) * l2_penalty
    penalty[0, 0] = 0.0
    parameters = np.linalg.solve(gram + penalty, rhs)
    return LinearRankModel(weights=parameters[1:], intercept=float(parameters[0]))


def final_rank_model_signature(*, source_sha256: str, settings: Phase5DSettings) -> str:
    payload = {
        "version": PHASE5D_VERSION,
        "source_sha256": source_sha256,
        "market": settings.target_market,
        "features": list(market_model_spec(settings.target_market).feature_columns),
        "clip_lower_quantile": settings.clip_lower_quantile,
        "clip_upper_quantile": settings.clip_upper_quantile,
        "quantile_sample_size": settings.quantile_sample_size,
        "ranking_l2_penalty": settings.ranking_l2_penalty,
        "random_seed": settings.random_seed,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def save_final_rank_model(
    *,
    bundle: FinalRankModelBundle,
    manifest_path: Path,
    arrays_path: Path,
) -> None:
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_arrays = arrays_path.with_suffix(arrays_path.suffix + ".tmp")
    with temporary_arrays.open("wb") as handle:
        np.savez_compressed(
            handle,
            lower=bundle.preprocessor.lower,
            upper=bundle.preprocessor.upper,
            mean=bundle.preprocessor.mean,
            std=bundle.preprocessor.std,
            weights=bundle.model.weights,
            intercept=np.array([bundle.model.intercept], dtype=np.float64),
        )
    temporary_arrays.replace(arrays_path)
    manifest = {
        "phase5d_version": PHASE5D_VERSION,
        "signature": bundle.signature,
        "feature_columns": list(bundle.feature_columns),
        "training_rows": bundle.training_rows,
        "signal_dates": bundle.signal_dates,
        "first_training_date": bundle.first_training_date,
        "last_training_date": bundle.last_training_date,
        "source_sha256": bundle.source_sha256,
        "target_mean": bundle.target_mean,
        "target_std": bundle.target_std,
        "sampled_rows": bundle.preprocessor.sampled_rows,
        "arrays_sha256": sha256_file(arrays_path),
    }
    _write_json_atomic(manifest_path, manifest)


def load_final_rank_model(*, manifest_path: Path, arrays_path: Path) -> FinalRankModelBundle:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(arrays_path) != manifest.get("arrays_sha256"):
        raise RuntimeError("Phase 5D 最終模型陣列 SHA 不一致")
    arrays = np.load(arrays_path)
    feature_columns = tuple(str(value) for value in manifest["feature_columns"])
    supported_feature_sets = {
        market_model_spec("tpex").feature_columns,
        market_model_spec("twse").feature_columns,
    }
    if feature_columns not in supported_feature_sets:
        raise RuntimeError("Phase 5D 最終模型特徵順序不屬於已驗證市場規格")
    preprocessor = Preprocessor(
        lower=arrays["lower"],
        upper=arrays["upper"],
        mean=arrays["mean"],
        std=arrays["std"],
        sampled_rows=int(manifest["sampled_rows"]),
        training_rows=int(manifest["training_rows"]),
    )
    model = LinearRankModel(
        weights=arrays["weights"],
        intercept=float(arrays["intercept"][0]),
    )
    return FinalRankModelBundle(
        signature=str(manifest["signature"]),
        feature_columns=feature_columns,
        preprocessor=preprocessor,
        model=model,
        training_rows=int(manifest["training_rows"]),
        signal_dates=int(manifest["signal_dates"]),
        first_training_date=str(manifest["first_training_date"]),
        last_training_date=str(manifest["last_training_date"]),
        source_sha256=str(manifest["source_sha256"]),
        target_mean=float(manifest["target_mean"]),
        target_std=float(manifest["target_std"]),
    )


def load_phase4e_oos_scores(path: Path, settings: Phase5DSettings) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Phase 5D 找不到 Phase 4E OOS 分數：{path}")
    frame = pd.read_csv(
        path,
        compression="gzip",
        dtype={"stock_id": "string", "signal_date": "string"},
        low_memory=False,
    )
    missing = [column for column in OOS_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Phase 5D OOS 分數缺少欄位：{missing}")
    selected = frame[list(OOS_REQUIRED_COLUMNS)].copy()
    selected["stock_id"] = selected["stock_id"].astype(str)
    selected["stock_name"] = selected["stock_name"].fillna("").astype(str)
    selected["signal_date"] = selected["signal_date"].astype(str)
    selected["test_year"] = pd.to_numeric(selected["test_year"], errors="coerce")
    for column in (
        "adjusted_return_20d",
        "adjusted_return_40d",
        TARGET_SCORE_COLUMN,
    ):
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    if selected[["test_year", TARGET_SCORE_COLUMN]].isna().any().any():
        raise RuntimeError("Phase 5D OOS 年度或分數無效")
    if not np.isfinite(selected[TARGET_SCORE_COLUMN]).all():
        raise RuntimeError("Phase 5D OOS 分數包含 NaN／Infinity")
    if not selected[TARGET_SCORE_COLUMN].between(0, 100, inclusive="both").all():
        raise RuntimeError("Phase 5D OOS 百分位超出 0～100")
    duplicates = int(selected.duplicated(["stock_id", "signal_date"]).sum())
    if duplicates:
        raise RuntimeError(f"Phase 5D OOS 股票＋日期重複：{duplicates}")
    counts = selected.groupby("signal_date")["stock_id"].transform("nunique")
    selected = selected[counts >= settings.minimum_daily_stocks].copy()
    selected["test_year"] = selected["test_year"].astype(int)
    return selected.sort_values(["signal_date", "stock_id"], kind="stable").reset_index(drop=True)


def replay_lifecycle_engine(
    frame: pd.DataFrame,
    *,
    settings: Phase5DSettings,
) -> dict[str, pd.DataFrame]:
    settings.validate()
    base = frame.copy()
    dates = sorted(base["signal_date"].unique())
    date_index = {value: index for index, value in enumerate(dates)}
    base["market_day_index"] = base["signal_date"].map(date_index).astype(int)
    for horizon in (20, 40):
        column = f"adjusted_return_{horizon}d"
        base[f"daily_universe_return_{horizon}d"] = base.groupby("signal_date")[
            column
        ].transform("mean")
        base[f"excess_return_{horizon}d"] = (
            base[column] - base[f"daily_universe_return_{horizon}d"]
        )
    rows_by_date = {
        date: group.set_index("stock_id", drop=False)
        for date, group in base.groupby("signal_date", sort=True)
    }
    active: dict[str, dict[str, Any]] = {}
    cooldown_until: dict[str, int] = {}
    confirmation_run: dict[str, int] = {}
    last_seen_index: dict[str, int] = {}
    previous_score: dict[str, float] = {}
    events: list[dict[str, Any]] = []
    notifications: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    event_lookup: dict[str, dict[str, Any]] = {}

    for current_date in dates:
        current_index = date_index[current_date]
        day = rows_by_date[current_date]
        current_stocks = set(day.index.astype(str))
        for stock_id in current_stocks:
            score = float(day.loc[stock_id, TARGET_SCORE_COLUMN])
            consecutive = last_seen_index.get(stock_id) == current_index - 1
            if consecutive and score >= settings.confirmation_threshold:
                confirmation_run[stock_id] = confirmation_run.get(stock_id, 0) + 1
            elif score >= settings.confirmation_threshold:
                confirmation_run[stock_id] = 1
            else:
                confirmation_run[stock_id] = 0

        for stock_id in list(active):
            state = active[stock_id]
            age = current_index - int(state["start_index"])
            row = day.loc[stock_id] if stock_id in day.index else None
            score = float(row[TARGET_SCORE_COLUMN]) if row is not None else math.nan
            if (
                not state["confirmed"]
                and row is not None
                and confirmation_run.get(stock_id, 0) >= settings.confirmation_days
            ):
                state["confirmed"] = True
                state["status"] = "LAYOUT_CONFIRMED"
                _append_notification(
                    notifications,
                    transitions,
                    event=state,
                    signal_date=current_date,
                    market_day_index=current_index,
                    notification_type="LAYOUT_CONFIRMED",
                    status="LAYOUT_CONFIRMED",
                    percentile=score,
                    event_age_days=age,
                    reason=(
                        f"連續{settings.confirmation_days}個交易日法人排名維持前20%"
                    ),
                )
            if age == settings.primary_tracking_days:
                if row is None:
                    _end_event(
                        state,
                        current_date=current_date,
                        current_index=current_index,
                        reason="DAY20_END_NO_SCORE",
                        percentile=math.nan,
                        notifications=notifications,
                        transitions=transitions,
                    )
                    cooldown_until[stock_id] = current_index + settings.cooldown_days
                    del active[stock_id]
                elif score >= settings.entry_threshold:
                    state["status"] = "EXTENDED_STRONG"
                    state["extended"] = True
                    _append_notification(
                        notifications,
                        transitions,
                        event=state,
                        signal_date=current_date,
                        market_day_index=current_index,
                        notification_type="DAY20_EXTEND_STRONG",
                        status="EXTENDED_STRONG",
                        percentile=score,
                        event_age_days=age,
                        reason="第20日仍在法人排名前10%，延長追蹤至第40日",
                    )
                elif score >= settings.confirmation_threshold:
                    state["status"] = "EXTENDED"
                    state["extended"] = True
                    _append_notification(
                        notifications,
                        transitions,
                        event=state,
                        signal_date=current_date,
                        market_day_index=current_index,
                        notification_type="DAY20_EXTEND",
                        status="EXTENDED",
                        percentile=score,
                        event_age_days=age,
                        reason="第20日仍在法人排名前20%，延長觀察至第40日",
                    )
                else:
                    _end_event(
                        state,
                        current_date=current_date,
                        current_index=current_index,
                        reason="DAY20_END_WEAKENED",
                        percentile=score,
                        notifications=notifications,
                        transitions=transitions,
                    )
                    cooldown_until[stock_id] = current_index + settings.cooldown_days
                    del active[stock_id]
            elif age >= settings.maximum_tracking_days:
                _end_event(
                    state,
                    current_date=current_date,
                    current_index=current_index,
                    reason="DAY40_END",
                    percentile=score,
                    notifications=notifications,
                    transitions=transitions,
                )
                cooldown_until[stock_id] = current_index + settings.cooldown_days
                del active[stock_id]

        for stock_id in sorted(current_stocks):
            row = day.loc[stock_id]
            score = float(row[TARGET_SCORE_COLUMN])
            previous = previous_score.get(stock_id)
            first_crossing = score >= settings.entry_threshold and (
                previous is None
                or last_seen_index.get(stock_id) != current_index - 1
                or previous < settings.entry_threshold
            )
            direct_confirmation = (
                confirmation_run.get(stock_id, 0) >= settings.confirmation_days
            )
            cooling = current_index <= cooldown_until.get(stock_id, -10**9)
            if stock_id not in active and not cooling and (direct_confirmation or first_crossing):
                event_id = f"P5D-{stock_id}-{current_date}"
                trigger = "DIRECT_LAYOUT_CONFIRMATION" if direct_confirmation else "FIRST_TOP10_ENTRY"
                confirmed = bool(direct_confirmation)
                status = "LAYOUT_CONFIRMED" if confirmed else "NEW_CANDIDATE"
                event = {
                    "event_id": event_id,
                    "stock_id": stock_id,
                    "stock_name": str(row["stock_name"]),
                    "signal_date": current_date,
                    "test_year": int(row["test_year"]),
                    "start_index": current_index,
                    "entry_percentile": score,
                    "entry_trigger": trigger,
                    "high_intensity_at_entry": int(score >= settings.high_intensity_threshold),
                    "high_intensity_threshold": settings.high_intensity_threshold,
                    "confirmed": confirmed,
                    "extended": False,
                    "status": status,
                    "end_signal_date": "",
                    "end_index": -1,
                    "end_reason": "",
                    "entry_return_20d": row["adjusted_return_20d"],
                    "entry_return_40d": row["adjusted_return_40d"],
                    "entry_excess_return_20d": row["excess_return_20d"],
                    "entry_excess_return_40d": row["excess_return_40d"],
                }
                active[stock_id] = event
                events.append(event)
                event_lookup[event_id] = event
                notification_type = (
                    "LAYOUT_CONFIRMED_DIRECT" if confirmed else "NEW_CANDIDATE"
                )
                reason = (
                    f"連續{settings.confirmation_days}日維持法人排名前20%，直接列為布局確認"
                    if confirmed
                    else f"法人排名首次進入同日{settings.target_market.upper()}前10%"
                )
                _append_notification(
                    notifications,
                    transitions,
                    event=event,
                    signal_date=current_date,
                    market_day_index=current_index,
                    notification_type=notification_type,
                    status=status,
                    percentile=score,
                    event_age_days=0,
                    reason=reason,
                )

            previous_score[stock_id] = score
            last_seen_index[stock_id] = current_index

    final_index = len(dates) - 1
    final_date = dates[-1]
    for stock_id, state in active.items():
        state["status"] = "OPEN_AT_DATA_END"
        transitions.append(
            {
                "event_id": state["event_id"],
                "stock_id": stock_id,
                "stock_name": state["stock_name"],
                "signal_date": final_date,
                "market_day_index": final_index,
                "event_age_days": final_index - int(state["start_index"]),
                "transition_type": "DATA_END",
                "status": "OPEN_AT_DATA_END",
                "percentile": np.nan,
                "reason": "樣本外資料結束，事件尚未走完20／40日生命週期",
            }
        )

    event_frame = pd.DataFrame(events)
    if event_frame.empty:
        raise RuntimeError("Phase 5D 狀態引擎沒有產生任何事件")
    event_frame = event_frame.drop(columns=["start_index"], errors="ignore")
    notification_frame = pd.DataFrame(notifications)
    transition_frame = pd.DataFrame(transitions)
    _validate_replay_outputs(event_frame, notification_frame)
    performance = build_replay_performance(event_frame)
    yearly = build_replay_yearly_performance(event_frame)
    bootstrap = build_replay_bootstrap(event_frame, settings=settings)
    daily_counts = build_daily_notification_counts(notification_frame)
    state_counts = build_state_transition_counts(transition_frame)
    audit = build_replay_audit(base, event_frame, notification_frame, settings)
    return {
        "events": event_frame,
        "notifications": notification_frame,
        "transitions": transition_frame,
        "performance": performance,
        "yearly": yearly,
        "bootstrap": bootstrap,
        "daily_counts": daily_counts,
        "state_counts": state_counts,
        "replay_audit": audit,
    }


def _append_notification(
    notifications: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    *,
    event: dict[str, Any],
    signal_date: str,
    market_day_index: int,
    notification_type: str,
    status: str,
    percentile: float,
    event_age_days: int,
    reason: str,
) -> None:
    high_intensity_threshold = float(
        event.get("high_intensity_threshold", DEFAULT_HIGH_INTENSITY_THRESHOLD)
    )
    high_intensity = bool(
        np.isfinite(percentile) and percentile >= high_intensity_threshold
    )
    row = {
        "event_id": event["event_id"],
        "stock_id": event["stock_id"],
        "stock_name": event["stock_name"],
        "signal_date": signal_date,
        "market_day_index": market_day_index,
        "event_age_days": event_age_days,
        "notification_type": notification_type,
        "status": status,
        "percentile": percentile,
        "high_intensity": int(high_intensity),
        "reason": reason,
        "trade_action": "TRACK_ONLY",
    }
    notifications.append(row)
    transitions.append(
        {
            **row,
            "transition_type": notification_type,
        }
    )


def _end_event(
    event: dict[str, Any],
    *,
    current_date: str,
    current_index: int,
    reason: str,
    percentile: float,
    notifications: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
) -> None:
    event["status"] = "ENDED"
    event["end_signal_date"] = current_date
    event["end_index"] = current_index
    event["end_reason"] = reason
    _append_notification(
        notifications,
        transitions,
        event=event,
        signal_date=current_date,
        market_day_index=current_index,
        notification_type=reason,
        status="ENDED",
        percentile=percentile,
        event_age_days=current_index - int(event["start_index"]),
        reason=(
            "第20日法人排名已跌出前20%，結束積極追蹤；不代表賣出"
            if reason == "DAY20_END_WEAKENED"
            else "第20日無可用排名，結束本次追蹤"
            if reason == "DAY20_END_NO_SCORE"
            else "已達第40日，結束本次法人布局事件"
        ),
    )


def _validate_replay_outputs(events: pd.DataFrame, notifications: pd.DataFrame) -> None:
    if int(events["event_id"].duplicated().sum()):
        raise RuntimeError("Phase 5D 事件 ID 重複")
    duplicate_notifications = int(
        notifications.duplicated(["event_id", "signal_date", "notification_type"]).sum()
    )
    if duplicate_notifications:
        raise RuntimeError(f"Phase 5D 通知重複：{duplicate_notifications}")
    if not notifications["trade_action"].eq("TRACK_ONLY").all():
        raise RuntimeError("Phase 5D 法人通知不可直接產生買進／賣出動作")


def build_replay_performance(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    latest_year = int(events["test_year"].max())
    periods = (
        ("all_oos", events),
        ("confirmation", events[events["test_year"] >= 2023]),
        (
            "confirmation_ex_latest",
            events[(events["test_year"] >= 2023) & (events["test_year"] < latest_year)],
        ),
    )
    for period_name, period in periods:
        for trigger, group in period.groupby("entry_trigger", sort=True):
            row = {
                "period": period_name,
                "entry_trigger": trigger,
                "event_count": int(len(group)),
                "unique_stocks": int(group["stock_id"].nunique()),
                "signal_dates": int(group["signal_date"].nunique()),
                "high_intensity_entry_rate": float(group["high_intensity_at_entry"].mean()),
                "confirmed_event_rate": float(group["confirmed"].mean()),
                "extended_event_rate": float(group["extended"].mean()),
            }
            row.update(_return_metrics(group, "entry_return_20d", "return_20d"))
            row.update(_return_metrics(group, "entry_return_40d", "return_40d"))
            row.update(
                _return_metrics(group, "entry_excess_return_20d", "excess_return_20d")
            )
            row.update(
                _return_metrics(group, "entry_excess_return_40d", "excess_return_40d")
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_replay_bootstrap(
    events: pd.DataFrame,
    *,
    settings: Phase5DSettings,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    latest_year = int(events["test_year"].max())
    periods = (
        ("confirmation", events[events["test_year"] >= 2023]),
        (
            "confirmation_ex_latest",
            events[(events["test_year"] >= 2023) & (events["test_year"] < latest_year)],
        ),
    )
    for period_name, period in periods:
        for trigger, group in period.groupby("entry_trigger", sort=True):
            for horizon in (20, 40):
                column = f"entry_excess_return_{horizon}d"
                selected = group[["signal_date", column]].copy()
                selected[column] = pd.to_numeric(selected[column], errors="coerce")
                selected = selected[np.isfinite(selected[column])]
                daily = selected.groupby("signal_date")[column].mean().sort_index()
                monthly = daily.groupby(daily.index.astype(str).str[:7]).mean().sort_index()
                if len(monthly) < 2:
                    lower = upper = bootstrap_mean = np.nan
                else:
                    lower, upper, bootstrap_mean = moving_block_bootstrap_mean_ci(
                        monthly.to_numpy(dtype=np.float64),
                        iterations=settings.bootstrap_iterations,
                        block_length=settings.bootstrap_block_months,
                        random_seed=_stable_seed(
                            settings.random_seed,
                            f"{period_name}-{trigger}-{horizon}",
                        ),
                    )
                rows.append(
                    {
                        "period": period_name,
                        "entry_trigger": trigger,
                        "horizon_days": horizon,
                        "event_count": int(len(group)),
                        "daily_observations": int(len(daily)),
                        "monthly_blocks": int(len(monthly)),
                        "point_estimate_daily_equal_weight_excess_return": float(daily.mean())
                        if len(daily)
                        else np.nan,
                        "bootstrap_mean_excess_return": bootstrap_mean,
                        "confidence_level": 0.95,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "ci_excludes_zero_positive": int(
                            np.isfinite(lower) and lower > 0
                        ),
                        "iterations": settings.bootstrap_iterations,
                        "block_months": min(settings.bootstrap_block_months, len(monthly))
                        if len(monthly)
                        else 0,
                    }
                )
    return pd.DataFrame(rows)


def build_replay_yearly_performance(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (trigger, year), group in events.groupby(["entry_trigger", "test_year"], sort=True):
        row = {
            "entry_trigger": trigger,
            "test_year": int(year),
            "event_count": int(len(group)),
            "unique_stocks": int(group["stock_id"].nunique()),
            "signal_dates": int(group["signal_date"].nunique()),
        }
        row.update(_return_metrics(group, "entry_return_20d", "return_20d"))
        row.update(_return_metrics(group, "entry_return_40d", "return_40d"))
        row.update(_return_metrics(group, "entry_excess_return_20d", "excess_return_20d"))
        row.update(_return_metrics(group, "entry_excess_return_40d", "excess_return_40d"))
        rows.append(row)
    return pd.DataFrame(rows)


def _return_metrics(frame: pd.DataFrame, column: str, prefix: str) -> dict[str, Any]:
    valid = frame[["signal_date", column]].copy()
    valid[column] = pd.to_numeric(valid[column], errors="coerce")
    valid = valid[np.isfinite(valid[column])]
    if valid.empty:
        return {
            f"{prefix}_sample_count": 0,
            f"{prefix}_daily_equal_weight": np.nan,
            f"{prefix}_sample_average": np.nan,
            f"{prefix}_sample_median": np.nan,
            f"{prefix}_positive_rate": np.nan,
            f"{prefix}_up_5pct_rate": np.nan,
            f"{prefix}_down_5pct_rate": np.nan,
        }
    values = valid[column].to_numpy(dtype=np.float64)
    daily = valid.groupby("signal_date")[column].mean()
    return {
        f"{prefix}_sample_count": int(len(valid)),
        f"{prefix}_daily_equal_weight": float(daily.mean()),
        f"{prefix}_sample_average": float(values.mean()),
        f"{prefix}_sample_median": float(np.median(values)),
        f"{prefix}_positive_rate": float((values > 0).mean()),
        f"{prefix}_up_5pct_rate": float((values >= 0.05).mean()),
        f"{prefix}_down_5pct_rate": float((values <= -0.05).mean()),
    }


def build_daily_notification_counts(notifications: pd.DataFrame) -> pd.DataFrame:
    counts = (
        notifications.groupby(["signal_date", "notification_type"], sort=True)
        .size()
        .rename("notification_count")
        .reset_index()
    )
    totals = (
        notifications.groupby("signal_date", sort=True)
        .size()
        .rename("notification_count")
        .reset_index()
    )
    totals["notification_type"] = "ALL"
    return pd.concat([counts, totals], ignore_index=True).sort_values(
        ["signal_date", "notification_type"], kind="stable"
    )


def build_state_transition_counts(transitions: pd.DataFrame) -> pd.DataFrame:
    return (
        transitions.groupby(["transition_type", "status"], sort=True)
        .size()
        .rename("transition_count")
        .reset_index()
    )


def build_replay_audit(
    base: pd.DataFrame,
    events: pd.DataFrame,
    notifications: pd.DataFrame,
    settings: Phase5DSettings,
) -> pd.DataFrame:
    values = {
        "phase5d_version": PHASE5D_VERSION,
        "source_rows": len(base),
        "source_stocks": base["stock_id"].nunique(),
        "source_signal_dates": base["signal_date"].nunique(),
        "first_signal_date": base["signal_date"].min(),
        "last_signal_date": base["signal_date"].max(),
        "event_rows": len(events),
        "notification_rows": len(notifications),
        "duplicate_event_ids": int(events["event_id"].duplicated().sum()),
        "duplicate_notifications": int(
            notifications.duplicated(["event_id", "signal_date", "notification_type"]).sum()
        ),
        "entry_threshold": settings.entry_threshold,
        "confirmation_threshold": settings.confirmation_threshold,
        "confirmation_days": settings.confirmation_days,
        "primary_tracking_days": settings.primary_tracking_days,
        "maximum_tracking_days": settings.maximum_tracking_days,
        "cooldown_days": settings.cooldown_days,
        "bootstrap_iterations": settings.bootstrap_iterations,
        "bootstrap_block_months": settings.bootstrap_block_months,
        "leakage_note": "事件觸發只使用當日及過去OOS分數；20／40日報酬只用於事後驗證。",
        "trade_action_note": "法人模型所有通知固定為TRACK_ONLY，不直接輸出買進或賣出。",
    }
    return pd.DataFrame([{"metric": key, "value": value} for key, value in values.items()])


def build_event_contributions(
    *,
    events: pd.DataFrame,
    training_frame: pd.DataFrame,
    bundle: FinalRankModelBundle,
) -> pd.DataFrame:
    keys = events[["event_id", "stock_id", "signal_date"]].copy()
    features = training_frame[["stock_id", "signal_date", *bundle.feature_columns]].copy()
    merged = keys.merge(
        features,
        on=["stock_id", "signal_date"],
        how="left",
        validate="one_to_one",
    )
    feature_values = merged[list(bundle.feature_columns)].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float64)
    complete = np.isfinite(feature_values).all(axis=1)
    output_rows: list[dict[str, Any]] = []
    for position, row in enumerate(merged.itertuples(index=False)):
        base = {
            "event_id": row.event_id,
            "stock_id": row.stock_id,
            "signal_date": row.signal_date,
            "explanation_available": int(complete[position]),
            "explanation_model_scope": "final_all_mature_history_non_oos",
        }
        if not complete[position]:
            output_rows.append(base)
            continue
        standardized = bundle.preprocessor.transform(feature_values[position : position + 1])[0]
        contributions = standardized * bundle.model.weights
        group_values: dict[str, float] = {
            "foreign": 0.0,
            "investment_trust": 0.0,
            "dealer_self": 0.0,
            "institutional_consensus": 0.0,
        }
        for feature, contribution in zip(bundle.feature_columns, contributions, strict=True):
            group_values[feature_group(feature)] += float(contribution)
        positive = np.argsort(contributions)[::-1][:3]
        negative = np.argsort(contributions)[:3]
        standardized_row = bundle.preprocessor.transform(
            feature_values[position : position + 1]
        )
        base.update(
            {
                "final_model_raw_score": float(
                    bundle.model.scores(standardized_row)[0]
                ),
                "foreign_contribution": group_values["foreign"],
                "investment_trust_contribution": group_values["investment_trust"],
                "dealer_self_contribution": group_values["dealer_self"],
                "institutional_consensus_contribution": group_values["institutional_consensus"],
                "top_positive_factors": "；".join(
                    f"{FEATURE_LABELS.get(bundle.feature_columns[index], bundle.feature_columns[index])}({contributions[index]:+.4f})"
                    for index in positive
                ),
                "top_negative_factors": "；".join(
                    f"{FEATURE_LABELS.get(bundle.feature_columns[index], bundle.feature_columns[index])}({contributions[index]:+.4f})"
                    for index in negative
                ),
            }
        )
        output_rows.append(base)
    return pd.DataFrame(output_rows)


def score_rank_cross_section(
    frame: pd.DataFrame,
    *,
    bundle: FinalRankModelBundle,
) -> pd.DataFrame:
    missing = [column for column in bundle.feature_columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Phase 5D 推論缺少特徵：{missing}")
    result = frame.copy()
    features = result[list(bundle.feature_columns)].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(features).all():
        raise RuntimeError("Phase 5D 推論特徵包含 NaN／Infinity")
    standardized = bundle.preprocessor.transform(features)
    result["return_rank_raw_score"] = bundle.model.scores(standardized)
    result["return_rank_daily_percentile"] = (
        result["return_rank_raw_score"].rank(method="average", pct=True) * 100.0
    )
    contributions = standardized * bundle.model.weights
    for group in (
        "foreign",
        "investment_trust",
        "dealer_self",
        "institutional_consensus",
    ):
        indices = [
            index
            for index, feature in enumerate(bundle.feature_columns)
            if feature_group(feature) == group
        ]
        result[f"{group}_contribution"] = contributions[:, indices].sum(axis=1)
    return result


def build_final_model_reports(
    *,
    bundle: FinalRankModelBundle,
    training_frame: pd.DataFrame,
    validation: dict[str, Any],
    settings: Phase5DSettings,
) -> dict[str, pd.DataFrame]:
    features = training_frame[list(bundle.feature_columns)].to_numpy(dtype=np.float64)
    targets = training_frame["future_return_rank_pct_20d"].to_numpy(dtype=np.float64)
    transformed = bundle.preprocessor.transform(features)
    predictions = bundle.model.scores(transformed)
    residuals = targets - predictions
    centered = targets - targets.mean()
    r2 = 1.0 - float(np.square(residuals).sum() / max(np.square(centered).sum(), 1e-12))
    correlation = float(np.corrcoef(targets, predictions)[0, 1])
    manifest = pd.DataFrame(
        [
            {"metric": "phase5d_version", "value": PHASE5D_VERSION},
            {"metric": "model_target", "value": "same_day_future_return_rank_20d"},
            {"metric": "market", "value": settings.target_market},
            {"metric": "feature_count", "value": len(bundle.feature_columns)},
            {"metric": "training_rows", "value": bundle.training_rows},
            {"metric": "training_signal_dates", "value": bundle.signal_dates},
            {"metric": "first_training_date", "value": bundle.first_training_date},
            {"metric": "last_training_date", "value": bundle.last_training_date},
            {"metric": "source_sha256", "value": bundle.source_sha256},
            {"metric": "model_signature", "value": bundle.signature},
            {"metric": "ranking_l2_penalty", "value": settings.ranking_l2_penalty},
            {"metric": "training_target_mean", "value": bundle.target_mean},
            {"metric": "training_target_std", "value": bundle.target_std},
            {"metric": "training_fit_r2_descriptive_only", "value": r2},
            {"metric": "training_fit_correlation_descriptive_only", "value": correlation},
            {"metric": "phase4e_oos_sha256", "value": validation["phase4e_oos_sha256"]},
            {
                "metric": "validation_note",
                "value": "最終全歷史模型訓練配適不可替代Phase 4E樣本外驗證。",
            },
        ]
    )
    coefficients = pd.DataFrame(
        [
            {
                "feature_order": index + 1,
                "feature_name": feature,
                "feature_label": FEATURE_LABELS.get(feature, feature),
                "feature_group": feature_group(feature),
                "coefficient": float(bundle.model.weights[index]),
            }
            for index, feature in enumerate(bundle.feature_columns)
        ]
    )
    coefficients = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "feature_order": 0,
                        "feature_name": "__INTERCEPT__",
                        "feature_label": "截距",
                        "feature_group": "intercept",
                        "coefficient": bundle.model.intercept,
                    }
                ]
            ),
            coefficients,
        ],
        ignore_index=True,
    )
    preprocessing = pd.DataFrame(
        [
            {
                "feature_order": index + 1,
                "feature_name": feature,
                "clip_lower": float(bundle.preprocessor.lower[index]),
                "clip_upper": float(bundle.preprocessor.upper[index]),
                "mean": float(bundle.preprocessor.mean[index]),
                "std": float(bundle.preprocessor.std[index]),
            }
            for index, feature in enumerate(bundle.feature_columns)
        ]
    )
    group_coefficients = (
        coefficients[coefficients["feature_order"] > 0]
        .groupby("feature_group", sort=True)
        .agg(
            feature_count=("feature_name", "size"),
            coefficient_sum=("coefficient", "sum"),
            absolute_coefficient_sum=("coefficient", lambda values: float(np.abs(values).sum())),
        )
        .reset_index()
    )
    return {
        "model_manifest": manifest,
        "model_coefficients": coefficients,
        "model_preprocessing": preprocessing,
        "model_group_coefficients": group_coefficients,
    }


def build_phase5d_summary(
    *,
    validation: dict[str, Any],
    bundle: FinalRankModelBundle,
    replay: dict[str, pd.DataFrame],
    settings: Phase5DSettings,
) -> pd.DataFrame:
    events = replay["events"]
    notifications = replay["notifications"]
    daily_all = replay["daily_counts"]
    daily_all = daily_all[daily_all["notification_type"] == "ALL"]
    latest_year = int(events["test_year"].max())
    main_period = events[(events["test_year"] >= 2023) & (events["test_year"] < latest_year)]
    metrics = {
        "phase5d_version": PHASE5D_VERSION,
        "pipeline_status": "PASS",
        "market": settings.target_market,
        "primary_model": "return_rank_score_20d",
        "primary_tracking_days": settings.primary_tracking_days,
        "maximum_tracking_days": settings.maximum_tracking_days,
        "entry_threshold": settings.entry_threshold,
        "high_intensity_threshold": settings.high_intensity_threshold,
        "confirmation_threshold": settings.confirmation_threshold,
        "confirmation_days": settings.confirmation_days,
        "cooldown_days": settings.cooldown_days,
        "final_model_training_rows": bundle.training_rows,
        "final_model_last_training_date": bundle.last_training_date,
        "replay_event_count": len(events),
        "replay_notification_count": len(notifications),
        "replay_confirmation_ex_latest_events": len(main_period),
        "average_notifications_per_active_date": float(daily_all["notification_count"].mean()),
        "maximum_notifications_on_one_date": int(daily_all["notification_count"].max()),
        "direct_trade_actions": int(notifications["trade_action"].ne("TRACK_ONLY").sum()),
        "ready_for_github_integration_decision": 1,
        "decision_note": "PASS代表最終模型產物與狀態重播完整；正式GitHub Actions整合仍須自行核對通知數與樣本外表現。",
        "explanation_note": "歷史重播的個股貢獻使用最終全成熟資料模型，只供解釋，不參與樣本外績效驗證。",
        "phase4e_oos_sha256": validation["phase4e_oos_sha256"],
    }
    return pd.DataFrame([{"metric": key, "value": value} for key, value in metrics.items()])


def export_phase5d_reports(
    *,
    output_dir: Path,
    reports: dict[str, pd.DataFrame],
) -> list[Path]:
    names = {
        "model_manifest": "phase5d_model_manifest.csv",
        "model_coefficients": "phase5d_model_coefficients.csv",
        "model_preprocessing": "phase5d_model_preprocessing.csv",
        "model_group_coefficients": "phase5d_model_group_coefficients.csv",
        "events": "phase5d_replay_events.csv.gz",
        "notifications": "phase5d_notification_replay.csv",
        "transitions": "phase5d_state_transitions.csv.gz",
        "performance": "phase5d_replay_performance.csv",
        "yearly": "phase5d_replay_yearly_stability.csv",
        "bootstrap": "phase5d_replay_bootstrap_confidence.csv",
        "daily_counts": "phase5d_daily_notification_counts.csv",
        "state_counts": "phase5d_state_transition_counts.csv",
        "replay_audit": "phase5d_replay_audit.csv",
        "summary": "phase5d_summary.csv",
    }
    paths: list[Path] = []
    for key, name in names.items():
        path = output_dir / name
        frame = reports[key]
        if name.endswith(".csv.gz"):
            _write_gzip_csv(path, frame)
        else:
            _write_csv(path, frame)
        paths.append(path)
    markdown = output_dir / "phase5d_local_notification_summary.md"
    markdown.write_text(_build_markdown_summary(reports), encoding="utf-8")
    paths.append(markdown)
    archive = output_dir / "phase5d_final_model_reports.zip"
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in paths:
            handle.write(path, arcname=path.name)
    temporary.replace(archive)
    paths.append(archive)
    return paths


def _build_markdown_summary(reports: dict[str, pd.DataFrame]) -> str:
    summary = dict(
        zip(
            reports["summary"]["metric"].astype(str),
            reports["summary"]["value"].astype(str),
            strict=True,
        )
    )
    counts = reports["state_counts"].sort_values("transition_count", ascending=False)
    market = summary.get("market", "tpex").upper()
    lines = [
        f"# Phase 5D {market} 法人模型與通知狀態重播",
        "",
        f"- 最終模型：`{summary.get('primary_model', '')}`",
        f"- 最終模型成熟訓練列：{summary.get('final_model_training_rows', '')}",
        f"- 歷史重播事件：{summary.get('replay_event_count', '')}",
        f"- 歷史重播通知：{summary.get('replay_notification_count', '')}",
        f"- 每個有效日期平均通知：{float(summary.get('average_notifications_per_active_date', 0)):.2f}",
        f"- 單日最高通知：{summary.get('maximum_notifications_on_one_date', '')}",
        "",
        "## 固定規則",
        "",
        f"- 首次進入同日 {market} 前 10%：法人候選。",
        "- 百分位至少 95：高強度標章。",
        "- 連續 5 日維持前 20%：法人布局確認。",
        "- 第 20 日仍在前 20%：延長至第 40 日。",
        "- 第 20 日跌出前 20%：結束積極追蹤，但不代表賣出。",
        "- 第 40 日：結束事件並進入 20 個交易日冷卻。",
        "",
        "## 狀態轉移數量",
        "",
        "| 轉移 | 狀態 | 次數 |",
        "|---|---|---:|",
    ]
    for row in counts.itertuples(index=False):
        lines.append(f"| {row.transition_type} | {row.status} | {int(row.transition_count)} |")
    lines.extend(
        [
            "",
            "> 法人模型所有通知固定為 TRACK_ONLY，不直接產生買進或賣出建議。",
        ]
    )
    return "\n".join(lines) + "\n"


def _stable_seed(base_seed: int, value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return base_seed + int.from_bytes(digest[:4], byteorder="big", signed=False)


def _metric_map(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    return dict(zip(frame["metric"], frame["value"], strict=True))


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)
    return path


def _write_gzip_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", compression="gzip")
    temporary.replace(path)
    return path


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)
