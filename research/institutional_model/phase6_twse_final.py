from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.institutional_model.database import ResearchDatabase
from research.institutional_model.market_model_spec import market_model_spec
from research.institutional_model.phase4_horizon import build_horizon_research_frame
from research.institutional_model.phase4_selection import resolve_phase3_shard_directory
from research.institutional_model.phase5_final_model import (
    FinalRankModelBundle,
    Phase5DSettings,
    build_or_load_final_rank_model,
)
from research.institutional_model.phase6_twse_40d import (
    Phase6BTWSE40DSettings,
    _to_horizon_settings,
    prepare_twse_40d_frame,
)


PHASE6D_VERSION = "phase6d-v1"
TARGET_MARKET = "twse"
TARGET_HORIZON_DAYS = 40
SELECTED_LIQUIDITY_UNIVERSE = "money100m_volume300lots"
ENTRY_PERCENTILE = 90.0
CONFIRMATION_DAYS = 3
TRACKING_DAYS = 40
COOLDOWN_DAYS = 20


@dataclass(frozen=True)
class Phase6DTWSEFinalSettings:
    minimum_daily_stocks: int = 50
    clip_lower_quantile: float = 0.005
    clip_upper_quantile: float = 0.995
    quantile_sample_size: int = 250_000
    ranking_l2_penalty: float = 0.001
    random_seed: int = 20260801

    def validate(self) -> None:
        if self.minimum_daily_stocks < 20:
            raise ValueError("Phase 6D 每日最低股票數不可低於 20")
        if not 0 <= self.clip_lower_quantile < self.clip_upper_quantile <= 1:
            raise ValueError("Phase 6D 截尾分位數設定無效")
        if self.quantile_sample_size <= 0:
            raise ValueError("Phase 6D 分位數抽樣筆數必須大於 0")
        if self.ranking_l2_penalty < 0:
            raise ValueError("Phase 6D ranking L2 不可小於 0")


@dataclass(frozen=True)
class Phase6DTWSEFinalResult:
    status: str
    training_rows: int
    signal_dates: int
    output_paths: tuple[Path, ...]


def run_phase6d_twse_final_model(
    *,
    database: ResearchDatabase,
    output_dir: Path | str,
    shard_root: Path | str,
    cache_root: Path | str,
    model_root: Path | str,
    settings: Phase6DTWSEFinalSettings | None = None,
    force: bool = False,
) -> Phase6DTWSEFinalResult:
    config = settings or Phase6DTWSEFinalSettings()
    config.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_dir = Path(model_root)
    model_dir.mkdir(parents=True, exist_ok=True)
    gate = validate_phase6d_gates(output)

    phase6b_settings = Phase6BTWSE40DSettings(
        minimum_daily_stocks=config.minimum_daily_stocks,
        clip_lower_quantile=config.clip_lower_quantile,
        clip_upper_quantile=config.clip_upper_quantile,
        quantile_sample_size=config.quantile_sample_size,
        ranking_l2_penalty=config.ranking_l2_penalty,
        random_seed=config.random_seed,
    )
    shard_dir = resolve_phase3_shard_directory(
        output_dir=output,
        shard_root=Path(shard_root),
    )
    raw = build_horizon_research_frame(
        database=database,
        shard_dir=shard_dir,
        cache_dir=Path(cache_root),
        settings=_to_horizon_settings(phase6b_settings),
        force=force,
    )
    frame = prepare_twse_40d_frame(raw, settings=phase6b_settings)
    feature_columns = market_model_spec(TARGET_MARKET).feature_columns
    training = frame[
        [
            "stock_id",
            "stock_name",
            "signal_date",
            "adjusted_return_40d",
            "future_return_rank_pct_40d",
            *feature_columns,
        ]
    ].copy()
    training = training.sort_values(["signal_date", "stock_id"], kind="stable").reset_index(drop=True)
    source_sha = _frame_sha256(training)
    phase5_settings = Phase5DSettings(
        target_market=TARGET_MARKET,
        minimum_daily_stocks=config.minimum_daily_stocks,
        clip_lower_quantile=config.clip_lower_quantile,
        clip_upper_quantile=config.clip_upper_quantile,
        quantile_sample_size=config.quantile_sample_size,
        ranking_l2_penalty=config.ranking_l2_penalty,
        random_seed=config.random_seed,
    )
    phase5_training = training.rename(
        columns={
            "adjusted_return_40d": "adjusted_return_20d",
            "future_return_rank_pct_40d": "future_return_rank_pct_20d",
        }
    )
    bundle = build_or_load_final_rank_model(
        training_frame=phase5_training,
        training_meta={"cache_sha256": source_sha},
        model_root=model_dir,
        settings=phase5_settings,
        force=force,
    )
    _augment_manifest(model_dir, bundle=bundle, gate=gate, settings=config)
    paths = export_phase6d_reports(
        output_dir=output,
        model_dir=model_dir,
        bundle=bundle,
        training=training,
        gate=gate,
        settings=config,
    )
    return Phase6DTWSEFinalResult(
        status="PASS",
        training_rows=len(training),
        signal_dates=int(training["signal_date"].nunique()),
        output_paths=tuple(paths),
    )


def validate_phase6d_gates(output_dir: Path) -> dict[str, str]:
    phase6b_path = output_dir / "phase6b_summary.csv"
    phase6c_path = output_dir / "phase6c_summary.csv"
    candidates_path = output_dir / "phase6c_rule_candidates.csv"
    missing = [str(path.name) for path in (phase6b_path, phase6c_path, candidates_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Phase 6D 缺少必要報告：{missing}")
    phase6b = _metric_map(phase6b_path)
    phase6c = _metric_map(phase6c_path)
    if phase6b.get("return_rank_40d_validation_pass") not in {"1", "1.0"}:
        raise RuntimeError("Phase 6B TWSE 40 日模型尚未通過")
    if phase6c.get("lifecycle_validation_pass") not in {"1", "1.0"}:
        raise RuntimeError("Phase 6C TWSE 生命週期尚未通過")
    candidates = pd.read_csv(candidates_path, encoding="utf-8-sig", low_memory=False)
    required = {"universe_id", "entry_threshold", "confirmation_days", "candidate_status"}
    missing_columns = sorted(required.difference(candidates.columns))
    if missing_columns:
        raise KeyError(f"Phase 6C 候選表缺少欄位：{missing_columns}")
    selected = candidates[
        (candidates["universe_id"].astype(str) == SELECTED_LIQUIDITY_UNIVERSE)
        & (pd.to_numeric(candidates["entry_threshold"], errors="coerce") == ENTRY_PERCENTILE)
        & (pd.to_numeric(candidates["confirmation_days"], errors="coerce") == CONFIRMATION_DAYS)
        & (candidates["candidate_status"].astype(str) == "strong_candidate")
    ]
    if selected.empty:
        raise RuntimeError("Phase 6C 未找到固定的 1億元＋300張、連續3日前10% 強候選")
    return {
        "phase6b_version": phase6b.get("phase6b_version", ""),
        "phase6c_version": phase6c.get("phase6c_version", ""),
        "phase6b_validation_pass": phase6b.get("return_rank_40d_validation_pass", ""),
        "phase6c_validation_pass": phase6c.get("lifecycle_validation_pass", ""),
    }


def export_phase6d_reports(
    *,
    output_dir: Path,
    model_dir: Path,
    bundle: FinalRankModelBundle,
    training: pd.DataFrame,
    gate: dict[str, str],
    settings: Phase6DTWSEFinalSettings,
) -> list[Path]:
    coefficients = pd.DataFrame(
        {
            "feature_name": list(bundle.feature_columns),
            "coefficient": bundle.model.weights.astype(float),
        }
    )
    coefficients_path = output_dir / "phase6d_coefficients.csv"
    coefficients.to_csv(coefficients_path, index=False, encoding="utf-8-sig")
    summary_rows: list[dict[str, Any]] = [
        {"metric": "phase6d_version", "value": PHASE6D_VERSION},
        {"metric": "status", "value": "PASS"},
        {"metric": "market", "value": TARGET_MARKET},
        {"metric": "target_horizon_days", "value": TARGET_HORIZON_DAYS},
        {"metric": "training_rows", "value": len(training)},
        {"metric": "signal_dates", "value": training["signal_date"].nunique()},
        {"metric": "first_training_date", "value": training["signal_date"].min()},
        {"metric": "last_training_date", "value": training["signal_date"].max()},
        {"metric": "feature_count", "value": len(bundle.feature_columns)},
        {"metric": "liquidity_universe", "value": SELECTED_LIQUIDITY_UNIVERSE},
        {"metric": "minimum_trading_money_20d", "value": 100_000_000},
        {"metric": "minimum_trading_volume_lots_20d", "value": 300},
        {"metric": "entry_percentile", "value": ENTRY_PERCENTILE},
        {"metric": "confirmation_days", "value": CONFIRMATION_DAYS},
        {"metric": "tracking_days", "value": TRACKING_DAYS},
        {"metric": "cooldown_days", "value": COOLDOWN_DAYS},
        {"metric": "model_signature", "value": bundle.signature},
        {"metric": "source_sha256", "value": bundle.source_sha256},
        {"metric": "phase6b_validation_pass", "value": gate["phase6b_validation_pass"]},
        {"metric": "phase6c_validation_pass", "value": gate["phase6c_validation_pass"]},
        {"metric": "deployment_model_ready", "value": 1},
    ]
    summary_path = output_dir / "phase6d_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    report_zip = output_dir / "phase6d_twse_final_model_reports.zip"
    deploy_zip = output_dir / "phase6d_twse_deploy_model.zip"
    with zipfile.ZipFile(report_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(summary_path, summary_path.name)
        archive.write(coefficients_path, coefficients_path.name)
        archive.write(model_dir / "phase5d_final_rank_model.json", "models/twse/phase5d_final_rank_model.json")
        archive.write(model_dir / "phase5d_final_rank_model.npz", "models/twse/phase5d_final_rank_model.npz")
    with zipfile.ZipFile(deploy_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(model_dir / "phase5d_final_rank_model.json", "phase5d_final_rank_model.json")
        archive.write(model_dir / "phase5d_final_rank_model.npz", "phase5d_final_rank_model.npz")
    return [summary_path, coefficients_path, report_zip, deploy_zip]


def _augment_manifest(
    model_dir: Path,
    *,
    bundle: FinalRankModelBundle,
    gate: dict[str, str],
    settings: Phase6DTWSEFinalSettings,
) -> None:
    path = model_dir / "phase5d_final_rank_model.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "phase6d_version": PHASE6D_VERSION,
            "market": TARGET_MARKET,
            "target": "same_day_future_return_rank_40d",
            "target_horizon_days": TARGET_HORIZON_DAYS,
            "liquidity_universe": SELECTED_LIQUIDITY_UNIVERSE,
            "minimum_trading_money_20d": 100_000_000,
            "minimum_trading_volume_lots_20d": 300,
            "entry_percentile": ENTRY_PERCENTILE,
            "confirmation_days": CONFIRMATION_DAYS,
            "tracking_days": TRACKING_DAYS,
            "cooldown_days": COOLDOWN_DAYS,
            "phase6b_version": gate["phase6b_version"],
            "phase6c_version": gate["phase6c_version"],
            "random_seed": settings.random_seed,
            "model_signature": bundle.signature,
        }
    )
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _frame_sha256(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _metric_map(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    return dict(zip(frame["metric"].astype(str), frame["value"].astype(str), strict=True))
