from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np

from research.institutional_model.phase3_dataset import FEATURE_COLUMNS
from research.institutional_model.phase4_model import (
    Phase4Settings,
    fit_preprocessor,
    fit_temperature,
    run_phase4a_rolling_baseline,
    softmax,
)


def test_phase4_preprocessor_uses_only_training_years() -> None:
    rng = np.random.default_rng(42)
    features = rng.normal(size=(200, len(FEATURE_COLUMNS))).astype(np.float32)
    years = np.array([2015] * 100 + [2020] * 100, dtype=np.int16)

    first = fit_preprocessor(
        features=features,
        years=years,
        train_start_year=2015,
        train_end_year=2015,
        sample_size=100,
        lower_quantile=0.01,
        upper_quantile=0.99,
        chunk_size=37,
        seed=7,
    )
    modified = features.copy()
    modified[100:] = 1_000_000
    second = fit_preprocessor(
        features=modified,
        years=years,
        train_start_year=2015,
        train_end_year=2015,
        sample_size=100,
        lower_quantile=0.01,
        upper_quantile=0.99,
        chunk_size=37,
        seed=7,
    )

    np.testing.assert_allclose(first.lower, second.lower)
    np.testing.assert_allclose(first.upper, second.upper)
    np.testing.assert_allclose(first.mean, second.mean)
    np.testing.assert_allclose(first.std, second.std)


def test_temperature_scaling_never_worsens_calibration_loss() -> None:
    logits = np.array(
        [
            [8.0, 1.0, 0.0],
            [7.0, 2.0, 0.0],
            [0.0, 8.0, 1.0],
            [0.0, 7.0, 2.0],
            [1.0, 0.0, 8.0],
            [2.0, 0.0, 7.0],
        ]
    )
    labels = np.array([0, 1, 1, 2, 2, 0], dtype=np.uint8)

    temperature, raw_loss, scaled_loss = fit_temperature(logits, labels)

    assert 0.25 <= temperature <= 4.0
    assert scaled_loss <= raw_loss + 1e-12
    np.testing.assert_allclose(softmax(logits / temperature).sum(axis=1), 1.0)


def test_phase4_rolling_pipeline_builds_reports_and_resumes(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    _write_phase3b_ready(output)
    manifest_rows = []
    for market, seed in (("twse", 11), ("tpex", 22)):
        path = output / f"phase3_training_{market}.csv.gz"
        row_count = _write_training_file(path, market=market, seed=seed)
        manifest_rows.append(
            {
                "file_name": path.name,
                "market_type": market,
                "row_count": row_count,
                "size_bytes": path.stat().st_size,
                "sha256": f"synthetic-{market}",
                "config_signature": "test",
                "label_rule_version": "rounded-return-10dp-v1",
            }
        )
    _write_csv(output / "phase3_dataset_manifest.csv", manifest_rows)

    settings = Phase4Settings(
        first_test_year=2019,
        minimum_training_years=3,
        calibration_years=1,
        quantile_sample_size=300,
        cache_chunk_size=173,
        training_chunk_size=173,
        batch_size=64,
        maximum_epochs=6,
        minimum_epochs=2,
        early_stopping_patience=2,
        learning_rate=0.03,
        random_seed=123,
    )
    first = run_phase4a_rolling_baseline(
        output_dir=output,
        cache_root=tmp_path / "cache",
        run_root=tmp_path / "runs",
        settings=settings,
    )

    assert first.status == "PASS"
    assert first.completed_folds == first.expected_folds == 6
    assert first.failed_folds == 0
    assert (output / "phase4a_validation_reports.zip").exists()
    assert (output / "phase4a_metrics.csv").exists()
    assert (output / "phase4a_index_deciles.csv").exists()
    with (output / "phase4a_summary.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        summary = {row["metric"]: row["value"] for row in csv.DictReader(handle)}
    assert float(summary["twse_log_loss_improvement"]) > 0
    assert float(summary["tpex_log_loss_improvement"]) > 0

    second = run_phase4a_rolling_baseline(
        output_dir=output,
        cache_root=tmp_path / "cache",
        run_root=tmp_path / "runs",
        settings=settings,
    )
    assert second.status == "PASS"
    assert second.completed_folds == 6


def _write_phase3b_ready(output: Path) -> None:
    _write_csv(
        output / "phase3b_summary.csv",
        [
            {"metric": "status", "value": "PASS"},
            {"metric": "ready_for_modeling", "value": "1"},
        ],
    )


def _write_training_file(path: Path, *, market: str, seed: int) -> int:
    rng = np.random.default_rng(seed)
    rows = []
    for year in range(2015, 2022):
        for index in range(120):
            latent = rng.normal()
            noisy_signal = latent + rng.normal(scale=0.35)
            if noisy_signal > 0.45:
                label = "UP"
                adjusted_return = 0.08
            elif noisy_signal < -0.45:
                label = "DOWN"
                adjusted_return = -0.08
            else:
                label = "FLAT"
                adjusted_return = 0.0
            row = {
                "signal_year": year,
                "label_10d": label,
                "adjusted_return_10d": adjusted_return,
            }
            for feature_index, feature in enumerate(FEATURE_COLUMNS):
                if feature_index == 0:
                    value = latent
                elif feature_index == 1:
                    value = latent * 0.5 + rng.normal(scale=0.5)
                else:
                    value = rng.normal(scale=1 + feature_index / 100)
                row[feature] = value
            rows.append(row)
    with gzip.open(path, "wt", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
