from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np

from research.institutional_model.phase3_dataset import FEATURE_COLUMNS
from research.institutional_model.phase4_model import Phase4Settings
from research.institutional_model.phase4_stability import (
    CORE_FEATURE_COLUMNS,
    blend_probabilities,
    build_confirmation_decisions,
    fit_prior_blend,
    run_phase4b_stability_research,
    select_development_candidates,
)


def test_core_feature_set_is_explicit_subset() -> None:
    assert len(CORE_FEATURE_COLUMNS) == 22
    assert len(set(CORE_FEATURE_COLUMNS)) == 22
    assert set(CORE_FEATURE_COLUMNS).issubset(FEATURE_COLUMNS)
    assert not any(name.startswith("selected_total_flow_pct") for name in CORE_FEATURE_COLUMNS)


def test_prior_blend_grid_never_worse_than_raw_or_history_on_calibration() -> None:
    probabilities = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.60, 0.30, 0.10],
            [0.10, 0.20, 0.70],
            [0.10, 0.30, 0.60],
        ]
    )
    labels = np.array([0, 1, 2, 1], dtype=np.uint8)
    priors = np.array([0.25, 0.50, 0.25])

    alpha, selected_loss = fit_prior_blend(
        probabilities=probabilities,
        labels=labels,
        priors=priors,
    )
    raw = blend_probabilities(probabilities, priors, 0.0)
    history = blend_probabilities(probabilities, priors, 1.0)

    assert 0.0 <= alpha <= 1.0
    from research.institutional_model.phase4_model import multiclass_log_loss

    assert selected_loss <= multiclass_log_loss(labels, raw) + 1e-12
    assert selected_loss <= multiclass_log_loss(labels, history) + 1e-12


def test_candidate_selection_and_confirmation_are_market_specific() -> None:
    development = []
    for candidate, spreads in (
        ("full40_l2_1e-4", [0.001, -0.001, 0.001, -0.001]),
        ("core22_l2_1e-3", [0.005, 0.004, 0.006, 0.003]),
    ):
        for year, spread in zip(range(2019, 2023), spreads, strict=True):
            development.append(
                _fake_result(
                    market="tpex",
                    year=year,
                    candidate=candidate,
                    spread=spread,
                    model_loss=1.0,
                    history_loss=1.01,
                )
            )
    selection = select_development_candidates(development)
    assert selection["tpex"]["selected_candidate_id"] == "core22_l2_1e-3"

    confirmation = [
        _fake_result(
            market="tpex",
            year=year,
            candidate="core22_l2_1e-3",
            spread=0.004,
            model_loss=1.0,
            history_loss=1.01,
        )
        for year in range(2023, 2027)
    ]
    decisions = build_confirmation_decisions(
        selections=selection,
        confirmation_results=confirmation,
    )
    assert decisions[0]["decision"] == "PROBABILITY_AND_RANKING"


def test_phase4b_pipeline_builds_reports_and_resumes(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    _write_csv(
        output / "phase3b_summary.csv",
        [
            {"metric": "status", "value": "PASS"},
            {"metric": "ready_for_modeling", "value": "1"},
        ],
    )
    training = output / "phase3_training_tpex.csv.gz"
    row_count = _write_training_file(training)
    _write_csv(
        output / "phase3_dataset_manifest.csv",
        [
            {
                "file_name": training.name,
                "market_type": "tpex",
                "row_count": row_count,
                "size_bytes": training.stat().st_size,
                "sha256": "phase4b-synthetic",
                "config_signature": "test",
                "label_rule_version": "rounded-return-10dp-v1",
            }
        ],
    )
    settings = Phase4Settings(
        first_test_year=2019,
        minimum_training_years=3,
        calibration_years=1,
        quantile_sample_size=500,
        cache_chunk_size=137,
        training_chunk_size=137,
        batch_size=64,
        maximum_epochs=2,
        minimum_epochs=1,
        early_stopping_patience=1,
        learning_rate=0.03,
        random_seed=321,
    )

    first = run_phase4b_stability_research(
        output_dir=output,
        cache_root=tmp_path / "cache",
        phase4a_run_root=tmp_path / "phase4a-runs",
        run_root=tmp_path / "phase4b-runs",
        settings=settings,
        markets=("tpex",),
    )
    assert first.status == "PASS"
    assert first.completed_candidate_folds == first.expected_candidate_folds == 16
    assert first.failed_candidate_folds == 0
    assert (output / "phase4b_validation_reports.zip").exists()
    assert (output / "phase4b_market_decisions.csv").exists()

    second = run_phase4b_stability_research(
        output_dir=output,
        cache_root=tmp_path / "cache",
        phase4a_run_root=tmp_path / "phase4a-runs",
        run_root=tmp_path / "phase4b-runs",
        settings=settings,
        markets=("tpex",),
    )
    assert second.status == "PASS"
    assert second.completed_candidate_folds == 16


def _fake_result(
    *,
    market: str,
    year: int,
    candidate: str,
    spread: float,
    model_loss: float,
    history_loss: float,
) -> dict[str, object]:
    return {
        "fold_summary": {
            "market": market,
            "test_year": year,
            "candidate_id": candidate,
            "feature_set": "core22" if candidate.startswith("core22") else "full40",
            "feature_count": 22 if candidate.startswith("core22") else 40,
            "l2_penalty": 0.001,
            "test_rows": 100,
            "high_minus_low_return": spread,
            "raw_test_log_loss": model_loss,
            "temperature_test_log_loss": model_loss + 0.001,
            "prior_blend_test_log_loss": model_loss,
            "historical_test_log_loss": history_loss,
            "raw_test_ece": 0.03,
            "temperature_test_ece": 0.04,
            "prior_blend_test_ece": 0.03,
            "historical_test_ece": 0.04,
        }
    }


def _write_training_file(path: Path) -> int:
    rng = np.random.default_rng(456)
    rows = []
    for year in range(2015, 2027):
        for _ in range(90):
            latent = rng.normal()
            noisy = latent + rng.normal(scale=0.4)
            if noisy > 0.45:
                label = "UP"
                adjusted_return = 0.08 + latent * 0.005
            elif noisy < -0.45:
                label = "DOWN"
                adjusted_return = -0.08 + latent * 0.005
            else:
                label = "FLAT"
                adjusted_return = latent * 0.005
            row = {
                "signal_year": year,
                "label_10d": label,
                "adjusted_return_10d": adjusted_return,
            }
            for index, feature in enumerate(FEATURE_COLUMNS):
                if feature == "foreign_flow_pct_1d":
                    value = latent
                elif feature == "investment_trust_flow_pct_5d":
                    value = latent * 0.5 + rng.normal(scale=0.4)
                else:
                    value = rng.normal()
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
