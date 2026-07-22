from __future__ import annotations

from pathlib import Path

from src.config_loader import load_yaml
from src.providers.fixture import load_fixture_bundle
from src.scoring.engine import ScoringEngine


ROOT = Path(__file__).resolve().parents[1]


def test_engine_generates_traceable_result() -> None:
    scoring = load_yaml(ROOT / "configs/scoring.yaml")
    user = load_yaml(ROOT / "configs/users/example.yaml")
    stock = user["stocks"][0]
    bundle = load_fixture_bundle(
        ROOT / "tests/fixtures/sample_bundle.json",
        "2330",
        stock,
    )
    result = ScoringEngine(scoring).analyze("example", stock, bundle, set())
    assert -100 <= result.objective_score <= 100
    assert -100 <= result.operation_score <= 100
    assert result.completeness == 100
    assert result.new_event_ids == ["fixture-order-2330"]
    assert result.name == "台積電"
    assert result.market == "TWSE"
    assert result.summary
