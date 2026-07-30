from __future__ import annotations

import pytest

from src.config_loader import (
    ConfigError,
    institutional_candidates_settings,
    validate_user_config,
)


def _config() -> dict:
    return {
        "user": {"id": "example"},
        "stocks": [
            {
                "symbol": "2330",
                "average_cost": 620,
                "profit_alerts": [30, 50, 75],
            }
        ],
    }


def test_minimal_holding_config_is_valid() -> None:
    config = _config()
    validate_user_config(config)
    assert institutional_candidates_settings(config) == {
        "enabled": True,
        "max_new_candidates": 6,
        "include_state_updates": True,
    }


def test_cost_and_profit_alerts_are_required() -> None:
    config = _config()
    del config["stocks"][0]["average_cost"]
    with pytest.raises(ConfigError, match="average_cost"):
        validate_user_config(config)

    config = _config()
    config["stocks"][0]["profit_alerts"] = []
    with pytest.raises(ConfigError, match="profit_alerts"):
        validate_user_config(config)


def test_old_scoring_fields_are_rejected() -> None:
    config = _config()
    config["stocks"][0]["peers"] = ["2303"]
    with pytest.raises(ConfigError, match="不再支援"):
        validate_user_config(config)


def test_profit_alerts_must_be_unique_and_sorted() -> None:
    config = _config()
    config["stocks"][0]["profit_alerts"] = [50, 30]
    with pytest.raises(ConfigError, match="由小到大"):
        validate_user_config(config)

    config = _config()
    config["stocks"][0]["profit_alerts"] = [30, 30]
    with pytest.raises(ConfigError, match="不可重複"):
        validate_user_config(config)
