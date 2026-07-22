from __future__ import annotations

import pytest

from src.config_loader import ConfigError, validate_user_config


def test_symbol_is_the_only_required_stock_identity_field() -> None:
    validate_user_config(
        {
            "user": {"id": "example"},
            "stocks": [{"symbol": "2330"}],
        }
    )


def test_holding_cost_is_required_only_when_holding_is_enabled() -> None:
    with pytest.raises(ConfigError, match="average_cost"):
        validate_user_config(
            {
                "user": {"id": "example"},
                "stocks": [{"symbol": "2330", "holding": {"enabled": True}}],
            }
        )
