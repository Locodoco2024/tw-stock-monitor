from __future__ import annotations

from research.institutional_model.adjusted_returns import calculate_holding_return
from research.institutional_model.corporate_actions import CorporateAction


def run_self_check() -> None:
    prices = [
        {"date": "2024-06-21", "open": 100.0, "close": 102.0},
        {"date": "2024-06-24", "open": 97.0, "close": 98.0},
        {"date": "2024-06-25", "open": 98.5, "close": 99.0},
    ]
    actions = {
        "2024-06-24": [
            CorporateAction(
                date="2024-06-24",
                action_type="dividend",
                reference_price=97.0,
                description="除息",
                source_dataset="synthetic",
            )
        ]
    }
    result = calculate_holding_return(prices, actions)
    expected_adjusted = (102 / 100) * (98 / 97) * (99 / 98) - 1
    assert abs(result.raw_return - (-0.01)) < 1e-12
    assert abs(result.adjusted_return - expected_adjusted) < 1e-12
    assert result.adjusted_return > 0.04
    print("自我檢查通過：除息機械式跌幅已由官方參考價還原。")
