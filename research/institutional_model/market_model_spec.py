from __future__ import annotations

from dataclasses import dataclass

from research.institutional_model.phase3_dataset import FEATURE_COLUMNS
from research.institutional_model.phase4_stability import CORE_FEATURE_COLUMNS


@dataclass(frozen=True)
class MarketModelSpec:
    market: str
    candidate_id: str
    feature_columns: tuple[str, ...]
    l2_penalty: float


_MARKET_MODEL_SPECS = {
    "tpex": MarketModelSpec(
        market="tpex",
        candidate_id="core22_l2_1e-3",
        feature_columns=tuple(CORE_FEATURE_COLUMNS),
        l2_penalty=0.001,
    ),
    "twse": MarketModelSpec(
        market="twse",
        candidate_id="full40_l2_1e-3",
        feature_columns=tuple(FEATURE_COLUMNS),
        l2_penalty=0.001,
    ),
}


def market_model_spec(market: str) -> MarketModelSpec:
    normalized = str(market).lower()
    try:
        return _MARKET_MODEL_SPECS[normalized]
    except KeyError as exc:
        raise ValueError(f"不支援的法人模型市場：{market}") from exc
