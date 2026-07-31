from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class Quote:
    symbol: str
    name: str
    price: float
    previous_close: float | None
    open_price: float | None
    high_price: float | None
    low_price: float | None
    change_percent: float | None
    trade_volume: float | None
    as_of: str | None
    exchange: str | None = None
    market: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class HoldingMonitorResult:
    user_id: str
    symbol: str
    name: str
    analyzed_at: str
    quote: Quote | None
    average_cost: float
    profit_return_pct: float | None
    active_profit_thresholds: list[float]
    new_profit_thresholds: list[float]
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "symbol": self.symbol,
            "name": self.name,
            "analyzed_at": self.analyzed_at,
            "quote": self.quote.to_dict() if self.quote else None,
            "average_cost": self.average_cost,
            "profit_return_pct": self.profit_return_pct,
            "active_profit_thresholds": self.active_profit_thresholds,
            "new_profit_thresholds": self.new_profit_thresholds,
            "errors": self.errors,
        }


@dataclass(slots=True, frozen=True)
class InstitutionalNotification:
    user_id: str
    symbol: str
    name: str
    signal_date: str
    event_id: str
    notification_type: str
    notify_mode: str
    percentile: float | None
    event_age_days: int | None
    reason: str
    positive_factors: str
    negative_factors: str
    is_configured_stock: bool
    trade_action: str = "TRACK_ONLY"
    estimated_cost_window_days: int | None = None
    estimated_cost_low: float | None = None
    estimated_cost_mid: float | None = None
    estimated_cost_high: float | None = None
    estimated_cost_buy_days: int | None = None
    signal_close: float | None = None
    signal_deviation_pct: float | None = None

    @property
    def state_key(self) -> str:
        return "|".join(
            [
                self.user_id,
                self.event_id,
                self.notification_type,
                self.signal_date,
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class HoldingStateRecord:
    active_profit_thresholds: list[float] = field(default_factory=list)
    migrated_from_legacy: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AppState:
    records: dict[str, HoldingStateRecord] = field(default_factory=dict)
    institutional_notification_keys: dict[str, list[str]] = field(default_factory=dict)
    updated_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat())
