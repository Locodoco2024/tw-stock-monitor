from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class RuleResult:
    rule_id: str
    module: str
    score: float
    message: str
    values: dict[str, Any] = field(default_factory=dict)
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ModuleResult:
    module: str
    weight: float
    score: float
    available_weight: float
    rules: list[RuleResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def opportunity(self) -> float:
        return sum(max(0.0, rule.score) for rule in self.rules)

    @property
    def risk(self) -> float:
        return sum(abs(min(0.0, rule.score)) for rule in self.rules)

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "weight": self.weight,
            "score": self.score,
            "available_weight": self.available_weight,
            "opportunity": self.opportunity,
            "risk": self.risk,
            "notes": self.notes,
            "rules": [rule.to_dict() for rule in self.rules],
        }


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
class OfficialEvent:
    event_id: str
    symbol: str
    company_name: str
    title: str
    description: str
    published_at: str
    source: str
    source_url: str | None = None

    def combined_text(self) -> str:
        return f"{self.title}\n{self.description}".strip()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StockInputBundle:
    quote: Quote | None
    resolved_name: str
    resolved_market: str
    benchmark: str
    prices: list[dict[str, Any]]
    market_prices: list[dict[str, Any]]
    peer_prices: dict[str, list[dict[str, Any]]]
    monthly_revenue: list[dict[str, Any]]
    peer_monthly_revenue: dict[str, list[dict[str, Any]]]
    financial_statements: list[dict[str, Any]]
    events: list[OfficialEvent]
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AnalysisResult:
    user_id: str
    symbol: str
    name: str
    market: str
    analyzed_at: str
    quote: Quote | None
    objective_score: float
    operation_score: float
    opportunity_score: float
    risk_score: float
    completeness: float
    objective_label: str
    operation_label: str
    holding_enabled: bool
    price_return_pct: float | None
    modules: list[ModuleResult]
    holding_rules: list[RuleResult]
    summary: str
    new_event_ids: list[str]
    errors: list[str]
    report_path: str | None = None
    report_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "symbol": self.symbol,
            "name": self.name,
            "market": self.market,
            "analyzed_at": self.analyzed_at,
            "quote": self.quote.to_dict() if self.quote else None,
            "objective_score": self.objective_score,
            "operation_score": self.operation_score,
            "opportunity_score": self.opportunity_score,
            "risk_score": self.risk_score,
            "completeness": self.completeness,
            "objective_label": self.objective_label,
            "operation_label": self.operation_label,
            "holding_enabled": self.holding_enabled,
            "price_return_pct": self.price_return_pct,
            "modules": [module.to_dict() for module in self.modules],
            "holding_rules": [rule.to_dict() for rule in self.holding_rules],
            "summary": self.summary,
            "new_event_ids": self.new_event_ids,
            "errors": self.errors,
            "report_path": self.report_path,
            "report_url": self.report_url,
        }


@dataclass(slots=True)
class NotificationDecision:
    should_send: bool
    reasons: list[str]


@dataclass(slots=True)
class StateRecord:
    operation_score: float
    operation_label: str
    risk_score: float
    notified_at: str | None
    processed_event_ids: list[str]
    triggered_rule_ids: list[str] = field(default_factory=list)

    @classmethod
    def empty(cls) -> "StateRecord":
        return cls(
            operation_score=0.0,
            operation_label="",
            risk_score=0.0,
            notified_at=None,
            processed_event_ids=[],
            triggered_rule_ids=[],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AppState:
    records: dict[str, StateRecord] = field(default_factory=dict)
    updated_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat())
