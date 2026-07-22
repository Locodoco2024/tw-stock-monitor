from __future__ import annotations

from datetime import datetime
from typing import Any

from src.models import AnalysisResult, ModuleResult, StockInputBundle
from src.scoring.company import score_company_capacity
from src.scoring.events import score_official_events
from src.scoring.holding import score_holding
from src.scoring.industry import score_industry_peers
from src.scoring.labels import objective_label, operation_label
from src.scoring.market_environment import score_market_environment
from src.scoring.market_pricing import score_market_pricing
from src.scoring.summary import build_summary


class ScoringEngine:
    def __init__(self, scoring: dict[str, Any]) -> None:
        self.scoring = scoring

    def analyze(
        self,
        user_id: str,
        stock: dict[str, Any],
        bundle: StockInputBundle,
        processed_event_ids: set[str] | None = None,
    ) -> AnalysisResult:
        processed_event_ids = processed_event_ids or set()
        modules_config = self.scoring.get("modules", {})

        company_module, trailing_revenue = score_company_capacity(
            bundle.monthly_revenue,
            bundle.financial_statements,
            self._weight(modules_config, "company_capacity"),
        )
        official_available = not any("官方重大訊息" in error for error in bundle.errors)
        event_module, _matched_event_ids = score_official_events(
            bundle.events,
            self.scoring,
            self._weight(modules_config, "official_events"),
            official_available,
            trailing_revenue,
        )
        pricing_module, pricing_context = score_market_pricing(
            bundle.prices,
            bundle.market_prices,
            bundle.peer_prices,
            bundle.quote,
            self.scoring,
            self._weight(modules_config, "market_pricing"),
        )
        industry_module = score_industry_peers(
            bundle.peer_prices,
            bundle.peer_monthly_revenue,
            self._weight(modules_config, "industry_peers"),
        )
        market_module = score_market_environment(
            bundle.market_prices,
            self._weight(modules_config, "market_environment"),
        )
        modules: list[ModuleResult] = [
            event_module,
            pricing_module,
            industry_module,
            company_module,
            market_module,
        ]

        raw_objective = sum(module.score for module in modules)
        objective_score = max(-100.0, min(100.0, raw_objective))
        completeness = self._completeness(modules)
        objective_score = self._apply_completeness_cap(objective_score, completeness)

        current_price = bundle.quote.price if bundle.quote else pricing_context.get("current_price")
        has_negative_official_event = any(rule.score < 0 for rule in event_module.rules)
        holding_rules, price_return_pct = score_holding(
            stock,
            current_price,
            objective_score,
            completeness,
            self.scoring,
            has_negative_official_event,
        )
        holding_enabled = bool(stock.get("holding", {}).get("enabled"))
        operation_raw = objective_score + sum(rule.score for rule in holding_rules)
        operation_score = max(-100.0, min(100.0, operation_raw))
        operation_score = self._apply_completeness_cap(operation_score, completeness)

        opportunity_score = min(
            100.0,
            sum(max(0.0, module.score) for module in modules)
            + sum(max(0.0, rule.score) for rule in holding_rules),
        )
        risk_score = min(
            100.0,
            sum(abs(min(0.0, module.score)) for module in modules)
            + sum(abs(min(0.0, rule.score)) for rule in holding_rules),
        )

        labels = self.scoring.get("labels", {})
        new_event_ids = [event.event_id for event in bundle.events if event.event_id not in processed_event_ids]
        result = AnalysisResult(
            user_id=user_id,
            symbol=str(stock["symbol"]),
            name=bundle.resolved_name,
            market=bundle.resolved_market,
            analyzed_at=datetime.now().astimezone().isoformat(),
            quote=bundle.quote,
            objective_score=round(objective_score, 1),
            operation_score=round(operation_score, 1),
            opportunity_score=round(opportunity_score, 1),
            risk_score=round(risk_score, 1),
            completeness=round(completeness, 1),
            objective_label=objective_label(objective_score, labels),
            operation_label=operation_label(operation_score, holding_enabled, labels),
            holding_enabled=holding_enabled,
            price_return_pct=round(price_return_pct, 2) if price_return_pct is not None else None,
            modules=modules,
            holding_rules=holding_rules,
            summary=build_summary(modules, holding_rules, completeness),
            new_event_ids=new_event_ids,
            errors=bundle.errors,
        )
        return result

    @staticmethod
    def _weight(modules_config: dict[str, Any], name: str) -> float:
        return float(modules_config.get(name, {}).get("weight", 0))

    @staticmethod
    def _completeness(modules: list[ModuleResult]) -> float:
        expected = sum(module.weight for module in modules)
        available = sum(module.available_weight for module in modules)
        return available / expected * 100 if expected else 0.0

    def _apply_completeness_cap(self, score: float, completeness: float) -> float:
        config = self.scoring.get("completeness", {})
        capped_minimum = float(config.get("capped_minimum", 40))
        warning_minimum = float(config.get("warning_minimum", 60))
        capped_score = float(config.get("capped_score", 39))
        if completeness < capped_minimum:
            return 0.0
        if completeness < warning_minimum:
            return max(-capped_score, min(capped_score, score))
        return score
