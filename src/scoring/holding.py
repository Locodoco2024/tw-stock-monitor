from __future__ import annotations

from typing import Any

from src.models import RuleResult


def score_holding(
    stock: dict[str, Any],
    current_price: float | None,
    objective_score: float,
    completeness: float,
    scoring: dict[str, Any],
    has_negative_official_event: bool,
) -> tuple[list[RuleResult], float | None]:
    holding = stock.get("holding", {})
    if not holding.get("enabled"):
        return [], None
    average_cost = float(holding["average_cost"])
    if current_price is None or average_cost <= 0:
        return [
            RuleResult(
                rule_id="holding.price_unavailable",
                module="holding",
                score=0,
                message="目前價格不可用，無法計算持倉價格報酬率",
            )
        ], None

    return_pct = (current_price / average_cost - 1) * 100
    rules: list[RuleResult] = []
    profit_take = stock.get("profit_take", {})
    alert_at = float(profit_take.get("alert_at_pct", 50))
    strong_at = float(profit_take.get("strong_alert_at_pct", 75))
    if return_pct >= strong_at:
        rules.append(
            RuleResult(
                rule_id="holding.strong_profit_take",
                module="holding",
                score=-45,
                message=(
                    f"價格報酬已達 {return_pct:+.1f}%，超過設定的 {strong_at:.0f}% 強停利提醒"
                ),
                values={"return_pct": return_pct, "threshold_pct": strong_at},
            )
        )
    elif return_pct >= alert_at:
        rules.append(
            RuleResult(
                rule_id="holding.profit_take",
                module="holding",
                score=-30,
                message=f"價格報酬已達 {return_pct:+.1f}%，超過設定的 {alert_at:.0f}% 停利提醒",
                values={"return_pct": return_pct, "threshold_pct": alert_at},
            )
        )
    elif return_pct >= 30:
        rules.append(
            RuleResult(
                rule_id="holding.profit_watch",
                module="holding",
                score=-10,
                message=f"價格報酬已達 {return_pct:+.1f}%，進入停利觀察區",
                values={"return_pct": return_pct},
            )
        )

    add_on = stock.get("add_on", {})
    if add_on.get("enabled") and return_pct < 0:
        loss_pct = abs(return_pct)
        thresholds = sorted(
            [float(item) for item in add_on.get("alert_at_loss_pct", [10, 20, 30])],
            reverse=True,
        )
        minimum_score = float(add_on.get("minimum_analysis_score", 40))
        maximum_adjustment = float(add_on.get("maximum_adjustment", 10))
        eligible = (
            objective_score >= minimum_score
            and completeness >= 80
            and not has_negative_official_event
        )
        matched_threshold = next((item for item in thresholds if loss_pct >= item), None)
        if matched_threshold is not None:
            if eligible:
                adjustment = min(maximum_adjustment, 5 + max(0, (matched_threshold - 10) / 10 * 3))
                rules.append(
                    RuleResult(
                        rule_id="holding.conditional_add_on",
                        module="holding",
                        score=adjustment,
                        message=(
                            f"價格低於成本 {loss_pct:.1f}%，且個股客觀分析仍符合條件，"
                            "列為分批加碼觀察"
                        ),
                        values={
                            "loss_pct": loss_pct,
                            "objective_score": objective_score,
                            "minimum_analysis_score": minimum_score,
                        },
                    )
                )
            else:
                reasons = []
                if objective_score < minimum_score:
                    reasons.append("客觀分析未達門檻")
                if completeness < 80:
                    reasons.append("資料完整度不足")
                if has_negative_official_event:
                    reasons.append("存在官方負面事件")
                rules.append(
                    RuleResult(
                        rule_id="holding.add_on_not_eligible",
                        module="holding",
                        score=0,
                        message=(
                            f"價格低於成本 {loss_pct:.1f}%，但不符合加碼條件："
                            + "、".join(reasons)
                        ),
                        values={"loss_pct": loss_pct},
                    )
                )
    return rules, return_pct
