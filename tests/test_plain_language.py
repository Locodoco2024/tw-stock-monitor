from __future__ import annotations

from src.models import ModuleResult, RuleResult
from src.scoring.plain_language import plain_message
from src.scoring.summary import build_summary


def test_plain_message_hides_technical_indicator_names() -> None:
    rule = RuleResult(
        rule_id="pricing.ema20_vs_ema60",
        module="market_pricing",
        score=-4,
        message="EMA20 低於 EMA60",
        values={"ema20": 98, "ema60": 102},
    )
    message = plain_message(rule)
    assert "EMA" not in message
    assert "中期" in message


def test_neutral_summary_explains_mixed_trend_in_plain_language() -> None:
    module = ModuleResult(
        module="market_pricing",
        weight=25,
        score=-1,
        available_weight=25,
        rules=[
            RuleResult(
                rule_id="pricing.price_vs_ema20",
                module="market_pricing",
                score=3,
                message="目前價格高於 EMA20",
            ),
            RuleResult(
                rule_id="pricing.ema20_vs_ema60",
                module="market_pricing",
                score=-4,
                message="EMA20 低於 EMA60",
            ),
        ],
    )
    summary = build_summary([module], [], 100, -1, True)
    assert "EMA" not in summary
    assert "短線略有回穩" in summary
    assert "明顯加碼或減碼" in summary


def test_profit_take_summary_explains_reason_for_reduction() -> None:
    holding_rule = RuleResult(
        rule_id="holding.profit_take",
        module="holding",
        score=-30,
        message="價格報酬已達 +52.0%",
        values={"return_pct": 52.0, "threshold_pct": 50},
    )
    summary = build_summary([], [holding_rule], 100, 28, True)
    assert "停利區間" in summary
    assert "分批停利" in summary
    assert "營運明顯惡化" in summary


def test_html_report_keeps_technical_terms_collapsed() -> None:
    from src.models import AnalysisResult
    from src.reports.html_report import render_stock_report

    module = ModuleResult(
        module="market_pricing",
        weight=25,
        score=3,
        available_weight=25,
        rules=[
            RuleResult(
                rule_id="pricing.price_vs_ema20",
                module="market_pricing",
                score=3,
                message="目前價格高於 EMA20",
            )
        ],
    )
    result = AnalysisResult(
        user_id="test",
        symbol="2330",
        name="台積電",
        market="TWSE",
        analyzed_at="2026-07-22T09:00:00+08:00",
        quote=None,
        objective_score=3,
        operation_score=3,
        opportunity_score=3,
        risk_score=0,
        completeness=100,
        objective_label="中性觀察",
        operation_label="續抱觀察",
        holding_enabled=True,
        price_return_pct=None,
        modules=[module],
        holding_rules=[],
        summary="股價短線偏穩，但目前沒有明確方向。",
        new_event_ids=[],
        errors=[],
    )
    document = render_stock_report(result)
    assert "分析涵蓋度" in document
    assert "+3 / 100" in document
    assert "查看計分細節與技術數值" in document
    assert "股價仍守在近期平均水準之上" in document
