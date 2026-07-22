from __future__ import annotations

from src.models import ModuleResult, RuleResult


MODULE_NAMES = {
    "official_events": "官方事件",
    "market_pricing": "股價反應",
    "industry_peers": "產業與同業",
    "company_capacity": "公司營運",
    "market_environment": "大盤環境",
    "holding": "持倉策略",
}


def plain_message(rule: RuleResult) -> str:
    """Convert a traceable scoring rule into wording for non-technical readers."""
    values = rule.values
    score = rule.score

    if rule.rule_id == "holding.strong_profit_take":
        return (
            f"目前獲利已達 {_signed(values.get('return_pct'))}%，超過你設定的強停利門檻，"
            "應優先評估分批落袋"
        )
    if rule.rule_id == "holding.profit_take":
        return (
            f"目前獲利已達 {_signed(values.get('return_pct'))}%，"
            "已進入你設定的停利區間"
        )
    if rule.rule_id == "holding.profit_watch":
        return (
            f"目前獲利已達 {_signed(values.get('return_pct'))}%，"
            "已接近停利區間，可開始留意分批落袋時機"
        )
    if rule.rule_id == "holding.conditional_add_on":
        return (
            f"股價目前低於成本 {_number(values.get('loss_pct'))}%，"
            "但公司與市場分析仍符合加碼門檻，因此列為分批加碼觀察"
        )
    if rule.rule_id == "holding.add_on_not_eligible":
        return (
            f"股價目前低於成本 {_number(values.get('loss_pct'))}%，"
            "但現有條件還不足以支持加碼"
        )
    if rule.rule_id == "holding.price_unavailable":
        return "目前缺少可用股價，暫時無法計算持倉損益"

    if rule.rule_id == "pricing.price_vs_ema20":
        return (
            "股價仍守在近期平均水準之上，短線偏穩"
            if score > 0
            else "股價已跌到近期平均水準之下，短線轉弱"
        )
    if rule.rule_id == "pricing.ema20_vs_ema60":
        return (
            "近期走勢比過去幾個月更強，中期動能改善"
            if score > 0
            else "近期走勢仍弱於過去幾個月，中期尚未明顯轉強"
        )
    if rule.rule_id == "pricing.ema60_vs_ema120":
        return (
            "中長期走勢仍維持正向"
            if score > 0
            else "中長期走勢仍承受壓力"
        )
    if rule.rule_id == "pricing.relative_return_5d":
        relative = abs(_float(values.get("relative_return_5d_pct")))
        basis = str(values.get("comparison_basis") or "大盤與同業")
        return (
            f"近五日表現比{basis}強 {relative:.1f} 個百分點"
            if score > 0
            else f"近五日表現比{basis}弱 {relative:.1f} 個百分點"
        )
    if rule.rule_id == "pricing.already_reflected":
        relative = _number(values.get("relative_return_5d_pct"))
        basis = str(values.get("comparison_basis") or "大盤與同業")
        return f"近五日已明顯領先{basis} {relative}%，部分正面因素可能已提前反映"
    if rule.rule_id == "pricing.extension_from_ema20":
        return "股價近期漲得較快，已明顯偏離平均水準，現在追價風險較高"
    if rule.rule_id == "pricing.volume_confirmation":
        return "上漲同時伴隨成交量放大，市場參與度提高"
    if rule.rule_id == "pricing.volume_decline":
        return "下跌時成交量放大，賣壓較明顯"
    if rule.rule_id == "pricing.rsi_moderate":
        return "近期買盤動能穩定，尚未明顯過熱"
    if rule.rule_id == "pricing.rsi_overheated":
        return "近期漲勢偏急，短線追價風險提高"

    if rule.rule_id == "industry.peer_median_return_20d":
        value = _signed(values.get("peer_return_20d_median_pct"))
        return (
            f"同業近一個月平均表現約為 {value}%，產業走勢偏強"
            if score > 0
            else f"同業近一個月平均表現約為 {value}%，產業走勢偏弱"
        )
    if rule.rule_id == "industry.breadth_ma20":
        ratio = _number(values.get("breadth_ma20_pct"), 0)
        return (
            f"約 {ratio}% 的同業近期走勢偏強"
            if score > 0
            else f"僅約 {ratio}% 的同業近期走勢偏強，多數公司仍偏弱"
        )
    if rule.rule_id == "industry.breadth_ma60":
        ratio = _number(values.get("breadth_ma60_pct"), 0)
        return (
            f"約 {ratio}% 的同業中期走勢偏強"
            if score > 0
            else f"僅約 {ratio}% 的同業中期走勢偏強，產業支撐不足"
        )
    if rule.rule_id == "industry.revenue_yoy":
        value = _signed(values.get("peer_revenue_yoy_median_pct"))
        return (
            f"同業近期營收較去年同期成長約 {value}%，產業需求偏正面"
            if score > 0
            else f"同業近期營收較去年同期變化約 {value}%，產業需求偏弱"
        )

    if rule.rule_id == "company.revenue_yoy_3m":
        value = _signed(values.get("revenue_yoy_3m_median_pct"))
        return (
            f"最近三個月營收較去年同期成長約 {value}%，營運動能偏正面"
            if score > 0
            else f"最近三個月營收較去年同期變化約 {value}%，營運動能轉弱"
        )
    if rule.rule_id == "company.revenue_acceleration":
        value = _float(values.get("revenue_yoy_acceleration_pct"))
        return (
            f"營收成長速度比前一季改善約 {abs(value):.1f} 個百分點"
            if score > 0
            else f"營收成長速度比前一季減弱約 {abs(value):.1f} 個百分點"
        )
    if rule.rule_id == "company.gross_margin":
        value = _number(values.get("gross_margin_pct"))
        if score < 0:
            return f"毛利率約 {value}%，產品獲利空間偏薄"
        if score >= 3:
            return f"毛利率約 {value}%，目前產品獲利空間相對穩定"
        return f"毛利率約 {value}%，目前仍維持正向獲利"
    if rule.rule_id == "company.operating_margin":
        value = _number(values.get("operating_margin_pct"))
        if score < 0:
            return f"本業目前仍處於虧損，營運壓力較高（營業利益率約 {value}%）"
        if score >= 3:
            return f"本業獲利能力穩定，營業利益率約 {value}%"
        return f"本業目前仍有獲利，營業利益率約 {value}%"
    if rule.rule_id == "company.net_income_growth":
        value = _signed(values.get("net_income_growth_pct"))
        return (
            f"最新稅後獲利較去年同期成長約 {value}%"
            if score > 0
            else f"最新稅後獲利較去年同期變化約 {value}%，獲利表現轉弱"
        )

    if rule.rule_id == "market.price_vs_ema20":
        return (
            "整體大盤短線仍偏穩"
            if score > 0
            else "整體大盤短線轉弱，個股容易受到拖累"
        )
    if rule.rule_id == "market.ema20_vs_ema60":
        return (
            "大盤近期走勢比過去幾個月更強，市場氣氛改善"
            if score > 0
            else "大盤近期走勢仍弱於過去幾個月，市場氣氛偏保守"
        )
    if rule.rule_id == "market.ema60_vs_ema120":
        return (
            "大盤中長期走勢仍維持正向"
            if score > 0
            else "大盤中長期走勢仍承受壓力"
        )
    if rule.rule_id == "market.return_20d":
        value = _signed(values.get("return_20d_pct"))
        return (
            f"大盤近一個月上漲約 {value}%，整體環境提供支撐"
            if score > 0
            else f"大盤近一個月變化約 {value}%，整體環境偏弱"
        )

    if rule.rule_id.startswith("event.materiality."):
        ratio = _number(values.get("annualized_revenue_ratio_pct"))
        return f"公告內容可量化，估算年化規模約占近一年營收 {ratio}%，影響具有實質性"
    if rule.rule_id.startswith("event."):
        title = _event_title(rule.message)
        if score > 0:
            return f"公司發布正面官方公告：{title}"
        if score < 0:
            return f"公司發布負面官方公告：{title}"
        return f"公司發布新的官方公告：{title}"

    return rule.message.rstrip("。；; ")


def module_plain_points(module: ModuleResult, limit: int = 3) -> list[str]:
    ordered = sorted(module.rules, key=lambda rule: abs(rule.score), reverse=True)
    points = [plain_message(rule) for rule in ordered[:limit]]
    if not points and module.notes:
        points = [note.rstrip("。；; ") for note in module.notes[:limit]]
    return points


def _event_title(message: str) -> str:
    if "：" in message:
        return message.rsplit("：", 1)[-1].strip()
    return message.strip()


def _float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _number(value: object, decimals: int = 1) -> str:
    return f"{_float(value):.{decimals}f}"


def _signed(value: object) -> str:
    return f"{_float(value):+.1f}"
