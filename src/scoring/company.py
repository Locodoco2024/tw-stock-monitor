from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.indicators import median
from src.models import ModuleResult, RuleResult
from src.scoring.common import cap_module, number


REVENUE_KEYS = ["revenue", "Revenue"]
REVENUE_YOY_KEYS = ["revenue_year_growth_rate", "YoY", "yoy", "revenue_yoy"]


def score_company_capacity(
    monthly_revenue: list[dict[str, Any]],
    financial_statements: list[dict[str, Any]],
    weight: float,
) -> tuple[ModuleResult, float | None]:
    if not monthly_revenue and not financial_statements:
        return cap_module(
            "company_capacity", weight, 0.0, [], ["月營收與財報資料均不足"]
        ), None

    rules: list[RuleResult] = []
    yoy_series = _revenue_yoy_series(monthly_revenue)
    trailing_revenue = _trailing_revenue(monthly_revenue)
    if yoy_series:
        latest_three = median(yoy_series[-3:])
        previous_three = median(yoy_series[-6:-3]) if len(yoy_series) >= 6 else None
        if latest_three is not None:
            if latest_three >= 20:
                score = 6
            elif latest_three >= 8:
                score = 4
            elif latest_three <= -20:
                score = -6
            elif latest_three <= -8:
                score = -4
            else:
                score = 0
            if score:
                rules.append(
                    RuleResult(
                        rule_id="company.revenue_yoy_3m",
                        module="company_capacity",
                        score=score,
                        message=f"最近三個月營收年增率中位數為 {latest_three:+.1f}%",
                        values={"revenue_yoy_3m_median_pct": latest_three},
                    )
                )
        if latest_three is not None and previous_three is not None:
            acceleration = latest_three - previous_three
            if acceleration >= 8:
                score = 3
            elif acceleration <= -8:
                score = -3
            else:
                score = 0
            if score:
                rules.append(
                    RuleResult(
                        rule_id="company.revenue_acceleration",
                        module="company_capacity",
                        score=score,
                        message=(
                            f"營收年增趨勢較前一個三個月"
                            f"{'改善' if score > 0 else '惡化'} {abs(acceleration):.1f} 個百分點"
                        ),
                        values={"revenue_yoy_acceleration_pct": acceleration},
                    )
                )

    metrics = _statement_metrics(financial_statements)
    gross_margin = metrics.get("gross_margin")
    operating_margin = metrics.get("operating_margin")
    net_income_growth = metrics.get("net_income_growth")
    if gross_margin is not None:
        rules.append(
            RuleResult(
                rule_id="company.gross_margin",
                module="company_capacity",
                score=3 if gross_margin >= 25 else -2 if gross_margin < 10 else 1,
                message=f"最新可辨識毛利率約 {gross_margin:.1f}%",
                values={"gross_margin_pct": gross_margin},
            )
        )
    if operating_margin is not None:
        rules.append(
            RuleResult(
                rule_id="company.operating_margin",
                module="company_capacity",
                score=3 if operating_margin >= 15 else -3 if operating_margin < 0 else 1,
                message=f"最新可辨識營業利益率約 {operating_margin:.1f}%",
                values={"operating_margin_pct": operating_margin},
            )
        )
    if net_income_growth is not None:
        score = 3 if net_income_growth >= 10 else -3 if net_income_growth <= -10 else 0
        if score:
            rules.append(
                RuleResult(
                    rule_id="company.net_income_growth",
                    module="company_capacity",
                    score=score,
                    message=f"最新可比稅後淨利年增率約 {net_income_growth:+.1f}%",
                    values={"net_income_growth_pct": net_income_growth},
                )
            )

    notes: list[str] = []
    if financial_statements and not metrics:
        notes.append("財報欄位未符合目前可辨識格式，財報內容未計分")
    return cap_module("company_capacity", weight, weight, rules, notes), trailing_revenue


def _revenue_yoy_series(rows: list[dict[str, Any]]) -> list[float]:
    sorted_rows = sorted(rows, key=lambda row: str(row.get("date") or row.get("revenue_month") or ""))
    values: list[float] = []
    for row in sorted_rows:
        value = _first_number(row, REVENUE_YOY_KEYS)
        if value is not None:
            values.append(value)
    return values


def _trailing_revenue(rows: list[dict[str, Any]]) -> float | None:
    sorted_rows = sorted(rows, key=lambda row: str(row.get("date") or row.get("revenue_month") or ""))
    values: list[float] = []
    for row in sorted_rows[-12:]:
        value = _first_number(row, REVENUE_KEYS)
        if value is not None:
            values.append(value)
    return sum(values) if len(values) >= 6 else None


def _statement_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    by_date: dict[str, dict[str, float]] = defaultdict(dict)
    aliases = {
        "revenue": {"Revenue", "OperatingRevenue", "營業收入合計", "營業收入"},
        "gross_profit": {"GrossProfit", "GrossProfitLoss", "營業毛利（毛損）", "營業毛利"},
        "operating_income": {"OperatingIncome", "OperatingIncomeLoss", "營業利益（損失）", "營業利益"},
        "net_income": {"IncomeAfterTaxes", "ProfitLoss", "本期淨利（淨損）", "本期淨利"},
    }
    for row in rows:
        date_key = str(row.get("date") or "")
        type_name = str(row.get("type") or row.get("account") or row.get("origin_name") or "")
        value = number(row.get("value"))
        if not date_key or value is None:
            continue
        for metric, names in aliases.items():
            if type_name in names:
                by_date[date_key][metric] = value
                break
    if not by_date:
        return {}
    dates = sorted(by_date)
    latest = by_date[dates[-1]]
    metrics: dict[str, float] = {}
    revenue = latest.get("revenue")
    if revenue and revenue != 0:
        if "gross_profit" in latest:
            metrics["gross_margin"] = latest["gross_profit"] / revenue * 100
        if "operating_income" in latest:
            metrics["operating_margin"] = latest["operating_income"] / revenue * 100
    if len(dates) >= 5 and "net_income" in latest:
        prior = by_date[dates[-5]].get("net_income")
        if prior not in (None, 0):
            metrics["net_income_growth"] = (latest["net_income"] / prior - 1) * 100
    return metrics


def _first_number(row: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = number(row.get(key))
        if value is not None:
            return value
    return None
