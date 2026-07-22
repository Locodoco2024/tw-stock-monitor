from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Iterable

from src.models import AnalysisResult, RuleResult


MODULE_NAMES = {
    "official_events": "官方事件與催化",
    "market_pricing": "市場定價狀態",
    "industry_peers": "產業與同業確認",
    "company_capacity": "公司承接能力",
    "market_environment": "大盤環境",
    "holding": "持倉策略調整",
}


def write_stock_report(result: AnalysisResult, output_dir: str | Path) -> Path:
    directory = Path(output_dir) / result.user_id
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{result.symbol}.html"
    target.write_text(render_stock_report(result), encoding="utf-8")
    json_target = directory / f"{result.symbol}.json"
    json_target.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target


def write_index(results: Iterable[AnalysisResult], output_dir: str | Path) -> Path:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in sorted(results, key=lambda item: (item.user_id, item.symbol)):
        path = f"{result.user_id}/{result.symbol}.html"
        rows.append(
            f"<tr><td>{_h(result.user_id)}</td><td><a href='{_h(path)}'>"
            f"{_h(result.symbol)} {_h(result.name)}</a></td>"
            f"<td>{_h(result.operation_label)}</td>"
            f"<td>{result.operation_score:+.1f}</td>"
            f"<td>{result.completeness:.0f}%</td></tr>"
        )
    document = _page(
        "台股監控報告",
        "<h1>台股監控報告</h1>"
        "<p class='muted'>規則分析結果，不代表未來漲跌機率或投資建議。</p>"
        "<table><thead><tr><th>使用者</th><th>股票</th><th>操作傾向</th>"
        "<th>指數</th><th>完整度</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>",
    )
    target = target_dir / "index.html"
    target.write_text(document, encoding="utf-8")
    return target


def render_stock_report(result: AnalysisResult) -> str:
    quote = result.quote
    headline = (
        f"<div class='score-card'><div class='score-label'>{_h(result.operation_label)}</div>"
        f"<div class='score'>{abs(result.operation_score):.0f}%</div></div>"
    )
    metrics = [
        ("個股分析", f"{_h(result.objective_label)} {abs(result.objective_score):.0f}%"),
        ("機會分數", f"{result.opportunity_score:.1f}"),
        ("風險分數", f"{result.risk_score:.1f}"),
        ("資料完整度", f"{result.completeness:.0f}%"),
    ]
    if quote:
        metrics.insert(0, ("目前價格", f"{quote.price:g}"))
    if result.price_return_pct is not None:
        metrics.insert(1, ("價格報酬", f"{result.price_return_pct:+.1f}%"))
    metric_html = "".join(
        f"<div class='metric'><span>{_h(label)}</span><strong>{_h(value)}</strong></div>"
        for label, value in metrics
    )

    module_html = "".join(_render_module(module) for module in result.modules)
    holding_html = ""
    if result.holding_rules:
        holding_html = (
            "<section><h2>持倉策略調整</h2>"
            + _rules_table(result.holding_rules)
            + "</section>"
        )
    errors = ""
    if result.errors:
        errors = (
            "<section><h2>資料取得警告</h2><ul>"
            + "".join(f"<li>{_h(error)}</li>" for error in result.errors)
            + "</ul></section>"
        )
    raw_json = html.escape(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    body = f"""
    <a href='../index.html'>← 回到總覽</a>
    <header><h1>{_h(result.symbol)} {_h(result.name)}</h1>
    <p class='muted'>分析時間：{_h(result.analyzed_at)}</p></header>
    {headline}
    <div class='metrics'>{metric_html}</div>
    <section><h2>簡短結論</h2><p class='summary'>{_h(result.summary)}</p></section>
    {holding_html}
    {module_html}
    {errors}
    <section><details><summary>完整 JSON 計算結果</summary><pre>{raw_json}</pre></details></section>
    """
    return _page(f"{result.symbol} {result.name} 分析", body)


def _render_module(module) -> str:
    name = MODULE_NAMES.get(module.module, module.module)
    notes = ""
    if module.notes:
        notes = "<ul class='notes'>" + "".join(
            f"<li>{_h(note)}</li>" for note in module.notes
        ) + "</ul>"
    return (
        f"<section><h2>{_h(name)}</h2>"
        f"<p>模組淨貢獻：<strong>{module.score:+.1f}</strong> / {module.weight:.0f}</p>"
        f"{notes}{_rules_table(module.rules)}</section>"
    )


def _rules_table(rules: list[RuleResult]) -> str:
    if not rules:
        return "<p class='muted'>本模組沒有觸發計分規則。</p>"
    rows = []
    for rule in rules:
        source = f"<a href='{_h(rule.source)}'>來源</a>" if rule.source else ""
        rows.append(
            f"<tr><td><code>{_h(rule.rule_id)}</code></td>"
            f"<td class='number'>{rule.score:+.1f}</td>"
            f"<td>{_h(rule.message)}</td><td>{source}</td></tr>"
        )
    return (
        "<table><thead><tr><th>規則</th><th>分數</th><th>說明</th><th></th></tr>"
        "</thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang='zh-Hant'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>{_h(title)}</title>
<style>
:root {{ color-scheme: light dark; font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
body {{ max-width: 1100px; margin: 0 auto; padding: 28px 18px 80px; line-height: 1.65; }}
a {{ color: #4f86f7; }}
header {{ margin: 18px 0; }}
.muted {{ opacity: .7; }}
.summary {{ font-size: 1.08rem; }}
.score-card {{ display: inline-flex; align-items: baseline; gap: 14px; border: 1px solid #8885; border-radius: 16px; padding: 18px 24px; margin: 10px 0 18px; }}
.score-label {{ font-size: 1.15rem; font-weight: 700; }}
.score {{ font-size: 2.3rem; font-weight: 800; }}
.metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }}
.metric {{ border: 1px solid #8885; border-radius: 12px; padding: 12px; display: flex; flex-direction: column; }}
.metric span {{ opacity: .7; font-size: .9rem; }}
.metric strong {{ font-size: 1.15rem; }}
section {{ margin-top: 30px; }}
table {{ width: 100%; border-collapse: collapse; overflow: hidden; }}
th, td {{ border-bottom: 1px solid #8884; padding: 9px; text-align: left; vertical-align: top; }}
th {{ background: #8882; }}
td.number {{ white-space: nowrap; font-variant-numeric: tabular-nums; }}
pre {{ white-space: pre-wrap; word-break: break-word; border: 1px solid #8885; border-radius: 12px; padding: 14px; overflow: auto; }}
.notes {{ opacity: .8; }}
</style>
</head><body>{body}</body></html>"""


def _h(value: object) -> str:
    return html.escape(str(value), quote=True)
