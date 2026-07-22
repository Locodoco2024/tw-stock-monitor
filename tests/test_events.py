from __future__ import annotations

from src.models import OfficialEvent
from src.scoring.events import extract_amount_twd, score_official_events


def _config() -> dict:
    return {
        "official_event_rules": {
            "negation_terms": ["並未", "否認", "澄清"],
            "rules": [
                {
                    "id": "confirmed_order",
                    "phrases": ["取得訂單", "簽訂供貨合約"],
                    "score": 10,
                    "title": "正式訂單",
                }
            ],
        }
    }


def test_extract_twd_amount() -> None:
    assert extract_amount_twd("合約金額新台幣 12.5 億元") == 1_250_000_000


def test_negated_order_is_not_scored() -> None:
    event = OfficialEvent(
        event_id="1",
        symbol="2330",
        company_name="測試",
        title="澄清媒體報導",
        description="本公司並未取得訂單，相關報導並非事實。",
        published_at="2026-01-01",
        source="TWSE",
    )
    result, matched = score_official_events([event], _config(), 25, True, None)
    assert result.score == 0
    assert matched == []


def test_confirmed_order_and_materiality_are_scored() -> None:
    event = OfficialEvent(
        event_id="2",
        symbol="2330",
        company_name="測試",
        title="簽訂供貨合約",
        description="合約金額新台幣20億元，期間2年。",
        published_at="2026-01-01",
        source="TWSE",
    )
    result, matched = score_official_events([event], _config(), 25, True, 5_000_000_000)
    assert result.score > 10
    assert matched == ["2"]
