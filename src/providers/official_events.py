from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Iterable

from src.models import OfficialEvent
from src.providers.http import HttpClient, ProviderError


TWSE_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap04_L"
TPEX_URL = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap04_O"


class OfficialEventProvider:
    def __init__(self, http: HttpClient | None = None) -> None:
        self.http = http or HttpClient(timeout=30)
        self._cache: dict[str, list[dict[str, Any]]] = {}

    def events(self, symbol: str, market: str) -> list[OfficialEvent]:
        market_upper = market.upper()
        source = "TWSE" if market_upper in {"TWSE", "TSE", "LISTED"} else "TPEx"
        url = TWSE_URL if source == "TWSE" else TPEX_URL
        rows = self._load(url)
        result: list[OfficialEvent] = []
        for row in rows:
            row_symbol = _value(row, ["公司代號", "公司代碼", "SecuritiesCompanyCode", "Code"])
            if str(row_symbol).strip() != str(symbol).strip():
                continue
            title = str(_value(row, ["主旨", "Subject", "Title"]) or "重大訊息").strip()
            description = str(
                _value(
                    row,
                    [
                        "說明",
                        "重大訊息說明",
                        "Description",
                        "說明事項",
                        "內容",
                    ],
                )
                or ""
            ).strip()
            published_at = _published_at(row)
            company_name = str(_value(row, ["公司名稱", "CompanyName", "簡稱"]) or symbol)
            raw_identity = "|".join([source, str(symbol), published_at, title, description])
            event_id = hashlib.sha256(raw_identity.encode("utf-8")).hexdigest()[:20]
            result.append(
                OfficialEvent(
                    event_id=event_id,
                    symbol=str(symbol),
                    company_name=company_name,
                    title=title,
                    description=description,
                    published_at=published_at,
                    source=source,
                    source_url=url,
                )
            )
        return sorted(result, key=lambda event: event.published_at, reverse=True)

    def _load(self, url: str) -> list[dict[str, Any]]:
        if url not in self._cache:
            payload = self.http.get_json(url)
            if not isinstance(payload, list):
                raise ProviderError(f"重大訊息 API 回傳格式異常: {url}")
            self._cache[url] = payload
        return self._cache[url]


def _value(row: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _published_at(row: dict[str, Any]) -> str:
    date_value = _value(
        row,
        ["發言日期", "發布日期", "Date", "日期", "事實發生日"],
    )
    time_value = _value(row, ["發言時間", "發布時間", "Time", "時間"])
    raw = " ".join(part for part in [str(date_value or ""), str(time_value or "")] if part).strip()
    if not raw:
        return datetime.now().astimezone().isoformat()
    return raw
