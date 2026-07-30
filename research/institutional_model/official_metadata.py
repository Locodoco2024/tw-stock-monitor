from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable

import requests

from research.institutional_model.database import ResearchDatabase


class OfficialMetadataError(RuntimeError):
    pass


@dataclass
class OfficialMarketMetadataClient:
    timeout: int = 60

    TWSE_COMPANY_INFO_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
    TPEX_COMPANY_INFO_URL = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O"
    TWSE_COMPANY_INFO_CSV_URL = (
        "https://mopsfin.twse.com.tw/opendata/t187ap03_L.csv"
    )
    TPEX_COMPANY_INFO_CSV_URL = (
        "https://mopsfin.twse.com.tw/opendata/t187ap03_O.csv"
    )
    TWSE_DELISTED_CSV_URL = (
        "https://www.twse.com.tw/company/suspendListingCsvAndHtml?type=open_data"
    )

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json,text/csv,text/plain,*/*",
                "User-Agent": "tw-stock-monitor-research/0.2.3",
            }
        )

    def fetch_company_info(self, market_type: str) -> list[dict[str, Any]]:
        if market_type == "twse":
            json_url = self.TWSE_COMPANY_INFO_URL
            csv_url = self.TWSE_COMPANY_INFO_CSV_URL
            listing_keys = ("上市日期",)
            source_prefix = "TWSE:t187ap03_L"
        elif market_type == "tpex":
            json_url = self.TPEX_COMPANY_INFO_URL
            csv_url = self.TPEX_COMPANY_INFO_CSV_URL
            listing_keys = ("上櫃日期", "掛牌日期")
            source_prefix = "TPEx:mopsfin_t187ap03_O"
        else:
            raise ValueError(f"不支援的市場類型：{market_type}")

        payload, transport = self._get_json_with_csv_fallback(json_url, csv_url)
        result: list[dict[str, Any]] = []
        for raw in payload:
            row = _normalized_keys(raw)
            stock_id = _first_text(row, "公司代號", "證券代號", "股票代號")
            if not stock_id:
                continue
            result.append(
                {
                    "stock_id": stock_id,
                    "stock_name": _first_text(
                        row, "公司簡稱", "公司名稱", "證券名稱"
                    ),
                    "market_type": market_type,
                    "listing_date": _first_date(row, *listing_keys),
                    "report_date": _first_date(row, "出表日期", "資料日期"),
                    "source_dataset": f"{source_prefix}:{transport}",
                    "raw_json": json.dumps(
                        raw, ensure_ascii=False, sort_keys=True
                    ),
                }
            )
        if not result:
            raise OfficialMetadataError(
                f"{market_type} 官方公司基本資料沒有可用紀錄"
            )
        return result

    def fetch_twse_delisted(self) -> list[dict[str, Any]]:
        try:
            response = self.session.get(self.TWSE_DELISTED_CSV_URL, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OfficialMetadataError(f"TWSE 終止上市清單下載失敗：{exc}") from exc

        text = _decode_response(response.content)
        reader = csv.DictReader(io.StringIO(text))
        result: list[dict[str, Any]] = []
        for raw in reader:
            row = _normalized_keys(raw)
            stock_id = _first_text(row, "上市編號", "公司代號", "證券代號")
            delisting_date = _first_date(row, "終止上市日期", "下市日期")
            if not stock_id or not delisting_date:
                continue
            result.append(
                {
                    "stock_id": stock_id,
                    "stock_name": _first_text(row, "公司名稱", "公司簡稱"),
                    "delisting_date": delisting_date,
                    "source_dataset": "TWSE:suspendListingCsvAndHtml",
                    "raw_json": json.dumps(raw, ensure_ascii=False, sort_keys=True),
                }
            )
        return result

    def _get_json_with_csv_fallback(
        self, json_url: str, csv_url: str
    ) -> tuple[list[dict[str, Any]], str]:
        try:
            return self._get_json(json_url), "json"
        except OfficialMetadataError as json_error:
            try:
                return self._get_csv(csv_url), "csv_fallback"
            except OfficialMetadataError as csv_error:
                raise OfficialMetadataError(
                    "官方公司基本資料 JSON 與 CSV 皆下載失敗："
                    f"JSON={json_error}；CSV={csv_error}"
                ) from csv_error

    def _get_json(self, url: str) -> list[dict[str, Any]]:
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise OfficialMetadataError(
                f"官方基本資料 JSON 下載失敗：{url}：{exc}"
            ) from exc
        if not isinstance(payload, list):
            raise OfficialMetadataError(f"官方基本資料不是陣列：{url}")
        return [row for row in payload if isinstance(row, dict)]

    def _get_csv(self, url: str) -> list[dict[str, Any]]:
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OfficialMetadataError(
                f"官方基本資料 CSV 下載失敗：{url}：{exc}"
            ) from exc
        text = _decode_response(response.content)
        return [dict(row) for row in csv.DictReader(io.StringIO(text))]


def refresh_official_metadata(
    database: ResearchDatabase,
    client: OfficialMarketMetadataClient | None = None,
) -> dict[str, int]:
    client = client or OfficialMarketMetadataClient()
    counts: dict[str, int] = {}
    for market_type in ("twse", "tpex"):
        rows = client.fetch_company_info(market_type)
        counts[f"company_info_{market_type}"] = _replace_company_info(
            database, market_type, rows
        )
    delisted = client.fetch_twse_delisted()
    counts["twse_delisted"] = _replace_twse_delisted(database, delisted)
    return counts


def _replace_company_info(
    database: ResearchDatabase,
    market_type: str,
    rows: Iterable[dict[str, Any]],
) -> int:
    materialized = list(rows)
    with database.connect() as connection:
        connection.execute(
            "DELETE FROM official_company_info WHERE market_type=?", (market_type,)
        )
        connection.executemany(
            """
            INSERT INTO official_company_info (
                stock_id, market_type, stock_name, listing_date, report_date,
                source_dataset, raw_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                (
                    row["stock_id"],
                    row["market_type"],
                    row["stock_name"],
                    row["listing_date"],
                    row["report_date"],
                    row["source_dataset"],
                    row["raw_json"],
                )
                for row in materialized
            ),
        )
    return len(materialized)


def _replace_twse_delisted(
    database: ResearchDatabase, rows: Iterable[dict[str, Any]]
) -> int:
    materialized = list(rows)
    with database.connect() as connection:
        connection.execute("DELETE FROM official_twse_delisted")
        connection.executemany(
            """
            INSERT INTO official_twse_delisted (
                stock_id, stock_name, delisting_date, source_dataset,
                raw_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                (
                    row["stock_id"],
                    row["stock_name"],
                    row["delisting_date"],
                    row["source_dataset"],
                    row["raw_json"],
                )
                for row in materialized
            ),
        )
    return len(materialized)


def normalize_date(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    text = (
        text.replace("年", "/")
        .replace("月", "/")
        .replace("日", "")
        .replace(".", "/")
        .replace("-", "/")
    )
    parts = [item for item in re.split(r"[/\s]+", text) if item]
    if len(parts) < 3:
        digits = re.sub(r"\D", "", text)
        if len(digits) == 8:
            parts = [digits[:4], digits[4:6], digits[6:8]]
        elif len(digits) == 7:
            parts = [digits[:3], digits[3:5], digits[5:7]]
        else:
            return None
    try:
        year = int(parts[0])
        month = int(parts[1])
        day = int(parts[2])
        if year < 1911:
            year += 1911
        return date(year, month, day).isoformat()
    except (ValueError, TypeError):
        return None


def _normalized_keys(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key).strip().lstrip("\ufeff"): value for key, value in row.items()}


def _first_text(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _first_date(row: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = normalize_date(row.get(key))
        if value:
            return value
    return None


def _decode_response(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "big5", "cp950"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")
