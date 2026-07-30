from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests


class FinMindResearchError(RuntimeError):
    pass


class FinMindQuotaExceeded(FinMindResearchError):
    pass


@dataclass(frozen=True)
class FinMindApiUsage:
    used: int | None
    limit: int | None

    @property
    def remaining(self) -> int | None:
        if self.used is None or self.limit is None:
            return None
        return max(0, self.limit - self.used)


@dataclass
class FinMindResearchClient:
    token: str | None = None
    timeout: int = 90
    retries: int = 4
    min_interval_seconds: float = 0.25

    BASE_URL = "https://api.finmindtrade.com/api/v4/data"
    USER_INFO_URL = "https://api.web.finmindtrade.com/v2/user_info"

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "tw-stock-monitor-research/0.2.1",
            }
        )
        if self.token:
            self.session.headers["Authorization"] = f"Bearer {self.token}"
        self._last_request_at = 0.0
        self.request_count = 0

    def fetch(
        self,
        dataset: str,
        *,
        data_id: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"dataset": dataset}
        if data_id:
            params["data_id"] = data_id
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        last_error: Exception | None = None
        for attempt in range(self.retries):
            self._wait_for_rate_limit()
            try:
                self.request_count += 1
                response = self.session.get(
                    self.BASE_URL,
                    params=params,
                    timeout=self.timeout,
                )
                if response.status_code == 429:
                    retry_after = float(response.headers.get("Retry-After", 5))
                    time.sleep(max(retry_after, 5.0) * (attempt + 1))
                    continue
                if response.status_code == 402:
                    try:
                        detail = response.json().get("msg")
                    except ValueError:
                        detail = response.text
                    raise FinMindQuotaExceeded(
                        detail or "FinMind API quota exceeded"
                    )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise FinMindResearchError(f"{dataset} 回傳不是 JSON 物件")
                if payload.get("status") == 402:
                    raise FinMindQuotaExceeded(
                        payload.get("msg") or "FinMind API quota exceeded"
                    )
                if payload.get("status") not in (None, 200):
                    raise FinMindResearchError(
                        f"{dataset}/{data_id or '*'} 查詢失敗: "
                        f"{payload.get('msg') or payload}"
                    )
                rows = payload.get("data") or []
                if not isinstance(rows, list):
                    raise FinMindResearchError(f"{dataset} data 不是陣列")
                return [row for row in rows if isinstance(row, dict)]
            except FinMindQuotaExceeded:
                raise
            except (requests.RequestException, ValueError, FinMindResearchError) as exc:
                last_error = exc
                if attempt < self.retries - 1:
                    time.sleep(2**attempt)
        raise FinMindResearchError(
            f"{dataset}/{data_id or '*'} 下載失敗: {last_error}"
        ) from last_error

    def get_api_usage(self) -> FinMindApiUsage | None:
        """Return the current FinMind account usage without interrupting backfill."""
        if not self.token:
            return None
        try:
            response = self.session.get(self.USER_INFO_URL, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                return None
            used = _optional_int(payload.get("user_count"))
            limit = _optional_int(payload.get("api_request_limit"))
            if used is None and limit is None:
                return None
            return FinMindApiUsage(used=used, limit=limit)
        except (requests.RequestException, ValueError):
            return None

    def _wait_for_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.min_interval_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_at = time.monotonic()


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
