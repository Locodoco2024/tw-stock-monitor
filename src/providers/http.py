from __future__ import annotations

import time
from typing import Any

import requests


class ProviderError(RuntimeError):
    pass


class HttpClient:
    def __init__(self, timeout: int = 20, retries: int = 3, backoff: float = 1.0) -> None:
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "tw-stock-monitor/0.1 (+https://github.com/)",
                "Accept": "application/json",
            }
        )

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt < self.retries - 1:
                    time.sleep(self.backoff * (2**attempt))
        raise ProviderError(f"GET {url} 失敗: {last_error}") from last_error

    def post_json(
        self,
        url: str,
        *,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                if not response.content:
                    return None
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt < self.retries - 1:
                    time.sleep(self.backoff * (2**attempt))
        raise ProviderError(f"POST {url} 失敗: {last_error}") from last_error
