from __future__ import annotations

import time
from typing import Any

import requests


class RetryingHttpClient:
    """Small HTTP client with retry behavior for external API polling."""

    def __init__(self, timeout_seconds: int, retries: int) -> None:
        self.timeout_seconds = timeout_seconds
        self.retries = retries

    def get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None

        for attempt in range(1, self.retries + 1):
            resp = None
            try:
                resp = requests.get(url, params=params, timeout=self.timeout_seconds)
                resp.raise_for_status()
                return resp.json()
            except requests.HTTPError as exc:
                detail = resp.text if resp is not None else ""
                try:
                    detail = resp.json().get("reason", detail) if resp is not None else detail
                except ValueError:
                    pass

                status_code = resp.status_code if resp is not None else None
                last_error = requests.HTTPError(f"{exc}. API detail: {detail}")
                retriable = status_code is not None and status_code >= 500

                if not retriable or attempt == self.retries:
                    raise last_error from exc
            except requests.RequestException as exc:
                last_error = exc
                if attempt == self.retries:
                    raise

            time.sleep(min(2 * attempt, 6))

        if last_error is not None:
            raise last_error

        raise RuntimeError("Unexpected request flow in RetryingHttpClient.get_json")
