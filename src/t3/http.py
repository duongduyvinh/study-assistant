"""POST JSON có timeout, retry với backoff và log lỗi HTTP (mã lỗi, lần thử thứ mấy). Log đi vào stderr."""
from __future__ import annotations

import logging
import time
from typing import Callable

import httpx

log = logging.getLogger("t3.http")

TRANSIENT_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class LLMError(RuntimeError):
    """Lỗi gọi dịch vụ ngoài sau khi đã hết lần thử hoặc gặp lỗi vĩnh viễn."""


def post_json(
    url: str,
    *,
    headers: dict[str, str],
    body: dict[str, object],
    timeout: float,
    max_retries: int,
    backoff: float,
    what: str,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    """Lỗi tạm thời (timeout, mạng, 408/429/5xx) thì thử lại; lỗi vĩnh viễn (4xx khác) thì dừng ngay."""
    attempts = max_retries + 1
    owns_client = client is None
    http = client or httpx.Client(timeout=timeout)
    try:
        for attempt in range(1, attempts + 1):
            wait = backoff * (2 ** (attempt - 1))
            try:
                resp = http.post(url, headers=headers, json=body, timeout=timeout)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                log.warning("%s: %s ở lần thử %d/%d", what, type(exc).__name__, attempt, attempts)
                if attempt == attempts:
                    raise LLMError(f"Error: {what} lỗi mạng sau {attempts} lần thử ({type(exc).__name__})") from exc
                sleep(wait)
                continue
            if resp.status_code < 400:
                return resp
            log.warning("%s: HTTP %d ở lần thử %d/%d", what, resp.status_code, attempt, attempts)
            if resp.status_code not in TRANSIENT_STATUS:
                raise LLMError(f"Error: {what} trả HTTP {resp.status_code} (lỗi vĩnh viễn): {resp.text[:200]}")
            if attempt == attempts:
                raise LLMError(f"Error: {what} trả HTTP {resp.status_code} sau {attempts} lần thử")
            retry_after = resp.headers.get("retry-after", "")
            sleep(float(retry_after) if retry_after.isdigit() else wait)
        raise LLMError(f"Error: {what} không có phản hồi")  # không tới được, để type checker yên tâm
    finally:
        if owns_client:
            http.close()
