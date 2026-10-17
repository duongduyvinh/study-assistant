"""Judge tuỳ chọn: Gemma 3 1B chạy cục bộ qua LM Studio (API kiểu OpenAI), chấm nhị phân từng expected_point.

Cần bật Local Server trong LM Studio (mặc định http://localhost:1234/v1) và nạp model google/gemma-3-1b.
Model 1B chấm nhiễu, nên số liệu chính của make eval không phụ thuộc vào judge.
"""
from __future__ import annotations

import logging

import httpx

log = logging.getLogger("t3.judge")

PROMPT = (
    "Task: Check if the following Vietnamese answer mentions the key point.\n"
    "Key point: {point}\n"
    "Answer: {answer}\n"
    "Question: Does the answer mention or cover this key point?\n"
    "Answer with only one word: YES or NO."
)


def judge_points(
    answer: str, points: tuple[str, ...], *, model: str = "google/gemma-3-1b",
    url: str = "http://localhost:1234/v1/chat/completions", timeout: float = 60.0,
) -> float | None:
    """Tỉ lệ ý được nêu (0 đến 1). None nếu không gọi được LM Studio."""
    if not points:
        return None
    yes = 0
    try:
        with httpx.Client(timeout=timeout) as client:
            for point in points:
                resp = client.post(url, json={
                    "model": model, "temperature": 0, "max_tokens": 5,
                    "messages": [{"role": "user", "content": PROMPT.format(point=point, answer=answer)}],
                })
                resp.raise_for_status()
                verdict = resp.json()["choices"][0]["message"]["content"].strip().upper()
                yes += verdict.startswith(("YES", "CO"))
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
        log.warning("Judge không chạy được (%s: %s)", type(exc).__name__, exc)
        return None
    return yes / len(points)
