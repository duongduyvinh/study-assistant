"""Judge tuỳ chọn: Gemma 3 1B chạy cục bộ qua Ollama, chấm nhị phân từng expected_point.

CHƯA TEST trong bản nháp này (cần Ollama + model gemma3:1b). Chỉ là bổ sung tham khảo:
model 1B chấm nhiễu, nên số liệu chính của make eval không phụ thuộc vào judge.
"""
from __future__ import annotations

import logging

import httpx

log = logging.getLogger("t3.judge")

PROMPT = (
    "Câu trả lời dưới đây có nêu ý sau không?\n"
    "Ý cần có: {point}\n"
    "Câu trả lời: {answer}\n"
    "Chỉ trả lời đúng một từ: CO hoặc KHONG."
)


def judge_points(
    answer: str, points: tuple[str, ...], *, model: str = "gemma3:1b",
    url: str = "http://localhost:11434/api/chat", timeout: float = 60.0,
) -> float | None:
    """Tỉ lệ ý được nêu (0 đến 1). None nếu không gọi được Ollama."""
    if not points:
        return None
    yes = 0
    try:
        with httpx.Client(timeout=timeout) as client:
            for point in points:
                resp = client.post(url, json={
                    "model": model, "stream": False, "options": {"temperature": 0},
                    "messages": [{"role": "user", "content": PROMPT.format(point=point, answer=answer)}],
                })
                resp.raise_for_status()
                verdict = resp.json()["message"]["content"].strip().upper()
                yes += verdict.startswith("CO")
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        log.warning("Judge không chạy được (%s: %s)", type(exc).__name__, exc)
        return None
    return yes / len(points)
