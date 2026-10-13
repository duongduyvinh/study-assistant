"""M3: một hàm call_llm duy nhất. Có timeout, retry, log lỗi HTTP và trả Usage."""
from __future__ import annotations

import re
import time

import httpx

from .config import Settings, load_settings
from .http import LLMError, post_json
from .types import ChatMessage, LLMFn, Usage

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """Dòng Qwen3 có thể chèn khối <think>; bỏ đi để parse action và trích dẫn không bị hỏng."""
    return _THINK.sub("", text).strip()


def call_llm(
    messages: list[ChatMessage],
    *,
    stage: str,
    settings: Settings | None = None,
    temperature: float = 0.0,
    max_tokens: int = 800,
    client: httpx.Client | None = None,
) -> tuple[str, Usage]:
    s = settings or load_settings()
    if not s.llm_api_key:
        raise LLMError("Error: thiếu LLM_API_KEY, hãy điền trong .env")
    body: dict[str, object] = {
        "model": s.llm_model,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "temperature": temperature,
        "max_tokens": max_tokens,
        **s.extra_body(),
    }
    started = time.monotonic()
    resp = post_json(
        f"{s.llm_base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {s.llm_api_key}"},
        body=body,
        timeout=s.llm_timeout,
        max_retries=s.llm_max_retries,
        backoff=s.llm_backoff,
        what=f"LLM[{stage}]",
        client=client,
    )
    latency_ms = int((time.monotonic() - started) * 1000)
    try:
        data = resp.json()
        text = data["choices"][0]["message"]["content"] or ""
        usage_raw = data.get("usage") or {}
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Error: phản hồi LLM sai định dạng: {resp.text[:200]}") from exc
    usage = Usage(
        stage=stage,
        model=str(data.get("model", s.llm_model)),
        prompt_tokens=int(usage_raw.get("prompt_tokens", 0)),
        completion_tokens=int(usage_raw.get("completion_tokens", 0)),
        latency_ms=latency_ms,
    )
    return strip_think(text), usage


def default_llm(settings: Settings | None = None) -> LLMFn:
    """LLMFn dùng call_llm thật, gắn sẵn settings."""
    s = settings or load_settings()

    def _fn(messages: list[ChatMessage], stage: str) -> tuple[str, Usage]:
        return call_llm(messages, stage=stage, settings=s)

    return _fn
