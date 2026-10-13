"""Embedder: Gemini (thật) và fake (băm từ khoá, chỉ để test offline)."""
from __future__ import annotations

import hashlib
import math
from typing import Protocol

import httpx

from .config import Settings
from .http import LLMError, post_json
from .text import syllables


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class GeminiEmbedder:
    """gemini-embedding-001 qua REST. Dùng taskType khác nhau cho tài liệu và câu hỏi.

    CHƯA TEST với key thật trong bản nháp này: kiểm tra tên field theo tài liệu Gemini API hiện hành.
    """

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        if not settings.gemini_api_key:
            raise LLMError("Error: thiếu GEMINI_API_KEY, hãy điền trong .env (hoặc đặt EMBEDDER=fake để test offline)")
        self._s = settings
        self._client = client
        self.name = f"gemini:{settings.embedding_model}"
        self.dim = settings.embedding_dim
        self._cache: dict[tuple[str, str], list[float]] = {}

    def _batch(self, texts: list[str], task_type: str) -> list[list[float]]:
        model = self._s.embedding_model
        body: dict[str, object] = {
            "requests": [
                {
                    "model": f"models/{model}",
                    "content": {"parts": [{"text": t}]},
                    "taskType": task_type,
                    "outputDimensionality": self.dim,
                }
                for t in texts
            ]
        }
        resp = post_json(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents",
            headers={"x-goog-api-key": self._s.gemini_api_key},
            body=body,
            timeout=self._s.llm_timeout,
            max_retries=self._s.llm_max_retries,
            backoff=self._s.llm_backoff,
            what="Embedding",
            client=self._client,
        )
        try:
            vectors = [e["values"] for e in resp.json()["embeddings"]]
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMError(f"Error: phản hồi embedding sai định dạng: {resp.text[:200]}") from exc
        if len(vectors) != len(texts):
            raise LLMError("Error: số vector trả về không khớp số văn bản gửi lên")
        # Khi cắt chiều (outputDimensionality) vector không còn chuẩn hoá, nên chuẩn hoá lại
        return [_normalize(v) for v in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 50):
            out.extend(self._batch(texts[i:i + 50], "RETRIEVAL_DOCUMENT"))
        return out

    def embed_query(self, text: str) -> list[float]:
        key = ("q", text)
        if key not in self._cache:  # cache để quét ngưỡng và so dense/hybrid không gọi API lặp lại
            self._cache[key] = self._batch([text], "RETRIEVAL_QUERY")[0]
        return self._cache[key]


class FakeEmbedder:
    """Túi từ băm vào dim chiều. Cosine phản ánh độ trùng từ, đủ để test plumbing.

    KHÔNG dùng cho số liệu eval nộp bài: nó không hiểu nghĩa.
    """

    def __init__(self, dim: int = 256) -> None:
        self.name = "fake"
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in syllables(text):
            h = int.from_bytes(hashlib.md5(tok.encode("utf-8")).digest()[:4], "big")
            vec[h % self.dim] += 1.0
        return _normalize(vec)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def make_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "fake":
        return FakeEmbedder()
    if settings.embedder == "gemini":
        return GeminiEmbedder(settings)
    raise ValueError(f"EMBEDDER không hợp lệ: {settings.embedder!r} (gemini hoặc fake)")
