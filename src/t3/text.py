"""Chuẩn hoá và tách token tiếng Việt cho BM25, độ phủ từ khoá và embedder giả."""
from __future__ import annotations

import re
import unicodedata

_TOKEN = re.compile(r"[^\W_]+(?:-[^\W_]+)*", re.UNICODE)

# Từ chức năng hay gặp trong câu hỏi, bỏ khi tính độ phủ từ khoá
STOPWORDS = frozenset(
    "là gì và của có các những này cho trong tuần học môn bài nào như thế được không một để với "
    "khi ở thì mà hay hoặc cần phải sao đâu bao nhiêu mấy em tôi mình".split()
)


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).lower()


def syllables(text: str) -> list[str]:
    """Âm tiết và token chữ-số, giữ nguyên dạng như if-else, h1-h6, web101."""
    return _TOKEN.findall(normalize(text))


def tokenize(text: str) -> list[str]:
    """Unigram + bigram âm tiết, dùng cho BM25."""
    syl = syllables(text)
    return syl + [f"{a}_{b}" for a, b in zip(syl, syl[1:])]


def content_syllables(text: str) -> list[str]:
    return [s for s in syllables(text) if s not in STOPWORDS]


def coverage(query: str, doc_text: str) -> float:
    """Tỉ lệ âm tiết nội dung của câu hỏi xuất hiện trong chunk (0 đến 1)."""
    q = set(content_syllables(query))
    if not q:
        return 0.0
    d = set(syllables(doc_text))
    return len(q & d) / len(q)
