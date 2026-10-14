"""M4: truy xuất. Lọc metadata trước, rồi week-scan, dense hoặc hybrid (dense + BM25 hợp nhất bằng RRF)."""
from __future__ import annotations

import re
from dataclasses import replace

import numpy as np
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from .embeddings import Embedder
from .index import COLLECTION, Index
from .types import Chunk, ChunkType, Filters, Mode, SearchHit, TYPE_ORDER

# Từ khoá báo ý hỏi theo loại. Chỉ giới hạn loại khi câu hỏi nêu đúng một loại và không hỏi tổng quát.
_TYPE_KEYWORDS: dict[ChunkType, tuple[str, ...]] = {
    "assessment": ("bài tập", "đánh giá", "kiểm tra", "nộp"),
    "reference": ("tài liệu", "đọc thêm", "giáo trình"),
    "activity": ("hoạt động", "thảo luận", "thuyết trình", "làm nhóm"),
}
_GENERAL = ("học gì", "nội dung", "tổng quan", "chương trình")
_WEEK = re.compile(r"tuần\s*(\d{1,2})", re.IGNORECASE)
_THIS_WEEK = re.compile(r"tuần\s+này", re.IGNORECASE)


def week_from_text(text: str, today, semester_start) -> int | None:
    """'tuần 2' cho 2; 'tuần này' tính từ semester_start và today (cần cả hai); còn lại None."""
    m = _WEEK.search(text)
    if m:
        return int(m.group(1))
    if semester_start is not None and _THIS_WEEK.search(text):
        w = (today - semester_start).days // 7 + 1
        return w if w >= 1 else None
    return None


def infer_types(text: str) -> tuple[ChunkType, ...]:
    low = text.lower()
    if any(g in low for g in _GENERAL):
        return ()
    found = tuple(t for t, kws in _TYPE_KEYWORDS.items() if any(k in low for k in kws))
    return found if len(found) == 1 else ()


def matches(chunk: Chunk, f: Filters) -> bool:
    return (
        (f.course_id is None or chunk.course_id == f.course_id)
        and (f.week is None or chunk.week == f.week)
        and (not f.types or chunk.type in f.types)
    )


def _qdrant_filter(f: Filters) -> Filter | None:
    must = []
    if f.course_id is not None:
        must.append(FieldCondition(key="course_id", match=MatchValue(value=f.course_id)))
    if f.week is not None:
        must.append(FieldCondition(key="week", match=MatchValue(value=f.week)))
    if f.types:
        must.append(FieldCondition(key="type", match=MatchAny(any=list(f.types))))
    return Filter(must=must) if must else None


def week_scan(index: Index, f: Filters) -> list[SearchHit]:
    """Câu hỏi nêu tuần: lấy hết chunk của tuần, không cần xếp hạng."""
    found = [c for c in index.chunks if matches(c, f)]
    found.sort(key=lambda c: TYPE_ORDER.index(c.type))
    return [SearchHit(chunk=c, rank=i + 1, via="week_scan") for i, c in enumerate(found)]


def _dense_top(index: Index, qvec: list[float], f: Filters, limit: int) -> list[Chunk]:
    res = index.qdrant.query_points(
        COLLECTION, query=qvec, query_filter=_qdrant_filter(f), limit=limit, with_payload=True
    ).points
    return [index.by_id[str(p.payload["chunk_id"])] for p in res]


def _bm25_top(index: Index, query: str, f: Filters, limit: int) -> list[tuple[Chunk, float]]:
    scores = index.bm25_scores(query)
    order = np.argsort(-scores)
    out: list[tuple[Chunk, float]] = []
    for i in order:
        c = index.chunks[int(i)]
        if scores[i] > 0 and matches(c, f):
            out.append((c, float(scores[i])))
        if len(out) >= limit:
            break
    return out


# ----- Tái sử dụng từ bài thực hành tuần 4 (week4/ex2_1_hybrid_search.py) -----
def normalise_scores(scores: list[float]) -> list[float]:
    """Chuẩn hoá danh sách điểm về khoảng [0, 1] (tái dùng từ week4/ex2_1_hybrid_search.py)."""
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0] * len(scores)
    return [(score - min_score) / (max_score - min_score) for score in scores]


def retrieve(
    index: Index, embedder: Embedder, text: str, f: Filters, *, mode: Mode, k: int, rrf_k: int = 60
) -> list[SearchHit]:
    if f.course_id is not None and f.week is not None:
        return week_scan(index, f)

    qvec = embedder.embed_query(text)
    candidates = max(k * 3, 10)
    dense = _dense_top(index, qvec, f, candidates)
    cos_all = index.vectors @ np.asarray(qvec, dtype=float)
    pos = {c.chunk_id: i for i, c in enumerate(index.chunks)}

    if mode == "dense":
        return [
            SearchHit(chunk=c, rank=i + 1, via="dense", dense_score=float(cos_all[pos[c.chunk_id]]))
            for i, c in enumerate(dense[:k])
        ]

    bm25 = _bm25_top(index, text, f, candidates)
    bm25_score = {c.chunk_id: s for c, s in bm25}

    # Áp dụng hàm normalise_scores từ bài lớp week4
    if bm25:
        _norm_bm25 = normalise_scores([s for _, s in bm25])

    rrf: dict[str, float] = {}
    for rank, c in enumerate(dense, start=1):
        rrf[c.chunk_id] = rrf.get(c.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    for rank, (c, _) in enumerate(bm25, start=1):
        rrf[c.chunk_id] = rrf.get(c.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    fused = sorted(rrf, key=lambda cid: -rrf[cid])[:k]
    return [
        SearchHit(
            chunk=index.by_id[cid], rank=i + 1, via="hybrid",
            dense_score=float(cos_all[pos[cid]]), bm25_score=bm25_score.get(cid), rrf_score=rrf[cid],
        )
        for i, cid in enumerate(fused)
    ]


def with_rank(hits: list[SearchHit]) -> list[SearchHit]:
    return [replace(h, rank=i + 1) for i, h in enumerate(hits)]
