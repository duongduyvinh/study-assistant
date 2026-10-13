"""M4: gate (Jev) = rerank + ngưỡng. Quyết định answer, abstain hoặc clarify; không gọi LLM.

rerank_score v1 = w * dense_cosine + (1 - w) * độ phủ từ khoá của câu hỏi trong chunk.
Điểm RRF không dùng để so ngưỡng vì nó chỉ dựa trên thứ hạng, không phản ánh độ liên quan.
Ngưỡng lấy từ bảng quét ngưỡng của `make eval`, không đoán.
"""
from __future__ import annotations

from dataclasses import replace

from .config import Settings
from .index import Index
from .text import coverage
from .types import Filters, GateDecision, Query, SearchHit

KNOWN_COURSES_HINT = "Các môn hiện có: "


def _courses_text(index: Index) -> str:
    return "; ".join(f"{c.course_id} ({c.subject})" for c in index.courses.values())


def course_overview(index: Index, course_id: str) -> str:
    topics: dict[int, str] = {}
    for c in index.chunks:
        if c.course_id == course_id:
            topics.setdefault(c.week, c.topic)
    return "; ".join(f"tuần {w}: {t}" for w, t in sorted(topics.items()))


def precheck(query: Query, f: Filters, index: Index) -> GateDecision | None:
    """Những trường hợp chốt được trước khi truy xuất. Thiếu hoặc sai mã môn thì clarify, không retrieval."""
    if f.course_id is None:
        example = next(iter(index.courses), "WEB101")
        return GateDecision(
            action="clarify", reason="missing_course_id",
            hint=f"Hãy gửi lại kèm mã môn, ví dụ: 'Tuần 2 môn {example} học gì?'. {KNOWN_COURSES_HINT}{_courses_text(index)}.",
        )
    if f.course_id not in index.courses:
        return GateDecision(
            action="clarify", reason="unknown_course",
            hint=f"Không có môn {f.course_id}. {KNOWN_COURSES_HINT}{_courses_text(index)}.",
        )
    return None


def rerank(text: str, hits: list[SearchHit], dense_weight: float) -> list[SearchHit]:
    scored = [
        replace(h, rerank_score=dense_weight * (h.dense_score or 0.0) + (1 - dense_weight) * coverage(text, h.chunk.text + " " + h.chunk.topic))
        for h in hits
    ]
    scored.sort(key=lambda h: -(h.rerank_score or 0.0))
    return [replace(h, rank=i + 1) for i, h in enumerate(scored)]


def decide(query: Query, f: Filters, hits: list[SearchHit], index: Index, settings: Settings) -> tuple[GateDecision, list[SearchHit]]:
    """Hàm thuần: cùng đầu vào cho cùng quyết định, nên quét ngưỡng không cần truy xuất lại."""
    overview = course_overview(index, f.course_id) if f.course_id else ""
    if not hits:
        reason = "no_chunks_for_week" if f.week is not None else "no_hits"
        return GateDecision(
            action="abstain", reason=reason,
            hint=f"Môn {f.course_id} hiện có: {overview}." if overview else None,
        ), []
    if hits[0].via == "week_scan":
        return GateDecision(action="answer", reason="week_found"), hits
    ranked = rerank(query.text, hits, settings.rerank_dense_weight)
    top1 = ranked[0].rerank_score or 0.0
    if top1 < settings.gate_threshold:
        return GateDecision(
            action="abstain", reason="top1_below_threshold", top1_score=top1, threshold=settings.gate_threshold,
            hint=f"Môn {f.course_id} hiện có: {overview}.",
        ), ranked
    return GateDecision(action="answer", reason="ok", top1_score=top1, threshold=settings.gate_threshold), ranked
