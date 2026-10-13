"""Luồng một request: parse -> precheck -> retrieve -> gate -> LLM -> kiểm trích dẫn -> kết thúc.

Mỗi lần gọi tạo một RunContext mới và không giữ trạng thái giữa các lần gọi. Chỉ nhánh answer gọi LLM.
"""
from __future__ import annotations

from datetime import date

from .config import Settings
from .embeddings import Embedder
from .gate import decide, precheck
from .http import LLMError
from .index import Index
from .prompts import NO_EVIDENCE, build_evidence, check_citations, fill, load_prompt
from .retrieve import infer_types, retrieve, week_from_text
from .types import ChatMessage, Citation, Filters, LLMFn, Mode, Query, RunContext, Usage

NOT_ENOUGH = "Mình chưa đủ căn cứ trong lesson plan để trả lời chắc chắn câu này."


def start(
    text: str, course_id: str | None, *, index: Index, mode: Mode = "hybrid", today: date | None = None,
    week: int | None = None,
) -> RunContext:
    today = today or date.today()
    cid = course_id.strip().upper() if course_id else None
    info = index.courses.get(cid) if cid else None
    wk = week if week is not None else week_from_text(text, today, info.semester_start if info else None)
    types = infer_types(text)
    query = Query(text=text, course_id=cid, week=wk, mode=mode, today=today)
    ctx = RunContext(query=query, filters=Filters(course_id=cid, week=wk, types=types))
    ctx.record("run_start", query=text, course_id=cid, week=wk, types=list(types), mode=mode, today=today.isoformat())
    return ctx


def retrieve_stage(ctx: RunContext, index: Index, embedder: Embedder, settings: Settings) -> None:
    assert ctx.filters is not None
    ctx.hits = retrieve(
        index, embedder, ctx.query.text, ctx.filters, mode=ctx.query.mode, k=settings.top_k, rrf_k=settings.rrf_k
    )
    ctx.record(
        "retrieval", via=ctx.hits[0].via if ctx.hits else "none", n_hits=len(ctx.hits),
        top=[h.chunk.chunk_id for h in ctx.hits[:settings.top_k]],
    )


def gate_stage(ctx: RunContext, index: Index, settings: Settings) -> None:
    assert ctx.filters is not None
    ctx.gate, ctx.hits = decide(ctx.query, ctx.filters, ctx.hits, index, settings)
    ctx.record(
        "gate", action=ctx.gate.action, reason=ctx.gate.reason, top1_score=ctx.gate.top1_score,
        threshold=ctx.gate.threshold,
    )


def prepare(
    text: str, course_id: str | None, *, index: Index, embedder: Embedder, settings: Settings,
    mode: Mode = "hybrid", today: date | None = None, week: int | None = None,
) -> RunContext:
    """start + precheck + retrieve + gate. Dừng ở quyết định gate, chưa gọi LLM. Tool tìm kiếm cũng dùng hàm này."""
    ctx = start(text, course_id, index=index, mode=mode, today=today, week=week)
    assert ctx.filters is not None
    pre = precheck(ctx.query, ctx.filters, index)
    if pre is not None:
        ctx.gate = pre
        ctx.record("gate", action=pre.action, reason=pre.reason)
        return ctx
    retrieve_stage(ctx, index, embedder, settings)
    gate_stage(ctx, index, settings)
    return ctx


def _finish(ctx: RunContext, status: str, answer: str | None) -> RunContext:
    ctx.status = status  # type: ignore[assignment]
    ctx.answer = answer
    if not ctx.usages:  # nhánh không gọi LLM vẫn có dòng USAGE bằng 0
        ctx.add_usage(Usage(stage="none"))
    ctx.record("final", status=status, citation_ok=ctx.citation_ok, total_tokens=ctx.total_tokens)
    return ctx


def no_llm_answer(ctx: RunContext) -> str:
    assert ctx.gate is not None
    hint = ctx.gate.hint or ""
    if ctx.gate.action == "clarify":
        return hint
    return f"Nội dung này không có trong lesson plan của môn {ctx.query.course_id}. {hint}".strip()


def answer_stage(ctx: RunContext, settings: Settings, llm: LLMFn) -> RunContext:
    chunks = [h.chunk for h in ctx.hits[:settings.max_evidence_chunks]]
    ctx.evidence = chunks
    allowed = {c.chunk_id for c in chunks}
    messages = [
        ChatMessage("system", load_prompt(settings.prompts_dir, "system")),
        ChatMessage(
            "user",
            fill(load_prompt(settings.prompts_dir, "answer"), evidence=build_evidence(chunks), question=ctx.query.text),
        ),
    ]
    text, usage = llm(messages, "answer")
    ctx.add_usage(usage)
    if NO_EVIDENCE in text:
        ctx.record("llm_no_evidence")
        return _finish(ctx, "abstained", f"Lesson plan của môn {ctx.query.course_id} không đủ thông tin để trả lời câu này.")

    check = check_citations(text, allowed)
    if not check.ok:
        ctx.record("citation_retry", cited=list(check.cited), invalid=list(check.invalid))
        messages += [
            ChatMessage("assistant", text),
            ChatMessage(
                "user",
                "Error: trích dẫn không hợp lệ hoặc thiếu. Chỉ dùng các id sau, mỗi ý kèm [id]: " + ", ".join(sorted(allowed)),
            ),
        ]
        text, usage = llm(messages, "answer")
        ctx.add_usage(usage)
        check = check_citations(text, allowed)

    ctx.citation_ok = check.ok
    if not check.ok:
        ctx.record("citation_failed", cited=list(check.cited), invalid=list(check.invalid))
        return _finish(ctx, "abstained", NOT_ENOUGH)

    by_id = {c.chunk_id: c for c in chunks}
    ctx.citations = [
        Citation(chunk_id=cid, course_id=by_id[cid].course_id, week=by_id[cid].week, type=by_id[cid].type)
        for cid in check.cited
    ]
    sources = sorted({(c.course_id, c.week, c.type) for c in ctx.citations})
    footer = "Nguồn: " + "; ".join(f"{cid} tuần {w} ({t})" for cid, w, t in sources)
    return _finish(ctx, "answered", f"{text}\n{footer}")


def run_question(
    text: str,
    course_id: str | None,
    *,
    index: Index,
    embedder: Embedder,
    settings: Settings,
    llm: LLMFn | None = None,
    mode: Mode = "hybrid",
    today: date | None = None,
    generate: bool = True,
) -> RunContext:
    """generate=False dừng sau gate (dùng cho eval không tốn token)."""
    ctx = prepare(text, course_id, index=index, embedder=embedder, settings=settings, mode=mode, today=today)
    assert ctx.gate is not None
    if ctx.gate.action != "answer":
        return _finish(ctx, "abstained" if ctx.gate.action == "abstain" else "clarified", no_llm_answer(ctx))
    if not generate:
        return _finish(ctx, "retrieved", None)

    if llm is None:
        from .llm import default_llm
        llm = default_llm(settings)
    try:
        return answer_stage(ctx, settings, llm)
    except LLMError as exc:
        ctx.record("error", message=str(exc))
        return _finish(ctx, "error", str(exc))
