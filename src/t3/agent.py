"""M5: agent loop tự viết, không dùng framework orchestration.

Nhận câu hỏi, gọi LLM, parse action, chạy tool, đưa kết quả vào lịch sử, lặp tới khi có câu trả lời
hoặc hết MAX_STEPS. Lịch sử `messages` chỉ sống trong một run, không có memory giữa các câu hỏi.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Literal, Protocol

from .config import Settings
from .prompts import check_citations, extract_citations, fill, load_prompt
from .types import (
    AgentStep, ChatMessage, Filters, LLMFn, Query, RunContext, ToolCall, ToolDescriptor, ToolResult,
)

_NOTE_ID = re.compile(r"note_id=([0-9a-f]+)")
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class ToolBackend(Protocol):
    """ToolBox (gọi thẳng) và MCPBackend (qua stdio) cùng thoả giao diện này."""

    def specs(self) -> list[ToolDescriptor]: ...
    def call(self, name: str, args: dict[str, object]) -> ToolResult: ...


@dataclass(frozen=True)
class ParsedAction:
    kind: Literal["tool", "final", "error"]
    call: ToolCall | None = None
    final: str | None = None
    error: str | None = None


def parse_action(raw: str) -> ParsedAction:
    """Bọc try/except: JSON hỏng hay thiếu ngoặc thì trả lỗi để nhắc LLM thử lại, không làm chết run."""
    text = _FENCE.sub("", raw.strip())
    lo, hi = text.find("{"), text.rfind("}")
    if lo == -1 or hi <= lo:
        return ParsedAction("error", error='Error: không thấy đối tượng JSON. Hãy trả {"tool": ..., "args": {...}} hoặc {"final": ...}')
    try:
        obj = json.loads(text[lo:hi + 1])
    except json.JSONDecodeError as exc:
        return ParsedAction("error", error=f"Error: JSON hỏng ({exc.msg} ở vị trí {exc.pos}). Hãy trả lại đúng một đối tượng JSON.")
    if not isinstance(obj, dict):
        return ParsedAction("error", error="Error: JSON phải là một đối tượng.")
    if "final" in obj:
        return ParsedAction("final", final=str(obj["final"]))
    if "tool" in obj:
        args = obj.get("args", {})
        if not isinstance(obj["tool"], str) or not isinstance(args, dict):
            return ParsedAction("error", error="Error: 'tool' phải là chuỗi và 'args' phải là đối tượng.")
        return ParsedAction("tool", call=ToolCall(name=obj["tool"], args=args))
    return ParsedAction("error", error="Error: thiếu khoá 'tool' hoặc 'final'.")


def _tools_text(specs: list[ToolDescriptor]) -> str:
    return "\n".join(f"- {d.name}({', '.join(d.params)}): {d.description}" for d in specs)


def run_agent(
    text: str,
    course_id: str | None,
    *,
    backend: ToolBackend,
    llm: LLMFn,
    settings: Settings,
    today: date | None = None,
) -> RunContext:
    cid = course_id.strip().upper() if course_id else None
    ctx = RunContext(
        query=Query(text=text, course_id=cid, today=today or date.today()), filters=Filters(course_id=cid)
    )
    ctx.record("run_start", query=text, course_id=cid, mode="agent", max_steps=settings.max_steps)

    specs = backend.specs()
    takes_course = {d.name for d in specs if any(p.rstrip("?") == "course_id" for p in d.params)}
    system = load_prompt(settings.prompts_dir, "system") + "\n\n" + fill(
        load_prompt(settings.prompts_dir, "agent"), tools=_tools_text(specs)
    )
    messages = [ChatMessage("system", system), ChatMessage("user", f"course_id: {cid}\nCâu hỏi: {text}")]

    seen_ids: set[str] = set()
    pending_note: str | None = None
    verify_nudged = cite_nudged = False

    for index in range(1, settings.max_steps + 1):
        raw, usage = llm(messages, "agent")
        ctx.add_usage(usage)
        messages.append(ChatMessage("assistant", raw))
        action = parse_action(raw)

        if action.kind == "error":
            ctx.steps.append(AgentStep(index, None, ToolResult(False, action.error or ""), usage))
            ctx.record("agent_action", index=index, kind="error", error=action.error, raw=raw[:300])
            messages.append(ChatMessage("user", action.error or "Error"))
            continue

        if action.kind == "final":
            final = action.final or ""
            if pending_note is not None and not verify_nudged:
                verify_nudged = True
                ctx.record("agent_action", index=index, kind="final_blocked", reason="chưa verify note")
                messages.append(ChatMessage(
                    "user", "Error: bạn đã lưu ghi chú nhưng chưa gọi list_study_notes để xác nhận. Hãy gọi nó trước khi kết thúc."
                ))
                continue
            check = check_citations(final, seen_ids) if seen_ids else None
            if check is not None and not check.ok and not cite_nudged:
                cite_nudged = True
                ctx.record("agent_action", index=index, kind="final_blocked", reason="trích dẫn", invalid=list(check.invalid))
                messages.append(ChatMessage(
                    "user", "Error: trích dẫn không hợp lệ hoặc thiếu. Chỉ dùng các id đã thấy: " + ", ".join(sorted(seen_ids))
                ))
                continue
            if pending_note is not None:
                ctx.record("verify", ok=False, reason="not_called", note_id=pending_note)
            ctx.steps.append(AgentStep(index, None, None, usage, final=final))
            ctx.record("agent_action", index=index, kind="final")
            ctx.answer = final
            ctx.citation_ok = check.ok if check is not None else None
            if check is not None:
                ctx.record("citations", cited=list(check.cited), invalid=list(check.invalid))
            ctx.status = "answered"
            ctx.record("final", status=ctx.status, citation_ok=ctx.citation_ok, total_tokens=ctx.total_tokens)
            return ctx

        call = action.call
        assert call is not None
        args = dict(call.args)
        if cid is not None and call.name in takes_course:
            args["course_id"] = cid  # dùng đúng mã môn người dùng đưa, LLM không được tự đổi
        ctx.record("tool_call", index=index, name=call.name, args=args)
        result = backend.call(call.name, args)
        ctx.record("tool_result", index=index, name=call.name, ok=result.ok, content=result.content[:600])
        ctx.steps.append(AgentStep(index, ToolCall(call.name, args), result, usage))
        seen_ids.update(extract_citations(result.content))

        if call.name == "save_study_note" and result.ok and (m := _NOTE_ID.search(result.content)):
            pending_note = m.group(1)
        elif call.name == "list_study_notes" and pending_note is not None:
            ctx.record("verify", ok=result.ok and f"note_id={pending_note}" in result.content, note_id=pending_note)
            pending_note = None
        messages.append(ChatMessage("user", f"Observation (tool {call.name}): {result.content}"))

    ctx.status = "error"
    ctx.answer = f"Error: vượt MAX_STEPS={settings.max_steps} mà chưa có câu trả lời cuối."
    ctx.record("final", status=ctx.status, citation_ok=ctx.citation_ok, total_tokens=ctx.total_tokens)
    return ctx
