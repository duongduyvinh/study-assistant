"""M6: hàm lõi của tool, dùng chung cho agent loop và MCP server.

- Tool không gọi LLM và không đọc RunContext: mọi thứ cần thiết nằm trong tham số.
- Lỗi trả chuỗi bắt đầu bằng "Error: " và không raise, để LLM đọc rồi tự sửa.
"""
from __future__ import annotations

from datetime import date
from typing import Callable

from .config import Settings
from .embeddings import Embedder, make_embedder
from .index import Index, load_index
from .notes import NoteStore
from .pipeline import prepare
from .prompts import format_chunk
from .types import Mode, ToolDescriptor, ToolResult, TYPE_ORDER

MAX_NOTE_CHARS = 1000


class ToolError(Exception):
    """Lỗi do tham số hoặc dữ liệu; được đổi thành 'Error: ...' ở ToolBox.call."""


class ToolBox:
    def __init__(self, index: Index, embedder: Embedder, settings: Settings, notes: NoteStore, today: date | None = None):
        self._index, self._embedder, self._s, self._notes = index, embedder, settings, notes
        self._today = today
        self._handlers: dict[str, tuple[ToolDescriptor, Callable[..., str]]] = {}
        self._register(
            ToolDescriptor(
                "search_lesson_plan",
                "Tìm trong lesson plan của một môn; trả quyết định gate (answer/abstain/clarify) và evidence có [chunk_id].",
                ("query", "course_id", "week?", "mode?"),
            ),
            self.search_lesson_plan,
        )
        self._register(
            ToolDescriptor("get_week_outline", "Lấy toàn bộ nội dung một tuần của một môn.", ("course_id", "week")),
            self.get_week_outline,
        )
        self._register(
            ToolDescriptor("get_course_outline", "Liệt kê các tuần và chủ đề của một môn.", ("course_id",)),
            self.get_course_outline,
        )
        self._register(
            ToolDescriptor(
                "save_study_note", "Lưu ghi chú ôn tập (GHI). Sau khi lưu phải gọi list_study_notes để xác nhận.",
                ("course_id", "week", "note"),
            ),
            self.save_study_note,
        )
        self._register(
            ToolDescriptor("list_study_notes", "Liệt kê ghi chú ôn tập đã lưu của một môn.", ("course_id",)),
            self.list_study_notes,
        )

    # ----- giao diện backend -----
    def _register(self, desc: ToolDescriptor, fn: Callable[..., str]) -> None:
        self._handlers[desc.name] = (desc, fn)

    def specs(self) -> list[ToolDescriptor]:
        return [d for d, _ in self._handlers.values()]

    def call(self, name: str, args: dict[str, object]) -> ToolResult:
        if name not in self._handlers:
            return ToolResult(False, f"Error: tool {name!r} không tồn tại. Các tool có sẵn: {', '.join(self._handlers)}")
        desc, fn = self._handlers[name]
        allowed = {p.rstrip("?") for p in desc.params}
        required = {p for p in desc.params if not p.endswith("?")}
        unknown = set(args) - allowed
        missing = required - set(args)
        if unknown:
            return ToolResult(False, f"Error: tham số lạ {sorted(unknown)} cho {name}. Hợp lệ: {sorted(allowed)}")
        if missing:
            return ToolResult(False, f"Error: thiếu tham số bắt buộc {sorted(missing)} cho {name}")
        try:
            return ToolResult(True, fn(**args))
        except ToolError as exc:
            return ToolResult(False, f"Error: {exc}")
        except Exception as exc:  # noqa: BLE001 - tool không được làm chết vòng lặp
            return ToolResult(False, f"Error: lỗi nội bộ khi chạy {name}: {type(exc).__name__}: {exc}")

    # ----- kiểm tra tham số -----
    def _course(self, course_id: object) -> str:
        cid = str(course_id).strip().upper()
        if cid not in self._index.courses:
            known = ", ".join(self._index.courses)
            raise ToolError(f"không có môn {cid!r}. Các môn hiện có: {known}")
        return cid

    def _week(self, cid: str, week: object) -> int:
        try:
            w = int(week)  # type: ignore[call-overload]
        except (TypeError, ValueError):
            raise ToolError(f"week phải là số nguyên, nhận {week!r}") from None
        mx = self._index.courses[cid].max_week
        if not 1 <= w <= mx:
            raise ToolError(f"môn {cid} chỉ có tuần 1 đến {mx}, không có tuần {w}")
        return w

    # ----- tool đọc -----
    def search_lesson_plan(self, query: str, course_id: str, week: object = None, mode: str = "hybrid") -> str:
        if not str(query).strip():
            raise ToolError("query rỗng")
        if mode not in ("dense", "hybrid"):
            raise ToolError("mode phải là dense hoặc hybrid")
        cid = str(course_id).strip().upper()
        try:
            wk = int(week) if week is not None else None  # type: ignore[call-overload]
        except (TypeError, ValueError):
            raise ToolError(f"week phải là số nguyên, nhận {week!r}") from None
        ctx = prepare(
            str(query), cid, index=self._index, embedder=self._embedder, settings=self._s,
            mode=mode, today=self._today, week=wk,  # type: ignore[arg-type]
        )
        gate = ctx.gate
        assert gate is not None
        head = f"decision: {gate.action} (reason={gate.reason}"
        if gate.top1_score is not None:
            head += f", top1={gate.top1_score:.2f}, threshold={gate.threshold}"
        head += ")"
        lines = [head]
        if gate.action != "answer":
            if gate.hint:
                lines.append(f"hint: {gate.hint}")
        else:
            lines += [format_chunk(h.chunk) for h in ctx.hits[:self._s.max_evidence_chunks]]
        return "\n".join(lines)

    def get_week_outline(self, course_id: str, week: object) -> str:
        cid = self._course(course_id)
        w = self._week(cid, week)
        chunks = sorted(
            (c for c in self._index.chunks if c.course_id == cid and c.week == w),
            key=lambda c: TYPE_ORDER.index(c.type),
        )
        if not chunks:
            raise ToolError(f"môn {cid} không có nội dung ở tuần {w}")
        return "\n".join(format_chunk(c) for c in chunks)

    def get_course_outline(self, course_id: str) -> str:
        cid = self._course(course_id)
        weeks: dict[int, tuple[str, list[str]]] = {}
        for c in self._index.chunks:
            if c.course_id == cid:
                topic, types = weeks.setdefault(c.week, (c.topic, []))
                types.append(c.type)
        info = self._index.courses[cid]
        rows = [f"{cid} · {info.subject}"]
        rows += [f"tuần {w}: {t} ({', '.join(ts)})" for w, (t, ts) in sorted(weeks.items())]
        return "\n".join(rows)

    def list_study_notes(self, course_id: str) -> str:
        cid = self._course(course_id)
        notes = self._notes.list(cid)
        if not notes:
            return f"(chưa có ghi chú nào cho môn {cid})"
        return "\n".join(f"note_id={n.note_id} | tuần {n.week} | {n.text} | {n.created_at}" for n in notes)

    # ----- tool ghi -----
    def save_study_note(self, course_id: str, week: object, note: str) -> str:
        cid = self._course(course_id)
        w = self._week(cid, week)
        text = str(note).strip()
        if not text:
            raise ToolError("note rỗng")
        if len(text) > MAX_NOTE_CHARS:
            raise ToolError(f"note dài quá {MAX_NOTE_CHARS} ký tự")
        saved = self._notes.add(cid, w, text)
        return f"ok: đã lưu note_id={saved.note_id} (môn {cid}, tuần {w})"


def build_toolbox(settings: Settings, today: date | None = None) -> ToolBox:
    embedder = make_embedder(settings)
    index = load_index(settings.index_dir, expected_embedder=embedder.name)
    return ToolBox(index, embedder, settings, NoteStore(settings.notes_path), today=today)
