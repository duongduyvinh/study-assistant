"""Kiểu dữ liệu dùng chung. Không truyền dict thô giữa các module (TraceEvent.data là ngoại lệ, chỉ logger đọc)."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Callable, Literal

ChunkType = Literal["concept", "assessment", "activity", "reference"]
Action = Literal["answer", "abstain", "clarify"]
Mode = Literal["dense", "hybrid"]
Status = Literal["running", "retrieved", "answered", "abstained", "clarified", "error"]

# Thứ tự hiển thị các loại chunk trong một tuần
TYPE_ORDER: tuple[ChunkType, ...] = ("concept", "activity", "assessment", "reference")


@dataclass(frozen=True)
class CourseInfo:
    course_id: str
    subject: str
    doc_id: str
    max_week: int
    semester_start: date | None = None


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    course_id: str
    subject: str
    week: int
    topic: str
    type: ChunkType
    clo_ids: tuple[str, ...]
    version: str
    owner: str
    text: str


@dataclass(frozen=True)
class Query:
    """Đầu vào bất biến của một request. today được tiêm vào để eval tái lập."""
    text: str
    course_id: str | None = None
    week: int | None = None
    mode: Mode = "hybrid"
    today: date = field(default_factory=date.today)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass(frozen=True)
class Filters:
    course_id: str | None
    week: int | None = None
    types: tuple[ChunkType, ...] = ()


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    rank: int
    via: str = ""  # week_scan | dense | hybrid
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


@dataclass(frozen=True)
class GateDecision:
    action: Action
    reason: str  # ok | week_found | missing_course_id | unknown_course | no_chunks_for_week | no_hits | top1_below_threshold
    top1_score: float | None = None
    threshold: float | None = None
    hint: str | None = None  # gợi ý hiển thị cho người dùng khi abstain hoặc clarify


@dataclass(frozen=True)
class Usage:
    stage: str  # agent | answer | judge | none
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, object]


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    content: str  # khi ok=False, content bắt đầu bằng "Error: "


@dataclass(frozen=True)
class ToolDescriptor:
    """Mô tả một tool cho LLM. Với MCP, được dựng động từ list_tools()."""
    name: str
    description: str
    params: tuple[str, ...]  # tên tham số, thêm dấu ? nếu tuỳ chọn


@dataclass(frozen=True)
class AgentStep:
    index: int
    call: ToolCall | None
    result: ToolResult | None
    usage: Usage
    final: str | None = None


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    course_id: str
    week: int
    type: ChunkType


@dataclass(frozen=True)
class CitationCheck:
    ok: bool
    cited: tuple[str, ...]
    invalid: tuple[str, ...]


@dataclass(frozen=True)
class Note:
    note_id: str
    course_id: str
    week: int
    text: str
    created_at: str


@dataclass(frozen=True)
class TraceEvent:
    ts: float
    event: str
    data: dict[str, object]  # ngoại lệ duy nhất cho dict: chỉ logger đọc


# Hàm gọi LLM: nhận (tin nhắn, tên stage), trả (text, Usage). Test thay bằng hàm giả.
LLMFn = Callable[[list[ChatMessage], str], tuple[str, Usage]]


@dataclass
class RunContext:
    """Trạng thái của đúng một câu hỏi. Tạo khi nhận câu hỏi, bỏ sau khi ghi log."""
    query: Query
    filters: Filters | None = None
    hits: list[SearchHit] = field(default_factory=list)
    gate: GateDecision | None = None
    evidence: list[Chunk] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
    usages: list[Usage] = field(default_factory=list)
    answer: str | None = None
    citations: list[Citation] = field(default_factory=list)
    citation_ok: bool | None = None
    status: Status = "running"
    trace: list[TraceEvent] = field(default_factory=list)

    def record(self, event: str, **data: object) -> None:
        self.trace.append(TraceEvent(time.time(), event, dict(data)))

    def add_usage(self, usage: Usage) -> None:
        self.usages.append(usage)
        self.record("usage", **asdict(usage), total_tokens=usage.total_tokens)

    @property
    def total_tokens(self) -> int:
        return sum(u.total_tokens for u in self.usages)

    def to_jsonl(self) -> str:
        """Một dòng JSON mỗi sự kiện, mọi dòng mang run_id."""
        rid = self.query.run_id
        return "\n".join(
            json.dumps({"run_id": rid, "ts": e.ts, "event": e.event, **e.data}, ensure_ascii=False, default=str)
            for e in self.trace
        )
