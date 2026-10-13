"""M1/M4: đọc lesson plan Markdown và chia chunk theo (tuần, loại).

Định dạng một file:

    ---
    doc_id: web101-lesson-plan
    course_id: WEB101
    subject: Lập trình web cơ bản
    version: 2026.1
    owner: <giảng viên>
    semester_start: 2026-09-28        # tuỳ chọn, dùng để hiểu "tuần này"
    ---
    ## Tuần 1: Giới thiệu HTML cơ bản
    CLO: CLO1, CLO2                   # tuỳ chọn
    ### Khái niệm | Hoạt động | Bài tập | Tài liệu
    nội dung...
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from .types import Chunk, ChunkType, CourseInfo, TYPE_ORDER

REQUIRED_KEYS = ("doc_id", "course_id", "subject", "version", "owner")

HEADING_TYPE: dict[str, ChunkType] = {
    "khái niệm": "concept",
    "nội dung": "concept",
    "hoạt động": "activity",
    "bài tập": "assessment",
    "đánh giá": "assessment",
    "tài liệu": "reference",
    "tài liệu tham khảo": "reference",
}

TYPE_LABEL_VI: dict[str, str] = {
    "concept": "khái niệm",
    "activity": "hoạt động",
    "assessment": "bài tập",
    "reference": "tài liệu",
}

_WEEK_RE = re.compile(r"^##\s+Tuần\s+(\d+)\s*[:\-–]\s*(.+?)\s*$", re.IGNORECASE)
_SECTION_RE = re.compile(r"^###\s+(.+?)\s*$")
_CLO_RE = re.compile(r"^CLO\s*:\s*(.+)$", re.IGNORECASE)


class LessonPlanError(ValueError):
    pass


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise LessonPlanError("Thiếu front matter (dòng --- ở đầu file)")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        raise LessonPlanError("Front matter không đóng bằng ---") from None
    meta: dict[str, str] = {}
    for raw in lines[1:end]:
        line = re.sub(r"\s+#.*$", "", raw).rstrip()  # bỏ chú thích cuối dòng
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise LessonPlanError(f"Dòng front matter sai định dạng: {raw!r}")
        meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, "\n".join(lines[end + 1:])


def parse_lesson_plan(path: Path) -> tuple[CourseInfo, list[Chunk]]:
    meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_KEYS if not meta.get(k)]
    if missing:
        raise LessonPlanError(f"{path.name}: thiếu {', '.join(missing)} trong front matter")
    start = date.fromisoformat(meta["semester_start"]) if meta.get("semester_start") else None

    chunks: list[Chunk] = []
    week: int | None = None
    topic = ""
    clos: tuple[str, ...] = ()
    section: ChunkType | None = None
    buf: list[str] = []
    seen: set[str] = set()

    def flush() -> None:
        nonlocal buf
        text = "\n".join(buf).strip()
        buf = []
        if section is None or week is None or not text:
            return
        chunk_id = f"{meta['course_id']}-w{week:02d}-{section}"
        if chunk_id in seen:
            raise LessonPlanError(f"{path.name}: trùng mục {section} ở tuần {week}")
        seen.add(chunk_id)
        chunks.append(Chunk(
            chunk_id=chunk_id, doc_id=meta["doc_id"], course_id=meta["course_id"], subject=meta["subject"],
            week=week, topic=topic, type=section, clo_ids=clos, version=meta["version"], owner=meta["owner"],
            text=text,
        ))

    for raw in body.splitlines():
        line = raw.rstrip()
        if (m := _WEEK_RE.match(line)):
            flush()
            week, topic, clos, section = int(m.group(1)), m.group(2), (), None
        elif (m := _SECTION_RE.match(line)):
            flush()
            key = m.group(1).strip().lower()
            if key not in HEADING_TYPE:
                raise LessonPlanError(f"{path.name}: mục lạ {m.group(1)!r}, chỉ nhận {sorted(HEADING_TYPE)}")
            if week is None:
                raise LessonPlanError(f"{path.name}: mục {m.group(1)!r} nằm trước dòng Tuần")
            section = HEADING_TYPE[key]
        elif week is not None and section is None and (m := _CLO_RE.match(line)):
            clos = tuple(c.strip() for c in m.group(1).split(",") if c.strip())
        elif section is not None and not line.lstrip().startswith("<!--"):
            buf.append(line)
    flush()

    if not chunks:
        raise LessonPlanError(f"{path.name}: không tìm thấy chunk nào (cần '## Tuần N: ...' và '### Khái niệm|...')")
    chunks.sort(key=lambda c: (c.week, TYPE_ORDER.index(c.type)))
    info = CourseInfo(
        course_id=meta["course_id"], subject=meta["subject"], doc_id=meta["doc_id"],
        max_week=max(c.week for c in chunks), semester_start=start,
    )
    return info, chunks


def load_all(data_dir: Path) -> tuple[dict[str, CourseInfo], list[Chunk]]:
    courses: dict[str, CourseInfo] = {}
    chunks: list[Chunk] = []
    files = sorted(data_dir.glob("*.md"))
    if not files:
        raise LessonPlanError(f"Không có file .md nào trong {data_dir}")
    for path in files:
        info, cs = parse_lesson_plan(path)
        if info.course_id in courses:
            raise LessonPlanError(f"{path.name}: course_id {info.course_id} đã có ở file khác")
        courses[info.course_id] = info
        chunks.extend(cs)
    return courses, chunks


def embed_text(chunk: Chunk) -> str:
    """Header ngữ cảnh + text. Chỉ dùng lúc embed, không lưu vào payload."""
    header = (
        f"Môn: {chunk.subject} ({chunk.course_id}) | Tuần {chunk.week}: {chunk.topic} | "
        f"Loại: {TYPE_LABEL_VI[chunk.type]}"
    )
    return f"{header}\n{chunk.text}"


def bm25_text(chunk: Chunk) -> str:
    return f"{chunk.course_id} {chunk.topic} {TYPE_LABEL_VI[chunk.type]} {chunk.text}"
