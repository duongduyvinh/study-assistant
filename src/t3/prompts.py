"""M3: nạp prompt từ prompts/*.txt, dựng khối evidence, kiểm tra trích dẫn dạng [chunk_id]."""
from __future__ import annotations

import re
from pathlib import Path

from .types import Chunk, CitationCheck

NO_EVIDENCE = "KHONG_DU_CAN_CU"
_CITE = re.compile(r"\[([A-Za-z0-9_]+-w\d{2}-[a-z]+)\]")


def load_prompt(prompts_dir: Path, name: str) -> str:
    return (prompts_dir / f"{name}.txt").read_text(encoding="utf-8").strip()


def fill(template: str, **values: str) -> str:
    """Thay {key} bằng giá trị. Không dùng str.format vì evidence có thể chứa dấu ngoặc nhọn."""
    out = template
    for key, value in values.items():
        out = out.replace("{" + key + "}", value)
    return out


def format_chunk(c: Chunk) -> str:
    return f"[{c.chunk_id}] {c.course_id} · Tuần {c.week} · {c.topic} · {c.type} (v{c.version})\n{c.text}"


def build_evidence(chunks: list[Chunk]) -> str:
    """Chỉ đưa vào prompt những trường LLM cần: id, môn, tuần, chủ đề, loại, phiên bản, nội dung."""
    return "\n".join(format_chunk(c) for c in chunks)


def extract_citations(text: str) -> list[str]:
    seen: list[str] = []
    for cid in _CITE.findall(text):
        if cid not in seen:
            seen.append(cid)
    return seen


def check_citations(answer: str, allowed_ids: set[str]) -> CitationCheck:
    cited = extract_citations(answer)
    invalid = tuple(c for c in cited if c not in allowed_ids)
    return CitationCheck(ok=bool(cited) and not invalid, cited=tuple(cited), invalid=invalid)
