"""Study note: trạng thái ghi duy nhất của hệ thống. Không phải memory: không tự động đưa vào prompt."""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from .types import Note


class NoteStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def _load(self) -> list[Note]:
        if not self._path.exists():
            return []
        rows = json.loads(self._path.read_text(encoding="utf-8"))
        return [Note(**r) for r in rows]

    def _save(self, notes: list[Note]) -> None:
        """Ghi nguyên tử: ghi file tạm rồi os.replace, để agent và MCP server cùng ghi không làm hỏng file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(
            json.dumps([n.__dict__ for n in notes], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(tmp, self._path)

    def add(self, course_id: str, week: int, text: str) -> Note:
        notes = self._load()
        note = Note(
            note_id=uuid.uuid4().hex[:8], course_id=course_id, week=week, text=text,
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
        notes.append(note)
        self._save(notes)
        return note

    def list(self, course_id: str) -> list[Note]:
        return [n for n in self._load() if n.course_id == course_id]
