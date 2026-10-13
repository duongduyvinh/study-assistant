from __future__ import annotations

from pathlib import Path

import pytest

from t3.config import Settings
from t3.ingest import LessonPlanError, embed_text, load_all, parse_front_matter, parse_lesson_plan
from t3.text import coverage, tokenize


def test_tokenize_keeps_codes_and_makes_bigrams():
    toks = tokenize("Câu lệnh if-else ở WEB101")
    assert "if-else" in toks and "web101" in toks
    assert "câu_lệnh" in toks


def test_coverage_ignores_stopwords():
    assert coverage("Box model là gì?", "Selector, Box Model (margin)") == 1.0
    assert coverage("React hook là gì?", "Selector, Box Model") == 0.0


def test_parse_real_plan_gives_chunk_per_week_and_type():
    info, chunks = parse_lesson_plan(Settings().data_dir / "web101.md")
    assert info.course_id == "WEB101" and info.max_week == 3
    assert [c.chunk_id for c in chunks][:2] == ["WEB101-w01-concept", "WEB101-w01-assessment"]
    assert chunks[0].text.startswith("Cấu trúc thẻ HTML")
    assert "Tuần 1" in embed_text(chunks[0]) and "khái niệm" in embed_text(chunks[0])


def test_load_all_has_overlapping_courses():
    courses, chunks = load_all(Settings().data_dir)
    assert set(courses) == {"WEB101", "WEB201"}
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def _write(tmp_path: Path, body: str, front: str | None = None) -> Path:
    front = front if front is not None else (
        "---\ndoc_id: d\ncourse_id: C1\nsubject: S\nversion: 1\nowner: o\n---\n"
    )
    p = tmp_path / "x.md"
    p.write_text(front + body, encoding="utf-8")
    return p


def test_missing_key_is_reported(tmp_path: Path):
    p = _write(tmp_path, "## Tuần 1: A\n### Khái niệm\nx\n", front="---\ndoc_id: d\n---\n")
    with pytest.raises(LessonPlanError, match="thiếu"):
        parse_lesson_plan(p)


def test_unknown_section_and_duplicates_fail(tmp_path: Path):
    with pytest.raises(LessonPlanError, match="mục lạ"):
        parse_lesson_plan(_write(tmp_path, "## Tuần 1: A\n### Linh tinh\nx\n"))
    with pytest.raises(LessonPlanError, match="trùng"):
        parse_lesson_plan(_write(tmp_path, "## Tuần 1: A\n### Khái niệm\nx\n### Nội dung\ny\n"))


def test_inline_comment_and_clo(tmp_path: Path):
    front = "---\ndoc_id: d\ncourse_id: C1\nsubject: S\nversion: 1\nowner: o\nsemester_start: 2026-09-28  # giả định\n---\n"
    info, chunks = parse_lesson_plan(_write(tmp_path, "## Tuần 1: A\nCLO: CLO1, CLO2\n### Bài tập\nlàm bài\n", front))
    assert info.semester_start.isoformat() == "2026-09-28"
    assert chunks[0].clo_ids == ("CLO1", "CLO2") and chunks[0].type == "assessment"


def test_front_matter_required():
    with pytest.raises(LessonPlanError):
        parse_front_matter("không có front matter")
