from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from conftest import ScriptedLLM
from t3.agent import run_agent
from t3.mcp_client import MCPBackend, default_server_params


@pytest.fixture()
def mcp_backend(index_dir: Path, tmp_path: Path):
    env = {"EMBEDDER": "fake", "INDEX_DIR": str(index_dir), "NOTES_PATH": str(tmp_path / "notes.json")}
    backend = MCPBackend(default_server_params(env), timeout=60)
    yield backend
    backend.close()


def test_mcp_discovers_tools_dynamically_and_stdout_is_clean(mcp_backend):
    specs = {d.name: d for d in mcp_backend.specs()}
    assert len(specs) >= 2
    assert {"save_study_note", "list_study_notes", "search_lesson_plan"} <= set(specs)
    assert "query" in specs["search_lesson_plan"].params and "week?" in specs["search_lesson_plan"].params
    # Nếu server in bừa ra stdout thì khung JSON-RPC đã hỏng và đến đây là không gọi được
    assert "WEB101" in mcp_backend.call("get_course_outline", {"course_id": "WEB101"}).content


def test_mcp_write_then_verify(mcp_backend):
    saved = mcp_backend.call("save_study_note", {"course_id": "WEB101", "week": 2, "note": "Ôn qua MCP"})
    assert saved.ok and "note_id=" in saved.content
    assert "Ôn qua MCP" in mcp_backend.call("list_study_notes", {"course_id": "WEB101"}).content
    err = mcp_backend.call("get_week_outline", {"course_id": "WEB101", "week": 99})
    assert not err.ok and err.content.startswith("Error:")


def test_no_print_in_modules_the_mcp_server_imports():
    src = Path(__file__).resolve().parents[1] / "src" / "t3"
    for name in ("mcp_server.py", "tools.py", "pipeline.py", "retrieve.py", "gate.py", "index.py", "notes.py",
                 "ingest.py", "embeddings.py", "llm.py", "http.py", "config.py", "text.py", "prompts.py", "types.py"):
        assert not re.search(r"^\s*print\(", (src / name).read_text(encoding="utf-8"), re.M), name


def test_agent_over_mcp_uses_discovered_tools(mcp_backend, settings):
    llm = ScriptedLLM([
        json.dumps({"tool": "save_study_note", "args": {"course_id": "WEB201", "week": 1, "note": "Ôn closure"}}),
        json.dumps({"tool": "list_study_notes", "args": {"course_id": "WEB201"}}),
        json.dumps({"final": "Đã lưu và xác nhận ghi chú."}),
    ])
    ctx = run_agent("Lưu ghi chú ôn closure cho tuần 1", "WEB201", backend=mcp_backend, llm=llm, settings=settings)
    system = llm.calls[0][0][0].content
    assert all(d.name in system for d in mcp_backend.specs())  # prompt dựng từ tool phát hiện được
    assert ctx.status == "answered"
    verify = [e for e in ctx.trace if e.event == "verify"]
    assert verify and verify[0].data["ok"] is True
