from __future__ import annotations

import re
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from t3.config import Settings
from t3.mcp_client import MCPBackend, default_server_params
from t3.run_eval import (
    EVAL_TODAY, abstain_metrics, evaluate, load_testset, points_hit, summarize, sweep,
)


# ---------- MCP qua stdio thật ----------
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


# ---------- eval ----------
def test_testset_meets_assignment_minimums():
    cases = load_testset(Settings().testset_path)
    assert len(cases) >= 12 and sum(c.type == "no_answer" for c in cases) >= 4
    assert {c.type for c in cases} == {"answerable", "near_miss", "no_answer", "ambiguous"}
    assert all(c.question and c.id for c in cases) and len({c.id for c in cases}) == len(cases)


def test_abstain_metrics_math():
    pairs = [("abstain", "abstain"), ("abstain", "answer"), ("answer", "abstain"), ("answer", "answer"), ("answer", "answer")]
    m = abstain_metrics(pairs)
    assert m["abstain_precision"] == 0.5 and m["abstain_recall"] == 0.5 and m["false_refusal_rate"] == pytest.approx(1 / 3)
    assert abstain_metrics([("answer", "answer")])["abstain_precision"] is None


def test_points_hit_is_diacritic_and_case_normalised():
    assert points_hit("Có Box Model và Selector", ("selector", "box model")) == 1.0
    assert points_hit("chỉ Selector", ("selector", "margin")) == 0.5
    assert points_hit(None, ("a",)) is None and points_hit("x", ()) is None


def test_evaluate_runs_both_configs_and_sweep(index, embedder, settings):
    cases = load_testset(settings.testset_path)
    rows = evaluate(settings, index, embedder, cases)
    assert len(rows) == 2 * len(cases) and {r.config for r in rows} == {"dense", "hybrid"}
    summary = {s["config"]: s for s in summarize(rows)}
    assert summary["hybrid"]["tokens"] == 0 and summary["hybrid"]["clarify_accuracy"] == 1.0
    # câu nêu tuần được lọc cứng nên recall = 1; câu ambiguous bị clarify trước khi truy xuất
    by_id = {(r.id, r.config): r for r in rows}
    assert by_id[("q01", "hybrid")].recall_at_k == 1.0 and by_id[("q01", "hybrid")].via == "week_scan"
    assert by_id[("q13", "dense")].action == "clarify" and by_id[("q07", "dense")].action == "abstain"
    sw = sweep(settings, index, embedder, cases)
    hybrid = [r for r in sw if r["config"] == "hybrid"]
    assert len(hybrid) == 17 and hybrid[0]["threshold"] == 0.1 and hybrid[-1]["threshold"] == 0.9
    # ngưỡng cao thì từ chối nhầm không giảm
    assert hybrid[-1]["false_refusal_rate"] >= hybrid[0]["false_refusal_rate"]
    assert EVAL_TODAY.isoformat() == "2026-10-10"


# ---------- agent chạy qua MCP: tool và prompt đều lấy từ list_tools() ----------
def test_agent_over_mcp_uses_discovered_tools(mcp_backend, settings):
    import json
    from t3.agent import run_agent
    from conftest import ScriptedLLM

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
