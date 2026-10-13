from __future__ import annotations

import json

from t3.agent import parse_action, run_agent
from t3.config import Settings

from conftest import ScriptedLLM


def j(**kw) -> str:
    return json.dumps(kw, ensure_ascii=False)


# ---------- tools ----------
def test_tool_errors_are_strings_not_exceptions(toolbox):
    assert toolbox.call("nope", {}).content.startswith("Error: tool 'nope'")
    assert toolbox.call("get_week_outline", {"course_id": "WEB101"}).content.startswith("Error: thiếu tham số")
    assert toolbox.call("get_week_outline", {"course_id": "WEB101", "week": 9}).content.startswith("Error: môn WEB101 chỉ có tuần 1 đến 3")
    assert toolbox.call("get_week_outline", {"course_id": "WEB101", "week": "abc"}).content.startswith("Error: week phải là số")
    assert toolbox.call("get_week_outline", {"course_id": "XXX", "week": 1}).content.startswith("Error: không có môn")
    assert toolbox.call("get_course_outline", {"course_id": "WEB101", "bậy": 1}).content.startswith("Error: tham số lạ")
    assert not toolbox.call("save_study_note", {"course_id": "WEB101", "week": 2, "note": "  "}).ok


def test_search_tool_returns_decision_and_ids(toolbox):
    ok = toolbox.call("search_lesson_plan", {"query": "Tuần 2 học gì?", "course_id": "web101"})
    assert ok.ok and ok.content.startswith("decision: answer (reason=week_found)")
    assert "[WEB101-w02-concept]" in ok.content
    abstain = toolbox.call("search_lesson_plan", {"query": "Tuần 9 học gì?", "course_id": "WEB101"})
    assert abstain.ok and "decision: abstain" in abstain.content and "hint:" in abstain.content
    clar = toolbox.call("search_lesson_plan", {"query": "Tuần 2 học gì?", "course_id": "ZZZ"})
    assert "decision: clarify (reason=unknown_course)" in clar.content


def test_write_tool_then_verify_by_read(toolbox):
    saved = toolbox.call("save_study_note", {"course_id": "WEB101", "week": 2, "note": "Ôn Box Model"})
    assert saved.ok and "note_id=" in saved.content
    note_id = saved.content.split("note_id=")[1].split(" ")[0]
    listed = toolbox.call("list_study_notes", {"course_id": "WEB101"})
    assert f"note_id={note_id}" in listed.content and "Ôn Box Model" in listed.content
    assert "chưa có ghi chú" in toolbox.call("list_study_notes", {"course_id": "WEB201"}).content


def test_at_least_one_write_tool_and_three_read_tools(toolbox):
    names = {d.name for d in toolbox.specs()}
    assert {"save_study_note", "list_study_notes", "search_lesson_plan", "get_week_outline", "get_course_outline"} <= names


# ---------- parse_action ----------
def test_parse_action_variants():
    assert parse_action(j(tool="get_week_outline", args={"course_id": "WEB101", "week": 2})).kind == "tool"
    assert parse_action("```json\n" + j(final="xong") + "\n```").final == "xong"
    assert parse_action("Tôi sẽ gọi tool:\n" + j(tool="x", args={}) + "\nxong").kind == "tool"
    for bad in ('{"tool": "x", "args": {"a": 1}', "không có json", "[1, 2]", '{"tool": 5}', '{"foo": 1}', '{"tool": "x", "args": 3}'):
        r = parse_action(bad)
        assert r.kind == "error" and r.error.startswith("Error:"), bad


# ---------- agent loop ----------
def agent(toolbox, settings, replies, question="Tuần 2 học gì?", course="WEB101"):
    llm = ScriptedLLM(replies)
    return run_agent(question, course, backend=toolbox, llm=llm, settings=settings), llm


def test_agent_search_then_final_with_citation(toolbox, settings):
    ctx, llm = agent(toolbox, settings, [
        j(tool="search_lesson_plan", args={"query": "Tuần 2 học gì?", "course_id": "WEB101"}),
        j(final="Tuần 2 học Selector, Box Model [WEB101-w02-concept]."),
    ])
    assert ctx.status == "answered" and ctx.citation_ok is True and len(ctx.steps) == 2
    assert ctx.total_tokens == 240
    assert "Observation (tool search_lesson_plan)" in llm.calls[1][0][-1].content
    events = [e.event for e in ctx.trace]
    assert events.count("tool_call") == 1 and "tool_result" in events


def test_agent_recovers_from_broken_json_and_tool_error(toolbox, settings):
    ctx, llm = agent(toolbox, settings, [
        '{"tool": "get_week_outline", "args": {"course_id": "WEB101",}}',  # JSON hỏng (thừa dấu phẩy)
        j(tool="get_week_outline", args={"course_id": "WEB101", "week": 9}),  # tool trả Error:
        j(tool="get_week_outline", args={"course_id": "WEB101", "week": 2}),
        j(final="Tuần 2: Selector, Box Model [WEB101-w02-concept]"),
    ])
    assert ctx.status == "answered"
    assert llm.calls[1][0][-1].content.startswith("Error: JSON hỏng")
    assert llm.calls[2][0][-1].content.startswith("Observation (tool get_week_outline): Error: môn WEB101 chỉ có tuần")


def test_agent_forces_user_course_id(toolbox, settings):
    ctx, _ = agent(toolbox, settings, [
        j(tool="get_course_outline", args={"course_id": "WEB201"}),  # LLM tự đổi môn
        j(final="xong"),
    ])
    call = ctx.steps[0].call
    assert call.args["course_id"] == "WEB101" and "WEB101" in ctx.steps[0].result.content


def test_agent_must_verify_after_write(toolbox, settings):
    ctx, llm = agent(toolbox, settings, [
        j(tool="save_study_note", args={"course_id": "WEB101", "week": 2, "note": "Ôn Box Model"}),
        j(final="Đã lưu."),  # quên verify -> bị chặn
        j(tool="list_study_notes", args={"course_id": "WEB101"}),
        j(final="Đã lưu ghi chú và xác nhận."),
    ])
    assert ctx.status == "answered"
    assert llm.calls[2][0][-1].content.startswith("Error: bạn đã lưu ghi chú nhưng chưa gọi list_study_notes")
    verify = [e for e in ctx.trace if e.event == "verify"]
    assert len(verify) == 1 and verify[0].data["ok"] is True


def test_agent_unverified_write_is_recorded(toolbox, settings):
    ctx, _ = agent(toolbox, settings, [
        j(tool="save_study_note", args={"course_id": "WEB101", "week": 2, "note": "x"}),
        j(final="Đã lưu."), j(final="Đã lưu thật."),
    ])
    assert ctx.status == "answered"
    assert [e.data["ok"] for e in ctx.trace if e.event == "verify"] == [False]


def test_agent_invalid_citation_is_nudged_once(toolbox, settings):
    ctx, llm = agent(toolbox, settings, [
        j(tool="get_week_outline", args={"course_id": "WEB101", "week": 2}),
        j(final="Tuần 2 học CSS [WEB999-w01-concept]"),
        j(final="Tuần 2 học CSS [WEB101-w02-concept]"),
    ])
    assert ctx.citation_ok is True and llm.calls[2][0][-1].content.startswith("Error: trích dẫn")


def test_agent_stops_at_max_steps(toolbox, settings):
    import dataclasses
    s = dataclasses.replace(settings, max_steps=3)
    loop = j(tool="get_course_outline", args={"course_id": "WEB101"})
    ctx, llm = agent(toolbox, s, [loop, loop, loop, loop])
    assert ctx.status == "error" and "MAX_STEPS=3" in ctx.answer and len(llm.calls) == 3
