from __future__ import annotations

import json
from datetime import date

from t3.http import LLMError
from t3.logging_jsonl import write_run_log
from t3.pipeline import NOT_ENOUGH, run_question
from t3.prompts import build_evidence, check_citations, extract_citations, fill

from conftest import ScriptedLLM

TODAY = date(2026, 10, 10)


def ask(index, embedder, settings, text, course, llm=None, **kw):
    return run_question(text, course, index=index, embedder=embedder, settings=settings, llm=llm, today=TODAY, **kw)


def test_answer_branch_cites_and_logs_usage(index, embedder, settings):
    llm = ScriptedLLM(["Tuần 2 học Selector và Box Model [WEB101-w02-concept]. Bài tập: trang trí CV [WEB101-w02-assessment]."])
    ctx = ask(index, embedder, settings, "Tuần 2 học gì?", "web101", llm)
    assert ctx.status == "answered" and ctx.citation_ok is True
    assert ctx.answer.rstrip().splitlines()[-1] == "Nguồn: WEB101 tuần 2 (assessment); WEB101 tuần 2 (concept)"
    assert [c.chunk_id for c in ctx.citations] == ["WEB101-w02-concept", "WEB101-w02-assessment"]
    assert ctx.total_tokens == 120 and ctx.usages[0].stage == "answer"
    # prompt chỉ chứa evidence của tuần 2, không lẫn môn khác
    user_msg = llm.calls[0][0][1].content
    assert "WEB101-w02-concept" in user_msg and "WEB201" not in user_msg and "week_scan" not in user_msg


def test_no_llm_branches_still_log_zero_usage(index, embedder, settings):
    llm = ScriptedLLM([])
    clarify = ask(index, embedder, settings, "Tuần 2 học gì?", None, llm)
    abstain = ask(index, embedder, settings, "Tuần 9 học gì?", "WEB101", llm)
    assert not llm.calls
    assert clarify.status == "clarified" and "mã môn" in clarify.answer
    assert abstain.status == "abstained" and "không có trong lesson plan" in abstain.answer
    for ctx in (clarify, abstain):
        assert ctx.usages[0].stage == "none" and ctx.total_tokens == 0
        assert any(json.loads(line)["event"] == "usage" for line in ctx.to_jsonl().splitlines())


def test_citation_retry_then_success(index, embedder, settings):
    llm = ScriptedLLM(["Học Box Model.", "Học Box Model [WEB101-w02-concept]."])
    ctx = ask(index, embedder, settings, "Tuần 2 học gì?", "WEB101", llm)
    assert ctx.status == "answered" and len(ctx.usages) == 2
    assert any(e.event == "citation_retry" for e in ctx.trace)
    assert llm.calls[1][0][-1].content.startswith("Error:")


def test_invented_source_is_refused_after_retry(index, embedder, settings):
    llm = ScriptedLLM(["Bịa [WEB999-w01-concept]", "Vẫn bịa [WEB999-w01-concept]"])
    ctx = ask(index, embedder, settings, "Tuần 2 học gì?", "WEB101", llm)
    assert ctx.status == "abstained" and ctx.answer == NOT_ENOUGH and ctx.citation_ok is False


def test_llm_can_decline(index, embedder, settings):
    ctx = ask(index, embedder, settings, "Tuần 2 học gì?", "WEB101", ScriptedLLM(["KHONG_DU_CAN_CU"]))
    assert ctx.status == "abstained" and ctx.citations == []


def test_llm_error_becomes_error_status(index, embedder, settings):
    def boom(messages, stage):
        raise LLMError("Error: LLM trả HTTP 500 sau 4 lần thử")

    ctx = ask(index, embedder, settings, "Tuần 2 học gì?", "WEB101", boom)
    assert ctx.status == "error" and ctx.answer.startswith("Error:")


def test_generate_false_stops_after_gate(index, embedder, settings):
    ctx = ask(index, embedder, settings, "Tuần 2 học gì?", "WEB101", generate=False)
    assert ctx.status == "retrieved" and ctx.answer is None and ctx.total_tokens == 0


def test_requests_do_not_share_state(index, embedder, settings):
    a = ask(index, embedder, settings, "Tuần 2 học gì?", "WEB101", generate=False)
    b = ask(index, embedder, settings, "Tuần 2 học gì?", "WEB101", generate=False)
    assert a.query.run_id != b.query.run_id and a.trace is not b.trace
    assert [h.chunk.chunk_id for h in a.hits] == [h.chunk.chunk_id for h in b.hits]


def test_log_file_one_line_per_event_with_run_id(index, embedder, settings):
    ctx = ask(index, embedder, settings, "Tuần 9 học gì?", "WEB101")
    path = write_run_log(ctx, settings.logs_dir, "t")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert {r["run_id"] for r in rows} == {ctx.query.run_id}
    assert [r["event"] for r in rows] == ["run_start", "retrieval", "gate", "usage", "final"]


def test_citation_helpers():
    assert extract_citations("a [WEB101-w01-concept] b [WEB101-w01-concept] [1]") == ["WEB101-w01-concept"]
    assert check_citations("không có nguồn", {"WEB101-w01-concept"}).ok is False
    assert check_citations("ok [WEB101-w01-concept]", {"WEB101-w01-concept"}).ok is True
    assert fill("{evidence} / {question}", evidence="a {b}", question="q") == "a {b} / q"
    assert build_evidence([]) == ""
