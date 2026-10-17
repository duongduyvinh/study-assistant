from __future__ import annotations

from dataclasses import replace
from datetime import date

from t3.gate import decide, precheck
from t3.retrieve import infer_types, normalise_scores, retrieve, week_from_text
from t3.types import Filters, Query


def test_week_from_text():
    start = date(2026, 9, 28)
    assert week_from_text("Tuần 2 học gì?", date(2026, 10, 10), start) == 2
    assert week_from_text("tuần này học gì", date(2026, 10, 10), start) == 2
    assert week_from_text("tuần này học gì", date(2026, 9, 1), start) is None  # học kỳ chưa bắt đầu
    assert week_from_text("tuần này học gì", date(2026, 10, 10), None) is None  # không biết ngày bắt đầu
    assert week_from_text("Box model là gì", date(2026, 10, 10), start) is None


def test_infer_types_only_when_single_and_specific():
    assert infer_types("Bài tập tuần 3 là gì?") == ("assessment",)
    assert infer_types("Tuần 2 học gì và có bài tập gì?") == ()
    assert infer_types("Tài liệu và bài tập tuần 1") == ()


def test_week_scan_returns_whole_week_in_type_order(index, embedder, settings):
    hits = retrieve(index, embedder, "Tuần 2 học gì?", Filters("WEB101", 2), mode="hybrid", k=5)
    assert [h.chunk.type for h in hits] == ["concept", "assessment"]
    assert all(h.via == "week_scan" for h in hits)


def test_type_filter_limits_week(index, embedder):
    hits = retrieve(index, embedder, "bài tập", Filters("WEB101", 2, ("assessment",)), mode="hybrid", k=5)
    assert [h.chunk.chunk_id for h in hits] == ["WEB101-w02-assessment"]


def test_course_filter_is_hard(index, embedder):
    for mode in ("dense", "hybrid"):
        hits = retrieve(index, embedder, "JavaScript closure DOM", Filters("WEB101"), mode=mode, k=5)
        assert hits and all(h.chunk.course_id == "WEB101" for h in hits)


def test_hybrid_has_rrf_and_bm25_scores_and_dense_has_none(index, embedder):
    assert normalise_scores([]) == []
    assert normalise_scores([5.0, 5.0]) == [1.0, 1.0]
    assert normalise_scores([2.0, 4.0, 6.0]) == [0.0, 0.5, 1.0]
    hy = retrieve(index, embedder, "Box Model margin padding", Filters("WEB101"), mode="hybrid", k=3)
    assert hy[0].chunk.chunk_id == "WEB101-w02-concept"
    assert hy[0].rrf_score and hy[0].bm25_score == 1.0 and hy[0].dense_score
    de = retrieve(index, embedder, "Box Model margin padding", Filters("WEB101"), mode="dense", k=3)
    assert de[0].rrf_score is None and de[0].via == "dense"


def test_precheck(index):
    q = Query("x")
    assert precheck(q, Filters(None), index).reason == "missing_course_id"
    unk = precheck(q, Filters("ABC999"), index)
    assert unk.action == "clarify" and "WEB101" in (unk.hint or "")
    assert precheck(q, Filters("WEB101"), index) is None


def test_gate_week_missing_abstains_with_hint(index, embedder, settings):
    f = Filters("WEB101", 9)
    hits = retrieve(index, embedder, "Tuần 9 học gì?", f, mode="hybrid", k=5)
    d, _ = decide(Query("Tuần 9 học gì?", "WEB101", 9), f, hits, index, settings)
    assert d.action == "abstain" and d.reason == "no_chunks_for_week" and "tuần 1" in (d.hint or "")


def test_gate_threshold_controls_answer_vs_abstain(index, embedder, settings):
    text, f = "Box model gồm những gì?", Filters("WEB101")
    hits = retrieve(index, embedder, text, f, mode="hybrid", k=5)
    q = Query(text, "WEB101")
    low, ranked = decide(q, f, hits, index, replace(settings, gate_threshold=0.1))
    high, _ = decide(q, f, hits, index, replace(settings, gate_threshold=0.99))
    assert low.action == "answer" and low.reason == "ok" and ranked[0].rerank_score is not None
    assert high.action == "abstain" and high.reason == "top1_below_threshold"
    # decide là hàm thuần: gọi lại cho cùng kết quả
    assert decide(q, f, hits, index, replace(settings, gate_threshold=0.1))[0] == low


def test_gate_abstains_on_out_of_scope(index, embedder, settings):
    text, f = "Cách nấu phở bò?", Filters("WEB101")
    hits = retrieve(index, embedder, text, f, mode="hybrid", k=5)
    assert decide(Query(text, "WEB101"), f, hits, index, settings)[0].action == "abstain"
