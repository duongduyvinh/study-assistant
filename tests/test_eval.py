from __future__ import annotations

import pytest

from t3.config import Settings
from t3.run_eval import (
    EVAL_TODAY, abstain_metrics, evaluate, load_testset, points_hit, summarize, sweep,
)


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
