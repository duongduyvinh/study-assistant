"""M8: make eval. Dense so với hybrid trên cùng testset, kèm bảng quét ngưỡng gate.

Chạy mặc định không gọi LLM (chỉ truy xuất + gate) nên không tốn token. Thêm --with-llm để đo
độ đúng câu trả lời, citation hợp lệ và tổng token.

Số liệu chỉ có ý nghĩa khi EMBEDDER=gemini. Với EMBEDDER=fake chỉ để kiểm tra code chạy.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

from .config import Settings, load_settings
from .embeddings import Embedder, make_embedder
from .gate import decide, precheck
from .http import LLMError
from .index import Index, IndexNotReady, load_index
from .pipeline import run_question, start
from .retrieve import retrieve
from .text import normalize
from .types import Mode

EVAL_TODAY = date(2026, 10, 10)  # cố định để "tuần này" trong testset tái lập được
EXPECTED_ACTION = {"answerable": "answer", "near_miss": "abstain", "no_answer": "abstain", "ambiguous": "clarify"}
MODES: tuple[Mode, ...] = ("dense", "hybrid")


@dataclass(frozen=True)
class TestCase:
    id: str
    question: str
    course_id: str | None
    type: str  # answerable | near_miss | no_answer | ambiguous
    gold_doc_ids: tuple[str, ...]
    expected_points: tuple[str, ...]
    expected_behavior: str


@dataclass(frozen=True)
class Row:
    id: str
    type: str
    config: str
    action: str
    expected: str
    action_correct: bool
    reason: str
    top1: float | None
    via: str
    recall_at_k: float | None
    status: str
    citation_ok: bool | None
    tokens: int
    points_hit: float | None


def load_testset(path: Path) -> list[TestCase]:
    cases: list[TestCase] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            cases.append(TestCase(
                id=d["id"], question=d["question"], course_id=d.get("course_id"), type=d["type"],
                gold_doc_ids=tuple(d.get("gold_doc_ids", ())), expected_points=tuple(d.get("expected_points", ())),
                expected_behavior=d.get("expected_behavior", ""),
            ))
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(f"{path.name} dòng {n}: {exc!r}") from exc
        if cases[-1].type not in EXPECTED_ACTION:
            raise ValueError(f"{path.name} dòng {n}: type {cases[-1].type!r} không hợp lệ")
    return cases


def points_hit(answer: str | None, points: tuple[str, ...]) -> float | None:
    if not points or answer is None:
        return None
    a = normalize(answer)
    return sum(normalize(p) in a for p in points) / len(points)


def evaluate(
    settings: Settings, index: Index, embedder: Embedder, cases: list[TestCase], *, with_llm: bool = False,
    judge: bool = False,
) -> list[Row]:
    rows: list[Row] = []
    for mode in MODES:
        for case in cases:
            ctx = run_question(
                case.question, case.course_id, index=index, embedder=embedder, settings=settings,
                mode=mode, today=EVAL_TODAY, generate=with_llm,
            )
            assert ctx.gate is not None
            k = settings.top_k
            got_docs = {h.chunk.doc_id for h in ctx.hits[:k]}
            recall = len(got_docs & set(case.gold_doc_ids)) / len(case.gold_doc_ids) if case.gold_doc_ids else None
            hit = points_hit(ctx.answer, case.expected_points) if with_llm else None
            if judge and ctx.answer and case.expected_points:
                from .judge import judge_points
                hit = judge_points(ctx.answer, case.expected_points)
            expected = EXPECTED_ACTION[case.type]
            rows.append(Row(
                id=case.id, type=case.type, config=mode, action=ctx.gate.action, expected=expected,
                action_correct=ctx.gate.action == expected, reason=ctx.gate.reason, top1=ctx.gate.top1_score,
                via=ctx.hits[0].via if ctx.hits else "none", recall_at_k=recall, status=ctx.status,
                citation_ok=ctx.citation_ok, tokens=ctx.total_tokens, points_hit=hit,
            ))
    return rows


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def abstain_metrics(pairs: list[tuple[str, str]]) -> dict[str, float | None]:
    """pairs = (expected_action, action). Dương tính = nên abstain."""
    tp = sum(e == "abstain" and a == "abstain" for e, a in pairs)
    pred = sum(a == "abstain" for _, a in pairs)
    pos = sum(e == "abstain" for e, _ in pairs)
    answerable = [(e, a) for e, a in pairs if e == "answer"]
    return {
        "abstain_precision": tp / pred if pred else None,
        "abstain_recall": tp / pos if pos else None,
        "false_refusal_rate": sum(a != "answer" for _, a in answerable) / len(answerable) if answerable else None,
    }


def summarize(rows: list[Row]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for mode in MODES:
        rs = [r for r in rows if r.config == mode]
        if not rs:
            continue
        m = abstain_metrics([(r.expected, r.action) for r in rs])
        amb = [r for r in rs if r.type == "ambiguous"]
        out.append({
            "config": mode, "n": len(rs),
            "recall@k": _mean([r.recall_at_k for r in rs if r.recall_at_k is not None]),
            "recall@k_no_week_scan": _mean([r.recall_at_k for r in rs if r.recall_at_k is not None and r.via != "week_scan"]),
            "action_accuracy": _mean([float(r.action_correct) for r in rs]),
            **m,
            "clarify_accuracy": _mean([float(r.action_correct) for r in amb]),
            "citation_ok_rate": _mean([float(r.citation_ok) for r in rs if r.citation_ok is not None]),
            "points_hit": _mean([r.points_hit for r in rs if r.points_hit is not None]),
            "tokens": sum(r.tokens for r in rs),
        })
    return out


def sweep(settings: Settings, index: Index, embedder: Embedder, cases: list[TestCase]) -> list[dict[str, object]]:
    """Truy xuất một lần mỗi câu, rồi chạy lại decide() ở từng ngưỡng (decide là hàm thuần)."""
    thresholds = [round(0.10 + 0.05 * i, 2) for i in range(17)]
    out: list[dict[str, object]] = []
    for mode in MODES:
        prepared = []
        for case in cases:
            ctx = start(case.question, case.course_id, index=index, mode=mode, today=EVAL_TODAY)
            assert ctx.filters is not None
            pre = precheck(ctx.query, ctx.filters, index)
            hits = [] if pre else retrieve(
                index, embedder, case.question, ctx.filters, mode=mode, k=settings.top_k, rrf_k=settings.rrf_k
            )
            prepared.append((case, ctx, pre, hits))
        for t in thresholds:
            s = replace(settings, gate_threshold=t)
            pairs: list[tuple[str, str]] = []
            for case, ctx, pre, hits in prepared:
                assert ctx.filters is not None
                decision = pre if pre else decide(ctx.query, ctx.filters, hits, index, s)[0]
                pairs.append((EXPECTED_ACTION[case.type], decision.action))
            out.append({"config": mode, "threshold": t, **abstain_metrics(pairs)})
    return out


def _fmt(v: object) -> str:
    return "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


def print_table(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    cols = list(rows[0])
    widths = [max(len(c), *(len(_fmt(r.get(c))) for r in rows)) for c in cols]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for r in rows:
        print("  ".join(_fmt(r.get(c)).ljust(w) for c, w in zip(cols, widths)))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Đánh giá dense so với hybrid trên testset")
    ap.add_argument("--testset", type=Path)
    ap.add_argument("--out", type=Path, help="thư mục ghi kết quả (mặc định eval_out/)")
    ap.add_argument("--with-llm", action="store_true", help="gọi LLM để đo độ đúng, citation, token")
    ap.add_argument("--judge", action="store_true", help="thêm judge Gemma 3 1B qua Ollama (chưa test)")
    ap.add_argument("--skip-sweep", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    settings = load_settings()
    try:
        embedder = make_embedder(settings)
        index = load_index(settings.index_dir, expected_embedder=embedder.name)
        cases = load_testset(args.testset or settings.testset_path)
    except (IndexNotReady, LLMError, ValueError, FileNotFoundError) as exc:
        print(f"{exc}", file=sys.stderr)
        sys.exit(1)
    out = args.out or settings.eval_dir

    if embedder.name == "fake":
        print("CẢNH BÁO: EMBEDDER=fake, số liệu bên dưới không phản ánh chất lượng thật.\n")
    rows = evaluate(settings, index, embedder, cases, with_llm=args.with_llm, judge=args.judge)
    write_csv(out / "results.csv", [r.__dict__ for r in rows])
    summary = summarize(rows)
    write_csv(out / "summary.csv", summary)
    print(f"Testset: {len(cases)} câu, top_k={settings.top_k}, ngưỡng gate={settings.gate_threshold}\n")
    print_table(summary)
    if not args.skip_sweep:
        sw = sweep(settings, index, embedder, cases)
        write_csv(out / "sweep.csv", sw)
        print("\nQuét ngưỡng gate (chọn điểm cân bằng giữa từ chối nhầm và bịa ngoài chương trình):")
        print_table(sw)
    print(f"\nĐã ghi {out}/results.csv, summary.csv" + ("" if args.skip_sweep else ", sweep.csv"))


if __name__ == "__main__":
    main()
