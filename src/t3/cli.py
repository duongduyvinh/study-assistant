"""Điểm vào dòng lệnh: setup, ask, agent, demo, eval."""
from __future__ import annotations

import argparse
import logging
import sys

from .agent import ToolBackend, run_agent
from .config import Settings, load_settings
from .embeddings import make_embedder
from .http import LLMError
from .index import IndexNotReady, build_index, load_index
from .ingest import LessonPlanError, load_all
from .llm import default_llm
from .logging_jsonl import write_run_log
from .pipeline import run_question
from .tools import build_toolbox
from .types import Mode, RunContext


def cmd_setup(s: Settings) -> None:
    courses, chunks = load_all(s.data_dir)
    embedder = make_embedder(s)
    build_index(chunks, courses, embedder, s.index_dir)
    print(f"Đã tạo index: {len(chunks)} chunk, {len(courses)} môn, embedder={embedder.name}, thư mục {s.index_dir}")


def _show(ctx: RunContext, s: Settings, label: str) -> None:
    gate = f"{ctx.gate.action} ({ctx.gate.reason})" if ctx.gate else "-"
    print(f"\n--- {label}\ncâu hỏi : {ctx.query.text}  [môn={ctx.query.course_id}]")
    print(f"gate     : {gate}")
    print(f"trạng thái: {ctx.status} | token: {ctx.total_tokens}")
    print(f"trả lời  :\n{ctx.answer}")
    print(f"log      : {write_run_log(ctx, s.logs_dir, label)}")


def cmd_ask(s: Settings, question: str, course: str | None, mode: Mode) -> None:
    index = load_index(s.index_dir, expected_embedder=make_embedder(s).name)
    ctx = run_question(question, course, index=index, embedder=make_embedder(s), settings=s, mode=mode)
    _show(ctx, s, "ask")


def _agent(s: Settings, question: str, course: str | None, backend: ToolBackend, label: str) -> None:
    ctx = run_agent(question, course, backend=backend, llm=default_llm(s), settings=s)
    _show(ctx, s, label)


def cmd_agent(s: Settings, question: str, course: str | None, backend_name: str) -> None:
    if backend_name == "mcp":
        from .mcp_client import MCPBackend
        backend = MCPBackend()
        try:
            _agent(s, question, course, backend, "agent-mcp")
        finally:
            backend.close()
    else:
        _agent(s, question, course, build_toolbox(s), "agent")


def cmd_demo(s: Settings) -> None:
    embedder = make_embedder(s)
    index = load_index(s.index_dir, expected_embedder=embedder.name)
    scenarios: list[tuple[str, str | None, str]] = [
        ("Tuần 2 học gì?", "WEB101", "demo-answer"),          # nhánh answer, có LLM
        ("Bài tập tuần 3 là gì?", "WEB101", "demo-assessment"),  # lọc theo loại, có LLM
        ("Tuần 2 học gì?", None, "demo-clarify"),               # thiếu mã môn, không LLM
        ("Tuần 9 học gì?", "WEB101", "demo-abstain"),           # tuần không tồn tại, không LLM
    ]
    for question, course, label in scenarios:
        ctx = run_question(question, course, index=index, embedder=embedder, settings=s)
        _show(ctx, s, label)
    # Agent: tool ghi + bước verify hiện trong trace
    _agent(
        s, "Tóm tắt tuần 2 rồi lưu ghi chú 'Ôn lại Box Model trước buổi sau' cho tuần 2.", "WEB101",
        build_toolbox(s), "demo-agent-note",
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(prog="t3")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup", help="đọc lesson plan, embed, ghi index")
    p = sub.add_parser("ask", help="hỏi một câu qua pipeline")
    p.add_argument("question")
    p.add_argument("--course")
    p.add_argument("--mode", choices=["dense", "hybrid"], default="hybrid")
    p = sub.add_parser("agent", help="hỏi qua agent loop")
    p.add_argument("question")
    p.add_argument("--course")
    p.add_argument("--backend", choices=["local", "mcp"], default="local")
    sub.add_parser("demo", help="kịch bản mẫu, mỗi lần chạy ghi một file log")
    args = ap.parse_args(argv)

    s = load_settings()
    try:
        if args.cmd == "setup":
            cmd_setup(s)
        elif args.cmd == "ask":
            cmd_ask(s, args.question, args.course, args.mode)
        elif args.cmd == "agent":
            cmd_agent(s, args.question, args.course, args.backend)
        elif args.cmd == "demo":
            cmd_demo(s)
    except (IndexNotReady, LLMError, LessonPlanError) as exc:
        print(f"{exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
