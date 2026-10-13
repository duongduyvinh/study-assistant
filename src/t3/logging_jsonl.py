"""Ghi log mỗi run thành một file jsonl có timestamp trong logs/."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .types import RunContext


def write_run_log(ctx: RunContext, logs_dir: Path, label: str = "run") -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    first_ts = ctx.trace[0].ts if ctx.trace else datetime.now().timestamp()
    stamp = datetime.fromtimestamp(first_ts).strftime("%Y%m%d-%H%M%S")
    path = logs_dir / f"{label}-{stamp}-{ctx.query.run_id}.jsonl"
    path.write_text(ctx.to_jsonl() + "\n", encoding="utf-8")
    return path
