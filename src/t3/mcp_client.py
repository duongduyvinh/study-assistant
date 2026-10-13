"""M7: MCP client Python. Khám phá tool bằng list_tools() (không hardcode tên tool).

MCPBackend thoả giao diện ToolBackend của agent loop, nên agent có thể chạy qua MCP:
    python -m t3.cli agent "..." --course WEB101 --backend mcp
Chạy riêng để xem tool và thử gọi:  python -m t3.mcp_client --course WEB101
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import threading
from datetime import timedelta

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import ROOT
from .types import ToolDescriptor, ToolResult


def default_server_params(extra_env: dict[str, str] | None = None) -> StdioServerParameters:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), **(extra_env or {})}
    return StdioServerParameters(command=sys.executable, args=["-m", "t3.mcp_server"], env=env)


def _param_names(schema: dict[str, object]) -> tuple[str, ...]:
    props = schema.get("properties", {})
    required = set(schema.get("required", []))  # type: ignore[arg-type]
    return tuple(k if k in required else f"{k}?" for k in props)  # type: ignore[union-attr]


class MCPBackend:
    """Giữ một phiên MCP mở trong một thread riêng, cho agent loop đồng bộ gọi vào."""

    def __init__(self, params: StdioServerParameters | None = None, timeout: float = 60.0) -> None:
        self._params = params or default_server_params()
        self._timeout = timeout
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._stop: asyncio.Event | None = None
        self._session: ClientSession | None = None
        self._specs: list[ToolDescriptor] = []
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise RuntimeError("Error: MCP server không sẵn sàng trong thời gian chờ")
        if self._error is not None:
            raise RuntimeError(f"Error: không kết nối được MCP server: {self._error!r}") from self._error

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._main())

    async def _main(self) -> None:
        self._stop = asyncio.Event()
        try:
            async with stdio_client(self._params) as (read, write):
                async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=self._timeout)) as session:
                    await session.initialize()
                    listed = await session.list_tools()  # discover động
                    self._specs = [
                        ToolDescriptor(
                            name=t.name,
                            description=(t.description or "").strip().splitlines()[0] if t.description else "",
                            params=_param_names(t.inputSchema),
                        )
                        for t in listed.tools
                    ]
                    self._session = session
                    self._ready.set()
                    await self._stop.wait()
        except BaseException as exc:  # noqa: BLE001
            self._error = exc
            self._ready.set()

    def specs(self) -> list[ToolDescriptor]:
        return list(self._specs)

    def call(self, name: str, args: dict[str, object]) -> ToolResult:
        assert self._session is not None
        try:
            fut = asyncio.run_coroutine_threadsafe(self._session.call_tool(name, args), self._loop)
            res = fut.result(self._timeout)
        except Exception as exc:  # noqa: BLE001 - không để tool làm chết vòng lặp agent
            return ToolResult(False, f"Error: gọi MCP tool {name} thất bại: {type(exc).__name__}: {exc}")
        text = "\n".join(c.text for c in res.content if getattr(c, "text", None))
        return ToolResult(ok=not res.isError and not text.startswith("Error:"), content=text)

    def close(self) -> None:
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout=10)


def main() -> None:
    ap = argparse.ArgumentParser(description="Khám phá và thử gọi tool của MCP server t3")
    ap.add_argument("--course", default="WEB101")
    ap.add_argument("--week", type=int, default=2)
    args = ap.parse_args()

    backend = MCPBackend()
    try:
        specs = backend.specs()
        print(f"Phát hiện {len(specs)} tool qua list_tools():")
        for d in specs:
            print(f"  - {d.name}({', '.join(d.params)}): {d.description}")
        names = {d.name for d in specs}
        samples: list[tuple[str, dict[str, object]]] = [
            ("get_course_outline", {"course_id": args.course}),
            ("get_week_outline", {"course_id": args.course, "week": args.week}),
            ("save_study_note", {"course_id": args.course, "week": args.week, "note": "Ôn Box Model trước buổi sau"}),
            ("list_study_notes", {"course_id": args.course}),
        ]
        for name, call_args in samples:
            if name not in names:
                continue
            print(f"\n>>> {name} {call_args}")
            print(backend.call(name, call_args).content)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
