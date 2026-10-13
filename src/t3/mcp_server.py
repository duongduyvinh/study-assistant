"""M7: MCP server tự viết bằng mcp SDK, chạy qua stdio.

Stdout là kênh JSON-RPC của giao thức: KHÔNG print trong file này hay trong module nó import.
Mọi log đi vào stderr. Chạy: python -m t3.mcp_server
"""
from __future__ import annotations

import logging
import sys

from mcp.server.fastmcp import FastMCP

from .config import load_settings
from .tools import ToolBox, build_toolbox

log = logging.getLogger("t3.mcp")
mcp = FastMCP("t3-lesson-plan")
_box: ToolBox | None = None


def _toolbox() -> ToolBox:
    global _box
    if _box is None:
        _box = build_toolbox(load_settings())
    return _box


def _run(name: str, **args: object) -> str:
    result = _toolbox().call(name, {k: v for k, v in args.items() if v is not None})
    return result.content  # lỗi đã là chuỗi "Error: ..."


@mcp.tool()
def search_lesson_plan(query: str, course_id: str, week: int | None = None, mode: str = "hybrid") -> str:
    """Tìm trong lesson plan của một môn. Trả quyết định gate (answer, abstain, clarify) và evidence có [chunk_id]."""
    return _run("search_lesson_plan", query=query, course_id=course_id, week=week, mode=mode)


@mcp.tool()
def get_week_outline(course_id: str, week: int) -> str:
    """Lấy toàn bộ nội dung một tuần của một môn."""
    return _run("get_week_outline", course_id=course_id, week=week)


@mcp.tool()
def get_course_outline(course_id: str) -> str:
    """Liệt kê các tuần và chủ đề của một môn."""
    return _run("get_course_outline", course_id=course_id)


@mcp.tool()
def save_study_note(course_id: str, week: int, note: str) -> str:
    """Lưu ghi chú ôn tập (tool GHI). Sau khi lưu hãy gọi list_study_notes để xác nhận."""
    return _run("save_study_note", course_id=course_id, week=week, note=note)


@mcp.tool()
def list_study_notes(course_id: str) -> str:
    """Liệt kê ghi chú ôn tập đã lưu của một môn."""
    return _run("list_study_notes", course_id=course_id)


def main() -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    _toolbox()  # nạp index ngay lúc khởi động để lỗi cấu hình hiện sớm, trên stderr
    log.info("t3 MCP server sẵn sàng (stdio)")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
