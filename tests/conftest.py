from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from t3.config import Settings
from t3.embeddings import FakeEmbedder
from t3.index import Index, build_index, load_index
from t3.ingest import load_all
from t3.notes import NoteStore
from t3.tools import ToolBox
from t3.types import ChatMessage, Usage


@pytest.fixture(scope="session")
def index_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    s = Settings()
    courses, chunks = load_all(s.data_dir)
    out = tmp_path_factory.mktemp("index")
    build_index(chunks, courses, FakeEmbedder(), out)
    return out


@pytest.fixture()
def settings(index_dir: Path, tmp_path: Path) -> Settings:
    # Ngưỡng 0.25 chỉ hợp với embedder giả; ngưỡng thật phải lấy từ make eval
    return replace(
        Settings(), embedder="fake", index_dir=index_dir, notes_path=tmp_path / "notes.json",
        logs_dir=tmp_path / "logs", gate_threshold=0.25, llm_backoff=0.0,
    )


@pytest.fixture(scope="session")
def index(index_dir: Path) -> Index:
    return load_index(index_dir, expected_embedder="fake")


@pytest.fixture()
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture()
def toolbox(index: Index, embedder: FakeEmbedder, settings: Settings) -> ToolBox:
    return ToolBox(index, embedder, settings, NoteStore(settings.notes_path))


class ScriptedLLM:
    """LLM giả: trả lần lượt các câu đã soạn, và nhớ các tin nhắn nhận được."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[list[ChatMessage], str]] = []

    def __call__(self, messages: list[ChatMessage], stage: str) -> tuple[str, Usage]:
        self.calls.append((list(messages), stage))
        if not self.replies:
            raise AssertionError("ScriptedLLM hết câu trả lời soạn sẵn")
        return self.replies.pop(0), Usage(stage=stage, model="fake", prompt_tokens=100, completion_tokens=20)
