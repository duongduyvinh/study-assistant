"""Đọc cấu hình từ .env và biến môi trường. Một nơi duy nhất, không rải os.getenv khắp code."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    # LLM
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: str = ""
    llm_model: str = "qwen/qwen3.8-27b"
    llm_extra_body: str = "{}"
    llm_timeout: float = 60.0
    llm_max_retries: int = 3
    llm_backoff: float = 1.5
    # Embedding
    gemini_api_key: str = ""
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768
    embedder: str = "gemini"  # gemini | fake
    # Pipeline
    top_k: int = 5
    gate_threshold: float = 0.5
    rrf_k: int = 60
    rerank_dense_weight: float = 0.7
    max_evidence_chunks: int = 6
    max_steps: int = 6
    # Đường dẫn
    root: Path = ROOT
    data_dir: Path = ROOT / "data" / "lesson_plans"
    index_dir: Path = ROOT / "data" / "index"
    notes_path: Path = ROOT / "data" / "notes.json"
    logs_dir: Path = ROOT / "logs"
    prompts_dir: Path = ROOT / "prompts"
    eval_dir: Path = ROOT / "eval_out"
    testset_path: Path = ROOT / "tests" / "testset.jsonl"

    def extra_body(self) -> dict[str, object]:
        try:
            value = json.loads(self.llm_extra_body or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM_EXTRA_BODY không phải JSON hợp lệ: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError("LLM_EXTRA_BODY phải là một đối tượng JSON")
        return value


def _read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_settings(env_file: Path | None = None) -> Settings:
    """Biến môi trường thắng .env."""
    env = _read_dotenv(env_file or ROOT / ".env")
    env.update({k: v for k, v in os.environ.items() if k in _KEYS})

    def get(key: str, default: str) -> str:
        return env.get(key, default)

    base = Settings()
    return Settings(
        llm_base_url=get("LLM_BASE_URL", base.llm_base_url),
        llm_api_key=get("LLM_API_KEY", base.llm_api_key),
        llm_model=get("LLM_MODEL", base.llm_model),
        llm_extra_body=get("LLM_EXTRA_BODY", base.llm_extra_body),
        gemini_api_key=get("GEMINI_API_KEY", base.gemini_api_key),
        embedding_model=get("EMBEDDING_MODEL", base.embedding_model),
        embedding_dim=int(get("EMBEDDING_DIM", str(base.embedding_dim))),
        embedder=get("EMBEDDER", base.embedder),
        top_k=int(get("TOP_K", str(base.top_k))),
        gate_threshold=float(get("GATE_THRESHOLD", str(base.gate_threshold))),
        max_steps=int(get("MAX_STEPS", str(base.max_steps))),
        index_dir=Path(get("INDEX_DIR", str(base.index_dir))),
        notes_path=Path(get("NOTES_PATH", str(base.notes_path))),
        logs_dir=Path(get("LOGS_DIR", str(base.logs_dir))),
        data_dir=Path(get("DATA_DIR", str(base.data_dir))),
    )


_KEYS = {
    "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "LLM_EXTRA_BODY", "GEMINI_API_KEY", "EMBEDDING_MODEL",
    "EMBEDDING_DIM", "EMBEDDER", "TOP_K", "GATE_THRESHOLD", "MAX_STEPS",
    "INDEX_DIR", "NOTES_PATH", "LOGS_DIR", "DATA_DIR",
}
