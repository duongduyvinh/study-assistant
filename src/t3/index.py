"""M4: index. Lưu chunk + vector ra jsonl (setup), nạp lại vào Qdrant :memory: và BM25 mỗi process.

Vì Qdrant :memory: mất khi process kết thúc, và chế độ local path chỉ cho một process mở,
mọi process (demo, eval, MCP server) tự nạp lại từ chunks.jsonl. Nạp không gọi API embedding.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from .embeddings import Embedder
from .ingest import bm25_text, embed_text
from .text import tokenize
from .types import Chunk, CourseInfo

COLLECTION = "lesson_plan"


class IndexNotReady(RuntimeError):
    pass


@dataclass
class Index:
    chunks: list[Chunk]
    vectors: np.ndarray  # (n, dim), đã chuẩn hoá
    courses: dict[str, CourseInfo]
    embedder_name: str
    qdrant: QdrantClient
    bm25: BM25Okapi
    by_id: dict[str, Chunk]

    def bm25_scores(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(tokenize(query)), dtype=float)


def chunk_to_payload(c: Chunk) -> dict[str, object]:
    d = asdict(c)
    d["clo_ids"] = list(c.clo_ids)
    return d


def chunk_from_payload(d: dict[str, object]) -> Chunk:
    data = dict(d)
    data["clo_ids"] = tuple(data.get("clo_ids", ()))  # type: ignore[arg-type]
    return Chunk(**data)  # type: ignore[arg-type]


def build_index(chunks: list[Chunk], courses: dict[str, CourseInfo], embedder: Embedder, index_dir: Path) -> None:
    """Embed một lần và ghi data/index/{chunks.jsonl, courses.json, meta.json}."""
    index_dir.mkdir(parents=True, exist_ok=True)
    vectors = embedder.embed_documents([embed_text(c) for c in chunks])
    with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c, v in zip(chunks, vectors):
            f.write(json.dumps({"chunk": chunk_to_payload(c), "vector": v}, ensure_ascii=False) + "\n")
    courses_json = {
        cid: {
            "course_id": ci.course_id, "subject": ci.subject, "doc_id": ci.doc_id, "max_week": ci.max_week,
            "semester_start": ci.semester_start.isoformat() if ci.semester_start else None,
        }
        for cid, ci in courses.items()
    }
    (index_dir / "courses.json").write_text(json.dumps(courses_json, ensure_ascii=False, indent=2), encoding="utf-8")
    (index_dir / "meta.json").write_text(
        json.dumps({"embedder": embedder.name, "dim": embedder.dim, "n_chunks": len(chunks)}), encoding="utf-8"
    )


def load_index(index_dir: Path, expected_embedder: str | None = None) -> Index:
    chunks_path = index_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise IndexNotReady(f"Chưa có index ở {index_dir}. Chạy `make setup` trước.")
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    if expected_embedder is not None and meta["embedder"] != expected_embedder:
        raise IndexNotReady(
            f"Index tạo bằng embedder {meta['embedder']!r} nhưng đang cấu hình {expected_embedder!r}. Chạy lại `make setup`."
        )
    chunks: list[Chunk] = []
    vecs: list[list[float]] = []
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        chunks.append(chunk_from_payload(row["chunk"]))
        vecs.append(row["vector"])
    vectors = np.asarray(vecs, dtype=float)

    courses_raw = json.loads((index_dir / "courses.json").read_text(encoding="utf-8"))
    courses = {
        cid: CourseInfo(
            course_id=d["course_id"], subject=d["subject"], doc_id=d["doc_id"], max_week=d["max_week"],
            semester_start=date.fromisoformat(d["semester_start"]) if d["semester_start"] else None,
        )
        for cid, d in courses_raw.items()
    }

    client = QdrantClient(":memory:")
    client.create_collection(COLLECTION, vectors_config=VectorParams(size=int(meta["dim"]), distance=Distance.COSINE))
    client.upsert(
        COLLECTION,
        points=[PointStruct(id=i, vector=vecs[i], payload=chunk_to_payload(c)) for i, c in enumerate(chunks)],
    )
    bm25 = BM25Okapi([tokenize(bm25_text(c)) for c in chunks])
    return Index(
        chunks=chunks, vectors=vectors, courses=courses, embedder_name=meta["embedder"],
        qdrant=client, bm25=bm25, by_id={c.chunk_id: c for c in chunks},
    )
