#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Sequence

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


def get_default_memory_dir() -> Path:
    env_path = os.getenv("ISSUE_RAG_MEMORY_DIR")
    if env_path:
        env_dir = Path(env_path)
        if env_dir.exists():
            return env_dir
    script_dir = Path(__file__).resolve().parent
    bundle_root = script_dir
    bundle_memory_dir = bundle_root / "csv0_memory_1k"
    if bundle_memory_dir.exists():
        return bundle_memory_dir
    project_memory_dir = bundle_root.parent.parent / "csv0_memory_1k"
    if project_memory_dir.exists():
        return project_memory_dir
    fallback = bundle_root.parent.parent / "csv0_memory_1k"
    if fallback.exists():
        return fallback
    return bundle_memory_dir


class Embedder:
    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        return self._model.encode(
            list(texts),
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )


@dataclass(frozen=True)
class MemoryEntry:
    memory_id: str
    embedding_text: str
    agent_memory: dict[str, Any]
    path: Path


class EmbeddingIndex:
    def __init__(self, entries: Sequence[MemoryEntry], matrix: np.ndarray) -> None:
        self.entries = list(entries)
        self.matrix = matrix
        self._id_to_row = {entry.memory_id: idx for idx, entry in enumerate(self.entries)}

    @classmethod
    def build(cls, entries: Sequence[MemoryEntry], embedder: Embedder) -> "EmbeddingIndex":
        vectors = embedder.encode([entry.embedding_text for entry in entries])
        return cls(entries, vectors)

    def select(self, allowed_ids: set[str]) -> "EmbeddingIndex":
        selected_entries: list[MemoryEntry] = []
        rows: list[np.ndarray] = []
        seen: set[str] = set()
        for entry in self.entries:
            if entry.memory_id in allowed_ids and entry.memory_id not in seen:
                seen.add(entry.memory_id)
                selected_entries.append(entry)
                rows.append(self.matrix[self._id_to_row[entry.memory_id]])
        if not selected_entries:
            msg = f"No memory entries found for: {', '.join(sorted(allowed_ids))}"
            raise ValueError(msg)
        matrix = np.stack(rows) if rows else np.empty((0, 0), dtype=self.matrix.dtype)
        return EmbeddingIndex(selected_entries, matrix)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[MemoryEntry, float]]:
        if not self.entries or query_vector.size == 0 or self.matrix.size == 0:
            return []
        scores = self.matrix @ query_vector
        k = min(max(top_k, 0), len(self.entries))
        if k == 0:
            return []
        indices = np.argsort(scores)[::-1][:k]
        return [(self.entries[index], float(scores[index])) for index in indices]


def normalize_memory_ids(raw_ids: Sequence[str] | None) -> set[str] | None:
    if not raw_ids:
        return None
    normalized: set[str] = set()
    for raw in raw_ids:
        value = raw.strip()
        if value:
            normalized.add(Path(value).stem)
    return normalized or None


def load_memory_entries(memory_dir: Path, focus_ids: set[str] | None) -> list[MemoryEntry]:
    if not memory_dir.exists():
        raise FileNotFoundError(f"Memory directory not found: {memory_dir}")
    entries: list[MemoryEntry] = []
    for path in sorted(memory_dir.glob("*_memory.json")):
        stem = path.stem
        if focus_ids and stem not in focus_ids:
            continue
        data = path.read_text(encoding="utf-8")
        payload = json_load(data)
        search_index = payload.get("search_index") or {}
        desc = search_index.get("description_for_embedding")
        keywords = search_index.get("keywords") or []
        parts: list[str] = []
        if isinstance(desc, str) and desc.strip():
            parts.append(desc.strip())
        if isinstance(keywords, Sequence) and not isinstance(keywords, (str, bytes)):
            keyword_text = " ".join(
                keyword.strip()
                for keyword in keywords
                if isinstance(keyword, str) and keyword.strip()
            )
            if keyword_text:
                parts.append(keyword_text)
        embedding_text = " ".join(parts).strip()
        agent_memory = payload.get("agent_memory")
        if not embedding_text or not isinstance(agent_memory, dict):
            continue
        entries.append(
            MemoryEntry(memory_id=stem, embedding_text=embedding_text, agent_memory=agent_memory, path=path)
        )
    if focus_ids and not entries:
        raise ValueError(f"No memory entries found for: {', '.join(sorted(focus_ids))}")
    if not entries:
        raise ValueError(f"No memory entries loaded from {memory_dir}")
    return entries


def json_load(text: str) -> dict[str, Any]:
    import json

    return json.loads(text)


class RagService:
    def __init__(self, memory_dir: Path, embedding_model: str) -> None:
        self.memory_dir = memory_dir
        self.embedding_model = embedding_model
        self.embedder = Embedder(embedding_model)
        self._lock = Lock()
        self._entries = load_memory_entries(self.memory_dir, None)
        self._index = EmbeddingIndex.build(self._entries, self.embedder)

    def refresh(self) -> None:
        with self._lock:
            self._entries = load_memory_entries(self.memory_dir, None)
            self._index = EmbeddingIndex.build(self._entries, self.embedder)

    def match(self, issue_text: str, top_k: int, focus_ids: set[str] | None) -> list[tuple[MemoryEntry, float]]:
        issue = issue_text.strip()
        if not issue:
            raise ValueError("Issue description must not be empty.")
        with self._lock:
            index = self._index if focus_ids is None else self._index.select(focus_ids)
        issue_vector = self.embedder.encode([issue])
        if issue_vector.size == 0:
            return []
        return index.search(issue_vector[0], top_k)


class MatchRequest(BaseModel):
    issue: str = Field(..., description="Issue description to search with.")
    top_k: int = Field(default=3, ge=0, le=50)
    memory_ids: list[str] | None = Field(default=None, description="Optional list of memory ids to focus on.")


class MatchResponse(BaseModel):
    matches: list[dict[str, Any]]


def create_app(service: RagService) -> FastAPI:
    app = FastAPI(title="issue_rag", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/match", response_model=MatchResponse)
    def match(request: MatchRequest) -> MatchResponse:
        try:
            focus_ids = normalize_memory_ids(request.memory_ids)
            matches = service.match(request.issue, request.top_k, focus_ids)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        payload = [
            {
                "memory_id": entry.memory_id,
                "similarity_score": score,
                "agent_memory": entry.agent_memory,
                "path": str(entry.path),
            }
            for entry, score in matches
        ]
        return MatchResponse(matches=payload)

    @app.post("/refresh")
    def refresh() -> dict[str, str]:
        service.refresh()
        return {"status": "ok"}

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve issue_rag over HTTP.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000).")
    parser.add_argument(
        "--memory-dir",
        type=Path,
        default=None,
        help="Directory containing *_memory.json files (defaults to csv0_memory_1k lookup).",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model identifier.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    memory_dir = args.memory_dir or get_default_memory_dir()
    if not memory_dir.exists():
        raise FileNotFoundError(f"Memory directory not found: {memory_dir}")
    service = RagService(memory_dir.resolve(), args.embedding_model)
    app = create_app(service)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

