from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np


class Embedder:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Install the 'sentence_transformers' package to run embedding retrieval."
            ) from exc
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
    text: str
    path: Path


@dataclass(frozen=True)
class EmbeddingIndex:
    entries: Sequence[MemoryEntry]
    matrix: np.ndarray

    @classmethod
    def build(cls, entries: Sequence[MemoryEntry], embedder: Embedder) -> EmbeddingIndex:
        vectors = embedder.encode([entry.text for entry in entries])
        return cls(entries=list(entries), matrix=vectors)

    def search(self, query_vector: np.ndarray, top_k: int) -> List[tuple[MemoryEntry, float]]:
        if len(self.entries) == 0:
            return []
        scores = self.matrix @ query_vector
        k = min(max(top_k, 0), len(self.entries))
        if k == 0:
            return []
        indices = np.argsort(scores)[::-1][:k]
        return [(self.entries[i], float(scores[i])) for i in indices]


def load_memory_entries(memory_dir: Path, focus: str | None) -> List[MemoryEntry]:
    entries: List[MemoryEntry] = []
    target = None
    if focus:
        target = focus if focus.endswith(".json") else f"{focus}.json"
    for path in sorted(memory_dir.glob("*_memory.json")):
        if target and path.name != target:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        search_index = data.get("search_index", {})
        desc = search_index.get("description_for_embedding", "") or ""
        keywords: Iterable[str] = search_index.get("keywords", []) or []
        combined = " ".join(part for part in [desc.strip(), " ".join(keywords)] if part)
        if not combined:
            continue
        entries.append(MemoryEntry(memory_id=path.stem, text=combined, path=path))
    if target and not entries:
        raise ValueError(f"No memory entries found for '{focus}'.")
    if not entries:
        raise ValueError(f"No memory entries loaded from {memory_dir}.")
    return entries


def load_issues(
    dataset_name: str,
    split: str,
    field: str,
    instance_filters: set[str] | None,
    limit: int | None,
) -> List[dict]:
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install the 'datasets' package to load SWE-bench Lite.") from exc
    dataset = load_dataset(dataset_name, split=split)
    issues: List[dict] = []
    for row in dataset:
        instance_id = row.get("instance_id", "")
        if instance_filters and instance_id not in instance_filters:
            continue
        text = row.get(field)
        if not isinstance(text, str) or not text.strip():
            continue
        issues.append(
            {
                "instance_id": instance_id,
                "repo": row.get("repo", ""),
                "text": text,
            }
        )
        if limit and len(issues) >= limit:
            break
    if instance_filters and not issues:
        raise ValueError(f"No issues found for: {', '.join(sorted(instance_filters))}")
    if not issues:
        raise ValueError("No issues loaded from dataset.")
    return issues


def match_issues_to_memory(
    issues: Sequence[dict],
    memories: Sequence[MemoryEntry],
    top_k: int,
    embedder: Embedder,
) -> List[tuple[dict, List[tuple[MemoryEntry, float]]]]:
    index = EmbeddingIndex.build(memories, embedder)
    issue_vectors = embedder.encode([issue["text"] for issue in issues])
    results: List[tuple[dict, List[tuple[MemoryEntry, float]]]] = []
    for issue, vector in zip(issues, issue_vectors):
        ranked = index.search(vector, top_k)
        results.append((issue, ranked))
    return results


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Match SWE-bench Lite issues to csv0_memory_1k memories via BM25."
    )
    parser.add_argument(
        "--memory-dir",
        type=Path,
        help="Directory containing *_memory.json files (defaults to script directory).",
    )
    parser.add_argument(
        "--memory-id",
        type=str,
        help="Optional memory file stem (e.g., 9_memory) to limit matching.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="SWE-bench/SWE-bench_Lite",
        help="Dataset identifier on Hugging Face hub.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to load.",
    )
    parser.add_argument(
        "--issue-field",
        type=str,
        default="problem_statement",
        help="Dataset field containing the issue text.",
    )
    parser.add_argument(
        "--instance-id",
        action="append",
        help="Filter to specific instance_id values (can be repeated).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Cap the number of issues processed.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of memories to return per issue (top matches).",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name used to produce semantic embeddings.",
    )
    return parser


def build_embedder(model_name: str) -> Embedder:
    return Embedder(model_name)


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    default_memory_dir = Path(__file__).resolve().parent
    memory_dir = (args.memory_dir or default_memory_dir).resolve()
    memories = load_memory_entries(memory_dir, args.memory_id)

    embedder = build_embedder(args.embedding_model)

    instance_filters = set(args.instance_id) if args.instance_id else None
    issues = load_issues(args.dataset, args.split, args.issue_field, instance_filters, args.limit)

    results = match_issues_to_memory(issues, memories, args.top_k, embedder)

    analysis_rows: List[tuple[str, str, str, float]] = []
    for issue, matches in results:
        for memory, score in matches:
            analysis_rows.append((issue["instance_id"], issue["repo"], memory.memory_id, score))

    for index, (issue, matches) in enumerate(results, start=1):
        if index > 10:
            break
        print(f"Issue: {issue['instance_id']} ({issue['repo']})")
        for rank, (memory, score) in enumerate(matches, start=1):
            print(f"  {rank}. {memory.memory_id} | score={score:.4f} | path={memory.path}")
        print()

    if analysis_rows:
        analysis_rows.sort(key=lambda row: row[3], reverse=True)
        print("Overall top similarity pairs:")
        for rank, (instance_id, repo, memory_id, score) in enumerate(analysis_rows[:10], start=1):
            print(f"  {rank}. issue={instance_id} ({repo}) | memory={memory_id} | score={score:.4f}")


if __name__ == "__main__":
    main()
