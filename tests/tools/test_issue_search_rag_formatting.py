import runpy
from pathlib import Path


_MODULE = runpy.run_path(
    str(
        Path(__file__).resolve().parents[2]
        / "tools"
        / "issue_search_rag"
        / "chromadb_service_1222.py"
    )
)
_format_results = _MODULE["_format_results"]


def test_format_results_falls_back_to_documents_and_infers_repo_from_id() -> None:
    raw = {
        "ids": [["01org/parameter-framework116938351"]],
        "metadatas": [[{}]],
        "documents": [["some description"]],
        "distances": [[0.647]],
    }

    formatted = _format_results(raw)

    assert len(formatted) == 1
    item = formatted[0]
    assert item["repo"] == "01org/parameter-framework"
    assert item["patch"] == "some description"
    assert item["similarity_score"] == 0.647
    assert "pr_number" not in item


