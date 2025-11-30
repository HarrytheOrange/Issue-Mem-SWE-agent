#!/usr/bin/env python3
"""
issue_memory_rag - Client tool for searching structured experience memories via local ChromaDB service.

Features:
- Connects to localhost:9013 (configurable via env).
- Accepts a natural language query.
- Returns top-k matching memory snippets with summaries.
- 零第三方依赖：使用 urllib 替代 requests
"""

import argparse
import json
import os
import sys
import textwrap
import urllib.error
import urllib.request
from typing import Optional, Dict, Any

DEFAULT_SERVICE_URL = "http://127.0.0.1:9013/search"
SERVICE_URL_ENV = "ISSUE_MEMORY_RAG_URL"
TOPK_ENV = "ISSUE_MEMORY_RAG_DEFAULT_TOPK"
TOPK_MIN, TOPK_MAX = 1, 10


def search_service(query: str, topk: int = 3) -> Optional[Dict[str, Any]]:
    """Send search request to the local ChromaDB service（纯原生库实现）."""
    service_url = os.environ.get(SERVICE_URL_ENV, DEFAULT_SERVICE_URL)
    payload = json.dumps({"query": query, "topk": topk}).encode("utf-8")
    req = urllib.request.Request(
        service_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.load(resp)
            if result.get("success"):
                return result
            print(f"❌ Search failed: {result.get('error', 'Unknown error')}", file=sys.stderr)
            return None

    except urllib.error.HTTPError as e:
        print(f"❌ HTTP error: {e.code}", file=sys.stderr)
        return None
    except urllib.error.URLError:
        print(f"❌ Cannot connect to service at {service_url}", file=sys.stderr)
        print("💡 Make sure the ChromaDB memory service is running on port 9013", file=sys.stderr)
        return None
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return None


def _truncate(text: str, max_len: int = 500) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def print_memory_entry(item: Dict[str, Any]) -> None:
    print(f"   Source File:  {item.get('source_file', 'N/A')}")
    print(f"   Keywords:     {item.get('keywords', 'N/A')}")
    print(f"   Similarity:   {item.get('similarity_score', 0):.4f}")

    description = item.get("description") or ""
    if description:
        print("   Description:")
        print(textwrap.indent(_truncate(description), "      "))

    episodic = item.get("episodic_memory") or ""
    if episodic:
        print("   Episodic Memory:")
        print(textwrap.indent(_truncate(episodic), "      "))

    semantic = item.get("semantic_memory") or ""
    if semantic:
        print("   Semantic Memory:")
        print(textwrap.indent(_truncate(semantic), "      "))

    procedural = item.get("procedural_memory") or ""
    if procedural:
        print("   Procedural Steps:")
        print(textwrap.indent(_truncate(procedural), "      "))


def main() -> int:
    env_topk = os.environ.get(TOPK_ENV)
    topk_default = 3
    if env_topk:
        try:
            parsed = int(env_topk)
            if TOPK_MIN <= parsed <= TOPK_MAX:
                topk_default = parsed
        except ValueError:
            pass

    parser = argparse.ArgumentParser(
        prog="issue_memory_rag",
        description="Search structured issue memories using the local ChromaDB service."
    )
    parser.add_argument(
        "query",
        help="The issue description or search query"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=topk_default,
        choices=range(TOPK_MIN, TOPK_MAX + 1),
        help=f"Number of results to return ({TOPK_MIN}-{TOPK_MAX}, default: {topk_default})"
    )

    args = parser.parse_args()

    result = search_service(args.query, args.topk)

    if not result:
        return 1

    total = result.get('total_results', 0)
    print(f"🔍 Memory search for: \"{args.query}\"")
    print(f"📊 Found {total} results")
    print("=" * 60)

    results = result.get('results', [])
    if not results:
        print("❌ No relevant memories found.")
        return 0

    for i, item in enumerate(results, 1):
        print(f"\n📋 Result {i}:")
        print_memory_entry(item)
        print("-" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())