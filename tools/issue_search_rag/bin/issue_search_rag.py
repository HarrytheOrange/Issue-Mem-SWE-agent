#!/usr/bin/env python3
"""
issue_search_rag - Client tool for searching patches via local ChromaDB service.

Features:
- Connects to localhost:9012
- Accepts a natural language query.
- Returns top-k matching patches with diff previews.
- 零第三方依赖：使用 urllib 替代 requests
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Optional, Dict, Any


def search_service(query: str, topk: int = 3) -> Optional[Dict[str, Any]]:
    """Send search request to the local ChromaDB service（纯原生库实现）."""
    service_url = "http://127.0.0.1:9012/search"
    payload = json.dumps({"query": query, "topk": topk}).encode("utf-8")
    req = urllib.request.Request(
        service_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            # 读取并解析 JSON
            result = json.load(resp)
            if result.get("success"):
                return result
            else:
                print(f"❌ Search failed: {result.get('error', 'Unknown error')}", file=sys.stderr)
                return None

    except urllib.error.HTTPError as e:
        print(f"❌ HTTP error: {e.code}", file=sys.stderr)
        return None
    except urllib.error.URLError as e:
        # 包含连接失败、端口未监听等
        print(f"❌ Cannot connect to service at {service_url}", file=sys.stderr)
        print("💡 Make sure the ChromaDB service is running on port 9012", file=sys.stderr)
        return None
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return None


# ---------------- 以下代码完全不变 ---------------- #
def print_patch_preview(patch_content: str, max_lines: int = 8):
    """Format and print the patch content with coloring indicators."""
    if not patch_content:
        print("   (No patch content)")
        return

    lines = patch_content.split('\n')
    preview_lines = lines[:max_lines]

    for line in preview_lines:
        if line.startswith('+'):
            print(f"   🟢 {line}")
        elif line.startswith('-'):
            print(f"   🔴 {line}")
        else:
            print(f"   ⚪ {line}")

    if len(lines) > max_lines:
        print(f"   ... ({len(lines) - max_lines} more lines)")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="issue_search_rag",
        description="Search for relevant patches/issues using the local ChromaDB service."
    )
    parser.add_argument(
        "query",
        help="The issue description or search query"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        choices=range(1, 11),
        help="Number of results to return (1-10, default: 3)"
    )

    args = parser.parse_args()

    # Execute search
    result = search_service(args.query, args.topk)

    if not result:
        return 1

    # Display results
    total = result.get('total_results', 0)
    print(f"🔍 Search results for: \"{args.query}\"")
    print(f"📊 Found {total} results")
    print("=" * 60)

    results = result.get('results', [])
    if not results:
        print("❌ No relevant results found.")
        return 0

    for i, item in enumerate(results, 1):
        print(f"\n📋 Result {i}:")
        print(f"   Repo:       {item.get('repo', 'N/A')}")
        print(f"   File:       {item.get('file', 'N/A')}")
        print(f"   PR Number:  {item.get('pr_number', 'N/A')}")
        print(f"   Score:      {item.get('similarity_score', 0):.4f}")
        print(f"   Patch:      {item.get('patch', '')}")
        # print(f"   Patch Preview:")
        # print_patch_preview(item.get('patch', ''))
        print("-" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())