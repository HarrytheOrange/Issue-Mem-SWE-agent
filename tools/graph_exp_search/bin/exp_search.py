#!/usr/bin/env python3
"""
exp_search - Search for bug fix experiences.
Dependencies: None (Standard Lib only)
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

DEFAULT_SERVER_URL = "http://127.0.0.1:9020/search"
SERVER_URL = os.environ.get("GRAPH_EXP_SEARCH_URL", DEFAULT_SERVER_URL)


def search_experience(query: str, top_k: int = 5) -> Optional[Dict[str, Any]]:
    payload = json.dumps({"query": query, "top_k": top_k}).encode("utf-8")
    req = urllib.request.Request(
        SERVER_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.load(resp)
    except urllib.error.URLError as e:
        print(f"❌ Connection Error: {e}", file=sys.stderr)
        print(f"💡 Ensure server is running at {SERVER_URL}", file=sys.stderr)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Search experience knowledge base.")
    parser.add_argument("query", help="Description of the bug or issue")
    parser.add_argument("--top_k", type=int, default=10, help="Number of results")
    args = parser.parse_args()

    print(f"🔎 Searching for: '{args.query}'...")
    data = search_experience(args.query, args.top_k)

    if not data or not data.get("success"):
        print("❌ Search failed or no response.")
        return 1

    results = data.get("results", [])
    if not results:
        print("⚪ No results found.")
        return 0

    print(f"✅ Found {len(results)} relevant experiences:\n")
    for i, res in enumerate(results, 1):
        print(f"[{i}] {res['sub_category']}")
        print(f"    ID:    {res['id']}")
        print(f"    Macro: {res['macro_category']}")
        print(f"    Desc:  {res['description'][:120]}...")
        print("-" * 50)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
