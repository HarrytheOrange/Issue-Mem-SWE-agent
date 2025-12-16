#!/usr/bin/env python3
import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

SERVER_URL = "http://172.30.182.85:9030/search"
DATA_FILE = (
    Path(__file__)
    .resolve()
    .parents[3]
    / "data"
    / "agentic_exp_data_1209"
    / "experience_data.json"
)

_EXPERIENCE_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def _load_experience_data() -> Dict[str, Dict[str, Any]]:
    global _EXPERIENCE_CACHE
    if _EXPERIENCE_CACHE is None:
        if not DATA_FILE.exists():
            print(f"⚠️ Experience data file not found: {DATA_FILE}", file=sys.stderr)
            _EXPERIENCE_CACHE = {}
        else:
            _EXPERIENCE_CACHE = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    return _EXPERIENCE_CACHE


def get_bug_description(unique_id: str) -> Optional[str]:
    data = _load_experience_data()
    entry = data.get(unique_id)
    if not entry:
        return None
    return entry.get("bug_description")


def search_experience(query: str, top_k: int = 10) -> Optional[Dict[str, Any]]:
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
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Search bug fix experiences.")
    parser.add_argument("query", help="Description of the bug")
    parser.add_argument("--top_k", type=int, default=10, help="Number of results")
    args = parser.parse_args()

    print(f"🔎 Searching for: '{args.query}'...")
    data = search_experience(args.query, args.top_k)

    if not data or not data.get("success"):
        print("❌ Search failed.")
        return 1

    results = data.get("results", [])
    if not results:
        print("⚪ No results found.")
        return 0

    print(f"✅ Found {len(results)} relevant items:\n")
    for i, res in enumerate(results, 1):
        unique_id = res.get("id", "")
        preview = (
            res.get("bug_description")
            or (get_bug_description(unique_id) if unique_id else None)
            or "No preview available."
        )
        # or res.get("content_preview")
            
        print(f"[{i}] ID: {res['id']}")
        print(f"    Issue Content: {preview}")
        print("-" * 50)
    # print("Please Use ID and exp_read tool to browse detailed bug fix experience before fix bug.\n")
    print("If you think some of the Issue Content is helpful to fix the bug, please use exp_read tool with ID to browse detailed bug fix experience before fix bug. You can also try to search again with different keywords to find a better match.\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
