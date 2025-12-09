#!/usr/bin/env python3
"""
exp_read - Get detailed memory and related nodes for a specific ID.
Dependencies: None (Standard Lib only)
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

# 注意：这里使用 GET 请求
DEFAULT_BASE_URL = "http://127.0.0.1:9020/get_experience"
BASE_URL = os.environ.get("GRAPH_EXP_READ_URL", DEFAULT_BASE_URL)


def get_details(sub_id: str) -> Optional[Dict[str, Any]]:
    params = urllib.parse.urlencode({'id': sub_id})
    url = f"{BASE_URL}?{params}"
    
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"❌ ID '{sub_id}' not found in knowledge base.", file=sys.stderr)
        else:
            print(f"❌ HTTP Error: {e.code}", file=sys.stderr)
        return None
    except urllib.error.URLError as e:
        print(f"❌ Connection Error: {e}", file=sys.stderr)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Read detailed experience memory.")
    parser.add_argument("id", help="The ID of the sub-cluster experience")
    args = parser.parse_args()

    resp = get_details(args.id)
    
    if not resp or not resp.get("success"):
        return 1

    data = resp.get("data", {})
    
    print("=" * 60)
    print(f"📘 EXPERIENCE: {data.get('name', 'Unknown')}")
    print(f"🆔 ID: {data.get('id')}")
    print("=" * 60)
    
    print(f"\n📝 DESCRIPTION:\n{data.get('description')}\n")
    
    print(f"🧠 ISSUE MEMORY (Fix Logic):")
    memory = data.get('issue_memory', '')
    # 简单的缩进展示
    for line in memory.split('\n'):
        print(f"   {line}")
    
    print("\n" + "-" * 60)
    print(f"🔗 RELATED NODES (Suggestions):")
    related = data.get('related_nodes', [])
    
    if not related:
        print("   (No related nodes recorded)")
    else:
        for node in related:
            print(f"   👉 {node['target_name']}")
            print(f"      ID:  {node['target_id']}")
            print(f"      Why: {node['relationship_desc']}")
            print("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
