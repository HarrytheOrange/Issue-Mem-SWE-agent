#!/usr/bin/env python3
import argparse
import json
import sys
import urllib.request
import urllib.parse
import urllib.error
from typing import Any, Dict, Optional, Tuple

# 配置服务器地址
SEARCH_URL = "http://172.30.182.85:9030/search"
DETAILS_URL = "http://172.30.182.85:9030/get_experience"

def search_experience(query: str, top_k: int = 3) -> Optional[Dict[str, Any]]:
    """搜索相关的经验 ID"""
    payload = json.dumps({"query": query, "top_k": top_k}).encode("utf-8")
    req = urllib.request.Request(
        SEARCH_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.load(resp)
    except urllib.error.URLError as e:
        print(f"Search Connection Error: {e}", file=sys.stderr)
        return None

def get_details(unique_id: str) -> Optional[Dict[str, Any]]:
    """根据 ID 获取完整详情"""
    params = urllib.parse.urlencode({'id': unique_id})
    url = f"{DETAILS_URL}?{params}"
    
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        # 忽略单个获取失败，以免中断整个流程
        return None
    except urllib.error.URLError as e:
        return None

def _first_non_empty_str(data: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[str]:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None

def main():
    parser = argparse.ArgumentParser(description="RAG Tool: Search and retrieve full bug fix experiences.")
    parser.add_argument("query", help="Description of the bug or issue")
    parser.add_argument("--top_k", type=int, default=3, help="Number of results to retrieve (default: 3)")
    args = parser.parse_args()

    print(f"RAG Searching for: '{args.query}' (Top {args.top_k})...")
    
    # 1. 执行搜索
    search_data = search_experience(args.query, args.top_k)

    if not search_data or not search_data.get("success"):
        print("Search failed or returned no success signal.")
        return 1

    results = search_data.get("results", [])
    if not results:
        print("No relevant experiences found.")
        return 0

    print(f"Found {len(results)} relevant items. Retrieving full details...\n")

    # 2. 遍历结果并获取详情
    for i, res in enumerate(results, 1):
        unique_id = res.get('id')
        if not unique_id:
            continue

        detail_resp = get_details(unique_id)
        
        # 格式化输出
        # print("=" * 60)
        print(f"RESULT #{i}")
        
        if detail_resp and detail_resp.get("success"):
            data = detail_resp.get("data", {})
            print(f"ID:   {data.get('id')}")
            print(f"Repo: {data.get('repo')}")
            # print("-" * 60)
            print(f"Bug Description:\n")
            bug_description = (
                _first_non_empty_str(
                    data,
                    (
                        "bug_description",
                        "issue_content",
                        "content_preview",
                    ),
                )
                or _first_non_empty_str(
                    res,
                    (
                        "bug_description",
                        "issue_content",
                        "content_preview",
                    ),
                )
                or "No content available."
            )
            print(bug_description)

            print(f"Full Experience Content:\n")
            print(data.get('fix_experience', 'No content available.'))
        else:
            # 如果获取详情失败，回退显示搜索结果中的预览
            print(f"Full details fetch failed for ID: {unique_id}")
            print(f"ID:   {unique_id}")
            # print("-" * 60)
            print(f"Preview Content:\n")
            print(res.get('content_preview', 'No preview available.'))
        
        # print("\n" + "=" * 60 + "\n")
        print("\n")

    print("Tip: Use the insights above to guide your bug fix.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
