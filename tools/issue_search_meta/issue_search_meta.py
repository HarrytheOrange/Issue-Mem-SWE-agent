#!/usr/bin/env python3
"""
issue_search_meta - Tool 1
Searches for issues and returns Repo/PR Number pairs.
"""

import argparse
import json
import sys
import urllib.request
import urllib.error

def search_service(query: str, topk: int = 3):
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
            return json.load(resp)
    except Exception as e:
        print(f"❌ Error connecting to search service: {e}", file=sys.stderr)
        return None

def main():
    parser = argparse.ArgumentParser(description="Search issues and get Repo/PR IDs.")
    parser.add_argument("query", help="The issue description")
    parser.add_argument("--topk", type=int, default=3, help="Number of results")
    args = parser.parse_args()

    result = search_service(args.query, args.topk)
    
    if not result or not result.get("success"):
        sys.exit(1)

    results = result.get('results', [])
    if not results:
        print("No results found.")
        return

    print(f"Context: Found {len(results)} relevant issues for query: \"{args.query}\"\n")

    for i, item in enumerate(results, 1):
        # 1. 安全获取所有字段
        repo = item.get('repo', 'N/A')
        pr = item.get('pr_number', 'N/A')
        score = item.get('similarity_score', 0)
        pr_title = item.get('pr_title', 'N/A')
        # 注意：给LLM看的数据不要截断，保持完整
        issue_content = item.get('issue_content', 'N/A') 
        pr_content = item.get('pr_content', 'N/A')

        # 2. 构建结构化字符串块
        # 使用明确的标签（如 <Repo>, <Title>）或 Markdown 标题
        # 这里使用 Markdown 风格，LLM 对这种格式的注意力机制效果很好
        entry_str = f"""
### Result Item {i}
- **Repository**: {repo}
- **PR Number**: {pr}
- **Similarity Score**: {score:.4f}
- **PR Title**: {pr_title}

#### Issue Content:
{issue_content}

#### PR Content:
{pr_content}

---
"""
        # 3. 输出
        print(entry_str)

if __name__ == "__main__":
    main()
