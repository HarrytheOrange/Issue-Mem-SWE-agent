#!/usr/bin/env python3
"""
issue_get_patch - Tool 2
Retrieves the patch content for a specific Repo and PR Number.
"""

import argparse
import json
import sys
import urllib.request
import urllib.error

def get_patch_service(repo: str, pr_number: str):
    service_url = "http://127.0.0.1:9012/get_patch"
    payload = json.dumps({"repo": repo, "pr_number": pr_number}).encode("utf-8")
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
        print(f"❌ Error connecting to service: {e}", file=sys.stderr)
        return None

def print_colored_diff(patch_content):
    """简单的高亮显示 Patch"""
    for line in patch_content.split('\n'):
        if line.startswith('+'):
            print(f"\033[32m{line}\033[0m") # Green
        elif line.startswith('-'):
            print(f"\033[31m{line}\033[0m") # Red
        elif line.startswith('@@'):
            print(f"\033[36m{line}\033[0m") # Cyan
        else:
            print(line)

def main():
    parser = argparse.ArgumentParser(description="Get patch content by Repo and PR Number.")
    parser.add_argument("repo", help="Repository name (e.g., owner/repo)")
    parser.add_argument("pr_number", help="Pull Request Number")
    args = parser.parse_args()

    result = get_patch_service(args.repo, args.pr_number)

    if not result or not result.get("success"):
        print("❌ Failed to retrieve patch.", file=sys.stderr)
        sys.exit(1)

    patches = result.get('results', [])
    if not patches:
        print(f"❌ No patch found for Repo: {args.repo}, PR: {args.pr_number}")
        sys.exit(0)

    print(f"📂 Showing patches for {args.repo}#{args.pr_number}")
    print("=" * 60)

    for item in patches:
        print(f"\n📄 File: {item.get('file', 'Unknown File')}")
        print("-" * 40)
        patch_content = item.get('patch', '')
        print_colored_diff(patch_content)
        print("\n")

if __name__ == "__main__":
    main()
