#!/usr/bin/env python3
import argparse
import json
import sys
import urllib.request
import urllib.parse
import urllib.error

BASE_URL = "http://172.30.182.85:9030/get_experience"

def get_details(unique_id):
    params = urllib.parse.urlencode({'id': unique_id})
    url = f"{BASE_URL}?{params}"
    
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"❌ ID '{unique_id}' not found.", file=sys.stderr)
        else:
            print(f"❌ HTTP Error: {e.code}", file=sys.stderr)
        return None
    except urllib.error.URLError as e:
        print(f"❌ Connection Error: {e}", file=sys.stderr)
        return None

def main():
    parser = argparse.ArgumentParser(description="Read fix experience by ID.")
    parser.add_argument("id", help="The unique ID (repo+issue_id)")
    args = parser.parse_args()

    resp = get_details(args.id)
    
    if not resp or not resp.get("success"):
        return 1

    data = resp.get("data", {})
    
    print("=" * 60)
    print(f"FIX EXPERIENCE")
    print(f"ID:   {data.get('id')}")
    print(f"Repo: {data.get('repo')}")
    print("=" * 60)
    
    print(f"\nContent:\n")
    print(data.get('fix_experience', 'No content available.'))
    print("\n" + "=" * 60)
    print("If you find this experience helpful, consider starting your bug fix now! If not, you can try to search and browse other experiences using the exp_search tool.")

    return 0

if __name__ == "__main__":
    sys.exit(main())
