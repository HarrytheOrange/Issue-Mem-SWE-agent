#!/usr/bin/env python3
import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

BASE_URL = "http://172.30.182.85:9030/get_experience"

DEEPSEEK_API_KEY = "1a6d8e05-0978-496b-87c1-fd4fb3885e7c"
DEEPSEEK_MODEL = "deepseek-v3-1-terminus"
DEEPSEEK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v1"


def _first_env(*names: str) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _find_dotenv(start_dir: Path, *, max_levels: int = 6) -> Optional[Path]:
    current = start_dir.resolve()
    for _ in range(max_levels):
        candidate = current / ".env"
        if candidate.is_file():
            return candidate
        if current.parent == current:
            break
        current = current.parent
    return None


def _parse_dotenv(dotenv_path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for raw_line in dotenv_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _first_from_maps(keys: Tuple[str, ...], primary: Dict[str, str], secondary: Dict[str, str]) -> Optional[str]:
    for k in keys:
        v = primary.get(k)
        if v:
            return v
    for k in keys:
        v = secondary.get(k)
        if v:
            return v
    return None


def _resolve_llm_config() -> Optional[Tuple[str, str, str]]:
    if not DEEPSEEK_BASE_URL or not DEEPSEEK_API_KEY or not DEEPSEEK_MODEL:
        return None
    return DEEPSEEK_BASE_URL.rstrip("/"), DEEPSEEK_API_KEY, DEEPSEEK_MODEL


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _post_json(
    url: str, payload: Dict[str, Any], headers: Dict[str, str], *, timeout: int
) -> Optional[Dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        print(f"LLM HTTP Error: {e.code}", file=sys.stderr)
        return None
    except urllib.error.URLError as e:
        print(f"LLM Connection Error: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"LLM Error: {e}", file=sys.stderr)
        return None


def _llm_says_helpful(problem_statement: str, document: str, *, unique_id: Optional[str] = None) -> bool:
    cfg = _resolve_llm_config()
    if cfg is None:
        return False
    api_base, api_key, model = cfg

    url = f"{api_base}/chat/completions"
    system = (
        "You are a strict relevance classifier for software engineering tasks. "
        "Given a SWE problem statement and a candidate fix-experience document, decide if the document could plausibly help solve the problem. "
        "If the document might help, answer HELPFUL. Only answer NOT_HELPFUL if it is clearly unrelated. "
        "Output exactly one token: HELPFUL or NOT_HELPFUL."
    )
    user = (
        "SWE problem statement:\n"
        f"{_truncate(problem_statement, 8000)}\n\n"
        "Candidate fix-experience document:\n"
        f"{_truncate(document, 12000)}\n"
    )
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": 8,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    resp = _post_json(url, payload, headers, timeout=30)
    if not resp:
        return False
    try:
        content = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "")  # type: ignore[union-attr]
    except Exception:
        return False
    decision_raw = str(content).strip()
    decision = decision_raw.upper()
    print(f"LLM_DECISION: {decision}", file=sys.stderr)
    if decision.startswith("HELPFUL"):
        return True
    return False


def get_details(unique_id: str) -> Optional[Dict[str, Any]]:
    params = urllib.parse.urlencode({"id": unique_id})
    url = f"{BASE_URL}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data: Dict[str, Any] = json.load(resp)
            return data
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"ID '{unique_id}' not found.", file=sys.stderr)
        else:
            print(f"HTTP Error: {e.code}", file=sys.stderr)
        return None
    except urllib.error.URLError as e:
        print(f"Connection Error: {e}", file=sys.stderr)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Read fix experience by ID (LLM-filtered for current problem).")
    parser.add_argument("id", help="The unique ID (repo+issue_id)")
    args = parser.parse_args()

    resp = get_details(args.id)
    if not resp or not resp.get("success"):
        return 1

    data = resp.get("data", {}) or {}
    fix_experience = str(data.get("fix_experience", "") or "")
    if not fix_experience.strip():
        print("NOT_HELPFUL")
        return 0

    problem_statement = os.environ.get("PROBLEM_STATEMENT", "")
    if not _llm_says_helpful(problem_statement, fix_experience, unique_id=args.id):
        print("NOT_HELPFUL")
        return 0

    print("FIX EXPERIENCE:")
    print(f"ID:   {data.get('id')}")
    print("Content:")
    print(fix_experience)
    print(
        "If you find this experience helpful, consider starting your bug fix now! "
        "If not, you can try to browse other related experiences with exp_read tool or search other experiences using the exp_search tool."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
