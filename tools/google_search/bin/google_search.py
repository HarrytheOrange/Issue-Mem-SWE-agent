#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SERPER_ENDPOINT = "https://google.serper.dev/search"
SERPAPI_ENDPOINT = "https://serpapi.com/search.json"


def _first_env(names: Sequence[str]) -> Optional[str]:
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


def _first_env_or_dotenv(names: Sequence[str]) -> Optional[str]:
    value = _first_env(names)
    if value:
        return value
    dotenv_path = _find_dotenv(Path.cwd()) or _find_dotenv(Path(__file__).resolve().parent)
    if dotenv_path is None:
        return None
    values = _parse_dotenv(dotenv_path)
    for name in names:
        v = values.get(name)
        if v:
            return v
    return None


def _read_json_response(resp: Any) -> Dict[str, Any]:
    try:
        charset = resp.headers.get_content_charset() or "utf-8"
    except Exception:
        charset = "utf-8"
    raw = resp.read()
    text = raw.decode(charset, errors="replace")
    data = json.loads(text)
    if not isinstance(data, dict):
        msg = f"Unexpected JSON response type: {type(data).__name__}"
        raise ValueError(msg)
    return data


def _http_json(
    url: str,
    *,
    method: str,
    headers: Dict[str, str],
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 20.0,
) -> Dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return _read_json_response(resp)


@dataclass(frozen=True)
class SearchResult:
    title: str
    link: str
    snippet: str


def _coerce_results(items: Any, topk: int) -> List[SearchResult]:
    if not isinstance(items, list):
        return []
    results: List[SearchResult] = []
    for item in items[: max(0, topk)]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "") or "")
        link = str(item.get("link", "") or item.get("url", "") or "")
        snippet = str(item.get("snippet", "") or item.get("description", "") or "")
        if not (title or link or snippet):
            continue
        results.append(SearchResult(title=title, link=link, snippet=snippet))
    return results


def search_serper(query: str, *, topk: int, api_key: str, timeout: float) -> Tuple[List[SearchResult], Dict[str, Any]]:
    payload: Dict[str, Any] = {"q": query, "num": int(topk)}
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    raw = _http_json(SERPER_ENDPOINT, method="POST", headers=headers, payload=payload, timeout=timeout)
    results = _coerce_results(raw.get("organic"), topk)
    return results, raw


def search_serpapi(query: str, *, topk: int, api_key: str, timeout: float) -> Tuple[List[SearchResult], Dict[str, Any]]:
    params = {"engine": "google", "q": query, "api_key": api_key, "num": str(int(topk))}
    url = f"{SERPAPI_ENDPOINT}?{urllib.parse.urlencode(params)}"
    headers = {"User-Agent": "swe-agent-google-search/1.0"}
    raw = _http_json(url, method="GET", headers=headers, payload=None, timeout=timeout)
    results = _coerce_results(raw.get("organic_results"), topk)
    return results, raw


def _print_results(results: List[SearchResult]) -> None:
    if not results:
        print("No results.")
        return
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r.title}".rstrip())
        if r.link:
            print(r.link)
        if r.snippet:
            print(r.snippet)
        print()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Google search tool (Serper / SerpAPI).")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--topk", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument(
        "--backend",
        choices=["auto", "serper", "serpapi"],
        default="auto",
        help="Which provider to use (default: auto)",
    )
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout in seconds (default: 20)")
    parser.add_argument("--json", action="store_true", help="Print raw JSON response")
    args = parser.parse_args(list(argv) if argv is not None else None)

    topk = max(1, int(args.topk))

    serper_key = _first_env_or_dotenv(["SERPER_API_KEY", "GOOGLE_SERPER_API_KEY", "SERPER_KEY"])
    serpapi_key = _first_env_or_dotenv(["SERPAPI_API_KEY", "GOOGLE_SERPAPI_API_KEY", "SERPAPI_KEY"])

    backend = str(args.backend)
    try:
        if backend == "serper" or (backend == "auto" and serper_key):
            if not serper_key:
                print(
                    "Missing Serper API key. Set one of: SERPER_API_KEY, GOOGLE_SERPER_API_KEY, SERPER_KEY",
                    file=sys.stderr,
                )
                return 2
            results, raw = search_serper(args.query, topk=topk, api_key=serper_key, timeout=float(args.timeout))
        else:
            if not serpapi_key:
                print(
                    "Missing API key. Set Serper (SERPER_API_KEY) or SerpAPI (SERPAPI_API_KEY).",
                    file=sys.stderr,
                )
                return 2
            results, raw = search_serpapi(args.query, topk=topk, api_key=serpapi_key, timeout=float(args.timeout))
    except urllib.error.HTTPError as e:
        print(f"HTTP error: {e.code}", file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"Connection error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(raw, ensure_ascii=False, indent=2))
        return 0

    _print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


