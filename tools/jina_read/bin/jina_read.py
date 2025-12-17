#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional, Sequence

JINA_READER_BASE = "https://r.jina.ai/"


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


def _parse_dotenv(dotenv_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
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


def _first_env_or_dotenv(*names: str) -> Optional[str]:
    value = _first_env(*names)
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


def _build_reader_url(url: str) -> str:
    url = url.strip()
    if url.startswith(JINA_READER_BASE):
        return url
    if url.startswith("http://") or url.startswith("https://"):
        return f"{JINA_READER_BASE}{url}"
    msg = "URL must start with http:// or https://"
    raise ValueError(msg)


def _read_text_response(resp: Any) -> str:
    try:
        charset = resp.headers.get_content_charset() or "utf-8"
    except Exception:
        charset = "utf-8"
    raw = resp.read()
    return raw.decode(charset, errors="replace")


def jina_read(url: str, *, api_key: Optional[str], timeout: float) -> str:
    reader_url = _build_reader_url(url)
    headers = {"User-Agent": "swe-agent-jina-read/1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(reader_url, headers=headers, method="GET")
    if api_key:
        # urllib normalizes some header casing depending on Python version. Some tests use
        # `get_header("X-Api-Key")`, so we set it explicitly on the request object.
        req.headers["X-Api-Key"] = api_key
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return _read_text_response(resp)


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Read web pages via Jina Reader (r.jina.ai).")
    parser.add_argument("url", help="Target URL (http/https)")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout in seconds (default: 20)")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=8000,
        help="Max characters to print (default: 8000). Use 0 to disable truncation.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    api_key = _first_env_or_dotenv("JINA_API_KEY", "JINA_KEY", "JINA_AI_API_KEY")

    try:
        text = jina_read(args.url, api_key=api_key, timeout=float(args.timeout))
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except urllib.error.HTTPError as e:
        print(f"HTTP error: {e.code}", file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"Connection error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(_truncate(text, int(args.max_chars)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


