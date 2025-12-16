from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List

TARGET_TOOL_NAMES = {"exp_search", "exp_read"}


@dataclass(slots=True)
class ToolUsage:
    step: int
    name: str
    action: str
    arguments: Any
    outputs: List[str]


@dataclass(slots=True)
class AnalysisStats:
    matched_files: int = 0
    files_with_hits: int = 0
    failed_files: int = 0
    total_calls: int = 0
    exp_search_calls: int = 0
    exp_read_calls: int = 0

    def update(self, usages: List[ToolUsage]) -> None:
        if not usages:
            return
        self.files_with_hits += 1
        self.total_calls += len(usages)
        for usage in usages:
            if usage.name == "exp_search":
                self.exp_search_calls += 1
            elif usage.name == "exp_read":
                self.exp_read_calls += 1

    def to_dict(self) -> dict[str, int]:
        return {
            "matched_files": self.matched_files,
            "files_with_tool_calls": self.files_with_hits,
            "failed_files": self.failed_files,
            "total_tool_calls": self.total_calls,
            "exp_search_calls": self.exp_search_calls,
            "exp_read_calls": self.exp_read_calls,
        }


def stringify_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [stringify_content(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        if value.get("type") == "text" and "text" in value:
            return stringify_content(value["text"])
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def normalize_text(value: Any) -> str:
    text = stringify_content(value)
    if not text:
        return ""
    lines = [line.rstrip() for line in text.splitlines()]
    trimmed_top = 0
    while trimmed_top < len(lines) and not lines[trimmed_top]:
        trimmed_top += 1
    trimmed_bottom = len(lines)
    while trimmed_bottom > trimmed_top and not lines[trimmed_bottom - 1]:
        trimmed_bottom -= 1
    lines = lines[trimmed_top:trimmed_bottom]
    normalized: list[str] = []
    blank_pending = False
    for line in lines:
        if not line:
            if not blank_pending:
                normalized.append("")
            blank_pending = True
        else:
            normalized.append(line)
            blank_pending = False
    return "\n".join(normalized)


def normalize_arguments(tool_name: str, raw_args: Any) -> Any:
    if raw_args is None:
        return {}
    if isinstance(raw_args, dict):
        return raw_args
    if not isinstance(raw_args, str):
        raw_args = str(raw_args)
    trimmed = raw_args.strip()
    if not trimmed:
        return {}
    if trimmed[0] in {'"', "'"} and trimmed[-1] == trimmed[0]:
        trimmed_inner = trimmed[1:-1]
    else:
        trimmed_inner = trimmed
    if tool_name == "exp_search":
        try:
            parsed = json.loads(trimmed)
            if isinstance(parsed, (dict, list, str, int, float, bool)) or parsed is None:
                return parsed
        except json.JSONDecodeError:
            pass
        return {"query": trimmed_inner}
    if tool_name == "exp_read":
        return {"id": trimmed_inner}
    return {"raw": trimmed}


def load_trajectory(path: Path) -> List[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    trajectory = data.get("trajectory")
    if not isinstance(trajectory, list):
        raise ValueError("Missing 'trajectory' list")
    if not all(isinstance(entry, dict) for entry in trajectory):
        raise ValueError("Trajectory entries must be objects")
    return list(trajectory)


def collect_tool_outputs(messages: List[dict[str, Any]]) -> dict[str, List[str]]:
    outputs: dict[str, List[str]] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "tool":
            continue
        call_ids = message.get("tool_call_ids")
        if not isinstance(call_ids, list):
            continue
        text = normalize_text(message.get("content"))
        if not text:
            continue
        for call_id in call_ids:
            if not isinstance(call_id, str):
                continue
            outputs.setdefault(call_id, []).append(text)
    return outputs


def extract_tool_usages(entries: List[dict[str, Any]]) -> List[ToolUsage]:
    usages: List[ToolUsage] = []
    for idx, item in enumerate(entries, start=1):
        query_messages = item.get("query")
        if not isinstance(query_messages, list):
            continue
        call_outputs = collect_tool_outputs(query_messages)
        entry_action = item.get("action")
        entry_action_text = entry_action.strip() if isinstance(entry_action, str) else ""
        fallback_result = normalize_text(item.get("observation"))
        for message in query_messages:
            if not isinstance(message, dict):
                continue
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                function_info = call.get("function")
                if not isinstance(function_info, dict):
                    continue
                name = function_info.get("name")
                if name not in TARGET_TOOL_NAMES:
                    continue
                raw_arguments = function_info.get("arguments", "")
                arguments = normalize_arguments(name, raw_arguments)
                call_id = call.get("id")
                outputs = []
                if isinstance(call_id, str):
                    outputs = list(call_outputs.get(call_id, []))
                if not outputs and fallback_result:
                    outputs = [fallback_result]
                action_text = entry_action_text
                if not action_text or not action_text.startswith(name):
                    raw_argument_text = stringify_content(raw_arguments)
                    action_text = f"{name} {raw_argument_text}".strip()
                usages.append(
                    ToolUsage(
                        step=idx,
                        name=name,
                        action=action_text or name,
                        arguments=arguments,
                        outputs=outputs,
                    )
                )
    return usages


def locate_traj_files(root: Path) -> List[Path]:
    files = [
        candidate for candidate in root.rglob("*.traj") if candidate.is_file()
    ]
    files.sort()
    return files


def build_report(
    root: Path,
    files: List[Path],
) -> tuple[dict[str, Any], AnalysisStats]:
    timestamp = datetime.now().isoformat(timespec="seconds")
    stats = AnalysisStats(matched_files=len(files))
    report: dict[str, Any] = {
        "generated_at": timestamp,
        "root_directory": str(root),
        "matched_traj_files": len(files),
        "files": [],
    }
    if not files:
        report["notes"] = "No matching trajectory files found."
        report["stats"] = stats.to_dict()
        return report, stats
    for file_path in files:
        file_entry: dict[str, Any] = {
            "relative_path": str(file_path.relative_to(root)),
        }
        try:
            entries = load_trajectory(file_path)
            usages = extract_tool_usages(entries)
            stats.update(usages)
            file_entry["tool_calls"] = [
                {
                    "order": order,
                    "step": usage.step,
                    "tool": usage.name,
                    "action": usage.action,
                    "arguments": usage.arguments,
                    "outputs": usage.outputs,
                }
                for order, usage in enumerate(usages, start=1)
            ]
        except Exception as exc:  # noqa: BLE001
            stats.failed_files += 1
            file_entry["error"] = str(exc)
        else:
            file_entry.setdefault("tool_calls", [])
        report["files"].append(file_entry)
    report["stats"] = stats.to_dict()
    return report, stats


def run(root: Path, report_path: Path) -> None:
    files = locate_traj_files(root)
    report, stats = build_report(root, files)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print("Trace analysis complete.")
    print(f"Report saved to: {report_path}")
    print(f"Matched traj files: {stats.matched_files}")
    print(f"Files with tool calls: {stats.files_with_hits}")
    print(f"Failed files: {stats.failed_files}")
    print(
        "Tool calls:"
        f" total={stats.total_calls},"
        f" exp_search={stats.exp_search_calls},"
        f" exp_read={stats.exp_read_calls}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze exp_search and exp_read usage inside trajectory files."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Base directory that contains run subdirectories.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional explicit report path. Defaults to <root>/trace_analysis_report.txt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Root directory not found: {root}")
    report_path = (
        args.report.expanduser().resolve()
        if args.report
        else root / "trace_analysis_report.json"
    )
    run(root, report_path)


if __name__ == "__main__":
    main()

