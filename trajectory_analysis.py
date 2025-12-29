from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from math import ceil, floor
from pathlib import Path
from statistics import median
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
class TraceSummary:
    total_calls: int
    exp_search_calls: int
    exp_read_calls: int
    exp_search_followed_by_exp_read_anywhere: int
    exp_search_followed_by_exp_read_immediately: int
    exp_read_with_prior_exp_search: int
    exp_read_without_prior_exp_search: int
    all_exp_search_followed_by_exp_read_anywhere: bool
    all_exp_search_followed_by_exp_read_immediately: bool

    @property
    def has_exp_search(self) -> bool:
        return self.exp_search_calls > 0

    @property
    def has_exp_read(self) -> bool:
        return self.exp_read_calls > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_target_tool_calls": self.total_calls,
            "exp_search_calls": self.exp_search_calls,
            "exp_read_calls": self.exp_read_calls,
            "has_exp_search": self.has_exp_search,
            "has_exp_read": self.has_exp_read,
            "exp_search_followed_by_exp_read_anywhere": (
                self.exp_search_followed_by_exp_read_anywhere
            ),
            "exp_search_without_following_read_anywhere": (
                self.exp_search_calls - self.exp_search_followed_by_exp_read_anywhere
            ),
            "exp_search_followed_by_exp_read_immediately": (
                self.exp_search_followed_by_exp_read_immediately
            ),
            "exp_search_without_following_read_immediately": (
                self.exp_search_calls - self.exp_search_followed_by_exp_read_immediately
            ),
            "exp_read_with_prior_exp_search": self.exp_read_with_prior_exp_search,
            "exp_read_without_prior_exp_search": self.exp_read_without_prior_exp_search,
            "all_exp_search_followed_by_exp_read_anywhere": (
                self.all_exp_search_followed_by_exp_read_anywhere
            ),
            "all_exp_search_followed_by_exp_read_immediately": (
                self.all_exp_search_followed_by_exp_read_immediately
            ),
        }


def percentile(sorted_values: list[int], p: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute percentile of empty list")
    if p <= 0:
        return float(sorted_values[0])
    if p >= 1:
        return float(sorted_values[-1])
    pos = (len(sorted_values) - 1) * p
    lo = floor(pos)
    hi = ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    lower = sorted_values[lo]
    upper = sorted_values[hi]
    return float(lower + (upper - lower) * (pos - lo))


def summarize_int_distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "histogram": {}}
    values_sorted = sorted(values)
    n = len(values_sorted)
    histogram = {str(k): v for k, v in sorted(Counter(values_sorted).items())}
    return {
        "n": n,
        "min": values_sorted[0],
        "max": values_sorted[-1],
        "mean": sum(values_sorted) / n,
        "median": float(median(values_sorted)),
        "p10": percentile(values_sorted, 0.10),
        "p25": percentile(values_sorted, 0.25),
        "p75": percentile(values_sorted, 0.75),
        "p90": percentile(values_sorted, 0.90),
        "histogram": histogram,
    }


def summarize_trace(usages: List[ToolUsage]) -> TraceSummary:
    names = [usage.name for usage in usages]
    total_calls = len(names)

    exp_search_calls = 0
    exp_read_calls = 0
    for name in names:
        if name == "exp_search":
            exp_search_calls += 1
        elif name == "exp_read":
            exp_read_calls += 1

    next_read_after: list[int | None] = [None] * total_calls
    next_read: int | None = None
    for idx in range(total_calls - 1, -1, -1):
        next_read_after[idx] = next_read
        if names[idx] == "exp_read":
            next_read = idx

    exp_search_followed_by_exp_read_anywhere = 0
    exp_search_followed_by_exp_read_immediately = 0
    for idx, name in enumerate(names):
        if name != "exp_search":
            continue
        if next_read_after[idx] is not None:
            exp_search_followed_by_exp_read_anywhere += 1
        if idx + 1 < total_calls and names[idx + 1] == "exp_read":
            exp_search_followed_by_exp_read_immediately += 1

    exp_read_with_prior_exp_search = 0
    exp_read_without_prior_exp_search = 0
    seen_search = False
    for name in names:
        if name == "exp_search":
            seen_search = True
        elif name == "exp_read":
            if seen_search:
                exp_read_with_prior_exp_search += 1
            else:
                exp_read_without_prior_exp_search += 1

    all_anywhere = (
        exp_search_calls > 0
        and exp_search_followed_by_exp_read_anywhere == exp_search_calls
    )
    all_immediately = (
        exp_search_calls > 0
        and exp_search_followed_by_exp_read_immediately == exp_search_calls
    )

    return TraceSummary(
        total_calls=total_calls,
        exp_search_calls=exp_search_calls,
        exp_read_calls=exp_read_calls,
        exp_search_followed_by_exp_read_anywhere=exp_search_followed_by_exp_read_anywhere,
        exp_search_followed_by_exp_read_immediately=(
            exp_search_followed_by_exp_read_immediately
        ),
        exp_read_with_prior_exp_search=exp_read_with_prior_exp_search,
        exp_read_without_prior_exp_search=exp_read_without_prior_exp_search,
        all_exp_search_followed_by_exp_read_anywhere=all_anywhere,
        all_exp_search_followed_by_exp_read_immediately=all_immediately,
    )


@dataclass(slots=True)
class AnalysisStats:
    matched_files: int = 0
    processed_files: int = 0
    files_with_hits: int = 0
    failed_files: int = 0
    total_calls: int = 0
    exp_search_calls: int = 0
    exp_read_calls: int = 0
    traces_with_exp_search: int = 0
    traces_with_exp_read: int = 0
    traces_with_search_and_read: int = 0
    traces_with_search_no_read: int = 0
    traces_with_read_no_search: int = 0
    traces_with_neither: int = 0
    traces_all_searches_followed_by_read_anywhere: int = 0
    traces_all_searches_followed_by_read_immediately: int = 0
    exp_search_followed_by_exp_read_anywhere: int = 0
    exp_search_followed_by_exp_read_immediately: int = 0
    exp_read_with_prior_exp_search: int = 0
    exp_read_without_prior_exp_search: int = 0
    total_calls_per_trace: list[int] = field(default_factory=list)
    exp_search_calls_per_trace: list[int] = field(default_factory=list)
    exp_read_calls_per_trace: list[int] = field(default_factory=list)
    exp_search_without_following_read_anywhere_per_trace: list[int] = field(
        default_factory=list
    )
    exp_search_without_following_read_immediately_per_trace: list[int] = field(
        default_factory=list
    )

    validation_file: str | None = None
    resolved_instances: int = 0
    unresolved_instances: int = 0
    unknown_instances: int = 0
    resolved_total_tool_calls: int = 0
    unresolved_total_tool_calls: int = 0
    resolved_exp_search_calls: int = 0
    unresolved_exp_search_calls: int = 0
    resolved_exp_read_calls: int = 0
    unresolved_exp_read_calls: int = 0

    def update(self, summary: TraceSummary, outcome: str) -> None:
        self.processed_files += 1
        self.total_calls += summary.total_calls
        self.exp_search_calls += summary.exp_search_calls
        self.exp_read_calls += summary.exp_read_calls

        self.total_calls_per_trace.append(summary.total_calls)
        self.exp_search_calls_per_trace.append(summary.exp_search_calls)
        self.exp_read_calls_per_trace.append(summary.exp_read_calls)
        self.exp_search_without_following_read_anywhere_per_trace.append(
            summary.exp_search_calls - summary.exp_search_followed_by_exp_read_anywhere
        )
        self.exp_search_without_following_read_immediately_per_trace.append(
            summary.exp_search_calls
            - summary.exp_search_followed_by_exp_read_immediately
        )

        if summary.total_calls:
            self.files_with_hits += 1

        has_search = summary.has_exp_search
        has_read = summary.has_exp_read
        if has_search:
            self.traces_with_exp_search += 1
        if has_read:
            self.traces_with_exp_read += 1
        if has_search and has_read:
            self.traces_with_search_and_read += 1
        elif has_search:
            self.traces_with_search_no_read += 1
        elif has_read:
            self.traces_with_read_no_search += 1
        else:
            self.traces_with_neither += 1

        self.exp_search_followed_by_exp_read_anywhere += (
            summary.exp_search_followed_by_exp_read_anywhere
        )
        self.exp_search_followed_by_exp_read_immediately += (
            summary.exp_search_followed_by_exp_read_immediately
        )
        self.exp_read_with_prior_exp_search += summary.exp_read_with_prior_exp_search
        self.exp_read_without_prior_exp_search += (
            summary.exp_read_without_prior_exp_search
        )
        if summary.all_exp_search_followed_by_exp_read_anywhere:
            self.traces_all_searches_followed_by_read_anywhere += 1
        if summary.all_exp_search_followed_by_exp_read_immediately:
            self.traces_all_searches_followed_by_read_immediately += 1

        if outcome == "resolved":
            self.resolved_instances += 1
            self.resolved_total_tool_calls += summary.total_calls
            self.resolved_exp_search_calls += summary.exp_search_calls
            self.resolved_exp_read_calls += summary.exp_read_calls
        elif outcome == "unresolved":
            self.unresolved_instances += 1
            self.unresolved_total_tool_calls += summary.total_calls
            self.unresolved_exp_search_calls += summary.exp_search_calls
            self.unresolved_exp_read_calls += summary.exp_read_calls
        else:
            self.unknown_instances += 1

    def to_dict(self) -> dict[str, Any]:
        processed = self.processed_files
        labeled = self.resolved_instances + self.unresolved_instances
        total_labeled_tool_calls = (
            self.resolved_total_tool_calls + self.unresolved_total_tool_calls
        )
        return {
            "matched_files": self.matched_files,
            "processed_files": processed,
            "files_with_tool_calls": self.files_with_hits,
            "failed_files": self.failed_files,
            "total_tool_calls": self.total_calls,
            "exp_search_calls": self.exp_search_calls,
            "exp_read_calls": self.exp_read_calls,
            "traces_with_exp_search": self.traces_with_exp_search,
            "traces_without_exp_search": processed - self.traces_with_exp_search,
            "traces_with_exp_read": self.traces_with_exp_read,
            "traces_without_exp_read": processed - self.traces_with_exp_read,
            "traces_with_search_and_read": self.traces_with_search_and_read,
            "traces_with_search_no_read": self.traces_with_search_no_read,
            "traces_with_read_no_search": self.traces_with_read_no_search,
            "traces_with_neither": self.traces_with_neither,
            "exp_search_calls_mean_per_trace": (
                self.exp_search_calls / processed if processed else 0.0
            ),
            "exp_read_calls_mean_per_trace": (
                self.exp_read_calls / processed if processed else 0.0
            ),
            "exp_search_calls_mean_per_trace_when_used": (
                self.exp_search_calls / self.traces_with_exp_search
                if self.traces_with_exp_search
                else 0.0
            ),
            "exp_read_calls_mean_per_trace_when_used": (
                self.exp_read_calls / self.traces_with_exp_read
                if self.traces_with_exp_read
                else 0.0
            ),
            "exp_search_followed_by_exp_read_anywhere": (
                self.exp_search_followed_by_exp_read_anywhere
            ),
            "exp_search_followed_by_exp_read_immediately": (
                self.exp_search_followed_by_exp_read_immediately
            ),
            "exp_read_with_prior_exp_search": self.exp_read_with_prior_exp_search,
            "exp_read_without_prior_exp_search": (
                self.exp_read_without_prior_exp_search
            ),
            "traces_all_searches_followed_by_read_anywhere": (
                self.traces_all_searches_followed_by_read_anywhere
            ),
            "traces_all_searches_followed_by_read_immediately": (
                self.traces_all_searches_followed_by_read_immediately
            ),
            "distributions": {
                "total_target_tool_calls_per_trace": summarize_int_distribution(
                    self.total_calls_per_trace
                ),
                "exp_search_calls_per_trace": summarize_int_distribution(
                    self.exp_search_calls_per_trace
                ),
                "exp_read_calls_per_trace": summarize_int_distribution(
                    self.exp_read_calls_per_trace
                ),
                "exp_search_without_following_read_anywhere_per_trace": (
                    summarize_int_distribution(
                        self.exp_search_without_following_read_anywhere_per_trace
                    )
                ),
                "exp_search_without_following_read_immediately_per_trace": (
                    summarize_int_distribution(
                        self.exp_search_without_following_read_immediately_per_trace
                    )
                ),
            },
            "instance_outcomes": {
                "validation_file": self.validation_file,
                "resolved_instances": self.resolved_instances,
                "unresolved_instances": self.unresolved_instances,
                "unknown_instances": self.unknown_instances,
                "resolved_instance_ratio": (
                    self.resolved_instances / labeled if labeled else 0.0
                ),
                "unresolved_instance_ratio": (
                    self.unresolved_instances / labeled if labeled else 0.0
                ),
                "resolved_total_tool_calls": self.resolved_total_tool_calls,
                "unresolved_total_tool_calls": self.unresolved_total_tool_calls,
                "resolved_tool_calls_share": (
                    self.resolved_total_tool_calls / total_labeled_tool_calls
                    if total_labeled_tool_calls
                    else 0.0
                ),
                "unresolved_tool_calls_share": (
                    self.unresolved_total_tool_calls / total_labeled_tool_calls
                    if total_labeled_tool_calls
                    else 0.0
                ),
                "resolved_mean_tool_calls_per_instance": (
                    self.resolved_total_tool_calls / self.resolved_instances
                    if self.resolved_instances
                    else 0.0
                ),
                "unresolved_mean_tool_calls_per_instance": (
                    self.unresolved_total_tool_calls / self.unresolved_instances
                    if self.unresolved_instances
                    else 0.0
                ),
                "resolved_mean_exp_search_calls_per_instance": (
                    self.resolved_exp_search_calls / self.resolved_instances
                    if self.resolved_instances
                    else 0.0
                ),
                "unresolved_mean_exp_search_calls_per_instance": (
                    self.unresolved_exp_search_calls / self.unresolved_instances
                    if self.unresolved_instances
                    else 0.0
                ),
                "resolved_mean_exp_read_calls_per_instance": (
                    self.resolved_exp_read_calls / self.resolved_instances
                    if self.resolved_instances
                    else 0.0
                ),
                "unresolved_mean_exp_read_calls_per_instance": (
                    self.unresolved_exp_read_calls / self.unresolved_instances
                    if self.unresolved_instances
                    else 0.0
                ),
            },
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
    seen_call_ids: set[str] = set()
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
                call_id = call.get("id")
                if isinstance(call_id, str):
                    # Some .traj files repeat historical tool calls inside later steps'
                    # query messages. Deduplicate to count actual calls only.
                    if call_id in seen_call_ids:
                        continue
                    seen_call_ids.add(call_id)
                raw_arguments = function_info.get("arguments", "")
                arguments = normalize_arguments(name, raw_arguments)
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


def locate_validation_file(root: Path) -> Path | None:
    candidates = [path for path in root.glob("*.validate-*.json") if path.is_file()]
    if not candidates:
        return None
    candidates.sort()
    return candidates[0]


def load_validation_ids(path: Path) -> tuple[set[str], set[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    resolved_raw = data.get("resolved_ids", [])
    unresolved_raw = data.get("unresolved_ids", [])

    resolved_ids: set[str] = set()
    if isinstance(resolved_raw, list):
        resolved_ids = {item for item in resolved_raw if isinstance(item, str)}

    unresolved_ids: set[str] = set()
    if isinstance(unresolved_raw, list):
        unresolved_ids = {item for item in unresolved_raw if isinstance(item, str)}

    return resolved_ids, unresolved_ids


def locate_traj_files(root: Path) -> List[Path]:
    files = [
        candidate for candidate in root.rglob("*.traj") if candidate.is_file()
    ]
    files.sort()
    return files


def build_report(
    root: Path,
    files: List[Path],
    resolved_ids: set[str] | None = None,
    unresolved_ids: set[str] | None = None,
    validation_file: Path | None = None,
) -> tuple[dict[str, Any], AnalysisStats]:
    timestamp = datetime.now().isoformat(timespec="seconds")
    stats = AnalysisStats(matched_files=len(files))
    stats.validation_file = str(validation_file) if validation_file else None
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
            summary = summarize_trace(usages)
            instance_id = file_path.stem
            outcome = "unknown"
            if resolved_ids is not None and instance_id in resolved_ids:
                outcome = "resolved"
            elif unresolved_ids is not None and instance_id in unresolved_ids:
                outcome = "unresolved"
            stats.update(summary, outcome)
            file_entry["summary"] = summary.to_dict()
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
    validation_file = locate_validation_file(root)
    resolved_ids: set[str] | None = None
    unresolved_ids: set[str] | None = None
    if validation_file is not None:
        resolved_ids, unresolved_ids = load_validation_ids(validation_file)
    report, stats = build_report(
        root,
        files,
        resolved_ids=resolved_ids,
        unresolved_ids=unresolved_ids,
        validation_file=validation_file,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print("Trace analysis complete.")
    print(f"Report saved to: {report_path}")
    print(f"Matched traj files: {stats.matched_files}")
    print(f"Processed traj files: {stats.processed_files}")
    print(f"Files with tool calls: {stats.files_with_hits}")
    print(f"Failed files: {stats.failed_files}")
    print(
        "Tool calls:"
        f" total={stats.total_calls},"
        f" exp_search={stats.exp_search_calls},"
        f" exp_read={stats.exp_read_calls}"
    )
    print(
        "Traces:"
        f" with_exp_search={stats.traces_with_exp_search},"
        f" with_exp_read={stats.traces_with_exp_read},"
        f" with_both={stats.traces_with_search_and_read},"
        f" with_neither={stats.traces_with_neither}"
    )
    if stats.validation_file:
        labeled = stats.resolved_instances + stats.unresolved_instances
        resolved_avg = (
            stats.resolved_total_tool_calls / stats.resolved_instances
            if stats.resolved_instances
            else 0.0
        )
        unresolved_avg = (
            stats.unresolved_total_tool_calls / stats.unresolved_instances
            if stats.unresolved_instances
            else 0.0
        )
        print(
            "Outcomes:"
            f" resolved={stats.resolved_instances},"
            f" unresolved={stats.unresolved_instances},"
            f" unknown={stats.unknown_instances},"
            f" accuracy={stats.resolved_instances / labeled if labeled else 0.0:.3f},"
            f" avg_tool_calls(resolved)={resolved_avg:.3f},"
            f" avg_tool_calls(unresolved)={unresolved_avg:.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze exp_search and exp_read usage inside trajectory files."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Trace directory that contains issue subdirectories and *.traj files.",
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

