import argparse
import json
from pathlib import Path

from tabulate import tabulate


def get_resolved(path: Path) -> set[str]:
    data = json.loads(path.read_text())
    if "resolved" in data:
        data["resolved_ids"] = data["resolved"]
    return set(data["resolved_ids"])


def get_submitted(path: Path) -> set[str]:
    return set(json.loads(path.read_text())["submitted_ids"])


def stats_single(path: Path) -> None:
    evaluated_ids = sorted(get_submitted(path))
    resolved_ids = sorted(get_resolved(path))
    print(f"Total evaluated: {len(evaluated_ids)}")
    print(f"Total resolved: {len(resolved_ids)}")


def compare_many(paths: list[Path]) -> None:
    evaluated_ids = {}
    resolved_ids = {}
    all_evaluated_ids = set()
    for path in paths:
        evaluated_ids[path] = sorted(get_submitted(path))
        resolved_ids[path] = sorted(get_resolved(path))
        all_evaluated_ids.update(evaluated_ids[path])
    header = ["ID"] + [i for i in range(len(paths))]
    table = [header]

    def get_emoji(id: str, path: Path) -> str:
        if id not in evaluated_ids[path]:
            return "❓"
        if id in resolved_ids[path]:
            return "✅"
        return "❌"

    for id in sorted(all_evaluated_ids):
        row = [id] + [get_emoji(id, path) for path in paths]
        table.append(row)
    print(tabulate(table, headers="firstrow"))

    header = ["Name", "Resolved", "Evaluated"]
    table = [header]
    for path in paths:
        row = [path.parent.name, len(resolved_ids[path]), len(evaluated_ids[path])]
        table.append(row)
    print(tabulate(table, headers="firstrow"))


def compare_pair(new_path: Path, old_path: Path, *, show_same=False) -> None:
    evaluated_ids = sorted(get_submitted(new_path))
    resolved_ids = sorted(get_resolved(new_path))
    old_evaluated_ids = sorted(get_submitted(old_path))
    old_resolved_ids = sorted(get_resolved(old_path))
    print(f"Total evaluated: new {len(evaluated_ids)}, old {len(old_evaluated_ids)}")
    print(f"Total resolved: new {len(resolved_ids)}, old {len(old_resolved_ids)}")
    print("-" * 80)
    print("Emoji legend:")
    print("❓: Not evaluated in old version, so guessing it's either 😀 or 👾")
    print("😀: Newly resolved in new version")
    print("✅: Resolved in both")
    print("❌: Resolved in old, not in new")
    print("👾: Unresolved in both")
    print("-" * 80)

    for id in evaluated_ids:
        resolved_now = id in resolved_ids
        resolved_before = id in old_resolved_ids
        if id not in old_evaluated_ids and resolved_now:
            emoji = "😀❓"
        elif id not in old_evaluated_ids and not resolved_now:
            emoji = "👾❓"
        elif resolved_now and not resolved_before:
            emoji = "😀"
        elif resolved_now and resolved_before:
            emoji = "✅"
            if not show_same:
                continue
        elif not resolved_now and resolved_before:
            emoji = "❌"
        else:
            emoji = "👾"
            if not show_same:
                continue
        print(f"{emoji} {id}")


def run_from_cli(_args: list[str] | None = None) -> None:
    def get_preds_path(path: Path) -> Path:
        if path.is_dir():
            return path / "results.json"
        return path

    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--show-same", action="store_true")
    args = parser.parse_args(_args)
    args.paths = [get_preds_path(path) for path in args.paths]
    if len(args.paths) == 1:
        stats_single(args.paths[0])
    elif len(args.paths) == 2:
        compare_pair(args.paths[0], args.paths[1], show_same=args.show_same)
    else:
        compare_many(args.paths)
