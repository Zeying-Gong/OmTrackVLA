#!/usr/bin/env python3
"""Aggregate sharded modular-oracle evaluation results."""

import argparse
import csv
import gzip
import json
import os
from collections import defaultdict
from pathlib import Path

EXPECTED_CONTROLLER_VERSION = int(os.environ.get("ORACLE_CONTROLLER_VERSION", "5"))
CSV_FIELDNAMES = [
    "task", "split", "dataset_index", "episode_key", "scene_id",
    "episode_id", "target_name", "success", "following_rate",
    "collision", "finish", "total_step", "perception", "controller",
    "result_path",
]
REQUIRED_SUMMARY_FIELDS = {
    "success", "following_rate", "collision", "finish", "total_step",
}


def expected_count(repo, task, split):
    path = repo / "data" / "datasets" / "track" / task.upper() / split / f"{split}.json.gz"
    with gzip.open(path, "rt") as handle:
        return len(json.load(handle)["episodes"])


def write_episode_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def load_result(path):
    try:
        value = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        reason = f"invalid JSON: {exc}"
    except OSError as exc:
        print(f"WARNING: cannot read result {path}: {exc}")
        return None, False
    else:
        task = path.parent.parent.parent.name
        split = path.parent.parent.name
        if not isinstance(value, dict):
            reason = "top-level JSON value is not an object"
        elif value.get("task") != task or value.get("split") != split:
            reason = (
                f"task/split does not match path: "
                f"{value.get('task')}/{value.get('split')} != {task}/{split}"
            )
        elif "summary" in value and (
            not isinstance(value["summary"], dict)
            or not REQUIRED_SUMMARY_FIELDS.issubset(value["summary"])
        ):
            reason = "summary is malformed or missing required fields"
        elif "summary" not in value and "error" not in value:
            reason = "result has neither summary nor error"
        else:
            return value, False

    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        print(f"WARNING: cannot remove invalid result {path}: {exc}")
        return None, False
    print(f"Removed invalid result {path} ({reason}); its episode will be rerun")
    return None, True


def clean_invalid_results(root):
    removed = 0
    scanned = 0
    for path in sorted(root.glob("*/*/episodes/*.json")):
        scanned += 1
        _, was_removed = load_result(path)
        removed += int(was_removed)
    print(f"Checked {scanned} result JSON files; removed {removed} invalid files")
    return removed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root")
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parent)
    parser.add_argument("--require-100-success", action="store_true")
    parser.add_argument("--clean-invalid-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_root)
    repo = Path(args.repo_root)
    if args.clean_invalid_only:
        clean_invalid_results(root)
        return
    groups = defaultdict(list)
    errors = defaultdict(int)
    stale = defaultdict(int)
    episodes = []
    episodes_by_group = defaultdict(list)
    for path in sorted(root.glob("*/*/episodes/*.json")):
        value, _ = load_result(path)
        if value is None:
            continue
        task = value["task"]
        split = value["split"]
        key = (task, split)
        if value.get("controller_version") != EXPECTED_CONTROLLER_VERSION:
            stale[key] += 1
            continue
        if "summary" in value:
            groups[key].append(value["summary"])
            s = value["summary"]
            episode_row = {
                "task": task,
                "split": split,
                "dataset_index": value.get("dataset_index", ""),
                "episode_key": value.get("episode_key", path.stem),
                "scene_id": value.get("scene_id", ""),
                "episode_id": value.get("episode_id", ""),
                "target_name": value.get("target_name", ""),
                "success": s.get("success", ""),
                "following_rate": s.get("following_rate", ""),
                "collision": s.get("collision", ""),
                "finish": s.get("finish", ""),
                "total_step": s.get("total_step", ""),
                "perception": value.get("perception", ""),
                "controller": value.get("controller", ""),
                "result_path": str(path),
            }
            episodes.append(episode_row)
            episodes_by_group[key].append(episode_row)
        else:
            errors[key] += 1

    report = {}
    print("task split completed expected errors stale SR% TR% CR% finish%")
    failed_requirement = False
    keys = sorted(set(groups) | set(errors) | set(stale))
    for task, split in keys:
        group_key = (task, split)
        values = groups[group_key]
        expected = expected_count(repo, task, split)
        count = len(values)
        err = errors[group_key]
        old = stale[group_key]
        mean = lambda field: sum(float(v[field]) for v in values) / count if count else 0.0
        row = {
            "completed": count,
            "expected": expected,
            "errors": err,
            "stale": old,
            "success_rate": mean("success"),
            "tracking_rate": mean("following_rate"),
            "collision_rate": mean("collision"),
            "finish_rate": mean("finish"),
        }
        report[f"{task}/{split}"] = row
        group_root = root / task / split
        group_root.mkdir(parents=True, exist_ok=True)
        (group_root / "summary.json").write_text(json.dumps(row, indent=2) + "\n")
        group_csv_path = group_root / "episodes.csv"
        write_episode_csv(group_csv_path, episodes_by_group[group_key])
        print(
            f"{task:>4} {split:>5} {count:>9} {expected:>8} {err:>6} {old:>5} "
            f"{row['success_rate']*100:>6.2f} {row['tracking_rate']*100:>6.2f} "
            f"{row['collision_rate']*100:>6.2f} {row['finish_rate']*100:>7.2f}"
        )
        if count + err != expected or err or old or row["success_rate"] < 1.0:
            failed_requirement = True
    (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    csv_path = root / "episodes.csv"
    write_episode_csv(csv_path, episodes)
    print(f"\nWrote {len(episodes)} aggregate episode rows to {csv_path}")
    for task, split in keys:
        group_csv_path = root / task / split / "episodes.csv"
        print(
            f"Wrote {len(episodes_by_group[(task, split)])} {task}/{split} "
            f"episode rows to {group_csv_path}"
        )

    if args.require_100_success and failed_requirement:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
