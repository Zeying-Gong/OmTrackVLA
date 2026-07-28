#!/usr/bin/env python3
"""Aggregate sharded modular-oracle evaluation results."""

import argparse
import gzip
import json
import os
from collections import defaultdict
from pathlib import Path

EXPECTED_CONTROLLER_VERSION = int(os.environ.get("ORACLE_CONTROLLER_VERSION", "5"))


def expected_count(repo, task, split):
    path = repo / "data" / "datasets" / "track" / task.upper() / split / f"{split}.json.gz"
    with gzip.open(path, "rt") as handle:
        return len(json.load(handle)["episodes"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root")
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parent)
    parser.add_argument("--require-100-success", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_root)
    repo = Path(args.repo_root)
    groups = defaultdict(list)
    errors = defaultdict(int)
    stale = defaultdict(int)
    for path in root.glob("*/*/episodes/*.json"):
        value = json.loads(path.read_text())
        key = (value["task"], value["split"])
        if value.get("controller_version") != EXPECTED_CONTROLLER_VERSION:
            stale[key] += 1
            continue
        if "summary" in value:
            groups[key].append(value["summary"])
        else:
            errors[key] += 1

    report = {}
    print("task split completed expected errors stale SR% TR% CR% finish%")
    failed_requirement = False
    keys = sorted(set(groups) | set(errors) | set(stale))
    for task, split in keys:
        values = groups[(task, split)]
        expected = expected_count(repo, task, split)
        count = len(values)
        err = errors[(task, split)]
        old = stale[(task, split)]
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
        print(
            f"{task:>4} {split:>5} {count:>9} {expected:>8} {err:>6} {old:>5} "
            f"{row['success_rate']*100:>6.2f} {row['tracking_rate']*100:>6.2f} "
            f"{row['collision_rate']*100:>6.2f} {row['finish_rate']*100:>7.2f}"
        )
        if count + err != expected or err or old or row["success_rate"] < 1.0:
            failed_requirement = True
    (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.require_100_success and failed_requirement:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
