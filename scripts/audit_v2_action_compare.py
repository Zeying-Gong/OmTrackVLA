#!/usr/bin/env python3
"""Fail-closed audit and per-task table for the V2-002 EVT comparison."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


TASKS = ("stt", "dt", "at")
MODES = ("direct", "waypoint_dt", "lookahead")


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    checkpoints = set()
    checkpoint_hashes = set()
    episode_pairing: dict[tuple[str, int], set[str]] = defaultdict(set)
    for task in TASKS:
        for mode in MODES:
            for index in range(4):
                path = args.root / task / mode / str(index) / "result.json"
                value = json.loads(path.read_text(encoding="utf-8"))
                if value["task"] != task or value["action_mode"] != mode:
                    raise ValueError(f"directory/result contract mismatch: {path}")
                if value["split"] != "train" or value.get("uwb_mode") != "missing":
                    raise ValueError(f"non-development or non-visual protocol: {path}")
                if value["summary"].get("success") is None:
                    raise ValueError(f"episode did not produce official SR: {path}")
                audit = value["input_audit"]
                if not audit.get("passed") or audit.get("gt_target_point_used"):
                    raise ValueError(f"privileged-input audit failed: {path}")
                if any(
                    set(step["policy"].get("action_candidates_normalized") or {})
                    != set(MODES)
                    for step in value["steps"]
                ):
                    raise ValueError(f"three action candidates were not retained: {path}")
                checkpoints.add(value["checkpoint"])
                checkpoint_hashes.add(value["loading"]["checkpoint_sha256"])
                episode_pairing[(task, index)].add(str(value["episode_id"]))
                rows.append({
                    "task": task,
                    "mode": mode,
                    "index": index,
                    "sr": float(value["summary"]["success"]),
                    "tr": float(value["summary"]["following_rate"]),
                    "collision": float(value["summary"]["collision"] > 0.0),
                })
    if len(checkpoints) != 1 or len(checkpoint_hashes) != 1:
        raise ValueError("rollouts did not use one shared checkpoint")
    mismatched_episodes = {
        f"{task}/{index}": sorted(values)
        for (task, index), values in episode_pairing.items()
        if len(values) != 1
    }
    if mismatched_episodes:
        raise ValueError(f"execution modes used different episodes: {mismatched_episodes}")

    per_task_mode = {}
    for task in TASKS:
        per_task_mode[task] = {}
        for mode in MODES:
            selected = [row for row in rows if row["task"] == task and row["mode"] == mode]
            per_task_mode[task][mode] = {
                "episodes": len(selected),
                "sr": mean([row["sr"] for row in selected]),
                "tr": mean([row["tr"] for row in selected]),
                "collision_rate": mean([row["collision"] for row in selected]),
            }
    report = {
        "status": "passed",
        "rollouts": len(rows),
        "checkpoint": next(iter(checkpoints)),
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "same_episode_across_execution_modes": True,
        "three_candidates_recorded_each_step": True,
        "split": "train",
        "uwb_mode": "missing",
        "per_task_mode": per_task_mode,
        "test_locked_used": False,
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
