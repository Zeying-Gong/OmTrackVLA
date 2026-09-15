#!/usr/bin/env python3
"""Aggregate non-locked EVT action-mode rollouts without hiding failures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.glob("*/*/*/result.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("split") != "train":
            raise ValueError(f"non-development split encountered: {path}")
        summary = value["summary"]
        rows.append({
            "task": value["task"],
            "mode": value["action_mode"],
            "dataset_index": value["dataset_index"],
            "success": summary.get("success"),
            "success_so_far": summary.get("official_success_so_far"),
            "tracking_rate": summary.get("following_rate"),
            "collision": summary.get("collision"),
            "steps": summary.get("steps"),
            "result": str(path),
        })
    grouped = {}
    for mode in ("direct", "waypoint_dt", "lookahead"):
        selected = [row for row in rows if row["mode"] == mode]
        definite = [row["success"] for row in selected if row["success"] is not None]
        grouped[mode] = {
            "rollouts": len(selected),
            "completed_with_official_sr": len(definite),
            "sr": None if not definite else sum(float(v) for v in definite) / len(definite),
            "tr": None if not selected else sum(float(v["tracking_rate"] or 0.0) for v in selected) / len(selected),
            "collision_rate": None if not selected else sum(float(v["collision"] or 0.0) > 0.0 for v in selected) / len(selected),
        }
    expected = 36
    result = {
        "schema_version": 1,
        "protocol": "EVT train development split; same v2 checkpoint; UWB missing",
        "expected_rollouts": expected,
        "completed_rollouts": len(rows),
        "complete": len(rows) == expected,
        "by_execution_mode": grouped,
        "rollouts": rows,
        "test_locked_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["by_execution_mode"], sort_keys=True))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
