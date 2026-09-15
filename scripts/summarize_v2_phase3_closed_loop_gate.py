#!/usr/bin/env python3
"""Aggregate paired parent/selected Architecture-v2 closed-loop rollouts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visibility-threshold", type=float, default=0.005)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    root = args.root.expanduser().resolve(strict=True)
    rows = []
    paired: dict[tuple[str, str], dict[str, dict]] = {}
    for model in ("parent", "selected"):
        for mode in ("polar_reactive", "se2_waypoint"):
            for task in ("stt", "dt", "at"):
                result_path = root / model / mode / f"{task}_900" / "result.json"
                value = json.loads(result_path.read_text(encoding="utf-8"))
                if value.get("split") != "train" or value.get("dataset_index") != 900:
                    raise ValueError(f"closed-loop provenance mismatch: {result_path}")
                steps = value["steps"]
                false_visible = false_invisible = gt_visible = gt_invisible = 0
                for step in steps:
                    prediction = float(step["policy"]["visibility_probability"]) >= args.visibility_threshold
                    visible = bool(step["evaluation_only_after_action"]["gt_visible"])
                    gt_visible += int(visible)
                    gt_invisible += int(not visible)
                    false_visible += int(not visible and prediction)
                    false_invisible += int(visible and not prediction)
                summary = value["summary"]
                row = {
                    "model": model,
                    "action_mode": mode,
                    "task": task,
                    "episode_id": str(value["episode_id"]),
                    "checkpoint_sha256": value["loading"]["checkpoint_sha256"],
                    "steps": len(steps),
                    "following_rate": float(summary["following_rate"]),
                    "collision": float(summary["collision"]),
                    "robot_path_m": float(summary["robot_path_m"]),
                    "target_losses": int(summary["target_losses"]),
                    "target_reacquisitions": int(summary["target_reacquisitions"]),
                    "false_visible_steps": false_visible,
                    "false_invisible_steps": false_invisible,
                    "gt_visible_steps": gt_visible,
                    "gt_invisible_steps": gt_invisible,
                    "result": str(result_path),
                }
                rows.append(row)
                paired.setdefault((mode, task), {})[model] = row
    deltas = []
    for (mode, task), pair in sorted(paired.items()):
        if set(pair) != {"parent", "selected"}:
            raise ValueError(f"incomplete pair: {mode}/{task}")
        parent, selected = pair["parent"], pair["selected"]
        if parent["episode_id"] != selected["episode_id"]:
            raise ValueError(f"episode mismatch: {mode}/{task}")
        deltas.append(
            {
                "action_mode": mode,
                "task": task,
                "episode_id": parent["episode_id"],
                "following_rate_delta_selected_minus_parent": selected["following_rate"] - parent["following_rate"],
                "collision_delta_selected_minus_parent": selected["collision"] - parent["collision"],
                "false_visible_steps_delta_selected_minus_parent": selected["false_visible_steps"] - parent["false_visible_steps"],
                "false_invisible_steps_delta_selected_minus_parent": selected["false_invisible_steps"] - parent["false_invisible_steps"],
            }
        )
    aggregate = {}
    for model in ("parent", "selected"):
        for mode in ("polar_reactive", "se2_waypoint"):
            subset = [row for row in rows if row["model"] == model and row["action_mode"] == mode]
            key = f"{model}/{mode}"
            aggregate[key] = {
                "episodes": len(subset),
                "mean_following_rate": sum(row["following_rate"] for row in subset) / len(subset),
                "collision_episodes": sum(row["collision"] > 0.0 for row in subset),
                "false_visible_steps": sum(row["false_visible_steps"] for row in subset),
                "false_invisible_steps": sum(row["false_invisible_steps"] for row in subset),
                "robot_path_m": sum(row["robot_path_m"] for row in subset),
            }
    payload = {
        "schema_version": 1,
        "stage": "v2_011_phase3_closed_loop_gate",
        "status": "complete",
        "protocol": {
            "split": "train",
            "dataset_index": 900,
            "tasks": ["stt", "dt", "at"],
            "action_modes": ["polar_reactive", "se2_waypoint"],
            "visibility_threshold": args.visibility_threshold,
            "paired_same_episode": True,
            "test_locked_used": False,
        },
        "aggregate": aggregate,
        "paired_deltas": deltas,
        "rollouts": rows,
        "test_locked_used": False,
    }
    output = args.output.expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite gate summary: {output}")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
