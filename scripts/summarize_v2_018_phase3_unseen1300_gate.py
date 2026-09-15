#!/usr/bin/env python3
"""Aggregate the three-model Architecture-v2 unseen-index1300 closed-loop gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


MODELS = ("v2_006b_parent", "v2_016_waypoint", "v2_017_calibrated")
MODES = ("polar_reactive", "se2_waypoint")
TASKS = ("stt", "dt", "at")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visibility-threshold", type=float, default=0.005)
    return parser.parse_args()


def safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def main() -> int:
    args = arguments()
    root = args.root.expanduser().resolve(strict=True)
    rows = []
    matched: dict[tuple[str, str], dict[str, dict]] = {}
    for model in MODELS:
        for mode in MODES:
            for task in TASKS:
                result_path = root / model / mode / f"{task}_1300" / "result.json"
                value = json.loads(result_path.read_text(encoding="utf-8"))
                if (
                    value.get("split") != "train"
                    or value.get("dataset_index") != 1300
                    or value.get("action_mode") != mode
                    or value.get("task") != task
                    or value.get("uwb_mode") != "missing"
                ):
                    raise ValueError(f"closed-loop provenance mismatch: {result_path}")
                false_visible = false_invisible = gt_visible = gt_invisible = 0
                for step in value["steps"]:
                    prediction = (
                        float(step["policy"]["visibility_probability"])
                        >= args.visibility_threshold
                    )
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
                    "scene_id": str(value["scene_id"]),
                    "checkpoint_sha256": value["loading"]["checkpoint_sha256"],
                    "steps": len(value["steps"]),
                    "success": float(summary["success"]),
                    "official_success_so_far": float(summary["official_success_so_far"]),
                    "following_rate": float(summary["following_rate"]),
                    "collision": float(summary["collision"]),
                    "robot_path_m": float(summary["robot_path_m"]),
                    "target_path_m": float(summary["target_path_m"]),
                    "target_losses": int(summary["target_losses"]),
                    "target_reacquisitions": int(summary["target_reacquisitions"]),
                    "false_visible_steps": false_visible,
                    "false_invisible_steps": false_invisible,
                    "gt_visible_steps": gt_visible,
                    "gt_invisible_steps": gt_invisible,
                    "false_visible_rate": safe_rate(false_visible, gt_invisible),
                    "false_invisible_rate": safe_rate(false_invisible, gt_visible),
                    "mean_inference_ms_after_warmup": float(
                        summary["mean_inference_ms_after_warmup"]
                    ),
                    "result": str(result_path),
                }
                rows.append(row)
                matched.setdefault((mode, task), {})[model] = row

    condition_deltas = []
    for (mode, task), group in sorted(matched.items()):
        if set(group) != set(MODELS):
            raise ValueError(f"incomplete matched condition: {mode}/{task}")
        episode_ids = {row["episode_id"] for row in group.values()}
        scene_ids = {row["scene_id"] for row in group.values()}
        if len(episode_ids) != 1 or len(scene_ids) != 1:
            raise ValueError(f"environment mismatch: {mode}/{task}")
        for reference, candidate in (
            ("v2_006b_parent", "v2_016_waypoint"),
            ("v2_006b_parent", "v2_017_calibrated"),
            ("v2_016_waypoint", "v2_017_calibrated"),
        ):
            old, new = group[reference], group[candidate]
            condition_deltas.append(
                {
                    "action_mode": mode,
                    "task": task,
                    "reference": reference,
                    "candidate": candidate,
                    "success_delta": new["success"] - old["success"],
                    "following_rate_delta": new["following_rate"] - old["following_rate"],
                    "collision_delta": new["collision"] - old["collision"],
                    "false_visible_steps_delta": new["false_visible_steps"] - old["false_visible_steps"],
                    "false_invisible_steps_delta": new["false_invisible_steps"] - old["false_invisible_steps"],
                    "robot_path_m_delta": new["robot_path_m"] - old["robot_path_m"],
                }
            )

    aggregate = {}
    for model in MODELS:
        for mode in MODES:
            subset = [
                row
                for row in rows
                if row["model"] == model and row["action_mode"] == mode
            ]
            gt_visible = sum(row["gt_visible_steps"] for row in subset)
            gt_invisible = sum(row["gt_invisible_steps"] for row in subset)
            false_visible = sum(row["false_visible_steps"] for row in subset)
            false_invisible = sum(row["false_invisible_steps"] for row in subset)
            aggregate[f"{model}/{mode}"] = {
                "episodes": len(subset),
                "success_rate": sum(row["success"] for row in subset) / len(subset),
                "mean_following_rate": sum(row["following_rate"] for row in subset) / len(subset),
                "collision_episode_rate": sum(row["collision"] > 0.0 for row in subset) / len(subset),
                "collision_sum": sum(row["collision"] for row in subset),
                "false_visible_steps": false_visible,
                "false_visible_rate": safe_rate(false_visible, gt_invisible),
                "false_invisible_steps": false_invisible,
                "false_invisible_rate": safe_rate(false_invisible, gt_visible),
                "target_losses": sum(row["target_losses"] for row in subset),
                "target_reacquisitions": sum(row["target_reacquisitions"] for row in subset),
                "robot_path_m": sum(row["robot_path_m"] for row in subset),
                "mean_inference_ms_after_warmup": sum(
                    row["mean_inference_ms_after_warmup"] for row in subset
                )
                / len(subset),
            }

    gates = {}
    for mode in MODES:
        prior = aggregate[f"v2_016_waypoint/{mode}"]
        calibrated = aggregate[f"v2_017_calibrated/{mode}"]
        checks = {
            "success_rate_not_lower": calibrated["success_rate"] >= prior["success_rate"],
            "following_rate_drop_at_most_0.02": (
                calibrated["mean_following_rate"]
                >= prior["mean_following_rate"] - 0.02
            ),
            "no_new_collision_episodes": (
                calibrated["collision_episode_rate"]
                <= prior["collision_episode_rate"]
            ),
            "false_visible_rate_not_higher": (
                calibrated["false_visible_rate"] <= prior["false_visible_rate"]
            ),
            "false_invisible_rate_not_higher": (
                calibrated["false_invisible_rate"] <= prior["false_invisible_rate"]
            ),
        }
        gates[mode] = {"passed": all(checks.values()), "checks": checks}
    overall_passed = all(value["passed"] for value in gates.values())
    payload = {
        "schema_version": 1,
        "stage": "v2_018_phase3_unseen1300_closed_loop_gate",
        "status": "passed" if overall_passed else "failed",
        "protocol": {
            "split": "train",
            "dataset_index": 1300,
            "previously_unseen_before_this_gate": True,
            "tasks": list(TASKS),
            "action_modes": list(MODES),
            "models": list(MODELS),
            "uwb_mode": "missing",
            "visibility_threshold": args.visibility_threshold,
            "matched_same_episode_and_scene": True,
            "test_locked_used": False,
        },
        "promotion_gate": {
            "reference": "v2_016_waypoint",
            "candidate": "v2_017_calibrated",
            "passed": overall_passed,
            "by_action_mode": gates,
        },
        "aggregate": aggregate,
        "condition_deltas": condition_deltas,
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
