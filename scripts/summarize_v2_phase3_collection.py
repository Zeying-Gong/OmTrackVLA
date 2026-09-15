#!/usr/bin/env python3
"""Summarize non-locked model-visited rollouts and propose diverse relabel anchors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-anchor-step", type=int, default=21)
    parser.add_argument("--visibility-threshold", type=float, default=0.005)
    parser.add_argument("--close-distance-m", type=float, default=2.2)
    parser.add_argument("--anchors-per-rollout", type=int, default=3)
    return parser.parse_args()


def step_record(raw: dict[str, Any], threshold: float, close_m: float) -> dict[str, Any]:
    policy = raw.get("policy") or {}
    evaluation = raw.get("evaluation_only_after_action") or {}
    probability = float(policy.get("visibility_probability", 0.0))
    gt_visible = bool(evaluation.get("gt_visible", False))
    distance = float(evaluation.get("gt_distance_m", float("inf")))
    return {
        "step": int(raw["step"]),
        "gt_visible": gt_visible,
        "predicted_visible": probability >= threshold,
        "visibility_probability": probability,
        "gt_distance_m": distance,
        "close": distance <= close_m,
        "collision": bool(evaluation.get("human_collision", False)),
        "policy_mode": str(policy.get("mode", "unknown")),
    }


def candidate_priority(current: dict[str, Any], previous: dict[str, Any] | None) -> tuple[str, int]:
    if previous is not None and not previous["gt_visible"] and current["gt_visible"]:
        return "reacquisition", 0
    if previous is not None and previous["gt_visible"] and not current["gt_visible"]:
        return "target_loss", 1
    if not current["gt_visible"] and current["predicted_visible"]:
        return "false_visible", 2
    if current["gt_visible"] and not current["predicted_visible"]:
        return "false_invisible", 3
    if current["gt_visible"] and current["close"]:
        return "close_visible", 4
    if not current["gt_visible"]:
        return "lost", 5
    return "visible", 6


def main() -> int:
    args = arguments()
    root = args.root.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve(strict=False)
    rollouts = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        result_path = directory / "result.json"
        result_kind = "complete"
        if not result_path.is_file():
            result_path = directory / "result.partial.json"
            result_kind = "partial"
        if not result_path.is_file():
            continue
        value = json.loads(result_path.read_text(encoding="utf-8"))
        if value.get("split") != "train" or value.get("input_audit", {}).get(
            "passed"
        ) is not True:
            raise ValueError(f"inadmissible rollout provenance: {result_path}")
        records = [
            step_record(item, args.visibility_threshold, args.close_distance_m)
            for item in value.get("steps", [])
        ]
        eligible = [item for item in records if item["step"] >= args.minimum_anchor_step]
        candidates = []
        seen_categories: set[str] = set()
        previous_by_step = {item["step"]: item for item in records}
        ranked = []
        for item in eligible:
            category, priority = candidate_priority(
                item, previous_by_step.get(item["step"] - 1)
            )
            ranked.append((priority, item["step"], category, item))
        for _, _, category, item in sorted(ranked):
            if category in seen_categories:
                continue
            candidates.append({"category": category, **item})
            seen_categories.add(category)
            if len(candidates) >= args.anchors_per_rollout:
                break
        if len(candidates) < args.anchors_per_rollout:
            selected_steps = {item["step"] for item in candidates}
            for _, _, category, item in sorted(ranked):
                if item["step"] in selected_steps:
                    continue
                candidates.append({"category": category, **item})
                selected_steps.add(item["step"])
                if len(candidates) >= args.anchors_per_rollout:
                    break
        rollouts.append(
            {
                "directory": str(directory),
                "result": str(result_path),
                "result_kind": result_kind,
                "task": value.get("task"),
                "dataset_index": value.get("dataset_index"),
                "episode_id": value.get("episode_id"),
                "steps": len(records),
                "eligible_anchor_steps": len(eligible),
                "visible_steps": sum(item["gt_visible"] for item in records),
                "invisible_steps": sum(not item["gt_visible"] for item in records),
                "false_visible_steps": sum(
                    not item["gt_visible"] and item["predicted_visible"] for item in records
                ),
                "false_invisible_steps": sum(
                    item["gt_visible"] and not item["predicted_visible"] for item in records
                ),
                "close_steps": sum(item["close"] for item in records),
                "collision_steps": sum(item["collision"] for item in records),
                "proposed_anchors": candidates,
            }
        )
    payload = {
        "schema_version": 1,
        "stage": "v2_phase3_model_visited_collection_summary",
        "status": "complete",
        "root": str(root),
        "protocol": {
            "split": "train",
            "minimum_anchor_step": args.minimum_anchor_step,
            "visibility_threshold": args.visibility_threshold,
            "close_distance_m": args.close_distance_m,
            "test_locked_used": False,
        },
        "rollouts": rollouts,
        "totals": {
            "rollouts": len(rollouts),
            "complete": sum(item["result_kind"] == "complete" for item in rollouts),
            "partial": sum(item["result_kind"] == "partial" for item in rollouts),
            "steps": sum(item["steps"] for item in rollouts),
            "eligible_anchor_steps": sum(item["eligible_anchor_steps"] for item in rollouts),
        },
        "test_locked_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
