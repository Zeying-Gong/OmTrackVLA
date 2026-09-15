#!/usr/bin/env python3
"""Select observable and safe-stop anchors from non-locked EVT train rollouts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-size", type=int, default=8)
    parser.add_argument("--history-stride-steps", type=int, default=3)
    parser.add_argument("--expert-horizon-steps", type=int, default=21)
    return parser.parse_args()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def admitted_rollout(path: Path) -> Mapping[str, Any]:
    value = load(path)
    audit = value.get("input_audit", {}) if isinstance(value, Mapping) else {}
    if (
        not isinstance(value, Mapping)
        or value.get("split") != "train"
        or value.get("uwb_mode") != "missing"
        or audit.get("passed") is not True
        or audit.get("direct_gt_target_point_argument_used") is not False
    ):
        raise ValueError(f"inadmissible Phase-3 train rollout: {path}")
    return value


def select_one(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if row["candidate_kind"] == key]
    if not candidates:
        return None
    if key == "observable_visible":
        # Hard visible examples have the lowest predicted visibility.
        return min(candidates, key=lambda row: row["visibility_probability"])
    # Hard absent examples have the highest false-visible confidence.
    return max(candidates, key=lambda row: row["visibility_probability"])


def main() -> int:
    args = arguments()
    output = args.output.expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite anchor selection: {output}")
    if (
        args.history_size != 8
        or args.history_stride_steps != 3
        or args.expert_horizon_steps != 21
    ):
        raise ValueError("Architecture-v2 anchor contract must remain 8/3/21")

    selected = []
    rollouts = []
    for unresolved in args.rollout:
        path = unresolved.expanduser().resolve(strict=True)
        value = admitted_rollout(path)
        steps = value.get("steps")
        if not isinstance(steps, list) or len(steps) < 22:
            raise ValueError(f"rollout is too short for Phase-3 anchors: {path}")
        visible = {0: True}
        for expected, record in enumerate(steps, 1):
            if record.get("step") != expected:
                raise ValueError(f"non-contiguous rollout steps: {path}")
            visible[expected] = bool(
                record["evaluation_only_after_action"]["gt_visible"]
            )
        rows = []
        last_expert_anchor = len(steps) - args.expert_horizon_steps
        for anchor in range(21, len(steps) + 1):
            record = steps[anchor - 1]
            history_steps = [
                anchor - args.history_stride_steps * age
                for age in reversed(range(args.history_size))
            ]
            history_visible = [visible[step] for step in history_steps]
            current_visible = visible[anchor]
            observable = any(history_visible)
            probability = float(record["policy"]["visibility_probability"])
            if current_visible and anchor <= last_expert_anchor:
                kind = "observable_visible"
            elif (
                not current_visible
                and observable
                and anchor <= last_expert_anchor
            ):
                kind = "observable_recent_loss"
            elif not observable:
                kind = "safe_stop"
            else:
                continue
            rows.append(
                {
                    "candidate_kind": kind,
                    "anchor_environment_step": anchor,
                    "history_environment_steps": history_steps,
                    "history_gt_visible": history_visible,
                    "current_gt_visible": current_visible,
                    "target_observable_in_model_input": observable,
                    "visibility_probability": probability,
                    "false_visible_at_0_005": (
                        not current_visible and probability >= 0.005
                    ),
                    "false_invisible_at_0_005": (
                        current_visible and probability < 0.005
                    ),
                    "gt_distance_m": float(
                        record["evaluation_only_after_action"]["gt_distance_m"]
                    ),
                }
            )
        chosen = []
        for kind in (
            "observable_visible",
            "observable_recent_loss",
            "safe_stop",
        ):
            item = select_one(rows, kind)
            if item is not None:
                item = dict(item)
                item.update(
                    {
                        "rollout_result": str(path),
                        "task": str(value["task"]),
                        "dataset_index": int(value["dataset_index"]),
                        "episode_id": str(value["episode_id"]),
                        "scene_id": str(value["scene_id"]),
                        "waypoint_supervision": (
                            "safe_stop" if kind == "safe_stop" else "expert_waypoint"
                        ),
                    }
                )
                chosen.append(item)
                selected.append(item)
        rollouts.append(
            {
                "rollout_result": str(path),
                "task": str(value["task"]),
                "dataset_index": int(value["dataset_index"]),
                "episode_id": str(value["episode_id"]),
                "steps": len(steps),
                "eligible_counts": {
                    kind: sum(row["candidate_kind"] == kind for row in rows)
                    for kind in (
                        "observable_visible",
                        "observable_recent_loss",
                        "safe_stop",
                    )
                },
                "selected": chosen,
            }
        )
    result = {
        "schema_version": 1,
        "stage": "v2_phase3_multiscene_anchor_selection_v1",
        "history_size": args.history_size,
        "history_stride_environment_steps": args.history_stride_steps,
        "expert_horizon_environment_steps": args.expert_horizon_steps,
        "selection_rule": {
            "observable_visible": "lowest policy visibility probability",
            "observable_recent_loss": "highest policy false-visible probability",
            "safe_stop": "highest policy false-visible probability",
            "one_per_kind_per_rollout": True,
        },
        "rollouts": rollouts,
        "selected": selected,
        "selected_count": len(selected),
        "split": "train",
        "test_locked_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
