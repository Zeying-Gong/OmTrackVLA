#!/usr/bin/env python3
"""Determine whether a Phase-3 target is observable in each causal RGB history."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from smoke_next007_phase3_sample import _load_json
from train_v2_phase3_model_visited import sha256, verify_manifest


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition", choices=("train", "recovery_val"), required=True)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite observability audit: {output}")
    manifest = verify_manifest(manifest_path)
    rows = []
    modes = Counter()
    for record in manifest["samples"]:
        sample_path = Path(record["path"]).resolve(strict=True)
        sample = _load_json(sample_path)
        source = sample["source"]
        rollout_path = Path(source["rollout_result"]).resolve(strict=True)
        if sha256(rollout_path) != source["rollout_result_sha256"]:
            raise ValueError(f"rollout checksum mismatch: {rollout_path}")
        rollout = _load_json(rollout_path)
        if rollout.get("split") != "train" or rollout.get("task") != record["task"]:
            raise ValueError(f"rollout provenance mismatch: {rollout_path}")
        visible_by_step = {
            int(item["step"]): bool(item["evaluation_only_after_action"]["gt_visible"])
            for item in rollout["steps"]
        }
        history_steps = [
            int(item["environment_step"])
            for item in sample["model_inputs"]["rgb_history"]
        ]
        history_visible = []
        for step in history_steps:
            if step == 0:
                # Initialization is admitted only with a target bbox in the first frame.
                history_visible.append(True)
            elif step in visible_by_step:
                history_visible.append(visible_by_step[step])
            else:
                raise ValueError(f"history step {step} is absent from {rollout_path}")
        anchor = int(source["anchor_environment_step"])
        current_visible = bool(sample["supervision"]["target_state"]["visible"])
        if anchor in visible_by_step and visible_by_step[anchor] != current_visible:
            raise ValueError(f"current visibility label drifted: {sample_path}")
        uwb_valid = bool(sample["model_inputs"]["uwb"]["valid"])
        observable = bool(any(history_visible) or uwb_valid)
        mode = "expert_waypoint" if observable else "safe_stop"
        modes[mode] += 1
        rows.append(
            {
                "sample_id": record["sample_id"],
                "task": record["task"],
                "category": record["category"],
                "sample_path": str(sample_path),
                "sample_sha256": record["sha256"],
                "history_steps": history_steps,
                "history_gt_visible": history_visible,
                "current_gt_visible": current_visible,
                "uwb_valid": uwb_valid,
                "target_observable_in_model_input": observable,
                "waypoint_supervision": mode,
                "safe_stop_reason": (
                    "target absent from the full causal RGB history and UWB invalid"
                    if mode == "safe_stop"
                    else None
                ),
            }
        )
    payload = {
        "schema_version": 1,
        "stage": "architecture_v2_phase3_observability_audit_v1",
        "status": "passed",
        "partition": args.partition,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256(manifest_path),
        "sample_count": len(rows),
        "supervision_counts": dict(sorted(modes.items())),
        "samples": rows,
        "rule": {
            "expert_waypoint": "target visible in at least one of the eight causal RGB frames, or UWB valid",
            "safe_stop": "target absent from all eight causal RGB frames and UWB invalid",
            "gt_direction_never_added_to_model_inputs": True,
        },
        "test_locked_used": False,
    }
    unsigned = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["audit_sha256"] = hashlib.sha256(unsigned).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "passed",
        "partition": args.partition,
        "samples": len(rows),
        "supervision_counts": dict(sorted(modes.items())),
        "audit": str(output),
        "audit_sha256": payload["audit_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
