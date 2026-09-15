#!/usr/bin/env python3
"""Freeze audited Architecture-v2 model-visited samples for a small Phase-3 pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from smoke_v2_phase3_minibatch import validate_v2_sample
from smoke_next007_phase3_sample import _load_json, _safe_file


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=Path, action="append", required=True)
    parser.add_argument("--minimum-samples", type=int, default=9)
    parser.add_argument("--minimum-per-task", type=int, default=3)
    return parser.parse_args()


def category(path: Path, task: str) -> str:
    name = path.parent.name
    if name.startswith(task + "_"):
        value = name[len(task) + 1 :]
        if "_step" in value:
            value = value.rsplit("_step", 1)[0]
        return value
    return "earlier_audited_smoke"


def main() -> int:
    args = arguments()
    output = args.output.expanduser().resolve(strict=False)
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_states: set[tuple[str, int, str, int]] = set()
    for unresolved in args.sample:
        path = unresolved.expanduser().resolve(strict=True)
        value = validate_v2_sample(_load_json(path), path)
        inputs = value["model_inputs"]
        root = path.parent.resolve(strict=True)
        initial = inputs["initial_rgb"]
        _safe_file(root, initial["rgb_path"], initial["sha256"])
        for image in inputs["rgb_history"]:
            _safe_file(root, image["rgb_path"], image["sha256"])
        sample_id = str(value["sample_id"])
        parts = sample_id.split("/")
        if len(parts) != 4 or parts[0] != "habitat" or parts[1] not in {
            "stt", "dt", "at"
        }:
            raise ValueError(f"invalid Architecture-v2 sample id: {sample_id}")
        if sample_id in seen_ids:
            raise ValueError(f"duplicate Architecture-v2 sample id: {sample_id}")
        seen_ids.add(sample_id)
        task = parts[1]
        source = value["source"]
        state_key = (
            task,
            int(source["dataset_index"]),
            str(source["episode_id"]),
            int(source["anchor_environment_step"]),
        )
        if state_key in seen_states:
            raise ValueError(f"duplicate model-visited state: {state_key}")
        seen_states.add(state_key)
        expert = np.asarray(
            value["supervision"]["expert_trajectory"]["waypoints_base_xy_m"],
            dtype=np.float64,
        )
        stop_required = bool(
            value["supervision"]["expert_trajectory"].get("stop_required", False)
        )
        path_length = float(np.linalg.norm(np.diff(expert, axis=0), axis=1).sum())
        terminal_distance = float(np.linalg.norm(expert[-1]))
        path_valid = (
            path_length <= 1.0e-7 and terminal_distance <= 1.0e-7
            if stop_required
            else 0.20 <= path_length <= 2.40
        )
        if not (math.isfinite(path_length) and path_valid):
            raise ValueError(f"implausible expert path length for {sample_id}: {path_length}")
        target = value["supervision"]["target_state"]
        records.append(
            {
                "sample_id": sample_id,
                "task": task,
                "category": category(path, task),
                "path": str(path),
                "sha256": sha256(path),
                "dataset_index": state_key[1],
                "episode_id": state_key[2],
                "anchor_environment_step": state_key[3],
                "visible": bool(target["visible"]),
                "polar_valid": bool(target["polar_valid"]),
                "stop_required": stop_required,
                "expert_path_length_m": path_length,
                "expert_terminal_distance_m": terminal_distance,
            }
        )
    records.sort(key=lambda item: (item["task"], item["dataset_index"], item["anchor_environment_step"]))
    counts = Counter(item["task"] for item in records)
    if len(records) < args.minimum_samples:
        raise ValueError(f"only {len(records)} samples; minimum is {args.minimum_samples}")
    for task in ("stt", "dt", "at"):
        if counts[task] < args.minimum_per_task:
            raise ValueError(f"task {task} has only {counts[task]} samples")
    visible = sum(item["visible"] for item in records)
    invisible = len(records) - visible
    if visible == 0 or invisible == 0:
        raise ValueError("Phase-3 pilot must contain visible and invisible target states")
    payload = {
        "schema_version": 1,
        "stage": "architecture_v2_phase3_model_visited_manifest_v1",
        "status": "frozen",
        "sample_count": len(records),
        "task_counts": dict(sorted(counts.items())),
        "visible_samples": visible,
        "invisible_samples": invisible,
        "history_size": 8,
        "history_stride_environment_steps": 3,
        "samples": records,
        "input_contract": {
            "split": "train",
            "model_visited_states": True,
            "expert_waypoints_label_side_only": True,
            "target_state_label_side_only": True,
            "uwb_valid": False,
            "test_locked_used": False,
        },
        "formal_large_scale_training": False,
        "test_locked_used": False,
    }
    unsigned = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_sha256"] = hashlib.sha256(unsigned).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {output}")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "frozen",
        "manifest": str(output),
        "sample_count": len(records),
        "task_counts": dict(sorted(counts.items())),
        "visible_samples": visible,
        "invisible_samples": invisible,
        "manifest_sha256": payload["manifest_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
