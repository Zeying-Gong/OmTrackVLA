#!/usr/bin/env python3
"""Verify Phase-3 recovery train/validation scene and state separation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_v2_phase3_model_visited import sha256, verify_manifest


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def partition(path: Path) -> tuple[set[str], set[tuple[str, int, str, int]]]:
    manifest = verify_manifest(path)
    scenes = set()
    states = set()
    for record in manifest["samples"]:
        sample_path = Path(record["path"]).resolve(strict=True)
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
        source = sample["source"]
        scenes.add(str(source["scene_id"]))
        states.add(
            (
                str(record["task"]),
                int(source["dataset_index"]),
                str(source["episode_id"]),
                int(source["anchor_environment_step"]),
            )
        )
    return scenes, states


def main() -> int:
    args = arguments()
    train_path = args.train_manifest.expanduser().resolve(strict=True)
    validation_path = args.validation_manifest.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite partition audit: {output}")
    train_scenes, train_states = partition(train_path)
    validation_scenes, validation_states = partition(validation_path)
    overlapping_scenes = sorted(train_scenes & validation_scenes)
    overlapping_states = sorted(train_states & validation_states)
    if overlapping_scenes or overlapping_states:
        raise ValueError(
            "Phase-3 train/validation leakage: "
            f"scenes={overlapping_scenes}, states={overlapping_states}"
        )
    result = {
        "schema_version": 1,
        "stage": "v2_phase3_recovery_partition_disjoint_audit_v1",
        "status": "passed",
        "train_manifest": str(train_path),
        "train_manifest_sha256": sha256(train_path),
        "validation_manifest": str(validation_path),
        "validation_manifest_sha256": sha256(validation_path),
        "train_scenes": sorted(train_scenes),
        "validation_scenes": sorted(validation_scenes),
        "overlapping_scenes": [],
        "overlapping_states": [],
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
