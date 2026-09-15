#!/usr/bin/env python3
"""Freeze audited train-split NEXT-007 relabel samples for Phase 3."""
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

from smoke_next007_phase3_sample import _load_json, validate_sample


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=Path, action="append", default=[])
    parser.add_argument("--pipeline-complete", type=Path, action="append", default=[])
    parser.add_argument("--minimum-samples", type=int, default=6)
    parser.add_argument("--history-size", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    if args.minimum_samples < 3:
        raise ValueError("formal Phase 3 needs at least three audited samples")
    repository = Path(__file__).resolve().parents[1]
    output = args.output.expanduser()
    if not output.is_absolute():
        output = repository / output
    output = output.resolve(strict=False)

    requested = [path.expanduser() for path in args.sample]
    pipelines: dict[Path, dict] = {}
    for unresolved in args.pipeline_complete:
        path = unresolved.expanduser()
        if not path.is_absolute():
            path = repository / path
        path = path.resolve(strict=True)
        value = _load_json(path)
        if (
            value.get("status") != "complete"
            or value.get("formal_training_eligible") is not True
            or value.get("test_locked_used") is not False
        ):
            raise ValueError(f"unadmitted Phase 3 pipeline completion: {path}")
        sample = Path(str(value["sample"])).resolve(strict=True)
        if sha256(sample) != value.get("sample_sha256"):
            raise ValueError(f"pipeline sample checksum mismatch: {path}")
        requested.append(sample)
        pipelines[sample] = {
            "path": str(path),
            "sha256": sha256(path),
            "relabel_report": value.get("relabel_report"),
            "relabel_report_sha256": value.get("relabel_report_sha256"),
        }

    records = []
    seen_ids: set[str] = set()
    for unresolved in requested:
        path = unresolved if unresolved.is_absolute() else repository / unresolved
        path = path.resolve(strict=True)
        value = validate_sample(
            _load_json(path),
            path,
            args.history_size,
            expected_split="train",
            require_formal_eligible=True,
        )
        source = value["source"]
        if source.get("test_locked_used") is not False:
            raise ValueError(f"sample does not explicitly forbid test_locked: {path}")
        sample_id = str(value["sample_id"])
        if sample_id in seen_ids:
            raise ValueError(f"duplicate Phase 3 model-visited state: {sample_id}")
        seen_ids.add(sample_id)
        parts = sample_id.split("/")
        if len(parts) < 4 or parts[0] != "habitat" or parts[1] not in {"stt", "dt", "at"}:
            raise ValueError(f"unexpected Phase 3 sample id: {sample_id}")
        records.append({
            "sample_id": sample_id,
            "task": parts[1],
            "path": str(path),
            "sha256": sha256(path),
            "dataset_index": int(source["dataset_index"]),
            "episode_id": str(source["episode_id"]),
            "scene_id": str(source["scene_id"]),
            "anchor_environment_step": int(source["anchor_environment_step"]),
            "pipeline": pipelines.get(path),
        })

    records.sort(key=lambda item: (item["task"], item["dataset_index"], item["sample_id"]))
    tasks = Counter(record["task"] for record in records)
    if len(records) < args.minimum_samples:
        raise ValueError(
            f"only {len(records)} admitted samples; minimum is {args.minimum_samples}"
        )
    missing_tasks = sorted({"stt", "dt", "at"} - set(tasks))
    if missing_tasks:
        raise ValueError(f"Phase 3 manifest lacks tasks: {missing_tasks}")

    payload = {
        "schema_version": 1,
        "stage": "next007_phase3_recovery_manifest_v1",
        "status": "admitted",
        "history_size": args.history_size,
        "sample_count": len(records),
        "task_counts": dict(sorted(tasks.items())),
        "samples": records,
        "input_contract": {
            "split": "train",
            "model_visited_states": True,
            "expert_labels_passed_quality_gate": True,
            "later_bbox_used": False,
            "gt_target_pose_or_depth_model_input": False,
            "test_locked_used": False,
        },
        "test_locked_used": False,
    }
    unsigned = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_sha256"] = hashlib.sha256(unsigned).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(json.dumps({
        "status": "admitted",
        "samples": len(records),
        "task_counts": dict(sorted(tasks.items())),
        "manifest": str(output),
        "manifest_sha256": payload["manifest_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
