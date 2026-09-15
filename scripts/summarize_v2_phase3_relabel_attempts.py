#!/usr/bin/env python3
"""Summarize selected Phase-3 relabel attempts without admitting failures."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    selection_path = args.selection.expanduser().resolve(strict=True)
    root = args.root.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite relabel summary: {output}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (
        selection.get("stage") != "v2_phase3_multiscene_anchor_selection_v1"
        or selection.get("split") != "train"
        or selection.get("test_locked_used") is not False
    ):
        raise ValueError("relabel selection provenance mismatch")

    rows = []
    success_by_task = Counter()
    success_by_kind = Counter()
    for item in selection["selected"]:
        tag = (
            f"{item['task']}_{item['dataset_index']}_{item['candidate_kind']}"
            f"_step{item['anchor_environment_step']}"
        )
        exit_path = root / "logs" / f"{tag}.EXIT_CODE"
        sample_path = root / "samples" / tag / "sample.json"
        exit_code = (
            int(exit_path.read_text(encoding="utf-8").strip())
            if exit_path.is_file()
            else None
        )
        passed = exit_code == 0 and sample_path.is_file()
        row = {
            "tag": tag,
            "task": item["task"],
            "dataset_index": item["dataset_index"],
            "episode_id": item["episode_id"],
            "anchor_environment_step": item["anchor_environment_step"],
            "candidate_kind": item["candidate_kind"],
            "waypoint_supervision": item["waypoint_supervision"],
            "exit_code": exit_code,
            "passed": passed,
            "sample": str(sample_path) if passed else None,
        }
        rows.append(row)
        if passed:
            success_by_task[item["task"]] += 1
            success_by_kind[item["candidate_kind"]] += 1
    result = {
        "schema_version": 1,
        "stage": "v2_phase3_multiscene_relabel_summary_v1",
        "status": "complete",
        "attempted": len(rows),
        "passed": sum(row["passed"] for row in rows),
        "failed": sum(not row["passed"] for row in rows),
        "passed_by_task": dict(sorted(success_by_task.items())),
        "passed_by_candidate_kind": dict(sorted(success_by_kind.items())),
        "attempts": rows,
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
