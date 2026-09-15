#!/usr/bin/env python3
"""Summarize paired EVT-Bench person-identification rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def episode_counts(data: dict[str, Any]) -> dict[str, int | float]:
    counts: dict[str, int | float] = {
        "steps": 0,
        "gt_visible": 0,
        "detector_target_available": 0,
        "detected": 0,
        "correct": 0,
        "iou_sum_on_outputs": 0.0,
        "memory_updates": 0,
        "correct_memory_updates": 0,
    }
    for step in data.get("steps", []):
        counts["steps"] += 1
        gt_visible = bool(step.get("perception_gt_visible", step.get("visible")))
        counts["gt_visible"] += int(gt_visible)
        candidates = step.get("perception_candidates") or []
        counts["detector_target_available"] += int(
            any(bool(candidate.get("target_match")) for candidate in candidates)
        )
        detected = float(step.get("perception_confidence") or 0.0) > 0.0
        correct = bool(step.get("target_selection_correct"))
        counts["detected"] += int(detected)
        counts["correct"] += int(correct)
        if detected:
            counts["iou_sum_on_outputs"] += float(step.get("target_bbox_iou") or 0.0)
        updated = bool(step.get("perception_memory_updated"))
        counts["memory_updates"] += int(updated)
        counts["correct_memory_updates"] += int(updated and correct)
    return counts


def aggregate_counts(items: Iterable[dict[str, int | float]]) -> dict[str, Any]:
    totals: dict[str, int | float] = {}
    for item in items:
        for key, value in item.items():
            totals[key] = totals.get(key, 0) + value
    detected = int(totals.get("detected", 0))
    correct = int(totals.get("correct", 0))
    gt_visible = int(totals.get("gt_visible", 0))
    updates = int(totals.get("memory_updates", 0))
    totals.update(
        detector_target_recall=_safe_ratio(
            totals.get("detector_target_available", 0), gt_visible
        ),
        perception_target_precision=_safe_ratio(correct, detected),
        perception_target_recall=_safe_ratio(correct, gt_visible),
        perception_mean_target_iou=_safe_ratio(
            totals.get("iou_sum_on_outputs", 0.0), detected
        ),
        perception_memory_update_precision=(
            _safe_ratio(totals.get("correct_memory_updates", 0), updates)
            if updates else None
        ),
    )
    return totals


def _episode_file(root: Path, method: str, task: str, index: int) -> Path:
    files = sorted((root / method / task / str(index) / task / "val" / "episodes").glob("*.json"))
    if len(files) != 1:
        raise RuntimeError(
            f"{method}/{task}/{index}: expected exactly one result, got {len(files)}"
        )
    return files[0]


def _trajectory_signature(data: dict[str, Any]) -> list[tuple[bool, float]]:
    return [
        (bool(step.get("visible")), round(float(step.get("distance_m") or 0.0), 6))
        for step in data.get("steps", [])
    ]


def summarize(
    root: Path,
    methods: tuple[str, ...] = ("kpr", "osnet"),
    tasks: tuple[str, ...] = ("stt", "dt", "at"),
    indices: tuple[int, ...] = (0, 1),
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": 2,
        "benchmark": "evt_bench",
        "split": "val",
        "tasks": list(tasks),
        "dataset_indices": list(indices),
        "test_locked_used": False,
        "methods": {},
        "paired_protocol": {"consistent": True, "mismatches": []},
    }
    loaded: dict[tuple[str, str, int], dict[str, Any]] = {}
    for method in methods:
        episode_rows = []
        method_counts = []
        task_counts: dict[str, list[dict[str, int | float]]] = {
            task: [] for task in tasks
        }
        for task in tasks:
            for index in indices:
                path = _episode_file(root, method, task, index)
                data = json.loads(path.read_text())
                loaded[(method, task, index)] = data
                counts = episode_counts(data)
                method_counts.append(counts)
                task_counts[task].append(counts)
                episode_rows.append(
                    {
                        "task": task,
                        "dataset_index": index,
                        "episode_id": str(data.get("episode_id")),
                        "scene": data.get("scene"),
                        "status": data.get("summary", {}).get("status"),
                        **aggregate_counts([counts]),
                    }
                )
        report["methods"][method] = {
            "aggregate": aggregate_counts(method_counts),
            "by_task": {
                task: aggregate_counts(task_counts[task]) for task in tasks
            },
            "episodes": episode_rows,
        }

    reference_method = methods[0]
    for method in methods[1:]:
        for task in tasks:
            for index in indices:
                reference = loaded[(reference_method, task, index)]
                candidate = loaded[(method, task, index)]
                if (
                    str(reference.get("episode_id")) != str(candidate.get("episode_id"))
                    or reference.get("scene") != candidate.get("scene")
                    or _trajectory_signature(reference) != _trajectory_signature(candidate)
                ):
                    report["paired_protocol"]["consistent"] = False
                    report["paired_protocol"]["mismatches"].append(
                        {
                            "reference_method": reference_method,
                            "method": method,
                            "task": task,
                            "dataset_index": index,
                        }
                    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--methods", nargs="+", default=("kpr", "osnet"))
    parser.add_argument("--tasks", nargs="+", default=("stt", "dt", "at"))
    parser.add_argument("--indices", nargs="+", type=int, default=(0, 1))
    args = parser.parse_args()
    report = summarize(
        args.root,
        methods=tuple(args.methods),
        tasks=tuple(args.tasks),
        indices=tuple(args.indices),
    )
    output = args.output or args.root / "REPORT.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
