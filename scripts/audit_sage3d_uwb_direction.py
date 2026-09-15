#!/usr/bin/env python3
"""Audit UWB-bearing and expert-waypoint direction agreement without RGB."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from omtrackvla.data.sage3d_policy import canonical_waypoints, target_position_base


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--descriptors-per-split", type=int, default=256)
    return parser.parse_args()


def selected_lateral(waypoints: np.ndarray) -> float:
    radius = np.linalg.norm(waypoints, axis=1)
    candidates = np.flatnonzero(radius >= 0.15)
    selected = int(candidates[0]) if candidates.size else int(radius.argmax())
    return float(waypoints[max(1, selected), 1])


def summary(uwb: np.ndarray, expert: np.ndarray) -> dict[str, Any]:
    bearing = np.abs(uwb) >= 0.05
    movement = np.abs(expert) >= 0.02
    both = bearing & movement
    return {
        "anchors": int(len(uwb)),
        "uwb_lateral_quantiles_m": np.quantile(
            uwb, [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
        ).tolist(),
        "expert_selected_lateral_quantiles_m": np.quantile(
            expert, [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
        ).tolist(),
        "expert_vs_uwb_lateral_correlation": float(np.corrcoef(expert, uwb)[0, 1]),
        "expert_vs_uwb_sign": {
            "uwb_threshold_m": 0.05,
            "expert_threshold_m": 0.02,
            "count": int(both.sum()),
            "agreement": float(
                (np.sign(expert[both]) == np.sign(uwb[both])).mean()
            )
            if both.any()
            else None,
        },
        "expert_nontrivial_lateral_count": int(movement.sum()),
        "expert_positive_fraction_when_nontrivial": float(
            (expert[movement] > 0.0).mean()
        )
        if movement.any()
        else None,
    }


def main() -> int:
    args = arguments()
    if args.descriptors_per_split <= 0:
        raise ValueError("descriptors-per-split must be positive")
    index_path = args.index.expanduser().resolve(strict=True)
    value = json.loads(index_path.read_text(encoding="utf-8"))
    if value.get("test_locked_used") is not False:
        raise ValueError("UWB direction audit refuses a locked index")
    root = Path(value["source_root"]).resolve(strict=True)
    stride = int(value["anchor_stride"])
    report: dict[str, Any] = {
        "schema_version": 1,
        "stage": "sage3d_uwb_expert_direction_audit",
        "test_locked_used": False,
        "index": str(index_path),
        "splits": {},
    }
    for split in ("train", "val"):
        descriptors = value["splits"][split]
        count = min(len(descriptors), args.descriptors_per_split)
        indices = np.linspace(0, len(descriptors) - 1, count, dtype=np.int64)
        uwb_values: list[float] = []
        expert_values: list[float] = []
        for descriptor_index in indices:
            descriptor = descriptors[int(descriptor_index)]
            source = root / descriptor["relative"] / "derived.json"
            steps = json.loads(source.read_text(encoding="utf-8"))["steps"]
            for anchor in range(
                int(descriptor["anchor_start"]),
                int(descriptor["anchor_end"]) + 1,
                stride,
            ):
                uwb_values.append(float(target_position_base(steps[anchor])[1]))
                expert_values.append(
                    selected_lateral(np.asarray(canonical_waypoints(steps, anchor)))
                )
        report["splits"][split] = summary(
            np.asarray(uwb_values), np.asarray(expert_values)
        )
        report["splits"][split]["sampled_descriptors"] = int(count)
    output = args.output.expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
