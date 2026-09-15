#!/usr/bin/env python3
"""Select a NEXT-026 checkpoint by fixed non-locked validation metrics."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    candidates = []
    for report_path in args.reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("status") != "complete"
            or report.get("split") != "val"
            or report.get("test_locked_used") is not False
        ):
            raise ValueError(f"invalid NEXT-026 validation report: {report_path}")
        safe = report["modes"]["safe_stop"]
        normal = report["normal_modes"]
        candidates.append(
            {
                "report": str(report_path),
                "checkpoint": str(report["checkpoint"]),
                "checkpoint_global_step": int(report["checkpoint_global_step"]),
                "normal_waypoint_ade_m": float(normal["waypoint_ade_m"]),
                "normal_waypoint_fde_m": float(normal["waypoint_fde_m"]),
                "safe_stop_accuracy": float(safe["stop_accuracy"]),
                "finite_prediction_coverage": float(
                    normal["finite_prediction_coverage"]
                ),
            }
        )
    eligible = [
        candidate
        for candidate in candidates
        if candidate["finite_prediction_coverage"] == 1.0
        and candidate["safe_stop_accuracy"] >= 0.99
    ]
    if not eligible:
        raise RuntimeError("no checkpoint passed finite/safe-stop selection guards")
    selected = min(
        eligible,
        key=lambda candidate: (
            candidate["normal_waypoint_ade_m"],
            candidate["normal_waypoint_fde_m"],
            -candidate["checkpoint_global_step"],
        ),
    )
    source = Path(selected["checkpoint"]).resolve(strict=True)
    destination = args.run_dir.resolve(strict=True) / "checkpoints" / "best.ckpt"
    if source != destination:
        shutil.copy2(source, destination)
    result = {
        "schema_version": 1,
        "task": "NEXT-026 validation checkpoint selection",
        "status": "passed",
        "criterion": (
            "finite=1 and safe_stop_accuracy>=0.99, then minimum normal ADE/FDE"
        ),
        "candidates": candidates,
        "selected": selected,
        "best_checkpoint": str(destination),
        "test_locked_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
