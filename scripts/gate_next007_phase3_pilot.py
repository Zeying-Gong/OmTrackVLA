#!/usr/bin/env python3
"""Gate Phase-3 pilot before spending time on Habitat closed-loop replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-complete", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    value = json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("test_locked_used") is not False:
        raise ValueError(f"invalid or locked report: {path}")
    return value


def main() -> int:
    args = arguments()
    training = load(args.training_complete)
    baseline = load(args.baseline_metrics)
    candidate = load(args.candidate_metrics)
    before = training["recovery_training_set_metrics_before"]
    after = training["recovery_training_set_metrics"]
    base_normal = baseline["normal_modes"]
    cand_normal = candidate["normal_modes"]
    safe_stop = candidate["modes"]["safe_stop"]

    gates = {
        "training_completed": training.get("status") == "training_complete",
        "recovery_ade_improved": float(after["ade_m"]) < float(before["ade_m"]),
        "recovery_fde_improved": float(after["fde_m"]) < float(before["fde_m"]),
        "recovery_path_ratio_closer_to_one": abs(1.0 - float(after["path_length_ratio"])) < abs(1.0 - float(before["path_length_ratio"])),
        "val_ade_retained_within_10_percent": float(cand_normal["waypoint_ade_m"]) <= 1.10 * float(base_normal["waypoint_ade_m"]),
        "val_fde_retained_within_10_percent": float(cand_normal["waypoint_fde_m"]) <= 1.10 * float(base_normal["waypoint_fde_m"]),
        "val_path_ratio_not_shorter_by_over_0_05": float(cand_normal["path_length_ratio_of_sums"]) >= float(base_normal["path_length_ratio_of_sums"]) - 0.05,
        "finite_prediction_coverage": float(cand_normal["finite_prediction_coverage"]) == 1.0,
        "safe_stop_accuracy": float(safe_stop["stop_accuracy"]) >= 0.99,
        "safe_stop_path_length": float(safe_stop["predicted_path_length_mean_m"]) <= 0.03,
    }
    passed = all(gates.values())
    report = {
        "schema_version": 1,
        "stage": "next007_phase3_recovery_pilot_gate_v1",
        "status": "passed" if passed else "failed",
        "gates": gates,
        "recovery_before": before,
        "recovery_after": after,
        "baseline_normal_modes": base_normal,
        "candidate_normal_modes": cand_normal,
        "candidate_safe_stop": safe_stop,
        "test_locked_used": False,
    }
    output = args.output.expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"status": report["status"], "gates": gates, "output": str(output)}, sort_keys=True))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
