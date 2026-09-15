#!/usr/bin/env python3
"""Summarize frozen-front-end ABL-09 against the converged Architecture-v1 arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--architecture-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _policy_metrics(value: dict[str, object]) -> dict[str, object]:
    return {
        "waypoint_ade_m": float(value["waypoint_ade_m"]),
        "waypoint_fde_m": float(value["waypoint_fde_m"]),
        "path_length_ratio": value.get(
            "path_length_ratio", value.get("path_length_ratio_mean")
        ),
        "finite_prediction_coverage": float(value["finite_prediction_coverage"]),
    }


def _comparison(baseline: dict[str, object], architecture: dict[str, object]) -> dict[str, float]:
    result = {}
    for field in ("waypoint_ade_m", "waypoint_fde_m"):
        base = float(baseline[field])
        full = float(architecture[field])
        result[f"{field}_baseline_minus_architecture"] = base - full
        result[f"{field}_relative_architecture_improvement"] = (
            (base - full) / base if base else 0.0
        )
    return result


def main() -> int:
    args = _arguments()
    baseline_payload = _load(args.baseline_metrics)
    architecture_payload = _load(args.architecture_summary)
    if baseline_payload.get("split") != "val" or baseline_payload.get("test_locked_used") is not False:
        raise ValueError("ABL-09 baseline must be an unlocked val evaluation")
    baseline_open = baseline_payload["benchmarks"]["B2-OPEN"]
    teacher_on = architecture_payload["teacher_on"]
    teacher_off = architecture_payload["teacher_off"]
    architecture_arms = (
        teacher_on["gru_curriculum"],
        teacher_on["single_step"],
        teacher_off["single_step"],
    )
    if any(arm.get("test_locked_used") is not False for arm in architecture_arms):
        raise ValueError("Architecture-v1 reference has invalid test_locked provenance")
    references = {
        "best_teacher_on_gru_curriculum": _policy_metrics(teacher_on["gru_curriculum"]),
        "teacher_on_single_step_36864": _policy_metrics(teacher_on["single_step"]),
        "teacher_off_single_step_36864": _policy_metrics(teacher_off["single_step"]),
    }
    baseline = {
        **_policy_metrics(baseline_open),
        "checkpoint_global_step": int(baseline_payload["checkpoint_global_step"]),
        "predicted_path_length_mean_m": baseline_open.get("predicted_path_length_mean_m"),
        "target_path_length_mean_m": baseline_open.get("target_path_length_mean_m"),
        "horizon_error_m": baseline_open.get("horizon_error_m"),
    }
    output = {
        "schema_version": 1,
        "ablation": "ABL-V1-09",
        "test_locked_used": False,
        "frozen_frontend_baseline": baseline,
        "architecture_v1_references": references,
        "comparison_to_best_architecture_v1": _comparison(
            baseline, references["best_teacher_on_gru_curriculum"]
        ),
        "comparison_to_teacher_on_single_step_36864": _comparison(
            baseline, references["teacher_on_single_step_36864"]
        ),
        "fairness": {
            "same_sage3d_run_level_train_val_split": True,
            "same_phase2_optimizer_steps_as_single_step": 36864,
            "same_phase2_samples_as_single_step": 1179648,
            "same_global_batch_size": 32,
            "same_condition_modes": [
                "visual_uwb",
                "visual_only",
                "uwb_only",
                "safe_stop",
            ],
            "validation_samples_per_mode": 1024,
            "ade_definition": "mean over nontrivial future points 1..7; point 0 is the zero anchor",
            "anchor_alignment": (
                "same episode split and stride, but frozen cache starts at initial+1 "
                "whereas E2E starts at max(history-1, initial+1), normally initial+3; "
                "sample IDs therefore differ by 1-2 simulator steps"
            ),
            "evt_bench_used_for_waypoint_training": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
