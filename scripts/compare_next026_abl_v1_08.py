#!/usr/bin/env python3
"""Compare direct and Architecture-v1 Phase-1-initialized NEXT-026 arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MATCHED_MANIFEST_FIELDS = (
    "seed",
    "world_size",
    "dataset_index_sha256",
    "samples_per_epoch",
    "planned_optimizer_steps",
    "maximum_optimizer_steps",
    "batch_size_per_device",
    "sequence_steps",
    "policy_stride",
    "loss_weights",
    "simulated_uwb",
    "perception_cache",
    "test_locked_used",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-run", type=Path, required=True)
    parser.add_argument("--phase1-init-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _improvement(baseline: float, candidate: float) -> dict[str, float]:
    return {
        "direct": baseline,
        "phase1_init": candidate,
        "absolute_reduction": baseline - candidate,
        "relative_reduction_percent": 100.0 * (baseline - candidate) / baseline,
    }


def main() -> int:
    args = arguments()
    direct_run = args.direct_run.expanduser().resolve(strict=True)
    phase1_run = args.phase1_init_run.expanduser().resolve(strict=True)
    direct_manifest = _json(direct_run / "run_manifest.json")
    phase1_manifest = _json(phase1_run / "run_manifest.json")
    mismatches = {
        field: {
            "direct": direct_manifest.get(field),
            "phase1_init": phase1_manifest.get(field),
        }
        for field in MATCHED_MANIFEST_FIELDS
        if direct_manifest.get(field) != phase1_manifest.get(field)
    }
    if mismatches:
        raise ValueError(f"ABL-V1-08 training protocol mismatch: {mismatches}")
    initialization = phase1_manifest.get("initialization", {})
    if (
        initialization.get("checkpoint_phase") != 1
        or initialization.get("parameter_coverage") != 1.0
        or initialization.get("missing_or_mismatched") != []
    ):
        raise ValueError("Phase 1 initialization provenance/coverage is incomplete")

    direct_metrics = _json(direct_run / "eval" / "metrics.json")
    phase1_metrics = _json(phase1_run / "eval" / "metrics.json")
    for name, metrics in (("direct", direct_metrics), ("phase1_init", phase1_metrics)):
        if (
            metrics.get("status") != "complete"
            or metrics.get("split") != "val"
            or metrics.get("samples_per_mode") != 1024
            or metrics.get("test_locked_used") is not False
            or metrics.get("perception_cache") is not False
            or metrics.get("simulated_uwb") is not True
        ):
            raise ValueError(f"{name} evaluation provenance is not comparable")

    modes = {}
    for mode in ("visual_uwb", "visual_only", "uwb_only", "safe_stop"):
        direct_mode = direct_metrics["modes"][mode]
        phase1_mode = phase1_metrics["modes"][mode]
        modes[mode] = {
            "waypoint_ade_m": _improvement(
                float(direct_mode["waypoint_ade_m"]),
                float(phase1_mode["waypoint_ade_m"]),
            ),
            "waypoint_fde_m": _improvement(
                float(direct_mode["waypoint_fde_m"]),
                float(phase1_mode["waypoint_fde_m"]),
            ),
            "bbox_iou_delta": float(phase1_mode["bbox_iou"])
            - float(direct_mode["bbox_iou"]),
            "visibility_accuracy_delta": float(
                phase1_mode["visibility_accuracy"]
            )
            - float(direct_mode["visibility_accuracy"]),
            "ego_translation_error_m_delta": float(
                phase1_mode["ego_translation_error_m"]
            )
            - float(direct_mode["ego_translation_error_m"]),
            "ego_yaw_error_rad_delta": float(phase1_mode["ego_yaw_error_rad"])
            - float(direct_mode["ego_yaw_error_rad"]),
            "finite_prediction_coverage": float(
                phase1_mode["finite_prediction_coverage"]
            ),
            "stop_accuracy": float(phase1_mode["stop_accuracy"]),
        }

    direct_normal = direct_metrics["normal_modes"]
    phase1_normal = phase1_metrics["normal_modes"]
    normal = {
        "waypoint_ade_m": _improvement(
            float(direct_normal["waypoint_ade_m"]),
            float(phase1_normal["waypoint_ade_m"]),
        ),
        "waypoint_fde_m": _improvement(
            float(direct_normal["waypoint_fde_m"]),
            float(phase1_normal["waypoint_fde_m"]),
        ),
        "finite_prediction_coverage": float(
            phase1_normal["finite_prediction_coverage"]
        ),
    }
    hard_gates = {
        "finite_prediction_coverage_is_one": normal[
            "finite_prediction_coverage"
        ]
        == 1.0,
        "safe_stop_accuracy_at_least_0_99": modes["safe_stop"]["stop_accuracy"]
        >= 0.99,
        "normal_ade_improved": normal["waypoint_ade_m"]["absolute_reduction"]
        > 0.0,
        "normal_fde_improved": normal["waypoint_fde_m"]["absolute_reduction"]
        > 0.0,
    }
    direct_selection = _json(direct_run / "eval" / "checkpoint_selection.json")
    phase1_selection = _json(phase1_run / "eval" / "checkpoint_selection.json")
    report = {
        "schema_version": 1,
        "task": "NEXT-026 ABL-V1-08 Phase 1 initialization comparison",
        "status": "passed" if all(hard_gates.values()) else "failed",
        "training_protocol_equal": True,
        "matched_manifest_fields": list(MATCHED_MANIFEST_FIELDS),
        "direct_run": str(direct_run),
        "phase1_init_run": str(phase1_run),
        "initialization": initialization,
        "selected_checkpoint_steps": {
            "direct": direct_selection["selected"]["checkpoint_global_step"],
            "phase1_init": phase1_selection["selected"]["checkpoint_global_step"],
        },
        "normal_modes": normal,
        "modes": modes,
        "hard_gates": hard_gates,
        "scope": {
            "phase1_init_comparison_complete": True,
            "osnet_teacher_on_off_complete": False,
            "evt_bench_used_for_training": False,
            "test_locked_used": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
