#!/usr/bin/env python3
"""Export per-sample predictions for observable Phase-3 pilot checkpoints."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from train_v2_phase3_model_visited import (
    load_recovery_samples,
    recovery_batch,
    recovery_inputs,
    resolve,
    sha256,
    verify_manifest,
)
from train_v2_phase3_observable import (
    apply_observability,
    verify_observability_audit,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("checkpoint must use LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise ValueError("checkpoint label and path must be non-empty")
    return label, Path(raw_path)


def parameter_delta_norm(
    first: Mapping[str, Any], second: Mapping[str, Any], prefix: str, torch: Any
) -> float:
    squares = []
    for name, value in first.items():
        if name.startswith(prefix):
            squares.append((second[name].float() - value.float()).square().sum())
    if not squares:
        raise ValueError(f"parameter prefix absent: {prefix}")
    return float(torch.stack(squares).sum().sqrt())


def main() -> int:
    args = arguments()
    config_path = args.config.expanduser().resolve(strict=True)
    output_path = args.output.expanduser().resolve(strict=False)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic: {output_path}")

    import torch
    import yaml

    from omtrackvla.models.end_to_end import (
        ArchitectureV1Ablation,
        ArchitectureV1Config,
        load_official_da3_small_l11,
    )
    from omtrackvla.models.end_to_end_v2 import (
        ArchitectureV2DecoderConfig,
        ArchitectureV2FollowPolicy,
    )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (
        not isinstance(config, Mapping)
        or config.get("method") != "architecture_v2_evt_perception_polar"
        or config.get("test_locked_used") is not False
    ):
        raise ValueError("diagnostic config provenance mismatch")
    repository = config_path.parents[2]
    for field in ("da3_source", "da3_runtime"):
        extra = resolve(repository, config[field]).resolve(strict=True)
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    architecture = ArchitectureV1Config(**config["architecture"])
    decoder = ArchitectureV2DecoderConfig(**config["decoder"])
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("observable diagnostic requires CUDA")
    torch.cuda.set_device(device)

    manifest_path = resolve(repository, config["recovery_val_manifest"]).resolve(
        strict=True
    )
    audit_path = resolve(
        repository, config["recovery_val_observability_audit"]
    ).resolve(strict=True)
    manifest = verify_manifest(manifest_path)
    audit = verify_observability_audit(
        audit_path, manifest_path, "recovery_val"
    )
    samples, metadata = load_recovery_samples(manifest, architecture, torch)
    apply_observability(samples, metadata, audit, torch)

    da3, _ = load_official_da3_small_l11(
        resolve(repository, config["da3_model"]),
        architecture,
        ArchitectureV1Ablation(backbone_tuning="adapter"),
    )
    policy = ArchitectureV2FollowPolicy(da3, architecture, decoder).to(device)
    checkpoints = []
    seen_labels = set()
    for raw in args.checkpoint:
        label, raw_path = parse_checkpoint(raw)
        if label in seen_labels:
            raise ValueError(f"duplicate checkpoint label: {label}")
        seen_labels.add(label)
        path = resolve(repository, raw_path).resolve(strict=True)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if (
            checkpoint.get("method") != config["method"]
            or checkpoint.get("test_locked_used") is not False
        ):
            raise ValueError(f"checkpoint provenance mismatch: {path}")
        checkpoints.append((label, path, checkpoint))

    rows = []
    states = {}
    use_bfloat16 = torch.cuda.is_bf16_supported()
    for label, path, checkpoint in checkpoints:
        policy.load_state_dict(checkpoint["model"], strict=True)
        policy.eval()
        states[label] = checkpoint["model"]
        predictions = []
        with torch.inference_mode():
            for index, (sample, item) in enumerate(zip(samples, metadata)):
                batch = recovery_batch(samples, [index], device)
                with torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=use_bfloat16
                ):
                    model_output = policy(**recovery_inputs(batch))
                predicted = model_output["waypoints"][0].float()
                target = batch["waypoints"][0].float()
                errors = torch.linalg.vector_norm(
                    predicted[1:] - target[1:], dim=-1
                )
                path_length = torch.linalg.vector_norm(
                    predicted[1:] - predicted[:-1], dim=-1
                ).sum()
                visibility_logit = float(
                    model_output["visibility_logit"][0, 0].float()
                )
                stop_logit = float(model_output["stop_logit"][0, 0].float())
                predictions.append(
                    {
                        "sample_id": item["sample_id"],
                        "task": item["task"],
                        "category": item["category"],
                        "waypoint_supervision": item["waypoint_supervision"],
                        "gt_visible": bool(sample["visible"]),
                        "stop_target": bool(sample["stop"]),
                        "visibility_logit": visibility_logit,
                        "visibility_probability": float(
                            torch.sigmoid(torch.tensor(visibility_logit))
                        ),
                        "stop_logit": stop_logit,
                        "stop_probability": float(
                            torch.sigmoid(torch.tensor(stop_logit))
                        ),
                        "ade_m": float(errors.mean()),
                        "fde_m": float(errors[-1]),
                        "predicted_path_m": float(path_length),
                        "predicted_waypoints_xy_m": predicted.cpu().tolist(),
                    }
                )
        rows.append(
            {
                "label": label,
                "checkpoint": str(path),
                "checkpoint_sha256": sha256(path),
                "global_step": checkpoint.get("global_step"),
                "samples": predictions,
            }
        )

    deltas = []
    baseline_label = checkpoints[0][0]
    for label, _, _ in checkpoints[1:]:
        deltas.append(
            {
                "from": baseline_label,
                "to": label,
                "visibility_head_delta_norm": parameter_delta_norm(
                    states[baseline_label], states[label],
                    "trajectory.visibility_head.", torch,
                ),
                "stop_head_delta_norm": parameter_delta_norm(
                    states[baseline_label], states[label],
                    "trajectory.stop_head.", torch,
                ),
                "delta_head_delta_norm": parameter_delta_norm(
                    states[baseline_label], states[label],
                    "trajectory.delta_head.", torch,
                ),
            }
        )
    result = {
        "schema_version": 1,
        "stage": "v2_013_phase3_observable_per_sample_diagnostic",
        "deployment_visibility_threshold": float(
            config["training"]["deployment_visibility_threshold"]
        ),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "checkpoints": rows,
        "parameter_deltas": deltas,
        "test_locked_used": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
