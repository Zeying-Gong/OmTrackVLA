#!/usr/bin/env python3
"""Small Architecture-v2 Phase-3 pilot with clean SAGE3D replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from smoke_next007_phase3_sample import _load_json, _rgb_tensor, _safe_file
from smoke_v2_phase3_minibatch import validate_v2_sample


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(repository: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository / path


def verify_manifest(path: Path) -> dict[str, Any]:
    value = _load_json(path)
    if not isinstance(value, dict):
        raise ValueError("Phase-3 manifest is not an object")
    unsigned = dict(value)
    expected = unsigned.pop("manifest_sha256", None)
    actual = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if expected != actual:
        raise ValueError("Phase-3 manifest checksum mismatch")
    if (
        value.get("stage") != "architecture_v2_phase3_model_visited_manifest_v1"
        or value.get("status") != "frozen"
        or value.get("test_locked_used") is not False
        or value.get("formal_large_scale_training") is not False
        or value.get("history_size") != 8
        or value.get("history_stride_environment_steps") != 3
    ):
        raise ValueError("Phase-3 manifest admission mismatch")
    if set(value.get("task_counts", {})) != {"stt", "dt", "at"}:
        raise ValueError("Phase-3 manifest lacks STT/DT/AT coverage")
    return value


def load_recovery_samples(
    manifest: Mapping[str, Any], architecture: Any, torch: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples = []
    metadata = []
    for record in manifest["samples"]:
        path = Path(str(record["path"])).resolve(strict=True)
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Phase-3 sample checksum mismatch: {path}")
        value = validate_v2_sample(_load_json(path), path)
        if value["sample_id"] != record["sample_id"]:
            raise ValueError(f"Phase-3 sample identity changed: {path}")
        root = path.parent.resolve(strict=True)
        inputs = value["model_inputs"]
        initial = inputs["initial_rgb"]
        initial_path = _safe_file(root, initial["rgb_path"], initial["sha256"])
        history_paths = [
            _safe_file(root, item["rgb_path"], item["sha256"])
            for item in inputs["rgb_history"]
        ]
        uwb = inputs["uwb"]
        expert = value["supervision"]["expert_trajectory"]
        state = value["supervision"]["target_state"]
        bbox = state["bbox_xyxy_norm"] or [0.0, 0.0, 0.0, 0.0]
        samples.append(
            {
                "initial_rgb": _rgb_tensor(
                    initial_path, architecture.image_height, architecture.image_width, torch
                ),
                "initial_bbox": torch.tensor(
                    inputs["initial_bbox_xyxy_norm"], dtype=torch.float32
                ),
                "ego_rgb": torch.stack(
                    [
                        _rgb_tensor(
                            image,
                            architecture.image_height,
                            architecture.image_width,
                            torch,
                        )
                        for image in history_paths
                    ]
                ),
                "visual_initialization_valid": torch.tensor(1.0),
                "rgb_valid": torch.tensor(1.0),
                "binding_valid": torch.tensor(1.0),
                "uwb_xy": torch.tensor(
                    uwb["relative_position_base_xy_m"], dtype=torch.float32
                ),
                "uwb_covariance_xy": torch.tensor(
                    uwb["covariance_base_xy_m2"], dtype=torch.float32
                ),
                "uwb_quality": torch.tensor(float(uwb["quality_01"])),
                "uwb_age_s": torch.tensor(float(uwb["age_s"])),
                "uwb_valid": torch.tensor(0.0),
                "camera_intrinsics": torch.tensor(
                    inputs["camera_intrinsics"], dtype=torch.float32
                ),
                "camera_from_base": torch.tensor(
                    inputs["camera_from_base"], dtype=torch.float32
                ),
                "waypoints": torch.tensor(
                    expert["waypoints_base_xy_m"], dtype=torch.float32
                ),
                "waypoint_mask": torch.tensor(expert["valid_mask"], dtype=torch.bool),
                "bbox": torch.tensor(bbox, dtype=torch.float32),
                "visible": torch.tensor(float(state["visible"])),
                "angle": torch.tensor(state["angle_sincos"], dtype=torch.float32),
                "distance": torch.tensor(float(state["distance_m"])),
                "polar_valid": torch.tensor(bool(state["polar_valid"])),
            }
        )
        metadata.append(
            {
                "sample_id": value["sample_id"],
                "task": record["task"],
                "category": record["category"],
                "path": str(path),
                "sha256": record["sha256"],
            }
        )
    return samples, metadata


def recovery_batch(samples: list[dict[str, Any]], indices: list[int], device: Any) -> dict[str, Any]:
    keys = samples[0].keys()
    return {
        key: __import__("torch").stack([samples[index][key] for index in indices]).to(
            device, non_blocking=True
        )
        for key in keys
    }


def recovery_inputs(batch: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "initial_rgb",
        "initial_bbox",
        "ego_rgb",
        "visual_initialization_valid",
        "rgb_valid",
        "binding_valid",
        "uwb_xy",
        "uwb_covariance_xy",
        "uwb_quality",
        "uwb_age_s",
        "uwb_valid",
        "camera_intrinsics",
        "camera_from_base",
    )
    return {key: batch[key] for key in keys}


def recovery_loss(
    outputs: Mapping[str, Any], batch: Mapping[str, Any], weights: Mapping[str, float], F: Any
) -> tuple[Any, dict[str, Any]]:
    mask = batch["waypoint_mask"].bool().clone()
    mask[:, 0] = False
    expanded = mask[..., None].expand_as(outputs["waypoints"])
    waypoint = F.smooth_l1_loss(
        outputs["waypoints"], batch["waypoints"], reduction="none"
    )[expanded].mean()
    visible = batch["visible"] > 0.5
    bbox = (
        F.smooth_l1_loss(outputs["bbox_pred"][visible], batch["bbox"][visible])
        if visible.any()
        else outputs["bbox_pred"].sum() * 0.0
    )
    visibility = F.binary_cross_entropy_with_logits(
        outputs["visibility_logit"].squeeze(-1), batch["visible"]
    )
    polar_valid = batch["polar_valid"].bool()
    if polar_valid.any():
        predicted_angle = F.normalize(
            outputs["target_angle_sincos"][polar_valid].float(), dim=1
        )
        target_angle = F.normalize(batch["angle"][polar_valid].float(), dim=1)
        angle = (1.0 - (predicted_angle * target_angle).sum(dim=1)).mean()
        distance = F.smooth_l1_loss(
            __import__("torch").log1p(
                outputs["target_distance_m"].squeeze(-1)[polar_valid]
            ),
            __import__("torch").log1p(batch["distance"][polar_valid]),
        )
    else:
        angle = outputs["target_angle_sincos"].sum() * 0.0
        distance = outputs["target_distance_m"].sum() * 0.0
    values = {
        "waypoint": waypoint,
        "bbox": bbox,
        "visibility": visibility,
        "angle": angle,
        "distance": distance,
    }
    total = sum(float(weights.get(name, 0.0)) * value for name, value in values.items())
    values["total"] = total
    return total, values


def recovery_metrics(policy: Any, samples: list[dict[str, Any]], device: Any, torch: Any) -> dict[str, float]:
    totals = {
        "ade_sum": 0.0,
        "fde_sum": 0.0,
        "path_ratio_sum": 0.0,
        "visibility_correct": 0.0,
        "bbox_abs_sum": 0.0,
        "bbox_count": 0.0,
        "angle_abs_deg_sum": 0.0,
        "distance_abs_sum": 0.0,
        "polar_count": 0.0,
        "samples": 0.0,
    }
    policy.eval()
    use_bfloat16 = torch.cuda.is_bf16_supported()
    with torch.inference_mode():
        for start in range(0, len(samples), 2):
            indices = list(range(start, min(start + 2, len(samples))))
            batch = recovery_batch(samples, indices, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
                outputs = policy(**recovery_inputs(batch))
            prediction = outputs["waypoints"].float()
            target = batch["waypoints"].float()
            error = torch.linalg.vector_norm(prediction[:, 1:] - target[:, 1:], dim=-1)
            pred_length = torch.linalg.vector_norm(
                prediction[:, 1:] - prediction[:, :-1], dim=-1
            ).sum(dim=-1)
            target_length = torch.linalg.vector_norm(
                target[:, 1:] - target[:, :-1], dim=-1
            ).sum(dim=-1).clamp_min(1.0e-6)
            totals["ade_sum"] += float(error.sum())
            totals["fde_sum"] += float(error[:, -1].sum())
            totals["path_ratio_sum"] += float((pred_length / target_length).sum())
            visible_prediction = outputs["visibility_logit"].squeeze(-1) >= 0.0
            totals["visibility_correct"] += float(
                (visible_prediction == batch["visible"].bool()).sum()
            )
            visible = batch["visible"] > 0.5
            if visible.any():
                bbox_error = (outputs["bbox_pred"][visible] - batch["bbox"][visible]).abs()
                totals["bbox_abs_sum"] += float(bbox_error.sum())
                totals["bbox_count"] += float(bbox_error.numel())
            polar_valid = batch["polar_valid"].bool()
            if polar_valid.any():
                predicted_angle = torch.nn.functional.normalize(
                    outputs["target_angle_sincos"][polar_valid].float(), dim=1
                )
                target_angle = torch.nn.functional.normalize(
                    batch["angle"][polar_valid].float(), dim=1
                )
                cosine = (predicted_angle * target_angle).sum(dim=1).clamp(-1.0, 1.0)
                totals["angle_abs_deg_sum"] += float(
                    torch.acos(cosine).sum() * 180.0 / math.pi
                )
                totals["distance_abs_sum"] += float(
                    (
                        outputs["target_distance_m"].squeeze(-1)[polar_valid]
                        - batch["distance"][polar_valid]
                    ).abs().sum()
                )
                totals["polar_count"] += float(polar_valid.sum())
            totals["samples"] += len(indices)
    count = max(1.0, totals["samples"])
    return {
        "samples": int(totals["samples"]),
        "ade_m": totals["ade_sum"] / (count * 7.0),
        "fde_m": totals["fde_sum"] / count,
        "path_length_ratio": totals["path_ratio_sum"] / count,
        "visibility_accuracy": totals["visibility_correct"] / count,
        "bbox_mae_normalized": totals["bbox_abs_sum"] / max(1.0, totals["bbox_count"]),
        "polar_angle_mae_deg": totals["angle_abs_deg_sum"] / max(1.0, totals["polar_count"]),
        "polar_distance_mae_m": totals["distance_abs_sum"] / max(1.0, totals["polar_count"]),
    }


def clean_metrics(
    policy: Any,
    dataset: Any,
    sample_count: int,
    device: Any,
    torch: Any,
    default_collate: Any,
    step_inputs: Any,
    targets_fn: Any,
) -> dict[str, float]:
    totals = {"ade_sum": 0.0, "fde_sum": 0.0, "samples": 0.0}
    policy.eval()
    use_bfloat16 = torch.cuda.is_bf16_supported()
    indices = list(range(min(sample_count, dataset.base_length)))
    with torch.inference_mode():
        for start in range(0, len(indices), 2):
            selected = indices[start : start + 2]
            batch = default_collate([dataset[index] for index in selected])
            targets = targets_fn(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
                outputs = policy(**step_inputs(batch, device))
            error = torch.linalg.vector_norm(
                outputs["waypoints"].float()[:, 1:] - targets["waypoints"].float()[:, 1:],
                dim=-1,
            )
            totals["ade_sum"] += float(error.sum())
            totals["fde_sum"] += float(error[:, -1].sum())
            totals["samples"] += len(selected)
    count = max(1.0, totals["samples"])
    return {
        "samples": int(totals["samples"]),
        "ade_m": totals["ade_sum"] / (count * 7.0),
        "fde_m": totals["fde_sum"] / count,
    }


def atomic_torch_save(torch: Any, path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def main() -> int:
    args = arguments()
    config_path = args.config.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite Phase-3 pilot: {output_dir}")

    import torch
    import yaml
    from torch.nn import functional as F
    from torch.utils.data._utils.collate import default_collate

    from omtrackvla.data.end_to_end_training import Sage3DEndToEndSequenceDataset
    from omtrackvla.models.end_to_end import (
        ArchitectureV1Ablation,
        ArchitectureV1Config,
        load_official_da3_small_l11,
    )
    from omtrackvla.models.end_to_end_v2 import (
        ArchitectureV2DecoderConfig,
        ArchitectureV2FollowPolicy,
        parameter_inventory_v2,
        waypoint_gradient_report_v2,
    )
    from omtrackvla.training.end_to_end_v2 import _losses as clean_loss_fn
    from omtrackvla.training.end_to_end_v2 import _step_inputs as clean_step_inputs
    from omtrackvla.training.end_to_end_v2 import _targets as clean_targets

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (
        not isinstance(config, Mapping)
        or config.get("method") != "architecture_v2_evt_perception_polar"
        or config.get("formal_large_scale_training") is not False
        or config.get("test_locked_used") is not False
    ):
        raise ValueError("expected a non-locked Architecture-v2 Phase-3 pilot config")
    repository = config_path.parents[2]
    for field in ("da3_source", "da3_runtime"):
        extra = resolve(repository, config[field]).resolve(strict=True)
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    architecture = ArchitectureV1Config(**config["architecture"])
    decoder = ArchitectureV2DecoderConfig(**config["decoder"])
    if architecture.history_size != 8 or decoder.history_stride_raw != 3:
        raise ValueError("Phase-3 pilot requires eight frames at 0.1-second spacing")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Phase-3 pilot requires CUDA")
    torch.cuda.set_device(device)
    random.seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    torch.set_float32_matmul_precision("high")

    manifest_path = resolve(repository, config["recovery_manifest"]).resolve(strict=True)
    manifest = verify_manifest(manifest_path)
    recovery_samples, recovery_metadata = load_recovery_samples(
        manifest, architecture, torch
    )
    clean_modes = tuple(config["data"]["clean_condition_modes"])
    clean_dataset = Sage3DEndToEndSequenceDataset(
        resolve(repository, config["data"]["sequence_index"]),
        split="train",
        config=architecture,
        modes=clean_modes,
        history_stride_raw=decoder.history_stride_raw,
    )
    clean_val_dataset = Sage3DEndToEndSequenceDataset(
        resolve(repository, config["data"]["sequence_index"]),
        split="viz_val",
        config=architecture,
        modes=clean_modes,
        history_stride_raw=decoder.history_stride_raw,
    )

    da3, da3_loading = load_official_da3_small_l11(
        resolve(repository, config["da3_model"]),
        architecture,
        ArchitectureV1Ablation(backbone_tuning="adapter"),
    )
    policy = ArchitectureV2FollowPolicy(da3, architecture, decoder).to(device)
    parent_path = resolve(repository, config["parent_checkpoint"]).resolve(strict=True)
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if (
        parent.get("method") != "architecture_v2_evt_perception_polar"
        or parent.get("test_locked_used") is not False
    ):
        raise ValueError("Phase-3 parent checkpoint provenance mismatch")
    policy.load_state_dict(parent["model"], strict=True)
    for name, parameter in policy.da3.named_parameters():
        parameter.requires_grad_(".adapter." in name)
    inventory = parameter_inventory_v2(policy)
    training = config["training"]
    adapter_parameters = []
    polar_parameters = []
    policy_parameters = []
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".adapter." in name:
            adapter_parameters.append(parameter)
        elif name.startswith("trajectory.target_polar_head."):
            polar_parameters.append(parameter)
        else:
            policy_parameters.append(parameter)
    if not adapter_parameters or not polar_parameters or not policy_parameters:
        raise RuntimeError("Phase-3 optimizer parameter partition is incomplete")
    optimizer = torch.optim.AdamW(
        [
            {"params": policy_parameters, "lr": float(training["policy_learning_rate"]), "name": "policy"},
            {"params": polar_parameters, "lr": float(training["polar_learning_rate"]), "name": "polar"},
            {"params": adapter_parameters, "lr": float(training["adapter_learning_rate"]), "name": "da3_adapter"},
        ],
        weight_decay=float(training["weight_decay"]),
    )
    maximum_steps = int(args.max_steps or training["max_steps"])
    warmup = int(training["warmup_steps"])

    def schedule(step: int) -> float:
        if step < warmup:
            return max(1.0e-3, (step + 1) / max(1, warmup))
        progress = (step - warmup) / max(1, maximum_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    use_bfloat16 = bool(training.get("bfloat16", True)) and torch.cuda.is_bf16_supported()
    clean_weights = {str(k): float(v) for k, v in training["clean_loss_weights"].items()}
    recovery_weights = {
        str(k): float(v) for k, v in training["recovery_loss_weights"].items()
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output_dir / "config.yaml")
    shutil.copy2(manifest_path, output_dir / "recovery_manifest.json")
    start = time.time()

    before_recovery = recovery_metrics(policy, recovery_samples, device, torch)
    before_clean = clean_metrics(
        policy,
        clean_val_dataset,
        int(config["data"]["clean_validation_samples"]),
        device,
        torch,
        default_collate,
        clean_step_inputs,
        clean_targets,
    )
    (output_dir / "METRICS_BEFORE.json").write_text(
        json.dumps({"recovery": before_recovery, "clean_sage3d": before_clean}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # The deployment waypoint objective alone must reach the visual backbone and policy.
    policy.train()
    policy.da3.eval()
    probe_batch = recovery_batch(recovery_samples, [0], device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
        probe_output = policy(**recovery_inputs(probe_batch))
        _, probe_values = recovery_loss(
            probe_output, probe_batch, {"waypoint": 1.0}, F
        )
    probe_values["waypoint"].backward()
    recovery_gradients = waypoint_gradient_report_v2(policy)
    required = (
        "da3_adapter",
        "l11_projector",
        "scene_resampler",
        "context_encoder",
        "trajectory_decoder",
        "delta_head",
    )
    missing = [name for name in required if recovery_gradients.get(name, 0.0) <= 0.0]
    if missing:
        raise RuntimeError(f"recovery waypoint-only gradient is zero: {missing}")
    optimizer.zero_grad(set_to_none=True)
    uwb_probe = default_collate([clean_dataset[0]])
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
        uwb_output = policy(**clean_step_inputs(uwb_probe, device))
        uwb_target = clean_targets(uwb_probe, device)
        _, uwb_values = clean_loss_fn(uwb_output, uwb_target, {"waypoint": 1.0})
    uwb_values["waypoint"].backward()
    uwb_gradients = waypoint_gradient_report_v2(policy)
    if uwb_gradients.get("uwb_encoder", 0.0) <= 0.0:
        raise RuntimeError("clean visual+UWB waypoint loss did not reach UWB encoder")
    optimizer.zero_grad(set_to_none=True)
    (output_dir / "WAYPOINT_ONLY_BACKWARD.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "recovery_visual_only": recovery_gradients,
                "clean_visual_uwb": uwb_gradients,
                "recovery_required_nonzero": list(required),
                "recovery_uwb_zero_expected": True,
                "loss_source": "waypoint_only",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    run_manifest = {
        "schema_version": 1,
        "stage": config["stage"],
        "method": config["method"],
        "formal_large_scale_training": False,
        "maximum_optimizer_steps": maximum_steps,
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_sha256": sha256(parent_path),
        "recovery_manifest": str(manifest_path),
        "recovery_manifest_sha256": sha256(manifest_path),
        "recovery_samples": len(recovery_samples),
        "recovery_task_counts": manifest["task_counts"],
        "clean_dataset_sequences": len(clean_dataset),
        "clean_validation_sequences": len(clean_val_dataset),
        "clean_dataset_index": str(clean_dataset.index_path),
        "clean_dataset_index_sha256": clean_dataset.index_sha256,
        "clean_condition_modes": list(clean_modes),
        "optimizer": {
            "policy_learning_rate": float(training["policy_learning_rate"]),
            "polar_learning_rate": float(training["polar_learning_rate"]),
            "adapter_learning_rate": float(training["adapter_learning_rate"]),
            "policy_parameter_tensors": len(policy_parameters),
            "polar_parameter_tensors": len(polar_parameters),
            "adapter_parameter_tensors": len(adapter_parameters),
        },
        "loss_weights": {"clean": clean_weights, "recovery": recovery_weights},
        "da3_pretrained_loading": da3_loading,
        "parameter_inventory": inventory,
        "history": "8 frames at 0.1-second spacing",
        "gpu_zero_used": False,
        "test_locked_used": False,
    }
    (output_dir / "RUN_MANIFEST.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    clean_weight = float(training["clean_weight"])
    recovery_weight = float(training["recovery_weight"])
    log_path = output_dir / "train_log.jsonl"
    for step in range(maximum_steps):
        policy.train()
        policy.da3.eval()
        optimizer.zero_grad(set_to_none=True)
        mode = step % len(clean_modes)
        pool_size = (clean_dataset.base_length - 1 - mode) // len(clean_modes) + 1
        clean_index = mode + len(clean_modes) * random.randrange(pool_size)
        clean_batch = default_collate([clean_dataset[clean_index]])
        clean_target = clean_targets(clean_batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
            clean_output = policy(**clean_step_inputs(clean_batch, device))
            clean_total, clean_values = clean_loss_fn(
                clean_output, clean_target, clean_weights
            )
        (clean_weight * clean_total).backward()
        del clean_output, clean_total

        recovery_index = step % len(recovery_samples)
        rec_batch = recovery_batch(recovery_samples, [recovery_index], device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
            rec_output = policy(**recovery_inputs(rec_batch))
            rec_total, rec_values = recovery_loss(
                rec_output, rec_batch, recovery_weights, F
            )
        (recovery_weight * rec_total).backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            policy_parameters + polar_parameters + adapter_parameters,
            float(training["gradient_clip_norm"]),
        )
        optimizer.step()
        scheduler.step()
        global_step = step + 1
        if global_step == 1 or global_step % int(training["log_every_steps"]) == 0:
            record = {
                "global_step": global_step,
                "clean_index": clean_index,
                "clean_mode": clean_modes[mode],
                "recovery_index": recovery_index,
                "recovery_sample_id": recovery_metadata[recovery_index]["sample_id"],
                "clean_losses": {
                    name: float(value.detach().float().cpu())
                    for name, value in clean_values.items()
                },
                "recovery_losses": {
                    name: float(value.detach().float().cpu())
                    for name, value in rec_values.items()
                },
                "gradient_norm": float(gradient_norm.detach().float().cpu()),
                "learning_rates": {
                    group["name"]: float(group["lr"]) for group in optimizer.param_groups
                },
                "elapsed_s": time.time() - start,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
        if global_step % int(training["checkpoint_every_steps"]) == 0:
            checkpoint = {
                "schema_version": 1,
                "phase": 3,
                "method": config["method"],
                "stage": config["stage"],
                "global_step": global_step,
                "model": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "parent_checkpoint_sha256": sha256(parent_path),
                "recovery_manifest_sha256": sha256(manifest_path),
                "formal_large_scale_training": False,
                "test_locked_used": False,
            }
            atomic_torch_save(
                torch,
                output_dir / "checkpoints" / f"step_{global_step:07d}.ckpt",
                checkpoint,
            )

    after_recovery = recovery_metrics(policy, recovery_samples, device, torch)
    after_clean = clean_metrics(
        policy,
        clean_val_dataset,
        int(config["data"]["clean_validation_samples"]),
        device,
        torch,
        default_collate,
        clean_step_inputs,
        clean_targets,
    )
    gate = config["selection_gate"]
    recovery_required = before_recovery["ade_m"] * (
        1.0 - float(gate["minimum_recovery_ade_relative_improvement"])
    )
    clean_allowed = before_clean["ade_m"] * (
        1.0 + float(gate["maximum_clean_ade_relative_regression"])
    ) + float(gate["maximum_clean_ade_absolute_slack_m"])
    selected = (
        after_recovery["ade_m"] <= recovery_required
        and after_clean["ade_m"] <= clean_allowed
    )
    final_checkpoint = {
        "schema_version": 1,
        "phase": 3,
        "method": config["method"],
        "stage": config["stage"],
        "global_step": maximum_steps,
        "model": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "parent_checkpoint_sha256": sha256(parent_path),
        "recovery_manifest_sha256": sha256(manifest_path),
        "formal_large_scale_training": False,
        "test_locked_used": False,
    }
    last_path = output_dir / "checkpoints" / "last.ckpt"
    atomic_torch_save(torch, last_path, final_checkpoint)
    best_path = None
    if selected:
        best_path = output_dir / "checkpoints" / "best.ckpt"
        shutil.copy2(last_path, best_path)
    report = {
        "status": "pilot_complete",
        "selected": selected,
        "selection_gate": {
            "recovery_ade_required_max_m": recovery_required,
            "clean_ade_allowed_max_m": clean_allowed,
        },
        "before": {"recovery": before_recovery, "clean_sage3d": before_clean},
        "after": {"recovery": after_recovery, "clean_sage3d": after_clean},
        "optimizer_steps": maximum_steps,
        "last_checkpoint": str(last_path),
        "best_checkpoint": str(best_path) if best_path is not None else None,
        "elapsed_s": time.time() - start,
        "formal_large_scale_training": False,
        "gpu_zero_used": False,
        "test_locked_used": False,
    }
    (output_dir / "PILOT_COMPLETE.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
