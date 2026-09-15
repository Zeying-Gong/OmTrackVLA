#!/usr/bin/env python3
"""Controlled Phase-3 recovery training with clean Phase-2 replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from smoke_next007_phase3_minibatch import _stack_samples


MODEL_INPUT_KEYS = (
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


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--max-steps", type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(repository: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repository / path


def distributed(torch: Any) -> tuple[int, int, int, Any]:
    from torch import distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("NEXT-007 formal Phase 3 requires CUDA")
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return world_size, rank, local_rank, torch.device("cuda", local_rank)


def validate_manifest(path: Path) -> tuple[dict[str, Any], list[Path]]:
    value = load_json(path)
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("stage") != "next007_phase3_recovery_manifest_v1"
        or value.get("status") != "admitted"
        or value.get("test_locked_used") is not False
        or value.get("input_contract", {}).get("split") != "train"
    ):
        raise ValueError("invalid NEXT-007 Phase 3 recovery manifest")
    unsigned = dict(value)
    expected = unsigned.pop("manifest_sha256", None)
    actual = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if expected != actual:
        raise ValueError("Phase 3 recovery manifest checksum mismatch")
    samples = []
    for record in value.get("samples", []):
        sample = Path(str(record["path"])).resolve(strict=True)
        if sha256(sample) != record.get("sha256"):
            raise ValueError(f"Phase 3 sample checksum mismatch: {sample}")
        samples.append(sample)
    if len(samples) != int(value.get("sample_count", -1)) or len(samples) < 3:
        raise ValueError("Phase 3 manifest sample count mismatch")
    return value, samples


def clean_step_batch(batch: Mapping[str, Any], device: Any) -> tuple[dict[str, Any], Any, Any]:
    sequence_keys = {
        "ego_rgb",
        "rgb_valid",
        "binding_valid",
        "uwb_xy",
        "uwb_covariance_xy",
        "uwb_quality",
        "uwb_age_s",
        "uwb_valid",
    }
    inputs = {}
    for key in MODEL_INPUT_KEYS:
        tensor = batch[key]
        if key in sequence_keys:
            tensor = tensor[:, 0]
        inputs[key] = tensor.to(device, non_blocking=True)
    return (
        inputs,
        batch["target_waypoints"][:, 0].to(device, non_blocking=True),
        batch["waypoint_mask"][:, 0].to(device, non_blocking=True),
    )


def clean_step_supervision(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    """Select the first policy step while retaining Phase-2 auxiliary labels."""

    keys = (
        "target_waypoints",
        "waypoint_mask",
        "stop_target",
        "target_bbox",
        "target_visible",
        "identity_label_valid",
        "binding_target",
        "ego_motion_target",
    )
    return {key: batch[key][:, 0].to(device, non_blocking=True) for key in keys}


def select_recovery(
    inputs: Mapping[str, Any], target: Any, mask: Any, indices: list[int]
) -> tuple[dict[str, Any], Any, Any]:
    return (
        {key: value[indices] for key, value in inputs.items()},
        target[indices],
        mask[indices],
    )


def concatenate_inputs(torch: Any, clean: Mapping[str, Any], recovery: Mapping[str, Any]) -> dict[str, Any]:
    if set(clean) != set(recovery):
        raise ValueError("clean/recovery model input keys differ")
    return {key: torch.cat((clean[key], recovery[key]), dim=0) for key in clean}


def path_length(torch: Any, waypoints: Any) -> Any:
    return torch.linalg.vector_norm(waypoints[:, 1:] - waypoints[:, :-1], dim=-1).sum(-1)


def summarize_predictions(torch: Any, output: Mapping[str, Any], target: Any, mask: Any) -> dict[str, float]:
    predicted = output["waypoints"].detach().float()
    target = target.detach().float()
    valid = mask.bool()
    distance = torch.linalg.vector_norm(predicted - target, dim=-1)
    predicted_path = path_length(torch, predicted)
    target_path = path_length(torch, target)
    return {
        "waypoint_loss": float((distance[valid] ** 2).mean().cpu()),
        "ade_m": float(distance[valid].mean().cpu()),
        "fde_m": float(distance[:, -1].mean().cpu()),
        "predicted_path_length_m": float(predicted_path.mean().cpu()),
        "target_path_length_m": float(target_path.mean().cpu()),
        "path_length_ratio": float((predicted_path.sum() / target_path.sum().clamp_min(1.0e-8)).cpu()),
        "predicted_terminal_radius_m": float(torch.linalg.vector_norm(predicted[:, -1], dim=-1).mean().cpu()),
        "target_terminal_radius_m": float(torch.linalg.vector_norm(target[:, -1], dim=-1).mean().cpu()),
        "stop_probability": float(
            torch.sigmoid(output["stop_logit"].detach().float()).mean().cpu()
        ),
    }


def _optimizer_parameter_category(name: str) -> str:
    """Assign top-level and nested GRU parameters to the configured GRU LR."""

    if ".adapter." in name:
        return "adapter"
    if name.startswith("gru.") or ".gru." in name:
        return "gru"
    return "policy"


def main() -> int:
    args = arguments()
    config_path = args.config.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    config = load_json(config_path) if config_path.suffix == ".json" else None
    if not isinstance(config, dict):
        raise ValueError("NEXT-007 Phase 3 config must be JSON")
    if (
        config.get("schema_version") != 1
        or config.get("phase") != 3
        or config.get("method") != "architecture_v1_end_to_end"
        or config.get("formal_training") is not True
        or config.get("test_locked_used") is not False
    ):
        raise ValueError("invalid formal NEXT-007 Phase 3 config")

    repository = Path(__file__).resolve().parents[1]
    base_config_path = resolve(repository, config["base_model_config"]).resolve(strict=True)
    import torch
    import yaml
    from torch import distributed as dist
    from torch.nn import functional as F
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data._utils.collate import default_collate

    base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    if (
        base_config.get("method") != "architecture_v1_end_to_end"
        or base_config.get("test_locked_used") is not False
        or base_config.get("data", {}).get("split") != "train"
    ):
        raise ValueError("Phase 3 base model config is not admitted")

    world_size, rank, local_rank, device = distributed(torch)
    seed = int(config["seed"])
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.set_float32_matmul_precision("high")

    for field in ("da3_source", "da3_runtime"):
        path = resolve(repository, base_config[field]).resolve(strict=True)
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from omtrackvla.data.end_to_end_training import CONDITION_MODES, Sage3DEndToEndSequenceDataset
    from omtrackvla.models.end_to_end import (
        ArchitectureV1Ablation,
        ArchitectureV1Config,
        EndToEndFollowPolicy,
        compute_architecture_v1_loss,
        load_official_da3_small_l11,
        parameter_inventory,
        waypoint_gradient_report,
        waypoint_only_loss,
    )

    architecture = ArchitectureV1Config(**base_config.get("architecture", {}))
    architecture.validate()
    ablation = ArchitectureV1Ablation(**base_config.get("ablation", {}))
    ablation.validate()
    manifest_path = resolve(repository, config["recovery_manifest"]).resolve(strict=True)
    manifest, sample_paths = validate_manifest(manifest_path)
    parent_path = resolve(repository, config["parent_checkpoint"]).resolve(strict=True)
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if (
        parent.get("phase") not in {2, 3}
        or parent.get("method") != "architecture_v1_end_to_end"
        or parent.get("test_locked_used") is not False
    ):
        raise ValueError("Phase 3 parent checkpoint is not admitted")

    clean_dataset = Sage3DEndToEndSequenceDataset(
        resolve(repository, base_config["data"]["sequence_index"]),
        split="train",
        config=architecture,
        modes=CONDITION_MODES,
    )
    da3, da3_loading = load_official_da3_small_l11(
        resolve(repository, base_config["da3_model"]), architecture, ablation,
        resolve(repository, base_config["dinov2_model"]) if base_config.get("dinov2_model") else None,
    )
    policy = EndToEndFollowPolicy(da3, architecture, ablation).to(device)
    policy.load_state_dict(parent["model"], strict=True)
    recovery_inputs, recovery_target, recovery_mask, recovery_metadata = _stack_samples(
        sample_paths, architecture, torch, device
    )
    if len(recovery_metadata) != int(manifest["sample_count"]):
        raise RuntimeError("loaded recovery sample count changed")

    training = config["training"]
    trainable_prefixes = tuple(
        str(value) for value in training.get("trainable_parameter_prefixes", [])
    )
    if trainable_prefixes:
        for name, parameter in policy.named_parameters():
            parameter.requires_grad_(name.startswith(trainable_prefixes))
    clean_loss_weights = {
        str(name): float(value)
        for name, value in training.get("clean_loss_weights", {}).items()
    }
    recovery_distillation_weights = {
        str(name): float(value)
        for name, value in training.get("recovery_distillation_weights", {}).items()
    }
    distillation_outputs = {
        "bbox": "bbox_pred",
        "visibility": "visibility_logit",
        "binding": "binding_logit",
        "ego": "xi_hat",
        "stop": "stop_logit",
    }
    unknown_distillation = sorted(
        set(recovery_distillation_weights) - set(distillation_outputs)
    )
    if unknown_distillation:
        raise ValueError(
            f"unknown recovery distillation outputs: {unknown_distillation}"
        )
    recovery_teacher_outputs: dict[str, Any] = {}
    if any(weight != 0.0 for weight in recovery_distillation_weights.values()):
        policy.eval()
        with torch.inference_mode():
            teacher_outputs = policy(**recovery_inputs)
        recovery_teacher_outputs = {
            name: teacher_outputs[output_name].detach().float().clone()
            for name, output_name in distillation_outputs.items()
            if recovery_distillation_weights.get(name, 0.0) != 0.0
        }
    configured_steps = int(training["max_steps"])
    maximum_steps = int(args.max_steps or configured_steps)
    if maximum_steps <= 0 or maximum_steps > configured_steps:
        raise ValueError("invalid Phase 3 optimizer-step budget")
    clean_batch_size = int(training["clean_batch_size_per_device"])
    recovery_batch_size = int(training["recovery_batch_size_per_device"])
    if clean_batch_size != len(CONDITION_MODES) or recovery_batch_size <= 0:
        raise ValueError("Phase 3 v1 requires one clean sample per condition mode and recovery samples")

    trainable = [(name, parameter) for name, parameter in policy.named_parameters() if parameter.requires_grad]
    adapter_parameters = [
        parameter for name, parameter in trainable
        if _optimizer_parameter_category(name) == "adapter"
    ]
    gru_parameters = [
        parameter for name, parameter in trainable
        if _optimizer_parameter_category(name) == "gru"
    ]
    policy_parameters = [
        parameter for name, parameter in trainable
        if _optimizer_parameter_category(name) == "policy"
    ]
    optimizer_groups = []
    if adapter_parameters:
        optimizer_groups.append({
            "params": adapter_parameters,
            "lr": float(training["adapter_learning_rate"]),
            "name": "da3_adapter",
        })
    if policy_parameters:
        optimizer_groups.append({
            "params": policy_parameters,
            "lr": float(training["policy_learning_rate"]),
            "name": "new_policy",
        })
    if gru_parameters:
        optimizer_groups.append({
            "params": gru_parameters,
            "lr": float(training["gru_learning_rate"]),
            "name": "gru",
        })
    if not optimizer_groups:
        raise ValueError("Phase 3 has no trainable parameters")
    optimizer = torch.optim.AdamW(
        optimizer_groups, weight_decay=float(training["weight_decay"])
    )
    warmup = int(training["warmup_steps"])

    def schedule(step: int) -> float:
        if step < warmup:
            return max(1.0e-3, float(step + 1) / max(1, warmup))
        progress = (step - warmup) / max(1, maximum_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    start_step = 0
    if args.resume_from is not None:
        resume_path = args.resume_from.expanduser().resolve(strict=True)
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        if (
            resume.get("phase") != 3
            or resume.get("config_sha256") != sha256(config_path)
            or resume.get("recovery_manifest_sha256") != sha256(manifest_path)
            or resume.get("test_locked_used") is not False
        ):
            raise ValueError("Phase 3 resume checkpoint does not match this run")
        policy.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        start_step = int(resume["global_step"])

    wrapped = policy
    if world_size > 1:
        wrapped = DistributedDataParallel(policy, device_ids=[local_rank], find_unused_parameters=True)
    use_bfloat16 = bool(training.get("bfloat16", True)) and torch.cuda.is_bf16_supported()
    gradient_clip = float(training["gradient_clip_norm"])
    clean_weight = float(training["clean_waypoint_weight"])
    recovery_weight = float(training["recovery_waypoint_weight"])
    checkpoint_every = int(training["checkpoint_every_steps"])
    log_every = int(training["log_every_steps"])

    output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        shutil.copy2(config_path, output_dir / "config.json")
        shutil.copy2(manifest_path, output_dir / "recovery_manifest.json")
        (output_dir / "run_manifest.json").write_text(json.dumps({
            "schema_version": 1,
            "task": "NEXT-007 controlled formal Phase 3 recovery training",
            "stage": config["stage"],
            "world_size": world_size,
            "gpu_policy": "CUDA_VISIBLE_DEVICES must exclude physical GPU 0",
            "parent_checkpoint": str(parent_path),
            "parent_checkpoint_sha256": sha256(parent_path),
            "recovery_manifest": str(manifest_path),
            "recovery_manifest_sha256": sha256(manifest_path),
            "recovery_samples": len(sample_paths),
            "recovery_task_counts": manifest["task_counts"],
            "clean_dataset_sequences": len(clean_dataset),
            "clean_dataset_index": str(clean_dataset.index_path),
            "clean_dataset_index_sha256": clean_dataset.index_sha256,
            "clean_condition_modes": list(CONDITION_MODES),
            "maximum_optimizer_steps": maximum_steps,
            "clean_batch_size_per_device": clean_batch_size,
            "recovery_batch_size_per_device": recovery_batch_size,
            "clean_waypoint_weight": clean_weight,
            "recovery_waypoint_weight": recovery_weight,
            "clean_loss_weights": clean_loss_weights,
            "recovery_distillation_weights": recovery_distillation_weights,
            "parameter_inventory": parameter_inventory(policy),
            "trainable_parameter_prefixes": list(trainable_prefixes),
            "trainable_parameter_names": [name for name, _ in trainable],
            "frozen_parameter_names": [
                name
                for name, parameter in policy.named_parameters()
                if not parameter.requires_grad
            ],
            "da3_pretrained_loading": da3_loading,
            "waypoint_loss_is_only_training_loss": True,
            "simulated_uwb_in_clean_replay": True,
            "evt_bench_used_for_waypoint_training": False,
            "test_locked_used": False,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if world_size > 1:
        dist.barrier()

    initial_summary = None
    if rank == 0:
        policy.eval()
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16
        ):
            initial_output = policy(**recovery_inputs)
        initial_summary = summarize_predictions(
            torch, initial_output, recovery_target, recovery_mask
        )
        (output_dir / "recovery_metrics_before.json").write_text(
            json.dumps(initial_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if world_size > 1:
        dist.barrier()

    def recovery_indices(step: int) -> list[int]:
        start = (step * world_size * recovery_batch_size + rank * recovery_batch_size) % len(sample_paths)
        return [(start + offset) % len(sample_paths) for offset in range(recovery_batch_size)]

    def clean_indices(step: int) -> list[int]:
        indices = []
        sequence = step * world_size + rank
        for mode in range(len(CONDITION_MODES)):
            pool_size = (len(clean_dataset) - 1 - mode) // len(CONDITION_MODES) + 1
            position = (seed * 1_000_003 + sequence * 97_409 + mode * 65_537) % pool_size
            indices.append(mode + len(CONDITION_MODES) * position)
        return indices

    def checkpoint_payload(step: int) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "phase": 3,
            "method": "architecture_v1_end_to_end",
            "stage": config["stage"],
            "global_step": step,
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "architecture": architecture.to_dict(),
            "ablation": ablation.to_dict(),
            "config_sha256": sha256(config_path),
            "recovery_manifest_sha256": sha256(manifest_path),
            "parent_checkpoint": str(parent_path),
            "parent_checkpoint_sha256": sha256(parent_path),
            "training_only_dynamics": False,
            "training_only_heads": {},
            "da3_pretrained_loading": da3_loading,
            "input_contract": {
                "raw_rgb": True,
                "initial_bbox_once": True,
                "perception_cache": False,
                "external_later_bbox": False,
                "gt_depth_or_target_pose_input": False,
                "evt_bench_waypoint_training": False,
            },
            "test_locked_used": False,
        }

    def save_checkpoint(path: Path, step: int) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint_payload(step), temporary)
        temporary.replace(path)

    policy.train()
    training_started = time.time()
    first_probe_done = (output_dir / "waypoint_only_gradient_report.json").is_file()
    log_path = output_dir / "train_log.jsonl"
    running = {
        "clean": 0.0,
        "clean_waypoint": 0.0,
        "recovery": 0.0,
        "distillation": 0.0,
        "total": 0.0,
        "path_ratio": 0.0,
        "gradient": 0.0,
    }
    running_steps = 0

    for step in range(start_step, maximum_steps):
        clean_batch = default_collate([clean_dataset[index] for index in clean_indices(step)])
        clean_inputs, clean_target, clean_mask = clean_step_batch(clean_batch, device)
        clean_supervision = clean_step_supervision(clean_batch, device)
        rec_inputs, rec_target, rec_mask = select_recovery(
            recovery_inputs, recovery_target, recovery_mask, recovery_indices(step)
        )
        combined_inputs = concatenate_inputs(torch, clean_inputs, rec_inputs)
        combined_target = torch.cat((clean_target, rec_target), dim=0)
        combined_mask = torch.cat((clean_mask, rec_mask), dim=0)

        if not first_probe_done:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16
            ):
                probe_outputs = policy(**combined_inputs)
                probe_clean_output = {
                    key: value[:clean_batch_size]
                    for key, value in probe_outputs.items()
                    if torch.is_tensor(value)
                    and value.shape[:1]
                    == (clean_batch_size + recovery_batch_size,)
                }
                probe_recovery_output = {
                    key: value[clean_batch_size:]
                    for key, value in probe_outputs.items()
                    if torch.is_tensor(value)
                    and value.shape[:1]
                    == (clean_batch_size + recovery_batch_size,)
                }
                probe_clean_loss = waypoint_only_loss(
                    probe_clean_output, clean_target, clean_mask
                )
                probe_recovery_loss = waypoint_only_loss(
                    probe_recovery_output, rec_target, rec_mask
                )
                probe_loss = (
                    clean_weight * probe_clean_loss
                    + recovery_weight * probe_recovery_loss
                )
            probe_loss.backward()
            gradients = waypoint_gradient_report(policy)
            required = list(training.get(
                "waypoint_gradient_required",
                ["fusion", "gru", "da3_adapter"],
            ))
            missing = [
                name for name in required if gradients.get(name, 0.0) <= 0.0
            ]
            if missing:
                raise RuntimeError(
                    f"Phase 3 waypoint-only gradient is zero: {missing}"
                )
            if rank == 0:
                (output_dir / "waypoint_only_gradient_report.json").write_text(
                    json.dumps({
                        "status": "passed",
                        "loss_source": "waypoint_only_clean_plus_recovery",
                        "loss": float(probe_loss.detach().float().cpu()),
                        "clean_loss": float(
                            probe_clean_loss.detach().float().cpu()
                        ),
                        "recovery_loss": float(
                            probe_recovery_loss.detach().float().cpu()
                        ),
                        "gradient_norms": gradients,
                        "required_nonzero": required,
                        "auxiliary_or_distillation_loss_in_probe": False,
                        "test_locked_used": False,
                    }, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            optimizer.zero_grad(set_to_none=True)
            first_probe_done = True
            if world_size > 1:
                dist.barrier()

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
            outputs = wrapped(**combined_inputs)
            clean_output = {key: value[:clean_batch_size] for key, value in outputs.items() if torch.is_tensor(value) and value.shape[:1] == (clean_batch_size + recovery_batch_size,)}
            rec_output = {key: value[clean_batch_size:] for key, value in outputs.items() if torch.is_tensor(value) and value.shape[:1] == (clean_batch_size + recovery_batch_size,)}
            clean_waypoint_loss = waypoint_only_loss(
                clean_output, clean_target, clean_mask
            )
            if clean_loss_weights:
                clean_loss, _ = compute_architecture_v1_loss(
                    clean_output, clean_supervision, clean_loss_weights
                )
            else:
                clean_loss = clean_waypoint_loss
            recovery_loss = waypoint_only_loss(rec_output, rec_target, rec_mask)
            distillation_loss = recovery_loss * 0.0
            for name, teacher in recovery_teacher_outputs.items():
                output_name = distillation_outputs[name]
                selected_teacher = teacher[recovery_indices(step)].to(
                    dtype=rec_output[output_name].dtype
                )
                distillation_loss = distillation_loss + float(
                    recovery_distillation_weights[name]
                ) * F.smooth_l1_loss(
                    rec_output[output_name], selected_teacher
                )
            loss = (
                clean_weight * clean_loss
                + recovery_weight * recovery_loss
                + distillation_loss
            )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite Phase 3 loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainable], gradient_clip)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError(f"non-finite Phase 3 gradient at step {step}")
        optimizer.step()
        scheduler.step()
        global_step = step + 1
        with torch.no_grad():
            predicted_path = path_length(torch, rec_output["waypoints"].detach().float()).sum()
            target_path = path_length(torch, rec_target.detach().float()).sum().clamp_min(1.0e-8)
            ratio = predicted_path / target_path
        running["clean"] += float(clean_loss.detach())
        running["clean_waypoint"] += float(clean_waypoint_loss.detach())
        running["recovery"] += float(recovery_loss.detach())
        running["distillation"] += float(distillation_loss.detach())
        running["total"] += float(loss.detach())
        running["path_ratio"] += float(ratio.detach())
        running["gradient"] += float(gradient_norm.detach())
        running_steps += 1

        if global_step % log_every == 0:
            vector = torch.tensor([
                running["clean"], running["recovery"], running["total"],
                running["clean_waypoint"], running["distillation"],
                running["path_ratio"], running["gradient"], float(running_steps),
            ], dtype=torch.float64, device=device)
            if world_size > 1:
                dist.all_reduce(vector, op=dist.ReduceOp.SUM)
            denominator = max(1.0, float(vector[-1]))
            record = {
                "global_step": global_step,
                "loss_clean": float(vector[0] / denominator),
                "loss_recovery": float(vector[1] / denominator),
                "loss_total": float(vector[2] / denominator),
                "loss_clean_waypoint": float(vector[3] / denominator),
                "loss_recovery_distillation": float(vector[4] / denominator),
                "recovery_path_length_ratio": float(vector[5] / denominator),
                "gradient_norm_mean": float(vector[6] / denominator),
                "learning_rates": {group["name"]: float(group["lr"]) for group in optimizer.param_groups},
                "elapsed_s": time.time() - training_started,
            }
            if rank == 0:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                print(json.dumps(record, sort_keys=True), flush=True)
            running = {name: 0.0 for name in running}
            running_steps = 0

        if global_step % checkpoint_every == 0 or global_step == maximum_steps:
            if world_size > 1:
                dist.barrier()
            if rank == 0:
                checkpoint_dir = output_dir / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                save_checkpoint(checkpoint_dir / "last.ckpt", global_step)
                save_checkpoint(checkpoint_dir / f"step_{global_step:07d}.ckpt", global_step)
            if world_size > 1:
                dist.barrier()

    policy.eval()
    if rank == 0:
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
            final_output = policy(**recovery_inputs)
        final_summary = summarize_predictions(torch, final_output, recovery_target, recovery_mask)
        checkpoint_dir = output_dir / "checkpoints"
        shutil.copy2(checkpoint_dir / "last.ckpt", checkpoint_dir / "best.ckpt")
        summary = {
            "status": "training_complete",
            "stage": config["stage"],
            "global_step": maximum_steps,
            "recovery_samples": len(sample_paths),
            "recovery_training_set_metrics_before": initial_summary,
            "recovery_training_set_metrics": final_summary,
            "best_checkpoint": str(checkpoint_dir / "best.ckpt"),
            "best_checkpoint_sha256": sha256(checkpoint_dir / "best.ckpt"),
            "elapsed_s": time.time() - training_started,
            "test_locked_used": False,
        }
        (output_dir / "TRAINING_COMPLETE.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, sort_keys=True), flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
