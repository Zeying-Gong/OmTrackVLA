"""Small-budget Architecture-v2 dual-head training and SAGE3D validation."""
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
import torch
import yaml
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

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


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--initial-checkpoint", type=Path)
    return parser.parse_args()


def _resolve(repository: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distributed() -> tuple[int, int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("Architecture-v2 training requires CUDA")
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    return world, rank, local_rank, torch.device("cuda", local_rank)


def _seed(seed: int, rank: int) -> None:
    seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _step_inputs(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
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
    result: dict[str, torch.Tensor] = {}
    for key in keys:
        value = batch[key].to(device, non_blocking=True)
        result[key] = value[:, 0] if key in sequence_keys else value
    return result


def _targets(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "waypoints": batch["target_waypoints"][:, 0].to(device, non_blocking=True),
        "waypoint_yaws": batch["target_waypoint_yaws"][:, 0].to(
            device, non_blocking=True
        ),
        "waypoint_mask": batch["waypoint_mask"][:, 0].to(device, non_blocking=True),
        "action": batch["direct_action_target"][:, 0].to(device, non_blocking=True),
        "stop": batch["stop_target"][:, 0].to(device, non_blocking=True),
        "bbox": batch["target_bbox"][:, 0].to(device, non_blocking=True),
        "visible": batch["target_visible"][:, 0].to(device, non_blocking=True),
        "identity_valid": batch["identity_label_valid"][:, 0].to(
            device, non_blocking=True
        ).bool(),
    }


def _losses(
    outputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mask = targets["waypoint_mask"].bool().clone()
    mask[:, 0] = False
    expanded = mask[..., None].expand_as(outputs["waypoints"])
    waypoint = F.smooth_l1_loss(
        outputs["waypoints"], targets["waypoints"], reduction="none"
    )[expanded].mean()
    segment_mask = mask[:, 1:] & targets["waypoint_mask"][:, :-1].bool()
    segment_expanded = segment_mask[..., None].expand_as(outputs["waypoint_deltas"])
    target_delta = targets["waypoints"][:, 1:] - targets["waypoints"][:, :-1]
    waypoint_delta = F.smooth_l1_loss(
        outputs["waypoint_deltas"], target_delta, reduction="none"
    )[segment_expanded].mean()
    waypoint_terminal = F.smooth_l1_loss(
        outputs["waypoints"][:, -1], targets["waypoints"][:, -1]
    )
    action = F.smooth_l1_loss(
        outputs["direct_action_m_s_rad_s"], targets["action"]
    )
    stop = F.binary_cross_entropy_with_logits(
        outputs["stop_logit"].squeeze(-1), targets["stop"]
    )
    zero = outputs["waypoints"].sum() * 0.0
    waypoint_yaw = zero
    waypoint_yaw_delta = zero
    if "waypoint_yaws" in outputs:
        yaw_error = outputs["waypoint_yaws"] - targets["waypoint_yaws"]
        yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))
        waypoint_yaw = F.smooth_l1_loss(
            yaw_error[mask], torch.zeros_like(yaw_error[mask])
        )
        target_yaw_delta = (
            targets["waypoint_yaws"][:, 1:] - targets["waypoint_yaws"][:, :-1]
        )
        target_yaw_delta = torch.atan2(
            torch.sin(target_yaw_delta), torch.cos(target_yaw_delta)
        )
        yaw_delta_error = outputs["waypoint_yaw_deltas"] - target_yaw_delta
        yaw_delta_error = torch.atan2(
            torch.sin(yaw_delta_error), torch.cos(yaw_delta_error)
        )
        waypoint_yaw_delta = F.smooth_l1_loss(
            yaw_delta_error[segment_mask],
            torch.zeros_like(yaw_delta_error[segment_mask]),
        )
    bbox = zero
    visibility = zero
    if "bbox_pred" in outputs and "visibility_logit" in outputs:
        identity_valid = targets["identity_valid"]
        visible = identity_valid & targets["visible"].bool()
        if visible.any():
            bbox = F.smooth_l1_loss(
                outputs["bbox_pred"][visible], targets["bbox"][visible]
            )
        if identity_valid.any():
            visibility = F.binary_cross_entropy_with_logits(
                outputs["visibility_logit"].squeeze(-1)[identity_valid],
                targets["visible"][identity_valid],
            )
    values = {
        "waypoint": waypoint,
        "waypoint_delta": waypoint_delta,
        "waypoint_terminal": waypoint_terminal,
        "waypoint_yaw": waypoint_yaw,
        "waypoint_yaw_delta": waypoint_yaw_delta,
        "action": action,
        "stop": stop,
        "bbox": bbox,
        "visibility": visibility,
    }
    total = sum(float(weights.get(name, 0.0)) * value for name, value in values.items())
    values["total"] = total
    return total, values


def _atomic_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def main() -> int:
    args = arguments()
    config_path = args.config.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (
        config.get("schema_version") != 1
        or config.get("method") != "architecture_v2_dual_head"
        or config.get("test_locked_used") is not False
    ):
        raise ValueError("expected a non-locked Architecture-v2 dual-head config")
    repository = config_path.parents[2]
    world, rank, local_rank, device = _distributed()
    _seed(int(config["seed"]), rank)
    torch.set_float32_matmul_precision("high")

    architecture = ArchitectureV1Config(**config["architecture"])
    decoder = ArchitectureV2DecoderConfig(**config.get("decoder", {}))
    if architecture.history_size != 8 or decoder.history_stride_raw != 3:
        raise ValueError("Architecture-v2 contract requires 8 frames at raw stride 3")
    data_config = config["data"]
    balance_index = data_config.get("visibility_balance_index")
    balance_path = (
        _resolve(repository, balance_index) if balance_index is not None else None
    )
    dataset = Sage3DEndToEndSequenceDataset(
        _resolve(repository, data_config["sequence_index"]),
        split="train",
        config=architecture,
        modes=tuple(data_config["condition_modes"]),
        history_stride_raw=decoder.history_stride_raw,
        visibility_balance_index=balance_path,
        visibility_negative_repeats=int(
            data_config.get("visibility_negative_repeats", 0)
        ),
        small_bbox_repeats=int(data_config.get("small_bbox_repeats", 0)),
    )

    for field in ("da3_source", "da3_runtime"):
        path = _resolve(repository, config[field]).resolve(strict=True)
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    ablation = ArchitectureV1Ablation(backbone_tuning="adapter")
    da3, da3_loading = load_official_da3_small_l11(
        _resolve(repository, config["da3_model"]), architecture, ablation
    )
    policy = ArchitectureV2FollowPolicy(da3, architecture, decoder).to(device)
    initial_loading = None
    if args.initial_checkpoint is not None:
        initial_checkpoint = args.initial_checkpoint.expanduser().resolve(strict=True)
        payload = torch.load(initial_checkpoint, map_location="cpu", weights_only=False)
        if (
            payload.get("method") != "architecture_v2_dual_head"
            or payload.get("test_locked_used") is not False
        ):
            raise ValueError("initial checkpoint method/provenance mismatch")
        target_state = policy.state_dict()
        source_state = payload["model"]
        unexpected_source = [
            name for name in source_state if name not in target_state
        ]
        shape_mismatched = [
            name for name, value in source_state.items()
            if name in target_state and value.shape != target_state[name].shape
        ]
        compatible_state = {
            name: value for name, value in source_state.items()
            if name in target_state and value.shape == target_state[name].shape
        }
        incompatible = policy.load_state_dict(compatible_state, strict=False)
        allowed_missing = (
            "trajectory.yaw_delta_head.",
            "trajectory.bbox_head.",
            "trajectory.visibility_head.",
        )
        disallowed_missing = [
            name for name in incompatible.missing_keys
            if not name.startswith(allowed_missing)
        ]
        disallowed_shape_mismatch = [
            name for name in shape_mismatched
            if not name.startswith(allowed_missing)
        ]
        if (
            unexpected_source
            or incompatible.unexpected_keys
            or disallowed_missing
            or disallowed_shape_mismatch
        ):
            raise RuntimeError(
                "unsafe Architecture-v2 warm start: "
                f"missing={disallowed_missing}, unexpected={unexpected_source}, "
                f"shape_mismatch={disallowed_shape_mismatch}"
            )
        initial_loading = {
            "checkpoint": str(initial_checkpoint),
            "checkpoint_sha256": _sha256(initial_checkpoint),
            "checkpoint_step": payload.get("global_step"),
            "missing_new_parameters": list(incompatible.missing_keys),
            "shape_mismatched_new_parameters": shape_mismatched,
            "unexpected_parameters": unexpected_source,
        }
    if bool(config["training"].get("freeze_da3", True)):
        for parameter in policy.da3.parameters():
            parameter.requires_grad_(False)
    inventory = parameter_inventory_v2(policy)
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable Architecture-v2 parameters")
    wrapped: nn.Module = policy
    if world > 1:
        wrapped = DistributedDataParallel(
            policy, device_ids=[local_rank], find_unused_parameters=True
        )

    training = config["training"]
    batch_size = int(training["batch_size_per_device"])
    maximum_steps = int(args.max_steps or training["max_steps"])
    sampler = DistributedSampler(dataset, world, rank, shuffle=True, seed=int(config["seed"]))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=int(training["num_workers"]),
        pin_memory=True,
        drop_last=True,
        persistent_workers=int(training["num_workers"]) > 0,
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    warmup = int(training["warmup_steps"])

    def schedule(step: int) -> float:
        if step < warmup:
            return max(1.0e-3, (step + 1) / max(1, warmup))
        progress = (step - warmup) / max(1, maximum_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    use_bfloat16 = bool(training.get("bfloat16", True)) and torch.cuda.is_bf16_supported()
    loss_weights = {str(k): float(v) for k, v in training["loss_weights"].items()}
    config_hash = _sha256(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        shutil.copy2(config_path, output_dir / "config.yaml")
        (output_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "method": "architecture_v2_dual_head",
                    "stage": config["stage"],
                    "world_size": world,
                    "physical_gpu_zero_forbidden": True,
                    "dataset_sequences": len(dataset),
                    "dataset_base_sequences": dataset.base_length,
                    "visibility_negative_anchors": len(
                        dataset.visibility_negative_indices
                    ),
                    "visibility_negative_repeats": (
                        dataset.visibility_negative_repeats
                    ),
                    "small_bbox_anchors": len(dataset.small_bbox_indices),
                    "small_bbox_repeats": dataset.small_bbox_repeats,
                    "sequence_index": str(dataset.index_path),
                    "history_size": 8,
                    "history_stride_raw": 3,
                    "action_units": ["m/s", "m/s", "rad/s"],
                    "same_checkpoint_for_all_execution_modes": True,
                    "da3_pretrained_loading": da3_loading,
                    "parameter_inventory": inventory,
                    "loss_weights": loss_weights,
                    "initial_checkpoint_loading": initial_loading,
                    "maximum_optimizer_steps": maximum_steps,
                    "test_locked_used": False,
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
    if world > 1:
        dist.barrier()

    log_path = output_dir / "train_log.jsonl"
    step = 0
    epoch = 0
    start = time.time()
    first_probe = True
    while step < maximum_steps:
        sampler.set_epoch(epoch)
        policy.train()
        policy.da3.eval()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            inputs = _step_inputs(batch, device)
            targets = _targets(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
                outputs = wrapped(**inputs)
                loss, values = _losses(outputs, targets, loss_weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            loss.backward()
            if first_probe:
                gradients = waypoint_gradient_report_v2(policy)
                required = (
                    "l11_projector", "scene_resampler", "context_encoder",
                    "trajectory_decoder", "delta_head", "uwb_encoder", "motion_pair_head",
                )
                if not bool(training.get("freeze_da3", True)):
                    required = required + ("da3_adapter",)
                missing = [name for name in required if gradients[name] <= 0.0]
                if missing:
                    raise RuntimeError(f"v2 first backward has zero gradients: {missing}")
                action_head_gradient = float(torch.stack([
                    p.grad.detach().float().square().sum()
                    for p in policy.trajectory.direct_action_head.parameters()
                    if p.grad is not None
                ]).sum().sqrt().cpu())
                if action_head_gradient <= 0.0:
                    raise RuntimeError("direct action loss did not reach direct action head")
                new_head_gradients = {}
                for name in ("yaw_delta_head", "bbox_head", "visibility_head"):
                    module = getattr(policy.trajectory, name, None)
                    if module is None:
                        continue
                    squares = [
                        parameter.grad.detach().float().square().sum()
                        for parameter in module.parameters()
                        if parameter.grad is not None
                    ]
                    norm = float(torch.stack(squares).sum().sqrt().cpu()) if squares else 0.0
                    if norm <= 0.0:
                        raise RuntimeError(f"{name} loss did not reach its deployment head")
                    new_head_gradients[name] = norm
                if rank == 0:
                    (output_dir / "first_backward_report.json").write_text(
                        json.dumps({
                            "status": "passed",
                            "gradient_norms": gradients,
                            "direct_action_head": action_head_gradient,
                            "new_deployment_heads": new_head_gradients,
                            "da3_frozen": bool(training.get("freeze_da3", True)),
                        }, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                first_probe = False
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable, float(training["gradient_clip_norm"])
            )
            optimizer.step()
            scheduler.step()
            step += 1
            if rank == 0 and (step == 1 or step % int(training["log_every_steps"]) == 0):
                record = {
                    "global_step": step,
                    "losses": {name: float(value.detach().float().cpu()) for name, value in values.items()},
                    "gradient_norm": float(gradient_norm.detach().float().cpu()),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "elapsed_s": time.time() - start,
                }
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                print(json.dumps(record, sort_keys=True), flush=True)
            if step % int(training["checkpoint_every_steps"]) == 0 or step >= maximum_steps:
                if world > 1:
                    dist.barrier()
                if rank == 0:
                    payload = {
                        "schema_version": 1,
                        "phase": 2,
                        "method": "architecture_v2_dual_head",
                        "stage": config["stage"],
                        "global_step": step,
                        "model": policy.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "config_sha256": config_hash,
                        "test_locked_used": False,
                    }
                    _atomic_save(output_dir / "checkpoints" / "last.ckpt", payload)
                    _atomic_save(output_dir / "checkpoints" / f"step_{step:07d}.ckpt", payload)
                if world > 1:
                    dist.barrier()
            if step >= maximum_steps:
                break
        epoch += 1

    # Measure held-out SAGE3D geometry using the exact checkpointed model.
    val_dataset = Sage3DEndToEndSequenceDataset(
        _resolve(repository, data_config["sequence_index"]),
        split="viz_val",
        config=architecture,
        modes=("visual_only",),
        history_stride_raw=decoder.history_stride_raw,
        visibility_balance_index=balance_path,
        visibility_negative_repeats=1 if balance_path is not None else 0,
        small_bbox_repeats=1 if balance_path is not None else 0,
    )
    val_count = min(int(training.get("validation_samples", 256)), len(val_dataset))
    if val_dataset.visibility_negative_indices:
        negative_count = min(val_count // 2, len(val_dataset.visibility_negative_indices))
        negative_indices = [
            val_dataset.base_length + index for index in range(negative_count)
        ]
        positive_budget = val_count - negative_count
        small_bbox_count = min(
            positive_budget // 2, len(val_dataset.small_bbox_indices)
        )
        small_bbox_offset = val_dataset.base_length + len(
            val_dataset.visibility_negative_indices
        ) * val_dataset.visibility_negative_repeats
        small_bbox_indices = [
            small_bbox_offset + index for index in range(small_bbox_count)
        ]
        negative_base = set(val_dataset.visibility_negative_indices)
        positive_indices = []
        for index in range(val_dataset.base_length):
            if index not in negative_base:
                positive_indices.append(index)
                if len(positive_indices) >= positive_budget - small_bbox_count:
                    break
        validation_indices = positive_indices + small_bbox_indices + negative_indices
    else:
        validation_indices = list(range(val_count))
    val_subset = Subset(val_dataset, validation_indices)
    val_sampler = DistributedSampler(val_subset, world, rank, shuffle=False)
    val_loader = DataLoader(val_subset, batch_size=batch_size, sampler=val_sampler, num_workers=0)
    # ADE sum, FDE sum, direct-action MAE sum, samples, yaw absolute-error
    # sum/count, visibility correct/count, bbox absolute-error sum/count.
    sums = torch.zeros(10, dtype=torch.float64, device=device)
    policy.eval()
    with torch.inference_mode():
        for batch in val_loader:
            targets = _targets(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
                outputs = policy(**_step_inputs(batch, device))
            error = torch.linalg.vector_norm(outputs["waypoints"] - targets["waypoints"], dim=-1)
            action_error = (outputs["direct_action_m_s_rad_s"] - targets["action"]).abs().mean(dim=-1)
            yaw_sum = yaw_count = visibility_correct = visibility_count = 0.0
            bbox_sum = bbox_count = 0.0
            if "waypoint_yaws" in outputs:
                yaw_error = outputs["waypoint_yaws"][:, 1:] - targets["waypoint_yaws"][:, 1:]
                yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error)).abs()
                yaw_sum = float(yaw_error.sum())
                yaw_count = float(yaw_error.numel())
            if "visibility_logit" in outputs:
                valid = targets["identity_valid"]
                if valid.any():
                    prediction = outputs["visibility_logit"].squeeze(-1) >= 0.0
                    visibility_correct = float(
                        (prediction[valid] == targets["visible"][valid].bool()).sum()
                    )
                    visibility_count = float(valid.sum())
                    visible = valid & targets["visible"].bool()
                    if visible.any():
                        bbox_error = (
                            outputs["bbox_pred"][visible] - targets["bbox"][visible]
                        ).abs()
                        bbox_sum = float(bbox_error.sum())
                        bbox_count = float(bbox_error.numel())
            sums += torch.tensor([
                float(error[:, 1:].sum()), float(error[:, -1].sum()),
                float(action_error.sum()), float(error.shape[0]),
                yaw_sum, yaw_count, visibility_correct, visibility_count,
                bbox_sum, bbox_count,
            ], dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(sums)
    if rank == 0:
        effective = max(1.0, float(sums[3]))
        validation = {
            "split": "viz_val",
            "samples": int(sums[3]),
            "ade_m": float(sums[0] / (effective * 7.0)),
            "fde_m": float(sums[1] / effective),
            "direct_action_mae_physical": float(sums[2] / effective),
            "waypoint_yaw_mae_deg": float(
                sums[4] / max(1.0, float(sums[5])) * 180.0 / math.pi
            ),
            "visibility_accuracy": float(
                sums[6] / max(1.0, float(sums[7]))
            ),
            "bbox_mae_normalized": float(
                sums[8] / max(1.0, float(sums[9]))
            ),
            "test_locked_used": False,
        }
        (output_dir / "SAGE3D_METRICS.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        last = output_dir / "checkpoints" / "last.ckpt"
        shutil.copy2(last, output_dir / "checkpoints" / "best.ckpt")
        complete = {
            "status": "training_complete",
            "global_step": step,
            "checkpoint": str(output_dir / "checkpoints" / "best.ckpt"),
            "sage3d": validation,
            "elapsed_s": time.time() - start,
            "test_locked_used": False,
        }
        (output_dir / "TRAINING_COMPLETE.json").write_text(
            json.dumps(complete, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(complete, sort_keys=True), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
