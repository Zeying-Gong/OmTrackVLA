"""Distributed open-loop evaluation for Architecture v1 NEXT-026."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import distributed as dist
from torch.utils.data import DataLoader, Subset

from omtrackvla.data.end_to_end_training import (
    CONDITION_MODES,
    Sage3DEndToEndSequenceDataset,
)
from omtrackvla.models.end_to_end import (
    ArchitectureV1Ablation,
    ArchitectureV1Config,
    EndToEndFollowPolicy,
    compute_architecture_v1_loss,
    load_official_da3_small_l11,
)
from omtrackvla.models.end_to_end_phase1 import ArchitectureV1Phase1Model


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
ACCUMULATORS = (
    "sequences",
    "policy_steps",
    "finite_steps",
    "ade_sum",
    "fde_sum",
    "stop_correct",
    "stop_positive",
    "stop_probability_sum",
    "bbox_iou_sum",
    "bbox_count",
    "visibility_correct",
    "visibility_count",
    "binding_correct",
    "binding_count",
    "ego_translation_error_sum",
    "ego_yaw_error_sum",
    "predicted_path_length_sum",
    "target_path_length_sum",
    "path_length_ratio_sum",
    "path_length_ratio_count",
)
DYNAMICS_LOSSES = (
    "future_latent",
    "future_target_xy",
    "future_visibility",
    "world_action",
    "inverse",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "viz_val"), default="val")
    parser.add_argument("--samples-per-mode", type=int, default=1024)
    parser.add_argument("--batch-size-per-device", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def _distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("Architecture v1 evaluation requires CUDA")
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return world_size, rank, local_rank, torch.device("cuda", local_rank)


def _resolve(repository: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_indices(length: int, samples_per_mode: int) -> list[int]:
    selections = []
    for mode in range(len(CONDITION_MODES)):
        pool = np.arange(mode, length, len(CONDITION_MODES), dtype=np.int64)
        if samples_per_mode > len(pool):
            raise ValueError(
                f"samples_per_mode={samples_per_mode} exceeds mode pool={len(pool)}"
            )
        offsets = np.linspace(
            0, len(pool) - 1, samples_per_mode, dtype=np.int64
        )
        selections.append(pool[offsets].tolist())
    return [
        selections[mode][sample]
        for sample in range(samples_per_mode)
        for mode in range(len(CONDITION_MODES))
    ]


def _bbox_iou(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    upper_left = torch.maximum(predicted[..., :2], target[..., :2])
    lower_right = torch.minimum(predicted[..., 2:], target[..., 2:])
    intersection = (lower_right - upper_left).clamp_min(0.0).prod(dim=-1)
    predicted_area = (predicted[..., 2:] - predicted[..., :2]).clamp_min(0.0).prod(
        dim=-1
    )
    target_area = (target[..., 2:] - target[..., :2]).clamp_min(0.0).prod(dim=-1)
    return intersection / (predicted_area + target_area - intersection).clamp_min(
        1.0e-8
    )


def _path_length(waypoints: torch.Tensor) -> torch.Tensor:
    if waypoints.shape[-1] != 2 or waypoints.shape[-2] < 2:
        raise ValueError("waypoints must end in [horizon,2]")
    return torch.linalg.vector_norm(
        waypoints[..., 1:, :] - waypoints[..., :-1, :], dim=-1
    ).sum(dim=-1)


def _horizon_diagnostics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    waypoint_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if predicted.shape != target.shape or waypoint_mask.shape != predicted.shape[:-1]:
        raise ValueError("waypoint diagnostic shape mismatch")
    finite = torch.isfinite(predicted).all(dim=(-1, -2))
    valid = waypoint_mask.bool() & finite[..., None]
    error = torch.linalg.vector_norm(predicted - target, dim=-1)
    predicted_radius = torch.linalg.vector_norm(
        predicted - predicted[..., :1, :], dim=-1
    )
    target_radius = torch.linalg.vector_norm(target - target[..., :1, :], dim=-1)
    return error, predicted_radius, target_radius, valid


def _empty_stats(device: torch.device) -> torch.Tensor:
    return torch.zeros(
        len(CONDITION_MODES), len(ACCUMULATORS), dtype=torch.float64, device=device
    )


def main() -> int:
    args = arguments()
    config_path = args.config.expanduser().resolve(strict=True)
    checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)
    output_path = args.output.expanduser().resolve(strict=False)
    config: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (
        config.get("method") != "architecture_v1_end_to_end"
        or config.get("test_locked_used") is not False
        or args.split == "test_locked"
    ):
        raise ValueError("invalid or locked NEXT-026 evaluation request")
    repository = config_path.parents[2]
    world_size, rank, _, device = _distributed()
    architecture = ArchitectureV1Config(**config.get("architecture", {}))
    ablation = ArchitectureV1Ablation(**config.get("ablation", {}))
    ablation.validate()
    dataset = Sage3DEndToEndSequenceDataset(
        _resolve(repository, config["data"]["sequence_index"]),
        split=args.split,
        config=architecture,
        modes=CONDITION_MODES,
    )
    selected = _selected_indices(len(dataset), args.samples_per_mode)
    if len(selected) % (world_size * args.batch_size_per_device):
        raise ValueError("evaluation selection must divide evenly across ranks")
    rank_indices = selected[rank::world_size]
    loader = DataLoader(
        Subset(dataset, rank_indices),
        batch_size=args.batch_size_per_device,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    for field in ("da3_source", "da3_runtime"):
        path = _resolve(repository, config[field]).resolve(strict=True)
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    da3, da3_loading = load_official_da3_small_l11(
        _resolve(repository, config["da3_model"]),
        architecture,
        ablation,
        (
            _resolve(repository, config["dinov2_model"])
            if config.get("dinov2_model") is not None
            else None
        ),
    )
    policy = EndToEndFollowPolicy(da3, architecture, ablation).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("phase") not in {2, 3}
        or checkpoint.get("method") != "architecture_v1_end_to_end"
        or checkpoint.get("test_locked_used") is not False
    ):
        raise ValueError("checkpoint is not an admitted Architecture v1 Phase 2/3 model")
    policy.load_state_dict(checkpoint["model"], strict=True)
    dynamics_enabled = checkpoint.get("training_only_dynamics") is True
    dynamics_model: ArchitectureV1Phase1Model | None = None
    if dynamics_enabled:
        head_state = checkpoint.get("training_only_heads")
        if not isinstance(head_state, dict):
            raise ValueError("dynamics checkpoint has no training-only head state")
        dynamics_model = ArchitectureV1Phase1Model(
            policy,
            action_input_mode=str(
                config.get("training", {}).get("action_input_mode", "correct")
            ),
        ).to(device)
        missing, unexpected = dynamics_model.load_state_dict(head_state, strict=False)
        invalid_missing = [name for name in missing if not name.startswith("policy.")]
        if invalid_missing or unexpected:
            raise ValueError(
                "training-only head checkpoint mismatch: "
                f"missing={invalid_missing}, unexpected={list(unexpected)}"
            )
        dynamics_model.eval()
    else:
        policy.eval()
    use_bfloat16 = torch.cuda.is_bf16_supported()
    stats = _empty_stats(device)
    horizon_stats = torch.zeros(
        len(CONDITION_MODES), architecture.horizon, 4,
        dtype=torch.float64, device=device
    )
    dynamics_stats = torch.zeros(
        len(DYNAMICS_LOSSES) + 1, dtype=torch.float64, device=device
    )

    with torch.inference_mode():
        for batch in loader:
            tensor_batch = {
                key: value.to(device, non_blocking=True)
                if torch.is_tensor(value)
                else value
                for key, value in batch.items()
            }
            inputs = {key: tensor_batch[key] for key in MODEL_INPUT_KEYS}
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16
            ):
                if dynamics_model is None:
                    outputs = policy.forward_sequence(**inputs)
                else:
                    outputs = dynamics_model(
                        transition_action=tensor_batch["transition_action"],
                        **inputs,
                    )
                    _, dynamics_losses = compute_architecture_v1_loss(
                        outputs,
                        tensor_batch,
                        config["training"]["loss_weights"],
                        world_action_variant=str(
                            config["training"].get("world_action_variant", "b")
                        ),
                    )
                    for index, name in enumerate(DYNAMICS_LOSSES):
                        dynamics_stats[index] += float(
                            dynamics_losses[name].detach().float()
                        )
                    dynamics_stats[-1] += 1.0
            predicted = outputs["waypoints"].float()
            target = tensor_batch["target_waypoints"].float()
            point_error = torch.linalg.vector_norm(predicted - target, dim=-1)
            (
                horizon_error,
                predicted_radius,
                target_radius,
                horizon_valid,
            ) = _horizon_diagnostics(
                predicted, target, tensor_batch["waypoint_mask"]
            )
            ade = point_error[..., 1:].mean(dim=-1)
            fde = point_error[..., -1]
            finite = torch.isfinite(predicted).all(dim=(-1, -2))
            stop_probability = torch.sigmoid(outputs["stop_logit"].float().squeeze(-1))
            stop_target = tensor_batch["stop_target"]
            stop_correct = (stop_probability >= 0.5) == stop_target.bool()
            bbox_iou = _bbox_iou(
                outputs["bbox_pred"].float(), tensor_batch["target_bbox"]
            )
            bbox_valid = (
                tensor_batch["identity_label_valid"].bool()
                & tensor_batch["target_visible"].bool()
            )
            visibility_correct = (
                outputs["visibility_logit"].squeeze(-1) >= 0.0
            ) == tensor_batch["target_visible"].bool()
            visibility_valid = tensor_batch["identity_label_valid"].bool()
            binding_correct = (
                outputs["binding_logit"].squeeze(-1) >= 0.0
            ) == tensor_batch["binding_target"].bool()
            predicted_motion = outputs["xi_hat"].float()
            target_motion = tensor_batch["ego_motion_target"]
            translation_error = torch.linalg.vector_norm(
                predicted_motion[..., :2] - target_motion[..., :2], dim=-1
            )
            predicted_yaw = torch.atan2(
                predicted_motion[..., 2], predicted_motion[..., 3]
            )
            target_yaw = torch.atan2(target_motion[..., 2], target_motion[..., 3])
            yaw_error = torch.atan2(
                torch.sin(predicted_yaw - target_yaw),
                torch.cos(predicted_yaw - target_yaw),
            ).abs()
            predicted_path_length = _path_length(predicted)
            target_path_length = _path_length(target)
            path_length_valid = target_path_length > 1.0e-6
            path_length_ratio = predicted_path_length / target_path_length.clamp_min(
                1.0e-6
            )

            mode_indices = tensor_batch["mode_index"]
            sequence_steps = predicted.shape[1]
            for mode in range(len(CONDITION_MODES)):
                selected_sequences = mode_indices == mode
                selected_steps = selected_sequences[:, None].expand(-1, sequence_steps)
                row = stats[mode]
                row[0] += selected_sequences.sum()
                row[1] += selected_steps.sum()
                row[2] += finite[selected_steps].sum()
                row[3] += ade[selected_steps].sum()
                row[4] += fde[selected_steps].sum()
                row[5] += stop_correct[selected_steps].sum()
                row[6] += (
                    stop_target.bool()[selected_steps]
                    & (stop_probability[selected_steps] >= 0.5)
                ).sum()
                row[7] += stop_probability[selected_steps].sum()
                bbox_selection = selected_steps & bbox_valid
                row[8] += bbox_iou[bbox_selection].sum()
                row[9] += bbox_selection.sum()
                visibility_selection = selected_steps & visibility_valid
                row[10] += visibility_correct[visibility_selection].sum()
                row[11] += visibility_selection.sum()
                row[12] += binding_correct[selected_steps].sum()
                row[13] += selected_steps.sum()
                row[14] += translation_error[selected_steps].sum()
                row[15] += yaw_error[selected_steps].sum()
                path_selection = selected_steps & path_length_valid & finite
                row[16] += predicted_path_length[path_selection].sum()
                row[17] += target_path_length[path_selection].sum()
                row[18] += path_length_ratio[path_selection].sum()
                row[19] += path_selection.sum()
                horizon_selection = selected_steps[..., None] & horizon_valid
                horizon_stats[mode, :, 0] += (
                    horizon_error.masked_fill(~horizon_selection, 0.0)
                ).sum(dim=(0, 1))
                horizon_stats[mode, :, 1] += (
                    predicted_radius.masked_fill(~horizon_selection, 0.0)
                ).sum(dim=(0, 1))
                horizon_stats[mode, :, 2] += (
                    target_radius.masked_fill(~horizon_selection, 0.0)
                ).sum(dim=(0, 1))
                horizon_stats[mode, :, 3] += horizon_selection.sum(dim=(0, 1))

    if world_size > 1:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        dist.all_reduce(horizon_stats, op=dist.ReduceOp.SUM)
        dist.all_reduce(dynamics_stats, op=dist.ReduceOp.SUM)
    if rank == 0:
        per_mode: dict[str, object] = {}
        for mode_index, mode_name in enumerate(CONDITION_MODES):
            row = stats[mode_index].cpu()
            steps = max(1.0, float(row[1]))
            bbox_count = max(1.0, float(row[9]))
            visibility_count = max(1.0, float(row[11]))
            binding_count = max(1.0, float(row[13]))
            path_count = max(1.0, float(row[19]))
            horizon = horizon_stats[mode_index].cpu()
            horizon_count = horizon[:, 3].clamp_min(1.0)
            per_mode[mode_name] = {
                "sequences": int(row[0]),
                "policy_steps": int(row[1]),
                "finite_prediction_coverage": float(row[2] / steps),
                "waypoint_ade_m": float(row[3] / steps),
                "waypoint_fde_m": float(row[4] / steps),
                "stop_accuracy": float(row[5] / steps),
                "stop_positive_count": int(row[6]),
                "stop_probability_mean": float(row[7] / steps),
                "bbox_iou": float(row[8] / bbox_count),
                "bbox_count": int(row[9]),
                "visibility_accuracy": float(row[10] / visibility_count),
                "visibility_count": int(row[11]),
                "binding_accuracy": float(row[12] / binding_count),
                "ego_translation_error_m": float(row[14] / steps),
                "ego_yaw_error_rad": float(row[15] / steps),
                "predicted_path_length_mean_m": float(row[16] / path_count),
                "target_path_length_mean_m": float(row[17] / path_count),
                "path_length_ratio_mean": float(row[18] / path_count),
                "path_length_ratio_of_sums": float(
                    row[16] / max(1.0e-12, float(row[17]))
                ),
                "path_length_ratio_count": int(row[19]),
                "waypoint_error_by_horizon_m": (
                    horizon[:, 0] / horizon_count
                ).tolist(),
                "predicted_radius_by_horizon_m": (
                    horizon[:, 1] / horizon_count
                ).tolist(),
                "target_radius_by_horizon_m": (
                    horizon[:, 2] / horizon_count
                ).tolist(),
            }
        normal = stats[:3].sum(dim=0).cpu()
        normal_steps = max(1.0, float(normal[1]))
        normal_path_count = max(1.0, float(normal[19]))
        normal_horizon = horizon_stats[:3].sum(dim=0).cpu()
        normal_horizon_count = normal_horizon[:, 3].clamp_min(1.0)
        report = {
            "schema_version": 1,
            "task": "NEXT-026 Architecture v1 open-loop evaluation",
            "status": "complete",
            "stage": checkpoint["stage"],
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "checkpoint_global_step": int(checkpoint["global_step"]),
            "split": args.split,
            "samples_per_mode": args.samples_per_mode,
            "world_size": world_size,
            "normal_modes": {
                "policy_steps": int(normal[1]),
                "finite_prediction_coverage": float(normal[2] / normal_steps),
                "waypoint_ade_m": float(normal[3] / normal_steps),
                "waypoint_fde_m": float(normal[4] / normal_steps),
                "predicted_path_length_mean_m": float(
                    normal[16] / normal_path_count
                ),
                "target_path_length_mean_m": float(normal[17] / normal_path_count),
                "path_length_ratio_mean": float(normal[18] / normal_path_count),
                "path_length_ratio_of_sums": float(
                    normal[16] / max(1.0e-12, float(normal[17]))
                ),
                "path_length_ratio_count": int(normal[19]),
                "waypoint_error_by_horizon_m": (
                    normal_horizon[:, 0] / normal_horizon_count
                ).tolist(),
                "predicted_radius_by_horizon_m": (
                    normal_horizon[:, 1] / normal_horizon_count
                ).tolist(),
                "target_radius_by_horizon_m": (
                    normal_horizon[:, 2] / normal_horizon_count
                ).tolist(),
            },
            "modes": per_mode,
            "da3_pretrained_loading": da3_loading,
            "simulated_uwb": True,
            "perception_cache": False,
            "training_only_dynamics": (
                {
                    name: float(
                        dynamics_stats[index]
                        / max(1.0, float(dynamics_stats[-1]))
                    )
                    for index, name in enumerate(DYNAMICS_LOSSES)
                }
                if dynamics_enabled
                else False
            ),
            "test_locked_used": False,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, sort_keys=True), flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
