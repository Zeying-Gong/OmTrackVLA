"""Formal distributed Architecture v1 training for NEXT-026."""
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
from torch.nn.parallel import DistributedDataParallel
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
    parameter_inventory,
    waypoint_gradient_report,
    waypoint_only_loss,
)
from omtrackvla.models.end_to_end_phase1 import (
    ArchitectureV1Phase1Model,
    phase1_gradient_report,
)


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


class SequencePolicy(nn.Module):
    def __init__(self, policy: EndToEndFollowPolicy) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.policy.forward_sequence(**inputs)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument("--resume-from", type=Path)
    checkpoint.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--training-heads-checkpoint", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size-per-device", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-episodes", type=int)
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("phase") != 2
        or value.get("method") != "architecture_v1_end_to_end"
        or value.get("formal_training") is not True
    ):
        raise ValueError("NEXT-026 requires a formal Architecture v1 Phase 2 config")
    if value.get("test_locked_used") is not False:
        raise ValueError("NEXT-026 config must explicitly forbid test_locked")
    return value


def _distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("formal Architecture v1 training requires CUDA")
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return world_size, rank, local_rank, torch.device("cuda", local_rank)


def _seed_everything(seed: int, rank: int) -> None:
    value = seed + rank
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _resolve(repository: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository / path


def _config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected_indices(
    dataset_length: int,
    samples_per_epoch: int,
    *,
    seed: int,
    epoch: int,
) -> list[int]:
    if samples_per_epoch > dataset_length:
        raise ValueError(
            f"samples_per_epoch={samples_per_epoch} exceeds dataset={dataset_length}"
        )
    span = dataset_length - samples_per_epoch
    start = 0 if span == 0 else (seed * 1_000_003 + epoch * 97_409) % (span + 1)
    return list(range(start, start + samples_per_epoch))


def _initialization_report(
    model: EndToEndFollowPolicy, checkpoint_path: Path
) -> dict[str, object]:
    value = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if value.get("method") != "architecture_v1_end_to_end" or value.get(
        "phase"
    ) not in {1, 2}:
        raise ValueError("initialization must be an Architecture v1 checkpoint")
    state = value.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("initialization checkpoint has no model state")
    current = model.state_dict()
    compatible = {
        str(key): tensor
        for key, tensor in state.items()
        if key in current and tuple(tensor.shape) == tuple(current[key].shape)
    }
    model.load_state_dict(compatible, strict=False)
    matched_parameters = sum(current[key].numel() for key in compatible)
    total_parameters = sum(tensor.numel() for tensor in current.values())
    trained_names = {
        str(name) for name in value.get("trained_policy_parameter_names", [])
    }
    trained_compatible = sorted(trained_names.intersection(compatible))
    trained_parameters = sum(current[key].numel() for key in trained_compatible)
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_phase": value.get("phase"),
        "checkpoint_stage": value.get("stage"),
        "matched_tensors": len(compatible),
        "model_tensors": len(current),
        "matched_parameters": matched_parameters,
        "model_parameters": total_parameters,
        "parameter_coverage": matched_parameters / max(1, total_parameters),
        "missing_or_mismatched": sorted(set(current) - set(compatible)),
        "phase1_trained_tensors": len(trained_compatible),
        "phase1_trained_parameters": trained_parameters,
        "phase1_trained_parameter_names": trained_compatible,
    }


def _checkpoint_payload(
    *,
    policy: EndToEndFollowPolicy,
    training_model: nn.Module,
    dynamics_enabled: bool,
    training_only_initialization: Mapping[str, object],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    next_batch_in_epoch: int,
    global_step: int,
    best_training_loss: float,
    config: Mapping[str, object],
    config_sha256: str,
    da3_loading: Mapping[str, object],
    initialization: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "phase": 2,
        "method": "architecture_v1_end_to_end",
        "stage": config["stage"],
        "epoch": int(epoch),
        "next_batch_in_epoch": int(next_batch_in_epoch),
        "global_step": int(global_step),
        "best_training_loss": float(best_training_loss),
        "model": policy.state_dict(),
        "training_only_dynamics": bool(dynamics_enabled),
        "training_only_heads": {
            name: tensor
            for name, tensor in training_model.state_dict().items()
            if not name.startswith("policy.")
        },
        "training_only_initialization": dict(training_only_initialization),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "architecture": policy.config.to_dict(),
        "ablation": policy.ablation.to_dict(),
        "config_sha256": config_sha256,
        "da3_pretrained_loading": dict(da3_loading),
        "initialization": dict(initialization),
        "input_contract": {
            "raw_rgb": True,
            "initial_bbox_once": True,
            "uwb_kind": "simulated_uwb",
            "perception_cache": False,
            "external_later_bbox": False,
            "gt_depth_or_target_pose_input": False,
        },
        "test_locked_used": False,
    }


def _save_checkpoint(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _model_inputs(batch: Mapping[str, object], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].to(device, non_blocking=True)
        for key in MODEL_INPUT_KEYS
    }


def _training_model_inputs(
    batch: Mapping[str, object],
    device: torch.device,
    *,
    dynamics_enabled: bool,
) -> dict[str, torch.Tensor]:
    inputs = _model_inputs(batch, device)
    if dynamics_enabled:
        inputs["transition_action"] = batch["transition_action"].to(
            device, non_blocking=True
        )
    return inputs


def _load_training_only_heads(
    model: nn.Module, state: Mapping[str, torch.Tensor]
) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    invalid_missing = [name for name in missing if not name.startswith("policy.")]
    if invalid_missing or unexpected:
        raise ValueError(
            "training-only head checkpoint mismatch: "
            f"missing={invalid_missing}, unexpected={list(unexpected)}"
        )


def _initialize_training_only_heads(
    model: nn.Module, checkpoint_path: Path
) -> dict[str, object]:
    value = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    phase = value.get("phase")
    if value.get("method") != "architecture_v1_end_to_end" or phase not in (1, 2):
        raise ValueError(
            "training-only heads must come from Architecture v1 Phase 1 or Phase 2"
        )
    source_key = "phase1_model" if phase == 1 else "training_only_heads"
    source = value.get(source_key)
    if not isinstance(source, Mapping):
        raise ValueError(
            f"Architecture v1 Phase {phase} checkpoint has no complete "
            "training-head state"
        )
    expected = {
        name: tensor
        for name, tensor in model.state_dict().items()
        if not name.startswith("policy.")
    }
    compatible = {
        str(name): tensor
        for name, tensor in source.items()
        if name in expected and tuple(tensor.shape) == tuple(expected[name].shape)
    }
    if set(compatible) != set(expected):
        raise ValueError(
            f"Phase {phase} training-only head coverage is incomplete: "
            f"missing={sorted(set(expected) - set(compatible))}"
        )
    _load_training_only_heads(model, compatible)
    parameters = sum(tensor.numel() for tensor in expected.values())
    return {
        "kind": f"architecture_v1_phase{phase}_checkpoint",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_phase": phase,
        "checkpoint_stage": value.get("stage"),
        "matched_tensors": len(compatible),
        "model_tensors": len(expected),
        "matched_parameters": parameters,
        "model_parameters": parameters,
        "parameter_coverage": 1.0,
        "missing_or_mismatched": [],
    }


def _tensor_batch(batch: Mapping[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _waypoint_probe(
    policy: EndToEndFollowPolicy,
    batch: Mapping[str, object],
    device: torch.device,
    use_bfloat16: bool,
) -> dict[str, object]:
    policy.zero_grad(set_to_none=True)
    tensor_batch = _tensor_batch(batch, device)
    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16
    ):
        outputs = policy.forward_sequence(**_model_inputs(tensor_batch, device))
        loss = waypoint_only_loss(
            outputs,
            tensor_batch["target_waypoints"],
            tensor_batch["waypoint_mask"],
        )
    loss.backward()
    gradients = waypoint_gradient_report(policy)
    required = ["fusion"]
    if policy.ablation.temporal_fusion == "gru":
        required.append("gru")
    if policy.ablation.backbone_tuning == "adapter":
        required.append("da3_adapter")
    missing = [name for name in required if gradients[name] <= 0.0]
    policy.zero_grad(set_to_none=True)
    if missing:
        raise RuntimeError(f"formal waypoint-only probe has zero gradients: {missing}")
    return {
        "loss": float(loss.detach().float().cpu()),
        "gradient_norms": gradients,
        "required_nonzero": required,
        "status": "passed",
    }


def main() -> int:
    args = arguments()
    args.config = args.config.expanduser().resolve(strict=True)
    args.output_dir = args.output_dir.expanduser().resolve(strict=False)
    if args.resume_from is not None and args.training_heads_checkpoint is not None:
        raise ValueError("resume cannot also replace training-only heads")
    config = _load_config(args.config)
    repository = args.config.parents[2]
    world_size, rank, local_rank, device = _distributed()
    seed = int(config["seed"])
    _seed_everything(seed, rank)
    torch.set_float32_matmul_precision("high")

    architecture = ArchitectureV1Config(**config.get("architecture", {}))
    architecture.validate()
    ablation = ArchitectureV1Ablation(**config.get("ablation", {}))
    ablation.validate()
    data = config["data"]
    training = config["training"]
    loss_weights = {
        str(name): float(value)
        for name, value in training["loss_weights"].items()
    }
    world_action_variant = str(training.get("world_action_variant", "b"))
    if world_action_variant not in {"b", "latent_only"}:
        raise ValueError("world_action_variant must be 'b' or 'latent_only'")
    action_input_mode = str(training.get("action_input_mode", "correct"))
    if action_input_mode not in {"correct", "zero", "shuffled"}:
        raise ValueError("action_input_mode must be correct, zero, or shuffled")
    dynamics_enabled = any(
        loss_weights.get(name, 0.0) != 0.0
        for name in ("world_action", "inverse")
    )
    dataset = Sage3DEndToEndSequenceDataset(
        _resolve(repository, data["sequence_index"]),
        split="train",
        config=architecture,
        modes=tuple(data.get("condition_modes", CONDITION_MODES)),
        max_episodes=args.max_episodes,
    )
    batch_size = int(
        args.batch_size_per_device or training["batch_size_per_device"]
    )
    samples_per_epoch = int(training["samples_per_epoch"])
    if samples_per_epoch % (world_size * batch_size):
        raise ValueError(
            "samples_per_epoch must be divisible by world_size * batch_size"
        )
    epochs = int(training["epochs"])
    planned_steps = epochs * samples_per_epoch // (world_size * batch_size)
    maximum_steps = int(args.max_steps or planned_steps)
    if maximum_steps > planned_steps:
        raise ValueError("max_steps exceeds the configured formal training budget")

    da3_source = _resolve(repository, config["da3_source"]).resolve(strict=True)
    da3_runtime = _resolve(repository, config["da3_runtime"]).resolve(strict=True)
    for extra_path in (da3_source, da3_runtime):
        if str(extra_path) not in sys.path:
            sys.path.insert(0, str(extra_path))
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
    inventory = parameter_inventory(policy)
    gru_initialization = str(
        training.get("gru_initialization_after_checkpoint", "preserve")
    )
    if gru_initialization not in {"preserve", "near_identity"}:
        raise ValueError(
            "gru_initialization_after_checkpoint must be preserve or near_identity"
        )
    initialization: dict[str, object] = {
        "kind": "direct_phase2",
        "checkpoint": None,
        "old_resnet18_checkpoint_loaded": False,
    }
    start_epoch = 0
    start_batch = 0
    global_step = 0
    best_training_loss = float("inf")
    resume_value: Mapping[str, object] | None = None
    if args.resume_from is not None:
        if gru_initialization != "preserve":
            raise ValueError("resume cannot reinitialize GRU parameters")
        resume_path = args.resume_from.expanduser().resolve(strict=True)
        resume_value = torch.load(resume_path, map_location="cpu", weights_only=False)
        if (
            resume_value.get("phase") != 2
            or resume_value.get("method") != "architecture_v1_end_to_end"
            or resume_value.get("config_sha256") != _config_sha256(args.config)
        ):
            raise ValueError("resume checkpoint does not match this NEXT-026 run")
        policy.load_state_dict(resume_value["model"], strict=True)
        start_epoch = int(resume_value["epoch"])
        start_batch = int(resume_value["next_batch_in_epoch"])
        global_step = int(resume_value["global_step"])
        best_training_loss = float(resume_value["best_training_loss"])
        initialization = dict(resume_value.get("initialization", initialization))
    elif args.init_checkpoint is not None:
        initialization = _initialization_report(
            policy, args.init_checkpoint.expanduser().resolve(strict=True)
        )
        initialization["kind"] = "architecture_v1_checkpoint"
        initialization["old_resnet18_checkpoint_loaded"] = False
        if gru_initialization == "near_identity":
            policy.initialize_gru_near_identity()
            initialization["gru_reinitialized"] = "near_identity"
    elif gru_initialization != "preserve":
        raise ValueError("GRU reinitialization requires --init-checkpoint")

    training_model: nn.Module
    training_only_initialization: dict[str, object] = {
        "kind": "not_enabled",
        "checkpoint": None,
    }
    if dynamics_enabled:
        training_model = ArchitectureV1Phase1Model(
            policy, action_input_mode=action_input_mode
        )
        if resume_value is not None:
            head_state = resume_value.get("training_only_heads")
            if not isinstance(head_state, Mapping):
                raise ValueError("curriculum resume checkpoint has no training-only heads")
            _load_training_only_heads(training_model, head_state)
            training_only_initialization = dict(
                resume_value.get(
                    "training_only_initialization",
                    {"kind": "resume", "checkpoint": str(args.resume_from)},
                )
            )
        elif args.training_heads_checkpoint is not None:
            training_only_initialization = _initialize_training_only_heads(
                training_model,
                args.training_heads_checkpoint.expanduser().resolve(strict=True),
            )
        else:
            training_only_initialization = {
                "kind": "newly_initialized",
                "checkpoint": None,
            }
    else:
        training_model = SequencePolicy(policy)
    training_model = training_model.to(device)

    trainable = [
        (name, parameter)
        for name, parameter in training_model.named_parameters()
        if parameter.requires_grad
    ]
    adapter_parameters = [
        parameter for name, parameter in trainable if ".adapter." in name
    ]
    gru_learning_rate = training.get("gru_learning_rate")
    gru_parameters = [
        parameter
        for name, parameter in trainable
        if ".gru." in name and ".adapter." not in name
    ]
    policy_parameters = [
        parameter
        for name, parameter in trainable
        if ".adapter." not in name
        and (gru_learning_rate is None or ".gru." not in name)
    ]
    optimizer_groups = [
        {
            "params": adapter_parameters,
            "lr": float(training["adapter_learning_rate"]),
            "name": "da3_adapter",
        },
        {
            "params": policy_parameters,
            "lr": float(training["policy_learning_rate"]),
            "name": "new_policy",
        },
    ]
    if gru_learning_rate is not None:
        optimizer_groups.append(
            {
                "params": gru_parameters,
                "lr": float(gru_learning_rate),
                "name": "gru",
            }
        )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=float(training["weight_decay"]),
    )
    warmup = int(training["warmup_steps"])

    def schedule(step: int) -> float:
        if step < warmup:
            return max(1.0e-3, float(step + 1) / max(1, warmup))
        progress = (step - warmup) / max(1, maximum_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    if resume_value is not None:
        optimizer.load_state_dict(resume_value["optimizer"])
        scheduler.load_state_dict(resume_value["scheduler"])

    wrapped: nn.Module = training_model
    if world_size > 1:
        wrapped = DistributedDataParallel(
            training_model,
            device_ids=[local_rank],
            find_unused_parameters=True,
        )
    use_bfloat16 = bool(training.get("bfloat16", True)) and torch.cuda.is_bf16_supported()
    workers = int(
        args.num_workers if args.num_workers is not None else training["num_workers"]
    )
    config_hash = _config_sha256(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        shutil.copy2(args.config, args.output_dir / "config.yaml")
        (args.output_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "task": "NEXT-026 formal Architecture v1 Phase 2 training",
                    "stage": config["stage"],
                    "seed": seed,
                    "world_size": world_size,
                    "dataset_sequences": len(dataset),
                    "dataset_index": str(dataset.index_path),
                    "dataset_index_sha256": dataset.index_sha256,
                    "condition_modes": list(dataset.modes),
                    "samples_per_epoch": samples_per_epoch,
                    "epochs": epochs,
                    "planned_optimizer_steps": planned_steps,
                    "maximum_optimizer_steps": maximum_steps,
                    "batch_size_per_device": batch_size,
                    "sequence_steps": dataset.sequence_steps,
                    "policy_stride": dataset.policy_stride,
                    "ablation": ablation.to_dict(),
                    "loss_weights": loss_weights,
                    "world_action_variant": world_action_variant,
                    "action_input_mode": action_input_mode,
                    "da3_source": str(da3_source),
                    "da3_runtime": str(da3_runtime),
                    "da3_pretrained_loading": da3_loading,
                    "parameter_inventory": inventory,
                    "training_only_dynamics": dynamics_enabled,
                    "training_only_head_parameters": sum(
                        parameter.numel()
                        for name, parameter in training_model.named_parameters()
                        if not name.startswith("policy.")
                    ),
                    "training_only_heads_initialized": (
                        training_only_initialization["kind"]
                    ),
                    "training_only_initialization": training_only_initialization,
                    "initialization": initialization,
                    "simulated_uwb": True,
                    "perception_cache": False,
                    "test_locked_used": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if world_size > 1:
        dist.barrier()

    log_path = args.output_dir / "train_log.jsonl"
    checkpoint_every = int(training["checkpoint_every_steps"])
    log_every = int(training["log_every_steps"])
    gradient_clip = float(training["gradient_clip_norm"])
    first_probe_done = (args.output_dir / "waypoint_only_gradient_report.json").is_file()
    curriculum_probe_path = args.output_dir / "curriculum_gradient_report.json"
    curriculum_probe_done = curriculum_probe_path.is_file()
    running = {
        name: 0.0
        for name in dict.fromkeys(
            (
                *loss_weights,
                "future_latent",
                "future_target_xy",
                "future_visibility",
                "total",
            )
        )
    }
    running_batches = 0
    training_started = time.time()
    stop_training = global_step >= maximum_steps

    for epoch in range(start_epoch, epochs):
        if stop_training:
            break
        selected = _selected_indices(
            len(dataset), samples_per_epoch, seed=seed, epoch=epoch
        )
        rank_indices = selected[rank::world_size]
        epoch_batch_start = start_batch if epoch == start_epoch else 0
        sample_start = epoch_batch_start * batch_size
        rank_indices = rank_indices[sample_start:]
        loader = DataLoader(
            Subset(dataset, rank_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=workers > 0,
        )
        policy.train()
        for local_batch, batch in enumerate(loader, start=epoch_batch_start):
            if global_step >= maximum_steps:
                stop_training = True
                break
            if not first_probe_done:
                probe = _waypoint_probe(policy, batch, device, use_bfloat16)
                if rank == 0:
                    (args.output_dir / "waypoint_only_gradient_report.json").write_text(
                        json.dumps(probe, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                first_probe_done = True
                if world_size > 1:
                    dist.barrier()

            tensor_batch = _tensor_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16
            ):
                outputs = wrapped(
                    **_training_model_inputs(
                        tensor_batch, device, dynamics_enabled=dynamics_enabled
                    )
                )
                loss, losses = compute_architecture_v1_loss(
                    outputs,
                    tensor_batch,
                    loss_weights,
                    world_action_variant=world_action_variant,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite NEXT-026 loss at optimizer step {global_step}"
                )
            loss.backward()
            if dynamics_enabled and not curriculum_probe_done:
                if not isinstance(training_model, ArchitectureV1Phase1Model):
                    raise RuntimeError("dynamics curriculum wrapper is missing")
                gradients = waypoint_gradient_report(policy)
                gradients.update(phase1_gradient_report(training_model))
                required = [
                    "fusion",
                    "action_embedding",
                    "forward_dynamics",
                    "future_world_head",
                    "inverse_dynamics",
                ]
                if policy.ablation.temporal_fusion == "gru":
                    required.append("gru")
                if policy.ablation.backbone_tuning == "adapter":
                    required.append("da3_adapter")
                if world_action_variant == "b":
                    required.extend(
                        ("future_target_xy_head", "future_visibility_head")
                    )
                missing = [name for name in required if gradients[name] <= 0.0]
                if missing:
                    raise RuntimeError(
                        f"curriculum full-loss probe has zero gradients: {missing}"
                    )
                if rank == 0:
                    curriculum_probe_path.write_text(
                        json.dumps(
                            {
                                "status": "passed",
                                "loss": float(loss.detach().float().cpu()),
                                "losses": {
                                    name: float(value.detach().float().cpu())
                                    for name, value in losses.items()
                                },
                                "gradient_norms": gradients,
                                "required_nonzero": list(required),
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                curriculum_probe_done = True
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters := [parameter for _, parameter in trainable],
                gradient_clip,
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"non-finite NEXT-026 gradient at optimizer step {global_step}"
                )
            optimizer.step()
            scheduler.step()
            global_step += 1
            running_batches += 1
            for name in running:
                if name in losses:
                    running[name] += float(losses[name].detach())

            if global_step % log_every == 0:
                vector = torch.tensor(
                    [running[name] for name in running]
                    + [float(running_batches), float(gradient_norm.detach())],
                    dtype=torch.float64,
                    device=device,
                )
                if world_size > 1:
                    dist.all_reduce(vector, op=dist.ReduceOp.SUM)
                batches = max(1.0, float(vector[-2]))
                record = {
                    "epoch": epoch,
                    "next_batch_in_epoch": local_batch + 1,
                    "global_step": global_step,
                    "losses": {
                        name: float(vector[index] / batches)
                        for index, name in enumerate(running)
                    },
                    "gradient_norm_mean": float(vector[-1] / world_size),
                    "learning_rates": {
                        group["name"]: float(group["lr"])
                        for group in optimizer.param_groups
                    },
                    "elapsed_s": time.time() - training_started,
                }
                if rank == 0:
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                    print(json.dumps(record, sort_keys=True), flush=True)
                running = {name: 0.0 for name in running}
                running_batches = 0

            should_checkpoint = (
                global_step % checkpoint_every == 0
                or global_step >= maximum_steps
            )
            if should_checkpoint:
                if world_size > 1:
                    dist.barrier()
                current_loss = float(loss.detach())
                best_training_loss = min(best_training_loss, current_loss)
                if rank == 0:
                    payload = _checkpoint_payload(
                        policy=policy,
                        training_model=training_model,
                        dynamics_enabled=dynamics_enabled,
                        training_only_initialization=training_only_initialization,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        next_batch_in_epoch=local_batch + 1,
                        global_step=global_step,
                        best_training_loss=best_training_loss,
                        config=config,
                        config_sha256=config_hash,
                        da3_loading=da3_loading,
                        initialization=initialization,
                    )
                    _save_checkpoint(
                        args.output_dir / "checkpoints" / "last.ckpt", payload
                    )
                    _save_checkpoint(
                        args.output_dir
                        / "checkpoints"
                        / f"step_{global_step:07d}.ckpt",
                        payload,
                    )
                if world_size > 1:
                    dist.barrier()
            if global_step >= maximum_steps:
                stop_training = True
                break
        start_batch = 0

    if rank == 0:
        last_checkpoint = args.output_dir / "checkpoints" / "last.ckpt"
        if not last_checkpoint.is_file():
            raise RuntimeError("formal training ended without a checkpoint")
        shutil.copy2(last_checkpoint, args.output_dir / "checkpoints" / "best.ckpt")
        summary = {
            "status": "training_complete",
            "stage": config["stage"],
            "global_step": global_step,
            "maximum_optimizer_steps": maximum_steps,
            "best_checkpoint": str(args.output_dir / "checkpoints" / "best.ckpt"),
            "elapsed_s": time.time() - training_started,
            "test_locked_used": False,
        }
        (args.output_dir / "TRAINING_COMPLETE.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, sort_keys=True), flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
