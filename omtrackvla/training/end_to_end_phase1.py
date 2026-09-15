"""Formal Architecture-v1-isomorphic Phase 1 pretraining for NEXT-026."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from torch import distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset, default_collate

from omtrackvla.data.end_to_end_phase1 import (
    ArchitectureV1IdentityDataset,
    ArchitectureV1InternGeometryDataset,
    ArchitectureV1Phase1Dataset,
)
from omtrackvla.models.end_to_end import (
    ArchitectureV1Config,
    EndToEndFollowPolicy,
    load_official_da3_small_l11,
    parameter_inventory,
)
from omtrackvla.models.end_to_end_phase1 import (
    ArchitectureV1Phase1Model,
    compute_phase1_loss,
    load_frozen_osnet_teacher,
    phase1_gradient_report,
)
from omtrackvla.training.end_to_end_v1 import (
    MODEL_INPUT_KEYS,
    _config_sha256,
    _distributed,
    _model_inputs,
    _resolve,
    _save_checkpoint,
    _seed_everything,
    _selected_indices,
    _tensor_batch,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size-per-device", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-units-per-dataset", type=int)
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("phase") != 1
        or value.get("method") != "architecture_v1_end_to_end"
        or value.get("formal_training") is not True
        or value.get("test_locked_used") is not False
    ):
        raise ValueError("Architecture v1 Phase 1 requires a formal non-locked config")
    return value


def _phase1_model_inputs(
    batch: Mapping[str, object], device: torch.device
) -> dict[str, torch.Tensor]:
    result = _model_inputs(batch, device)
    result["transition_action"] = batch["transition_action"].to(
        device, non_blocking=True
    )
    result["teacher_bbox"] = batch["target_bbox"].to(device, non_blocking=True)
    result["teacher_valid"] = (
        batch["identity_label_valid"].to(device, non_blocking=True).bool()
        & batch["target_visible"].to(device, non_blocking=True).bool()
    )
    return result


def _checkpoint_payload(
    *,
    model: ArchitectureV1Phase1Model,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    next_batch_in_epoch: int,
    global_step: int,
    config: Mapping[str, object],
    config_sha256: str,
    da3_loading: Mapping[str, object],
    trained_policy_parameter_names: list[str],
    identity_teacher_report: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "phase": 1,
        "method": "architecture_v1_end_to_end",
        "stage": config["stage"],
        "epoch": int(epoch),
        "next_batch_in_epoch": int(next_batch_in_epoch),
        "global_step": int(global_step),
        # ``model`` is deliberately the deployment-policy namespace so Phase 2
        # can load it with exact key/shape accounting. The removable heads are
        # preserved separately for Phase 1 resume and diagnostics.
        "model": model.policy.state_dict(),
        "phase1_model": model.state_dict(),
        "training_only_heads": {
            key: value
            for key, value in model.state_dict().items()
            if not key.startswith("policy.")
        },
        "trained_policy_parameter_names": trained_policy_parameter_names,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "architecture": model.policy.config.to_dict(),
        "config_sha256": config_sha256,
        "da3_pretrained_loading": dict(da3_loading),
        "identity_teacher": dict(identity_teacher_report),
        "input_contract": {
            "raw_rgb": True,
            "initial_bbox_once": True,
            "perception_cache": False,
            "external_later_bbox": False,
            "gt_depth_or_target_pose_input": False,
            "policy_waypoint_supervision": False,
        },
        "test_locked_used": False,
    }


def _probe(
    model: ArchitectureV1Phase1Model,
    dataset: ArchitectureV1Phase1Dataset,
    device: torch.device,
    use_bfloat16: bool,
) -> tuple[dict[str, object], list[str]]:
    # Probe one admitted identity record and one independent Intern geometry
    # record so every required Phase 1 branch is exercised deterministically.
    batch = default_collate([dataset.identity[0], dataset.geometry[0]])
    tensor_batch = _tensor_batch(batch, device)
    model.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16
    ):
        outputs = model(**_phase1_model_inputs(tensor_batch, device))
        loss, losses = compute_phase1_loss(
            outputs,
            tensor_batch,
            dataset.loss_weights,
            visibility_negative_weight=dataset.visibility_negative_weight,
        )
    loss.backward()
    gradients = phase1_gradient_report(model)
    required = [
        "da3_adapter",
        "target_attention",
        "scene_attention",
        "motion_pair_head",
        "forward_dynamics",
        "inverse_dynamics",
    ]
    if model.identity_teacher is not None:
        required.append("identity_teacher_projector")
    missing = [name for name in required if gradients[name] <= 0.0]
    if missing:
        raise RuntimeError(f"Architecture v1 Phase 1 probe has zero gradients: {missing}")
    trained_policy_parameter_names = sorted(
        name.removeprefix("policy.")
        for name, parameter in model.named_parameters()
        if name.startswith("policy.")
        and parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and float(parameter.grad.detach().float().norm()) > 0.0
    )
    report = {
        "status": "passed",
        "loss": float(loss.detach().float().cpu()),
        "losses": {
            name: float(value.detach().float().cpu()) for name, value in losses.items()
        },
        "gradient_norms": gradients,
        "required_nonzero": list(required),
        "trained_policy_parameter_names": trained_policy_parameter_names,
        "test_locked_used": False,
    }
    model.zero_grad(set_to_none=True)
    return report, trained_policy_parameter_names


def main() -> int:
    args = arguments()
    args.config = args.config.expanduser().resolve(strict=True)
    args.output_dir = args.output_dir.expanduser().resolve(strict=False)
    config = _load_config(args.config)
    repository = args.config.parents[2]
    world_size, rank, local_rank, device = _distributed()
    seed = int(config["seed"])
    _seed_everything(seed, rank)
    torch.set_float32_matmul_precision("high")

    architecture = ArchitectureV1Config(**config.get("architecture", {}))
    architecture.validate()
    data = config["data"]
    training = config["training"]
    samples_per_epoch = int(training["samples_per_epoch"])
    maximum_units = (
        args.max_units_per_dataset
        if args.max_units_per_dataset is not None
        else data.get("max_units_per_dataset")
    )
    identity = ArchitectureV1IdentityDataset(
        manifest_path=_resolve(repository, data["manifest"]),
        split="train",
        roots={
            name: _resolve(repository, path)
            for name, path in data["roots"].items()
        },
        datasets=tuple(data["identity_datasets"]),
        sage3d_sidecar=_resolve(repository, data["sage3d_sidecar"]),
        config=architecture,
        max_units_per_dataset=maximum_units,
    )
    geometry = ArchitectureV1InternGeometryDataset(
        manifest_path=_resolve(repository, data["manifest"]),
        split="train",
        root=_resolve(repository, data["roots"]["intern_data_n1"]),
        config=architecture,
        maximum_gap=int(data["geometry_maximum_gap"]),
        samples_per_epoch=samples_per_epoch,
        seed=seed,
        max_units=maximum_units,
    )
    dataset = ArchitectureV1Phase1Dataset(
        identity,
        geometry,
        samples_per_epoch=samples_per_epoch,
        identity_fraction=float(data["identity_fraction"]),
        identity_sampling=str(data["identity_sampling"]),
        seed=seed,
    )
    dataset.loss_weights = {
        str(name): float(value)
        for name, value in training["loss_weights"].items()
    }
    dataset.visibility_negative_weight = float(training["visibility_negative_weight"])

    batch_size = int(
        args.batch_size_per_device or training["batch_size_per_device"]
    )
    if samples_per_epoch % (world_size * batch_size):
        raise ValueError("samples_per_epoch must divide evenly across DDP workers")
    epochs = int(training["epochs"])
    planned_steps = epochs * samples_per_epoch // (world_size * batch_size)
    maximum_steps = int(args.max_steps or planned_steps)
    if maximum_steps > planned_steps:
        raise ValueError("max_steps exceeds the configured Phase 1 budget")

    for extra in (config["da3_source"], config["da3_runtime"]):
        resolved = _resolve(repository, extra).resolve(strict=True)
        if str(resolved) not in sys.path:
            sys.path.insert(0, str(resolved))
    da3, da3_loading = load_official_da3_small_l11(
        _resolve(repository, config["da3_model"]), architecture
    )
    policy = EndToEndFollowPolicy(da3, architecture).to(device)
    identity_teacher_kind = str(training.get("identity_teacher", "off"))
    if identity_teacher_kind == "off":
        identity_teacher = None
        identity_teacher_report: dict[str, object] = {
            "kind": "off",
            "deployment_input": False,
        }
        identity_teacher_dim = 512
    elif identity_teacher_kind == "osnet":
        identity_teacher, identity_teacher_report = load_frozen_osnet_teacher(
            _resolve(repository, config["osnet_teacher_weights"]),
            _resolve(repository, config["osnet_teacher_code"]),
        )
        identity_teacher_dim = int(identity_teacher_report["output_dim"])
    else:
        raise ValueError("identity_teacher must be 'off' or 'osnet'")
    model = ArchitectureV1Phase1Model(
        policy,
        identity_teacher=identity_teacher,
        identity_teacher_dim=identity_teacher_dim,
    ).to(device)
    use_bfloat16 = bool(training.get("bfloat16", True)) and torch.cuda.is_bf16_supported()
    config_hash = _config_sha256(args.config)
    start_epoch = 0
    start_batch = 0
    global_step = 0
    trained_policy_parameter_names: list[str] = []
    resume_value: Mapping[str, object] | None = None
    if args.resume_from is not None:
        resume_path = args.resume_from.expanduser().resolve(strict=True)
        resume_value = torch.load(resume_path, map_location="cpu", weights_only=False)
        if (
            resume_value.get("phase") != 1
            or resume_value.get("method") != "architecture_v1_end_to_end"
            or resume_value.get("config_sha256") != config_hash
        ):
            raise ValueError("resume checkpoint does not match this Phase 1 run")
        model.load_state_dict(resume_value["phase1_model"], strict=True)
        start_epoch = int(resume_value["epoch"])
        start_batch = int(resume_value["next_batch_in_epoch"])
        global_step = int(resume_value["global_step"])
        trained_policy_parameter_names = list(
            resume_value.get("trained_policy_parameter_names", [])
        )

    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    adapters = [parameter for name, parameter in trainable if ".adapter." in name]
    others = [parameter for name, parameter in trainable if ".adapter." not in name]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": adapters,
                "lr": float(training["adapter_learning_rate"]),
                "name": "da3_adapter",
            },
            {
                "params": others,
                "lr": float(training["policy_learning_rate"]),
                "name": "phase1_policy_and_heads",
            },
        ],
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not trained_policy_parameter_names:
        probe, trained_policy_parameter_names = _probe(
            model, dataset, device, use_bfloat16
        )
        if rank == 0:
            (args.output_dir / "phase1_gradient_report.json").write_text(
                json.dumps(probe, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        shutil.copy2(args.config, args.output_dir / "config.yaml")
        heads_parameters = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if not name.startswith("policy.")
        )
        (args.output_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "task": "NEXT-026 Architecture-v1-isomorphic Phase 1 pretraining",
                    "stage": config["stage"],
                    "seed": seed,
                    "world_size": world_size,
                    "datasets": [
                        "intern_data_n1",
                        "sage3d_extracted",
                        "tpt_bench_clean_v2",
                    ],
                    "identity_records": len(identity),
                    "geometry_records_per_epoch": len(geometry),
                    "samples_per_epoch": samples_per_epoch,
                    "planned_optimizer_steps": planned_steps,
                    "maximum_optimizer_steps": maximum_steps,
                    "batch_size_per_device": batch_size,
                    "loss_weights": dataset.loss_weights,
                    "visibility_negative_weight": dataset.visibility_negative_weight,
                    "da3_pretrained_loading": da3_loading,
                    "parameter_inventory": parameter_inventory(policy),
                    "training_only_head_parameters": heads_parameters,
                    "trained_policy_parameter_names": trained_policy_parameter_names,
                    "identity_teacher": identity_teacher_report,
                    "policy_waypoint_supervision": False,
                    "perception_cache": False,
                    "test_locked_used": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    wrapped: nn.Module = model
    if world_size > 1:
        wrapped = DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True
        )
    if resume_value is not None:
        del resume_value
    workers = int(
        args.num_workers if args.num_workers is not None else training["num_workers"]
    )
    checkpoint_every = int(training["checkpoint_every_steps"])
    log_every = int(training["log_every_steps"])
    gradient_clip = float(training["gradient_clip_norm"])
    log_path = args.output_dir / "train_log.jsonl"
    running = {
        name: 0.0
        for name in (
            "bbox",
            "visibility",
            "identity_or_binding",
            "osnet_identity",
            "ego",
            "future_latent",
            "future_target_xy",
            "future_visibility",
            "world_action",
            "inverse",
            "total",
        )
    }
    running_batches = 0
    started = time.time()

    for epoch in range(start_epoch, epochs):
        if global_step >= maximum_steps:
            break
        selected = _selected_indices(
            len(dataset), samples_per_epoch, seed=seed, epoch=epoch
        )
        rank_indices = selected[rank::world_size]
        epoch_batch_start = start_batch if epoch == start_epoch else 0
        rank_indices = rank_indices[epoch_batch_start * batch_size :]
        loader = DataLoader(
            Subset(dataset, rank_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=workers > 0,
        )
        model.train()
        for local_batch, batch in enumerate(loader, start=epoch_batch_start):
            if global_step >= maximum_steps:
                break
            tensor_batch = _tensor_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16
            ):
                outputs = wrapped(**_phase1_model_inputs(tensor_batch, device))
                loss, losses = compute_phase1_loss(
                    outputs,
                    tensor_batch,
                    dataset.loss_weights,
                    visibility_negative_weight=dataset.visibility_negative_weight,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite Phase 1 loss at step {global_step}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in trainable], gradient_clip
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"non-finite Phase 1 gradient at step {global_step}"
                )
            optimizer.step()
            scheduler.step()
            global_step += 1
            running_batches += 1
            for name in running:
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
                count = max(1.0, float(vector[-2]))
                record = {
                    "epoch": epoch,
                    "next_batch_in_epoch": local_batch + 1,
                    "global_step": global_step,
                    "losses": {
                        name: float(vector[index] / count)
                        for index, name in enumerate(running)
                    },
                    "gradient_norm_mean": float(vector[-1] / world_size),
                    "learning_rates": {
                        group["name"]: float(group["lr"])
                        for group in optimizer.param_groups
                    },
                    "elapsed_s": time.time() - started,
                }
                if rank == 0:
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                    print(json.dumps(record, sort_keys=True), flush=True)
                running = {name: 0.0 for name in running}
                running_batches = 0

            if global_step % checkpoint_every == 0 or global_step >= maximum_steps:
                if world_size > 1:
                    dist.barrier()
                if rank == 0:
                    payload = _checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        next_batch_in_epoch=local_batch + 1,
                        global_step=global_step,
                        config=config,
                        config_sha256=config_hash,
                        da3_loading=da3_loading,
                        trained_policy_parameter_names=trained_policy_parameter_names,
                        identity_teacher_report=identity_teacher_report,
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
        start_batch = 0

    if rank == 0:
        last = args.output_dir / "checkpoints" / "last.ckpt"
        if not last.is_file():
            raise RuntimeError("Phase 1 ended without a checkpoint")
        shutil.copy2(last, args.output_dir / "checkpoints" / "best.ckpt")
        summary = {
            "status": "training_complete",
            "stage": config["stage"],
            "global_step": global_step,
            "maximum_optimizer_steps": maximum_steps,
            "best_checkpoint": str(args.output_dir / "checkpoints" / "best.ckpt"),
            "elapsed_s": time.time() - started,
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
