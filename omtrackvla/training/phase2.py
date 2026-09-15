"""Distributed Phase 2 waypoint training over frozen perception records."""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

from omtrackvla.data.phase2 import Sage3DPolicyDataset
from omtrackvla.models.phase2 import Phase2WaypointPolicy, compute_phase2_loss


def _load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict) or value.get("phase") != 2:
        raise ValueError("the Phase 2 trainer requires a phase: 2 mapping")
    return value


def _resolve(repository: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository / path


def _seed_everything(seed: int, rank: int) -> None:
    value = seed + rank
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _cache_path(
    repository: Path, data: dict[str, object], split: str, override: Path | None
) -> Path:
    if override is not None:
        return override
    caches = data.get("perception_caches")
    if not isinstance(caches, dict) or split not in caches:
        raise ValueError(f"Phase 2 perception cache is not configured for split={split}")
    return _resolve(repository, caches[split])


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    best_loss: float,
    config: dict[str, object],
) -> dict[str, object]:
    module = model.module if isinstance(model, DistributedDataParallel) else model
    return {
        "schema_version": 1,
        "phase": 2,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_loss": float(best_loss),
        "model": module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "model_config": config["model"],
        "policy_input": "frozen_rgb_person_perception_v1",
    }


def _selected_indices(
    dataset_length: int,
    samples_per_epoch: int,
    *,
    seed: int,
    epoch: int,
) -> list[int]:
    """Select the same deterministic epoch-dependent contiguous window as E2E."""

    if dataset_length <= 0 or samples_per_epoch <= 0:
        raise ValueError("dataset length and samples_per_epoch must be positive")
    if samples_per_epoch > dataset_length:
        raise ValueError(
            f"samples_per_epoch={samples_per_epoch} exceeds dataset={dataset_length}"
        )
    span = dataset_length - samples_per_epoch
    start = 0 if span == 0 else (seed * 1_000_003 + epoch * 97_409) % (span + 1)
    return list(range(start, start + samples_per_epoch))


def _learning_rate_scale(step: int, *, warmup_steps: int, maximum_steps: int) -> float:
    if warmup_steps < 0 or maximum_steps <= 0 or warmup_steps >= maximum_steps:
        raise ValueError("learning-rate schedule requires 0 <= warmup_steps < maximum_steps")
    if step < warmup_steps:
        return max(1.0e-3, float(step + 1) / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, maximum_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run(args, distributed_context) -> int:
    config = _load_config(args.config)
    repository = args.config.resolve().parents[2]
    world_size, rank, _, device = distributed_context()
    seed = int(config.get("seed", 20260909))
    _seed_everything(seed, rank)
    data = config["data"]
    training = config["training"]
    maximum_units = args.max_units_per_dataset
    if maximum_units is None:
        maximum_units = data.get("max_units_per_dataset")
    dataset = Sage3DPolicyDataset(
        _resolve(repository, data["manifest"]),
        split="train",
        root=_resolve(repository, data["root"]),
        sidecar_root=_resolve(repository, data["sidecar_root"]),
        policy_admission=_resolve(repository, data["policy_admission"]),
        perception_cache=_cache_path(repository, data, "train", args.perception_cache),
        history_size=int(data["history_size"]),
        image_size=int(data["image_size"]),
        max_units=maximum_units,
        allow_partial_cache=bool(args.allow_partial_cache),
    )
    samples_per_epoch = int(training.get("samples_per_epoch", len(dataset)))
    if samples_per_epoch <= 0:
        raise ValueError("Phase 2 samples_per_epoch must be positive")
    if samples_per_epoch > len(dataset):
        raise ValueError(
            f"Phase 2 samples_per_epoch={samples_per_epoch} exceeds dataset={len(dataset)}"
        )
    workers = int(args.num_workers if args.num_workers is not None else training["num_workers"])
    batch_size = int(args.batch_size_per_device or training["batch_size_per_device"])
    global_batch_size = world_size * batch_size
    if samples_per_epoch % global_batch_size:
        raise ValueError(
            "Phase 2 samples_per_epoch must divide evenly by the global batch size"
        )
    steps_per_epoch = samples_per_epoch // global_batch_size
    if steps_per_epoch <= 0:
        raise ValueError("Phase 2 training selection is smaller than one distributed batch")
    epochs = int(training["epochs"])
    maximum_steps = int(training.get("maximum_steps", epochs * steps_per_epoch))
    stop_after_steps = int(args.max_steps or maximum_steps)
    if maximum_steps <= 0 or maximum_steps > epochs * steps_per_epoch:
        raise ValueError("Phase 2 maximum_steps exceeds the configured epoch budget")
    if stop_after_steps <= 0 or stop_after_steps > maximum_steps:
        raise ValueError("Phase 2 --max-steps exceeds training.maximum_steps")

    model = Phase2WaypointPolicy(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    warmup_steps = int(training.get("warmup_steps", 0))
    schedule = lambda step: _learning_rate_scale(
        step, warmup_steps=warmup_steps, maximum_steps=maximum_steps
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        if checkpoint.get("phase") != 2:
            raise ValueError("Phase 2 can only resume a Phase 2 checkpoint")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" not in checkpoint:
            raise ValueError("Phase 2 resume checkpoint has no scheduler state")
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_loss = float(checkpoint["best_loss"])
    elif args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("phase") != 2:
            raise ValueError(
                "Phase 2 uses the separately frozen detector/ReID front end; "
                "--init-checkpoint must therefore be a Phase 2 waypoint checkpoint"
            )
        model.load_state_dict(checkpoint["model"], strict=False)

    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[device.index])
    use_bfloat16 = bool(training.get("bfloat16", True)) and torch.cuda.is_bf16_supported()
    metrics_path = args.output_dir / "train_metrics.json"
    history = []
    if args.resume_from and metrics_path.is_file():
        previous_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if isinstance(previous_metrics.get("history"), list):
            history = list(previous_metrics["history"])
    stop = False
    log_path = args.output_dir / "train_log.jsonl"
    started = time.time()
    running_loss = 0.0
    running_gradient_norm = 0.0
    running_batches = 0
    for epoch in range(start_epoch, epochs):
        indices = _selected_indices(
            len(dataset), samples_per_epoch, seed=seed, epoch=epoch
        )
        training_dataset = Subset(dataset, indices)
        sampler = DistributedSampler(
            training_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=bool(training.get("shuffle_samples", False)),
            seed=seed,
        )
        sampler.set_epoch(epoch)
        loader = DataLoader(
            training_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=workers > 0,
        )
        if len(loader) != steps_per_epoch:
            raise RuntimeError("Phase 2 epoch produced an unexpected optimizer-step count")
        model.train()
        loss_sum = 0.0
        batches = 0
        for batch in loader:
            tensor_batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
                outputs = model(
                    tensor_batch["visual_xy"],
                    tensor_batch["visual_confidence"],
                    tensor_batch["visual_valid"],
                    tensor_batch["uwb_xy"],
                    tensor_batch["uwb_quality"],
                    tensor_batch["uwb_valid"],
                    tensor_batch["uwb_age_s"],
                    tensor_batch["condition_index"],
                )
                loss = compute_phase2_loss(outputs, tensor_batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite Phase 2 loss at step {global_step}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"non-finite Phase 2 gradient at step {global_step}"
                )
            optimizer.step()
            scheduler.step()
            global_step += 1
            batches += 1
            loss_sum += float(loss.detach())
            running_loss += float(loss.detach())
            running_gradient_norm += float(gradient_norm.detach())
            running_batches += 1
            if global_step % int(training["log_every_steps"]) == 0:
                interval_values = torch.tensor(
                    (running_loss, running_gradient_norm, float(running_batches)),
                    dtype=torch.float64,
                    device=device,
                )
                if world_size > 1:
                    dist.all_reduce(interval_values, op=dist.ReduceOp.SUM)
                interval = max(1.0, float(interval_values[2]))
                if rank == 0:
                    record = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "loss_mean": float(interval_values[0]) / interval,
                        "gradient_norm_mean": float(interval_values[1]) / interval,
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "elapsed_s": time.time() - started,
                    }
                    args.output_dir.mkdir(parents=True, exist_ok=True)
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                    print(json.dumps(record, sort_keys=True), flush=True)
                running_loss = 0.0
                running_gradient_norm = 0.0
                running_batches = 0
            if global_step >= stop_after_steps:
                stop = True
                break

        vector = torch.tensor((loss_sum, float(batches)), dtype=torch.float64, device=device)
        if world_size > 1:
            dist.all_reduce(vector, op=dist.ReduceOp.SUM)
        epoch_loss = float(vector[0] / max(float(vector[1]), 1.0))
        history.append({"epoch": epoch, "global_step": global_step, "total": epoch_loss})
        if rank == 0:
            payload = _checkpoint_payload(
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                min(best_loss, epoch_loss),
                config,
            )
            _save_checkpoint(args.output_dir / "checkpoints" / "last.ckpt", payload)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                payload["best_loss"] = best_loss
                _save_checkpoint(args.output_dir / "checkpoints" / "best.ckpt", payload)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(
                json.dumps(
                    {
                        "history": history,
                        "best_loss": best_loss,
                        "global_step": global_step,
                        "budget": {
                            "epochs": epochs,
                            "samples_per_epoch": samples_per_epoch,
                            "global_batch_size": global_batch_size,
                            "maximum_steps": maximum_steps,
                        },
                        "selection": "epoch_dependent_contiguous_window_v1",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        if world_size > 1:
            dist.barrier()
        if stop:
            break
    if rank == 0:
        print(json.dumps({"phase": 2, "global_step": global_step, "best_loss": best_loss}))
    if world_size > 1:
        dist.destroy_process_group()
    return 0
