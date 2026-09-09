"""Distributed Phase 2 waypoint training over frozen perception records."""
from __future__ import annotations

import json
import random
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
        "model_config": config["model"],
        "policy_input": "frozen_rgb_person_perception_v1",
    }


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
    if samples_per_epoch < len(dataset):
        indices = np.linspace(0, len(dataset) - 1, samples_per_epoch, dtype=np.int64).tolist()
        training_dataset = Subset(dataset, indices)
    else:
        training_dataset = dataset
    sampler = DistributedSampler(
        training_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=bool(training.get("shuffle_samples", False)),
        seed=seed,
    )
    workers = int(args.num_workers if args.num_workers is not None else training["num_workers"])
    loader = DataLoader(
        training_dataset,
        batch_size=int(args.batch_size_per_device or training["batch_size_per_device"]),
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=workers > 0,
    )
    if not len(loader):
        raise ValueError("Phase 2 training selection is smaller than one distributed batch")

    model = Phase2WaypointPolicy(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        if checkpoint.get("phase") != 2:
            raise ValueError("Phase 2 can only resume a Phase 2 checkpoint")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
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
    epochs = int(training["epochs"])
    history = []
    stop = False
    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
            optimizer.step()
            global_step += 1
            batches += 1
            loss_sum += float(loss.detach())
            if rank == 0 and global_step % int(training["log_every_steps"]) == 0:
                print(json.dumps({"epoch": epoch, "global_step": global_step, "loss": float(loss)}))
            if args.max_steps is not None and global_step >= args.max_steps:
                stop = True
                break

        vector = torch.tensor((loss_sum, float(batches)), dtype=torch.float64, device=device)
        if world_size > 1:
            dist.all_reduce(vector, op=dist.ReduceOp.SUM)
        epoch_loss = float(vector[0] / max(float(vector[1]), 1.0))
        history.append({"epoch": epoch, "global_step": global_step, "total": epoch_loss})
        if rank == 0:
            payload = _checkpoint_payload(
                model, optimizer, epoch, global_step, min(best_loss, epoch_loss), config
            )
            _save_checkpoint(args.output_dir / "checkpoints" / "last.ckpt", payload)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                payload["best_loss"] = best_loss
                _save_checkpoint(args.output_dir / "checkpoints" / "best.ckpt", payload)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "train_metrics.json").write_text(
                json.dumps({"history": history, "best_loss": best_loss}, indent=2) + "\n",
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
