"""Distributed trainers for OmTrackVLA phases."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from omtrackvla.data.phase1 import ContractIdentityDataset, InternGeometryDataset, Phase1MultiTaskDataset
from omtrackvla.models.phase1 import Phase1WorldIdentityModel, compute_phase1_loss


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument("--resume-from", type=Path)
    checkpoint.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-units-per-dataset", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--batch-size-per-device", type=int)
    parser.add_argument("--perception-cache", type=Path)
    parser.add_argument("--allow-partial-cache", action="store_true")
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict) or value.get("phase") != 1:
        raise ValueError("the Phase 1 trainer requires a phase: 1 mapping")
    return value


def _distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("OmTrackVLA training requires CUDA")
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
        "phase": 1,
        "epoch": epoch,
        "global_step": global_step,
        "best_loss": best_loss,
        "model": module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": config["model"],
    }


def _save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> int:
    args = _arguments()
    if args.phase == 2:
        from omtrackvla.training.phase2 import run

        if not args.run_manifest.is_file():
            raise FileNotFoundError(f"run manifest not found: {args.run_manifest}")
        return run(args, _distributed)
    if args.phase != 1:
        raise ValueError("this entry point currently implements Phase 1 and Phase 2 only")
    if not args.run_manifest.is_file():
        raise FileNotFoundError(f"run manifest not found: {args.run_manifest}")
    config = _load_config(args.config)
    repository = args.config.resolve().parents[2]
    world_size, rank, _, device = _distributed()
    seed = int(config.get("seed", 20260907))
    _seed_everything(seed, rank)

    data_config = config["data"]
    training = config["training"]
    manifest_path = _resolve(repository, data_config["manifest"])
    roots = {key: Path(value) for key, value in data_config["roots"].items()}
    max_units = args.max_units_per_dataset
    if max_units is None:
        max_units = data_config.get("max_units_per_dataset")
    samples_per_epoch = int(training["samples_per_epoch"])
    identity = ContractIdentityDataset(
        manifest_path,
        split="train",
        roots=roots,
        datasets=tuple(
            data_config.get(
                "identity_datasets", ("sage3d_extracted", "tpt_bench_clean_v2")
            )
        ),
        sage3d_sidecar=data_config.get("sage3d_sidecar"),
        image_size=int(data_config["image_size"]),
        history_size=int(data_config["history_size"]),
        max_units_per_dataset=max_units,
    )
    geometry = InternGeometryDataset(
        manifest_path,
        split="train",
        root=roots["intern_data_n1"],
        image_size=int(data_config["image_size"]),
        maximum_gap=int(data_config["geometry_maximum_gap"]),
        history_size=int(data_config["history_size"]),
        samples_per_epoch=samples_per_epoch,
        seed=seed,
        max_units=max_units,
    )
    dataset = Phase1MultiTaskDataset(
        identity,
        geometry,
        samples_per_epoch=samples_per_epoch,
        identity_fraction=float(data_config["identity_fraction"]),
        identity_sampling=str(data_config.get("identity_sampling", "proportional")),
        seed=seed,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)
    workers = int(args.num_workers if args.num_workers is not None else training["num_workers"])
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size_per_device or training["batch_size_per_device"]),
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=workers > 0,
    )

    model = Phase1WorldIdentityModel(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])
    )
    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_loss = float(checkpoint["best_loss"])
    elif args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=False)

    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True)
    use_bfloat16 = bool(training.get("bfloat16", True)) and torch.cuda.is_bf16_supported()
    loss_weights = training["loss_weights"]
    epochs = int(training["epochs"])
    maximum_steps = args.max_steps
    history = []
    stop = False
    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)
        model.train()
        sums = {name: 0.0 for name in (*loss_weights, "total")}
        batches = 0
        for batch in loader:
            tensor_batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
                outputs = model(
                    tensor_batch["frame0"],
                    tensor_batch["frame1"],
                    tensor_batch["motion"],
                    tensor_batch["history"],
                    tensor_batch["history_mask"],
                )
                loss, losses = compute_phase1_loss(
                    outputs,
                    tensor_batch,
                    loss_weights,
                    visibility_negative_weight=float(
                        training.get("visibility_negative_weight", 1.0)
                    ),
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite Phase 1 loss at step {global_step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
            optimizer.step()
            global_step += 1
            batches += 1
            for name, value in losses.items():
                sums[name] += float(value.detach())
            if rank == 0 and global_step % int(training["log_every_steps"]) == 0:
                print(json.dumps({"epoch": epoch, "global_step": global_step, "loss": float(loss.detach())}))
            if maximum_steps is not None and global_step >= maximum_steps:
                stop = True
                break

        vector = torch.tensor([sums[name] for name in sums] + [float(batches)], device=device)
        if world_size > 1:
            dist.all_reduce(vector, op=dist.ReduceOp.SUM)
        total_batches = max(1.0, float(vector[-1]))
        epoch_metrics = {
            name: float(vector[index] / total_batches) for index, name in enumerate(sums)
        }
        epoch_metrics.update({"epoch": epoch, "global_step": global_step})
        history.append(epoch_metrics)
        current_loss = epoch_metrics["total"]
        if rank == 0:
            payload = _checkpoint_payload(model, optimizer, epoch, global_step, min(best_loss, current_loss), config)
            _save_checkpoint(args.output_dir / "checkpoints" / "last.ckpt", payload)
            if current_loss < best_loss:
                best_loss = current_loss
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
        print(json.dumps({"phase": 1, "global_step": global_step, "best_loss": best_loss}, sort_keys=True))
    if world_size > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
