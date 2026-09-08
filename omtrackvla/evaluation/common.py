"""Shared Phase 1 evaluation helpers."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import yaml
from torch import distributed as dist

from omtrackvla.data.phase1 import ContractIdentityDataset, InternGeometryDataset
from omtrackvla.models.phase1 import Phase1WorldIdentityModel


def load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def resolve(repository: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository / path


def distributed_device() -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 1 evaluation requires CUDA")
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    return world_size, rank, torch.device("cuda", local_rank)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[Phase1WorldIdentityModel, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("phase") != 1 or not isinstance(checkpoint.get("model_config"), dict):
        raise ValueError("checkpoint is not a Phase 1 checkpoint")
    model = Phase1WorldIdentityModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, checkpoint


def build_datasets(
    benchmark_path: Path,
    split: str,
    identity_limit_units: int | None = None,
    geometry_limit_units: int | None = None,
) -> tuple[ContractIdentityDataset, InternGeometryDataset, dict[str, object]]:
    benchmark = load_yaml(benchmark_path)
    repository = benchmark_path.resolve().parents[2]
    train_path = resolve(repository, benchmark["train_config"])
    train = load_yaml(train_path)
    data = train["data"]
    manifest = resolve(repository, data["manifest"])
    roots = {key: Path(value) for key, value in data["roots"].items()}
    identity = ContractIdentityDataset(
        manifest,
        split=split,
        roots=roots,
        datasets=tuple(
            data.get(
                "identity_datasets", ("sage3d_extracted", "tpt_bench_clean_v2")
            )
        ),
        sage3d_sidecar=data.get("sage3d_sidecar"),
        image_size=int(data["image_size"]),
        history_size=int(data["history_size"]),
        max_units_per_dataset=identity_limit_units,
    )
    geometry = InternGeometryDataset(
        manifest,
        split=split,
        root=roots["intern_data_n1"],
        image_size=int(data["image_size"]),
        maximum_gap=int(data["geometry_maximum_gap"]),
        history_size=int(data["history_size"]),
        samples_per_epoch=int(benchmark["geometry_samples"]),
        seed=int(benchmark.get("seed", train.get("seed", 20260907))),
        max_units=geometry_limit_units,
    )
    return identity, geometry, benchmark


def bbox_iou(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(predicted[:, :2], target[:, :2])
    bottom_right = torch.minimum(predicted[:, 2:], target[:, 2:])
    intersection = (bottom_right - top_left).clamp_min(0.0).prod(dim=1)
    predicted_area = (predicted[:, 2:] - predicted[:, :2]).clamp_min(0.0).prod(dim=1)
    target_area = (target[:, 2:] - target[:, :2]).clamp_min(0.0).prod(dim=1)
    return intersection / (predicted_area + target_area - intersection).clamp_min(1e-8)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
