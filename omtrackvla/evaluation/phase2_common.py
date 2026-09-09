"""Shared checkpoint and dataset helpers for Phase 2 evaluation."""
from __future__ import annotations

from pathlib import Path

import torch

from omtrackvla.data.phase2 import Sage3DPolicyDataset
from omtrackvla.evaluation.common import load_yaml, resolve
from omtrackvla.models.phase2 import Phase2WaypointPolicy


def load_phase2_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[Phase2WaypointPolicy, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("phase") != 2 or not isinstance(checkpoint.get("model_config"), dict):
        raise ValueError("checkpoint is not a Phase 2 checkpoint")
    model = Phase2WaypointPolicy(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, checkpoint


def build_phase2_dataset(
    benchmark_path: Path,
    split: str,
    *,
    maximum_units: int | None = None,
    perception_cache: Path | None = None,
    include_render_data: bool = False,
    allow_partial_cache: bool = False,
) -> tuple[Sage3DPolicyDataset, dict[str, object]]:
    benchmark = load_yaml(benchmark_path)
    if benchmark.get("phase") != 2:
        raise ValueError("Phase 2 benchmark config must declare phase: 2")
    repository = benchmark_path.resolve().parents[2]
    train = load_yaml(resolve(repository, benchmark["train_config"]))
    data = train["data"]
    caches = data.get("perception_caches")
    if perception_cache is None:
        if not isinstance(caches, dict) or split not in caches:
            raise ValueError(f"Phase 2 perception cache is not configured for split={split}")
        perception_cache = resolve(repository, caches[split])
    dataset = Sage3DPolicyDataset(
        resolve(repository, data["manifest"]),
        split=split,
        root=resolve(repository, data["root"]),
        sidecar_root=resolve(repository, data["sidecar_root"]),
        policy_admission=resolve(repository, data["policy_admission"]),
        perception_cache=perception_cache,
        history_size=int(data["history_size"]),
        image_size=int(data["image_size"]),
        max_units=maximum_units,
        include_render_data=include_render_data,
        allow_partial_cache=allow_partial_cache,
    )
    return dataset, benchmark


def model_inputs(batch: dict[str, object], device: torch.device) -> dict[str, torch.Tensor]:
    keys = (
        "visual_xy",
        "visual_confidence",
        "visual_valid",
        "uwb_xy",
        "uwb_quality",
        "uwb_valid",
        "uwb_age_s",
        "condition_index",
    )
    return {key: batch[key].to(device, non_blocking=True) for key in keys}
