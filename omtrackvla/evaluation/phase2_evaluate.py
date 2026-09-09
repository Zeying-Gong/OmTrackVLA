"""Open-loop Phase 2 waypoint and safety evaluation."""
from __future__ import annotations

import json
import math

import numpy as np
import torch
from torch import distributed as dist
from torch.utils.data import DataLoader, Subset

from omtrackvla.data.phase2 import CONDITION_MODES
from omtrackvla.evaluation.common import bbox_iou, load_yaml, write_json
from omtrackvla.evaluation.phase2_common import (
    build_phase2_dataset,
    load_phase2_model,
    model_inputs,
)


def _uniform(values: list[int], maximum: int) -> list[int]:
    count = min(len(values), int(maximum))
    if count <= 0:
        return []
    positions = np.linspace(0, len(values) - 1, count, dtype=np.int64)
    return [values[int(position)] for position in positions]


def _wrapped_angle_error(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    predicted_angle = torch.atan2(predicted[:, 1], predicted[:, 0])
    target_angle = torch.atan2(target[:, 1], target[:, 0])
    return torch.abs(torch.atan2(torch.sin(predicted_angle - target_angle), torch.cos(predicted_angle - target_angle)))


def _mode_sums(model, dataset, indices, batch_size: int, device: torch.device) -> torch.Tensor:
    # count, finite, ADE, FDE, heading, exact-stop, pred-visible, gt-visible,
    # visibility-intersection, visible-pair-count, visible-pair-IoU
    sums = torch.zeros(11, dtype=torch.float64, device=device)
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, num_workers=0)
    with torch.inference_mode():
        for batch in loader:
            outputs = model(**model_inputs(batch, device))
            predicted = outputs["waypoints"]
            target = batch["waypoints"].to(device)
            mask = batch["waypoint_mask"].to(device).bool()
            finite = torch.isfinite(predicted).all(dim=(1, 2))
            errors = torch.linalg.vector_norm(predicted - target, dim=2)
            future_mask = mask[:, 1:]
            ade = (errors[:, 1:] * future_mask).sum(dim=1) / future_mask.sum(dim=1).clamp_min(1)
            fde = errors[:, -1]
            heading = _wrapped_angle_error(predicted[:, -1], target[:, -1])
            exact_stop = predicted.abs().amax(dim=(1, 2)) <= 1e-8
            predicted_visible = batch["predicted_visible"].to(device) > 0.5
            target_visible = batch["target_visible"].to(device) > 0.5
            both_visible = predicted_visible & target_visible
            sums[0] += predicted.shape[0]
            sums[1] += finite.sum()
            sums[2] += ade[finite].sum()
            sums[3] += fde[finite].sum()
            sums[4] += heading[finite].sum()
            sums[5] += exact_stop.sum()
            sums[6] += predicted_visible.sum()
            sums[7] += target_visible.sum()
            sums[8] += both_visible.sum()
            if both_visible.any():
                iou = bbox_iou(
                    batch["predicted_bbox"].to(device)[both_visible],
                    batch["target_bbox"].to(device)[both_visible],
                )
                sums[9] += iou.numel()
                sums[10] += iou.sum()
    return sums


def _metrics(values: torch.Tensor, *, safety: bool = False) -> dict[str, object]:
    count = float(values[0])
    finite = float(values[1])
    result = {
        "sample_count": int(count),
        "finite_prediction_coverage": finite / max(count, 1.0),
        "waypoint_ade_m": float(values[2]) / max(finite, 1.0),
        "waypoint_fde_m": float(values[3]) / max(finite, 1.0),
        "endpoint_heading_mae_rad": float(values[4]) / max(finite, 1.0),
    }
    if safety:
        result["exact_stop_rate"] = float(values[5]) / max(count, 1.0)
    return result


def run(args, distributed_device) -> int:
    world_size, rank, device = distributed_device()
    model, checkpoint = load_phase2_model(args.checkpoint, device)
    benchmark_header = load_yaml(args.config)
    benchmark_split = str(benchmark_header.get("split", "val"))
    dataset, benchmark = build_phase2_dataset(
        args.config,
        benchmark_split,
        maximum_units=args.max_units_per_dataset,
        perception_cache=args.perception_cache,
        allow_partial_cache=args.allow_partial_cache,
    )
    samples_per_mode = int(args.samples_per_mode or benchmark["samples_per_mode"])
    if samples_per_mode <= 0:
        raise ValueError("Phase 2 samples_per_mode must be positive")
    batch_size = int(benchmark["batch_size_per_device"])
    values = {}
    for mode_index, mode in enumerate(CONDITION_MODES):
        candidates = list(range(mode_index, len(dataset), len(CONDITION_MODES)))
        indices = _uniform(candidates, samples_per_mode)[rank::world_size]
        mode_values = _mode_sums(model, dataset, indices, batch_size, device)
        if world_size > 1:
            dist.all_reduce(mode_values)
        values[mode] = mode_values

    if rank == 0:
        mode_metrics = {
            mode: _metrics(value, safety=mode == "safe_stop")
            for mode, value in values.items()
        }
        normal = sum((values[mode] for mode in CONDITION_MODES[:3]), torch.zeros_like(values[CONDITION_MODES[0]]))
        visual = values["visual_uwb"] + values["visual_only"]
        predicted_visible = float(visual[6])
        target_visible = float(visual[7])
        intersection = float(visual[8])
        metrics = {
            "schema_version": 1,
            "phase": 2,
            "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
            "split": benchmark_split,
            "benchmarks": {
                "B2-OPEN": {
                    **_metrics(normal),
                    "modes": mode_metrics,
                },
                "B2-SAFETY": {
                    "sample_count": mode_metrics["safe_stop"]["sample_count"],
                    "exact_stop_rate": mode_metrics["safe_stop"]["exact_stop_rate"],
                    "finite_prediction_coverage": mode_metrics["safe_stop"]["finite_prediction_coverage"],
                },
                "B2-PERCEPTION": {
                    "sample_count": int(visual[0]),
                    "predicted_visible_count": int(predicted_visible),
                    "target_visible_count": int(target_visible),
                    "visibility_precision": intersection / max(predicted_visible, 1.0),
                    "visibility_recall": intersection / max(target_visible, 1.0),
                    "bbox_iou_mean_both_visible": float(visual[10]) / max(float(visual[9]), 1.0),
                    "provenance": "frozen_front_end_diagnostic_not_policy_input_bbox",
                },
                "B2-CLOSED": {
                    "episode_count": 0,
                    "status": "not_available_in_offline_sage3d_cache",
                },
            },
        }
        write_json(args.metrics_out, metrics)
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(
            "# Phase 2 evaluation\n\n"
            f"Checkpoint step: `{metrics['checkpoint_global_step']}`  \n"
            f"Split: `{benchmark_split}`  \n"
            "Scope: open-loop SAGE3D waypoint, frozen-perception diagnostics, and hard safe-stop. "
            "Closed-loop Habitat evaluation is not yet available and is not claimed.\n\n"
            "```json\n" + json.dumps(metrics["benchmarks"], indent=2, sort_keys=True) + "\n```\n",
            encoding="utf-8",
        )
        print(json.dumps(metrics, sort_keys=True))
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0
