"""Evaluate OmTrackVLA phase checkpoints."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch.utils.data import DataLoader, Subset

from omtrackvla.data.phase1 import InternGeometryDataset
from omtrackvla.evaluation.common import (
    bbox_iou,
    build_datasets,
    distributed_device,
    load_model,
    load_yaml,
    resolve,
    write_json,
)
from omtrackvla.models.phase1 import Phase1WorldIdentityModel


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--max-units-per-dataset", type=int)
    parser.add_argument("--identity-samples", type=int)
    parser.add_argument("--geometry-samples", type=int)
    parser.add_argument("--probe-train-samples", type=int)
    parser.add_argument("--probe-val-samples", type=int)
    parser.add_argument("--samples-per-mode", type=int)
    parser.add_argument("--perception-cache", type=Path)
    parser.add_argument("--allow-partial-cache", action="store_true")
    return parser.parse_args()


def _uniform_indices(length: int, maximum: int) -> list[int]:
    count = min(length, maximum)
    if count <= 0:
        return []
    return np.linspace(0, length - 1, count, dtype=np.int64).tolist()


def _identity_metrics(model, dataset, indices, batch_size, device) -> torch.Tensor:
    sums = torch.zeros(9, dtype=torch.float64, device=device)
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, num_workers=0)
    with torch.inference_mode():
        for batch in loader:
            frame0 = batch["frame0"].to(device)
            frame1 = batch["frame1"].to(device)
            target_bbox = batch["target_bbox"].to(device)
            visible = batch["target_visible"].to(device) > 0.5
            output = model(
                frame0,
                frame1,
                history=batch["history"].to(device),
                history_mask=batch["history_mask"].to(device),
            )
            predicted_visible = (
                torch.sigmoid(output["visibility_logit"])
                >= model.visibility_probability_threshold
            )
            sums[0] += visible.numel()
            sums[1] += (predicted_visible == visible).sum()
            sums[2] += visible.sum()
            invisible = ~visible
            sums[6] += invisible.sum()
            sums[7] += (predicted_visible & invisible).sum()
            sums[8] += torch.isfinite(output["bbox"]).all(dim=1).sum()
            if visible.any():
                iou = bbox_iou(output["bbox"][visible], target_bbox[visible])
                pred_center = (output["bbox"][visible, :2] + output["bbox"][visible, 2:]) / 2.0
                true_center = (target_bbox[visible, :2] + target_bbox[visible, 2:]) / 2.0
                sums[3] += iou.sum()
                sums[4] += (iou >= 0.5).sum()
                sums[5] += torch.linalg.vector_norm(pred_center - true_center, dim=1).sum()
    return sums


def _geometry_metrics(model, dataset, indices, batch_size, device) -> torch.Tensor:
    sums = torch.zeros(8, dtype=torch.float64, device=device)
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, num_workers=0)
    with torch.inference_mode():
        for batch in loader:
            frame0 = batch["frame0"].to(device)
            frame1 = batch["frame1"].to(device)
            target = batch["motion"].to(device)
            predicted = model(frame0, frame1)["motion"]
            finite = torch.isfinite(predicted).all(dim=1)
            translation = torch.linalg.vector_norm(predicted[:, :2] - target[:, :2], dim=1)
            yaw = torch.atan2(torch.sin(predicted[:, 2] - target[:, 2]), torch.cos(predicted[:, 2] - target[:, 2])).abs()
            sums[0] += target.shape[0]
            sums[1] += finite.sum()
            sums[2] += translation.sum()
            sums[3] += translation.square().sum()
            sums[4] += yaw.sum()
            sums[5] += yaw.square().sum()
            target_norm = torch.linalg.vector_norm(target[:, :2], dim=1)
            scale_valid = target_norm >= 0.05
            if scale_valid.any():
                predicted_norm = torch.linalg.vector_norm(predicted[:, :2], dim=1)
                sums[6] += (predicted_norm[scale_valid] / target_norm[scale_valid] - 1.0).abs().sum()
                sums[7] += scale_valid.sum()
    return sums


def _feature_matrix(model: Phase1WorldIdentityModel, dataset, count: int, device: torch.device):
    indices = _uniform_indices(len(dataset), count)
    loader = DataLoader(Subset(dataset, indices), batch_size=16, num_workers=0)
    features, labels = [], []
    with torch.inference_mode():
        for batch in loader:
            frame0 = batch["frame0"].to(device)
            frame1 = batch["frame1"].to(device)
            _, feature0 = model.encoder(frame0)
            _, feature1 = model.encoder(frame1)
            features.append(torch.cat((feature0, feature1, feature1 - feature0), dim=1).cpu())
            labels.append(batch["motion"].cpu())
    return torch.cat(features).double(), torch.cat(labels).double()


def _ridge(train_x, train_y, val_x, val_y, regularization: float) -> float:
    train_x = torch.cat((train_x, torch.ones(train_x.shape[0], 1, dtype=train_x.dtype)), dim=1)
    val_x = torch.cat((val_x, torch.ones(val_x.shape[0], 1, dtype=val_x.dtype)), dim=1)
    gram = train_x.T @ train_x
    identity = torch.eye(gram.shape[0], dtype=gram.dtype)
    identity[-1, -1] = 0.0
    weights = torch.linalg.solve(gram + regularization * identity, train_x.T @ train_y)
    return float(torch.mean((val_x @ weights - val_y) ** 2))


def _standardize_features(
    train_x: torch.Tensor, val_x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standardize a probe representation using train statistics only."""

    mean = train_x.mean(dim=0)
    scale = train_x.std(dim=0, unbiased=False)
    scale = torch.where(scale > 1e-12, scale, torch.ones_like(scale))
    return (train_x - mean) / scale, (val_x - mean) / scale


def _probe_metrics(model, benchmark_path: Path, benchmark, device, maximum_units):
    repository = benchmark_path.resolve().parents[2]
    train_config = load_yaml(resolve(repository, benchmark["train_config"]))
    data = train_config["data"]
    roots = {key: Path(value) for key, value in data["roots"].items()}
    manifest = resolve(repository, data["manifest"])
    train_count = int(benchmark["probe_train_samples"])
    val_count = int(benchmark["probe_val_samples"])
    train_set = InternGeometryDataset(
        manifest, "train", roots["intern_data_n1"], int(data["image_size"]),
        int(data["geometry_maximum_gap"]), int(data["history_size"]),
        train_count, int(benchmark["seed"]) + 101, maximum_units,
    )
    val_set = InternGeometryDataset(
        manifest, "val", roots["intern_data_n1"], int(data["image_size"]),
        int(data["geometry_maximum_gap"]), int(data["history_size"]),
        val_count, int(benchmark["seed"]) + 202, maximum_units,
    )
    trained_x, train_y = _feature_matrix(model, train_set, train_count, device)
    trained_val_x, val_y = _feature_matrix(model, val_set, val_count, device)
    state = torch.random.get_rng_state()
    torch.manual_seed(int(benchmark["seed"]) + 303)
    random_model = Phase1WorldIdentityModel(**train_config["model"]).to(device).eval()
    random_x, _ = _feature_matrix(random_model, train_set, train_count, device)
    random_val_x, _ = _feature_matrix(random_model, val_set, val_count, device)
    torch.random.set_rng_state(state)
    trained_x, trained_val_x = _standardize_features(trained_x, trained_val_x)
    random_x, random_val_x = _standardize_features(random_x, random_val_x)
    regularization = float(benchmark.get("probe_ridge_regularization", 1e-3))
    trained_mse = _ridge(trained_x, train_y, trained_val_x, val_y, regularization)
    random_mse = _ridge(random_x, train_y, random_val_x, val_y, regularization)
    improvement = (random_mse - trained_mse) / max(random_mse, 1e-12)
    return {
        "train_samples": train_count,
        "validation_samples": val_count,
        "feature_standardization": "train_mean_std",
        "pretrained_probe_mse": trained_mse,
        "from_scratch_probe_mse": random_mse,
        "relative_improvement": improvement,
    }


def main() -> int:
    args = _arguments()
    if args.phase == 2:
        from omtrackvla.evaluation.phase2_evaluate import run

        return run(args, distributed_device)
    if args.phase != 1:
        raise ValueError("this evaluator currently implements Phase 1 and Phase 2 only")
    world_size, rank, device = distributed_device()
    model, checkpoint = load_model(args.checkpoint, device)
    benchmark_header = load_yaml(args.config)
    benchmark_split = str(benchmark_header.get("split", "val"))
    identity, geometry, benchmark = build_datasets(
        args.config,
        split=benchmark_split,
        identity_limit_units=args.max_units_per_dataset,
        geometry_limit_units=args.max_units_per_dataset,
    )
    benchmark = dict(benchmark)
    for argument, key in (
        (args.identity_samples, "identity_samples"),
        (args.geometry_samples, "geometry_samples"),
        (args.probe_train_samples, "probe_train_samples"),
        (args.probe_val_samples, "probe_val_samples"),
    ):
        if argument is not None:
            if argument <= 0:
                raise ValueError(f"{key} override must be positive")
            benchmark[key] = argument
    batch_size = int(benchmark["batch_size_per_device"])
    identity_indices = _uniform_indices(len(identity), int(benchmark["identity_samples"]))[rank::world_size]
    geometry_indices = _uniform_indices(len(geometry), int(benchmark["geometry_samples"]))[rank::world_size]
    identity_sums = _identity_metrics(model, identity, identity_indices, batch_size, device)
    geometry_sums = _geometry_metrics(model, geometry, geometry_indices, batch_size, device)
    if world_size > 1:
        dist.all_reduce(identity_sums)
        dist.all_reduce(geometry_sums)

    if rank == 0:
        identity_count = float(identity_sums[0])
        visible_count = float(identity_sums[2])
        invisible_count = float(identity_sums[6])
        geometry_count = float(geometry_sums[0])
        metrics = {
            "schema_version": 1,
            "phase": 1,
            "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
            "split": benchmark_split,
            "benchmarks": {
                "B1-ID": {
                    "sample_count": int(identity_count),
                    "visible_sample_count": int(visible_count),
                    "invisible_sample_count": int(invisible_count),
                    "visibility_probability_threshold": model.visibility_probability_threshold,
                    "visibility_accuracy": float(identity_sums[1] / max(identity_count, 1.0)),
                    "bbox_iou_mean_visible": float(identity_sums[3] / max(visible_count, 1.0)),
                    "tracking_success_iou_0_5": float(identity_sums[4] / max(visible_count, 1.0)),
                    "center_error_norm_mean": float(identity_sums[5] / max(visible_count, 1.0)),
                    "absent_false_positive_rate": (
                        float(identity_sums[7] / invisible_count) if invisible_count else None
                    ),
                    "finite_prediction_coverage": float(identity_sums[8] / max(identity_count, 1.0)),
                },
                "B1-GEO": {
                    "sample_count": int(geometry_count),
                    "finite_prediction_coverage": float(geometry_sums[1] / max(geometry_count, 1.0)),
                    "translation_mae_m": float(geometry_sums[2] / max(geometry_count, 1.0)),
                    "translation_rmse_m": math.sqrt(float(geometry_sums[3] / max(geometry_count, 1.0))),
                    "yaw_mae_rad": float(geometry_sums[4] / max(geometry_count, 1.0)),
                    "yaw_rmse_rad": math.sqrt(float(geometry_sums[5] / max(geometry_count, 1.0))),
                    "scale_relative_error_mean": (
                        float(geometry_sums[6] / geometry_sums[7]) if geometry_sums[7] else None
                    ),
                },
                "B1-PROBE": _probe_metrics(model, args.config, benchmark, device, args.max_units_per_dataset),
            },
        }
        write_json(args.metrics_out, metrics)
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(
            "# Phase 1 evaluation\n\n"
            f"Checkpoint step: `{metrics['checkpoint_global_step']}`  \n"
            f"Split: `{benchmark_split}`\n\n"
            "```json\n" + json.dumps(metrics["benchmarks"], indent=2, sort_keys=True) + "\n```\n",
            encoding="utf-8",
        )
        print(json.dumps(metrics, sort_keys=True))
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
