"""Perception/Polar-only EVT adaptation and fixed-dev evaluation for v2."""
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
from PIL import Image, ImageDraw
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset

from omtrackvla.data.evt_perception import EVTPerceptionPolarDataset
from omtrackvla.models.end_to_end import (
    ArchitectureV1Ablation,
    ArchitectureV1Config,
    load_official_da3_small_l11,
)
from omtrackvla.models.end_to_end_v2 import (
    ArchitectureV2DecoderConfig,
    ArchitectureV2FollowPolicy,
    parameter_inventory_v2,
)


METHOD = "architecture_v2_evt_perception_polar"
INPUT_KEYS = (
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


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--render-samples", type=int, default=32)
    return parser.parse_args()


def _resolve(repository: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else repository / path).resolve(strict=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distributed() -> tuple[int, int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("EVT perception adaptation requires CUDA")
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    return world, rank, local_rank, torch.device("cuda", local_rank)


def _seed(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def _load_checkpoint(
    policy: ArchitectureV2FollowPolicy, path: Path
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("method") not in {"architecture_v2_dual_head", METHOD}
        or payload.get("test_locked_used") is not False
    ):
        raise ValueError("checkpoint method/provenance mismatch")
    target = policy.state_dict()
    source = payload["model"]
    unexpected = sorted(name for name in source if name not in target)
    mismatched = sorted(
        name
        for name, value in source.items()
        if name in target and value.shape != target[name].shape
    )
    compatible = {
        name: value
        for name, value in source.items()
        if name in target and value.shape == target[name].shape
    }
    incompatible = policy.load_state_dict(compatible, strict=False)
    allowed_missing_prefixes = ("trajectory.target_polar_head.",)
    disallowed_missing = [
        name
        for name in incompatible.missing_keys
        if not name.startswith(allowed_missing_prefixes)
    ]
    if unexpected or mismatched or incompatible.unexpected_keys or disallowed_missing:
        raise RuntimeError(
            "unsafe v2 EVT warm start: "
            f"missing={disallowed_missing}, unexpected={unexpected}, "
            f"shape_mismatch={mismatched}"
        )
    report = {
        "checkpoint": str(path),
        "checkpoint_sha256": _sha256(path),
        "source_method": payload.get("method"),
        "source_stage": payload.get("stage"),
        "source_global_step": payload.get("global_step"),
        "missing_new_parameters": list(incompatible.missing_keys),
        "unexpected_parameters": unexpected,
        "shape_mismatches": mismatched,
    }
    return payload, report


def _build_policy(
    config: Mapping[str, Any], repository: Path, device: torch.device
) -> tuple[ArchitectureV2FollowPolicy, dict[str, Any]]:
    architecture = ArchitectureV1Config(**config["architecture"])
    decoder = ArchitectureV2DecoderConfig(**config["decoder"])
    if (
        architecture.history_size != 8
        or decoder.history_size != 8
        or not decoder.predict_target_diagnostics
        or not decoder.predict_target_polar
    ):
        raise ValueError("EVT adaptation requires frozen v2 eight-frame Polar contract")
    for field in ("da3_source", "da3_runtime"):
        path = _resolve(repository, config[field])
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    da3, loading = load_official_da3_small_l11(
        _resolve(repository, config["da3_model"]),
        architecture,
        ArchitectureV1Ablation(backbone_tuning="adapter"),
    )
    return ArchitectureV2FollowPolicy(da3, architecture, decoder).to(device), loading


def _inputs(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].to(device, non_blocking=True)
        for key in INPUT_KEYS
    }


def _targets(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "bbox": batch["target_bbox"].to(device, non_blocking=True),
        "visible": batch["target_visible"].to(device, non_blocking=True),
        "angle_sincos": batch["target_angle_sincos"].to(device, non_blocking=True),
        "angle_valid": batch["target_angle_valid"].to(device, non_blocking=True).bool(),
        "distance_m": batch["target_distance_m"].to(device, non_blocking=True),
        "distance_valid": batch["target_distance_valid"].to(
            device, non_blocking=True
        ).bool(),
    }


def _losses(
    outputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    *,
    positive_fraction: float,
    weights: Mapping[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = outputs["visibility_logit"].sum() * 0.0
    visible = targets["visible"].bool()
    bbox = (
        F.smooth_l1_loss(outputs["bbox_pred"][visible], targets["bbox"][visible])
        if visible.any()
        else outputs["bbox_pred"].sum() * 0.0
    )
    visibility_raw = F.binary_cross_entropy_with_logits(
        outputs["visibility_logit"].squeeze(-1), targets["visible"], reduction="none"
    )
    positive_fraction = min(max(float(positive_fraction), 1.0e-4), 1.0 - 1.0e-4)
    class_weight = torch.where(
        visible,
        visibility_raw.new_tensor(0.5 / positive_fraction),
        visibility_raw.new_tensor(0.5 / (1.0 - positive_fraction)),
    )
    visibility = (visibility_raw * class_weight).mean()
    angle_valid = targets["angle_valid"]
    if angle_valid.any():
        predicted_angle = F.normalize(
            outputs["target_angle_sincos"][angle_valid].float(), dim=1, eps=1.0e-6
        )
        target_angle = F.normalize(
            targets["angle_sincos"][angle_valid].float(), dim=1, eps=1.0e-6
        )
        angle = (1.0 - (predicted_angle * target_angle).sum(dim=1)).mean()
    else:
        angle = outputs["target_angle_sincos"].sum() * 0.0
    distance_valid = targets["distance_valid"]
    if distance_valid.any():
        distance = F.smooth_l1_loss(
            torch.log1p(outputs["target_distance_m"].squeeze(-1)[distance_valid]),
            torch.log1p(targets["distance_m"][distance_valid]),
        )
    else:
        distance = outputs["target_distance_m"].sum() * 0.0
    values = {
        "bbox": bbox,
        "visibility": visibility,
        "angle": angle,
        "distance": distance,
    }
    total = sum(float(weights.get(name, 0.0)) * value for name, value in values.items())
    values["total"] = total
    return total, values


def _bbox_iou(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(predicted[:, :2], target[:, :2])
    bottom_right = torch.minimum(predicted[:, 2:], target[:, 2:])
    intersection = (bottom_right - top_left).clamp_min(0.0).prod(dim=1)
    predicted_area = (predicted[:, 2:] - predicted[:, :2]).clamp_min(0.0).prod(dim=1)
    target_area = (target[:, 2:] - target[:, :2]).clamp_min(0.0).prod(dim=1)
    return intersection / (predicted_area + target_area - intersection).clamp_min(1.0e-8)


def _empty_metric() -> dict[str, float]:
    return {
        "samples": 0.0,
        "bbox_iou_sum": 0.0,
        "bbox_mae_sum": 0.0,
        "bbox_count": 0.0,
        "tp": 0.0,
        "tn": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "angle_abs_rad_sum": 0.0,
        "angle_count": 0.0,
        "distance_abs_sum": 0.0,
        "distance_sq_sum": 0.0,
        "distance_bias_sum": 0.0,
        "distance_count": 0.0,
    }


def _finalize_metric(value: Mapping[str, float]) -> dict[str, Any]:
    tp, tn, fp, fn = (float(value[name]) for name in ("tp", "tn", "fp", "fn"))
    positive = tp + fn
    negative = tn + fp
    total = positive + negative
    recall = tp / max(1.0, positive)
    specificity = tn / max(1.0, negative)
    precision = tp / max(1.0, tp + fp)
    return {
        "samples": int(value["samples"]),
        "bbox_iou": value["bbox_iou_sum"] / max(1.0, value["bbox_count"]),
        "bbox_mae_normalized": value["bbox_mae_sum"]
        / max(1.0, value["bbox_count"] * 4.0),
        "bbox_visible_count": int(value["bbox_count"]),
        "visibility": {
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "accuracy": (tp + tn) / max(1.0, total),
            "balanced_accuracy": 0.5 * (recall + specificity),
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": 2.0 * precision * recall / max(1.0e-12, precision + recall),
        },
        "angle_mae_deg": value["angle_abs_rad_sum"]
        / max(1.0, value["angle_count"])
        * 180.0
        / math.pi,
        "angle_valid_count": int(value["angle_count"]),
        "distance_mae_m": value["distance_abs_sum"]
        / max(1.0, value["distance_count"]),
        "distance_rmse_m": math.sqrt(
            value["distance_sq_sum"] / max(1.0, value["distance_count"])
        ),
        "distance_bias_m": value["distance_bias_sum"]
        / max(1.0, value["distance_count"]),
        "distance_valid_count": int(value["distance_count"]),
    }


def _render_contact_sheet(
    dataset: EVTPerceptionPolarDataset,
    predictions: list[dict[str, Any]],
    output: Path,
    maximum: int,
) -> None:
    if maximum <= 0:
        return
    selected: list[dict[str, Any]] = []
    for case in sorted({row["case_id"] for row in predictions}):
        rows = [row for row in predictions if row["case_id"] == case]
        selected.extend([row for row in rows if row["target_visible"]][:4])
        selected.extend([row for row in rows if not row["target_visible"]][:4])
    selected = selected[:maximum]
    if not selected:
        return
    tile_w, tile_h = 384, 430
    columns = 4
    rows_count = math.ceil(len(selected) / columns)
    sheet = Image.new("RGB", (columns * tile_w, rows_count * tile_h), "white")
    for tile, row in enumerate(selected):
        sample_index = int(row["sample_index"])
        case_index, frame_index = dataset.samples[sample_index]
        record = dataset.cases[case_index].records[frame_index]
        with Image.open(record.image_path) as source:
            image = source.convert("RGB").resize((tile_w, 384))
        draw = ImageDraw.Draw(image)
        if row["target_visible"]:
            box = row["target_bbox"]
            draw.rectangle(
                tuple(int(value * 384) for value in box), outline=(0, 255, 0), width=3
            )
        box = row["predicted_bbox"]
        draw.rectangle(
            tuple(int(value * 384) for value in box), outline=(255, 0, 255), width=3
        )
        caption = Image.new("RGB", (tile_w, tile_h - 384), "white")
        caption_draw = ImageDraw.Draw(caption)
        caption_draw.text(
            (4, 2),
            f"{row['case_id']} f{frame_index} GT/P={int(row['target_visible'])}/{row['predicted_visible_probability']:.2f}",
            fill="black",
        )
        caption_draw.text(
            (4, 18),
            f"angle GT/P={row['target_angle_deg']:.1f}/{row['predicted_angle_deg']:.1f}  dist={row['target_distance_m']:.2f}/{row['predicted_distance_m']:.2f}m",
            fill="black",
        )
        sheet.paste(image, ((tile % columns) * tile_w, (tile // columns) * tile_h))
        sheet.paste(
            caption,
            ((tile % columns) * tile_w, (tile // columns) * tile_h + 384),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def evaluate(
    policy: ArchitectureV2FollowPolicy,
    dataset: EVTPerceptionPolarDataset,
    device: torch.device,
    *,
    batch_size: int,
    workers: int,
    use_bfloat16: bool,
    output_dir: Path,
    render_samples: int,
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    totals: dict[str, dict[str, float]] = {"all": _empty_metric()}
    predictions: list[dict[str, Any]] = []
    sample_offset = 0
    policy.eval()
    with torch.inference_mode():
        for batch in loader:
            targets = _targets(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
                outputs = policy(**_inputs(batch, device))
            predicted_bbox = outputs["bbox_pred"].float()
            predicted_visible = torch.sigmoid(outputs["visibility_logit"].squeeze(-1).float())
            predicted_angle_vector = F.normalize(
                outputs["target_angle_sincos"].float(), dim=1, eps=1.0e-6
            )
            predicted_angle = torch.atan2(
                predicted_angle_vector[:, 0], predicted_angle_vector[:, 1]
            )
            target_angle = torch.atan2(
                targets["angle_sincos"][:, 0], targets["angle_sincos"][:, 1]
            )
            angle_error = torch.atan2(
                torch.sin(predicted_angle - target_angle),
                torch.cos(predicted_angle - target_angle),
            ).abs()
            predicted_distance = outputs["target_distance_m"].squeeze(-1).float()
            distance_error = predicted_distance - targets["distance_m"]
            iou = _bbox_iou(predicted_bbox, targets["bbox"])
            visible = targets["visible"].bool()
            visible_prediction = predicted_visible >= 0.5
            tasks = [str(value) for value in batch["task"]]
            case_ids = [str(value) for value in batch["case_id"]]
            for index, task in enumerate(tasks):
                for name in ("all", task):
                    metric = totals.setdefault(name, _empty_metric())
                    metric["samples"] += 1.0
                    truth = bool(visible[index])
                    prediction = bool(visible_prediction[index])
                    metric[
                        "tp" if truth and prediction else "fn" if truth else "fp" if prediction else "tn"
                    ] += 1.0
                    if truth:
                        metric["bbox_iou_sum"] += float(iou[index])
                        metric["bbox_mae_sum"] += float(
                            (predicted_bbox[index] - targets["bbox"][index]).abs().sum()
                        )
                        metric["bbox_count"] += 1.0
                    if bool(targets["angle_valid"][index]):
                        metric["angle_abs_rad_sum"] += float(angle_error[index])
                        metric["angle_count"] += 1.0
                    if bool(targets["distance_valid"][index]):
                        error = float(distance_error[index])
                        metric["distance_abs_sum"] += abs(error)
                        metric["distance_sq_sum"] += error * error
                        metric["distance_bias_sum"] += error
                        metric["distance_count"] += 1.0
                predictions.append(
                    {
                        "sample_index": sample_offset + index,
                        "case_id": case_ids[index],
                        "task": task,
                        "frame_index": int(batch["frame_index"][index]),
                        "target_visible": bool(visible[index]),
                        "predicted_visible_probability": float(predicted_visible[index]),
                        "target_bbox": [float(value) for value in targets["bbox"][index]],
                        "predicted_bbox": [float(value) for value in predicted_bbox[index]],
                        "target_angle_deg": float(target_angle[index] * 180.0 / math.pi),
                        "predicted_angle_deg": float(predicted_angle[index] * 180.0 / math.pi),
                        "angle_valid": bool(targets["angle_valid"][index]),
                        "target_distance_m": float(targets["distance_m"][index]),
                        "predicted_distance_m": float(predicted_distance[index]),
                    }
                )
            sample_offset += len(tasks)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "partition": dataset.partition,
        "fixed_case_count": len(dataset.cases),
        "metrics": {name: _finalize_metric(value) for name, value in totals.items()},
        "history_sampling": {
            "frames": 8,
            "interval_s": dataset.history_interval_s,
            "method": "causal_timestamp_floor_with_left_padding",
        },
        "uwb_valid_for_every_sample": False,
        "gt_pose_or_bbox_used_as_model_input": False,
        "optimizer_steps": 0,
        "official_evt_validation": False,
        "test_locked_used": False,
    }
    truth = np.asarray([row["target_visible"] for row in predictions], dtype=bool)
    scores = np.asarray(
        [row["predicted_visible_probability"] for row in predictions], dtype=np.float64
    )
    positive_scores = scores[truth]
    negative_scores = scores[~truth]
    if positive_scores.size and negative_scores.size:
        auroc = float(
            (positive_scores[:, None] > negative_scores[None, :]).mean()
            + 0.5 * (positive_scores[:, None] == negative_scores[None, :]).mean()
        )
        best = None
        for threshold in np.unique(np.concatenate(([0.0], scores, [1.0]))):
            predicted = scores >= threshold
            recall = float(predicted[truth].mean())
            specificity = float((~predicted[~truth]).mean())
            candidate = (
                0.5 * (recall + specificity),
                float(threshold),
                recall,
                specificity,
            )
            if best is None or candidate[0] > best[0]:
                best = candidate
        metrics["metrics"]["all"]["visibility"]["auroc"] = auroc
        metrics["metrics"]["all"]["visibility"]["dev_tuned_threshold"] = best[1]
        metrics["metrics"]["all"]["visibility"][
            "dev_tuned_balanced_accuracy"
        ] = best[0]
        metrics["metrics"]["all"]["visibility"]["dev_tuned_recall"] = best[2]
        metrics["metrics"]["all"]["visibility"][
            "dev_tuned_specificity"
        ] = best[3]
    (output_dir / "EVT_DEV_METRICS.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "DEV_PREDICTIONS.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    _render_contact_sheet(
        dataset, predictions, output_dir / "DEV_CONTACT_SHEET.png", render_samples
    )
    return metrics


def _atomic_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def main() -> int:
    args = arguments()
    config_path = args.config.expanduser().resolve(strict=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (
        config.get("schema_version") != 1
        or config.get("method") != METHOD
        or config.get("test_locked_used") is not False
    ):
        raise ValueError("expected non-locked Architecture-v2 EVT perception config")
    repository = config_path.parents[2]
    output_dir = args.output_dir.expanduser().resolve()
    world, rank, local_rank, device = _distributed()
    seed = int(config["seed"])
    _seed(seed, rank)
    torch.set_float32_matmul_precision("high")
    policy, da3_loading = _build_policy(config, repository, device)
    checkpoint = args.checkpoint if args.eval_only else args.initial_checkpoint
    if checkpoint is None:
        raise ValueError("a checkpoint is required")
    checkpoint = checkpoint.expanduser().resolve(strict=True)
    _, loading = _load_checkpoint(policy, checkpoint)
    architecture = ArchitectureV1Config(**config["architecture"])
    data_config = config["data"]
    use_bfloat16 = bool(config["training"].get("bfloat16", True)) and torch.cuda.is_bf16_supported()

    if args.eval_only:
        if world != 1:
            raise ValueError("fixed dev evaluation must use one process")
        dataset = EVTPerceptionPolarDataset(
            _resolve(repository, data_config["dev_audit"]),
            partition="dev",
            config=architecture,
            history_interval_s=float(data_config["history_interval_s"]),
            polar_valid_only_when_visible=bool(
                data_config.get("polar_valid_only_when_visible", False)
            ),
        )
        metrics = evaluate(
            policy,
            dataset,
            device,
            batch_size=int(config["evaluation"]["batch_size"]),
            workers=int(config["evaluation"]["num_workers"]),
            use_bfloat16=use_bfloat16,
            output_dir=output_dir,
            render_samples=args.render_samples,
        )
        (output_dir / "LOADING_REPORT.json").write_text(
            json.dumps(
                {
                    "checkpoint": loading,
                    "da3_pretrained_loading": da3_loading,
                    "parameter_inventory": parameter_inventory_v2(policy),
                    "test_locked_used": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(metrics, sort_keys=True), flush=True)
        return 0

    if not bool(data_config.get("user_authorized_perception_training", False)):
        raise PermissionError("config lacks explicit perception-only user authorization")
    dataset = EVTPerceptionPolarDataset(
        _resolve(repository, data_config["train_audit"]),
        partition="train",
        config=architecture,
        history_interval_s=float(data_config["history_interval_s"]),
        user_authorized_perception_training=True,
        polar_valid_only_when_visible=bool(
            data_config.get("polar_valid_only_when_visible", False)
        ),
    )
    training = config["training"]
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    allowed_heads = {"bbox_head", "visibility_head", "target_polar_head"}
    head_names = tuple(
        str(name)
        for name in training.get(
            "optimizer_heads", ["bbox_head", "visibility_head", "target_polar_head"]
        )
    )
    if not head_names or len(set(head_names)) != len(head_names) or any(
        name not in allowed_heads for name in head_names
    ):
        raise ValueError("optimizer_heads must be a non-empty unique diagnostic subset")
    for name in head_names:
        module = getattr(policy.trajectory, name)
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    inventory = parameter_inventory_v2(policy)
    wrapped: nn.Module = policy
    if world > 1:
        wrapped = DistributedDataParallel(
            policy, device_ids=[local_rank], find_unused_parameters=False
        )
    optimizer_dataset: torch.utils.data.Dataset = dataset
    if bool(training.get("train_visible_only", False)):
        optimizer_dataset = Subset(dataset, dataset.visible_indices)
    sampler = DistributedSampler(
        optimizer_dataset, world, rank, shuffle=True, seed=seed
    )
    loader = DataLoader(
        optimizer_dataset,
        batch_size=int(training["batch_size_per_device"]),
        sampler=sampler,
        num_workers=int(training["num_workers"]),
        pin_memory=True,
        drop_last=True,
        persistent_workers=int(training["num_workers"]) > 0,
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    maximum_steps = int(args.max_steps or training["max_steps"])
    warmup = int(training["warmup_steps"])

    def schedule(step: int) -> float:
        if step < warmup:
            return max(1.0e-3, (step + 1) / max(1, warmup))
        progress = (step - warmup) / max(1, maximum_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    positive_fraction = len(dataset.visible_indices) / len(dataset)
    loss_weights = {
        str(name): float(value)
        for name, value in training["loss_weights"].items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        shutil.copy2(config_path, output_dir / "config.yaml")
        (output_dir / "RUN_MANIFEST.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "method": METHOD,
                    "stage": config["stage"],
                    "world_size": world,
                    "physical_gpu_zero_forbidden": True,
                    "dataset_samples": len(dataset),
                    "dataset_cases": len(dataset.cases),
                    "visible_samples": len(dataset.visible_indices),
                    "invisible_samples": len(dataset.invisible_indices),
                    "optimizer_samples": len(optimizer_dataset),
                    "train_visible_only": bool(
                        training.get("train_visible_only", False)
                    ),
                    "history_sampling": "8 frames, causal timestamp floor, 0.1 s",
                    "uwb_disabled_to_prevent_GT_Polar_leakage": True,
                    "teacher_actions_loaded": False,
                    "waypoint_labels_loaded": False,
                    "optimizer_scope": list(head_names),
                    "maximum_optimizer_steps": maximum_steps,
                    "user_authorized_perception_training": True,
                    "source_audit_optimizer_flag_preserved_as_false": True,
                    "authorization_scope": "perception_and_polar_heads_only",
                    "da3_pretrained_loading": da3_loading,
                    "initial_checkpoint_loading": loading,
                    "parameter_inventory": inventory,
                    "loss_weights": loss_weights,
                    "test_locked_used": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if world > 1:
        dist.barrier()

    log_path = output_dir / "train_log.jsonl"
    step = 0
    epoch = 0
    start = time.time()
    first_backward = True
    first_gradient_norms = {name: 0.0 for name in head_names}
    policy.eval()
    for name in head_names:
        getattr(policy.trajectory, name).train()
    while step < maximum_steps:
        sampler.set_epoch(epoch)
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            targets = _targets(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
                outputs = wrapped(**_inputs(batch, device))
                loss, values = _losses(
                    outputs,
                    targets,
                    positive_fraction=positive_fraction,
                    weights=loss_weights,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            loss.backward()
            if first_backward:
                gradient_norms = {}
                for name in head_names:
                    squares = [
                        parameter.grad.detach().float().square().sum()
                        for parameter in getattr(policy.trajectory, name).parameters()
                        if parameter.grad is not None
                    ]
                    gradient_norms[name] = (
                        float(torch.stack(squares).sum().sqrt().cpu()) if squares else 0.0
                    )
                    first_gradient_norms[name] = max(
                        first_gradient_norms[name], gradient_norms[name]
                    )
                frozen_with_grad = [
                    name
                    for name, parameter in policy.named_parameters()
                    if not parameter.requires_grad and parameter.grad is not None
                ]
                if frozen_with_grad:
                    raise RuntimeError(
                        f"head-only backward contract failed: frozen_grad={frozen_with_grad[:5]}"
                    )
                if all(value > 0.0 for value in first_gradient_norms.values()) and rank == 0:
                    (output_dir / "FIRST_BACKWARD_REPORT.json").write_text(
                        json.dumps(
                            {
                                "status": "passed",
                                "head_gradient_norms": first_gradient_norms,
                                "frozen_parameters_with_gradient": [],
                                "waypoint_or_action_loss_used": False,
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                if all(value > 0.0 for value in first_gradient_norms.values()):
                    first_backward = False
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable, float(training["gradient_clip_norm"])
            )
            optimizer.step()
            scheduler.step()
            step += 1
            if rank == 0 and (
                step == 1 or step % int(training["log_every_steps"]) == 0
            ):
                row = {
                    "global_step": step,
                    "losses": {
                        name: float(value.detach().float().cpu())
                        for name, value in values.items()
                    },
                    "gradient_norm": float(gradient_norm.detach().float().cpu()),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "elapsed_s": time.time() - start,
                }
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                print(json.dumps(row, sort_keys=True), flush=True)
            checkpoint_every = int(training["checkpoint_every_steps"])
            if step % checkpoint_every == 0 or step >= maximum_steps:
                if world > 1:
                    dist.barrier()
                if rank == 0:
                    payload = {
                        "schema_version": 1,
                        "phase": "evt_perception_polar_head_only",
                        "method": METHOD,
                        "stage": config["stage"],
                        "global_step": step,
                        "model": policy.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "head_only": True,
                        "teacher_actions_loaded": False,
                        "waypoint_labels_loaded": False,
                        "test_locked_used": False,
                    }
                    _atomic_save(output_dir / "checkpoints" / "last.ckpt", payload)
                    _atomic_save(
                        output_dir / "checkpoints" / f"step_{step:07d}.ckpt", payload
                    )
                if world > 1:
                    dist.barrier()
            if step >= maximum_steps:
                break
        epoch += 1
    if rank == 0:
        last = output_dir / "checkpoints" / "last.ckpt"
        shutil.copy2(last, output_dir / "checkpoints" / "best.ckpt")
        completion = {
            "status": "head_only_smoke_complete",
            "global_step": step,
            "checkpoint": str(output_dir / "checkpoints" / "best.ckpt"),
            "elapsed_s": time.time() - start,
            "formal_large_scale_training": False,
            "test_locked_used": False,
        }
        (output_dir / "TRAINING_COMPLETE.json").write_text(
            json.dumps(completion, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(completion, sort_keys=True), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
