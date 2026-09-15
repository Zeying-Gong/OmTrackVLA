#!/usr/bin/env python3
"""Train a small observable-target Phase-3 pilot with independent recovery validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from train_v2_phase3_model_visited import (
    atomic_torch_save,
    clean_metrics,
    load_recovery_samples,
    recovery_batch,
    recovery_inputs,
    resolve,
    sha256,
    verify_manifest,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def verify_observability_audit(
    path: Path, manifest_path: Path, partition: str
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(value)
    expected = unsigned.pop("audit_sha256", None)
    actual = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if expected != actual:
        raise ValueError("observability audit checksum mismatch")
    if (
        value.get("stage") != "architecture_v2_phase3_observability_audit_v1"
        or value.get("status") != "passed"
        or value.get("partition") != partition
        or Path(value["source_manifest"]).resolve() != manifest_path.resolve()
        or value.get("source_manifest_sha256") != sha256(manifest_path)
        or value.get("test_locked_used") is not False
    ):
        raise ValueError("observability audit admission mismatch")
    return value


def apply_observability(
    samples: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    audit: Mapping[str, Any],
    torch: Any,
) -> None:
    by_id = {item["sample_id"]: item for item in audit["samples"]}
    if set(by_id) != {item["sample_id"] for item in metadata}:
        raise ValueError("observability audit sample set mismatch")
    for sample, item in zip(samples, metadata):
        record = by_id[item["sample_id"]]
        mode = record["waypoint_supervision"]
        if mode not in {"expert_waypoint", "safe_stop"}:
            raise ValueError("unknown waypoint observability mode")
        if mode == "safe_stop":
            if record["target_observable_in_model_input"] is not False or any(
                record["history_gt_visible"]
            ):
                raise ValueError("unsafe safe-stop observability record")
            sample["waypoints"] = torch.zeros_like(sample["waypoints"])
        sample["stop"] = torch.tensor(float(mode == "safe_stop"))
        item["waypoint_supervision"] = mode


def observable_loss(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    weights: Mapping[str, float],
    threshold: float,
    torch: Any,
    F: Any,
) -> tuple[Any, dict[str, Any]]:
    mask = batch["waypoint_mask"].bool().clone()
    mask[:, 0] = False
    expanded = mask[..., None].expand_as(outputs["waypoints"])
    waypoint = F.smooth_l1_loss(
        outputs["waypoints"], batch["waypoints"], reduction="none"
    )[expanded].mean()
    stop = F.binary_cross_entropy_with_logits(
        outputs["stop_logit"].squeeze(-1), batch["stop"]
    )
    logits = outputs["visibility_logit"].squeeze(-1).float()
    boundary = math.log(threshold / (1.0 - threshold))
    visible = batch["visible"] > 0.5
    visibility_margin = torch.where(
        visible,
        F.softplus(logits.new_tensor(boundary) - logits),
        F.softplus(logits - logits.new_tensor(boundary)),
    ).mean()
    bbox = (
        F.smooth_l1_loss(outputs["bbox_pred"][visible], batch["bbox"][visible])
        if visible.any()
        else outputs["bbox_pred"].sum() * 0.0
    )
    polar_valid = batch["polar_valid"].bool()
    if polar_valid.any():
        predicted_angle = F.normalize(
            outputs["target_angle_sincos"][polar_valid].float(), dim=1
        )
        target_angle = F.normalize(batch["angle"][polar_valid].float(), dim=1)
        angle = (1.0 - (predicted_angle * target_angle).sum(dim=1)).mean()
        distance = F.smooth_l1_loss(
            torch.log1p(outputs["target_distance_m"].squeeze(-1)[polar_valid]),
            torch.log1p(batch["distance"][polar_valid]),
        )
    else:
        angle = outputs["target_angle_sincos"].sum() * 0.0
        distance = outputs["target_distance_m"].sum() * 0.0
    values = {
        "waypoint": waypoint,
        "stop": stop,
        "visibility_margin": visibility_margin,
        "bbox": bbox,
        "angle": angle,
        "distance": distance,
    }
    total = sum(float(weights.get(name, 0.0)) * value for name, value in values.items())
    values["total"] = total
    return total, values


def observable_metrics(
    policy: Any,
    samples: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    threshold: float,
    device: Any,
    torch: Any,
) -> dict[str, Any]:
    totals = {
        "ade": 0.0,
        "fde": 0.0,
        "expert_ade": 0.0,
        "expert_count": 0,
        "safe_ade": 0.0,
        "safe_path": 0.0,
        "safe_count": 0,
        "false_visible": 0,
        "false_invisible": 0,
        "visible": 0,
        "invisible": 0,
        "stop_correct": 0,
        "bbox_sum": 0.0,
        "bbox_count": 0,
    }
    policy.eval()
    use_bfloat16 = torch.cuda.is_bf16_supported()
    with torch.inference_mode():
        for start in range(0, len(samples), 2):
            indices = list(range(start, min(start + 2, len(samples))))
            batch = recovery_batch(samples, indices, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
                outputs = policy(**recovery_inputs(batch))
            predicted = outputs["waypoints"].float()
            target = batch["waypoints"].float()
            error = torch.linalg.vector_norm(predicted[:, 1:] - target[:, 1:], dim=-1)
            path_length = torch.linalg.vector_norm(
                predicted[:, 1:] - predicted[:, :-1], dim=-1
            ).sum(dim=-1)
            totals["ade"] += float(error.sum())
            totals["fde"] += float(error[:, -1].sum())
            predicted_visible = torch.sigmoid(
                outputs["visibility_logit"].squeeze(-1).float()
            ) >= threshold
            visible = batch["visible"].bool()
            totals["false_visible"] += int(((~visible) & predicted_visible).sum())
            totals["false_invisible"] += int((visible & (~predicted_visible)).sum())
            totals["visible"] += int(visible.sum())
            totals["invisible"] += int((~visible).sum())
            predicted_stop = torch.sigmoid(outputs["stop_logit"].squeeze(-1).float()) >= 0.5
            totals["stop_correct"] += int((predicted_stop == batch["stop"].bool()).sum())
            if visible.any():
                bbox_error = (outputs["bbox_pred"][visible] - batch["bbox"][visible]).abs()
                totals["bbox_sum"] += float(bbox_error.sum())
                totals["bbox_count"] += int(bbox_error.numel())
            for local_index, global_index in enumerate(indices):
                ade_sum = float(error[local_index].sum())
                if metadata[global_index]["waypoint_supervision"] == "safe_stop":
                    totals["safe_ade"] += ade_sum
                    totals["safe_path"] += float(path_length[local_index])
                    totals["safe_count"] += 1
                else:
                    totals["expert_ade"] += ade_sum
                    totals["expert_count"] += 1
    count = max(1, len(samples))
    return {
        "samples": len(samples),
        "ade_m": totals["ade"] / (count * 7.0),
        "fde_m": totals["fde"] / count,
        "expert_waypoint_ade_m": totals["expert_ade"]
        / max(1, totals["expert_count"] * 7.0),
        "safe_stop_ade_m": totals["safe_ade"] / max(1, totals["safe_count"] * 7.0),
        "safe_stop_predicted_path_m": totals["safe_path"] / max(1, totals["safe_count"]),
        "expert_waypoint_samples": totals["expert_count"],
        "safe_stop_samples": totals["safe_count"],
        "false_visible_steps": totals["false_visible"],
        "false_invisible_steps": totals["false_invisible"],
        "visible_samples": totals["visible"],
        "invisible_samples": totals["invisible"],
        "visibility_accuracy_at_deployment_threshold": 1.0
        - (totals["false_visible"] + totals["false_invisible"]) / count,
        "stop_accuracy": totals["stop_correct"] / count,
        "bbox_mae_normalized": totals["bbox_sum"] / max(1, totals["bbox_count"]),
    }


def checkpoint_payload(
    policy: Any,
    optimizer: Any,
    scheduler: Any,
    config: Mapping[str, Any],
    step: int,
    parent_sha: str,
    train_manifest_sha: str,
    val_manifest_sha: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": 3,
        "method": config["method"],
        "stage": config["stage"],
        "global_step": step,
        "model": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "parent_checkpoint_sha256": parent_sha,
        "train_manifest_sha256": train_manifest_sha,
        "recovery_val_manifest_sha256": val_manifest_sha,
        "formal_large_scale_training": False,
        "test_locked_used": False,
    }


def main() -> int:
    args = arguments()
    config_path = args.config.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite observable Phase-3 pilot: {output_dir}")

    import torch
    import yaml
    from torch.nn import functional as F
    from torch.utils.data._utils.collate import default_collate

    from omtrackvla.data.end_to_end_training import Sage3DEndToEndSequenceDataset
    from omtrackvla.models.end_to_end import (
        ArchitectureV1Ablation,
        ArchitectureV1Config,
        load_official_da3_small_l11,
    )
    from omtrackvla.models.end_to_end_v2 import (
        ArchitectureV2DecoderConfig,
        ArchitectureV2FollowPolicy,
        parameter_inventory_v2,
        waypoint_gradient_report_v2,
    )
    from omtrackvla.training.end_to_end_v2 import _losses as clean_loss_fn
    from omtrackvla.training.end_to_end_v2 import _step_inputs as clean_step_inputs
    from omtrackvla.training.end_to_end_v2 import _targets as clean_targets

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (
        not isinstance(config, Mapping)
        or config.get("method") != "architecture_v2_evt_perception_polar"
        or config.get("formal_large_scale_training") is not False
        or config.get("test_locked_used") is not False
    ):
        raise ValueError("observable Phase-3 config provenance mismatch")
    repository = config_path.parents[2]
    for field in ("da3_source", "da3_runtime"):
        extra = resolve(repository, config[field]).resolve(strict=True)
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    architecture = ArchitectureV1Config(**config["architecture"])
    decoder = ArchitectureV2DecoderConfig(**config["decoder"])
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("observable Phase-3 pilot requires CUDA")
    torch.cuda.set_device(device)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")

    train_manifest_path = resolve(repository, config["train_manifest"]).resolve(strict=True)
    val_manifest_path = resolve(repository, config["recovery_val_manifest"]).resolve(strict=True)
    train_manifest = verify_manifest(train_manifest_path)
    val_manifest = verify_manifest(val_manifest_path)
    train_audit_path = resolve(repository, config["train_observability_audit"]).resolve(strict=True)
    val_audit_path = resolve(repository, config["recovery_val_observability_audit"]).resolve(strict=True)
    train_audit = verify_observability_audit(train_audit_path, train_manifest_path, "train")
    val_audit = verify_observability_audit(val_audit_path, val_manifest_path, "recovery_val")
    train_samples, train_metadata = load_recovery_samples(train_manifest, architecture, torch)
    val_samples, val_metadata = load_recovery_samples(val_manifest, architecture, torch)
    apply_observability(train_samples, train_metadata, train_audit, torch)
    apply_observability(val_samples, val_metadata, val_audit, torch)

    modes = tuple(config["data"]["clean_condition_modes"])
    sequence_index = resolve(repository, config["data"]["sequence_index"])
    clean_train = Sage3DEndToEndSequenceDataset(
        sequence_index,
        split="train",
        config=architecture,
        modes=modes,
        history_stride_raw=decoder.history_stride_raw,
    )
    clean_val = Sage3DEndToEndSequenceDataset(
        sequence_index,
        split="viz_val",
        config=architecture,
        modes=modes,
        history_stride_raw=decoder.history_stride_raw,
    )
    da3, da3_loading = load_official_da3_small_l11(
        resolve(repository, config["da3_model"]),
        architecture,
        ArchitectureV1Ablation(backbone_tuning="adapter"),
    )
    policy = ArchitectureV2FollowPolicy(da3, architecture, decoder).to(device)
    parent_path = resolve(repository, config["parent_checkpoint"]).resolve(strict=True)
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    if parent.get("method") != config["method"] or parent.get("test_locked_used") is not False:
        raise ValueError("observable Phase-3 parent provenance mismatch")
    policy.load_state_dict(parent["model"], strict=True)
    for name, parameter in policy.da3.named_parameters():
        parameter.requires_grad_(".adapter." in name)
    training = config["training"]
    adapter_parameters, polar_parameters, policy_parameters = [], [], []
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".adapter." in name:
            adapter_parameters.append(parameter)
        elif name.startswith("trajectory.target_polar_head."):
            polar_parameters.append(parameter)
        else:
            policy_parameters.append(parameter)
    if not adapter_parameters or not polar_parameters or not policy_parameters:
        raise RuntimeError("observable optimizer partition is incomplete")
    optimizer = torch.optim.AdamW(
        [
            {"params": policy_parameters, "lr": float(training["policy_learning_rate"]), "name": "policy"},
            {"params": polar_parameters, "lr": float(training["polar_learning_rate"]), "name": "polar"},
            {"params": adapter_parameters, "lr": float(training["adapter_learning_rate"]), "name": "da3_adapter"},
        ],
        weight_decay=float(training["weight_decay"]),
    )
    maximum_steps = int(training["max_steps"])
    warmup = int(training["warmup_steps"])

    def schedule(step: int) -> float:
        if step < warmup:
            return max(1.0e-3, (step + 1) / max(1, warmup))
        progress = (step - warmup) / max(1, maximum_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    use_bfloat16 = bool(training["bfloat16"]) and torch.cuda.is_bf16_supported()
    clean_weights = {str(k): float(v) for k, v in training["clean_loss_weights"].items()}
    recovery_weights = {str(k): float(v) for k, v in training["recovery_loss_weights"].items()}
    threshold = float(training["deployment_visibility_threshold"])
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output_dir / "config.yaml")
    shutil.copy2(train_manifest_path, output_dir / "train_manifest.json")
    shutil.copy2(val_manifest_path, output_dir / "recovery_val_manifest.json")
    shutil.copy2(train_audit_path, output_dir / "train_observability_audit.json")
    shutil.copy2(val_audit_path, output_dir / "recovery_val_observability_audit.json")
    start = time.time()

    def evaluate(step: int) -> dict[str, Any]:
        value = {
            "global_step": step,
            "train_recovery": observable_metrics(
                policy, train_samples, train_metadata, threshold, device, torch
            ),
            "recovery_val": observable_metrics(
                policy, val_samples, val_metadata, threshold, device, torch
            ),
            "clean_sage3d_val": clean_metrics(
                policy,
                clean_val,
                int(config["data"]["clean_validation_samples"]),
                device,
                torch,
                default_collate,
                clean_step_inputs,
                clean_targets,
            ),
            "test_locked_used": False,
        }
        (output_dir / f"EVAL_STEP_{step:03d}.json").write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({"event": "evaluation", **value}, sort_keys=True), flush=True)
        return value

    baseline = evaluate(0)
    # Waypoint-only deployment gradient checks remain separate from auxiliaries.
    policy.train()
    policy.da3.eval()
    expert_index = next(
        index for index, item in enumerate(train_metadata)
        if item["waypoint_supervision"] == "expert_waypoint"
    )
    probe = recovery_batch(train_samples, [expert_index], device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
        output = policy(**recovery_inputs(probe))
        _, values = observable_loss(
            output, probe, {"waypoint": 1.0}, threshold, torch, F
        )
    values["waypoint"].backward()
    recovery_gradients = waypoint_gradient_report_v2(policy)
    required = (
        "da3_adapter", "l11_projector", "scene_resampler",
        "context_encoder", "trajectory_decoder", "delta_head",
    )
    missing = [name for name in required if recovery_gradients.get(name, 0.0) <= 0.0]
    if missing:
        raise RuntimeError(f"observable waypoint-only gradient is zero: {missing}")
    optimizer.zero_grad(set_to_none=True)
    uwb_probe = default_collate([clean_train[0]])
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
        output = policy(**clean_step_inputs(uwb_probe, device))
        target = clean_targets(uwb_probe, device)
        _, values = clean_loss_fn(output, target, {"waypoint": 1.0})
    values["waypoint"].backward()
    uwb_gradients = waypoint_gradient_report_v2(policy)
    if uwb_gradients.get("uwb_encoder", 0.0) <= 0.0:
        raise RuntimeError("observable clean waypoint loss missed UWB encoder")
    optimizer.zero_grad(set_to_none=True)
    (output_dir / "WAYPOINT_ONLY_BACKWARD.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "recovery_expert_waypoint": recovery_gradients,
                "clean_visual_uwb": uwb_gradients,
                "required_nonzero": list(required),
                "loss_source": "waypoint_only",
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    parent_sha = sha256(parent_path)
    train_manifest_sha = sha256(train_manifest_path)
    val_manifest_sha = sha256(val_manifest_path)
    run_manifest = {
        "schema_version": 1,
        "stage": config["stage"],
        "method": config["method"],
        "formal_large_scale_training": False,
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_sha256": parent_sha,
        "train_samples": len(train_samples),
        "train_task_counts": train_manifest["task_counts"],
        "recovery_val_samples": len(val_samples),
        "recovery_val_task_counts": val_manifest["task_counts"],
        "train_supervision_counts": train_audit["supervision_counts"],
        "recovery_val_supervision_counts": val_audit["supervision_counts"],
        "clean_weight": float(training["clean_weight"]),
        "recovery_weight": float(training["recovery_weight"]),
        "deployment_visibility_threshold": threshold,
        "loss_weights": {"clean": clean_weights, "recovery": recovery_weights},
        "da3_pretrained_loading": da3_loading,
        "parameter_inventory": parameter_inventory_v2(policy),
        "optimizer_parameter_tensors": {
            "policy": len(policy_parameters),
            "polar": len(polar_parameters),
            "da3_adapter": len(adapter_parameters),
        },
        "maximum_optimizer_steps": maximum_steps,
        "history": "8 frames at 0.1-second spacing",
        "gpu_zero_used": False,
        "test_locked_used": False,
    }
    (output_dir / "RUN_MANIFEST.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    clean_weight = float(training["clean_weight"])
    recovery_weight = float(training["recovery_weight"])
    log_path = output_dir / "train_log.jsonl"
    evaluations = [baseline]
    checkpoint_paths: dict[int, Path] = {}
    for step in range(maximum_steps):
        policy.train()
        policy.da3.eval()
        optimizer.zero_grad(set_to_none=True)
        mode_index = step % len(modes)
        pool_size = (clean_train.base_length - 1 - mode_index) // len(modes) + 1
        clean_index = mode_index + len(modes) * random.randrange(pool_size)
        batch = default_collate([clean_train[clean_index]])
        target = clean_targets(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
            output = policy(**clean_step_inputs(batch, device))
            clean_total, clean_values = clean_loss_fn(output, target, clean_weights)
        (clean_weight * clean_total).backward()
        del output, clean_total
        recovery_index = step % len(train_samples)
        batch = recovery_batch(train_samples, [recovery_index], device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
            output = policy(**recovery_inputs(batch))
            recovery_total, recovery_values = observable_loss(
                output, batch, recovery_weights, threshold, torch, F
            )
        (recovery_weight * recovery_total).backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            policy_parameters + polar_parameters + adapter_parameters,
            float(training["gradient_clip_norm"]),
        )
        optimizer.step()
        scheduler.step()
        global_step = step + 1
        if global_step == 1 or global_step % int(training["log_every_steps"]) == 0:
            record = {
                "global_step": global_step,
                "clean_mode": modes[mode_index],
                "recovery_sample_id": train_metadata[recovery_index]["sample_id"],
                "recovery_waypoint_supervision": train_metadata[recovery_index]["waypoint_supervision"],
                "clean_losses": {name: float(value.detach().float().cpu()) for name, value in clean_values.items()},
                "recovery_losses": {name: float(value.detach().float().cpu()) for name, value in recovery_values.items()},
                "gradient_norm": float(gradient_norm.detach().float().cpu()),
                "learning_rates": {group["name"]: float(group["lr"]) for group in optimizer.param_groups},
                "elapsed_s": time.time() - start,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
        if global_step % int(training["evaluate_every_steps"]) == 0:
            path = output_dir / "checkpoints" / f"step_{global_step:07d}.ckpt"
            atomic_torch_save(
                torch,
                path,
                checkpoint_payload(
                    policy,
                    optimizer,
                    scheduler,
                    config,
                    global_step,
                    parent_sha,
                    train_manifest_sha,
                    val_manifest_sha,
                ),
            )
            checkpoint_paths[global_step] = path
            evaluations.append(evaluate(global_step))

    gate = config["selection_gate"]
    baseline_val = baseline["recovery_val"]
    baseline_clean = baseline["clean_sage3d_val"]
    val_ade_max = baseline_val["ade_m"] * (
        1.0 - float(gate["minimum_recovery_val_ade_relative_improvement"])
    )
    false_visible_max = max(
        0,
        baseline_val["false_visible_steps"]
        - int(gate["minimum_false_visible_reduction"]),
    )
    false_invisible_max = baseline_val["false_invisible_steps"] + int(
        gate["maximum_false_invisible_increase"]
    )
    clean_ade_max = baseline_clean["ade_m"] * (
        1.0 + float(gate["maximum_clean_ade_relative_regression"])
    ) + float(gate["maximum_clean_ade_absolute_slack_m"])
    expert_ade_max = baseline_val["expert_waypoint_ade_m"] * (
        1.0 + float(gate.get("maximum_expert_ade_relative_regression", math.inf))
    ) + float(gate.get("maximum_expert_ade_absolute_slack_m", 0.0))
    safe_path_max = baseline_val["safe_stop_predicted_path_m"] * (
        1.0 - float(gate.get("minimum_safe_stop_path_relative_improvement", 0.0))
    )
    candidates = []
    for evaluation in evaluations[1:]:
        val = evaluation["recovery_val"]
        clean = evaluation["clean_sage3d_val"]
        passed = (
            val["ade_m"] <= val_ade_max
            and val["false_visible_steps"] <= false_visible_max
            and val["false_invisible_steps"] <= false_invisible_max
            and val["expert_waypoint_ade_m"] <= expert_ade_max
            and val["safe_stop_predicted_path_m"] <= safe_path_max
            and clean["ade_m"] <= clean_ade_max
        )
        candidates.append(
            {
                "global_step": evaluation["global_step"],
                "passed": passed,
                "recovery_val": val,
                "clean_sage3d_val": clean,
            }
        )
    passed_candidates = [item for item in candidates if item["passed"]]
    selected = min(passed_candidates, key=lambda item: item["recovery_val"]["ade_m"]) if passed_candidates else None
    best_path = None
    if selected is not None:
        best_path = output_dir / "checkpoints" / "best.ckpt"
        shutil.copy2(checkpoint_paths[selected["global_step"]], best_path)
    result = {
        "status": "pilot_complete",
        "selected": selected is not None,
        "selected_global_step": selected["global_step"] if selected else None,
        "best_checkpoint": str(best_path) if best_path else None,
        "best_checkpoint_sha256": sha256(best_path) if best_path else None,
        "baseline": baseline,
        "candidates": candidates,
        "selection_gate": {
            "recovery_val_ade_required_max_m": val_ade_max,
            "false_visible_steps_allowed_max": false_visible_max,
            "false_invisible_steps_allowed_max": false_invisible_max,
            "expert_waypoint_ade_allowed_max_m": expert_ade_max,
            "safe_stop_predicted_path_required_max_m": safe_path_max,
            "clean_ade_allowed_max_m": clean_ade_max,
        },
        "elapsed_s": time.time() - start,
        "formal_large_scale_training": False,
        "gpu_zero_used": False,
        "test_locked_used": False,
    }
    (output_dir / "PILOT_COMPLETE.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
