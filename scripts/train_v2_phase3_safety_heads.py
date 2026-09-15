#!/usr/bin/env python3
"""Calibrate only Architecture-v2 visibility/stop heads on recovery states."""
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
    load_recovery_samples,
    recovery_batch,
    recovery_inputs,
    resolve,
    sha256,
    verify_manifest,
)
from train_v2_phase3_observable import (
    apply_observability,
    observable_metrics,
    verify_observability_audit,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def state_subset_sha256(
    state: Mapping[str, Any], excluded_prefixes: tuple[str, ...]
) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        if name.startswith(excluded_prefixes):
            continue
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main() -> int:
    args = arguments()
    config_path = args.config.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite safety calibration: {output_dir}")

    import torch
    import yaml
    from torch.nn import functional as F

    from omtrackvla.models.end_to_end import (
        ArchitectureV1Ablation,
        ArchitectureV1Config,
        load_official_da3_small_l11,
    )
    from omtrackvla.models.end_to_end_v2 import (
        ArchitectureV2DecoderConfig,
        ArchitectureV2FollowPolicy,
    )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (
        not isinstance(config, Mapping)
        or config.get("method") != "architecture_v2_evt_perception_polar"
        or config.get("formal_large_scale_training") is not False
        or config.get("test_locked_used") is not False
    ):
        raise ValueError("safety calibration config provenance mismatch")
    repository = config_path.parents[2]
    for field in ("da3_source", "da3_runtime"):
        extra = resolve(repository, config[field]).resolve(strict=True)
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    architecture = ArchitectureV1Config(**config["architecture"])
    decoder = ArchitectureV2DecoderConfig(**config["decoder"])
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("safety calibration requires CUDA")
    torch.cuda.set_device(device)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_manifest_path = resolve(repository, config["train_manifest"]).resolve(
        strict=True
    )
    val_manifest_path = resolve(
        repository, config["recovery_val_manifest"]
    ).resolve(strict=True)
    train_manifest = verify_manifest(train_manifest_path)
    val_manifest = verify_manifest(val_manifest_path)
    train_audit_path = resolve(
        repository, config["train_observability_audit"]
    ).resolve(strict=True)
    val_audit_path = resolve(
        repository, config["recovery_val_observability_audit"]
    ).resolve(strict=True)
    train_audit = verify_observability_audit(
        train_audit_path, train_manifest_path, "train"
    )
    val_audit = verify_observability_audit(
        val_audit_path, val_manifest_path, "recovery_val"
    )
    train_samples, train_metadata = load_recovery_samples(
        train_manifest, architecture, torch
    )
    val_samples, val_metadata = load_recovery_samples(
        val_manifest, architecture, torch
    )
    apply_observability(train_samples, train_metadata, train_audit, torch)
    apply_observability(val_samples, val_metadata, val_audit, torch)

    da3, da3_loading = load_official_da3_small_l11(
        resolve(repository, config["da3_model"]),
        architecture,
        ArchitectureV1Ablation(backbone_tuning="adapter"),
    )
    policy = ArchitectureV2FollowPolicy(da3, architecture, decoder).to(device)
    base_path = resolve(repository, config["base_checkpoint"]).resolve(strict=True)
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    if base.get("method") != config["method"] or base.get("test_locked_used") is not False:
        raise ValueError("safety calibration base checkpoint provenance mismatch")
    gradient_path = resolve(repository, config["base_gradient_report"]).resolve(
        strict=True
    )
    gradient_report = json.loads(gradient_path.read_text(encoding="utf-8"))
    if gradient_report.get("status") != "passed" or gradient_report.get("loss_source") != "waypoint_only":
        raise ValueError("base checkpoint lacks waypoint-only gradient admission")
    policy.load_state_dict(base["model"], strict=True)

    allowed_prefixes = tuple(str(value) for value in config["allowed_trainable_prefixes"])
    expected_prefixes = (
        "trajectory.visibility_head.",
        "trajectory.stop_head.",
    )
    if allowed_prefixes != expected_prefixes:
        raise ValueError("safety calibration trainable prefixes changed")
    trainable = []
    trainable_names = []
    for name, parameter in policy.named_parameters():
        allowed = name.startswith(allowed_prefixes)
        parameter.requires_grad_(allowed)
        if allowed:
            trainable.append(parameter)
            trainable_names.append(name)
    if not trainable or not any(
        name.startswith("trajectory.visibility_head.") for name in trainable_names
    ) or not any(name.startswith("trajectory.stop_head.") for name in trainable_names):
        raise RuntimeError("safety-head optimizer partition is incomplete")
    base_frozen_sha = state_subset_sha256(base["model"], allowed_prefixes)

    training = config["training"]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    threshold = float(training["deployment_visibility_threshold"])
    boundary = math.log(threshold / (1.0 - threshold))
    maximum_steps = int(training["max_steps"])
    evaluate_every = int(training["evaluate_every_steps"])
    use_bfloat16 = bool(training["bfloat16"]) and torch.cuda.is_bf16_supported()
    pools = {
        "visible": [
            index for index, sample in enumerate(train_samples) if bool(sample["visible"])
        ],
        "recent_loss": [
            index for index, sample in enumerate(train_samples)
            if not bool(sample["visible"]) and not bool(sample["stop"])
        ],
        "safe_stop": [
            index for index, sample in enumerate(train_samples) if bool(sample["stop"])
        ],
    }
    if any(not values for values in pools.values()):
        raise ValueError(f"safety calibration sampling pool is empty: {pools}")

    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output_dir / "config.yaml")
    start = time.time()

    def evaluate(step: int) -> dict[str, Any]:
        value = {
            "calibration_step": step,
            "recovery_val": observable_metrics(
                policy, val_samples, val_metadata, threshold, device, torch
            ),
            "test_locked_used": False,
        }
        (output_dir / f"EVAL_STEP_{step:03d}.json").write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({"event": "evaluation", **value}, sort_keys=True), flush=True)
        return value

    baseline = evaluate(0)
    evaluations = [baseline]
    checkpoints: dict[int, Path] = {}
    log_path = output_dir / "train_log.jsonl"
    pool_names = tuple(pools)
    for step in range(maximum_steps):
        policy.eval()
        optimizer.zero_grad(set_to_none=True)
        pool_name = pool_names[step % len(pool_names)]
        sample_index = random.choice(pools[pool_name])
        batch = recovery_batch(train_samples, [sample_index], device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
            outputs = policy(**recovery_inputs(batch))
            shifted_visibility = outputs["visibility_logit"].squeeze(-1).float() - boundary
            visibility_loss = F.binary_cross_entropy_with_logits(
                shifted_visibility, batch["visible"].float()
            )
            stop_loss = F.binary_cross_entropy_with_logits(
                outputs["stop_logit"].squeeze(-1).float(), batch["stop"].float()
            )
            total = (
                float(training["visibility_weight"]) * visibility_loss
                + float(training["stop_weight"]) * stop_loss
            )
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(training["gradient_clip_norm"])
        )
        optimizer.step()
        calibration_step = step + 1
        if calibration_step == 1 or calibration_step % 16 == 0:
            record = {
                "calibration_step": calibration_step,
                "sampling_pool": pool_name,
                "sample_id": train_metadata[sample_index]["sample_id"],
                "visibility_loss": float(visibility_loss.detach().cpu()),
                "stop_loss": float(stop_loss.detach().cpu()),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "elapsed_s": time.time() - start,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
        if calibration_step % evaluate_every == 0:
            checkpoint = {
                "schema_version": 1,
                "phase": 3,
                "method": config["method"],
                "stage": config["stage"],
                "global_step": base.get("global_step"),
                "calibration_step": calibration_step,
                "model": policy.state_dict(),
                "base_checkpoint_sha256": sha256(base_path),
                "frozen_parameter_sha256": base_frozen_sha,
                "allowed_changed_prefixes": list(allowed_prefixes),
                "formal_large_scale_training": False,
                "test_locked_used": False,
            }
            path = output_dir / "checkpoints" / f"step_{calibration_step:07d}.ckpt"
            atomic_torch_save(torch, path, checkpoint)
            checkpoints[calibration_step] = path
            evaluations.append(evaluate(calibration_step))

    current_frozen_sha = state_subset_sha256(policy.state_dict(), allowed_prefixes)
    if current_frozen_sha != base_frozen_sha:
        raise RuntimeError("safety calibration changed a frozen model tensor")
    gate = config["selection_gate"]
    base_metrics = baseline["recovery_val"]
    false_visible_max = max(
        0,
        base_metrics["false_visible_steps"]
        - int(gate["minimum_false_visible_reduction"]),
    )
    false_invisible_max = base_metrics["false_invisible_steps"] + int(
        gate["maximum_false_invisible_increase"]
    )
    drift_max = float(gate["maximum_waypoint_metric_drift_m"])
    candidates = []
    for evaluation in evaluations[1:]:
        metrics = evaluation["recovery_val"]
        waypoint_drift = max(
            abs(metrics[name] - base_metrics[name])
            for name in (
                "ade_m",
                "fde_m",
                "expert_waypoint_ade_m",
                "safe_stop_ade_m",
                "safe_stop_predicted_path_m",
            )
        )
        passed = (
            metrics["false_visible_steps"] <= false_visible_max
            and metrics["false_invisible_steps"] <= false_invisible_max
            and metrics["stop_accuracy"] >= float(gate["minimum_stop_accuracy"])
            and waypoint_drift <= drift_max
        )
        candidates.append(
            {
                "calibration_step": evaluation["calibration_step"],
                "passed": passed,
                "waypoint_metric_max_abs_drift_m": waypoint_drift,
                "recovery_val": metrics,
            }
        )
    passed_candidates = [item for item in candidates if item["passed"]]
    selected = min(passed_candidates, key=lambda item: item["calibration_step"]) if passed_candidates else None
    best_path = None
    if selected is not None:
        best_path = output_dir / "checkpoints" / "best.ckpt"
        shutil.copy2(checkpoints[selected["calibration_step"]], best_path)
    result = {
        "status": "pilot_complete",
        "selected": selected is not None,
        "selected_calibration_step": selected["calibration_step"] if selected else None,
        "best_checkpoint": str(best_path) if best_path else None,
        "best_checkpoint_sha256": sha256(best_path) if best_path else None,
        "baseline": baseline,
        "candidates": candidates,
        "selection_gate": {
            "false_visible_steps_allowed_max": false_visible_max,
            "false_invisible_steps_allowed_max": false_invisible_max,
            "maximum_waypoint_metric_drift_m": drift_max,
            "minimum_stop_accuracy": float(gate["minimum_stop_accuracy"]),
        },
        "trainable_parameter_names": trainable_names,
        "frozen_parameter_sha256_before": base_frozen_sha,
        "frozen_parameter_sha256_after": current_frozen_sha,
        "frozen_parameters_bitwise_unchanged": True,
        "da3_pretrained_loading": da3_loading,
        "base_waypoint_gradient_report": str(gradient_path),
        "base_waypoint_gradient_report_sha256": sha256(gradient_path),
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
