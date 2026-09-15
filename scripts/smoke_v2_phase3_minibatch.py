#!/usr/bin/env python3
"""Overfit a few audited model-visited v2 samples without saving a checkpoint."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from smoke_next007_phase3_sample import (
    _load_json,
    _render_comparison,
    _rgb_tensor,
    _safe_file,
    _sha256,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, action="append", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--policy-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--adapter-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260915)
    return parser.parse_args()


def validate_v2_sample(value: object, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("v2 Phase-3 sample schema mismatch")
    if value.get("stage") != "next007_phase3_model_visited_relabel_smoke":
        raise ValueError("v2 Phase-3 sample stage mismatch")
    if value.get("formal_training_eligible") is not True:
        raise ValueError("v2 Phase-3 smoke requires train-split samples")
    source = value.get("source")
    if not isinstance(source, Mapping) or (
        source.get("split") != "train" or source.get("test_locked_used") is not False
    ):
        raise ValueError("v2 Phase-3 sample provenance is not admitted")
    inputs = value.get("model_inputs")
    if not isinstance(inputs, Mapping) or inputs.get("condition_mode") != "visual_only":
        raise ValueError("v2 Phase-3 smoke requires visual-only model inputs")
    history = inputs.get("rgb_history")
    if not isinstance(history, list) or len(history) != 8:
        raise ValueError("v2 Phase-3 RGB history must contain eight frames")
    history_steps = [int(record["environment_step"]) for record in history]
    if any(second - first != 3 for first, second in zip(history_steps, history_steps[1:])):
        raise ValueError("v2 Phase-3 RGB history must use a three-step stride")
    if history_steps[-1] != int(source["anchor_environment_step"]):
        raise ValueError("v2 Phase-3 history does not end at its anchor")
    sampling = inputs.get("history_sampling")
    if not isinstance(sampling, Mapping) or (
        sampling.get("stride_environment_steps") != 3
    ):
        raise ValueError("v2 Phase-3 history sampling metadata is missing")
    supervision = value.get("supervision")
    expert = supervision.get("expert_trajectory") if isinstance(supervision, Mapping) else None
    target_state = supervision.get("target_state") if isinstance(supervision, Mapping) else None
    waypoints = np.asarray(
        expert.get("waypoints_base_xy_m") if isinstance(expert, Mapping) else None,
        dtype=np.float32,
    )
    if waypoints.shape != (8, 2) or not np.isfinite(waypoints).all():
        raise ValueError("v2 Phase-3 expert trajectory must be finite 8x2")
    if not isinstance(target_state, Mapping) or (
        target_state.get("gt_used_only_on_label_side") is not True
    ):
        raise ValueError("v2 Phase-3 target-state label boundary is missing")
    angle = np.asarray(target_state.get("angle_sincos"), dtype=np.float32)
    distance = float(target_state.get("distance_m", float("nan")))
    if angle.shape != (2,) or not np.isfinite(angle).all() or not math.isfinite(distance):
        raise ValueError("v2 Phase-3 Polar label is invalid")
    return value


def main() -> int:
    args = arguments()
    if len(args.sample) < 2 or args.steps <= 0:
        raise ValueError("v2 Phase-3 smoke requires at least two samples")
    config_path = args.config.expanduser().resolve(strict=True)
    checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite v2 Phase-3 smoke: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    import torch
    import torch.nn.functional as F
    import yaml

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping) or (
        config.get("method") != "architecture_v2_evt_perception_polar"
        or config.get("test_locked_used") is not False
    ):
        raise ValueError("v2 Phase-3 smoke requires the admitted Polar config")
    repository = config_path.parents[2]
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))

    def resolve(value: str | Path) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else repository / candidate

    for field in ("da3_source", "da3_runtime"):
        extra = resolve(config[field]).resolve(strict=True)
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))

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

    architecture = ArchitectureV1Config(**config["architecture"])
    decoder = ArchitectureV2DecoderConfig(**config["decoder"])
    ablation = ArchitectureV1Ablation(backbone_tuning="adapter")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("v2 Phase-3 smoke requires CUDA")
    torch.cuda.set_device(device)

    loaded = []
    tensors = []
    for unresolved in args.sample:
        path = unresolved.expanduser().resolve(strict=True)
        sample = validate_v2_sample(_load_json(path), path)
        root = path.parent.resolve(strict=True)
        inputs = sample["model_inputs"]
        initial = inputs["initial_rgb"]
        initial_path = _safe_file(root, initial["rgb_path"], initial["sha256"])
        history_paths = [
            _safe_file(root, record["rgb_path"], record["sha256"])
            for record in inputs["rgb_history"]
        ]
        uwb = inputs["uwb"]
        expert = sample["supervision"]["expert_trajectory"]
        state = sample["supervision"]["target_state"]
        bbox = state["bbox_xyxy_norm"] or [0.0, 0.0, 0.0, 0.0]
        tensors.append({
            "initial_rgb": _rgb_tensor(
                initial_path, architecture.image_height, architecture.image_width, torch
            ),
            "initial_bbox": torch.tensor(inputs["initial_bbox_xyxy_norm"], dtype=torch.float32),
            "ego_rgb": torch.stack([
                _rgb_tensor(image, architecture.image_height, architecture.image_width, torch)
                for image in history_paths
            ]),
            "uwb_xy": torch.tensor(uwb["relative_position_base_xy_m"], dtype=torch.float32),
            "uwb_covariance_xy": torch.tensor(uwb["covariance_base_xy_m2"], dtype=torch.float32),
            "uwb_quality": torch.tensor(float(uwb["quality_01"])),
            "uwb_age_s": torch.tensor(float(uwb["age_s"])),
            "waypoints": torch.tensor(expert["waypoints_base_xy_m"], dtype=torch.float32),
            "waypoint_mask": torch.tensor(expert["valid_mask"], dtype=torch.bool),
            "bbox": torch.tensor(bbox, dtype=torch.float32),
            "visible": torch.tensor(float(state["visible"])),
            "angle": torch.tensor(state["angle_sincos"], dtype=torch.float32),
            "distance": torch.tensor(float(state["distance_m"])),
            "polar_valid": torch.tensor(bool(state["polar_valid"])),
            "camera_intrinsics": torch.tensor(inputs["camera_intrinsics"], dtype=torch.float32),
            "camera_from_base": torch.tensor(inputs["camera_from_base"], dtype=torch.float32),
        })
        loaded.append({"path": path, "value": sample, "current_rgb": history_paths[-1]})

    sample_ids = [str(item["value"]["sample_id"]) for item in loaded]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("v2 Phase-3 smoke contains duplicate states")

    def stack(name: str) -> Any:
        return torch.stack([item[name] for item in tensors]).to(device)

    count = len(tensors)
    model_inputs = {
        "initial_rgb": stack("initial_rgb"),
        "initial_bbox": stack("initial_bbox"),
        "ego_rgb": stack("ego_rgb"),
        "visual_initialization_valid": torch.ones(count, device=device),
        "rgb_valid": torch.ones(count, device=device),
        "binding_valid": torch.ones(count, device=device),
        "uwb_xy": stack("uwb_xy"),
        "uwb_covariance_xy": stack("uwb_covariance_xy"),
        "uwb_quality": stack("uwb_quality"),
        "uwb_age_s": stack("uwb_age_s"),
        "uwb_valid": torch.zeros(count, device=device),
        "camera_intrinsics": stack("camera_intrinsics"),
        "camera_from_base": stack("camera_from_base"),
    }
    targets = {name: stack(name) for name in (
        "waypoints", "waypoint_mask", "bbox", "visible", "angle", "distance", "polar_valid"
    )}

    da3, da3_loading = load_official_da3_small_l11(
        resolve(config["da3_model"]), architecture, ablation,
        resolve(config["dinov2_model"]) if config.get("dinov2_model") else None,
    )
    policy = ArchitectureV2FollowPolicy(da3, architecture, decoder).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("method") != "architecture_v2_evt_perception_polar" or (
        checkpoint.get("test_locked_used") is not False
    ):
        raise ValueError("v2 Phase-3 parent checkpoint is not admitted")
    policy.load_state_dict(checkpoint["model"], strict=True)
    policy.train()
    use_bfloat16 = torch.cuda.is_bf16_supported()

    def compute_losses(output: Mapping[str, Any]) -> dict[str, Any]:
        expanded = targets["waypoint_mask"][..., None].expand_as(output["waypoints"])
        waypoint = F.smooth_l1_loss(
            output["waypoints"], targets["waypoints"], reduction="none"
        )[expanded].mean()
        visible = targets["visible"] > 0.5
        bbox = (
            F.smooth_l1_loss(output["bbox_pred"][visible], targets["bbox"][visible])
            if visible.any() else output["bbox_pred"].sum() * 0.0
        )
        visibility = F.binary_cross_entropy_with_logits(
            output["visibility_logit"].squeeze(-1), targets["visible"]
        )
        polar_valid = targets["polar_valid"].bool()
        if polar_valid.any():
            predicted_angle = F.normalize(
                output["target_angle_sincos"][polar_valid].float(), dim=1
            )
            target_angle = F.normalize(targets["angle"][polar_valid].float(), dim=1)
            angle = (1.0 - (predicted_angle * target_angle).sum(dim=1)).mean()
            distance = F.smooth_l1_loss(
                torch.log1p(output["target_distance_m"].squeeze(-1)[polar_valid]),
                torch.log1p(targets["distance"][polar_valid]),
            )
        else:
            angle = output["target_angle_sincos"].sum() * 0.0
            distance = output["target_distance_m"].sum() * 0.0
        total = waypoint + 0.1 * (bbox + visibility + angle + distance)
        return {
            "total": total,
            "waypoint": waypoint,
            "bbox": bbox,
            "visibility": visibility,
            "angle": angle,
            "distance": distance,
        }

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
        initial_output = policy(**model_inputs)
        initial_losses = compute_losses(initial_output)
    policy.zero_grad(set_to_none=True)
    initial_losses["waypoint"].backward()
    waypoint_gradients = waypoint_gradient_report_v2(policy)
    required = (
        "da3_adapter", "l11_projector", "scene_resampler",
        "context_encoder", "trajectory_decoder", "delta_head",
    )
    missing = [name for name in required if waypoint_gradients.get(name, 0.0) <= 0.0]
    if missing:
        raise RuntimeError(f"v2 Phase-3 waypoint loss has zero gradients: {missing}")

    adapter_parameters = []
    policy_parameters = []
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        (adapter_parameters if ".adapter." in name else policy_parameters).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": policy_parameters, "lr": args.policy_learning_rate},
        {"params": adapter_parameters, "lr": args.adapter_learning_rate},
    ], weight_decay=1.0e-4)
    history = [{name: float(value.detach().float().cpu()) for name, value in initial_losses.items()}]
    before = initial_output["waypoints"].detach().float().cpu().tolist()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
            output = policy(**model_inputs)
            losses = compute_losses(output)
        if not bool(torch.isfinite(losses["total"])):
            raise RuntimeError("v2 Phase-3 smoke loss became non-finite")
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for group in optimizer.param_groups for parameter in group["params"]], 1.0
        )
        optimizer.step()
        history.append({name: float(value.detach().float().cpu()) for name, value in losses.items()})

    policy.eval()
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=use_bfloat16
    ):
        final_output = policy(**model_inputs)
        final_losses = compute_losses(final_output)
    final_values = {
        name: float(value.detach().float().cpu()) for name, value in final_losses.items()
    }
    history.append(final_values)
    if final_values["total"] >= history[0]["total"]:
        raise RuntimeError("v2 Phase-3 smoke did not reduce its total loss")
    after = final_output["waypoints"].detach().float().cpu().tolist()
    expert = targets["waypoints"].detach().float().cpu().tolist()
    visuals = []
    for index, sample in enumerate(loaded):
        before_path = output_dir / f"sample_{index:02d}_before.png"
        after_path = output_dir / f"sample_{index:02d}_after.png"
        _render_comparison(sample["current_rgb"], before[index], expert[index], before_path)
        _render_comparison(sample["current_rgb"], after[index], expert[index], after_path)
        visuals.append({
            "before": str(before_path), "before_sha256": _sha256(before_path),
            "after": str(after_path), "after_sha256": _sha256(after_path),
        })
    report = {
        "schema_version": 1,
        "stage": "v2_phase3_model_visited_minibatch_smoke",
        "status": "passed",
        "samples": [{
            "sample_id": item["value"]["sample_id"],
            "path": str(item["path"]),
            "sha256": _sha256(item["path"]),
        } for item in loaded],
        "parent_checkpoint": str(checkpoint_path),
        "parent_checkpoint_sha256": _sha256(checkpoint_path),
        "optimizer_steps": args.steps,
        "checkpoint_saved": False,
        "formal_training_started": False,
        "loss_weights": {
            "waypoint": 1.0, "bbox": 0.1, "visibility": 0.1,
            "angle": 0.1, "distance": 0.1,
        },
        "loss_history": history,
        "waypoint_only_gradient_norms": waypoint_gradients,
        "waypoint_required_nonzero": list(required),
        "visualizations": visuals,
        "da3_pretrained_loading": da3_loading,
        "parameter_inventory": parameter_inventory_v2(policy),
        "input_audit": {
            "all_samples_train_split": True,
            "history": "8 frames at three Habitat steps (0.1 s)",
            "uwb_valid": False,
            "gt_target_state_used_only_as_training_label": True,
            "test_locked_used": False,
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "passed",
        "samples": len(loaded),
        "loss_initial": history[0],
        "loss_final": final_values,
        "waypoint_only_gradient_norms": waypoint_gradients,
        "checkpoint_saved": False,
        "report": str(report_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
