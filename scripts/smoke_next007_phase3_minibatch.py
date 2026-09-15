#!/usr/bin/env python3
"""Overfit a few train-split relabels in memory before formal Phase 3."""
from __future__ import annotations

import argparse
import json
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
    validate_sample,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, action="append", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--policy-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--adapter-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260913)
    return parser.parse_args()


def _stack_samples(
    sample_paths: list[Path], architecture: Any, torch: Any, device: Any
) -> tuple[dict[str, Any], Any, Any, list[dict[str, Any]]]:
    loaded: list[dict[str, Any]] = []
    tensors: list[dict[str, Any]] = []
    for unresolved in sample_paths:
        path = unresolved.expanduser().resolve(strict=True)
        sample = validate_sample(
            _load_json(path),
            path,
            architecture.history_size,
            expected_split="train",
            require_formal_eligible=True,
        )
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
        tensors.append({
            "initial_rgb": _rgb_tensor(
                initial_path, architecture.image_height, architecture.image_width, torch
            ),
            "initial_bbox": torch.tensor(
                inputs["initial_bbox_xyxy_norm"], dtype=torch.float32
            ),
            "ego_rgb": torch.stack([
                _rgb_tensor(
                    image, architecture.image_height, architecture.image_width, torch
                )
                for image in history_paths
            ]),
            "uwb_xy": torch.tensor(
                uwb["relative_position_base_xy_m"], dtype=torch.float32
            ),
            "uwb_covariance_xy": torch.tensor(
                uwb["covariance_base_xy_m2"], dtype=torch.float32
            ),
            "uwb_quality": torch.tensor(float(uwb["quality_01"])),
            "uwb_age_s": torch.tensor(float(uwb["age_s"])),
            "camera_intrinsics": torch.tensor(
                inputs["camera_intrinsics"], dtype=torch.float32
            ),
            "camera_from_base": torch.tensor(
                inputs["camera_from_base"], dtype=torch.float32
            ),
            "target": torch.tensor(
                expert["waypoints_base_xy_m"], dtype=torch.float32
            ),
            "mask": torch.tensor(expert["valid_mask"], dtype=torch.bool),
        })
        loaded.append({
            "path": path,
            "value": sample,
            "current_rgb": history_paths[-1],
        })

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
    return model_inputs, stack("target"), stack("mask"), loaded


def main() -> int:
    args = arguments()
    if len(args.sample) < 2 or args.steps <= 0:
        raise ValueError("NEXT-007 minibatch smoke requires >=2 samples and positive steps")
    config_path = args.config.expanduser().resolve(strict=True)
    checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import yaml

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (
        not isinstance(config, Mapping)
        or config.get("method") != "architecture_v1_end_to_end"
        or config.get("test_locked_used") is not False
    ):
        raise ValueError("NEXT-007 minibatch requires Architecture v1")
    repository = config_path.parents[2]
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))

    def resolve(value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else repository / path

    for field in ("da3_source", "da3_runtime"):
        extra = resolve(config[field]).resolve(strict=True)
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))

    from omtrackvla.models.end_to_end import (
        ArchitectureV1Ablation,
        ArchitectureV1Config,
        EndToEndFollowPolicy,
        load_official_da3_small_l11,
        waypoint_gradient_report,
        waypoint_only_loss,
    )

    architecture = ArchitectureV1Config(**config.get("architecture", {}))
    architecture.validate()
    ablation = ArchitectureV1Ablation(**config.get("ablation", {}))
    ablation.validate()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("NEXT-007 minibatch smoke requires CUDA")
    torch.cuda.set_device(device)
    model_inputs, target, mask, samples = _stack_samples(
        list(args.sample), architecture, torch, device
    )
    sample_ids = [str(sample["value"]["sample_id"]) for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("NEXT-007 minibatch contains duplicate model-visited states")
    da3, da3_loading = load_official_da3_small_l11(
        resolve(config["da3_model"]), architecture, ablation,
        resolve(config["dinov2_model"]) if config.get("dinov2_model") else None,
    )
    policy = EndToEndFollowPolicy(da3, architecture, ablation).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("phase") != 2
        or checkpoint.get("method") != "architecture_v1_end_to_end"
        or checkpoint.get("test_locked_used") is not False
    ):
        raise ValueError("NEXT-007 minibatch checkpoint is not admitted")
    policy.load_state_dict(checkpoint["model"], strict=True)
    policy.train()
    adapter_parameters = []
    policy_parameters = []
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        (adapter_parameters if ".adapter." in name else policy_parameters).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": policy_parameters, "lr": args.policy_learning_rate},
            {"params": adapter_parameters, "lr": args.adapter_learning_rate},
        ],
        weight_decay=1.0e-4,
    )
    use_bfloat16 = torch.cuda.is_bf16_supported()

    def forward_loss() -> tuple[Any, Any]:
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16
        ):
            output = policy(**model_inputs)
            loss = waypoint_only_loss(output, target, mask)
        return output, loss

    optimizer.zero_grad(set_to_none=True)
    initial_output, initial_loss = forward_loss()
    initial_loss.backward()
    gradients = waypoint_gradient_report(policy)
    required = ["fusion", "gru"]
    if ablation.backbone_tuning == "adapter":
        required.append("da3_adapter")
    missing = [name for name in required if gradients.get(name, 0.0) <= 0.0]
    if missing:
        raise RuntimeError(f"NEXT-007 minibatch has zero gradients: {missing}")
    before = initial_output["waypoints"].detach().float().cpu().tolist()
    losses = [float(initial_loss.detach().float().cpu())]

    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        _, loss = forward_loss()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("NEXT-007 minibatch loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for group in optimizer.param_groups for parameter in group["params"]],
            1.0,
        )
        optimizer.step()
        losses.append(float(loss.detach().float().cpu()))

    policy.eval()
    with torch.inference_mode():
        final_output, final_loss = forward_loss()
    final_value = float(final_loss.detach().float().cpu())
    losses.append(final_value)
    if final_value >= losses[0]:
        raise RuntimeError(
            f"NEXT-007 minibatch did not reduce loss: {losses[0]} -> {final_value}"
        )
    after = final_output["waypoints"].detach().float().cpu().tolist()
    expert = target.detach().float().cpu().tolist()
    visuals = []
    for index, sample in enumerate(samples):
        before_path = output_dir / f"sample_{index:02d}_before.png"
        after_path = output_dir / f"sample_{index:02d}_after.png"
        _render_comparison(sample["current_rgb"], before[index], expert[index], before_path)
        _render_comparison(sample["current_rgb"], after[index], expert[index], after_path)
        visuals.append({
            "before": str(before_path),
            "before_sha256": _sha256(before_path),
            "after": str(after_path),
            "after_sha256": _sha256(after_path),
        })
    report = {
        "schema_version": 1,
        "stage": "next007_phase3_train_minibatch_optimizer_smoke",
        "status": "passed",
        "samples": [
            {
                "path": str(sample["path"]),
                "sha256": _sha256(sample["path"]),
                "sample_id": sample["value"]["sample_id"],
                "split": sample["value"]["source"]["split"],
            }
            for sample in samples
        ],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "optimizer_steps": args.steps,
        "checkpoint_saved": False,
        "formal_training_started": False,
        "waypoint_only_loss_history": losses,
        "initial_gradient_norms": gradients,
        "required_nonzero": required,
        "before_waypoints": before,
        "after_waypoints": after,
        "expert_waypoints": expert,
        "visualizations": visuals,
        "da3_pretrained_loading": da3_loading,
        "input_audit": {
            "all_samples_train_split": True,
            "uwb_valid": False,
            "gt_target_pose_or_depth_used": False,
            "later_bbox_used": False,
            "test_locked_used": False,
        },
    }
    report_path = output_dir / "report.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    print(json.dumps({
        "status": "passed",
        "samples": len(samples),
        "loss_initial": losses[0],
        "loss_final": final_value,
        "gradient_norms": gradients,
        "optimizer_steps": args.steps,
        "checkpoint_saved": False,
        "report": str(report_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
