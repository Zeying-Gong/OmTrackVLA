#!/usr/bin/env python3
"""Run waypoint-only backward on one audited NEXT-007 relabel sample."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_file(root: Path, relative: str, expected_sha256: str) -> Path:
    path = (root / relative).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"sample image escaped its directory: {relative}") from error
    if _sha256(path) != expected_sha256:
        raise ValueError(f"sample image checksum mismatch: {relative}")
    return path


def validate_sample(
    value: object,
    sample_path: Path,
    history_size: int,
    *,
    expected_split: str = "val",
    require_formal_eligible: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("NEXT-007 sample schema mismatch")
    if value.get("stage") != "next007_phase3_model_visited_relabel_smoke":
        raise ValueError("NEXT-007 sample stage mismatch")
    if value.get("formal_training_eligible") is not require_formal_eligible:
        raise ValueError("NEXT-007 sample formal-training eligibility mismatch")
    source = value.get("source")
    if not isinstance(source, Mapping) or source.get("test_locked_used") is not False:
        raise ValueError("NEXT-007 sample must explicitly exclude test_locked")
    if source.get("split") != expected_split:
        raise ValueError(f"NEXT-007 smoke expected split={expected_split}")
    inputs = value.get("model_inputs")
    if not isinstance(inputs, Mapping) or inputs.get("condition_mode") != "visual_only":
        raise ValueError("NEXT-007 smoke must reproduce the no-UWB visual failure")
    history = inputs.get("rgb_history")
    if not isinstance(history, list) or len(history) != history_size:
        raise ValueError(
            f"NEXT-007 RGB history must contain exactly {history_size} frames"
        )
    steps = [int(record["environment_step"]) for record in history]
    if any(second != first + 1 for first, second in zip(steps, steps[1:])):
        raise ValueError("NEXT-007 RGB history steps must be contiguous")
    uwb = inputs.get("uwb")
    if not isinstance(uwb, Mapping) or uwb.get("valid") is not False:
        raise ValueError("NEXT-007 validation failure must not invent UWB input")
    supervision = value.get("supervision")
    expert = supervision.get("expert_trajectory") if isinstance(supervision, Mapping) else None
    waypoints = np.asarray(
        expert.get("waypoints_base_xy_m") if isinstance(expert, Mapping) else None,
        dtype=np.float32,
    )
    mask = np.asarray(
        expert.get("valid_mask") if isinstance(expert, Mapping) else None,
        dtype=bool,
    )
    if waypoints.shape != (8, 2) or mask.shape != (8,) or not (
        np.isfinite(waypoints).all() and mask.all()
    ):
        raise ValueError("NEXT-007 expert target must be a finite valid 8x2 path")
    if not np.allclose(waypoints[0], 0.0, atol=1e-7):
        raise ValueError("NEXT-007 expert waypoint zero must be the anchor")
    if supervision.get("gt_used_only_on_label_side") is not True:
        raise ValueError("NEXT-007 sample does not prove the GT label boundary")
    return value


def _rgb_tensor(path: Path, height: int, width: int, torch: Any) -> Any:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _render_comparison(
    current_rgb: Path,
    predicted: Sequence[Sequence[float]],
    expert: Sequence[Sequence[float]],
    output: Path,
) -> None:
    with Image.open(current_rgb) as source:
        rgb = source.convert("RGB").resize((384, 384))
    canvas = Image.new("RGB", (768, 384), "white")
    canvas.paste(rgb, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), "model input at failure state", fill=(255, 255, 255))
    origin = (576, 350)
    scale = 120.0
    draw.line((origin[0], 24, origin[0], origin[1]), fill=(190, 190, 190), width=1)
    draw.line((408, origin[1], 748, origin[1]), fill=(190, 190, 190), width=1)

    def points(values: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
        return [
            (
                int(round(origin[0] - float(left) * scale)),
                int(round(origin[1] - float(forward) * scale)),
            )
            for forward, left in values
        ]

    expert_points = points(expert)
    predicted_points = points(predicted)
    draw.line(expert_points, fill=(230, 130, 30), width=4, joint="curve")
    draw.line(predicted_points, fill=(30, 180, 60), width=4, joint="curve")
    for point in expert_points:
        draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=(230, 130, 30))
    for point in predicted_points:
        draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=(30, 180, 60))
    draw.text((410, 10), "orange: expert   green: current model", fill=(20, 20, 20))
    draw.text((590, 28), "forward", fill=(90, 90, 90))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> int:
    args = arguments()
    sample_path = args.sample.expanduser().resolve(strict=True)
    config_path = args.config.expanduser().resolve(strict=True)
    checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import yaml

    config_value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (
        not isinstance(config_value, Mapping)
        or config_value.get("method") != "architecture_v1_end_to_end"
        or config_value.get("test_locked_used") is not False
    ):
        raise ValueError("Phase-3 smoke requires an admitted Architecture-v1 config")
    repository = config_path.parents[2]
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))

    def resolve(value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else repository / path

    for field in ("da3_source", "da3_runtime"):
        extra = resolve(config_value[field]).resolve(strict=True)
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))

    from omtrackvla.models.end_to_end import (
        ArchitectureV1Ablation,
        ArchitectureV1Config,
        EndToEndFollowPolicy,
        load_official_da3_small_l11,
        parameter_inventory,
        waypoint_gradient_report,
        waypoint_only_loss,
    )

    architecture = ArchitectureV1Config(**config_value.get("architecture", {}))
    architecture.validate()
    ablation = ArchitectureV1Ablation(**config_value.get("ablation", {}))
    ablation.validate()
    sample = validate_sample(_load_json(sample_path), sample_path, architecture.history_size)
    root = sample_path.parent.resolve(strict=True)
    inputs_value = sample["model_inputs"]
    initial_record = inputs_value["initial_rgb"]
    initial_path = _safe_file(root, initial_record["rgb_path"], initial_record["sha256"])
    history_paths = [
        _safe_file(root, record["rgb_path"], record["sha256"])
        for record in inputs_value["rgb_history"]
    ]

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("NEXT-007 real DA3 smoke requires CUDA")
    torch.cuda.set_device(device)
    da3, da3_loading = load_official_da3_small_l11(
        resolve(config_value["da3_model"]), architecture, ablation,
        resolve(config_value["dinov2_model"])
        if config_value.get("dinov2_model") else None,
    )
    policy = EndToEndFollowPolicy(da3, architecture, ablation).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("phase") != 2
        or checkpoint.get("method") != "architecture_v1_end_to_end"
        or checkpoint.get("test_locked_used") is not False
    ):
        raise ValueError("NEXT-007 smoke checkpoint is not an admitted Phase-2 model")
    policy.load_state_dict(checkpoint["model"], strict=True)
    policy.train()

    uwb = inputs_value["uwb"]
    model_inputs = {
        "initial_rgb": _rgb_tensor(
            initial_path, architecture.image_height, architecture.image_width, torch
        )[None].to(device),
        "initial_bbox": torch.tensor(
            inputs_value["initial_bbox_xyxy_norm"], dtype=torch.float32, device=device
        )[None],
        "ego_rgb": torch.stack([
            _rgb_tensor(path, architecture.image_height, architecture.image_width, torch)
            for path in history_paths
        ])[None].to(device),
        "visual_initialization_valid": torch.ones(1, device=device),
        "rgb_valid": torch.ones(1, device=device),
        "binding_valid": torch.ones(1, device=device),
        "uwb_xy": torch.tensor(
            uwb["relative_position_base_xy_m"], dtype=torch.float32, device=device
        )[None],
        "uwb_covariance_xy": torch.tensor(
            uwb["covariance_base_xy_m2"], dtype=torch.float32, device=device
        )[None],
        "uwb_quality": torch.tensor([uwb["quality_01"]], device=device),
        "uwb_age_s": torch.tensor([uwb["age_s"]], device=device),
        "uwb_valid": torch.zeros(1, device=device),
        "camera_intrinsics": torch.tensor(
            inputs_value["camera_intrinsics"], dtype=torch.float32, device=device
        )[None],
        "camera_from_base": torch.tensor(
            inputs_value["camera_from_base"], dtype=torch.float32, device=device
        )[None],
    }
    target_value = sample["supervision"]["expert_trajectory"]
    target = torch.tensor(
        target_value["waypoints_base_xy_m"], dtype=torch.float32, device=device
    )[None]
    mask = torch.tensor(target_value["valid_mask"], dtype=torch.bool, device=device)[None]

    policy.zero_grad(set_to_none=True)
    use_bfloat16 = torch.cuda.is_bf16_supported()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16):
        outputs = policy(**model_inputs)
        loss = waypoint_only_loss(outputs, target, mask)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("NEXT-007 waypoint-only loss is non-finite")
    loss.backward()
    gradients = waypoint_gradient_report(policy)
    required = ["fusion", "gru"]
    if ablation.backbone_tuning == "adapter":
        required.append("da3_adapter")
    missing = [name for name in required if gradients.get(name, 0.0) <= 0.0]
    if missing:
        raise RuntimeError(f"NEXT-007 waypoint-only gradient is zero: {missing}")

    predicted = outputs["waypoints"][0].detach().float().cpu().tolist()
    expert = target[0].detach().float().cpu().tolist()
    visualization = output_dir / "waypoint_comparison.png"
    _render_comparison(history_paths[-1], predicted, expert, visualization)
    report = {
        "schema_version": 1,
        "stage": "next007_phase3_single_sample_waypoint_backward_smoke",
        "status": "passed",
        "sample": str(sample_path),
        "sample_sha256": _sha256(sample_path),
        "sample_formal_training_eligible": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_global_step": checkpoint.get("global_step"),
        "waypoint_only_loss": float(loss.detach().float().cpu()),
        "gradient_norms": gradients,
        "required_nonzero": required,
        "predicted_waypoints_base_xy_m": predicted,
        "expert_waypoints_base_xy_m": expert,
        "prediction_path_length_m": float(np.linalg.norm(
            np.diff(np.asarray(predicted), axis=0), axis=1
        ).sum()),
        "expert_path_length_m": float(np.linalg.norm(
            np.diff(np.asarray(expert), axis=0), axis=1
        ).sum()),
        "visualization": str(visualization),
        "visualization_sha256": _sha256(visualization),
        "da3_pretrained_loading": da3_loading,
        "parameters": parameter_inventory(policy),
        "model_inputs": {
            "rgb_history_steps": [
                int(record["environment_step"])
                for record in inputs_value["rgb_history"]
            ],
            "uwb_valid": False,
            "gt_target_pose_or_depth_used": False,
            "later_bbox_used": False,
        },
        "optimizer_step_performed": False,
        "formal_training_started": False,
        "test_locked_used": False,
    }
    report_path = output_dir / "report.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    print(json.dumps({
        "status": "passed",
        "loss": report["waypoint_only_loss"],
        "gradient_norms": gradients,
        "prediction_path_length_m": report["prediction_path_length_m"],
        "expert_path_length_m": report["expert_path_length_m"],
        "report": str(report_path),
        "visualization": str(visualization),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
