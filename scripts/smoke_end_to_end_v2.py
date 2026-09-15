#!/usr/bin/env python3
"""Run Architecture-v2 0-step/1-step RGB-to-waypoint smoke and render it."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from typing import Mapping

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from omtrackvla.data.end_to_end import (
    load_sage3d_end_to_end_smoke_batch,
    to_device,
)
from omtrackvla.models.end_to_end import (
    ArchitectureV1Config,
    StubDA3SmallL11Backbone,
    load_official_da3_small_l11,
    waypoint_only_loss,
)
from omtrackvla.models.end_to_end_v2 import (
    ArchitectureV2DecoderConfig,
    ArchitectureV2FollowPolicy,
    parameter_inventory_v2,
    waypoint_gradient_report_v2,
)


DEFAULT_EPISODE = "0001_83992/stt/0/go2_realsense_d435i"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("stub", "da3"), default="stub")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--sage3d-root", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=REPOSITORY_ROOT / "configs/manifests/phase1_v1.json",
    )
    parser.add_argument(
        "--policy-admission",
        type=Path,
        default=REPOSITORY_ROOT / "results/sage3d_policy_v1_audit/admission.json",
    )
    parser.add_argument("--episode", default=DEFAULT_EPISODE)
    parser.add_argument("--initial-index", type=int, default=0)
    parser.add_argument("--anchor-index", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visualization", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    return parser.parse_args()


def _cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float().cpu()


def render_initial_prediction(
    *,
    batch: Mapping[str, torch.Tensor],
    outputs: Mapping[str, torch.Tensor],
    path: Path,
    episode: str,
    anchor_index: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    path.parent.mkdir(parents=True, exist_ok=True)
    history = _cpu(batch["ego_rgb"])[0].permute(0, 2, 3, 1).numpy()
    initial = _cpu(batch["initial_rgb"])[0].permute(1, 2, 0).numpy()
    bbox = _cpu(batch["initial_bbox"])[0].numpy()
    appearance = _cpu(outputs["appearance_attention"])[0].reshape(20, 36).numpy()
    geometry = _cpu(outputs["uwb_geometry_attention"])[0].reshape(20, 36).numpy()
    scene = _cpu(outputs["scene_resampler_attention"])[0].mean(dim=(1, 2))
    scene = scene.reshape(8, 20, 36).numpy()
    predicted = _cpu(outputs["waypoints"])[0].numpy()
    expert = _cpu(batch["target_waypoints"])[0].numpy()

    figure, axes = plt.subplots(3, 4, figsize=(18, 11), constrained_layout=True)
    height, width = initial.shape[:2]
    axes[0, 0].imshow(initial)
    axes[0, 0].add_patch(
        Rectangle(
            (bbox[0] * width, bbox[1] * height),
            (bbox[2] - bbox[0]) * width,
            (bbox[3] - bbox[1]) * height,
            fill=False,
            edgecolor="lime",
            linewidth=2,
        )
    )
    axes[0, 0].set_title("Initialization RGB + target bbox")
    axes[0, 0].axis("off")

    for axis, heat, title in (
        (axes[0, 1], appearance, "Current target appearance attention"),
        (axes[0, 2], geometry, "Current UWB Gaussian attention"),
    ):
        axis.imshow(history[-1])
        axis.imshow(
            heat,
            cmap="magma",
            alpha=0.55,
            interpolation="bilinear",
            extent=(0, history.shape[2], history.shape[1], 0),
        )
        axis.set_title(title)
        axis.axis("off")

    trajectory = axes[0, 3]
    trajectory.plot(expert[:, 1], expert[:, 0], "o-", color="royalblue", label="expert")
    trajectory.plot(
        predicted[:, 1], predicted[:, 0], "o-", color="limegreen", label="random v2"
    )
    trajectory.scatter([0.0], [0.0], color="black", marker="x", s=80)
    trajectory.set_xlabel("left y (m)")
    trajectory.set_ylabel("forward x (m)")
    trajectory.set_aspect("equal", adjustable="datalim")
    trajectory.grid(True, alpha=0.3)
    trajectory.legend()
    trajectory.set_title("0-step trajectory (before any training)")

    raw_offsets = tuple(range(-21, 1, 3))
    for index, axis in enumerate(axes[1:].reshape(-1)):
        axis.imshow(history[index])
        axis.imshow(
            scene[index],
            cmap="viridis",
            alpha=0.42,
            interpolation="bilinear",
            extent=(0, history.shape[2], history.shape[1], 0),
        )
        axis.set_title(f"history raw offset {raw_offsets[index]} / scene")
        axis.axis("off")
    figure.suptitle(
        f"Architecture v2 initial-weight smoke | {episode} | anchor {anchor_index}",
        fontsize=14,
    )
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> int:
    args = arguments()
    if args.learning_rate <= 0.0:
        raise ValueError("learning rate must be positive")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    sensor_config = ArchitectureV1Config(history_size=8)
    decoder_config = ArchitectureV2DecoderConfig(history_size=8, history_stride_raw=3)

    # Reuse the admitted loader for the dense anchor-21..anchor window, then
    # select every third 30-Hz frame to obtain the frozen 8-frame/10-Hz input.
    dense_loader_config = replace(sensor_config, history_size=22)
    batch = load_sage3d_end_to_end_smoke_batch(
        root=args.sage3d_root,
        sidecar_root=args.sidecar_root,
        split_manifest=args.split_manifest,
        policy_admission=args.policy_admission,
        episode_relative=args.episode,
        initial_index=args.initial_index,
        anchor_index=args.anchor_index,
        config=dense_loader_config,
    )
    batch["ego_rgb"] = batch["ego_rgb"][:, :: decoder_config.history_stride_raw]
    if batch["ego_rgb"].shape[1] != sensor_config.history_size:
        raise RuntimeError("dense history did not produce exactly eight strided frames")
    batch = to_device(batch, device)

    if args.backend == "da3":
        if args.model is None:
            raise ValueError("--model is required for the official DA3 backend")
        da3, loading = load_official_da3_small_l11(args.model, sensor_config)
    else:
        da3 = StubDA3SmallL11Backbone(sensor_config)
        loading = {"backend": "stub", "pretrained_weight_loading": False}
    model = ArchitectureV2FollowPolicy(
        da3, sensor_config, decoder_config
    ).to(device)
    if any("gru" in name.lower() for name, _ in model.named_parameters()):
        raise RuntimeError("Architecture v2 unexpectedly contains a GRU parameter")
    inventory = parameter_inventory_v2(model)
    model_inputs = {
        key: value
        for key, value in batch.items()
        if key not in {"target_waypoints", "waypoint_mask"}
    }
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    autocast_enabled = device.type == "cuda"
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16 if autocast_enabled else torch.float32,
        enabled=autocast_enabled,
    ):
        outputs_zero = model(**model_inputs)
        loss_zero = waypoint_only_loss(
            outputs_zero, batch["target_waypoints"], batch["waypoint_mask"]
        )
    loss_zero.backward()
    gradients = waypoint_gradient_report_v2(model)
    missing = [name for name, norm in gradients.items() if norm <= 0.0]
    if missing:
        raise RuntimeError(
            f"waypoint-only backward has zero required gradients: {missing}"
        )
    render_initial_prediction(
        batch=batch,
        outputs=outputs_zero,
        path=args.visualization,
        episode=args.episode,
        anchor_index=args.anchor_index,
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16 if autocast_enabled else torch.float32,
        enabled=autocast_enabled,
    ):
        outputs_one = model(**model_inputs)
        loss_one = waypoint_only_loss(
            outputs_one, batch["target_waypoints"], batch["waypoint_mask"]
        )
    if not torch.isfinite(loss_one):
        raise RuntimeError("1-step waypoint loss is not finite")

    report = {
        "schema_version": 1,
        "task": "V2-001 Architecture v2 0-step/1-step smoke",
        "status": "passed",
        "backend": args.backend,
        "seed": args.seed,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "data_source": {
            "kind": "admitted_sage3d_train",
            "episode": args.episode,
            "initial_index": args.initial_index,
            "anchor_index": args.anchor_index,
            "phase1_split": "train",
        },
        "temporal_contract": {
            "source_hz": 30,
            "policy_hz": 10,
            "history_size": 8,
            "raw_stride": 3,
            "raw_offsets": list(range(-21, 1, 3)),
            "span_s": 0.7,
            "scene_latents_per_frame": decoder_config.scene_latents,
            "total_scene_latents": (
                decoder_config.history_size * decoder_config.scene_latents
            ),
        },
        "deployment_contract": {
            "raw_rgb_to_da3": True,
            "target_appearance_current_frame": True,
            "uwb_gaussian_geometry_path": True,
            "uwb_independent_continuous_path": True,
            "se2_ego_bottleneck": True,
            "trajectory_transformer": True,
            "gru": False,
            "auxiliary_losses_used": [],
            "gt_depth_or_target_pose_input": False,
        },
        "output_shapes": {
            "waypoints": list(outputs_zero["waypoints"].shape),
            "stop_logit": list(outputs_zero["stop_logit"].shape),
            "uwb_patch_bias": list(outputs_zero["uwb_patch_bias"].shape),
            "scene_resampler_attention": list(
                outputs_zero["scene_resampler_attention"].shape
            ),
        },
        "zero_step_waypoint_only_loss": float(loss_zero.detach().float().cpu()),
        "one_step_waypoint_only_loss": float(loss_one.detach().float().cpu()),
        "waypoint_only_gradient_norms": gradients,
        "parameter_inventory": inventory,
        "da3_pretrained_loading": loading,
        "zero_step_waypoints": _cpu(outputs_zero["waypoints"])[0].tolist(),
        "one_step_waypoints": _cpu(outputs_one["waypoints"])[0].tolist(),
        "expert_waypoints": _cpu(batch["target_waypoints"])[0].tolist(),
        "visualization": str(args.visualization),
        "test_locked_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
