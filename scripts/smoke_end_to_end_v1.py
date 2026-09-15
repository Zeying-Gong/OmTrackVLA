#!/usr/bin/env python3
"""Run a waypoint-only Architecture v1 backward smoke on raw RGB."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from omtrackvla.data.end_to_end import (
    load_raw_rgb_smoke_batch,
    load_sage3d_end_to_end_smoke_batch,
    to_device,
)
from omtrackvla.models.end_to_end import (
    ArchitectureV1Config,
    EndToEndFollowPolicy,
    StubDA3SmallL11Backbone,
    load_official_da3_small_l11,
    parameter_inventory,
    waypoint_gradient_report,
    waypoint_only_loss,
)


DEFAULT_SAMPLE = (
    REPOSITORY_ROOT
    / "example_datasets/samples/sage3d_extracted/0001_83992/stt/0/go2_realsense_d435i"
)
DEFAULT_EPISODE = "0001_83992/stt/0/go2_realsense_d435i"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("stub", "da3"), default="stub")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--bbox", nargs=4, type=float, default=(0.25, 0.10, 0.65, 0.95))
    parser.add_argument("--sage3d-root", type=Path)
    parser.add_argument("--sidecar-root", type=Path)
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
    parser.add_argument("--anchor-index", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260911)
    return parser.parse_args()


def initialize_distributed(device_name: str) -> tuple[torch.device, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not device_name.startswith("cuda"):
            raise ValueError("multi-process NEXT-025 smoke requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return torch.device("cuda", local_rank), rank, world_size
    return torch.device(device_name), rank, world_size


def main() -> int:
    args = arguments()
    device, rank, world_size = initialize_distributed(args.device)
    torch.manual_seed(args.seed + rank)
    config = ArchitectureV1Config()
    use_admitted_sage3d = args.sage3d_root is not None or args.sidecar_root is not None
    if use_admitted_sage3d:
        if args.sage3d_root is None or args.sidecar_root is None:
            raise ValueError("--sage3d-root and --sidecar-root must be supplied together")
        batch = load_sage3d_end_to_end_smoke_batch(
            root=args.sage3d_root,
            sidecar_root=args.sidecar_root,
            split_manifest=args.split_manifest,
            policy_admission=args.policy_admission,
            episode_relative=args.episode,
            initial_index=args.initial_index,
            anchor_index=args.anchor_index,
            config=config,
        )
        data_source = {
            "kind": "admitted_sage3d_train",
            "episode": args.episode,
            "initial_index": args.initial_index,
            "anchor_index": args.anchor_index,
            "sidecar_admission_checked": True,
            "policy_admission_checked": True,
            "phase1_split": "train",
        }
        waypoint_target = "admitted_sage3d_expert"
    else:
        image_paths = [
            args.sample_dir / "rgb" / f"{index:05d}.jpg" for index in range(5)
        ]
        batch = load_raw_rgb_smoke_batch(
            image_paths=image_paths,
            camera_info=args.sample_dir / "camera_info.json",
            initial_bbox_xyxy_norm=args.bbox,
            config=config,
        )
        data_source = {"kind": "repository_sample"}
        waypoint_target = "smoke-only synthetic trajectory"
    batch = to_device(batch, device)

    if args.backend == "da3":
        if args.model is None:
            raise ValueError("--model is required for the official DA3 backend")
        da3, loading = load_official_da3_small_l11(args.model, config)
    else:
        da3 = StubDA3SmallL11Backbone(config)
        loading = {"backend": "stub", "pretrained_weight_loading": False}
    model = EndToEndFollowPolicy(da3, config).to(device)
    inventory = parameter_inventory(model)
    wrapped: torch.nn.Module = model
    if world_size > 1:
        wrapped = DistributedDataParallel(
            model,
            device_ids=[device.index],
            find_unused_parameters=True,
        )
    model_inputs = {
        key: value
        for key, value in batch.items()
        if key not in {"target_waypoints", "waypoint_mask"}
    }
    outputs = wrapped(**model_inputs)
    loss = waypoint_only_loss(
        outputs,
        batch["target_waypoints"],
        batch["waypoint_mask"],
    )
    loss.backward()
    gradients = waypoint_gradient_report(model)
    required = ("fusion", "gru", "da3_adapter")
    missing = [name for name in required if gradients[name] <= 0.0]
    if missing:
        raise RuntimeError(f"waypoint-only backward has zero required gradients: {missing}")
    loss_value = loss.detach().float()
    if world_size > 1:
        dist.all_reduce(loss_value)
        loss_value /= world_size
    report = {
        "schema_version": 1,
        "task": "NEXT-025 Architecture v1 waypoint-only backward smoke",
        "status": "passed",
        "backend": args.backend,
        "world_size": world_size,
        "seed": args.seed,
        "data_source": data_source,
        "architecture": config.to_dict(),
        "input_contract": {
            "raw_rgb": True,
            "initial_bbox_once": True,
            "uwb_kind": "simulated_uwb",
            "perception_cache": False,
            "external_later_bbox": False,
            "gt_depth_or_target_pose_input": False,
            "waypoint_target": waypoint_target,
        },
        "output_shapes": {
            "waypoints": list(outputs["waypoints"].shape),
            "stop_logit": list(outputs["stop_logit"].shape),
            "l11_patch_bias": list(outputs["uwb_patch_bias"].shape),
            "xi_hat": list(outputs["xi_hat"].shape),
        },
        "waypoint_only_loss": float(loss_value.cpu()),
        "waypoint_only_gradient_norms": gradients,
        "parameter_inventory": inventory,
        "da3_pretrained_loading": loading,
        "auxiliary_losses_used": [],
        "test_locked_used": False,
    }
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, sort_keys=True))
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
