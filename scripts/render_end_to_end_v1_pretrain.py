#!/usr/bin/env python3
"""Render an initial-weight Architecture v1 dashboard before formal training."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from omtrackvla.data.end_to_end import load_sage3d_end_to_end_smoke_batch, to_device
from omtrackvla.evaluation.end_to_end_render import render_pretrain_dashboard
from omtrackvla.models.end_to_end import (
    ArchitectureV1Config,
    EndToEndFollowPolicy,
    load_official_da3_small_l11,
    parameter_inventory,
    waypoint_only_loss,
)


DEFAULT_SAGE3D_ROOT = Path("/data/nfs/share/OmTrackVLA/data/sage3d_extracted")
DEFAULT_SIDECAR_ROOT = REPOSITORY_ROOT / "results/sage3d_bbox_sidecar_v1"
DEFAULT_EPISODE = "0001_83992/stt/0/go2_realsense_d435i"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sage3d-root", type=Path, default=DEFAULT_SAGE3D_ROOT)
    parser.add_argument("--sidecar-root", type=Path, default=DEFAULT_SIDECAR_ROOT)
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
    parser.add_argument("--report", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260911)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = arguments()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    config = ArchitectureV1Config()
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
    batch = to_device(batch, device)
    da3, loading = load_official_da3_small_l11(args.model, config)
    model = EndToEndFollowPolicy(da3, config).to(device).eval()
    model_inputs = {
        key: value
        for key, value in batch.items()
        if key not in {"target_waypoints", "waypoint_mask"}
    }
    with torch.inference_mode():
        outputs = model(**model_inputs)
        loss = waypoint_only_loss(
            outputs, batch["target_waypoints"], batch["waypoint_mask"]
        )
    metadata = {
        "backend": "official DA3-SMALL L11 + initial policy weights",
        "episode": args.episode,
        "initial_index": args.initial_index,
        "anchor_index": args.anchor_index,
        "da3_coverage": loading["parameter_coverage"],
        "waypoint_loss": float(loss.cpu()),
    }
    dashboard = render_pretrain_dashboard(
        batch,
        outputs,
        metadata,
        grid_shape=(config.grid_height, config.grid_width),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), dashboard):
        raise RuntimeError(f"failed to write visualization: {args.output}")
    decoded = cv2.imread(str(args.output), cv2.IMREAD_COLOR)
    if decoded is None or decoded.shape != dashboard.shape:
        raise RuntimeError("written visualization failed the decode/shape check")
    report_path = args.report or args.output.with_suffix(".json")
    report = {
        "schema_version": 1,
        "task": "Architecture v1 pre-training qualitative inspection",
        "status": "passed",
        "untrained_policy_parameters": True,
        "optimizer_steps": 0,
        "seed": args.seed,
        "data_source": {
            "dataset": "sage3d_extracted",
            "split": "train",
            "episode": args.episode,
            "initial_index": args.initial_index,
            "anchor_index": args.anchor_index,
            "waypoint_target": "admitted_sage3d_expert",
            "uwb_kind": "simulated_uwb",
            "perception_cache": False,
            "test_locked_used": False,
        },
        "da3_pretrained_loading": loading,
        "parameter_inventory": parameter_inventory(model),
        "diagnostics": {
            "waypoint_smooth_l1": float(loss.cpu()),
            "predicted_waypoints": outputs["waypoints"][0].cpu().tolist(),
            "expert_waypoints": batch["target_waypoints"][0].cpu().tolist(),
            "stop_probability": float(
                torch.sigmoid(outputs["stop_logit"][0, 0]).cpu()
            ),
            "xi_hat": outputs["xi_hat"][0].cpu().tolist(),
            "uwb_xy": batch["uwb_xy"][0].cpu().tolist(),
            "uwb_mean_uv": outputs["uwb_mean_uv"][0].cpu().tolist(),
            "uwb_gamma": float(outputs["uwb_gamma"][0].cpu()),
            "uwb_rho_fov": float(outputs["uwb_rho_fov"][0].cpu()),
        },
        "visualization": {
            "path": str(args.output),
            "height": int(decoded.shape[0]),
            "width": int(decoded.shape[1]),
            "sha256": _sha256(args.output),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
