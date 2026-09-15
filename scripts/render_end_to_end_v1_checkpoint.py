#!/usr/bin/env python3
"""Render a trained Architecture v1 checkpoint on the fixed SAGE3D sample."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import cv2
import torch
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from omtrackvla.data.end_to_end import load_sage3d_end_to_end_smoke_batch, to_device
from omtrackvla.evaluation.end_to_end_render import render_pretrain_dashboard
from omtrackvla.models.end_to_end import (
    ArchitectureV1Ablation,
    ArchitectureV1Config,
    EndToEndFollowPolicy,
    load_official_da3_small_l11,
    waypoint_only_loss,
)


DEFAULT_EPISODE = "0001_83992/stt/0/go2_realsense_d435i"


def checkpoint_backend_label(checkpoint: Mapping[str, object]) -> str:
    """Describe the trained policy without conflating its initialization arm."""

    if checkpoint.get("phase") == 3:
        stage = str(checkpoint.get("stage", "unknown stage"))
        return f"trained Architecture v1 Phase 3 ({stage})"
    initialization = checkpoint.get("initialization")
    if isinstance(initialization, Mapping):
        kind = initialization.get("kind")
        if kind == "direct_phase2":
            return "trained Architecture v1 Phase 2 (direct DA3 initialization)"
        if (
            kind == "architecture_v1_checkpoint"
            and initialization.get("checkpoint_phase") == 1
        ):
            return "trained Architecture v1 Phase 2 (Phase 1 initialization)"
    stage = str(checkpoint.get("stage", "unknown stage"))
    return f"trained Architecture v1 Phase 2 ({stage})"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--episode", default=DEFAULT_EPISODE)
    parser.add_argument("--initial-index", type=int, default=0)
    parser.add_argument("--anchor-index", type=int, default=106)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = arguments()
    config_path = args.config.expanduser().resolve(strict=True)
    checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)
    config_value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (
        config_value.get("method") != "architecture_v1_end_to_end"
        or config_value.get("test_locked_used") is not False
    ):
        raise ValueError("renderer requires a non-locked Architecture v1 config")
    for field in ("da3_source", "da3_runtime"):
        path = Path(config_value[field]).resolve(strict=True)
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    device = torch.device(args.device)
    architecture = ArchitectureV1Config(**config_value.get("architecture", {}))
    ablation = ArchitectureV1Ablation(**config_value.get("ablation", {}))
    ablation.validate()
    batch = load_sage3d_end_to_end_smoke_batch(
        root="/data/nfs/share/OmTrackVLA/data/sage3d_extracted",
        sidecar_root=REPOSITORY_ROOT / "results/sage3d_bbox_sidecar_v1",
        split_manifest=REPOSITORY_ROOT / "configs/manifests/phase1_v1.json",
        policy_admission=REPOSITORY_ROOT
        / "results/sage3d_policy_v1_audit/admission.json",
        episode_relative=args.episode,
        initial_index=args.initial_index,
        anchor_index=args.anchor_index,
        config=architecture,
    )
    batch = to_device(batch, device)
    da3, loading = load_official_da3_small_l11(
        config_value["da3_model"],
        architecture,
        ablation,
        config_value.get("dinov2_model"),
    )
    model = EndToEndFollowPolicy(da3, architecture, ablation).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("method") != "architecture_v1_end_to_end"
        or checkpoint.get("test_locked_used") is not False
    ):
        raise ValueError("renderer checkpoint provenance mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    model_inputs = {
        key: value
        for key, value in batch.items()
        if key not in {"target_waypoints", "waypoint_mask"}
    }
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
    ):
        outputs = model(**model_inputs)
        loss = waypoint_only_loss(
            outputs, batch["target_waypoints"], batch["waypoint_mask"]
        )
    metadata = {
        "backend": checkpoint_backend_label(checkpoint),
        "policy_state": "trained",
        "checkpoint_step": int(checkpoint["global_step"]),
        "temporal_fusion": model.ablation.temporal_fusion,
        "episode": args.episode,
        "initial_index": args.initial_index,
        "anchor_index": args.anchor_index,
        "da3_coverage": loading["parameter_coverage"],
        "waypoint_loss": float(loss.float().cpu()),
    }
    dashboard = render_pretrain_dashboard(
        batch,
        outputs,
        metadata,
        grid_shape=(architecture.grid_height, architecture.grid_width),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), dashboard):
        raise RuntimeError(f"failed to write visualization: {args.output}")
    decoded = cv2.imread(str(args.output), cv2.IMREAD_COLOR)
    if decoded is None or decoded.shape != dashboard.shape:
        raise RuntimeError("written visualization failed decode/shape validation")
    report = {
        "schema_version": 1,
        "task": "NEXT-026 trained Architecture v1 qualitative inspection",
        "status": "passed",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_global_step": int(checkpoint["global_step"]),
        "stage": checkpoint["stage"],
        "initialization": checkpoint.get("initialization", {}),
        "data_source": {
            "dataset": "sage3d_extracted",
            "split": "train",
            "episode": args.episode,
            "initial_index": args.initial_index,
            "anchor_index": args.anchor_index,
            "uwb_kind": "simulated_uwb",
            "perception_cache": False,
            "test_locked_used": False,
        },
        "diagnostics": {
            "waypoint_smooth_l1": float(loss.float().cpu()),
            "predicted_waypoints": outputs["waypoints"][0].float().cpu().tolist(),
            "expert_waypoints": batch["target_waypoints"][0].cpu().tolist(),
            "stop_probability": float(
                torch.sigmoid(outputs["stop_logit"][0, 0].float()).cpu()
            ),
            "binding_probability": float(
                torch.sigmoid(outputs["binding_logit"][0, 0].float()).cpu()
            ),
            "xi_hat": outputs["xi_hat"][0].float().cpu().tolist(),
            "uwb_gamma": float(outputs["uwb_gamma"][0].float().cpu()),
        },
        "da3_pretrained_loading": loading,
        "visualization": {
            "path": str(args.output),
            "height": int(decoded.shape[0]),
            "width": int(decoded.shape[1]),
            "sha256": _sha256(args.output),
        },
        "test_locked_used": False,
    }
    report_path = args.report or args.output.with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
