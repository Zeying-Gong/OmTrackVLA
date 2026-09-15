#!/usr/bin/env python3
"""Render fixed before/after overlays for audited Phase-3 recovery states."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from smoke_next007_phase3_minibatch import _stack_samples
from smoke_next007_phase3_sample import _render_comparison, _sha256
from train_next007_phase3 import validate_manifest


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--before-checkpoint", type=Path, required=True)
    parser.add_argument("--after-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-task", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def path_length(points: list[list[float]]) -> float:
    value = np.asarray(points, dtype=np.float64)
    return float(np.linalg.norm(value[1:] - value[:-1], axis=-1).sum())


def metrics(predicted: list[list[float]], expert: list[list[float]]) -> dict[str, float]:
    predicted_value = np.asarray(predicted, dtype=np.float64)
    expert_value = np.asarray(expert, dtype=np.float64)
    error = np.linalg.norm(predicted_value - expert_value, axis=-1)
    predicted_path = path_length(predicted)
    expert_path = path_length(expert)
    return {
        "ade_m": float(error.mean()),
        "fde_m": float(error[-1]),
        "predicted_path_length_m": predicted_path,
        "expert_path_length_m": expert_path,
        "path_length_ratio": predicted_path / max(expert_path, 1.0e-8),
    }


def main() -> int:
    args = arguments()
    if args.samples_per_task <= 0:
        raise ValueError("samples-per-task must be positive")
    config_path = args.config.expanduser().resolve(strict=True)
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    before_path = args.before_checkpoint.expanduser().resolve(strict=True)
    after_path = args.after_checkpoint.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (
        not isinstance(config, Mapping)
        or config.get("method") != "architecture_v1_end_to_end"
        or config.get("test_locked_used") is not False
    ):
        raise ValueError("invalid Architecture v1 rendering config")
    repository = config_path.parents[2]

    def resolve(value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else repository / path

    for field in ("da3_source", "da3_runtime"):
        path = resolve(config[field]).resolve(strict=True)
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from omtrackvla.models.end_to_end import (
        ArchitectureV1Ablation,
        ArchitectureV1Config,
        EndToEndFollowPolicy,
        load_official_da3_small_l11,
    )

    manifest, all_paths = validate_manifest(manifest_path)
    path_by_id = {
        str(record["sample_id"]): Path(str(record["path"])).resolve(strict=True)
        for record in manifest["samples"]
    }
    selected_ids = []
    for task in ("stt", "dt", "at"):
        candidates = [
            str(record["sample_id"])
            for record in manifest["samples"]
            if record["task"] == task
        ]
        selected_ids.extend(candidates[: args.samples_per_task])
    selected_paths = [path_by_id[sample_id] for sample_id in selected_ids]
    if len(selected_paths) < 3:
        raise ValueError("recovery visualization lacks one of the three tasks")

    architecture = ArchitectureV1Config(**config.get("architecture", {}))
    ablation = ArchitectureV1Ablation(**config.get("ablation", {}))
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Phase 3 recovery rendering requires CUDA")
    torch.cuda.set_device(device)
    inputs, target, _, loaded = _stack_samples(
        selected_paths, architecture, torch, device
    )
    da3, da3_loading = load_official_da3_small_l11(
        resolve(config["da3_model"]), architecture, ablation,
        resolve(config["dinov2_model"]) if config.get("dinov2_model") else None,
    )
    policy = EndToEndFollowPolicy(da3, architecture, ablation).to(device)

    def predict(checkpoint_path: Path) -> list[list[list[float]]]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            checkpoint.get("phase") not in {2, 3}
            or checkpoint.get("method") != "architecture_v1_end_to_end"
            or checkpoint.get("test_locked_used") is not False
        ):
            raise ValueError(f"unadmitted Architecture v1 checkpoint: {checkpoint_path}")
        policy.load_state_dict(checkpoint["model"], strict=True)
        policy.eval()
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16,
            enabled=torch.cuda.is_bf16_supported(),
        ):
            output = policy(**inputs)
        return output["waypoints"].detach().float().cpu().tolist()

    before = predict(before_path)
    after = predict(after_path)
    expert = target.detach().float().cpu().tolist()
    records: list[dict[str, Any]] = []
    for index, sample in enumerate(loaded):
        before_image = output_dir / f"sample_{index:02d}_before.png"
        after_image = output_dir / f"sample_{index:02d}_after.png"
        _render_comparison(sample["current_rgb"], before[index], expert[index], before_image)
        _render_comparison(sample["current_rgb"], after[index], expert[index], after_image)
        records.append({
            "sample_id": sample["value"]["sample_id"],
            "sample": str(sample["path"]),
            "before": metrics(before[index], expert[index]),
            "after": metrics(after[index], expert[index]),
            "before_image": str(before_image),
            "before_image_sha256": _sha256(before_image),
            "after_image": str(after_image),
            "after_image_sha256": _sha256(after_image),
        })
    report = {
        "schema_version": 1,
        "stage": "next007_phase3_recovery_fixed_visualization_v1",
        "status": "complete",
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "manifest_total_samples": len(all_paths),
        "before_checkpoint": str(before_path),
        "before_checkpoint_sha256": _sha256(before_path),
        "after_checkpoint": str(after_path),
        "after_checkpoint_sha256": _sha256(after_path),
        "records": records,
        "da3_pretrained_loading": da3_loading,
        "test_locked_used": False,
    }
    report_path = output_dir / "report.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    print(json.dumps({"status": "complete", "samples": len(records), "report": str(report_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
