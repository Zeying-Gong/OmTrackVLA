#!/usr/bin/env python3
"""Measure whether Architecture-v1 waypoint decisions use UWB direction.

This is a read-only diagnostic.  It evaluates the admitted validation split and
replays the UWB stream from an existing NEXT-027 result; it never opens
``test_locked`` and never changes a checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from omtrackvla.data.end_to_end_training import Sage3DEndToEndSequenceDataset
from omtrackvla.evaluation.end_to_end_closed_loop import (
    _load_policy_after_first_render,
    habitat_camera_calibration,
)
from omtrackvla.evaluation.end_to_end_evaluate import MODEL_INPUT_KEYS
from omtrackvla.models.end_to_end import ArchitectureV1Config


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--closed-loop-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def _resolve(repository: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository / path


def _selected_lateral(waypoints: torch.Tensor) -> torch.Tensor:
    """Match the deployed action adapter's waypoint selection."""

    radius = torch.linalg.vector_norm(waypoints, dim=-1)
    reaches = radius >= 0.15
    first = reaches.to(torch.int64).argmax(dim=-1)
    fallback = radius.argmax(dim=-1)
    selected = torch.where(reaches.any(dim=-1), first, fallback).clamp_min(1)
    return waypoints.gather(
        -2,
        selected[..., None, None].expand(*selected.shape, 1, 2),
    ).squeeze(-2)[..., 1]


class DirectionStats:
    def __init__(self) -> None:
        self.count = 0
        self.target_count = 0
        self.uwb_count = 0
        self.target_matches = 0
        self.uwb_matches = 0
        self.ade_sum = 0.0
        self.predicted_lateral_sum = 0.0
        self.target_lateral_sum = 0.0
        self.uwb_lateral_sum = 0.0
        self.predicted: list[float] = []
        self.target: list[float] = []
        self.uwb: list[float] = []

    def update(
        self,
        predicted_waypoints: torch.Tensor,
        target_waypoints: torch.Tensor,
        uwb_xy: torch.Tensor,
    ) -> None:
        predicted = _selected_lateral(predicted_waypoints).detach().float().cpu()
        target = _selected_lateral(target_waypoints).detach().float().cpu()
        uwb = uwb_xy[..., 1].detach().float().cpu()
        ade = torch.linalg.vector_norm(
            predicted_waypoints.detach().float().cpu()
            - target_waypoints.detach().float().cpu(),
            dim=-1,
        )[..., 1:].mean(dim=-1)
        pred_flat = predicted.reshape(-1)
        target_flat = target.reshape(-1)
        uwb_flat = uwb.reshape(-1)
        self.count += int(pred_flat.numel())
        self.ade_sum += float(ade.sum())
        self.predicted_lateral_sum += float(pred_flat.sum())
        self.target_lateral_sum += float(target_flat.sum())
        self.uwb_lateral_sum += float(uwb_flat.sum())
        target_mask = target_flat.abs() >= 0.02
        uwb_mask = uwb_flat.abs() >= 0.05
        self.target_count += int(target_mask.sum())
        self.uwb_count += int(uwb_mask.sum())
        self.target_matches += int(
            ((pred_flat.sign() == target_flat.sign()) & target_mask).sum()
        )
        self.uwb_matches += int(
            ((pred_flat.sign() == uwb_flat.sign()) & uwb_mask).sum()
        )
        self.predicted.extend(float(value) for value in pred_flat)
        self.target.extend(float(value) for value in target_flat)
        self.uwb.extend(float(value) for value in uwb_flat)

    def report(self) -> dict[str, Any]:
        def correlation(first: list[float], second: list[float]) -> float | None:
            if len(first) < 2:
                return None
            value = float(np.corrcoef(first, second)[0, 1])
            return value if math.isfinite(value) else None

        return {
            "policy_steps": self.count,
            "waypoint_ade_m": self.ade_sum / max(self.count, 1),
            "selected_lateral_mean_m": self.predicted_lateral_sum / max(self.count, 1),
            "target_selected_lateral_mean_m": self.target_lateral_sum / max(self.count, 1),
            "uwb_lateral_mean_m": self.uwb_lateral_sum / max(self.count, 1),
            "selected_vs_target_lateral_sign": {
                "threshold_m": 0.02,
                "count": self.target_count,
                "agreement": self.target_matches / max(self.target_count, 1),
            },
            "selected_vs_uwb_lateral_sign": {
                "threshold_m": 0.05,
                "count": self.uwb_count,
                "agreement": self.uwb_matches / max(self.uwb_count, 1),
            },
            "selected_target_lateral_correlation": correlation(
                self.predicted, self.target
            ),
            "selected_uwb_lateral_correlation": correlation(self.predicted, self.uwb),
        }


def validation_counterfactuals(
    policy: Any,
    architecture: ArchitectureV1Config,
    device: torch.device,
    config_path: Path,
    config: Mapping[str, Any],
    *,
    samples: int,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    repository = config_path.parents[2]
    dataset = Sage3DEndToEndSequenceDataset(
        _resolve(repository, config["data"]["sequence_index"]),
        split="val",
        config=architecture,
        modes=("uwb_only",),
    )
    count = min(int(samples), len(dataset))
    # Spread samples over the full validation split without reading test_locked.
    indices = np.linspace(0, len(dataset) - 1, count, dtype=np.int64).tolist()
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    variants = {
        "dataset_uwb_only": DirectionStats(),
        "dataset_uwb_only_flip_lateral": DirectionStats(),
        "dataset_uwb_only_uwb_invalid": DirectionStats(),
        "blank_visual_rgb_invalid": DirectionStats(),
        "blank_visual_rgb_invalid_flip_lateral": DirectionStats(),
    }
    use_bfloat16 = torch.cuda.is_bf16_supported()
    with torch.inference_mode():
        for batch in loader:
            tensor_batch = {
                key: value.to(device, non_blocking=True)
                if torch.is_tensor(value)
                else value
                for key, value in batch.items()
            }
            base = {key: tensor_batch[key] for key in MODEL_INPUT_KEYS}
            for name, stats in variants.items():
                inputs = dict(base)
                if "blank_visual" in name:
                    inputs["initial_rgb"] = torch.zeros_like(inputs["initial_rgb"])
                    inputs["ego_rgb"] = torch.zeros_like(inputs["ego_rgb"])
                    inputs["visual_initialization_valid"] = torch.zeros_like(
                        inputs["visual_initialization_valid"]
                    )
                    inputs["rgb_valid"] = torch.zeros_like(inputs["rgb_valid"])
                    inputs["binding_valid"] = torch.zeros_like(inputs["binding_valid"])
                if "flip_lateral" in name:
                    inputs["uwb_xy"] = inputs["uwb_xy"].clone()
                    inputs["uwb_xy"][..., 1].neg_()
                if "uwb_invalid" in name:
                    inputs["uwb_xy"] = torch.zeros_like(inputs["uwb_xy"])
                    inputs["uwb_covariance_xy"] = torch.zeros_like(
                        inputs["uwb_covariance_xy"]
                    )
                    inputs["uwb_quality"] = torch.zeros_like(inputs["uwb_quality"])
                    inputs["uwb_valid"] = torch.zeros_like(inputs["uwb_valid"])
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16
                ):
                    outputs = policy.forward_sequence(**inputs)
                stats.update(
                    outputs["waypoints"],
                    tensor_batch["target_waypoints"],
                    inputs["uwb_xy"],
                )
    report = {name: stats.report() for name, stats in variants.items()}
    baseline = np.asarray(variants["dataset_uwb_only"].predicted, dtype=np.float64)
    for name in variants:
        if name == "dataset_uwb_only":
            continue
        compared = np.asarray(variants[name].predicted, dtype=np.float64)
        report[name]["mean_abs_selected_lateral_change_vs_dataset_uwb_only_m"] = (
            float(np.abs(compared - baseline).mean())
        )
    return report


def logged_stream_counterfactuals(
    policy: Any,
    architecture: ArchitectureV1Config,
    device: torch.device,
    result_path: Path,
) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("uwb_mode") != "simulated-noiseless":
        raise ValueError("closed-loop replay requires a simulated-noiseless result")
    steps = result.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("closed-loop result has no steps")
    height = int(architecture.image_height)
    width = int(architecture.image_width)
    zero_initial = torch.zeros(1, 3, height, width, device=device)
    zero_history = torch.zeros(
        1, architecture.history_size, 3, height, width, device=device
    )
    intrinsics, camera_from_base = habitat_camera_calibration(
        (512, 512, 3), height, width, 90.0
    )
    static = {
        "initial_rgb": zero_initial,
        "initial_bbox": torch.zeros(1, 4, device=device),
        "ego_rgb": zero_history,
        "visual_initialization_valid": torch.zeros(1, device=device),
        "rgb_valid": torch.zeros(1, device=device),
        "binding_valid": torch.zeros(1, device=device),
        "camera_intrinsics": torch.tensor(intrinsics, device=device)[None],
        "camera_from_base": torch.tensor(camera_from_base, device=device)[None],
    }
    variants = {
        "as_recorded": {"flip": False, "valid": True},
        "flip_lateral": {"flip": True, "valid": True},
        "uwb_invalid": {"flip": False, "valid": False},
    }
    state: dict[str, dict[str, Any]] = {
        name: {"hidden": None, "target": None, "selected": [], "uwb": []}
        for name in variants
    }
    use_bfloat16 = torch.cuda.is_bf16_supported()
    with torch.inference_mode():
        for record in steps:
            measurement = record["policy"]["uwb_measurement"]
            original_xy = np.asarray(measurement["xy_m"], dtype=np.float32)
            for name, specification in variants.items():
                xy = original_xy.copy()
                if specification["flip"]:
                    xy[1] *= -1.0
                valid = float(specification["valid"])
                inputs = {
                    **static,
                    "uwb_xy": torch.tensor(xy, device=device)[None],
                    "uwb_covariance_xy": torch.zeros(1, 2, 2, device=device),
                    "uwb_quality": torch.tensor([valid], device=device),
                    "uwb_age_s": torch.zeros(1, device=device),
                    "uwb_valid": torch.tensor([valid], device=device),
                    "hidden_state": state[name]["hidden"],
                    "_target_memory_override": state[name]["target"],
                }
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16
                ):
                    output = policy(**inputs)
                state[name]["hidden"] = output["hidden_state"]
                state[name]["target"] = output["target_memory_next"]
                selected = float(_selected_lateral(output["waypoints"])[0].float())
                state[name]["selected"].append(selected)
                state[name]["uwb"].append(float(xy[1]))

    report: dict[str, Any] = {}
    for name, values in state.items():
        selected = np.asarray(values["selected"], dtype=np.float64)
        uwb = np.asarray(values["uwb"], dtype=np.float64)
        mask = np.abs(uwb) >= 0.05
        tail = np.arange(len(uwb)) >= 27
        report[name] = {
            "steps": int(len(uwb)),
            "selected_lateral_mean_m": float(selected.mean()),
            "uwb_lateral_mean_m": float(uwb.mean()),
            "selected_vs_uwb_lateral_sign": {
                "count": int(mask.sum()),
                "agreement": float((np.sign(selected[mask]) == np.sign(uwb[mask])).mean())
                if mask.any()
                else None,
            },
            "steps_28_to_50": {
                "selected_lateral_mean_m": float(selected[tail].mean()),
                "uwb_lateral_mean_m": float(uwb[tail].mean()),
                "sign_agreement": float(
                    (np.sign(selected[tail]) == np.sign(uwb[tail])).mean()
                ),
                "selected_lateral_m": selected[tail].tolist(),
                "uwb_lateral_m": uwb[tail].tolist(),
            },
        }
    baseline = np.asarray(state["as_recorded"]["selected"], dtype=np.float64)
    for name in ("flip_lateral", "uwb_invalid"):
        compared = np.asarray(state[name]["selected"], dtype=np.float64)
        report[name]["mean_abs_selected_lateral_change_vs_recorded_m"] = float(
            np.abs(compared - baseline).mean()
        )
    return report


def logged_target_only_rgb_counterfactuals(
    policy: Any,
    architecture: ArchitectureV1Config,
    device: torch.device,
    result_path: Path,
) -> dict[str, Any]:
    """Replay current RGB while removing only the stale target reference.

    This matches the training meaning of ``uwb_only``: RGB remains available
    for scene geometry, while UWB supplies the target-location condition and
    may bind a current visual token.
    """

    result = json.loads(result_path.read_text(encoding="utf-8"))
    steps = result.get("steps")
    if result.get("uwb_mode") != "simulated-noiseless" or not steps:
        raise ValueError("target-only replay requires a simulated-noiseless result")
    frame_directory = result_path.parent / "rollout_frames"
    frames: list[torch.Tensor] = []
    for index in range(len(steps)):
        composite = cv2.imread(
            str(frame_directory / f"frame_{index:06d}.png"), cv2.IMREAD_COLOR
        )
        if composite is None:
            raise ValueError(f"missing logged rollout frame {index}")
        source_width = composite.shape[1] - composite.shape[0]
        if source_width <= 0:
            raise ValueError("logged rollout frame has no RGB panel")
        rgb = cv2.cvtColor(composite[:, :source_width], cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb,
            (int(architecture.image_width), int(architecture.image_height)),
            interpolation=cv2.INTER_LINEAR,
        )
        frames.append(
            torch.from_numpy(resized.copy())
            .to(device=device, dtype=torch.float32)
            .permute(2, 0, 1)
            .contiguous()
            .div_(255.0)
        )
    intrinsics, camera_from_base = habitat_camera_calibration(
        (composite.shape[0], source_width, 3),
        int(architecture.image_height),
        int(architecture.image_width),
        90.0,
    )
    variants = {
        "target_only_rgb": {"flip": False, "valid": True},
        "target_only_rgb_flip_lateral": {"flip": True, "valid": True},
        "target_only_rgb_uwb_invalid": {"flip": False, "valid": False},
    }
    states: dict[str, dict[str, Any]] = {}
    for name in variants:
        history = deque(maxlen=int(architecture.history_size))
        for _ in range(int(architecture.history_size)):
            history.append(frames[0])
        states[name] = {
            "hidden": None,
            "target": None,
            "binding": torch.zeros(1, device=device),
            "history": history,
            "selected": [],
            "uwb": [],
        }
    common = {
        "initial_rgb": frames[0][None],
        "initial_bbox": torch.zeros(1, 4, device=device),
        "visual_initialization_valid": torch.zeros(1, device=device),
        "rgb_valid": torch.ones(1, device=device),
        "camera_intrinsics": torch.tensor(intrinsics, device=device)[None],
        "camera_from_base": torch.tensor(camera_from_base, device=device)[None],
    }
    use_bfloat16 = torch.cuda.is_bf16_supported()
    with torch.inference_mode():
        for index, record in enumerate(steps):
            original_xy = np.asarray(
                record["policy"]["uwb_measurement"]["xy_m"], dtype=np.float32
            )
            for name, specification in variants.items():
                current = states[name]
                current["history"].append(frames[index])
                xy = original_xy.copy()
                if specification["flip"]:
                    xy[1] *= -1.0
                valid = float(specification["valid"])
                if not specification["valid"]:
                    xy[:] = 0.0
                inputs = {
                    **common,
                    "ego_rgb": torch.stack(tuple(current["history"]))[None],
                    "binding_valid": current["binding"],
                    "uwb_xy": torch.tensor(xy, device=device)[None],
                    "uwb_covariance_xy": torch.zeros(1, 2, 2, device=device),
                    "uwb_quality": torch.tensor([valid], device=device),
                    "uwb_age_s": torch.zeros(1, device=device),
                    "uwb_valid": torch.tensor([valid], device=device),
                    "hidden_state": current["hidden"],
                    "_target_memory_override": current["target"],
                }
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16
                ):
                    output = policy(**inputs)
                current["hidden"] = output["hidden_state"]
                current["target"] = output["target_memory_next"]
                predicted_binding = (
                    torch.sigmoid(output["binding_logit"]).reshape(1)
                    * inputs["uwb_valid"]
                    * inputs["rgb_valid"]
                )
                current["binding"] = torch.maximum(
                    current["binding"], predicted_binding
                )
                current["selected"].append(
                    float(_selected_lateral(output["waypoints"])[0].float())
                )
                current["uwb"].append(float(original_xy[1] if valid else 0.0))

    report: dict[str, Any] = {}
    for name, values in states.items():
        selected = np.asarray(values["selected"], dtype=np.float64)
        uwb = np.asarray(values["uwb"], dtype=np.float64)
        mask = np.abs(uwb) >= 0.05
        tail = np.arange(len(uwb)) >= 27
        report[name] = {
            "steps": int(len(uwb)),
            "selected_lateral_mean_m": float(selected.mean()),
            "uwb_lateral_mean_m": float(uwb.mean()),
            "selected_vs_uwb_lateral_sign": {
                "count": int(mask.sum()),
                "agreement": float(
                    (np.sign(selected[mask]) == np.sign(uwb[mask])).mean()
                )
                if mask.any()
                else None,
            },
            "steps_28_to_50": {
                "selected_lateral_mean_m": float(selected[tail].mean()),
                "uwb_lateral_mean_m": float(uwb[tail].mean()),
                "sign_agreement": float(
                    (np.sign(selected[tail]) == np.sign(uwb[tail])).mean()
                )
                if np.any(np.abs(uwb[tail]) >= 0.05)
                else None,
            },
            "final_binding_probability": float(values["binding"].cpu()),
        }
    baseline = np.asarray(states["target_only_rgb"]["selected"], dtype=np.float64)
    for name in ("target_only_rgb_flip_lateral", "target_only_rgb_uwb_invalid"):
        compared = np.asarray(states[name]["selected"], dtype=np.float64)
        report[name]["mean_abs_selected_lateral_change_vs_target_only_rgb_m"] = (
            float(np.abs(compared - baseline).mean())
        )
    return report


def main() -> int:
    args = arguments()
    if args.samples <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("invalid diagnostic sample/loader settings")
    config_path = args.config.expanduser().resolve(strict=True)
    checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)
    result_path = args.closed_loop_result.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve(strict=False)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("test_locked_used") is not False:
        raise ValueError("diagnostic refuses configs that used test_locked")
    policy, architecture, device, loading = _load_policy_after_first_render(
        config_path, checkpoint_path, args.device
    )
    report = {
        "schema_version": 1,
        "stage": "next027_uwb_counterfactual_diagnostic",
        "test_locked_used": False,
        "checkpoint": str(checkpoint_path),
        "loading": loading,
        "validation": validation_counterfactuals(
            policy,
            architecture,
            device,
            config_path,
            config,
            samples=args.samples,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        ),
        "logged_closed_loop_blank_visual": logged_stream_counterfactuals(
            policy, architecture, device, result_path
        ),
        "logged_closed_loop_target_only_rgb": logged_target_only_rgb_counterfactuals(
            policy, architecture, device, result_path
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"event": "complete", "output": str(output), **report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
