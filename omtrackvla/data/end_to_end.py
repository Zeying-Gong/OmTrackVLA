"""Raw-RGB input helpers for the Architecture v1 NEXT-025 smoke path."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from omtrackvla.models.end_to_end import ArchitectureV1Config


FORBIDDEN_DEPLOYMENT_KEYS = {
    "cache",
    "perception_cache",
    "visual_xy",
    "visual_confidence",
    "predicted_bbox",
    "target_bbox",
    "target_pose",
    "target_position",
    "target_point",
    "depth",
    "gt_motion",
}


def reject_forbidden_deployment_inputs(value: Mapping[str, Any]) -> None:
    """Reject labels, frozen perception products, and GT geometry recursively."""

    def visit(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                normalized = str(key).lower()
                if normalized in FORBIDDEN_DEPLOYMENT_KEYS or any(
                    token in normalized
                    for token in (
                        "perception_cache",
                        "future_bbox",
                        "external_bbox",
                        "target_pose",
                        "target_position",
                        "target_point",
                        "gt_depth",
                        "gt_motion",
                    )
                ):
                    raise ValueError(f"forbidden deployment input at {path}.{key}")
                visit(child, f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")

    visit(value, "model_inputs")


def _rgb_tensor(path: Path, config: ArchitectureV1Config) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize(
            (config.image_width, config.image_height), Image.Resampling.BILINEAR
        )
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def sage_camera_calibration(
    camera_info: str | Path | Mapping[str, Any],
    config: ArchitectureV1Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert the recorded ROS base/camera calibration to OpenCV projection."""

    if isinstance(camera_info, (str, Path)):
        value = json.loads(Path(camera_info).read_text(encoding="utf-8"))
    else:
        value = dict(camera_info)
    camera = value["camera"]
    width = float(camera["width"])
    height = float(camera["height"])
    source = torch.tensor(camera["intrinsics"]["k"], dtype=torch.float32)
    intrinsics = source.clone()
    intrinsics[0] *= config.image_width / width
    intrinsics[1] *= config.image_height / height
    intrinsics[2] = torch.tensor([0.0, 0.0, 1.0])

    extrinsics = camera.get("extrinsics_robot_to_camera", {})
    camera_link = extrinsics.get("camera_link")
    if camera.get("axes") != "ros" or camera_link not in {
        "base",
        "base_link",
        "pelvis",
    }:
        raise ValueError(
            "Architecture v1 only accepts audited ROS-axis robot-aligned cameras"
        )
    camera_center_base = torch.tensor(
        extrinsics.get("translation", [0.0, 0.0, 0.0]), dtype=torch.float32
    )
    # canonical base: x forward, y left, z up; OpenCV camera: x right, y down, z forward
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    camera_from_base = torch.eye(4, dtype=torch.float32)
    camera_from_base[:3, :3] = rotation
    camera_from_base[:3, 3] = -(rotation @ camera_center_base)
    return intrinsics, camera_from_base


def load_raw_rgb_smoke_batch(
    *,
    image_paths: Sequence[str | Path],
    camera_info: str | Path | Mapping[str, Any],
    initial_bbox_xyxy_norm: Sequence[float],
    config: ArchitectureV1Config | None = None,
    uwb_xy_m: Sequence[float] = (2.0, 0.0),
    uwb_covariance_m2: Sequence[Sequence[float]] = ((0.04, 0.0), (0.0, 0.04)),
    target_waypoints: Sequence[Sequence[float]] | None = None,
) -> dict[str, torch.Tensor]:
    """Load one real-image smoke batch without any perception cache or GT point input."""

    config = config or ArchitectureV1Config()
    config.validate()
    paths = [Path(path).expanduser().resolve(strict=True) for path in image_paths]
    if len(paths) != config.history_size + 1:
        raise ValueError(
            f"expected one initialization image plus {config.history_size} history images"
        )
    bbox = torch.tensor(initial_bbox_xyxy_norm, dtype=torch.float32)
    if bbox.shape != (4,) or not bool(
        (bbox >= 0).all()
        and (bbox <= 1).all()
        and bbox[2] > bbox[0]
        and bbox[3] > bbox[1]
    ):
        raise ValueError("initial bbox must be valid normalized xyxy")
    intrinsics, camera_from_base = sage_camera_calibration(camera_info, config)
    if target_waypoints is None:
        target_waypoints = [
            [0.15 * index, 0.02 * index] for index in range(config.horizon)
        ]
        target_waypoints[0] = [0.0, 0.0]
    target = torch.tensor(target_waypoints, dtype=torch.float32)
    if target.shape != (config.horizon, 2) or not torch.allclose(
        target[0], torch.zeros(2)
    ):
        raise ValueError("smoke waypoint target must be [8,2] with point zero at the origin")
    model_inputs = {
        "initial_rgb": _rgb_tensor(paths[0], config)[None],
        "initial_bbox": bbox[None],
        "ego_rgb": torch.stack([_rgb_tensor(path, config) for path in paths[1:]])[None],
        "visual_initialization_valid": torch.ones(1),
        "rgb_valid": torch.ones(1),
        "binding_valid": torch.ones(1),
        "uwb_xy": torch.tensor(uwb_xy_m, dtype=torch.float32)[None],
        "uwb_covariance_xy": torch.tensor(uwb_covariance_m2, dtype=torch.float32)[None],
        "uwb_quality": torch.ones(1),
        "uwb_age_s": torch.zeros(1),
        "uwb_valid": torch.ones(1),
        "camera_intrinsics": intrinsics[None],
        "camera_from_base": camera_from_base[None],
    }
    reject_forbidden_deployment_inputs(model_inputs)
    model_inputs["target_waypoints"] = target[None]
    model_inputs["waypoint_mask"] = torch.ones(1, config.horizon, dtype=torch.bool)
    return model_inputs


def load_sage3d_end_to_end_smoke_batch(
    *,
    root: str | Path,
    sidecar_root: str | Path,
    split_manifest: str | Path,
    policy_admission: str | Path,
    episode_relative: str,
    initial_index: int,
    anchor_index: int,
    config: ArchitectureV1Config | None = None,
) -> dict[str, torch.Tensor]:
    """Build one admitted, cache-free SAGE3D policy batch for NEXT-025."""

    from omtrackvla.data.sage3d_policy import (
        canonical_waypoints,
        load_policy_admission,
        sha256_file,
        target_position_base,
    )
    from omtrackvla.data.sage3d_sidecar import (
        episode_sidecar_path,
        load_admitted_sidecar,
        validate_episode_sidecar,
    )

    config = config or ArchitectureV1Config()
    config.validate()
    root = Path(root).expanduser().resolve(strict=True)
    sidecar_root = Path(sidecar_root).expanduser().resolve(strict=True)
    split_manifest = Path(split_manifest).expanduser().resolve(strict=True)
    policy_admission = Path(policy_admission).expanduser().resolve(strict=True)
    load_policy_admission(policy_admission, root)
    split_value = json.loads(split_manifest.read_text(encoding="utf-8"))
    if not isinstance(split_value, Mapping) or split_value.get("schema_version") != 1:
        raise ValueError("Phase 1 split manifest must use schema_version=1")
    unsigned_manifest = dict(split_value)
    expected_manifest_sha256 = unsigned_manifest.pop("manifest_sha256", None)
    actual_manifest_sha256 = hashlib.sha256(
        json.dumps(unsigned_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if expected_manifest_sha256 != actual_manifest_sha256:
        raise ValueError("Phase 1 split manifest checksum mismatch")
    episode_parts = PurePosixPath(episode_relative).parts
    if not episode_parts or PurePosixPath(episode_relative).is_absolute() or ".." in episode_parts:
        raise ValueError("unsafe SAGE3D episode path")
    sage_manifest = split_value.get("datasets", {}).get("sage3d_extracted", {})
    if sage_manifest.get("split_unit") != "run":
        raise ValueError("Phase 1 SAGE3D split unit must be run")
    if episode_parts[0] not in sage_manifest.get("splits", {}).get("train", []):
        raise ValueError("NEXT-025 smoke episode must belong to the Phase 1 train split")
    manifest, _ = load_admitted_sidecar(sidecar_root)
    metadata = manifest["episodes"].get(episode_relative)
    if not isinstance(metadata, Mapping):
        raise ValueError(f"episode is absent from the admitted sidecar: {episode_relative}")
    sidecar_path = episode_sidecar_path(sidecar_root, episode_relative)
    if sha256_file(sidecar_path) != metadata["sidecar_sha256"]:
        raise ValueError("SAGE3D episode sidecar checksum mismatch")
    sidecar = validate_episode_sidecar(
        json.loads(sidecar_path.read_text(encoding="utf-8")), episode_relative
    )
    episode = (root / episode_relative).resolve(strict=True)
    if root not in episode.parents:
        raise ValueError("SAGE3D episode escaped the configured root")
    derived_path = episode / "derived.json"
    camera_info_path = episode / "camera_info.json"
    if sha256_file(derived_path) != sidecar["source_derived_sha256"]:
        raise ValueError("SAGE3D derived.json no longer matches the admitted sidecar")
    if sha256_file(camera_info_path) != sidecar["source_camera_info_sha256"]:
        raise ValueError("SAGE3D camera_info.json no longer matches the admitted sidecar")
    derived = json.loads(derived_path.read_text(encoding="utf-8"))
    steps = derived["steps"]
    labels = sidecar["steps"]
    if not (0 <= initial_index < anchor_index < len(steps)):
        raise ValueError("SAGE3D initialization/anchor indices are invalid")
    if anchor_index + 21 >= len(steps):
        raise ValueError("SAGE3D anchor lacks the frozen +21 waypoint horizon")
    initial_label = labels[initial_index]
    if not initial_label.get("visible") or initial_label.get("bbox_xyxy") is None:
        raise ValueError("SAGE3D initialization must use an admitted visible bbox")
    camera_info_value = json.loads(camera_info_path.read_text(encoding="utf-8"))
    camera = camera_info_value["camera"]
    width, height = float(camera["width"]), float(camera["height"])
    x1, y1, x2, y2 = (float(value) for value in initial_label["bbox_xyxy"])
    bbox = [x1 / width, y1 / height, x2 / width, y2 / height]
    history_indices = list(
        range(anchor_index - config.history_size + 1, anchor_index + 1)
    )
    if history_indices[0] < 0:
        raise ValueError("SAGE3D anchor lacks the required RGB history")
    image_paths = [episode / "rgb" / f"{initial_index:05d}.jpg"] + [
        episode / "rgb" / f"{index:05d}.jpg" for index in history_indices
    ]
    return load_raw_rgb_smoke_batch(
        image_paths=image_paths,
        camera_info=camera_info_value,
        initial_bbox_xyxy_norm=bbox,
        config=config,
        uwb_xy_m=target_position_base(steps[anchor_index]),
        uwb_covariance_m2=((0.04, 0.0), (0.0, 0.04)),
        target_waypoints=canonical_waypoints(steps, anchor_index),
    )


def to_device(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}
