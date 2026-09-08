"""Geometry and validation helpers for the repaired SAGE3D label sidecar.

The source dataset is immutable.  Repaired boxes and visibility decisions are
stored in a separate, versioned directory and are only consumable after an
independent admission report has passed.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

import numpy as np


SIDECAR_SCHEMA_VERSION = 1
GENERATION_SPEC_ID = "sage3d-bbox-depth-v1"


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def safe_source_path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe source-relative path: {relative}")
    candidate = (root / Path(*pure.parts)).resolve(strict=False)
    try:
        candidate.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise ValueError(f"path escapes source root: {relative}") from error
    return candidate


def episode_sidecar_path(sidecar_root: Path, source_path: str) -> Path:
    pure = PurePosixPath(source_path)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"unsafe SAGE3D episode path: {source_path}")
    return sidecar_root / "episodes" / Path(*pure.parts[:-1]) / f"{pure.parts[-1]}.json"


def _camera_basis(yaw: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Robot/base convention is x-forward, y-left, z-up.  Image x is right.
    forward = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)
    left = np.array([-math.sin(yaw), math.cos(yaw), 0.0], dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return forward, left, up


def project_target_bbox(
    step: Mapping[str, object],
    camera_info: Mapping[str, object],
    *,
    person_height_m: float = 1.7,
    person_width_m: float = 0.5,
    minimum_forward_m: float = 0.05,
) -> dict[str, object]:
    """Project the target capsule using the recorded camera translation.

    The old extractor dotted world displacement with the robot's *left* axis
    and added it to image x.  Here image x uses the true right axis, while the
    camera translation remains expressed in robot x-forward/y-left/z-up.
    """
    camera = camera_info.get("camera")
    if not isinstance(camera, Mapping):
        raise ValueError("camera_info.camera must be an object")
    if camera.get("model") != "pinhole" or camera.get("axes") != "ros":
        raise ValueError("only the recorded ROS-axis pinhole cameras are supported")
    intrinsics = camera.get("intrinsics")
    extrinsics = camera.get("extrinsics_robot_to_camera")
    if not isinstance(intrinsics, Mapping) or not isinstance(extrinsics, Mapping):
        raise ValueError("camera intrinsics/extrinsics are missing")
    translation = np.asarray(extrinsics.get("translation"), dtype=np.float64)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("camera translation must contain three finite values")

    robot = np.asarray(step.get("robot_pos"), dtype=np.float64)
    target = np.asarray(step.get("target_pos"), dtype=np.float64)
    if robot.shape != (3,) or target.shape != (3,):
        raise ValueError("robot_pos and target_pos must each contain three values")
    yaw = float(step["robot_yaw"])
    forward, left, up = _camera_basis(yaw)
    right = -left
    camera_origin = (
        robot
        + forward * translation[0]
        + left * translation[1]
        + up * translation[2]
    )
    expected_depth_m = float(np.dot(target - camera_origin, forward))
    if expected_depth_m <= minimum_forward_m:
        return {
            "bbox_xyxy": None,
            "raw_bbox_xyxy": None,
            "expected_depth_m": expected_depth_m,
            "projection_reason": "behind_camera",
        }

    width = int(camera["width"])
    height = int(camera["height"])
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    image_points: list[tuple[float, float]] = []
    half_width = 0.5 * float(person_width_m)
    for target_height, side in (
        (0.0, 0.0),
        (float(person_height_m), 0.0),
        (0.5 * float(person_height_m), half_width),
        (0.5 * float(person_height_m), -half_width),
    ):
        point = target.copy()
        point[2] = target_height
        point += right * side
        displacement = point - camera_origin
        camera_forward = float(np.dot(displacement, forward))
        if camera_forward <= minimum_forward_m:
            return {
                "bbox_xyxy": None,
                "raw_bbox_xyxy": None,
                "expected_depth_m": expected_depth_m,
                "projection_reason": "behind_camera",
            }
        camera_right = float(np.dot(displacement, right))
        camera_up = float(np.dot(displacement, up))
        image_points.append(
            (
                cx + fx * camera_right / camera_forward,
                cy - fy * camera_up / camera_forward,
            )
        )

    raw = [
        min(point[0] for point in image_points),
        min(point[1] for point in image_points),
        max(point[0] for point in image_points),
        max(point[1] for point in image_points),
    ]
    clipped = [
        max(0.0, raw[0]),
        max(0.0, raw[1]),
        min(float(width), raw[2]),
        min(float(height), raw[3]),
    ]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return {
            "bbox_xyxy": None,
            "raw_bbox_xyxy": raw,
            "expected_depth_m": expected_depth_m,
            "projection_reason": "out_of_view",
        }
    return {
        "bbox_xyxy": clipped,
        "raw_bbox_xyxy": raw,
        "expected_depth_m": expected_depth_m,
        "projection_reason": "projected",
    }


def depth_visibility_evidence(
    raw_depth: np.ndarray,
    bbox_xyxy: Sequence[float],
    expected_depth_m: float,
    *,
    depth_scale_to_m: float = 0.001,
    inner_margin_x: float = 0.20,
    inner_margin_y: float = 0.15,
    absolute_tolerance_m: float = 0.30,
    relative_tolerance: float = 0.15,
    minimum_support_fraction: float = 0.20,
    maximum_near_fraction: float = 0.50,
) -> dict[str, object]:
    """Classify visibility from aligned depth in the projected torso region."""
    depth = np.asarray(raw_depth)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2 or not depth.size:
        raise ValueError(f"depth image must be HxW, got {depth.shape}")
    metric = depth.astype(np.float32, copy=False) * float(depth_scale_to_m)
    height, width = metric.shape
    x0, y0, x1, y1 = map(float, bbox_xyxy)
    box_width, box_height = x1 - x0, y1 - y0
    xa = max(0, int(math.floor(x0 + inner_margin_x * box_width)))
    xb = min(width, int(math.ceil(x1 - inner_margin_x * box_width)))
    ya = max(0, int(math.floor(y0 + inner_margin_y * box_height)))
    yb = min(height, int(math.ceil(y1 - inner_margin_y * box_height)))
    if xb <= xa or yb <= ya:
        return {
            "valid_pixel_fraction": 0.0,
            "support_fraction": 0.0,
            "near_fraction": 0.0,
            "median_depth_m": None,
            "tolerance_m": max(absolute_tolerance_m, relative_tolerance * expected_depth_m),
            "visibility_reason": "empty_depth_region",
            "visible": False,
        }
    region = metric[ya:yb, xa:xb]
    valid_mask = np.isfinite(region) & (region > 0.05)
    values = region[valid_mask]
    valid_fraction = float(valid_mask.mean())
    tolerance = max(float(absolute_tolerance_m), float(relative_tolerance) * expected_depth_m)
    if not values.size:
        return {
            "valid_pixel_fraction": valid_fraction,
            "support_fraction": 0.0,
            "near_fraction": 0.0,
            "median_depth_m": None,
            "tolerance_m": tolerance,
            "visibility_reason": "missing_depth",
            "visible": False,
        }
    support = float(np.mean(np.abs(values - expected_depth_m) <= tolerance))
    near = float(np.mean(values < expected_depth_m - tolerance))
    if near > maximum_near_fraction:
        reason = "occluded_by_nearer_surface"
        visible = False
    elif support < minimum_support_fraction:
        reason = "depth_inconsistent"
        visible = False
    else:
        reason = "visible"
        visible = True
    return {
        "valid_pixel_fraction": valid_fraction,
        "support_fraction": support,
        "near_fraction": near,
        "median_depth_m": float(np.median(values)),
        "tolerance_m": tolerance,
        "visibility_reason": reason,
        "visible": visible,
    }


def load_admitted_sidecar(sidecar_root: Path) -> tuple[dict[str, object], dict[str, object]]:
    manifest_path = sidecar_root / "manifest.json"
    admission_path = sidecar_root / "admission.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    with admission_path.open("r", encoding="utf-8") as handle:
        admission = json.load(handle)
    if manifest.get("schema_version") != SIDECAR_SCHEMA_VERSION:
        raise ValueError("unsupported SAGE3D sidecar schema")
    if manifest.get("dataset_id") != "sage3d_extracted":
        raise ValueError("sidecar dataset_id must be sage3d_extracted")
    if manifest.get("generation_spec_id") != GENERATION_SPEC_ID:
        raise ValueError("unsupported SAGE3D generation spec")
    if manifest.get("selection") != "full":
        raise ValueError("a subset SAGE3D sidecar cannot be admitted for training")
    if admission.get("status") != "passed":
        raise ValueError("SAGE3D sidecar has not passed its admission audit")
    if admission.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("SAGE3D admission report does not match manifest.json")
    return manifest, admission


def validate_episode_sidecar(value: object, source_path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("SAGE3D episode sidecar must be an object")
    if value.get("schema_version") != SIDECAR_SCHEMA_VERSION:
        raise ValueError("unsupported SAGE3D episode sidecar schema")
    if value.get("generation_spec_id") != GENERATION_SPEC_ID:
        raise ValueError("SAGE3D episode sidecar generation spec mismatch")
    if value.get("source_path") != source_path:
        raise ValueError("SAGE3D episode sidecar source_path mismatch")
    if not isinstance(value.get("steps"), list):
        raise ValueError("SAGE3D episode sidecar steps must be a list")
    return value
