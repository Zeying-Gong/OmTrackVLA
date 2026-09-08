"""Explicit camera/base transformations for Phase 1 geometry supervision.

The canonical planar frame follows the WP-1 contract: x points forward, y
points left, z points up, and positive yaw is counter-clockwise about +z.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _matrix4(value: np.ndarray | Iterable[float], name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.size != 16:
        raise ValueError(f"{name} must contain 16 values")
    matrix = matrix.reshape(4, 4)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains a non-finite value")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{name} is not a homogeneous transform")
    return matrix


def _se2_from_relative(relative: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            relative[0, 3],
            relative[1, 3],
            math.atan2(relative[1, 0], relative[0, 0]),
        ],
        dtype=np.float32,
    )


def relative_w2c_to_base_se2(
    anchor_world_to_camera: np.ndarray | Iterable[float],
    target_world_to_camera: np.ndarray | Iterable[float],
    camera_from_base: np.ndarray | Iterable[float],
) -> np.ndarray:
    """Convert two OpenCV-style w2c poses into canonical base-frame SE(2).

    ``camera_from_base`` is ``^C T_B``. DA3 emits ``^C_t T_W``. The
    conversion therefore follows the frozen convention exactly::

        ^W T_C_t = inverse(^C_t T_W)
        ^W T_B_t = ^W T_C_t @ ^C T_B
        ^B_t T_B_u = inverse(^W T_B_t) @ ^W T_B_u
    """

    anchor_w2c = _matrix4(anchor_world_to_camera, "anchor_world_to_camera")
    target_w2c = _matrix4(target_world_to_camera, "target_world_to_camera")
    camera_base = _matrix4(camera_from_base, "camera_from_base")
    world_base_anchor = np.linalg.inv(anchor_w2c) @ camera_base
    world_base_target = np.linalg.inv(target_w2c) @ camera_base
    return _se2_from_relative(np.linalg.inv(world_base_anchor) @ world_base_target)


def intern_pair_to_canonical_se2(
    anchor_world_from_camera: np.ndarray | Iterable[float],
    target_world_from_camera: np.ndarray | Iterable[float],
    base_from_camera: np.ndarray | Iterable[float],
) -> np.ndarray:
    """Convert an InternData-N1 pose pair to x-forward/y-left SE(2).

    InternData-N1 stores ``action`` as a camera pose in the Habitat world and
    stores the fixed camera extrinsic as ``^B T_C``. Its planar body axes are
    represented as ``(right, forward)`` in the resulting matrix. The final
    ``(old_y, -old_x)`` basis change is the same conversion used by the
    audited InternNav loader, but is explicit here instead of being hidden in
    waypoint code.
    """

    world_camera_anchor = _matrix4(anchor_world_from_camera, "anchor_world_from_camera")
    world_camera_target = _matrix4(target_world_from_camera, "target_world_from_camera")
    base_camera = _matrix4(base_from_camera, "base_from_camera")
    world_base_anchor = world_camera_anchor @ np.linalg.inv(base_camera)
    world_base_target = world_camera_target @ np.linalg.inv(base_camera)
    relative = np.linalg.inv(world_base_anchor) @ world_base_target
    old = _se2_from_relative(relative)
    return np.asarray([old[1], -old[0], old[2]], dtype=np.float32)


def robust_translation_scale(
    predicted_xy: np.ndarray,
    reference_xy: np.ndarray,
    minimum_motion_m: float = 0.05,
) -> float:
    """Return a robust metric scale from matched planar displacements.

    Arbitrary-scale clips with no reliable moving pair are rejected rather
    than silently returning a magic scale.
    """

    predicted = np.asarray(predicted_xy, dtype=np.float64)
    reference = np.asarray(reference_xy, dtype=np.float64)
    if predicted.shape != reference.shape or predicted.ndim != 2 or predicted.shape[1] != 2:
        raise ValueError("predicted_xy and reference_xy must both have shape [N, 2]")
    if not np.isfinite(predicted).all() or not np.isfinite(reference).all():
        raise ValueError("scale inputs contain non-finite values")
    predicted_norm = np.linalg.norm(predicted, axis=1)
    reference_norm = np.linalg.norm(reference, axis=1)
    valid = (predicted_norm >= minimum_motion_m) & (reference_norm >= minimum_motion_m)
    if not np.any(valid):
        raise ValueError("no displacement is large enough for scale calibration")
    scale = float(np.median(reference_norm[valid] / predicted_norm[valid]))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("calibrated scale must be finite and positive")
    return scale
