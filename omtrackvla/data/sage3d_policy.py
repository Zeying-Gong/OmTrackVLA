"""Canonical SAGE3D person-follow policy geometry and timing helpers."""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


POLICY_SPEC_ID = "sage3d-policy-se2-30hz-v1"
TRANSFORM_SPEC_ID = "sage3d-habitat-world-to-base-se2-v1"
CLOCK_SPEC_ID = "sage3d-control-step-30hz-v1"
SIMULATION_SPEC_ID = "sage3d-pose-derived-noiseless-uwb-v1"
CONTROL_HZ = 30.0
WAYPOINT_HORIZON = 8
WAYPOINT_STRIDE_STEPS = 3


def _planar_pose(step: Mapping[str, object], key: str) -> np.ndarray:
    value = np.asarray(step.get(key), dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"{key} must contain three finite world coordinates")
    return value[:2]


def base_basis(yaw: float) -> tuple[np.ndarray, np.ndarray]:
    """Return world-frame unit vectors for canonical base forward and left."""
    yaw = float(yaw)
    if not math.isfinite(yaw):
        raise ValueError("robot_yaw must be finite")
    return (
        np.asarray((math.cos(yaw), math.sin(yaw)), dtype=np.float64),
        np.asarray((-math.sin(yaw), math.cos(yaw)), dtype=np.float64),
    )


def world_delta_to_base(delta_world_xy: Sequence[float], yaw: float) -> list[float]:
    delta = np.asarray(delta_world_xy, dtype=np.float64)
    if delta.shape != (2,) or not np.isfinite(delta).all():
        raise ValueError("world planar delta must contain two finite values")
    forward, left = base_basis(yaw)
    return [float(np.dot(delta, forward)), float(np.dot(delta, left))]


def target_position_base(step: Mapping[str, object]) -> list[float]:
    robot = _planar_pose(step, "robot_pos")
    target = _planar_pose(step, "target_pos")
    return world_delta_to_base(target - robot, float(step["robot_yaw"]))


def canonical_waypoints(
    steps: Sequence[Mapping[str, object]],
    anchor_index: int,
    *,
    horizon: int = WAYPOINT_HORIZON,
    stride_steps: int = WAYPOINT_STRIDE_STEPS,
) -> list[list[float]]:
    """Build absolute local positions at offsets 0, stride, ..., (H-1)*stride.

    Source ``waypoints_ego`` starts at the next control step.  The WP-1
    contract instead requires the current base pose as point zero, so policy
    adapters must recompute the trajectory from recorded robot poses.
    """
    anchor_index = int(anchor_index)
    horizon = int(horizon)
    stride_steps = int(stride_steps)
    if anchor_index < 0 or horizon < 2 or stride_steps < 1:
        raise ValueError("invalid waypoint request")
    final_index = anchor_index + (horizon - 1) * stride_steps
    if final_index >= len(steps):
        raise ValueError("insufficient future robot poses for waypoint horizon")
    anchor = steps[anchor_index]
    anchor_position = _planar_pose(anchor, "robot_pos")
    yaw = float(anchor["robot_yaw"])
    waypoints = []
    for point_index in range(horizon):
        future = steps[anchor_index + point_index * stride_steps]
        future_position = _planar_pose(future, "robot_pos")
        waypoints.append(world_delta_to_base(future_position - anchor_position, yaw))
    waypoints[0] = [0.0, 0.0]
    return waypoints


def source_waypoints(
    steps: Sequence[Mapping[str, object]],
    anchor_index: int,
    *,
    horizon: int = WAYPOINT_HORIZON,
    stride_steps: int = WAYPOINT_STRIDE_STEPS,
) -> list[list[float]]:
    """Reproduce the immutable extractor's +1,+4,...,+22 convention."""
    anchor_index = int(anchor_index)
    final_index = anchor_index + 1 + (int(horizon) - 1) * int(stride_steps)
    if anchor_index < 0 or final_index >= len(steps):
        raise ValueError("insufficient future poses for source waypoint convention")
    anchor = steps[anchor_index]
    anchor_position = _planar_pose(anchor, "robot_pos")
    yaw = float(anchor["robot_yaw"])
    return [
        world_delta_to_base(
            _planar_pose(steps[anchor_index + 1 + point * stride_steps], "robot_pos")
            - anchor_position,
            yaw,
        )
        for point in range(int(horizon))
    ]


def waypoint_time_offsets_s(
    *,
    horizon: int = WAYPOINT_HORIZON,
    stride_steps: int = WAYPOINT_STRIDE_STEPS,
    control_hz: float = CONTROL_HZ,
) -> list[float]:
    if horizon < 2 or stride_steps < 1 or not math.isfinite(control_hz) or control_hz <= 0:
        raise ValueError("invalid waypoint timing")
    return [point * stride_steps / control_hz for point in range(horizon)]


def moving_speed_ratio(
    steps: Sequence[Mapping[str, object]],
    commanded_speed_mps: float,
    *,
    control_hz: float = CONTROL_HZ,
) -> float | None:
    """Estimate recorded/commanded target speed while excluding waits/acceleration."""
    speed = float(commanded_speed_mps)
    if len(steps) < 2 or speed <= 0.0 or control_hz <= 0.0:
        return None
    positions = np.asarray([_planar_pose(step, "target_pos") for step in steps])
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    moving = distances[distances >= 0.25 * speed / control_hz]
    if moving.size < 3:
        return None
    return float(np.median(moving) * control_hz / speed)
