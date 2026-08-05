#!/usr/bin/env python3
"""First-frame smoke test for a modular person-following pipeline.

The perception and control implementations in this file are deliberately
replaceable.  The initial version uses Habitat panoptic and agent poses as
oracle inputs, while exposing the same target observation that a learned RGB
front end will produce later.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import imageio.v2 as imageio
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw


RGB_KEY = "agent_1_articulated_agent_jaw_rgb"
DEPTH_KEY = "agent_1_articulated_agent_jaw_depth"
PANOPTIC_KEY = "agent_1_articulated_agent_jaw_panoptic"
ACTION_NAMES = (
    "agent_0_humanoid_navigate_action",
    "agent_1_base_velocity",
    "agent_2_oracle_nav_randcoord_action_obstacle",
    "agent_3_oracle_nav_randcoord_action_obstacle",
    "agent_4_oracle_nav_randcoord_action_obstacle",
    "agent_5_oracle_nav_randcoord_action_obstacle",
    "agent_6_oracle_nav_randcoord_action_obstacle",
    "agent_7_oracle_nav_randcoord_action_obstacle",
    "agent_8_oracle_nav_randcoord_action_obstacle",
)
DEFAULT_SCENE_DATASET = (
    "data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
)


@dataclass(frozen=True)
class TargetObservation:
    visible: bool
    bbox_xyxy: Optional[Tuple[int, int, int, int]]
    footpoint_uv: Optional[Tuple[float, float]]
    relative_xy: Tuple[float, float]
    range_m: float
    bearing_rad: float
    mask_area: int
    confidence: float


@dataclass(frozen=True)
class ContinuousAction:
    forward: float
    lateral: float
    yaw: float

    def as_habitat(self) -> list[float]:
        return [self.forward, self.lateral, self.yaw]


@dataclass(frozen=True)
class ControlDecision:
    action: ContinuousAction
    mode: str
    guidance_bearing_rad: float
    waypoint_world: Optional[Tuple[float, float, float]]


def target_mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return an inclusive xyxy box for a binary target mask."""
    mask = np.asarray(mask, dtype=bool).squeeze()
    if mask.ndim != 2 or not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_to_footpoint(
    bbox: Optional[Tuple[int, int, int, int]],
) -> Optional[Tuple[float, float]]:
    """Use the detector-compatible bottom center of a bbox as pixel goal."""
    if bbox is None:
        return None
    x1, _, x2, y2 = bbox
    return (x1 + x2) / 2.0, float(y2)


def local_target(robot, target) -> Tuple[float, float]:
    """Return target position as robot-local (forward, left), in metres."""
    delta = target.base_pos - robot.base_pos
    local = robot.sim_obj.transformation.inverted().transform_vector(delta)
    return float(local.x), float(-local.z)


class OraclePerception:
    """Ground-truth target grounding from Habitat panoptic and agent poses."""

    def __call__(
        self,
        rgb: np.ndarray,
        panoptic: np.ndarray,
        target_semantic_id: int,
        relative_xy: Sequence[float],
    ) -> TargetObservation:
        del rgb  # Kept in the interface for a later RGB perception replacement.
        mask = np.asarray(panoptic).squeeze() == int(target_semantic_id)
        bbox = target_mask_to_bbox(mask)
        footpoint = bbox_to_footpoint(bbox)
        forward, left = float(relative_xy[0]), float(relative_xy[1])
        range_m = math.hypot(forward, left)
        bearing = math.atan2(left, forward)
        area = int(mask.sum())
        return TargetObservation(
            visible=bbox is not None,
            bbox_xyxy=bbox,
            footpoint_uv=footpoint,
            relative_xy=(forward, left),
            range_m=range_m,
            bearing_rad=bearing,
            mask_area=area,
            confidence=1.0 if bbox is not None else 0.0,
        )


class OracleFollowController:
    """Simple heading-gated continuous follower using oracle target geometry."""

    def __init__(
        self,
        desired_distance_m: float = 1.5,
        distance_gain: float = 0.6,
        heading_gain: float = 1.2,
        max_forward: float = 0.25,
        max_reverse: float = 0.15,
        max_yaw: float = 0.35,
        stop_forward_angle_deg: float = 55.0,
    ) -> None:
        self.desired_distance_m = desired_distance_m
        self.distance_gain = distance_gain
        self.heading_gain = heading_gain
        self.max_forward = max_forward
        self.max_reverse = max_reverse
        self.max_yaw = max_yaw
        self.stop_forward_angle = math.radians(stop_forward_angle_deg)

    def __call__(self, target: TargetObservation) -> ContinuousAction:
        if not target.visible:
            return ContinuousAction(0.0, 0.0, 0.0)

        yaw = float(np.clip(
            self.heading_gain * target.bearing_rad, -self.max_yaw, self.max_yaw
        ))
        distance_error = target.range_m - self.desired_distance_m
        forward = float(np.clip(
            self.distance_gain * distance_error, -self.max_reverse, self.max_forward
        ))

        # Rotate first for large target bearings; taper translation before cutoff.
        abs_bearing = abs(target.bearing_rad)
        if abs_bearing >= self.stop_forward_angle:
            forward = 0.0
        else:
            forward *= max(0.0, math.cos(abs_bearing))
        return ContinuousAction(forward, 0.0, yaw)


class OracleNavmeshFollower:
    """Continuous navmesh follower that maintains a safe distance band."""

    def __init__(
        self,
        min_distance_m: float = 1.25,
        max_distance_m: float = 1.75,
        max_forward: float = 1.0,
        max_lateral: float = 1.0,
        max_yaw: float = 1.0,
        heading_gain: float = 1.2,
        stop_forward_angle_deg: float = 55.0,
        retreat_step_m: float = 0.6,
        incoming_hold_distance_m: float = 3.0,
        incoming_retreat_distance_m: float = 2.0,
        incoming_motion_threshold_m: float = 0.03,
        incoming_memory_steps: int = 4,
        radial_distance_gain: float = 2.0,
        target_motion_gain: float = 2.5,
        target_motion_smoothing: float = 0.5,
        translation_slew_per_step: float = 0.25,
        emergency_translation_slew: float = 0.6,
        startup_approach_scale: float = 0.65,
        pass_yield_after_steps: int = 10,
        pass_yield_steps: int = 6,
        tracking_mask_min_pixels: int = 10000,
        approach_slowdown_distance_m: float = 0.75,
        hold_tolerance_m: float = 0.05,
        evasion_start_distance_m: float = 2.0,
        evasion_step_m: float = 0.9,
        tracking_distance_m: Optional[float] = None,
        prioritize_visibility: bool = False,
        incoming_safe_retreat_distance_m: float = 0.0,
        emergency_safe_retreat_distance_m: float = 0.0,
        tracking_mask_max_pixels: int = 0,
        visibility_reframe_after_steps: int = 0,
        coordinate_approach_min_scale: float = 0.1,
        lost_target_policy: str = "coordinate",
        lost_brake_steps: int = 2,
        lost_search_yaw: float = 0.35,
        lost_search_period_steps: int = 8,
        lost_coast_steps: int = 3,
        lost_coast_min_range_m: float = 2.0,
        lost_coast_max_translation: float = 0.35,
        lost_retreat_steps: int = 3,
    ) -> None:
        if min_distance_m >= max_distance_m:
            raise ValueError("min_distance_m must be smaller than max_distance_m")
        if lost_target_policy not in ("coordinate", "stop-search"):
            raise ValueError(f"Unsupported lost target policy: {lost_target_policy}")
        if lost_brake_steps < 0:
            raise ValueError("lost_brake_steps must be non-negative")
        if lost_search_period_steps <= 0:
            raise ValueError("lost_search_period_steps must be positive")
        if lost_coast_steps < 0:
            raise ValueError("lost_coast_steps must be non-negative")
        if lost_coast_max_translation < 0.0:
            raise ValueError("lost_coast_max_translation must be non-negative")
        if lost_retreat_steps < 0:
            raise ValueError("lost_retreat_steps must be non-negative")
        self.min_distance_m = min_distance_m
        self.max_distance_m = max_distance_m
        self.max_forward = max_forward
        self.max_lateral = max_lateral
        self.max_yaw = max_yaw
        self.heading_gain = heading_gain
        self.stop_forward_angle = math.radians(stop_forward_angle_deg)
        self.retreat_step_m = retreat_step_m
        self.incoming_hold_distance_m = incoming_hold_distance_m
        self.incoming_retreat_distance_m = incoming_retreat_distance_m
        self.incoming_motion_threshold_m = incoming_motion_threshold_m
        self.incoming_memory_steps = incoming_memory_steps
        self.radial_distance_gain = radial_distance_gain
        self.target_motion_gain = target_motion_gain
        self.target_motion_smoothing = target_motion_smoothing
        self.translation_slew_per_step = translation_slew_per_step
        self.emergency_translation_slew = emergency_translation_slew
        self.startup_approach_scale = startup_approach_scale
        self.pass_yield_after_steps = pass_yield_after_steps
        self.pass_yield_steps = pass_yield_steps
        self.tracking_mask_min_pixels = tracking_mask_min_pixels
        self.approach_slowdown_distance_m = approach_slowdown_distance_m
        self.hold_tolerance_m = hold_tolerance_m
        self.evasion_start_distance_m = evasion_start_distance_m
        self.evasion_step_m = evasion_step_m
        self.tracking_distance_m = (
            max_distance_m if tracking_distance_m is None else tracking_distance_m
        )
        self.prioritize_visibility = prioritize_visibility
        self.incoming_safe_retreat_distance_m = incoming_safe_retreat_distance_m
        self.emergency_safe_retreat_distance_m = emergency_safe_retreat_distance_m
        self.tracking_mask_max_pixels = tracking_mask_max_pixels
        self.visibility_reframe_after_steps = visibility_reframe_after_steps
        self.coordinate_approach_min_scale = coordinate_approach_min_scale
        self.lost_target_policy = lost_target_policy
        self.lost_brake_steps = int(lost_brake_steps)
        self.lost_search_yaw = float(lost_search_yaw)
        self.lost_search_period_steps = int(lost_search_period_steps)
        self.lost_coast_steps = int(lost_coast_steps)
        self.lost_coast_min_range_m = float(lost_coast_min_range_m)
        self.lost_coast_max_translation = float(lost_coast_max_translation)
        self.lost_retreat_steps = int(lost_retreat_steps)
        self.reset()

    def reset(self, evasion_side: Optional[float] = None) -> None:
        self._previous_target_position = None
        self._incoming_steps_remaining = 0
        self._previous_forward = 0.0
        self._previous_lateral = 0.0
        self._target_has_moved = False
        self._filtered_target_motion = np.zeros(3, dtype=np.float32)
        self._incoming_duration = 0
        self._pass_yield_steps_remaining = 0
        self._pass_yield_used = False
        self._pass_yield_active = False
        self._evasion_side = evasion_side
        self._low_mask_steps = 0
        self._lost_steps = 0
        self._last_seen_bearing = 0.0
        self._search_direction = 1.0

    def _lost_target_decision(self, target: TargetObservation) -> ControlDecision:
        self._lost_steps += 1
        if self._lost_steps == 1:
            if abs(self._last_seen_bearing) > 1e-3:
                self._search_direction = math.copysign(1.0, self._last_seen_bearing)
            elif self._evasion_side is not None:
                self._search_direction = float(self._evasion_side)

        retreat = bool(
            self._lost_steps <= self.lost_retreat_steps
            and target.range_m < self.min_distance_m
        )
        coast = bool(
            not retreat
            and
            self._lost_steps <= self.lost_coast_steps
            and target.range_m > self.lost_coast_min_range_m
        )
        if retreat:
            target_forward = math.cos(self._last_seen_bearing)
            target_left = math.sin(self._last_seen_bearing)
            forward = self._previous_forward
            lateral = self._previous_lateral
            if forward * target_forward + lateral * target_left >= -0.05:
                forward = -self.max_forward * target_forward
                lateral = -self.max_lateral * target_left
            forward, lateral = self._limit_translation(
                forward, lateral, emergency=True
            )
            mode = "lost_retreat"
            yaw = 0.0
        elif coast:
            forward, lateral = self._limit_translation(
                float(np.clip(
                    max(0.0, self._previous_forward),
                    0.0,
                    self.lost_coast_max_translation,
                )),
                float(np.clip(
                    self._previous_lateral,
                    -self.lost_coast_max_translation,
                    self.lost_coast_max_translation,
                )),
                emergency=True,
            )
            mode = "lost_coast"
            yaw = float(np.clip(
                self.heading_gain * self._last_seen_bearing,
                -self.max_yaw,
                self.max_yaw,
            ))
        else:
            forward, lateral = self._limit_translation(
                0.0, 0.0, emergency=True
            )
        if not retreat and not coast and (
            self._lost_steps <= self.lost_coast_steps + self.lost_brake_steps
        ):
            mode = "lost_brake"
            yaw = 0.0
        elif not retreat and not coast:
            search_step = (
                self._lost_steps - self.lost_coast_steps
                - self.lost_brake_steps - 1
            )
            phase = search_step // self.lost_search_period_steps
            direction = self._search_direction * (-1.0 if phase % 2 else 1.0)
            mode = "lost_search"
            yaw = float(np.clip(
                direction * self.lost_search_yaw, -self.max_yaw, self.max_yaw
            ))
        return ControlDecision(
            action=ContinuousAction(forward, lateral, yaw),
            mode=mode,
            guidance_bearing_rad=self._last_seen_bearing,
            waypoint_world=None,
        )

    def _update_incoming(self, robot_pos, target_pos) -> Tuple[bool, float]:
        target_toward_robot = 0.0
        target_motion = np.zeros(3, dtype=np.float32)
        if self._previous_target_position is not None:
            target_motion = target_pos - self._previous_target_position
            target_to_robot = robot_pos - target_pos
            target_motion[1] = 0.0
            target_to_robot[1] = 0.0
            norm = float(np.linalg.norm(target_to_robot))
            if norm > 1e-6:
                target_toward_robot = float(
                    np.dot(target_motion, target_to_robot / norm)
                )
            if float(np.linalg.norm(target_motion[[0, 2]])) > self.incoming_motion_threshold_m:
                self._target_has_moved = True
        alpha = self.target_motion_smoothing
        self._filtered_target_motion = (
            alpha * target_motion + (1.0 - alpha) * self._filtered_target_motion
        )
        self._previous_target_position = target_pos.copy()

        if target_toward_robot > self.incoming_motion_threshold_m:
            self._target_has_moved = True
            self._incoming_steps_remaining = self.incoming_memory_steps
        elif target_toward_robot < -self.incoming_motion_threshold_m:
            self._target_has_moved = True
            self._incoming_steps_remaining = 0
        elif self._incoming_steps_remaining:
            self._incoming_steps_remaining -= 1
        incoming = self._incoming_steps_remaining > 0
        if incoming:
            self._incoming_duration += 1
            if (
                not self._pass_yield_used
                and self._incoming_duration >= self.pass_yield_after_steps
            ):
                self._pass_yield_steps_remaining = self.pass_yield_steps
                self._pass_yield_used = True
        else:
            self._incoming_duration = 0
            self._pass_yield_steps_remaining = 0
            self._pass_yield_used = False
        self._pass_yield_active = self._pass_yield_steps_remaining > 0
        if self._pass_yield_active:
            self._pass_yield_steps_remaining -= 1
            if self._pass_yield_steps_remaining == 0:
                self._incoming_duration = 0
                self._pass_yield_used = False
        return incoming, target_toward_robot

    def _motion_tracking_translation(
        self,
        target: TargetObservation,
        target_motion_local: Tuple[float, float],
        desired_distance_m: Optional[float] = None,
    ) -> Tuple[float, float]:
        if desired_distance_m is None:
            desired_distance_m = self.min_distance_m
        radial_speed = self.radial_distance_gain * (
            target.range_m - desired_distance_m
        )
        bearing_forward = math.cos(target.bearing_rad)
        bearing_left = math.sin(target.bearing_rad)
        radial_motion = (
            target_motion_local[0] * bearing_forward
            + target_motion_local[1] * bearing_left
        )
        tangent_forward = target_motion_local[0] - radial_motion * bearing_forward
        tangent_left = target_motion_local[1] - radial_motion * bearing_left
        forward = (
            self.target_motion_gain * tangent_forward
            + radial_speed * bearing_forward
        )
        lateral = (
            self.target_motion_gain * tangent_left
            + radial_speed * bearing_left
        )
        return (
            float(np.clip(forward, -self.max_forward, self.max_forward)),
            float(np.clip(lateral, -self.max_lateral, self.max_lateral)),
        )

    def _limit_translation(self, forward: float, lateral: float, emergency=False):
        limit = (
            self.emergency_translation_slew
            if emergency else self.translation_slew_per_step
        )
        forward = float(np.clip(
            forward, self._previous_forward - limit, self._previous_forward + limit
        ))
        lateral = float(np.clip(
            lateral, self._previous_lateral - limit, self._previous_lateral + limit
        ))
        self._previous_forward = forward
        self._previous_lateral = lateral
        return forward, lateral

    @staticmethod
    def _local_xy(robot, world_point: Sequence[float]) -> Tuple[float, float]:
        delta = np.asarray(world_point, dtype=np.float32) - np.asarray(
            robot.base_pos, dtype=np.float32
        )
        local = robot.sim_obj.transformation.inverted().transform_vector(delta)
        return float(local.x), float(-local.z)

    @staticmethod
    def _shortest_path(sim, start, goal):
        import habitat_sim

        snapped_goal = np.asarray(sim.pathfinder.snap_point(goal), dtype=np.float32)
        if not np.isfinite(snapped_goal).all():
            return None
        path = habitat_sim.ShortestPath()
        path.requested_start = np.asarray(start, dtype=np.float32)
        path.requested_end = snapped_goal
        if not sim.pathfinder.find_path(path):
            return None
        return [np.asarray(point, dtype=np.float32) for point in path.points]

    def _retreat_goal(self, sim, robot_pos, target_pos):
        planar_away = np.asarray(robot_pos, dtype=np.float32) - np.asarray(
            target_pos, dtype=np.float32
        )
        planar_away[1] = 0.0
        norm = float(np.linalg.norm(planar_away))
        if norm < 1e-6:
            return None
        planar_away /= norm

        # Test several directions around "away" and keep a reachable point
        # that increases target clearance. This avoids blindly reversing into a wall.
        best = None
        for angle_deg in (0, -30, 30, -60, 60, -90, 90):
            angle = math.radians(angle_deg)
            x, z = float(planar_away[0]), float(planar_away[2])
            direction = np.array(
                [x * math.cos(angle) - z * math.sin(angle), 0.0,
                 x * math.sin(angle) + z * math.cos(angle)],
                dtype=np.float32,
            )
            candidate = np.asarray(robot_pos, dtype=np.float32) + direction * self.retreat_step_m
            path = self._shortest_path(sim, robot_pos, candidate)
            if not path:
                continue
            goal = path[-1]
            clearance = float(np.linalg.norm((goal - target_pos)[[0, 2]]))
            path_length = sum(
                float(np.linalg.norm((b - a)[[0, 2]])) for a, b in zip(path, path[1:])
            )
            score = clearance - 0.2 * path_length
            if best is None or score > best[0]:
                best = (score, goal)
        return None if best is None else best[1]

    def _path_waypoint(self, sim, robot, goal):
        path = self._shortest_path(sim, robot.base_pos, goal)
        if not path:
            return None
        robot_pos = np.asarray(robot.base_pos, dtype=np.float32)
        for point in path[1:]:
            if float(np.linalg.norm((point - robot_pos)[[0, 2]])) >= 0.2:
                return point
        return path[-1]

    def _evasion_waypoint(self, sim, robot, target_pos):
        robot_pos = np.asarray(robot.base_pos, dtype=np.float32)
        motion = np.asarray(self._filtered_target_motion, dtype=np.float32).copy()
        motion[1] = 0.0
        norm = float(np.linalg.norm(motion))
        if norm < 1e-6:
            motion = robot_pos - np.asarray(target_pos, dtype=np.float32)
            motion[1] = 0.0
            norm = float(np.linalg.norm(motion))
        if norm < 1e-6:
            return None
        motion /= norm
        perpendicular = np.array([-motion[2], 0.0, motion[0]], dtype=np.float32)

        candidates = []
        sides = (self._evasion_side,) if self._evasion_side is not None else (-1.0, 1.0)
        for side in sides:
            candidate = robot_pos + side * perpendicular * self.evasion_step_m
            path = self._shortest_path(sim, robot_pos, candidate)
            if not path:
                continue
            endpoint = path[-1]
            lateral_progress = float(np.dot(endpoint - robot_pos, side * perpendicular))
            path_length = sum(
                float(np.linalg.norm((b - a)[[0, 2]]))
                for a, b in zip(path, path[1:])
            )
            candidates.append((lateral_progress - 0.1 * path_length, side, endpoint))
        if not candidates:
            return None
        _, self._evasion_side, goal = max(candidates, key=lambda item: item[0])
        return self._path_waypoint(sim, robot, goal)

    def __call__(self, sim, robot, target_agent, target: TargetObservation) -> ControlDecision:
        if not target.visible and self.lost_target_policy == "stop-search":
            return self._lost_target_decision(target)

        if target.visible:
            self._last_seen_bearing = target.bearing_rad
            if self._lost_steps:
                # Do not interpret the displacement accumulated while visual
                # tracking was lost as one frame of target motion.
                self._previous_target_position = None
                self._filtered_target_motion.fill(0.0)
                self._incoming_steps_remaining = 0
                self._incoming_duration = 0
                self._pass_yield_steps_remaining = 0
                self._pass_yield_active = False
                self._lost_steps = 0

        target_pos = np.asarray(target_agent.base_pos, dtype=np.float32)
        robot_pos = np.asarray(robot.base_pos, dtype=np.float32)
        coordinate_takeover = not target.visible
        control_target = target
        if coordinate_takeover:
            forward, left = self._local_xy(robot, target_pos)
            control_target = TargetObservation(
                visible=False,
                bbox_xyxy=None,
                footpoint_uv=None,
                relative_xy=(forward, left),
                range_m=math.hypot(forward, left),
                bearing_rad=math.atan2(left, forward),
                mask_area=0,
                confidence=0.0,
            )
        target_available = target.visible or coordinate_takeover
        incoming, target_toward_robot = self._update_incoming(robot_pos, target_pos)

        mask_too_small = bool(
            target.visible and target.mask_area <= self.tracking_mask_min_pixels
        )
        mask_too_large = bool(
            target.visible
            and self.tracking_mask_max_pixels > 0
            and target.mask_area >= self.tracking_mask_max_pixels
        )
        mask_valid = target.visible and not mask_too_small and not mask_too_large
        self._low_mask_steps = self._low_mask_steps + 1 if mask_too_small else 0

        safe_retreat = bool(
            sim is not None
            and target_available
            and (
                (
                    self.incoming_safe_retreat_distance_m > 0.0
                    and incoming
                    and control_target.range_m < self.incoming_safe_retreat_distance_m
                )
                or (
                    self.emergency_safe_retreat_distance_m > 0.0
                    and control_target.range_m < self.emergency_safe_retreat_distance_m
                )
            )
        )
        safe_retreat_goal = (
            self._retreat_goal(sim, robot_pos, target_pos) if safe_retreat else None
        )
        safe_retreat_waypoint = (
            self._path_waypoint(sim, robot, safe_retreat_goal)
            if safe_retreat_goal is not None else None
        )
        reframe_visibility = bool(
            sim is not None
            and self.visibility_reframe_after_steps > 0
            and self._low_mask_steps >= self.visibility_reframe_after_steps
            and target.range_m <= self.max_distance_m + 0.2
        )
        reframe_waypoint = (
            self._evasion_waypoint(sim, robot, target_pos)
            if reframe_visibility else None
        )

        # Yielding is only safe while the full incoming clearance is available.
        # The previous 0.8 m cutoff could stop the robot inside the 0.5 m
        # collision envelope plus one discrete humanoid stride.
        if safe_retreat_waypoint is not None:
            mode = "retreat_safety"
            waypoint = safe_retreat_waypoint
        elif reframe_waypoint is not None:
            mode = "reframe_visibility"
            waypoint = reframe_waypoint
        elif mask_too_large:
            mode = "track_visibility_clearance"
            waypoint = None
        elif (
            self._pass_yield_active
            and target.range_m >= self.incoming_retreat_distance_m
        ):
            mode = "yield_pass"
            waypoint = None
        elif (
            incoming
            and target_available
            and control_target.range_m <= self.incoming_hold_distance_m
        ):
            mode = "track_incoming"
            waypoint = None
        elif (
            target_available
            and control_target.range_m < self.max_distance_m - self.hold_tolerance_m
            and self._target_has_moved
            and (
                not self.prioritize_visibility
                or mask_valid
            )
        ):
            mode = "track_distance"
            waypoint = None
        elif (
            not self._target_has_moved
            and target_available
            and control_target.range_m <= self.incoming_retreat_distance_m
        ):
            # Do not consume the clearance before a stationary leader starts
            # moving. Several episodes begin inside 2 m and then send the
            # humanoid directly toward the robot.
            mode = "hold_startup"
            waypoint = None
        elif (
            control_target.range_m > self.max_distance_m + self.hold_tolerance_m
            or mask_too_small
        ):
            mode = (
                "approach_visibility"
                if mask_too_small
                else "approach"
            )
            waypoint = self._path_waypoint(sim, robot, target_pos)
        else:
            mode = "hold"
            waypoint = None

        if mode.startswith("track_"):
            guidance_bearing = control_target.bearing_rad
            target_motion_local = self._local_xy(
                robot, robot_pos + self._filtered_target_motion
            )
            forward, lateral = self._motion_tracking_translation(
                control_target,
                target_motion_local,
                desired_distance_m=(
                    self.incoming_retreat_distance_m
                    if mode == "track_incoming" else self.tracking_distance_m
                ),
            )
            if (
                mode == "track_incoming"
                and sim is not None
                and control_target.range_m <= self.evasion_start_distance_m
            ):
                evasion_waypoint = self._evasion_waypoint(
                    sim, robot, target_pos
                )
                if evasion_waypoint is not None:
                    _, evasion_left = self._local_xy(robot, evasion_waypoint)
                    if abs(evasion_left) > 0.05:
                        lateral = float(np.clip(
                            evasion_left / 0.35,
                            -self.max_lateral,
                            self.max_lateral,
                        ))
                        mode = "evade_incoming"
        elif waypoint is None:
            guidance_bearing = control_target.bearing_rad
            forward = 0.0
            lateral = 0.0
        elif mode == "retreat_safety":
            local_forward, local_left = self._local_xy(robot, waypoint)
            norm = math.hypot(local_forward, local_left)
            if norm > 1e-6:
                forward = float(np.clip(
                    local_forward / norm, -self.max_forward, self.max_forward
                ))
                lateral = float(np.clip(
                    local_left / norm, -self.max_lateral, self.max_lateral
                ))
            else:
                forward = 0.0
                lateral = 0.0
            guidance_bearing = control_target.bearing_rad
        else:
            local_forward, local_left = self._local_xy(robot, waypoint)
            guidance_bearing = math.atan2(local_left, local_forward)
            abs_bearing = abs(guidance_bearing)
            distance_scale = float(np.clip(
                (control_target.range_m - self.max_distance_m)
                / self.approach_slowdown_distance_m,
                (
                    self.coordinate_approach_min_scale
                    if coordinate_takeover else 0.1
                ),
                1.0,
            ))
            forward = (
                self.max_forward
                * (1.0 if self._target_has_moved else self.startup_approach_scale)
                * distance_scale
                if abs_bearing < self.stop_forward_angle else 0.0
            )
            forward *= max(0.0, math.cos(abs_bearing))
            lateral = float(np.clip(
                self.max_lateral * math.sin(guidance_bearing),
                -self.max_lateral,
                self.max_lateral,
            ))

        forward, lateral = self._limit_translation(
            float(forward),
            float(lateral),
            emergency=control_target.range_m < self.min_distance_m,
        )

        yaw_bearing = (
            control_target.bearing_rad
            if mode.startswith("track_")
            or mode in ("yield_pass", "retreat_safety", "reframe_visibility")
            else guidance_bearing
        )
        yaw = float(np.clip(
            self.heading_gain * yaw_bearing, -self.max_yaw, self.max_yaw
        ))
        waypoint_tuple = (
            None if waypoint is None
            else tuple(float(value) for value in np.asarray(waypoint))
        )
        if coordinate_takeover:
            mode += "_coordinate"
        return ControlDecision(
            ContinuousAction(float(forward), lateral, yaw),
            mode,
            float(guidance_bearing),
            waypoint_tuple,
        )


def select_episode(dataset, episode_id: str, scene_substring: str, match_index: int = 0):
    matches = [
        episode
        for episode in dataset.episodes
        if str(episode.episode_id) == str(episode_id)
        and scene_substring in episode.scene_id
    ]
    if not matches:
        raise ValueError(
            f"No episode id={episode_id!r} in scene containing {scene_substring!r}"
        )
    if match_index < 0 or match_index >= len(matches):
        raise ValueError(
            f"match_index={match_index} is outside the {len(matches)} matching episodes"
        )
    return matches[match_index], len(matches)


def configure(config, scene_dataset: str):
    from habitat.config import read_write

    with read_write(config):
        config.habitat.simulator.scene_dataset = scene_dataset
        # These measures are unnecessary for a one-frame smoke test and some
        # require a complete episode rollout.
        measurements = config.habitat.task.measurements
        for name in ("top_down_map_following",):
            if name in measurements:
                del measurements[name]
    return config


def annotate(
    rgb: np.ndarray,
    target: TargetObservation,
    decision: ControlDecision,
    title: str,
    official_following: Optional[float] = None,
) -> Image.Image:
    image = Image.fromarray(np.asarray(rgb)[..., :3].astype(np.uint8))
    draw = ImageDraw.Draw(image)
    width, height = image.size
    draw.line((width // 2, 0, width // 2, height), fill=(30, 144, 255), width=2)
    if target.bbox_xyxy is not None:
        draw.rectangle(target.bbox_xyxy, outline=(255, 40, 40), width=4)
    if target.footpoint_uv is not None:
        u, v = target.footpoint_uv
        radius = 5
        draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=(255, 220, 0))

    lines = [
        title,
        f"visible={target.visible} range={target.range_m:.3f}m bearing={math.degrees(target.bearing_rad):.1f}deg",
        f"mode={decision.mode} official_following={official_following}",
        f"action=[forward={decision.action.forward:.3f}, lateral={decision.action.lateral:.3f}, yaw={decision.action.yaw:.3f}]",
    ]
    y = 8
    for line in lines:
        box = draw.textbbox((8, y), line)
        draw.rectangle((box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2), fill=(0, 0, 0))
        draw.text((8, y), line, fill=(255, 255, 255))
        y = box[3] + 6
    return image


def main() -> None:
    import habitat
    from habitat.datasets import make_dataset

    import evt_bench  # noqa: F401 - registers TrackEnv, sensors, and actions

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("stt", "dt", "at"), default="stt")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--episode-id", default="4")
    parser.add_argument("--scene", default="16tymPtM7uS")
    parser.add_argument("--match-index", type=int, default=1)
    parser.add_argument("--scene-dataset", default=DEFAULT_SCENE_DATASET)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--min-distance", type=float, default=1.25)
    parser.add_argument("--max-distance", type=float, default=1.75)
    parser.add_argument("--max-forward", type=float, default=1.0)
    parser.add_argument("--max-lateral", type=float, default=1.0)
    parser.add_argument("--max-yaw", type=float, default=1.0)
    parser.add_argument("--video-fps", type=int, default=4)
    parser.add_argument("--output-dir", default="outputs/oracle_modular_follow")
    args = parser.parse_args()

    config_kind = "train" if args.split == "train" else "infer"
    config_path = (
        f"habitat-lab/habitat/config/benchmark/nav/track/"
        f"track_{config_kind}_{args.task}.yaml"
    )
    config = configure(habitat.get_config(config_path), args.scene_dataset)
    dataset = make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
    episode, match_count = select_episode(
        dataset, args.episode_id, args.scene, args.match_index
    )
    dataset.episodes = [episode]

    with habitat.TrackEnv(config=config, dataset=dataset) as env:
        observations = env.reset()
        missing = [key for key in (RGB_KEY, PANOPTIC_KEY) if key not in observations]
        if missing:
            raise KeyError(f"Missing required observations {missing}; got {sorted(observations)}")

        robot = env.sim.agents_mgr[1].articulated_agent
        target_agent = env.sim.agents_mgr[0].articulated_agent
        target_semantic_id = int(episode.info["main_human_semantic_id"])

        perception = OraclePerception()
        controller = OracleNavmeshFollower(
            min_distance_m=args.min_distance,
            max_distance_m=args.max_distance,
            max_forward=args.max_forward,
            max_lateral=args.max_lateral,
            max_yaw=args.max_yaw,
        )

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = (
            f"{args.task}_{args.split}_{args.scene}_episode_{args.episode_id}"
            f"_match_{args.match_index}_steps_{args.steps}"
        )
        run_dir = output_dir / stem
        frame_dir = run_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        rgb_path = run_dir / "frame_000_rgb.jpg"
        vis_path = run_dir / "frame_000_oracle.jpg"
        video_path = run_dir / "oracle_follow.mp4"
        json_path = output_dir / f"{stem}.json"

        frames = []
        step_records = []
        for step_index in range(args.steps + 1):
            relative_xy = local_target(robot, target_agent)
            target = perception(
                observations[RGB_KEY], observations[PANOPTIC_KEY],
                target_semantic_id, relative_xy,
            )
            decision = controller(env.sim, robot, target_agent, target)
            metrics = env.get_metrics()
            official_following = metrics.get("human_following")
            collision = metrics.get("human_collision")

            annotated = annotate(
                observations[RGB_KEY], target, decision,
                f"{args.task.upper()} {args.split} | {args.scene} | ep {args.episode_id} | step {step_index}",
                official_following,
            )
            frame_array = np.asarray(annotated)
            frames.append(frame_array)
            frame_path = frame_dir / f"{step_index:03d}.jpg"
            annotated.save(frame_path, quality=95)
            if step_index == 0:
                iio.imwrite(
                    rgb_path, np.asarray(observations[RGB_KEY])[..., :3], quality=95
                )
                annotated.save(vis_path, quality=95)

            step_records.append({
                "step": step_index,
                "perception": asdict(target),
                "control": {
                    **asdict(decision),
                    "habitat_action": decision.action.as_habitat(),
                    "normalized": True,
                },
                "official_human_following": official_following,
                "human_collision": collision,
                "frame": str(frame_path),
            })
            print(json.dumps({
                "step": step_index,
                "range_m": target.range_m,
                "mask_area": target.mask_area,
                "mode": decision.mode,
                "action": decision.action.as_habitat(),
                "official_human_following": official_following,
                "human_collision": collision,
            }), flush=True)

            if step_index == args.steps or env.episode_over:
                break
            observations = env.step({
                "action": ACTION_NAMES,
                "action_args": {
                    "agent_1_base_vel": decision.action.as_habitat(),
                },
            })

        if len(frames) > 1:
            imageio.mimsave(video_path, frames, fps=args.video_fps, macro_block_size=1)

        # Match trained_agent.py: metrics are counted after each executed action,
        # so the reset observation at record 0 is excluded.
        evaluated_steps = step_records[1:]
        following_steps = sum(
            float(record["official_human_following"] or 0.0)
            for record in evaluated_steps
        )
        collision = max(
            (float(record["human_collision"] or 0.0) for record in evaluated_steps),
            default=0.0,
        )
        total_steps = len(evaluated_steps)
        final_metrics = env.get_metrics()
        if total_steps < config.habitat.environment.max_episode_steps:
            success = float(bool(
                final_metrics.get("human_following_success", 0.0)
                and final_metrics.get("human_following", 0.0)
            ))
        else:
            success = float(bool(final_metrics.get("human_following", 0.0)))
        distances = [
            float(record["perception"]["range_m"]) for record in evaluated_steps
        ]
        summary = {
            "finish": bool(env.episode_over),
            "success": success,
            "following_rate": following_steps / total_steps if total_steps else 0.0,
            "following_step": following_steps,
            "total_step": total_steps,
            "collision": collision,
            "distance_start_m": distances[0] if distances else None,
            "distance_min_m": min(distances) if distances else None,
            "distance_mean_m": float(np.mean(distances)) if distances else None,
            "distance_max_m": max(distances) if distances else None,
            "distance_end_m": distances[-1] if distances else None,
            "final_metrics": {
                key: float(final_metrics[key])
                for key in (
                    "human_following", "human_following_success",
                    "human_collision", "distance_to_leader",
                )
                if key in final_metrics and np.isscalar(final_metrics[key])
            },
        }

        result = {
            "schema_version": 1,
            "config": config_path,
            "task": args.task,
            "split": args.split,
            "scene_id": episode.scene_id,
            "episode_id": str(episode.episode_id),
            "episode_match_index": args.match_index,
            "episode_match_count": match_count,
            "robot_start_position": episode.info.get("robot_position"),
            "instruction": episode.info.get("instruction"),
            "target_semantic_id": target_semantic_id,
            "tracking_limits": {
                "official_following_max_distance_m": 3.0,
                "official_success_distance_m": [1.0, 3.0],
                "official_facing_min_pixels": 10000,
                "official_facing_max_fraction": 0.3,
                "human_collision_distance_m": 0.5,
                "controller_distance_band_m": [
                    args.min_distance, args.max_distance
                ],
                "controller_max_forward": args.max_forward,
                "controller_max_lateral": args.max_lateral,
                "controller_max_yaw": args.max_yaw,
            },
            "requested_steps": args.steps,
            "executed_actions": max(0, len(step_records) - 1),
            "summary": summary,
            "steps": step_records,
            "rgb": str(rgb_path),
            "visualization": str(vis_path),
            "video": str(video_path) if len(frames) > 1 else None,
        }
        json_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({
            "event": "oracle_modular_follow_complete",
            "summary": summary,
            "json": str(json_path),
            "video": result["video"],
        }, indent=2))


if __name__ == "__main__":
    main()
