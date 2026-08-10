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
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import imageio.v2 as imageio
import imageio.v3 as iio
import cv2
import numpy as np
from PIL import Image, ImageDraw

from modular_obstacle_map import LocalObstacleMap


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


class NavmeshConnectivityDiagnostic:
    """Visualize raw NavMesh and a conservative Spot-clearance component."""

    def __init__(
        self,
        meters_per_pixel: float = 0.05,
        spot_radius_m: float = 0.50,
    ) -> None:
        self.meters_per_pixel = float(meters_per_pixel)
        self.spot_radius_m = float(spot_radius_m)
        self.reset()

    def reset(self) -> None:
        self._raw_free = None
        self._raw_labels = None
        self._spot_free = None
        self._spot_labels = None
        self._bounds = None
        self._raw_radius_m = None
        self.last_visualization = None
        self.last_connectivity = None
        self.initial_connectivity = None

    def _to_grid(self, position):
        if self._bounds is None or self._raw_free is None:
            return None
        lower, upper = self._bounds
        span_z = max(float(upper[2] - lower[2]), 1e-6)
        span_x = max(float(upper[0] - lower[0]), 1e-6)
        row = int(round(
            (float(position[2]) - float(lower[2]))
            * (self._raw_free.shape[0] - 1) / span_z
        ))
        col = int(round(
            (float(position[0]) - float(lower[0]))
            * (self._raw_free.shape[1] - 1) / span_x
        ))
        return col, row

    @staticmethod
    def _nearest_free(mask, point, max_radius_px):
        if point is None:
            return None
        x0, y0 = point
        height, width = mask.shape
        best = None
        for radius in range(max_radius_px + 1):
            x1, x2 = max(0, x0 - radius), min(width - 1, x0 + radius)
            y1, y2 = max(0, y0 - radius), min(height - 1, y0 + radius)
            for y in range(y1, y2 + 1):
                for x in range(x1, x2 + 1):
                    if max(abs(x - x0), abs(y - y0)) != radius or not mask[y, x]:
                        continue
                    distance_sq = (x - x0) ** 2 + (y - y0) ** 2
                    if best is None or distance_sq < best[0]:
                        best = (distance_sq, (x, y))
            if best is not None:
                return best[1]
        return None

    def _initialize(self, sim, height_m: float) -> None:
        pathfinder = sim.pathfinder
        raw_free = np.asarray(pathfinder.get_topdown_view(
            meters_per_pixel=self.meters_per_pixel,
            height=float(height_m),
        ), dtype=bool)
        self._raw_free = np.ascontiguousarray(raw_free)
        _, self._raw_labels = cv2.connectedComponents(
            self._raw_free.astype(np.uint8), connectivity=8
        )
        self._bounds = tuple(
            np.asarray(value, dtype=np.float32) for value in pathfinder.get_bounds()
        )
        settings = getattr(pathfinder, "nav_mesh_settings", None)
        self._raw_radius_m = float(getattr(settings, "agent_radius", 0.30))
        extra_radius_px = max(0, int(math.ceil(
            (self.spot_radius_m - self._raw_radius_m) / self.meters_per_pixel
        )))
        if extra_radius_px:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * extra_radius_px + 1, 2 * extra_radius_px + 1),
            )
            self._spot_free = cv2.erode(
                self._raw_free.astype(np.uint8), kernel, iterations=1
            ).astype(bool)
        else:
            self._spot_free = self._raw_free.copy()
        _, self._spot_labels = cv2.connectedComponents(
            self._spot_free.astype(np.uint8), connectivity=8
        )

    def update(self, sim, robot_position, target_position) -> None:
        robot_position = np.asarray(robot_position, dtype=np.float32)
        target_position = np.asarray(target_position, dtype=np.float32)
        if self._raw_free is None:
            self._initialize(sim, float(robot_position[1]))

        raw_robot_px = self._nearest_free(
            self._raw_free, self._to_grid(robot_position),
            max(1, int(round(1.0 / self.meters_per_pixel))),
        )
        raw_target_px = self._nearest_free(
            self._raw_free, self._to_grid(target_position),
            max(1, int(round(1.0 / self.meters_per_pixel))),
        )
        spot_robot_px = self._nearest_free(
            self._spot_free, self._to_grid(robot_position),
            max(1, int(round(1.0 / self.meters_per_pixel))),
        )
        spot_target_px = self._nearest_free(
            self._spot_free, self._to_grid(target_position),
            max(1, int(round(1.0 / self.meters_per_pixel))),
        )

        raw_connected = False
        geodesic_distance_m = None
        path_points = []
        try:
            import habitat_sim

            snapped_robot = np.asarray(
                sim.pathfinder.snap_point(robot_position), dtype=np.float32
            )
            snapped_target = np.asarray(
                sim.pathfinder.snap_point(target_position), dtype=np.float32
            )
            if np.isfinite(snapped_robot).all() and np.isfinite(snapped_target).all():
                path = habitat_sim.ShortestPath()
                path.requested_start = snapped_robot
                path.requested_end = snapped_target
                raw_connected = bool(sim.pathfinder.find_path(path))
                if raw_connected:
                    geodesic_distance_m = float(path.geodesic_distance)
                    path_points = [self._to_grid(point) for point in path.points]
        except (AttributeError, ImportError, RuntimeError, ValueError):
            raw_robot_label = (
                int(self._raw_labels[raw_robot_px[1], raw_robot_px[0]])
                if raw_robot_px is not None else 0
            )
            raw_target_label = (
                int(self._raw_labels[raw_target_px[1], raw_target_px[0]])
                if raw_target_px is not None else 0
            )
            raw_connected = bool(
                raw_robot_label > 0 and raw_robot_label == raw_target_label
            )

        robot_label = (
            int(self._spot_labels[spot_robot_px[1], spot_robot_px[0]])
            if spot_robot_px is not None else 0
        )
        target_label = (
            int(self._spot_labels[spot_target_px[1], spot_target_px[0]])
            if spot_target_px is not None else 0
        )
        spot_connected = bool(
            robot_label > 0 and target_label > 0 and robot_label == target_label
        )
        connectivity = {
            "raw_connected": raw_connected,
            "spot_connected": spot_connected,
            "raw_navmesh_radius_m": self._raw_radius_m,
            "spot_radius_m": self.spot_radius_m,
            "raw_geodesic_distance_m": geodesic_distance_m,
            "robot_spot_component": robot_label,
            "target_spot_component": target_label,
        }
        self.last_connectivity = connectivity
        if self.initial_connectivity is None:
            self.initial_connectivity = dict(connectivity)

        canvas = np.full((*self._raw_free.shape, 3), 45, dtype=np.uint8)
        canvas[self._raw_free] = (115, 115, 115)
        canvas[self._spot_free] = (225, 225, 225)
        if robot_label > 0:
            canvas[self._spot_labels == robot_label] = (185, 215, 245)
        if target_label > 0 and target_label != robot_label:
            canvas[self._spot_labels == target_label] = (245, 205, 185)
        valid_path = [point for point in path_points if point is not None]
        if len(valid_path) >= 2:
            cv2.polylines(
                canvas,
                [np.asarray(valid_path, dtype=np.int32).reshape((-1, 1, 2))],
                False,
                (0, 220, 220),
                2,
                cv2.LINE_AA,
            )
        if spot_robot_px is not None:
            cv2.circle(canvas, spot_robot_px, 4, (30, 70, 230), -1)
        if spot_target_px is not None:
            target_color = (40, 190, 60) if spot_connected else (220, 55, 45)
            cv2.circle(canvas, spot_target_px, 4, target_color, -1)

        points = [point for point in (spot_robot_px, spot_target_px) if point is not None]
        points.extend(valid_path)
        if points:
            xs, ys = zip(*points)
            margin = max(20, int(round(3.0 / self.meters_per_pixel)))
            min_size = max(120, int(round(12.0 / self.meters_per_pixel)))
            cx, cy = int(round((min(xs) + max(xs)) / 2)), int(round((min(ys) + max(ys)) / 2))
            span = max(max(xs) - min(xs), max(ys) - min(ys)) + 2 * margin
            span = min(max(span, min_size), max(canvas.shape[:2]))
            x1 = max(0, min(canvas.shape[1] - span, cx - span // 2))
            y1 = max(0, min(canvas.shape[0] - span, cy - span // 2))
            canvas = canvas[y1:y1 + span, x1:x1 + span]

        distance_label = (
            f"{geodesic_distance_m:.2f}m" if geodesic_distance_m is not None else "none"
        )
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 48), (255, 255, 255), -1)
        cv2.putText(
            canvas,
            f"NAVMESH raw={'YES' if raw_connected else 'NO'} spot={'YES' if spot_connected else 'NO'}",
            (5, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (15, 15, 15), 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"path={distance_label} raw_r={self._raw_radius_m:.2f} spot_r={self.spot_radius_m:.2f}",
            (5, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.27, (15, 15, 15), 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "gray=raw white=Spot blue=robot island",
            (5, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (15, 15, 15), 1, cv2.LINE_AA,
        )
        self.last_visualization = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)


def target_mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return an inclusive xyxy box for a binary target mask."""
    mask = np.asarray(mask, dtype=bool).squeeze()
    if mask.ndim != 2 or not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def mask_connected_bboxes(
    mask: np.ndarray, min_area: int = 20
) -> list[Tuple[int, int, int, int]]:
    """Split a semantic mask into inclusive boxes for visible components."""
    binary = np.asarray(mask, dtype=bool).squeeze()
    if binary.ndim != 2 or not binary.any():
        return []
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    boxes = []
    for label in range(1, count):
        x, y, width, height, area = (int(v) for v in stats[label])
        if area < int(min_area):
            continue
        boxes.append((x, y, x + width - 1, y + height - 1))
    boxes.sort(key=lambda box: (box[0], box[1]))
    return boxes


def select_target_component_bbox(
    mask: np.ndarray,
    relative_xy: Sequence[float],
    image_width: int,
    previous_bbox: Optional[Tuple[int, int, int, int]] = None,
    depth: Optional[np.ndarray] = None,
    projected_uv: Optional[Tuple[float, float]] = None,
) -> Optional[Tuple[int, int, int, int]]:
    """Select the semantic component consistent with target pose and history."""
    boxes = mask_connected_bboxes(mask)
    if not boxes:
        return None
    person_sized = [
        box for box in boxes
        if box[2] - box[0] + 1 >= 10 and box[3] - box[1] + 1 >= 20
    ]
    if person_sized:
        boxes = person_sized
    forward, left = (float(v) for v in relative_xy)
    half_width = 0.5 * float(image_width)
    expected_u = half_width
    if forward > 0.05:
        expected_u += half_width * left / forward
    expected_u = float(np.clip(expected_u, 0.0, max(0.0, image_width - 1.0)))
    previous_center = None
    if previous_bbox is not None:
        previous_center = 0.5 * (previous_bbox[0] + previous_bbox[2])
    metric_depth = None
    if depth is not None:
        metric_depth = np.asarray(depth, dtype=np.float32).squeeze()
        finite = metric_depth[np.isfinite(metric_depth)]
        if finite.size and float(np.nanmax(finite)) <= 1.0 + 1e-3:
            metric_depth = metric_depth * 10.0
    expected_range = math.hypot(forward, left)

    def box_depth(box):
        if metric_depth is None or metric_depth.shape != np.asarray(mask).squeeze().shape:
            return None
        x1, y1, x2, y2 = box
        component_mask = np.asarray(mask, dtype=bool).squeeze()[y1:y2 + 1, x1:x2 + 1]
        values = metric_depth[y1:y2 + 1, x1:x2 + 1][component_mask]
        values = values[np.isfinite(values) & (values > 0.05)]
        return float(np.median(values)) if values.size else None

    def score(box):
        center = 0.5 * (box[0] + box[2])
        pose_error = abs(center - expected_u) / max(float(image_width), 1.0)
        temporal_error = (
            abs(center - previous_center) / max(float(image_width), 1.0)
            if previous_center is not None else 0.0
        )
        measured_depth = box_depth(box)
        depth_error = (
            abs(measured_depth - expected_range) / max(expected_range, 0.5)
            if measured_depth is not None else 0.5
        )
        if projected_uv is not None:
            box_u = 0.5 * (box[0] + box[2])
            box_v = float(box[3])
            projection_error = math.hypot(
                box_u - float(projected_uv[0]),
                box_v - float(projected_uv[1]),
            ) / max(float(image_width), 1.0)
            return 2.0 * projection_error + 0.35 * depth_error + 0.10 * temporal_error
        return depth_error + 0.25 * pose_error + 0.10 * temporal_error

    return min(boxes, key=score)


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


def project_world_to_sensor(
    sim,
    sensor_key: str,
    world_point,
    image_shape,
    hfov_deg: float = 90.0,
) -> Optional[Tuple[float, float]]:
    """Project a world point into a Habitat pinhole sensor image."""
    try:
        sensor = sim._sensors[sensor_key]._sensor_object
        camera_point = sensor.node.absolute_transformation().inverted().transform_point(
            np.asarray(world_point, dtype=np.float32)
        )
        x, y, z = (float(camera_point[i]) for i in range(3))
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    forward = -z
    if forward <= 0.05:
        return None
    height, width = int(image_shape[0]), int(image_shape[1])
    fx = 0.5 * width / math.tan(math.radians(hfov_deg) * 0.5)
    vfov = 2.0 * math.atan((height / max(width, 1)) * math.tan(math.radians(hfov_deg) * 0.5))
    fy = 0.5 * height / math.tan(vfov * 0.5)
    return (
        0.5 * (width - 1) + fx * x / forward,
        0.5 * (height - 1) - fy * y / forward,
    )


class OraclePerception:
    """Ground-truth target grounding from Habitat panoptic and agent poses."""

    def __init__(self) -> None:
        self.reset()

    def reset(self, reference_rgb=None) -> None:
        del reference_rgb
        self._last_bbox = None
        self.last_candidate_diagnostics = []

    def __call__(
        self,
        rgb: np.ndarray,
        panoptic: np.ndarray,
        target_semantic_id: int,
        relative_xy: Sequence[float],
        depth: Optional[np.ndarray] = None,
        projected_uv: Optional[Tuple[float, float]] = None,
    ) -> TargetObservation:
        mask = np.asarray(panoptic).squeeze() == int(target_semantic_id)
        boxes = mask_connected_bboxes(mask)
        bbox = select_target_component_bbox(
            mask,
            relative_xy,
            image_width=np.asarray(rgb).shape[1],
            previous_bbox=self._last_bbox,
            depth=depth,
            projected_uv=projected_uv,
        )
        self.last_candidate_diagnostics = [
            {
                "bbox_xyxy": box,
                "confidence": 1.0,
                "selected": box == bbox,
                "semantic_id": int(target_semantic_id),
                "depth_m": self._component_depth(mask, box, depth),
                "projected_uv": projected_uv,
            }
            for box in boxes
        ]
        if bbox is not None:
            self._last_bbox = bbox
        footpoint = bbox_to_footpoint(bbox)
        forward, left = float(relative_xy[0]), float(relative_xy[1])
        range_m = math.hypot(forward, left)
        bearing = math.atan2(left, forward)
        area = 0
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            area = int(mask[y1:y2 + 1, x1:x2 + 1].sum())
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

    @staticmethod
    def _component_depth(mask, box, depth):
        if depth is None:
            return None
        metric = np.asarray(depth, dtype=np.float32).squeeze()
        finite = metric[np.isfinite(metric)]
        if finite.size and float(np.nanmax(finite)) <= 1.0 + 1e-3:
            metric = metric * 10.0
        binary = np.asarray(mask, dtype=bool).squeeze()
        if metric.shape != binary.shape:
            return None
        x1, y1, x2, y2 = box
        values = metric[y1:y2 + 1, x1:x2 + 1][binary[y1:y2 + 1, x1:x2 + 1]]
        values = values[np.isfinite(values) & (values > 0.05)]
        return float(np.median(values)) if values.size else None


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


class ModularReactiveFollower:
    """Reactive point-goal control using only a TargetObservation.

    It deliberately ignores Habitat simulator state, target-agent poses, and
    navmesh paths so it can isolate the control side of the modular pipeline.
    """

    def __init__(
        self,
        min_distance_m: float = 1.2,
        max_distance_m: float = 1.5,
        max_forward: float = 1.0,
        max_lateral: float = 1.0,
        max_yaw: float = 1.0,
        distance_gain: float = 1.0,
        lateral_gain: float = 0.8,
        heading_gain: float = 1.5,
        lost_search_yaw: float = 0.35,
        use_invisible_pointgoal: bool = False,
    ) -> None:
        if min_distance_m >= max_distance_m:
            raise ValueError("min_distance_m must be smaller than max_distance_m")
        self.min_distance_m = float(min_distance_m)
        self.max_distance_m = float(max_distance_m)
        self.max_forward = float(max_forward)
        self.max_lateral = float(max_lateral)
        self.max_yaw = float(max_yaw)
        self.distance_gain = float(distance_gain)
        self.lateral_gain = float(lateral_gain)
        self.heading_gain = float(heading_gain)
        self.lost_search_yaw = float(lost_search_yaw)
        self.use_invisible_pointgoal = bool(use_invisible_pointgoal)
        self.uses_invisible_pointgoal = self.use_invisible_pointgoal
        self.lost_retreat_steps = 0
        self.reset()

    def reset(self, evasion_side: Optional[float] = None) -> None:
        self._lost_steps = 0
        self._last_seen_bearing = 0.0
        self._search_direction = 1.0 if evasion_side is None else float(evasion_side)
        self._evasion_side = evasion_side

    def __call__(self, sim, robot, target_agent, target: TargetObservation) -> ControlDecision:
        del sim, robot, target_agent
        if not target.visible and not self.use_invisible_pointgoal:
            self._lost_steps += 1
            if self._lost_steps == 1 and abs(self._last_seen_bearing) > 1e-3:
                self._search_direction = math.copysign(1.0, self._last_seen_bearing)
            yaw = float(np.clip(
                self._search_direction * self.lost_search_yaw,
                -self.max_yaw,
                self.max_yaw,
            ))
            return ControlDecision(
                ContinuousAction(0.0, 0.0, yaw),
                "reactive_search",
                self._last_seen_bearing,
                None,
            )

        self._lost_steps = 0
        self._last_seen_bearing = float(target.bearing_rad)
        distance_error = float(target.range_m - self.max_distance_m)
        forward = self.distance_gain * distance_error
        if abs(target.bearing_rad) > math.radians(55.0):
            forward = 0.0
        else:
            forward *= max(0.0, math.cos(target.bearing_rad))
        lateral = self.lateral_gain * float(target.relative_xy[1])
        forward = float(np.clip(forward, -self.max_forward, self.max_forward))
        lateral = float(np.clip(lateral, -self.max_lateral, self.max_lateral))
        yaw = float(np.clip(
            self.heading_gain * target.bearing_rad,
            -self.max_yaw,
            self.max_yaw,
        ))
        if not target.visible:
            mode = "reactive_pointgoal"
        else:
            mode = "reactive_approach" if distance_error > 0.05 else "reactive_hold"
        return ControlDecision(
            ContinuousAction(forward, lateral, yaw),
            mode,
            float(target.bearing_rad),
            None,
        )


class MapReactiveFollower(ModularReactiveFollower):
    """Reactive follower augmented with an Ascent-style local obstacle map."""

    def __init__(
        self,
        *args,
        hfov_deg: float = 90.0,
        camera_height_m: float = 0.24,
        camera_pitch_deg: float = 5.0,
        camera_forward_offset_m: float = 0.0,
        camera_left_offset_m: float = 0.0,
        min_obstacle_height_m: float = 0.06,
        max_obstacle_height_m: float = 1.8,
        robot_radius_m: float = 0.30,
        min_static_hits: int = 1,
        map_memory_frames: Optional[int] = None,
        catchup_bias_mps: float = 0.12,
        planning_hysteresis_m: float = 0.20,
        predictive_human_safety: bool = False,
        motion_blocked_feedback: bool = False,
        safety_prediction_steps: float = 5.0,
        safety_release_margin_m: float = 0.20,
        motion_min_command: float = 0.25,
        motion_min_displacement_m: float = 0.035,
        motion_max_block_yaw: float = 0.35,
        incoming_activation_margin_m: float = 0.35,
        incoming_closing_threshold_m: float = 0.02,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.catchup_bias_mps = float(catchup_bias_mps)
        self.planning_hysteresis_m = max(0.0, float(planning_hysteresis_m))
        self.predictive_human_safety = bool(predictive_human_safety)
        self.motion_blocked_feedback = bool(motion_blocked_feedback)
        self.safety_prediction_steps = max(0.0, float(safety_prediction_steps))
        self.safety_release_margin_m = max(0.0, float(safety_release_margin_m))
        self.motion_min_command = max(0.0, float(motion_min_command))
        self.motion_min_displacement_m = max(
            0.0, float(motion_min_displacement_m)
        )
        self.motion_max_block_yaw = max(0.0, float(motion_max_block_yaw))
        self.incoming_activation_margin_m = max(
            0.0, float(incoming_activation_margin_m)
        )
        self.incoming_closing_threshold_m = max(
            0.0, float(incoming_closing_threshold_m)
        )
        self.obstacle_map = LocalObstacleMap(
            hfov_deg=hfov_deg,
            camera_height_m=camera_height_m,
            camera_pitch_deg=camera_pitch_deg,
            camera_forward_offset_m=camera_forward_offset_m,
            camera_left_offset_m=camera_left_offset_m,
            min_obstacle_height_m=min_obstacle_height_m,
            max_obstacle_height_m=max_obstacle_height_m,
            robot_radius_m=robot_radius_m,
            min_static_hits=min_static_hits,
            memory_frames=map_memory_frames,
        )
        self.last_map_mode = "map_uninitialized"
        self.last_map_clearance = dict(self.obstacle_map.last_clearance)
        self.last_map_visualization = self.obstacle_map.visualize()
        self.last_navmesh_calibration = None
        self.last_navmesh_calibration_visualization = None
        self._navmesh_navigable_map = None
        self._target_history_episode = deque(maxlen=24)
        self._visibility_portal_episode = None
        self.last_visibility_portal_episode = None
        self.last_visibility_portal_distance_m = None
        self._target_was_visible = False
        self._previous_target_episode = None
        self._target_velocity_episode_ema = np.zeros(2, dtype=np.float32)
        self._target_closing_ema = 0.0
        self._human_safety_active = False
        self._human_safety_steps = 0
        self._human_evasion_direction = self._search_direction
        self._occluded_pass_active = False
        self._occluded_pass_steps = 0
        self._post_pass_reframe_steps = 0
        self.last_target_closing_m_per_step = 0.0
        self.last_predicted_closest_range_m = None
        self._last_translation_command = None
        self._last_translation_pose = None
        self._motion_stall_steps = 0
        self._motion_stall_direction = None
        self.last_motion_displacement_m = None
        self.last_motion_progress_m = None
        self.last_motion_block_added = False

    def reset(self, evasion_side: Optional[float] = None) -> None:
        super().reset(evasion_side=evasion_side)
        if hasattr(self, "obstacle_map"):
            self.obstacle_map.reset()
            self._episode_origin = None
            self._episode_forward_axis = None
            self._episode_left_axis = None
            self._map_evasion_direction = self._search_direction
            self._map_blocked_steps = 0
            self._target_history_episode.clear()
            self._visibility_portal_episode = None
            self.last_visibility_portal_episode = None
            self.last_visibility_portal_distance_m = None
            self._target_was_visible = False
            self._previous_target_episode = None
            self._target_velocity_episode_ema = np.zeros(2, dtype=np.float32)
            self._target_closing_ema = 0.0
            self._human_safety_active = False
            self._human_safety_steps = 0
            self._human_evasion_direction = self._search_direction
            self._occluded_pass_active = False
            self._occluded_pass_steps = 0
            self._post_pass_reframe_steps = 0
            self.last_target_closing_m_per_step = 0.0
            self.last_predicted_closest_range_m = None
            self._last_translation_command = None
            self._last_translation_pose = None
            self._motion_stall_steps = 0
            self._motion_stall_direction = None
            self.last_motion_displacement_m = None
            self.last_motion_progress_m = None
            self.last_motion_block_added = False
            self.last_navmesh_calibration = None
            self.last_navmesh_calibration_visualization = None
            self._navmesh_navigable_map = None
            self.last_map_mode = "map_reset"

    def update_navmesh_calibration(self, sim) -> None:
        """Align Habitat NavMesh to the episodic obstacle-map grid."""
        if self._episode_origin is None:
            return
        obstacle_map = self.obstacle_map
        if self._navmesh_navigable_map is None:
            size = obstacle_map.grid_size_px
            navmesh = np.zeros((size, size), dtype=bool)
            origin = np.asarray(self._episode_origin, dtype=np.float32)
            forward_axis = np.asarray(self._episode_forward_axis, dtype=np.float32)
            left_axis = np.asarray(self._episode_left_axis, dtype=np.float32)
            for gy in range(size):
                forward = (obstacle_map.center_px - gy) / obstacle_map.pixels_per_meter
                for gx in range(size):
                    left = (obstacle_map.center_px - gx) / obstacle_map.pixels_per_meter
                    world = origin + forward * forward_axis + left * left_axis
                    navmesh[gy, gx] = bool(sim.pathfinder.is_navigable(world))
            self._navmesh_navigable_map = navmesh

        observed = (obstacle_map.static_hits > 0) | (obstacle_map.free_hits > 0)
        predicted_obstacle = obstacle_map.static_map.astype(bool)
        navmesh_settings = getattr(sim.pathfinder, "nav_mesh_settings", None)
        navmesh_radius_m = float(
            getattr(navmesh_settings, "agent_radius", obstacle_map.robot_radius_m)
        )
        calibration_radius_px = max(
            0, int(math.ceil(navmesh_radius_m * obstacle_map.pixels_per_meter))
        )
        if calibration_radius_px > 0:
            radius = calibration_radius_px
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
            )
            predicted_obstacle = cv2.dilate(
                predicted_obstacle.astype(np.uint8), kernel, iterations=1
            ).astype(bool)
        predicted_free = observed & ~predicted_obstacle
        navmesh_free = self._navmesh_navigable_map
        false_obstacle = observed & predicted_obstacle & navmesh_free
        unsafe_free = predicted_free & ~navmesh_free
        comparable_navmesh_free = observed & navmesh_free
        intersection = int(np.sum(predicted_free & comparable_navmesh_free))
        union = int(np.sum(predicted_free | comparable_navmesh_free))
        obstacle_count = int(np.sum(observed & predicted_obstacle))
        free_count = int(np.sum(predicted_free))
        calibration = {
            "observed_cells": int(np.sum(observed)),
            "navmesh_agent_radius_m": navmesh_radius_m,
            "control_map_robot_radius_m": obstacle_map.robot_radius_m,
            "false_obstacle_cells": int(np.sum(false_obstacle)),
            "unsafe_free_cells": int(np.sum(unsafe_free)),
            "false_obstacle_rate": (
                float(np.sum(false_obstacle)) / obstacle_count
                if obstacle_count else 0.0
            ),
            "unsafe_free_rate": (
                float(np.sum(unsafe_free)) / free_count if free_count else 0.0
            ),
            "navigable_iou": float(intersection) / union if union else 1.0,
        }
        self.last_navmesh_calibration = calibration

        canvas = np.full((*observed.shape, 3), 205, dtype=np.uint8)
        canvas[observed & navmesh_free] = (225, 245, 225)
        canvas[observed & ~navmesh_free] = (95, 95, 95)
        canvas[observed & predicted_obstacle & ~navmesh_free] = (35, 35, 35)
        canvas[false_obstacle] = (230, 70, 70)
        canvas[unsafe_free] = (70, 120, 235)
        robot_gx, robot_gy = obstacle_map._grid(
            np.array([obstacle_map.robot_pose[0]]),
            np.array([obstacle_map.robot_pose[1]]),
        )
        if 0 <= robot_gx[0] < canvas.shape[1] and 0 <= robot_gy[0] < canvas.shape[0]:
            cv2.circle(
                canvas, (int(robot_gx[0]), int(robot_gy[0])), 4, (0, 0, 255), -1
            )
        cv2.putText(
            canvas,
            "LOCAL CHECK: red=false obstacle blue=unsafe free",
            (4, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            (
                f"FO={calibration['false_obstacle_rate']:.3f} "
                f"UF={calibration['unsafe_free_rate']:.3f} "
                f"IoU={calibration['navigable_iou']:.3f}"
            ),
            (4, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        self.last_navmesh_calibration_visualization = cv2.cvtColor(
            canvas, cv2.COLOR_RGB2BGR
        )

    def update_observation(
        self, observations, target: TargetObservation, dynamic_mask=None, robot=None, target_agent=None
    ) -> None:
        depth = observations.get(DEPTH_KEY)
        if depth is None:
            self.last_map_mode = "map_missing_depth"
            return
        if dynamic_mask is not None:
            dynamic_mask = np.asarray(dynamic_mask, dtype=bool)
        elif target.bbox_xyxy is not None:
            dynamic_mask = np.zeros(np.asarray(depth).squeeze().shape, dtype=bool)
            x1, y1, x2, y2 = target.bbox_xyxy
            dynamic_mask[
                max(0, int(y1)):min(dynamic_mask.shape[0], int(y2) + 1),
                max(0, int(x1)):min(dynamic_mask.shape[1], int(x2) + 1),
            ] = True
        robot_pose = (0.0, 0.0, 0.0)
        if robot is not None:
            position = np.asarray(robot.base_pos, dtype=np.float32)
            transform = robot.sim_obj.transformation
            forward_axis = np.asarray(
                transform.transform_vector(np.array([1.0, 0.0, 0.0]))
            )
            left_axis = np.asarray(
                transform.transform_vector(np.array([0.0, 0.0, -1.0]))
            )
            if self._episode_origin is None:
                self._episode_origin = position.copy()
                self._episode_forward_axis = forward_axis.copy()
                self._episode_left_axis = left_axis.copy()
            delta = position - self._episode_origin
            episode_forward = float(np.dot(delta, self._episode_forward_axis))
            episode_left = float(np.dot(delta, self._episode_left_axis))
            yaw = math.atan2(
                float(np.dot(forward_axis, self._episode_left_axis)),
                float(np.dot(forward_axis, self._episode_forward_axis)),
            )
            robot_pose = (episode_forward, episode_left, yaw)
        self.obstacle_map.update(
            depth,
            target_bbox=target.bbox_xyxy,
            dynamic_mask=dynamic_mask,
            robot_pose=robot_pose,
        )
        self._update_motion_feedback(robot_pose)
        if target.visible and robot is not None and target_agent is not None:
            target_delta = np.asarray(target_agent.base_pos, dtype=np.float32) - np.asarray(self._episode_origin)
            self._target_history_episode.append((
                float(np.dot(target_delta, self._episode_forward_axis)),
                float(np.dot(target_delta, self._episode_left_axis)),
            ))
            self._visibility_portal_episode = None
        elif self._target_was_visible and self._target_history_episode:
            # Latch the last genuinely observed position at the visibility
            # boundary. Reaching this portal takes priority over maintaining
            # follow distance to an exact coordinate behind an unseen wall.
            self._visibility_portal_episode = self._target_history_episode[-1]
        self._target_was_visible = bool(target.visible)
        self.last_map_clearance = dict(self.obstacle_map.last_clearance)
        self.last_map_visualization = self.obstacle_map.visualize(target.relative_xy)

    def _update_motion_feedback(self, robot_pose) -> None:
        self.last_motion_block_added = False
        if (
            not self.motion_blocked_feedback
            or self._last_translation_command is None
            or self._last_translation_pose is None
        ):
            return
        command_forward, command_left = self._last_translation_command
        command_norm = math.hypot(command_forward, command_left)
        if command_norm < self.motion_min_command:
            return
        previous_forward, previous_left, previous_yaw = (
            self._last_translation_pose
        )
        delta = np.asarray((
            robot_pose[0] - previous_forward,
            robot_pose[1] - previous_left,
        ), dtype=np.float32)
        direction = np.asarray((
            math.cos(previous_yaw) * command_forward
            - math.sin(previous_yaw) * command_left,
            math.sin(previous_yaw) * command_forward
            + math.cos(previous_yaw) * command_left,
        ), dtype=np.float32)
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        displacement = float(np.linalg.norm(delta))
        progress = float(np.dot(delta, direction))
        self.last_motion_displacement_m = displacement
        self.last_motion_progress_m = progress
        blocked = (
            displacement < self.motion_min_displacement_m
            or progress < -self.motion_min_displacement_m
        )
        if not blocked:
            self._motion_stall_steps = 0
            self._motion_stall_direction = None
            return
        if (
            self._motion_stall_direction is None
            or float(np.dot(direction, self._motion_stall_direction)) < 0.85
        ):
            self._motion_stall_steps = 1
            self._motion_stall_direction = direction
        else:
            self._motion_stall_steps += 1
        if self._motion_stall_steps == 2:
            self.last_motion_block_added = self.obstacle_map.mark_motion_blocked(
                command_forward,
                command_left,
                robot_pose=self._last_translation_pose,
            )

    def record_action(self, action: ContinuousAction, mode: str = "") -> None:
        """Store a command so the next observation can detect rejected motion."""
        if (
            not self.motion_blocked_feedback
            or mode.startswith("map_predictive_human_")
            or abs(action.yaw) > self.motion_max_block_yaw
            or math.hypot(action.forward, action.lateral)
            < self.motion_min_command
        ):
            self._last_translation_command = None
            self._last_translation_pose = None
            self._motion_stall_steps = 0
            self._motion_stall_direction = None
            return
        self._last_translation_command = (
            float(action.forward), float(action.lateral)
        )
        self._last_translation_pose = tuple(self.obstacle_map.robot_pose)

    def __call__(self, sim, robot, target_agent, target: TargetObservation) -> ControlDecision:
        # Planning overlays are per control step. Refresh them after choosing
        # this step's waypoint so the video never shows a stale A* path.
        self.obstacle_map.clear_plan()
        if not target.visible and not self.use_invisible_pointgoal:
            self.last_map_visualization = self.obstacle_map.visualize(target.relative_xy)
            return super().__call__(sim, robot, target_agent, target)
        target_episode = self.obstacle_map._local_to_episode(
            *target.relative_xy
        )
        portal_active = False
        portal_distance = 0.0
        if not target.visible and self._visibility_portal_episode is not None:
            if (
                self._target_closing_ema
                >= self.incoming_closing_threshold_m
            ):
                # The person disappeared while passing toward the robot. The
                # last visible body position is behind the pass direction and
                # must not be treated as a doorway breadcrumb.
                self._visibility_portal_episode = None
            else:
                robot_forward, robot_left, _ = self.obstacle_map.robot_pose
                portal_distance = math.hypot(
                    self._visibility_portal_episode[0] - robot_forward,
                    self._visibility_portal_episode[1] - robot_left,
                )
                if portal_distance > 0.35:
                    portal_active = True
                else:
                    self._visibility_portal_episode = None
        self.last_visibility_portal_episode = self._visibility_portal_episode
        self.last_visibility_portal_distance_m = (
            float(portal_distance) if portal_active else None
        )
        closing = 0.0
        radial = np.asarray(target.relative_xy, dtype=np.float32)
        if self._previous_target_episode is not None:
            target_motion = np.asarray(target_episode) - np.asarray(
                self._previous_target_episode
            )
            self._target_velocity_episode_ema = (
                0.5 * self._target_velocity_episode_ema
                + 0.5 * target_motion.astype(np.float32)
            )
            robot_forward, robot_left, _ = self.obstacle_map.robot_pose
            radial = np.asarray(target_episode) - np.asarray(
                (robot_forward, robot_left)
            )
            radial_norm = float(np.linalg.norm(radial))
            if radial_norm > 1e-6:
                # Positive only when the person, rather than the robot's own
                # approach, moves toward the current robot position.
                closing = max(
                    0.0, -float(np.dot(target_motion, radial / radial_norm))
                )
        self._previous_target_episode = tuple(float(v) for v in target_episode)
        self._target_closing_ema = (
            0.65 * self._target_closing_ema + 0.35 * closing
        )
        target_velocity = self._target_velocity_episode_ema
        velocity_sq = float(np.dot(target_velocity, target_velocity))
        closest_time = 0.0
        if velocity_sq > 1e-6:
            closest_time = float(np.clip(
                -np.dot(radial, target_velocity) / velocity_sq,
                0.0,
                self.safety_prediction_steps,
            ))
        predicted_range = float(np.linalg.norm(
            radial + closest_time * target_velocity
        ))
        self.last_target_closing_m_per_step = float(self._target_closing_ema)
        self.last_predicted_closest_range_m = predicted_range
        if self.predictive_human_safety:
            safety_was_active = self._human_safety_active
            incoming_near_follow_band = (
                target.visible
                and
                self._target_closing_ema >= self.incoming_closing_threshold_m
                and target.range_m
                < self.max_distance_m + self.incoming_activation_margin_m
            )
            if (
                not portal_active
                and (
                    predicted_range < 0.65
                    or target.range_m < 0.75
                    or incoming_near_follow_band
                )
            ):
                self._human_safety_active = True
            elif (
                (
                    not target.visible
                    and target.range_m >= 1.0
                    and self._target_closing_ema
                    < 0.5 * self.incoming_closing_threshold_m
                )
                or (
                    target.range_m >= self.min_distance_m + self.safety_release_margin_m
                    and predicted_range >= 0.90
                    and self._target_closing_ema
                    < 0.5 * self.incoming_closing_threshold_m
                )
            ):
                self._human_safety_active = False
            if self._human_safety_active:
                self._human_safety_steps += 1
                if not target.visible:
                    self._occluded_pass_active = True
                    self._occluded_pass_steps += 1
                    if self._occluded_pass_steps > 3:
                        self._human_safety_active = False
                else:
                    self._occluded_pass_steps = 0
            if self._human_safety_active:
                if not safety_was_active:
                    clear_left = self.last_map_clearance.get("left", 0.0)
                    clear_right = self.last_map_clearance.get("right", 0.0)
                    if clear_left > clear_right + 0.15:
                        self._human_evasion_direction = 1.0
                    elif clear_right > clear_left + 0.15:
                        self._human_evasion_direction = -1.0
                # Retreat along the full 2-D direction away from the person.
                # C3 reversed only Vx but kept Vy pointing toward the person.
                retreat_error = max(
                    0.0,
                    self.min_distance_m + self.safety_release_margin_m
                    - predicted_range,
                )
                retreat_speed = float(np.clip(
                    0.35 + 1.5 * retreat_error, 0.35, self.max_forward
                ))
                away_forward = -math.cos(target.bearing_rad) * retreat_speed
                away_lateral = -math.sin(target.bearing_rad) * retreat_speed
                head_on = (
                    self._human_safety_steps >= 2
                    and self._target_closing_ema
                    >= self.incoming_closing_threshold_m
                    and abs(target.bearing_rad) < math.radians(40.0)
                )
                if head_on:
                    # Backing away cannot resolve a head-on encounter when the
                    # leader walks at comparable speed. Commit to one clear
                    # side instead of oscillating in front of the person.
                    if target.visible:
                        away_forward = min(away_forward, -0.30)
                        away_lateral = 0.55 * self._human_evasion_direction
                    else:
                        # During an occluded pass, create enough clearance to
                        # remain on the connected-room side before turning to
                        # re-acquire. A small visible-mode sidestep leaves Spot
                        # stranded at the doorway after the person passes.
                        away_forward = min(away_forward, -0.85)
                        away_lateral = 0.75 * self._human_evasion_direction
                yaw = float(np.clip(
                    self.heading_gain * target.bearing_rad,
                    -0.5 * self.max_yaw,
                    0.5 * self.max_yaw,
                ))
                self.last_map_visualization = self.obstacle_map.visualize(
                    target.relative_xy
                )
                return ControlDecision(
                    ContinuousAction(
                        float(np.clip(away_forward, -self.max_forward, self.max_forward)),
                        float(np.clip(away_lateral, -self.max_lateral, self.max_lateral)),
                        yaw,
                    ),
                    (
                        "map_predictive_human_yield"
                        if head_on else "map_predictive_human_retreat"
                    ),
                    float(target.bearing_rad),
                    None,
                )
            self._human_safety_steps = 0
            if self._occluded_pass_active:
                self._occluded_pass_active = False
                self._occluded_pass_steps = 0
                self._post_pass_reframe_steps = 10
        if target.visible:
            self._post_pass_reframe_steps = 0
        if self._post_pass_reframe_steps > 0:
            phase = 10 - self._post_pass_reframe_steps
            turn_direction = -self._human_evasion_direction
            forward_profile = (
                -0.25, -0.15, 0.0, 0.0, 0.10,
                0.15, 0.20, 0.25, 0.30, 0.35,
            )
            lateral_profile = (
                0.0, 0.08, 0.15, 0.20, 0.20,
                0.15, 0.10, 0.05, 0.0, 0.0,
            )
            yaw_profile = (
                1.0, 1.0, 1.0, 1.0, 1.0,
                0.90, 0.80, 0.70, 0.60, 0.50,
            )
            forward = forward_profile[phase]
            lateral = turn_direction * lateral_profile[phase]
            yaw = turn_direction * yaw_profile[phase]
            self._post_pass_reframe_steps -= 1
            return ControlDecision(
                ContinuousAction(forward, lateral, yaw),
                "map_post_pass_reframe",
                float(target.bearing_rad),
                None,
            )
        room_entry_active = not target.visible and not portal_active
        if (
            target.range_m < self.min_distance_m
            and not portal_active
            and not room_entry_active
        ):
            # The map goal is defined in front of the person; when already too
            # close, retreat using the original person-relative point goal.
            self.last_map_visualization = self.obstacle_map.visualize(target.relative_xy)
            return super().__call__(sim, robot, target_agent, target)
        if (
            target.visible
            and
            target.range_m <= self.max_distance_m + self.planning_hysteresis_m
            and not portal_active
        ):
            yaw = float(np.clip(
                self.heading_gain * target.bearing_rad,
                -self.max_yaw,
                self.max_yaw,
            ))
            self.last_map_visualization = self.obstacle_map.visualize(target.relative_xy)
            return ControlDecision(
                ContinuousAction(0.0, 0.0, yaw),
                "map_follow_band_hysteresis_hold",
                float(target.bearing_rad),
                None,
            )
        planning_history = list(self._target_history_episode)
        if (
            portal_active
            and (
                not planning_history
                or planning_history[-1] != self._visibility_portal_episode
            )
        ):
            planning_history.append(self._visibility_portal_episode)
        planning_distance = 0.75 if room_entry_active else self.max_distance_m
        waypoint_forward, waypoint_left, map_mode = self.obstacle_map.choose_waypoint(
            target.relative_xy,
            desired_distance_m=planning_distance,
            history_episode=planning_history,
            prefer_history_waypoint=portal_active,
        )
        self.last_map_mode = map_mode
        self.last_map_visualization = self.obstacle_map.visualize(target.relative_xy)
        if map_mode == "map_blocked":
            self._map_blocked_steps += 1
            front_clearance = self.last_map_clearance.get("front", 0.0)
            if front_clearance > 0.90:
                # A persistent map can become temporarily topologically
                # blocked by viewpoint noise. If the live depth corridor is
                # clearly open, cautiously re-acquire the moving person rather
                # than remaining parked behind an old obstacle trace.
                recovery = super().__call__(sim, robot, target_agent, target)
                return ControlDecision(
                    ContinuousAction(
                        float(np.clip(recovery.action.forward, 0.0, min(self.max_forward, 0.75))),
                        float(np.clip(recovery.action.lateral, -0.30, 0.30)),
                        float(np.clip(recovery.action.yaw, -0.45, 0.45)),
                    ),
                    "map_live_depth_recovery",
                    recovery.guidance_bearing_rad,
                    None,
                )
            clear_left = self.last_map_clearance.get("left", 0.0)
            clear_right = self.last_map_clearance.get("right", 0.0)
            direction = self._map_evasion_direction
            # Select once on entry and keep the same side until A* recovers.
            # Re-selecting from noisy live depth every frame causes a stable
            # left/right oscillation in front of chair and table legs.
            if self._map_blocked_steps == 1:
                if clear_left > clear_right + 0.20:
                    direction = 1.0
                elif clear_right > clear_left + 0.20:
                    direction = -1.0
            self._map_evasion_direction = direction
            cautious_forward = 0.12 if front_clearance > 0.60 else 0.0
            return ControlDecision(
                ContinuousAction(
                    cautious_forward, 0.45 * direction, 0.15 * direction
                ),
                "map_blocked_evade",
                float(target.bearing_rad),
                None,
            )
        self._map_blocked_steps = 0
        waypoint_bearing = math.atan2(waypoint_left, waypoint_forward)
        if abs(waypoint_left) > 0.05:
            self._map_evasion_direction = math.copysign(1.0, waypoint_left)
        # The carrot controls heading, not catch-up speed. Using its short
        # 0.55 m distance as the velocity error caps the robot below the
        # leader's walking speed and makes it fall progressively behind.
        distance_error = max(0.0, target.range_m - self.max_distance_m)
        if room_entry_active:
            distance_error = max(
                distance_error, target.range_m - planning_distance
            )
        if portal_active:
            # Portal traversal is a topological commitment, not a request to
            # close the current person distance. Keep translating toward the
            # doorway even while the person coordinate remains in-band.
            distance_error = max(distance_error, portal_distance)
        # A small bounded feed-forward prevents a moving leader from opening
        # the gap while the distance error is still below the velocity cap.
        forward = self.distance_gain * distance_error
        if distance_error > 0.10:
            forward += self.catchup_bias_mps
        heading_alignment = max(0.0, math.cos(waypoint_bearing))
        if abs(waypoint_bearing) > math.radians(55.0):
            forward = (
                min(0.18, self.max_forward)
                if self.last_map_clearance.get("front", 0.0) > 0.90
                else 0.0
            )
            lateral = 0.0
        else:
            forward *= heading_alignment
            lateral = self.lateral_gain * waypoint_left * heading_alignment
        yaw = self.heading_gain * waypoint_bearing
        action = ContinuousAction(
            float(np.clip(forward, -self.max_forward, self.max_forward)),
            float(np.clip(lateral, -self.max_lateral, self.max_lateral)),
            float(np.clip(yaw, -self.max_yaw, self.max_yaw)),
        )
        return ControlDecision(
            action,
            f"{map_mode}_waypoint",
            float(waypoint_bearing),
            None,
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


def configure(config, scene_dataset: str, keep_top_down_map: bool = False):
    from habitat.config import read_write

    with read_write(config):
        config.habitat.simulator.scene_dataset = scene_dataset
        # These measures are unnecessary for a one-frame smoke test and some
        # require a complete episode rollout.
        measurements = config.habitat.task.measurements
        if not keep_top_down_map:
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
    person_boxes=None,
) -> Image.Image:
    image = Image.fromarray(np.asarray(rgb)[..., :3].astype(np.uint8))
    draw = ImageDraw.Draw(image)
    width, height = image.size
    draw.line((width // 2, 0, width // 2, height), fill=(30, 144, 255), width=2)
    for box, is_target in person_boxes or []:
        if not is_target:
            draw.rectangle(box, outline=(40, 120, 255), width=3)
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
