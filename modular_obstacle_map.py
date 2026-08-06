"""Lightweight depth obstacle map for modular person-following control.

This follows Ascent's depth->point-cloud->inflated-map idea without importing
its frontier/VLM stack. The map is local and refreshed every observation; the
API is intentionally ready for future multi-person dynamic masks.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np


def _metric_depth(raw_depth: np.ndarray, max_depth_m: float) -> np.ndarray:
    depth = np.asarray(raw_depth, dtype=np.float32).squeeze()
    finite = depth[np.isfinite(depth)]
    if finite.size and float(np.nanmax(finite)) <= 1.0 + 1e-3:
        depth = depth * float(max_depth_m)
    return depth


class LocalObstacleMap:
    def __init__(
        self,
        image_width: int = 384,
        image_height: int = 384,
        hfov_deg: float = 90.0,
        max_depth_m: float = 10.0,
        grid_size_m: float = 12.0,
        pixels_per_meter: int = 10,
        robot_radius_m: float = 0.30,
        camera_height_m: float = 0.85,
        min_obstacle_height_m: float = 0.05,
        max_obstacle_height_m: float = 1.8,
    ) -> None:
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.hfov_rad = math.radians(float(hfov_deg))
        self.max_depth_m = float(max_depth_m)
        self.grid_size_m = float(grid_size_m)
        self.pixels_per_meter = int(pixels_per_meter)
        self.robot_radius_m = float(robot_radius_m)
        self.camera_height_m = float(camera_height_m)
        self.min_obstacle_height_m = float(min_obstacle_height_m)
        self.max_obstacle_height_m = float(max_obstacle_height_m)
        self.grid_size_px = int(round(grid_size_m * pixels_per_meter))
        self.center_px = self.grid_size_px // 2
        self.static_map = np.zeros(
            (self.grid_size_px, self.grid_size_px), dtype=np.uint8
        )
        self.dynamic_map = np.zeros_like(self.static_map)
        self.inflated_map = np.zeros_like(self.static_map)
        self.last_clearance = {"front": max_depth_m, "left": max_depth_m, "right": max_depth_m}
        self.last_target_dynamic_points = 0

    def _grid(self, forward: np.ndarray, left: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gx = np.rint(self.center_px + left * self.pixels_per_meter).astype(np.int32)
        gy = np.rint(self.center_px - forward * self.pixels_per_meter).astype(np.int32)
        return gx, gy

    def update(
        self,
        raw_depth: np.ndarray,
        target_bbox: Optional[Tuple[int, int, int, int]] = None,
        dynamic_mask: Optional[np.ndarray] = None,
    ) -> None:
        depth = _metric_depth(raw_depth, max_depth_m=self.max_depth_m)
        if depth.ndim != 2:
            depth = np.squeeze(depth)
        height, width = depth.shape[:2]
        ys, xs = np.indices((height, width))
        # Ignore returns immediately in front of/inside the robot body.  Jaw
        # depth often contains self-geometry at a few centimetres.
        valid = np.isfinite(depth) & (depth > 0.60) & (depth < self.max_depth_m)
        # Keep the lower/middle camera image where physical obstacles intersect
        # the robot's walking plane; suppress ceiling and sky pixels.
        # Approximate vertical projection and discard the floor/ceiling.  This
        # is the important distinction from treating every valid depth pixel
        # as a 2-D obstacle: pixels below the camera ray hit the floor.
        vfov_rad = math.radians(60.0)
        fy = 0.5 * height / math.tan(vfov_rad * 0.5)
        vertical = (0.5 * (height - 1) - ys.astype(np.float32)) * depth / fy
        obstacle_height = self.camera_height_m + vertical
        valid &= (
            (obstacle_height >= self.min_obstacle_height_m)
            & (obstacle_height <= self.max_obstacle_height_m)
        )
        if dynamic_mask is not None:
            dynamic_mask = np.asarray(dynamic_mask, dtype=bool)
            if dynamic_mask.shape == valid.shape:
                dynamic_pixels = valid & dynamic_mask
            else:
                dynamic_pixels = np.zeros_like(valid)
        elif target_bbox is not None:
            x1, y1, x2, y2 = (int(v) for v in target_bbox)
            dynamic_pixels = np.zeros_like(valid)
            dynamic_pixels[
                max(0, y1):min(height, y2 + 1),
                max(0, x1):min(width, x2 + 1),
            ] = True
            dynamic_pixels &= valid
        else:
            dynamic_pixels = np.zeros_like(valid)

        fx = 0.5 * width / math.tan(self.hfov_rad * 0.5)
        forward = depth
        left = -(xs.astype(np.float32) - (width - 1) * 0.5) * depth / fx
        gx, gy = self._grid(forward, left)
        inside = (
            valid
            & (gx >= 0) & (gx < self.grid_size_px)
            & (gy >= 0) & (gy < self.grid_size_px)
        )
        self.static_map.fill(0)
        self.dynamic_map.fill(0)
        static_pixels = inside & ~dynamic_pixels
        self.static_map[gy[static_pixels], gx[static_pixels]] = 1
        self.dynamic_map[gy[inside & dynamic_pixels], gx[inside & dynamic_pixels]] = 1
        self.last_target_dynamic_points = int(np.sum(inside & dynamic_pixels))

        radius_px = max(1, int(round(self.robot_radius_m * self.pixels_per_meter)))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1)
        )
        combined = np.maximum(self.static_map, self.dynamic_map)
        self.inflated_map = cv2.dilate(combined, kernel, iterations=1)
        yy, xx = np.ogrid[:self.grid_size_px, :self.grid_size_px]
        self.inflated_map[
            (xx - self.center_px) ** 2 + (yy - self.center_px) ** 2
            <= int(round((self.robot_radius_m * self.pixels_per_meter) ** 2))
        ] = 0
        self._update_clearance(forward, left, valid & ~dynamic_pixels)

    def _update_clearance(self, forward, left, valid):
        sectors = {
            "front": (np.abs(left) < 0.45, forward > 0.0),
            "left": (left > 0.20, forward > 0.0),
            "right": (left < -0.20, forward > 0.0),
        }
        for name, (side, ahead) in sectors.items():
            values = forward[valid & side & ahead]
            self.last_clearance[name] = (
                float(np.percentile(values, 10)) if values.size else self.max_depth_m
            )

    def _segment_is_free(self, start, end) -> bool:
        length = int(max(1, np.linalg.norm(np.asarray(end) - np.asarray(start))))
        xs = np.linspace(float(start[0]), float(end[0]), length).round().astype(int)
        ys = np.linspace(float(start[1]), float(end[1]), length).round().astype(int)
        inside = (
            (xs >= 0) & (xs < self.grid_size_px)
            & (ys >= 0) & (ys < self.grid_size_px)
        )
        return bool(np.all(inside) and not np.any(self.inflated_map[ys[inside], xs[inside]]))

    def choose_waypoint(
        self,
        target_relative_xy: Tuple[float, float],
        desired_distance_m: float = 1.35,
    ) -> Tuple[float, float, str]:
        """Choose a collision-aware local follow point around the target."""
        target_forward, target_left = (float(v) for v in target_relative_xy)
        target_range = math.hypot(target_forward, target_left)
        if target_range < 1e-6:
            return 0.0, 0.0, "map_hold"
        target_angle = math.atan2(target_left, target_forward)
        candidates = []
        # Direct target point plus lateral detours. The detours are a local
        # analogue of Ascent's map-selected frontier/carrot waypoint.
        for offset_angle in (0.0, -0.35, 0.35, -0.70, 0.70, -1.05, 1.05):
            angle = target_angle + offset_angle
            # The waypoint is where the robot should stand, not the target
            # centre.  Keep the requested follow distance from the person.
            radius = max(0.0, target_range - desired_distance_m)
            goal = (radius * math.cos(angle), radius * math.sin(angle))
            goal_px = self._grid(np.array([goal[0]]), np.array([goal[1]]))
            if not self._segment_is_free(
                (self.center_px, self.center_px), (int(goal_px[0][0]), int(goal_px[1][0]))
            ):
                continue
            score = (
                abs(offset_angle) * 0.8
                + abs(radius - desired_distance_m) * 0.35
                + abs(angle) * 0.15
            )
            candidates.append((score, goal[0], goal[1], offset_angle))
        if not candidates:
            return 0.0, 0.0, "map_blocked"
        _, forward, left, offset = min(candidates, key=lambda item: item[0])
        mode = "map_direct" if abs(offset) < 1e-6 else "map_detour"
        return float(forward), float(left), mode

    def visualize(self, target_relative_xy: Optional[Tuple[float, float]] = None) -> np.ndarray:
        canvas = np.full((self.grid_size_px, self.grid_size_px, 3), 245, dtype=np.uint8)
        canvas[self.inflated_map > 0] = (35, 35, 35)
        canvas[self.dynamic_map > 0] = (220, 70, 180)
        canvas[self.center_px - 2:self.center_px + 3, self.center_px - 2:self.center_px + 3] = (0, 0, 255)
        if target_relative_xy is not None:
            gx, gy = self._grid(
                np.array([target_relative_xy[0]]), np.array([target_relative_xy[1]])
            )
            if 0 <= gx[0] < self.grid_size_px and 0 <= gy[0] < self.grid_size_px:
                cv2.circle(canvas, (int(gx[0]), int(gy[0])), 5, (0, 180, 0), -1)
        return cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
