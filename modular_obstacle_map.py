"""Persistent depth obstacle map for modular person-following control.

The map follows Ascent's useful mapping core: depth points are height-filtered,
transformed into a fixed episodic frame, accumulated, inflated by the robot
radius, and searched for a collision-free carrot waypoint.  Person pixels are
excluded from the persistent static layer and rebuilt in an ephemeral dynamic
layer every frame.
"""

from __future__ import annotations

import heapq
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
    """Episodic top-down map with a local planning interface.

    ``robot_pose`` is ``(forward, left, yaw)`` in a fixed coordinate frame
    established at episode reset.  The public waypoint remains robot-local so
    the controller does not depend on Habitat-specific map conventions.
    """

    def __init__(
        self,
        image_width: int = 384,
        image_height: int = 384,
        hfov_deg: float = 90.0,
        max_depth_m: float = 10.0,
        grid_size_m: float = 20.0,
        pixels_per_meter: int = 10,
        robot_radius_m: float = 0.38,
        camera_height_m: float = 0.85,
        min_obstacle_height_m: float = 0.06,
        max_obstacle_height_m: float = 1.8,
        carrot_distance_m: float = 0.55,
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
        self.carrot_distance_m = float(carrot_distance_m)
        self.grid_size_px = int(round(grid_size_m * pixels_per_meter))
        self.center_px = self.grid_size_px // 2
        self._inflation_radius_px = max(
            1, int(math.ceil(self.robot_radius_m * self.pixels_per_meter))
        )
        self.reset()

    def reset(self) -> None:
        shape = (self.grid_size_px, self.grid_size_px)
        self.static_hits = np.zeros(shape, dtype=np.uint16)
        self.static_map = np.zeros(shape, dtype=np.uint8)
        self.dynamic_map = np.zeros(shape, dtype=np.uint8)
        self.inflated_map = np.zeros(shape, dtype=np.uint8)
        self.explored_map = np.zeros(shape, dtype=np.uint8)
        self.trajectory_map = np.zeros(shape, dtype=np.uint8)
        self.last_path_px = []
        self.robot_pose = (0.0, 0.0, 0.0)
        self.last_clearance = {
            "front": self.max_depth_m,
            "left": self.max_depth_m,
            "right": self.max_depth_m,
        }
        self.last_target_dynamic_points = 0

    def _grid(self, forward: np.ndarray, left: np.ndarray):
        gx = np.rint(self.center_px + left * self.pixels_per_meter).astype(np.int32)
        gy = np.rint(self.center_px - forward * self.pixels_per_meter).astype(np.int32)
        return gx, gy

    def _local_to_episode(self, forward, left, robot_pose=None):
        rf, rl, yaw = self.robot_pose if robot_pose is None else robot_pose
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        episode_forward = rf + cos_yaw * forward - sin_yaw * left
        episode_left = rl + sin_yaw * forward + cos_yaw * left
        return episode_forward, episode_left

    def _episode_to_local(self, forward, left):
        rf, rl, yaw = self.robot_pose
        df, dl = forward - rf, left - rl
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        return cos_yaw * df + sin_yaw * dl, -sin_yaw * df + cos_yaw * dl

    def update(
        self,
        raw_depth: np.ndarray,
        target_bbox: Optional[Tuple[int, int, int, int]] = None,
        dynamic_mask: Optional[np.ndarray] = None,
        robot_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.robot_pose = tuple(float(v) for v in robot_pose)
        depth = _metric_depth(raw_depth, max_depth_m=self.max_depth_m)
        if depth.ndim != 2:
            depth = np.squeeze(depth)
        height, width = depth.shape[:2]
        ys, xs = np.indices((height, width))

        valid = np.isfinite(depth) & (depth > 0.10) & (depth < self.max_depth_m)
        vfov_rad = 2.0 * math.atan(
            (height / max(width, 1)) * math.tan(self.hfov_rad * 0.5)
        )
        fy = 0.5 * height / math.tan(vfov_rad * 0.5)
        vertical = (0.5 * (height - 1) - ys.astype(np.float32)) * depth / fy
        obstacle_height = self.camera_height_m + vertical
        valid &= (
            (obstacle_height >= self.min_obstacle_height_m)
            & (obstacle_height <= self.max_obstacle_height_m)
        )

        if dynamic_mask is not None:
            dynamic_mask = np.asarray(dynamic_mask, dtype=bool)
            dynamic_pixels = valid & dynamic_mask if dynamic_mask.shape == valid.shape else np.zeros_like(valid)
        elif target_bbox is not None:
            x1, y1, x2, y2 = (int(v) for v in target_bbox)
            dynamic_pixels = np.zeros_like(valid)
            dynamic_pixels[max(0, y1):min(height, y2 + 1), max(0, x1):min(width, x2 + 1)] = True
            dynamic_pixels &= valid
        else:
            dynamic_pixels = np.zeros_like(valid)

        fx = 0.5 * width / math.tan(self.hfov_rad * 0.5)
        local_forward = depth
        local_left = -(xs.astype(np.float32) - 0.5 * (width - 1)) * depth / fx
        # Remove only the robot footprint, not all near returns: chair legs and
        # low furniture can legitimately be closer than 0.6 m.
        valid &= np.hypot(local_forward, local_left) > self.robot_radius_m + 0.04

        episode_forward, episode_left = self._local_to_episode(
            local_forward, local_left, self.robot_pose
        )
        gx, gy = self._grid(episode_forward, episode_left)
        inside = valid & (gx >= 0) & (gx < self.grid_size_px) & (gy >= 0) & (gy < self.grid_size_px)
        static_pixels = inside & ~dynamic_pixels

        # Persistent static evidence. A small close operation connects sparse
        # chair/table legs before robot-radius inflation.
        np.add.at(
            self.static_hits,
            (gy[static_pixels], gx[static_pixels]),
            1,
        )
        np.minimum(self.static_hits, np.iinfo(np.uint16).max, out=self.static_hits)
        raw_static = (self.static_hits > 0).astype(np.uint8)
        self.static_map = cv2.morphologyEx(
            raw_static, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
        )

        self.dynamic_map.fill(0)
        dynamic_inside = inside & dynamic_pixels
        self.dynamic_map[gy[dynamic_inside], gx[dynamic_inside]] = 1
        self.last_target_dynamic_points = int(np.sum(dynamic_inside))

        # Mark the current view as explored and retain the robot trajectory.
        robot_gx, robot_gy = self._grid(
            np.array([self.robot_pose[0]]), np.array([self.robot_pose[1]])
        )
        if 0 <= robot_gx[0] < self.grid_size_px and 0 <= robot_gy[0] < self.grid_size_px:
            cv2.circle(self.trajectory_map, (int(robot_gx[0]), int(robot_gy[0])), 1, 1, -1)
            cv2.circle(self.explored_map, (int(robot_gx[0]), int(robot_gy[0])), int(self.max_depth_m * self.pixels_per_meter), 1, -1)

        combined = np.maximum(self.static_map, self.dynamic_map)
        kernel_size = 2 * self._inflation_radius_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        self.inflated_map = cv2.dilate(combined, kernel, iterations=1)
        # The robot's current footprint must remain a legal A* start cell.
        if 0 <= robot_gx[0] < self.grid_size_px and 0 <= robot_gy[0] < self.grid_size_px:
            cv2.circle(
                self.inflated_map,
                (int(robot_gx[0]), int(robot_gy[0])),
                self._inflation_radius_px,
                0,
                -1,
            )
        self._update_clearance(local_forward, local_left, valid & ~dynamic_pixels)

    def _update_clearance(self, forward, left, valid):
        sectors = {
            "front": np.abs(left) < 0.45,
            "left": left > 0.20,
            "right": left < -0.20,
        }
        for name, sector in sectors.items():
            values = forward[valid & sector & (forward > 0.0)]
            self.last_clearance[name] = float(np.percentile(values, 10)) if values.size else self.max_depth_m

    def _nearest_free(self, point, max_radius=12):
        x0, y0 = int(point[0]), int(point[1])
        for radius in range(max_radius + 1):
            candidates = []
            for y in range(y0 - radius, y0 + radius + 1):
                for x in range(x0 - radius, x0 + radius + 1):
                    if max(abs(x - x0), abs(y - y0)) != radius:
                        continue
                    if 0 <= x < self.grid_size_px and 0 <= y < self.grid_size_px and not self.inflated_map[y, x]:
                        candidates.append((math.hypot(x - x0, y - y0), x, y))
            if candidates:
                _, x, y = min(candidates)
                return x, y
        return None

    def _astar(self, start, goal):
        start = self._nearest_free(start, self._inflation_radius_px + 2)
        goal = self._nearest_free(goal, int(0.8 * self.pixels_per_meter))
        if start is None or goal is None:
            return []
        moves = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414))
        queue = [(0.0, 0.0, start)]
        parent = {start: None}
        best = {start: 0.0}
        while queue:
            _, cost, current = heapq.heappop(queue)
            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = parent[current]
                return path[::-1]
            if cost > best.get(current, float("inf")):
                continue
            for dx, dy, step_cost in moves:
                nxt = (current[0] + dx, current[1] + dy)
                if not (0 <= nxt[0] < self.grid_size_px and 0 <= nxt[1] < self.grid_size_px):
                    continue
                if self.inflated_map[nxt[1], nxt[0]]:
                    continue
                new_cost = cost + step_cost
                if new_cost >= best.get(nxt, float("inf")):
                    continue
                best[nxt] = new_cost
                parent[nxt] = current
                heuristic = math.hypot(goal[0] - nxt[0], goal[1] - nxt[1])
                heapq.heappush(queue, (new_cost + heuristic, new_cost, nxt))
        return []

    def choose_waypoint(
        self,
        target_relative_xy: Tuple[float, float],
        desired_distance_m: float = 1.35,
    ) -> Tuple[float, float, str]:
        target_forward, target_left = (float(v) for v in target_relative_xy)
        target_range = math.hypot(target_forward, target_left)
        if target_range <= desired_distance_m:
            self.last_path_px = []
            return 0.0, 0.0, "map_hold"

        robot_forward, robot_left, _ = self.robot_pose
        target_episode = self._local_to_episode(target_forward, target_left)
        direction_f = (target_episode[0] - robot_forward) / target_range
        direction_l = (target_episode[1] - robot_left) / target_range
        follow_goal = (
            target_episode[0] - desired_distance_m * direction_f,
            target_episode[1] - desired_distance_m * direction_l,
        )
        start_px_arr = self._grid(np.array([robot_forward]), np.array([robot_left]))
        goal_px_arr = self._grid(np.array([follow_goal[0]]), np.array([follow_goal[1]]))
        start = (int(start_px_arr[0][0]), int(start_px_arr[1][0]))
        goal = (int(goal_px_arr[0][0]), int(goal_px_arr[1][0]))
        path = self._astar(start, goal)
        self.last_path_px = path
        if not path:
            return 0.0, 0.0, "map_blocked"

        carrot_index = min(
            len(path) - 1,
            max(1, int(round(self.carrot_distance_m * self.pixels_per_meter))),
        )
        carrot_x, carrot_y = path[carrot_index]
        carrot_episode_forward = (self.center_px - carrot_y) / self.pixels_per_meter
        carrot_episode_left = (carrot_x - self.center_px) / self.pixels_per_meter
        local_forward, local_left = self._episode_to_local(
            carrot_episode_forward, carrot_episode_left
        )
        direct = all(not self.inflated_map[y, x] for x, y in path[:carrot_index + 1])
        mode = "map_direct" if direct and len(path) <= carrot_index + 2 else "map_astar"
        return float(local_forward), float(local_left), mode

    def visualize(self, target_relative_xy: Optional[Tuple[float, float]] = None) -> np.ndarray:
        canvas = np.full((self.grid_size_px, self.grid_size_px, 3), 235, dtype=np.uint8)
        canvas[self.explored_map == 0] = (205, 205, 205)
        canvas[self.static_map > 0] = (70, 70, 70)
        canvas[self.inflated_map > 0] = (25, 25, 25)
        canvas[self.dynamic_map > 0] = (220, 70, 180)
        canvas[self.trajectory_map > 0] = (40, 110, 230)
        for x, y in self.last_path_px:
            if 0 <= x < self.grid_size_px and 0 <= y < self.grid_size_px:
                canvas[y, x] = (0, 210, 255)
        robot_gx, robot_gy = self._grid(
            np.array([self.robot_pose[0]]), np.array([self.robot_pose[1]])
        )
        if 0 <= robot_gx[0] < self.grid_size_px and 0 <= robot_gy[0] < self.grid_size_px:
            cv2.circle(canvas, (int(robot_gx[0]), int(robot_gy[0])), 4, (255, 0, 0), -1)
        if target_relative_xy is not None:
            tf, tl = self._local_to_episode(*target_relative_xy)
            gx, gy = self._grid(np.array([tf]), np.array([tl]))
            if 0 <= gx[0] < self.grid_size_px and 0 <= gy[0] < self.grid_size_px:
                cv2.circle(canvas, (int(gx[0]), int(gy[0])), 5, (0, 180, 0), -1)
        return cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
