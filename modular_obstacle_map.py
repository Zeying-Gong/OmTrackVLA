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
from collections import deque
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
        robot_radius_m: float = 0.30,
        camera_height_m: float = 0.24,
        camera_pitch_deg: float = 5.0,
        camera_forward_offset_m: float = 0.0,
        camera_left_offset_m: float = 0.0,
        min_obstacle_height_m: float = 0.06,
        max_obstacle_height_m: float = 1.8,
        near_obstacle_range_m: float = 0.90,
        near_obstacle_min_height_m: float = 0.10,
        carrot_distance_m: float = 0.55,
        memory_frames: Optional[int] = None,
        min_static_hits: int = 1,
        free_space_all_depth: bool = False,
    ) -> None:
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.hfov_rad = math.radians(float(hfov_deg))
        self.max_depth_m = float(max_depth_m)
        self.grid_size_m = float(grid_size_m)
        self.pixels_per_meter = int(pixels_per_meter)
        self.robot_radius_m = float(robot_radius_m)
        self.camera_height_m = float(camera_height_m)
        self.camera_pitch_rad = math.radians(float(camera_pitch_deg))
        self.camera_forward_offset_m = float(camera_forward_offset_m)
        self.camera_left_offset_m = float(camera_left_offset_m)
        self.min_obstacle_height_m = float(min_obstacle_height_m)
        self.max_obstacle_height_m = float(max_obstacle_height_m)
        self.near_obstacle_range_m = float(near_obstacle_range_m)
        self.near_obstacle_min_height_m = float(near_obstacle_min_height_m)
        self.carrot_distance_m = float(carrot_distance_m)
        if memory_frames is not None and int(memory_frames) <= 0:
            raise ValueError("memory_frames must be positive or None")
        self.memory_frames = None if memory_frames is None else int(memory_frames)
        if int(min_static_hits) <= 0:
            raise ValueError("min_static_hits must be positive")
        self.min_static_hits = int(min_static_hits)
        self.free_space_all_depth = bool(free_space_all_depth)
        self.grid_size_px = int(round(grid_size_m * pixels_per_meter))
        self.center_px = self.grid_size_px // 2
        self._inflation_radius_px = max(
            0, int(math.ceil(self.robot_radius_m * self.pixels_per_meter))
        )
        self.reset()

    def reset(self) -> None:
        shape = (self.grid_size_px, self.grid_size_px)
        self.static_hits = np.zeros(shape, dtype=np.uint16)
        self.free_hits = np.zeros(shape, dtype=np.uint16)
        self._static_memory = deque(maxlen=self.memory_frames)
        self.static_map = np.zeros(shape, dtype=np.uint8)
        self.dynamic_map = np.zeros(shape, dtype=np.uint8)
        self.inflated_map = np.zeros(shape, dtype=np.uint8)
        self.explored_map = np.zeros(shape, dtype=np.uint8)
        self.trajectory_map = np.zeros(shape, dtype=np.uint8)
        self.last_path_px = []
        self.last_start_px = None
        self.last_goal_px = None
        self.last_carrot_px = None
        self.last_history_px = []
        self.last_history_waypoint_px = None
        self.robot_pose = (0.0, 0.0, 0.0)
        self.last_clearance = {
            "front": self.max_depth_m,
            "left": self.max_depth_m,
            "right": self.max_depth_m,
        }
        self.last_target_dynamic_points = 0
        self.last_ground_filtered_points = 0
        self.last_ceiling_filtered_points = 0
        self.last_free_cells = 0

    def _grid(self, forward: np.ndarray, left: np.ndarray):
        # Top-down image x increases to the right; keep the agent's left side
        # on the visual left so the map is not a horizontal mirror of RGB.
        gx = np.rint(self.center_px - left * self.pixels_per_meter).astype(np.int32)
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

        depth_valid = np.isfinite(depth) & (depth > 0.10) & (depth < self.max_depth_m)
        valid = depth_valid.copy()
        vfov_rad = 2.0 * math.atan(
            (height / max(width, 1)) * math.tan(self.hfov_rad * 0.5)
        )
        fy = 0.5 * height / math.tan(vfov_rad * 0.5)
        camera_up = (0.5 * (height - 1) - ys.astype(np.float32)) * depth / fy
        cos_pitch = math.cos(self.camera_pitch_rad)
        sin_pitch = math.sin(self.camera_pitch_rad)
        # Positive pitch means the optical axis points downward.
        local_forward = cos_pitch * depth + sin_pitch * camera_up
        obstacle_height = (
            self.camera_height_m - sin_pitch * depth + cos_pitch * camera_up
        )
        self.last_ground_filtered_points = int(np.sum(
            depth_valid & (obstacle_height < self.min_obstacle_height_m)
        ))
        self.last_ceiling_filtered_points = int(np.sum(
            depth_valid & (obstacle_height > self.max_obstacle_height_m)
        ))
        valid &= (
            (obstacle_height >= self.min_obstacle_height_m)
            & (obstacle_height <= self.max_obstacle_height_m)
        )
        # Very close furniture can be clipped below the nominal obstacle
        # band by the low Spot camera. Admit only near returns that are still
        # clearly above the ground plane; ground rays remain excluded.
        near_valid = depth_valid & (depth <= self.near_obstacle_range_m) & (
            obstacle_height >= self.near_obstacle_min_height_m
        ) & (obstacle_height <= self.max_obstacle_height_m)
        valid |= near_valid

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
        local_left = -(xs.astype(np.float32) - 0.5 * (width - 1)) * depth / fx
        # Remove only the robot footprint, not all near returns: chair legs and
        # low furniture can legitimately be closer than 0.6 m.
        outside_footprint = (
            np.hypot(local_forward, local_left) > self.robot_radius_m + 0.04
        )
        valid &= outside_footprint

        episode_forward, episode_left = self._local_to_episode(
            local_forward + self.camera_forward_offset_m,
            local_left + self.camera_left_offset_m,
            self.robot_pose,
        )
        gx, gy = self._grid(episode_forward, episode_left)
        inside = valid & (gx >= 0) & (gx < self.grid_size_px) & (gy >= 0) & (gy < self.grid_size_px)
        if self.free_space_all_depth:
            ray_inside = (
                depth_valid
                & outside_footprint
                & (gx >= 0) & (gx < self.grid_size_px)
                & (gy >= 0) & (gy < self.grid_size_px)
            )
        else:
            # Conservative 2-D mode: do not let rays passing above low
            # furniture clear its top-down footprint.
            ray_inside = inside
        static_pixels = inside & ~dynamic_pixels

        frame_static = np.zeros_like(self.static_map, dtype=np.uint8)
        frame_static[gy[static_pixels], gx[static_pixels]] = 1
        # Every depth return also observes free space along the camera ray.
        # Without this clearing step, accumulated furniture edges can form an
        # artificial wall that leaves A* no corridor to traverse.
        frame_free = np.zeros_like(self.static_map, dtype=np.uint8)
        robot_gx, robot_gy = self._grid(
            np.array([self.robot_pose[0]]), np.array([self.robot_pose[1]])
        )
        start_px = (int(robot_gx[0]), int(robot_gy[0]))
        # Sampling a regular image grid preserves field-of-view coverage
        # without drawing every depth pixel.
        ray_samples = ray_inside & ((ys % 8) == 0) & ((xs % 8) == 0)
        sampled = np.flatnonzero(ray_samples.ravel())
        inside_flat_gx = gx.ravel()[sampled]
        inside_flat_gy = gy.ravel()[sampled]
        for end_x, end_y in zip(inside_flat_gx.tolist(), inside_flat_gy.tolist()):
            cv2.line(frame_free, start_px, (int(end_x), int(end_y)), 1, 1)
        # Only height-filtered obstacle endpoints are occupied. Ground and
        # ceiling endpoints remain free-space evidence.
        frame_free[gy[inside], gx[inside]] = 0
        self.last_free_cells = int(frame_free.sum())
        if self.memory_frames is None:
            # Full-history mode: retain static evidence for the episode.
            # Count observations per frame, not per projected depth pixel.
            # Many pixels land in the same grid cell; np.add.at on raw points
            # makes a one-frame artifact immediately exceed a multi-frame hit
            # threshold.
            self.static_hits += frame_static.astype(self.static_hits.dtype)
            np.minimum(self.static_hits, np.iinfo(np.uint16).max, out=self.static_hits)
            self.free_hits += frame_free.astype(self.free_hits.dtype)
            np.minimum(self.free_hits, np.iinfo(np.uint16).max, out=self.free_hits)
            raw_static = (
                (self.static_hits >= self.min_static_hits)
                & (self.static_hits * 3 >= self.free_hits)
            ).astype(np.uint8)
        else:
            # Finite-memory mode: old observations expire instead of permanently
            # fossilizing one-frame depth artifacts.
            self._static_memory.append((frame_static, frame_free))
            occupied = np.sum(
                np.stack([item[0] for item in self._static_memory]), axis=0
            )
            observed_free = np.sum(
                np.stack([item[1] for item in self._static_memory]), axis=0
            )
            raw_static = (
                (occupied >= self.min_static_hits)
                & (occupied * 3 >= observed_free)
            ).astype(np.uint8)
        self.static_map = cv2.morphologyEx(
            raw_static, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
        )

        self.dynamic_map.fill(0)
        dynamic_inside = inside & dynamic_pixels
        self.dynamic_map[gy[dynamic_inside], gx[dynamic_inside]] = 1
        self.last_target_dynamic_points = int(np.sum(dynamic_inside))

        # Mark the current view as explored and retain the robot trajectory.
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
        history_episode=None,
    ) -> Tuple[float, float, str]:
        target_forward, target_left = (float(v) for v in target_relative_xy)
        target_range = math.hypot(target_forward, target_left)
        if target_range <= desired_distance_m:
            self.clear_plan()
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
        direct_path = self._astar(start, goal)
        path = direct_path
        # If the current target is around a wall, first route through the most
        # recent reachable point in the person's demonstrated trajectory.
        # This preserves the human's already validated passage around corners.
        history_path = []
        self.last_history_waypoint_px = None
        self.last_history_px = []
        if history_episode:
            for hf, hl in history_episode:
                hx, hy = self._grid(np.asarray([hf]), np.asarray([hl]))
                if 0 <= hx[0] < self.grid_size_px and 0 <= hy[0] < self.grid_size_px:
                    self.last_history_px.append((int(hx[0]), int(hy[0])))
        direct_blocked = len(direct_path) > int(
            math.hypot(goal[0] - start[0], goal[1] - start[1]) * 1.35
        )
        if history_episode and direct_blocked:
            # Newer points have priority. Never choose a point substantially
            # behind the robot, which would make the controller backtrack.
            history_candidates = list(history_episode[:-2])[::-1]
            for hf, hl in history_candidates:
                delta_forward = hf - robot_forward
                if delta_forward < -0.20:
                    continue
                hx, hy = self._grid(np.asarray([hf]), np.asarray([hl]))
                candidate = self._astar(start, (int(hx[0]), int(hy[0])))
                if candidate and len(candidate) > 1:
                    history_path = candidate
                    self.last_history_waypoint_px = (int(hx[0]), int(hy[0]))
                    break
            if history_path:
                path = history_path
        self.last_path_px = path
        self.last_start_px = path[0] if path else start
        self.last_goal_px = path[-1] if path else goal
        self.last_carrot_px = None
        if not path:
            return 0.0, 0.0, "map_blocked"
        if len(path) == 1 and target_range > desired_distance_m + 0.10:
            return 0.0, 0.0, "map_blocked"

        carrot_index = min(
            len(path) - 1,
            max(1, int(round(self.carrot_distance_m * self.pixels_per_meter))),
        )
        carrot_x, carrot_y = path[carrot_index]
        self.last_carrot_px = (carrot_x, carrot_y)
        carrot_episode_forward = (self.center_px - carrot_y) / self.pixels_per_meter
        carrot_episode_left = (self.center_px - carrot_x) / self.pixels_per_meter
        local_forward, local_left = self._episode_to_local(
            carrot_episode_forward, carrot_episode_left
        )
        direct = all(not self.inflated_map[y, x] for x, y in path[:carrot_index + 1])
        mode = "map_direct" if direct and len(path) <= carrot_index + 2 else "map_astar"
        return float(local_forward), float(local_left), mode

    def clear_plan(self) -> None:
        """Clear per-control-step planning overlays without resetting the map."""
        self.last_path_px = []
        self.last_start_px = None
        self.last_goal_px = None
        self.last_carrot_px = None

    def visualize(self, target_relative_xy: Optional[Tuple[float, float]] = None) -> np.ndarray:
        canvas = np.full((self.grid_size_px, self.grid_size_px, 3), 235, dtype=np.uint8)
        canvas[self.explored_map == 0] = (205, 205, 205)
        # Raw static hits and the robot-clearance inflation are deliberately
        # rendered separately; otherwise the conservative inflated footprint
        # looks like the whole scene is occupied.
        canvas[self.static_map > 0] = (105, 105, 105)
        # The display keeps raw occupied cells gray; A* still uses the
        # separately inflated map internally for robot clearance.
        canvas[self.dynamic_map > 0] = (220, 70, 180)
        canvas[self.trajectory_map > 0] = (40, 110, 230)
        if len(self.last_history_px) >= 2:
            cv2.polylines(
                canvas,
                [np.asarray(self.last_history_px, dtype=np.int32).reshape((-1, 1, 2))],
                False,
                (255, 165, 0),
                1,
                cv2.LINE_AA,
            )
            for hx, hy in self.last_history_px[::max(1, len(self.last_history_px) // 12)]:
                cv2.circle(canvas, (hx, hy), 1, (255, 165, 0), -1)
        if self.last_history_waypoint_px is not None:
            cv2.drawMarker(
                canvas, self.last_history_waypoint_px, (255, 0, 255),
                cv2.MARKER_DIAMOND, 5, 1, cv2.LINE_AA,
            )
        valid_path = [
            (int(x), int(y)) for x, y in self.last_path_px
            if 0 <= x < self.grid_size_px and 0 <= y < self.grid_size_px
        ]
        if len(valid_path) >= 2:
            cv2.polylines(
                canvas,
                [np.asarray(valid_path, dtype=np.int32).reshape((-1, 1, 2))],
                False,
                (0, 210, 255),
                1,
                cv2.LINE_AA,
            )
            # Sparse 3-pixel-radius path nodes make the discrete A* route
            # visible without turning every grid cell into a large blob.
            node_stride = max(1, len(valid_path) // 24)
            for px, py in valid_path[::node_stride]:
                cv2.circle(canvas, (px, py), 3, (0, 210, 255), -1)
        elif valid_path:
            cv2.circle(canvas, valid_path[0], 2, (0, 210, 255), -1)
        if self.last_goal_px is not None:
            gx, gy = self.last_goal_px
            if 0 <= gx < self.grid_size_px and 0 <= gy < self.grid_size_px:
                cv2.circle(canvas, (gx, gy), 1, (235, 55, 45), -1)
        if self.last_carrot_px is not None:
            cx, cy = self.last_carrot_px
            if 0 <= cx < self.grid_size_px and 0 <= cy < self.grid_size_px:
                # A small ring and crosshair remain visible after the map is
                # resized into the video, without masking nearby cells.
                cv2.circle(canvas, (cx, cy), 2, (255, 225, 0), 1)
                cv2.line(canvas, (cx - 1, cy), (cx + 1, cy), (255, 225, 0), 1)
                cv2.line(canvas, (cx, cy - 1), (cx, cy + 1), (255, 225, 0), 1)
        robot_gx, robot_gy = self._grid(
            np.array([self.robot_pose[0]]), np.array([self.robot_pose[1]])
        )
        if 0 <= robot_gx[0] < self.grid_size_px and 0 <= robot_gy[0] < self.grid_size_px:
            cv2.circle(canvas, (int(robot_gx[0]), int(robot_gy[0])), 2, (255, 0, 0), -1)
        if target_relative_xy is not None:
            tf, tl = self._local_to_episode(*target_relative_xy)
            gx, gy = self._grid(np.array([tf]), np.array([tl]))
            if 0 <= gx[0] < self.grid_size_px and 0 <= gy[0] < self.grid_size_px:
                cv2.circle(canvas, (int(gx[0]), int(gy[0])), 2, (0, 180, 0), -1)
        memory_label = "all" if self.memory_frames is None else str(self.memory_frames)
        cv2.putText(canvas, f"memory={memory_label} gray=static black=inflated pink=dynamic", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.31, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(canvas, "cyan=A* yellow=carrot orange=human-traj magenta=hist-goal red=goal green=person blue=robot", (4, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.29, (20, 20, 20), 1, cv2.LINE_AA)
        return cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
