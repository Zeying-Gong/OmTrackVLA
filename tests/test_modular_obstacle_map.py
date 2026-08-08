import unittest

import numpy as np

from modular_obstacle_map import LocalObstacleMap


class LocalObstacleMapTest(unittest.TestCase):
    def test_default_memory_retains_full_episode(self):
        obstacle_map = LocalObstacleMap()

        self.assertIsNone(obstacle_map.memory_frames)

    def test_target_mask_is_dynamic_not_static(self):
        obstacle_map = LocalObstacleMap(image_width=64, image_height=64)
        depth = np.full((64, 64), 2.0, dtype=np.float32)
        mask = np.zeros_like(depth, dtype=bool)
        mask[24:40, 26:38] = True

        obstacle_map.update(depth, dynamic_mask=mask)

        self.assertGreater(obstacle_map.dynamic_map.sum(), 0)
        self.assertGreater(obstacle_map.last_target_dynamic_points, 0)

    def test_visualization_has_expected_shape(self):
        obstacle_map = LocalObstacleMap(grid_size_m=6.0, pixels_per_meter=10)
        image = obstacle_map.visualize((2.0, 0.0))
        self.assertEqual(image.shape, (60, 60, 3))

    def test_zero_robot_radius_disables_inflation(self):
        obstacle_map = LocalObstacleMap(
            image_width=64, image_height=64, robot_radius_m=0.0
        )
        obstacle_map.update(np.full((64, 64), 2.0, dtype=np.float32))

        np.testing.assert_array_equal(
            obstacle_map.inflated_map,
            np.maximum(obstacle_map.static_map, obstacle_map.dynamic_map),
        )

    def test_camera_offset_does_not_shift_robot_relative_target(self):
        obstacle_map = LocalObstacleMap(camera_forward_offset_m=0.24)
        obstacle_map.robot_pose = (1.0, 2.0, 0.0)

        self.assertEqual(obstacle_map._local_to_episode(0.0, 0.0), (1.0, 2.0))

    def test_static_obstacles_persist_across_robot_motion(self):
        obstacle_map = LocalObstacleMap(image_width=64, image_height=64)
        depth = np.full((64, 64), 2.0, dtype=np.float32)
        obstacle_map.update(depth, robot_pose=(0.0, 0.0, 0.0))
        first = obstacle_map.static_map.copy()

        empty_depth = np.full((64, 64), obstacle_map.max_depth_m, dtype=np.float32)
        obstacle_map.update(empty_depth, robot_pose=(0.5, 0.0, 0.0))

        self.assertTrue(np.all(obstacle_map.static_map[first > 0] > 0))
        self.assertGreater(obstacle_map.trajectory_map.sum(), 1)

    def test_static_hit_threshold_counts_frames_not_depth_pixels(self):
        obstacle_map = LocalObstacleMap(
            image_width=64, image_height=64, min_static_hits=2
        )
        depth = np.full((64, 64), 2.0, dtype=np.float32)

        obstacle_map.update(depth)
        self.assertEqual(obstacle_map.static_map.sum(), 0)
        obstacle_map.update(depth)
        self.assertGreater(obstacle_map.static_map.sum(), 0)

    def test_dynamic_layer_is_rebuilt_each_frame(self):
        obstacle_map = LocalObstacleMap(image_width=64, image_height=64)
        depth = np.full((64, 64), 2.0, dtype=np.float32)
        mask = np.zeros_like(depth, dtype=bool)
        mask[24:40, 26:38] = True
        obstacle_map.update(depth, dynamic_mask=mask)
        self.assertGreater(obstacle_map.dynamic_map.sum(), 0)

        obstacle_map.update(depth, dynamic_mask=np.zeros_like(mask))
        self.assertEqual(obstacle_map.dynamic_map.sum(), 0)

    def test_finite_memory_expires_old_static_obstacles(self):
        obstacle_map = LocalObstacleMap(
            image_width=64, image_height=64, memory_frames=2
        )
        obstacle_depth = np.full((64, 64), 2.0, dtype=np.float32)
        empty_depth = np.full_like(obstacle_depth, obstacle_map.max_depth_m)
        obstacle_map.update(obstacle_depth)
        self.assertGreater(obstacle_map.static_map.sum(), 0)

        obstacle_map.update(empty_depth)
        self.assertGreater(obstacle_map.static_map.sum(), 0)
        obstacle_map.update(empty_depth)
        self.assertEqual(obstacle_map.static_map.sum(), 0)

    def test_ground_plane_is_removed_by_camera_height_projection(self):
        obstacle_map = LocalObstacleMap(
            image_width=64, image_height=64, hfov_deg=90.0,
            camera_height_m=0.24, min_obstacle_height_m=0.08,
        )
        depth = np.full((64, 64), obstacle_map.max_depth_m, dtype=np.float32)
        fy = 32.0
        center_y = 31.5
        for y in range(40, 64):
            depth[y, :] = obstacle_map.camera_height_m * fy / (y - center_y)

        obstacle_map.update(depth)

        self.assertEqual(obstacle_map.static_map.sum(), 0)
        self.assertGreater(obstacle_map.last_ground_filtered_points, 0)

    def test_ground_plane_is_removed_with_downward_camera_pitch(self):
        obstacle_map = LocalObstacleMap(
            image_width=64, image_height=64, hfov_deg=90.0,
            camera_height_m=0.24, camera_pitch_deg=5.0,
            min_obstacle_height_m=0.08,
        )
        depth = np.full((64, 64), obstacle_map.max_depth_m, dtype=np.float32)
        fy = 32.0
        center_y = 31.5
        pitch = np.deg2rad(5.0)
        for y in range(32, 64):
            downward_ray = (
                np.sin(pitch)
                + np.cos(pitch) * (y - center_y) / fy
            )
            depth[y, :] = obstacle_map.camera_height_m / downward_ray

        obstacle_map.update(depth)

        self.assertEqual(obstacle_map.static_map.sum(), 0)
        self.assertGreater(obstacle_map.last_ground_filtered_points, 0)

    def test_all_depth_free_space_mode_uses_ground_rays(self):
        obstacle_map = LocalObstacleMap(
            image_width=64, image_height=64, hfov_deg=90.0,
            camera_height_m=0.24, camera_pitch_deg=0.0,
            min_obstacle_height_m=0.08, free_space_all_depth=True,
        )
        depth = np.full((64, 64), obstacle_map.max_depth_m, dtype=np.float32)
        fy = 32.0
        center_y = 31.5
        for y in range(40, 64):
            depth[y, :] = obstacle_map.camera_height_m * fy / (y - center_y)

        obstacle_map.update(depth)

        self.assertGreater(obstacle_map.last_ground_filtered_points, 0)
        self.assertGreater(obstacle_map.last_free_cells, 0)

    def test_follow_waypoint_stops_before_target(self):
        obstacle_map = LocalObstacleMap()
        forward, left, mode = obstacle_map.choose_waypoint(
            (3.0, 0.0), desired_distance_m=1.5
        )
        self.assertIn(mode, ("map_direct", "map_astar"))
        self.assertGreater(forward, 0.0)
        self.assertAlmostEqual(left, 0.0)

    def test_waypoint_preserves_target_left_direction(self):
        obstacle_map = LocalObstacleMap(robot_radius_m=0.0)
        forward, left, mode = obstacle_map.choose_waypoint(
            (3.0, 2.0), desired_distance_m=1.0
        )

        self.assertIn(mode, ("map_direct", "map_astar"))
        self.assertGreater(forward, 0.0)
        self.assertGreater(left, 0.0)


if __name__ == "__main__":
    unittest.main()
