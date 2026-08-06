import unittest

import numpy as np

from modular_obstacle_map import LocalObstacleMap


class LocalObstacleMapTest(unittest.TestCase):
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

    def test_static_obstacles_persist_across_robot_motion(self):
        obstacle_map = LocalObstacleMap(image_width=64, image_height=64)
        depth = np.full((64, 64), 2.0, dtype=np.float32)
        obstacle_map.update(depth, robot_pose=(0.0, 0.0, 0.0))
        first = obstacle_map.static_map.copy()

        empty_depth = np.full((64, 64), obstacle_map.max_depth_m, dtype=np.float32)
        obstacle_map.update(empty_depth, robot_pose=(0.5, 0.0, 0.0))

        self.assertTrue(np.all(obstacle_map.static_map[first > 0] > 0))
        self.assertGreater(obstacle_map.trajectory_map.sum(), 1)

    def test_dynamic_layer_is_rebuilt_each_frame(self):
        obstacle_map = LocalObstacleMap(image_width=64, image_height=64)
        depth = np.full((64, 64), 2.0, dtype=np.float32)
        mask = np.zeros_like(depth, dtype=bool)
        mask[24:40, 26:38] = True
        obstacle_map.update(depth, dynamic_mask=mask)
        self.assertGreater(obstacle_map.dynamic_map.sum(), 0)

        obstacle_map.update(depth, dynamic_mask=np.zeros_like(mask))
        self.assertEqual(obstacle_map.dynamic_map.sum(), 0)

    def test_follow_waypoint_stops_before_target(self):
        obstacle_map = LocalObstacleMap()
        forward, left, mode = obstacle_map.choose_waypoint(
            (3.0, 0.0), desired_distance_m=1.5
        )
        self.assertIn(mode, ("map_direct", "map_astar"))
        self.assertGreater(forward, 0.0)
        self.assertAlmostEqual(left, 0.0)


if __name__ == "__main__":
    unittest.main()
