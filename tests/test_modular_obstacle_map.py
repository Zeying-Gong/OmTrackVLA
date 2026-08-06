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

    def test_follow_waypoint_stops_before_target(self):
        obstacle_map = LocalObstacleMap()
        forward, left, mode = obstacle_map.choose_waypoint(
            (3.0, 0.0), desired_distance_m=1.5
        )
        self.assertEqual(mode, "map_direct")
        self.assertAlmostEqual(forward, 1.5)
        self.assertAlmostEqual(left, 0.0)


if __name__ == "__main__":
    unittest.main()
