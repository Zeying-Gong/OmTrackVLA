import math
import unittest

from omtrackvla.data.evt_perception import (
    bbox_bearing_sincos,
    causal_history_indices,
    polar_from_habitat_state,
)


class EVTPerceptionGeometryTest(unittest.TestCase):
    def test_habitat_transform_uses_forward_col0_and_left_minus_col2(self):
        angle, distance = polar_from_habitat_state(
            {
                "robot_transform_world": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "robot_world_xyz_m": [0.0, 0.0, 0.0],
                "target_world_xyz_m": [2.0, 0.0, -2.0],
            }
        )
        self.assertAlmostEqual(angle, math.pi / 4.0)
        self.assertAlmostEqual(distance, math.sqrt(8.0))

    def test_bbox_right_is_negative_left_bearing(self):
        sine, cosine = bbox_bearing_sincos([0.6, 0.1, 0.8, 0.9])
        self.assertLess(sine, 0.0)
        self.assertGreater(cosine, 0.0)
        self.assertAlmostEqual(math.hypot(sine, cosine), 1.0)

    def test_timestamp_history_is_causal_and_left_padded(self):
        times = [index * 0.05 for index in range(21)]
        indices = causal_history_indices(times, 20, count=8, interval_s=0.1)
        self.assertEqual(indices[-1], 20)
        self.assertEqual(len(indices), 8)
        self.assertTrue(all(a <= b for a, b in zip(indices, indices[1:])))
        anchor_time = times[20]
        for slot, index in enumerate(indices):
            desired = anchor_time - (7 - slot) * 0.1
            self.assertLessEqual(times[index], desired + 1.0e-9)
        self.assertEqual(
            causal_history_indices(times, 1, count=8, interval_s=0.1)[:7],
            (0, 0, 0, 0, 0, 0, 0),
        )


if __name__ == "__main__":
    unittest.main()
