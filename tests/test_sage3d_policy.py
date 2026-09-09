import unittest

import numpy as np

from omtrackvla.data.sage3d_policy import (
    canonical_waypoints,
    moving_speed_ratio,
    source_waypoints,
    target_position_base,
    waypoint_time_offsets_s,
)


def steps(count=24, speed=0.6):
    return [
        {
            "step": index,
            "robot_pos": [0.1 * index, 0.0, 0.4],
            "robot_yaw": 0.0,
            "target_pos": [2.0 + speed * index / 30.0, 1.0, 0.0],
            "target_local": [2.0 + speed * index / 30.0 - 0.1 * index, 1.0],
        }
        for index in range(count)
    ]


class Sage3DPolicyTest(unittest.TestCase):
    def test_target_and_waypoints_use_forward_left_base_axes(self):
        values = steps()
        self.assertTrue(np.allclose(target_position_base(values[0]), [2.0, 1.0]))
        waypoints = canonical_waypoints(values, 0)
        self.assertEqual(waypoints[0], [0.0, 0.0])
        self.assertTrue(np.allclose(waypoints[-1], [2.1, 0.0]))

    def test_source_and_contract_offsets_are_intentionally_different(self):
        values = steps()
        contract = canonical_waypoints(values, 0)
        source = source_waypoints(values, 0)
        self.assertTrue(np.allclose(contract[1], [0.3, 0.0]))
        self.assertTrue(np.allclose(source[0], [0.1, 0.0]))
        self.assertTrue(np.allclose(source[-1], [2.2, 0.0]))

    def test_waypoint_timing_is_30hz_with_three_step_stride(self):
        self.assertTrue(
            np.allclose(waypoint_time_offsets_s(), [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        )

    def test_moving_speed_ratio_recovers_commanded_speed(self):
        self.assertAlmostEqual(moving_speed_ratio(steps(speed=0.6), 0.6), 1.0)

    def test_short_horizon_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "insufficient future"):
            canonical_waypoints(steps(count=21), 0)


if __name__ == "__main__":
    unittest.main()
