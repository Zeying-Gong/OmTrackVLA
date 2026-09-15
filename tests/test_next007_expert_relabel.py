import unittest

import numpy as np

from scripts.probe_next007_expert_relabel import (
    anchor_basis,
    causal_history_steps,
    forecast_goal_index,
    position_in_anchor_frame,
    replay_actions,
    target_progress_statistics,
    trajectory_statistics,
    waypoint_offsets,
)


class Next007ExpertRelabelTest(unittest.TestCase):
    def test_saved_policy_actions_require_contiguous_finite_steps(self):
        rollout = {
            "steps": [
                {
                    "step": 1,
                    "policy": {
                        "action": {"forward": 0.3, "lateral": -0.1, "yaw": 0.2}
                    },
                },
                {
                    "step": 2,
                    "policy": {
                        "action": {"forward": 0.0, "lateral": 0.0, "yaw": 0.0}
                    },
                },
            ]
        }
        self.assertEqual(replay_actions(rollout), [[0.3, -0.1, 0.2], [0.0, 0.0, 0.0]])
        rollout["steps"][1]["step"] = 3
        with self.assertRaisesRegex(ValueError, "contiguous"):
            replay_actions(rollout)

    def test_habitat_transform_maps_world_motion_to_forward_left(self):
        transform = np.array(
            [
                [0.0, 0.0, -1.0, 10.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 20.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        forward, left = anchor_basis(transform)
        point = np.array([10.0, 0.0, 21.0])
        xy = position_in_anchor_frame(point, [10.0, 0.0, 20.0], forward, left)
        np.testing.assert_allclose(xy, [1.0, 0.0], atol=1e-7)
        left_point = np.array([11.0, 0.0, 20.0])
        left_xy = position_in_anchor_frame(
            left_point, [10.0, 0.0, 20.0], forward, left
        )
        np.testing.assert_allclose(left_xy, [0.0, 1.0], atol=1e-7)

    def test_architecture_v1_offsets_are_exactly_eight_points(self):
        self.assertEqual(waypoint_offsets(21, 3), [0, 3, 6, 9, 12, 15, 18, 21])
        with self.assertRaisesRegex(ValueError, "exactly 8"):
            waypoint_offsets(18, 3)

    def test_v2_history_uses_stride_three_and_left_padding(self):
        self.assertEqual(
            causal_history_steps(21, history_size=8, history_stride_steps=3),
            [0, 3, 6, 9, 12, 15, 18, 21],
        )
        self.assertEqual(
            causal_history_steps(5, history_size=8, history_stride_steps=3),
            [0, 0, 0, 0, 0, 0, 2, 5],
        )

    def test_forecast_goal_index_clamps_to_horizon(self):
        self.assertEqual(forecast_goal_index(0, 21, 6), 6)
        self.assertEqual(forecast_goal_index(18, 21, 6), 21)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            forecast_goal_index(0, 21, -1)

    def test_trajectory_statistics_distinguish_path_and_terminal_radius(self):
        stats = trajectory_statistics([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
        self.assertAlmostEqual(stats["path_length_m"], 2.0)
        self.assertAlmostEqual(stats["terminal_radius_m"], 0.0)
        self.assertAlmostEqual(stats["maximum_segment_m"], 1.0)

    def test_target_progress_compares_with_stationary_anchor(self):
        stats = target_progress_statistics([0.5, 0.0], [2.0, 0.0])
        self.assertAlmostEqual(
            stats["stationary_anchor_final_target_distance_m"], 2.0
        )
        self.assertAlmostEqual(stats["expert_final_target_distance_m"], 1.5)
        self.assertAlmostEqual(
            stats["distance_improvement_over_stationary_anchor_m"], 0.5
        )
        self.assertAlmostEqual(
            stats["terminal_motion_target_alignment_cosine"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
