import math
import unittest
from types import SimpleNamespace

import numpy as np

from oracle_modular_follow import (
    OracleFollowController,
    OracleNavmeshFollower,
    TargetObservation,
    bbox_to_footpoint,
    target_mask_to_bbox,
)
from oracle_modular_follow_v6 import OracleNavmeshFollowerV6
from rgb_person_perception import bbox_depth_to_relative, bbox_iou, metric_depth


def make_target(forward, left, visible=True):
    distance = math.hypot(forward, left)
    return TargetObservation(
        visible=visible,
        bbox_xyxy=(10, 5, 20, 30) if visible else None,
        footpoint_uv=(15.0, 30.0) if visible else None,
        relative_xy=(forward, left),
        range_m=distance,
        bearing_rad=math.atan2(left, forward),
        mask_area=100 if visible else 0,
        confidence=1.0 if visible else 0.0,
    )


class OracleModularFollowTest(unittest.TestCase):
    def test_rgb_bbox_geometry_uses_depth_without_oracle_pose(self):
        depth = np.full((100, 100), 0.2, dtype=np.float32)
        forward, left = bbox_depth_to_relative((40, 10, 60, 90), depth)
        self.assertAlmostEqual(forward, 2.0)
        self.assertAlmostEqual(left, -0.02, places=2)

    def test_normalized_depth_is_converted_to_metres(self):
        depth = metric_depth(np.full((2, 2, 1), 0.3, dtype=np.float32))
        np.testing.assert_allclose(depth, 3.0)

    def test_bbox_iou_handles_overlap_and_disjoint_boxes(self):
        self.assertAlmostEqual(bbox_iou((0, 0, 10, 10), (5, 0, 15, 10)), 1.0 / 3.0)
        self.assertEqual(bbox_iou((0, 0, 2, 2), (3, 3, 4, 4)), 0.0)

    def test_mask_geometry_uses_xyxy_and_bottom_contact(self):
        mask = np.zeros((20, 30), dtype=bool)
        mask[4:15, 8:18] = True
        self.assertEqual(target_mask_to_bbox(mask), (8, 4, 17, 14))
        self.assertEqual(bbox_to_footpoint((8, 4, 17, 14)), (12.5, 14.0))

    def test_centered_far_target_moves_forward_without_turning(self):
        action = OracleFollowController()(make_target(3.0, 0.0))
        self.assertGreater(action.forward, 0.0)
        self.assertEqual(action.lateral, 0.0)
        self.assertEqual(action.yaw, 0.0)

    def test_left_target_turns_left_and_large_bearing_stops_translation(self):
        action = OracleFollowController()(make_target(0.5, 2.0))
        self.assertEqual(action.forward, 0.0)
        self.assertGreater(action.yaw, 0.0)

    def test_invisible_target_stops(self):
        action = OracleFollowController()(make_target(2.0, 0.0, visible=False))
        self.assertEqual(action.as_habitat(), [0.0, 0.0, 0.0])

    def test_navmesh_follower_defaults_match_habitat_action_limits(self):
        controller = OracleNavmeshFollower()
        self.assertEqual(controller.max_forward, 1.0)
        self.assertEqual(controller.max_lateral, 1.0)
        self.assertEqual(controller.max_yaw, 1.0)
        controller._previous_target_position = np.ones(3)
        controller.reset()
        self.assertIsNone(controller._previous_target_position)

    def test_reset_can_force_the_opposite_evasion_side(self):
        controller = OracleNavmeshFollower()
        controller.reset(evasion_side=-1.0)
        self.assertEqual(controller._evasion_side, -1.0)

    def test_incoming_motion_persists_across_stationary_animation_frames(self):
        controller = OracleNavmeshFollower(incoming_memory_steps=3)
        robot = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        target = np.array([2.0, 0.0, 0.0], dtype=np.float32)
        self.assertEqual(controller._update_incoming(robot, target), (False, 0.0))

        target = np.array([1.8, 0.0, 0.0], dtype=np.float32)
        incoming, motion = controller._update_incoming(robot, target)
        self.assertTrue(incoming)
        self.assertGreater(motion, 0.0)
        for _ in range(2):
            incoming, motion = controller._update_incoming(robot, target)
            self.assertTrue(incoming)
            self.assertEqual(motion, 0.0)

    def test_target_moving_away_clears_incoming_memory(self):
        controller = OracleNavmeshFollower(incoming_memory_steps=4)
        robot = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        controller._update_incoming(
            robot, np.array([2.0, 0.0, 0.0], dtype=np.float32)
        )
        incoming, _ = controller._update_incoming(
            robot, np.array([1.8, 0.0, 0.0], dtype=np.float32)
        )
        self.assertTrue(incoming)
        incoming, motion = controller._update_incoming(
            robot, np.array([2.0, 0.0, 0.0], dtype=np.float32)
        )
        self.assertFalse(incoming)
        self.assertLess(motion, 0.0)

    def test_invisible_target_inside_old_hold_band_uses_coordinates(self):
        controller = OracleNavmeshFollower(min_distance_m=1.2, max_distance_m=1.5)
        controller._path_waypoint = lambda *args: None
        robot = SimpleNamespace(base_pos=np.zeros(3, dtype=np.float32))
        target_agent = SimpleNamespace(
            base_pos=np.array([1.4, 0.0, 0.0], dtype=np.float32)
        )
        decision = controller(
            None, robot, target_agent, make_target(1.4, 0.0, visible=False)
        )
        self.assertEqual(decision.mode, "approach_coordinate")

    def test_incoming_motion_uses_radial_tracking_inside_three_metres(self):
        controller = OracleNavmeshFollower()
        controller._path_waypoint = lambda *args: None
        controller._local_xy = lambda *args: (0.0, 0.0)
        robot = SimpleNamespace(base_pos=np.zeros(3, dtype=np.float32))
        target_agent = SimpleNamespace(
            base_pos=np.array([2.6, 0.0, 0.0], dtype=np.float32)
        )
        controller(None, robot, target_agent, make_target(2.6, 0.0))
        target_agent.base_pos = np.array([2.3, 0.0, 0.0], dtype=np.float32)
        decision = controller(None, robot, target_agent, make_target(2.3, 0.0))
        self.assertEqual(decision.mode, "track_incoming")

    def test_incoming_uses_retreat_clearance_as_distance_target(self):
        controller = OracleNavmeshFollower(incoming_retreat_distance_m=2.0)
        controller._update_incoming = lambda *args: (True, 0.1)
        controller._local_xy = lambda *args: (0.0, 0.0)
        desired_distances = []
        controller._motion_tracking_translation = (
            lambda target, motion, desired_distance_m=None:
            (desired_distances.append(desired_distance_m) or (0.0, 0.0))
        )
        robot = SimpleNamespace(base_pos=np.zeros(3, dtype=np.float32))
        target_agent = SimpleNamespace(
            base_pos=np.array([1.5, 0.0, 0.0], dtype=np.float32)
        )
        decision = controller(
            None, robot, target_agent, make_target(1.5, 0.0)
        )
        self.assertEqual(decision.mode, "track_incoming")
        self.assertEqual(desired_distances, [2.0])

    def test_yield_pass_never_stops_inside_retreat_clearance(self):
        controller = OracleNavmeshFollower(incoming_retreat_distance_m=2.0)
        controller._update_incoming = lambda *args: (True, 0.1)
        controller._local_xy = lambda *args: (0.0, 0.0)
        controller._pass_yield_active = True
        robot = SimpleNamespace(base_pos=np.zeros(3, dtype=np.float32))
        target_agent = SimpleNamespace(
            base_pos=np.array([1.5, 0.0, 0.0], dtype=np.float32)
        )
        decision = controller(
            None, robot, target_agent, make_target(1.5, 0.0)
        )
        self.assertEqual(decision.mode, "track_incoming")

    def test_incoming_uses_navmesh_evasion_inside_two_metres(self):
        controller = OracleNavmeshFollower(evasion_start_distance_m=2.0)
        controller._update_incoming = lambda *args: (True, 0.1)
        controller._local_xy = lambda *args: (0.0, 0.35)
        controller._evasion_waypoint = lambda *args: np.ones(3)
        robot = SimpleNamespace(base_pos=np.zeros(3, dtype=np.float32))
        target_agent = SimpleNamespace(
            base_pos=np.array([1.5, 0.0, 0.0], dtype=np.float32)
        )
        decision = controller(
            object(), robot, target_agent, make_target(1.5, 0.0)
        )
        self.assertEqual(decision.mode, "evade_incoming")
        self.assertEqual(decision.action.lateral, 0.25)

    def test_motion_tracking_combines_target_velocity_and_distance(self):
        controller = OracleNavmeshFollower(
            min_distance_m=1.2,
            max_distance_m=1.5,
            radial_distance_gain=2.0,
            target_motion_gain=2.5,
        )
        target = make_target(1.2, 0.0)
        self.assertEqual(
            controller._motion_tracking_translation(target, (0.2, 0.1)),
            (0.0, 0.25),
        )
        target = make_target(0.85, 0.0)
        forward, lateral = controller._motion_tracking_translation(target, (0.0, 0.0))
        self.assertAlmostEqual(forward, -0.7)
        self.assertEqual(lateral, 0.0)
        target = make_target(1.2, 0.0)
        forward, _ = controller._motion_tracking_translation(
            target, (0.0, 0.0), desired_distance_m=1.5
        )
        self.assertAlmostEqual(forward, -0.6)

    def test_normal_tracking_targets_upper_distance_for_stride_margin(self):
        controller = OracleNavmeshFollower(
            min_distance_m=1.2, max_distance_m=1.5
        )
        controller._target_has_moved = True
        controller._update_incoming = lambda *args: (False, 0.0)
        controller._local_xy = lambda *args: (0.0, 0.0)
        desired_distances = []
        controller._motion_tracking_translation = (
            lambda target, motion, desired_distance_m=None:
            (desired_distances.append(desired_distance_m) or (0.0, 0.0))
        )
        robot = SimpleNamespace(base_pos=np.zeros(3, dtype=np.float32))
        target_agent = SimpleNamespace(
            base_pos=np.array([1.3, 0.0, 0.0], dtype=np.float32)
        )
        decision = controller(
            None, robot, target_agent, make_target(1.3, 0.0)
        )
        self.assertEqual(decision.mode, "track_distance")
        self.assertEqual(desired_distances, [1.5])

    def test_translation_slew_limits_startup_but_allows_emergency_reverse(self):
        controller = OracleNavmeshFollower(
            translation_slew_per_step=0.25,
            emergency_translation_slew=0.6,
        )
        self.assertEqual(controller._limit_translation(1.0, 0.0), (0.25, 0.0))
        self.assertEqual(controller._limit_translation(1.0, 0.0), (0.5, 0.0))
        self.assertAlmostEqual(
            controller._limit_translation(-1.0, 0.0, emergency=True)[0], -0.1
        )

    def test_target_motion_releases_startup_approach_limit(self):
        controller = OracleNavmeshFollower()
        robot = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        controller._update_incoming(
            robot, np.array([2.0, 0.0, 0.0], dtype=np.float32)
        )
        self.assertFalse(controller._target_has_moved)
        controller._update_incoming(
            robot, np.array([1.8, 0.0, 0.0], dtype=np.float32)
        )
        self.assertTrue(controller._target_has_moved)

    def test_stationary_target_inside_retreat_clearance_is_held(self):
        controller = OracleNavmeshFollower(incoming_retreat_distance_m=2.0)
        robot = SimpleNamespace(base_pos=np.zeros(3, dtype=np.float32))
        target_agent = SimpleNamespace(
            base_pos=np.array([1.6, 0.0, 0.0], dtype=np.float32)
        )
        decision = controller(
            None, robot, target_agent, make_target(1.6, 0.0)
        )
        self.assertEqual(decision.mode, "hold_startup")
        self.assertEqual(decision.action.forward, 0.0)

    def test_tangential_target_motion_releases_startup_hold(self):
        controller = OracleNavmeshFollower(incoming_motion_threshold_m=0.03)
        robot = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        controller._update_incoming(
            robot, np.array([1.6, 0.0, 0.0], dtype=np.float32)
        )
        controller._update_incoming(
            robot, np.array([1.6, 0.0, 0.1], dtype=np.float32)
        )
        self.assertTrue(controller._target_has_moved)

    def test_persistent_incoming_motion_triggers_one_pass_yield(self):
        controller = OracleNavmeshFollower(
            incoming_memory_steps=4,
            pass_yield_after_steps=3,
            pass_yield_steps=2,
        )
        robot = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        target = np.array([2.0, 0.0, 0.0], dtype=np.float32)
        controller._update_incoming(robot, target)
        for expected_active in (False, False, True, True):
            target[0] -= 0.1
            controller._update_incoming(robot, target)
            self.assertEqual(controller._pass_yield_active, expected_active)
        self.assertFalse(controller._pass_yield_used)
        self.assertEqual(controller._incoming_duration, 0)

    def test_v5_defaults_leave_new_safety_policy_disabled(self):
        controller = OracleNavmeshFollower(
            min_distance_m=1.2, max_distance_m=1.5
        )
        self.assertEqual(controller.tracking_distance_m, 1.5)
        self.assertFalse(controller.prioritize_visibility)
        self.assertEqual(controller.incoming_safe_retreat_distance_m, 0.0)
        self.assertEqual(controller.emergency_safe_retreat_distance_m, 0.0)
        self.assertEqual(controller.tracking_mask_max_pixels, 0)
        self.assertEqual(controller.visibility_reframe_after_steps, 0)
        self.assertEqual(controller.coordinate_approach_min_scale, 0.1)

    def test_v6_prioritizes_low_mask_visibility_over_distance_tracking(self):
        controller = OracleNavmeshFollowerV6(
            min_distance_m=1.2,
            max_distance_m=1.5,
            tracking_mask_min_pixels=10000,
        )
        controller._target_has_moved = True
        controller._update_incoming = lambda *args: (False, 0.0)
        controller._path_waypoint = lambda *args: np.array(
            [1.3, 0.0, 0.0], dtype=np.float32
        )
        controller._local_xy = lambda *args: (1.0, 0.0)
        robot = SimpleNamespace(base_pos=np.zeros(3, dtype=np.float32))
        target_agent = SimpleNamespace(
            base_pos=np.array([1.3, 0.0, 0.0], dtype=np.float32)
        )

        decision = controller(
            object(), robot, target_agent, make_target(1.3, 0.0)
        )

        self.assertEqual(decision.mode, "approach_visibility")
        self.assertEqual(controller.tracking_distance_m, 1.3)

    def test_v6_uses_navmesh_retreat_for_close_incoming_target(self):
        controller = OracleNavmeshFollowerV6(
            min_distance_m=1.2, max_distance_m=1.5
        )
        controller._update_incoming = lambda *args: (True, 0.1)
        controller._retreat_goal = lambda *args: np.array(
            [-1.0, 0.0, 0.0], dtype=np.float32
        )
        controller._path_waypoint = lambda *args: np.array(
            [-1.0, 0.0, 0.0], dtype=np.float32
        )
        controller._local_xy = lambda robot, point: (float(point[0]), 0.0)
        robot = SimpleNamespace(base_pos=np.zeros(3, dtype=np.float32))
        target_agent = SimpleNamespace(
            base_pos=np.array([1.5, 0.0, 0.0], dtype=np.float32)
        )

        decision = controller(
            object(), robot, target_agent, make_target(1.5, 0.0)
        )

        self.assertEqual(decision.mode, "retreat_safety")
        self.assertLess(decision.action.forward, 0.0)

    def test_v6_retreats_when_target_mask_is_too_large(self):
        controller = OracleNavmeshFollowerV6(
            min_distance_m=1.2, max_distance_m=1.5
        )
        controller._update_incoming = lambda *args: (False, 0.0)
        controller._local_xy = lambda *args: (0.0, 0.0)
        robot = SimpleNamespace(base_pos=np.zeros(3, dtype=np.float32))
        target_agent = SimpleNamespace(
            base_pos=np.array([1.1, 0.0, 0.0], dtype=np.float32)
        )
        target = make_target(1.1, 0.0)
        target = TargetObservation(
            **{**target.__dict__, "mask_area": controller.tracking_mask_max_pixels}
        )

        decision = controller(object(), robot, target_agent, target)

        self.assertEqual(decision.mode, "track_visibility_clearance")
        self.assertLess(decision.action.forward, 0.0)

    def test_v6_stops_retreating_after_close_target_becomes_invisible(self):
        controller = OracleNavmeshFollowerV6(
            min_distance_m=1.2, max_distance_m=1.5
        )
        controller._update_incoming = lambda *args: (True, 0.1)
        controller._path_waypoint = lambda *args: None
        robot = SimpleNamespace(base_pos=np.zeros(3, dtype=np.float32))
        target_agent = SimpleNamespace(
            base_pos=np.array([0.8, 0.0, 0.0], dtype=np.float32)
        )

        decision = controller(
            object(), robot, target_agent, make_target(0.8, 0.0, visible=False)
        )

        self.assertNotEqual(decision.mode, "retreat_safety")


if __name__ == "__main__":
    unittest.main()
