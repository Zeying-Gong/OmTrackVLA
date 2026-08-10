import math
import unittest
from types import SimpleNamespace

import numpy as np

from oracle_modular_follow import (
    OracleFollowController,
    OraclePerception,
    OracleNavmeshFollower,
    ContinuousAction,
    MapReactiveFollower,
    ModularReactiveFollower,
    NavmeshConnectivityDiagnostic,
    TargetObservation,
    bbox_to_footpoint,
    target_mask_to_bbox,
    mask_connected_bboxes,
    select_target_component_bbox,
)
from oracle_modular_follow_v6 import OracleNavmeshFollowerV6
from rgb_person_perception import (
    RGBPersonPerception,
    bbox_depth_to_relative,
    bbox_iou,
    color_histogram,
    metric_depth,
)


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


class OracleComponentSelectionTest(unittest.TestCase):
    def test_connected_target_regions_are_separate_boxes(self):
        mask = np.zeros((40, 80), dtype=bool)
        mask[10:30, 5:20] = True
        mask[8:32, 55:75] = True

        boxes = mask_connected_bboxes(mask)

        self.assertEqual(boxes, [(5, 10, 19, 29), (55, 8, 74, 31)])

    def test_relative_target_bearing_selects_matching_component(self):
        mask = np.zeros((40, 80), dtype=bool)
        mask[10:30, 5:20] = True
        mask[8:32, 55:75] = True

        bbox = select_target_component_bbox(
            mask, relative_xy=(2.0, 1.0), image_width=80
        )

        self.assertEqual(bbox, (55, 8, 74, 31))

    def test_oracle_perception_marks_only_selected_component(self):
        panoptic = np.zeros((40, 80), dtype=np.int32)
        panoptic[10:30, 5:20] = 1092
        panoptic[8:32, 55:75] = 1092
        perception = OraclePerception()

        target = perception(
            np.zeros((40, 80, 3), dtype=np.uint8),
            panoptic,
            target_semantic_id=1092,
            relative_xy=(2.0, 1.0),
        )

        self.assertEqual(target.bbox_xyxy, (55, 8, 74, 31))
        self.assertEqual(
            sum(bool(c["selected"]) for c in perception.last_candidate_diagnostics),
            1,
        )

    def test_component_depth_prevents_projection_identity_swap(self):
        mask = np.zeros((40, 80), dtype=bool)
        mask[10:30, 5:20] = True
        mask[8:32, 55:75] = True
        depth = np.full((40, 80), 10.0, dtype=np.float32)
        depth[10:30, 5:20] = 2.0
        depth[8:32, 55:75] = 0.8

        bbox = select_target_component_bbox(
            mask,
            relative_xy=(2.0, 1.0),
            image_width=80,
            previous_bbox=(55, 8, 74, 31),
            depth=depth,
        )

        self.assertEqual(bbox, (5, 10, 19, 29))

    def test_sensor_projection_has_priority_over_mirror_depth(self):
        mask = np.zeros((40, 80), dtype=bool)
        mask[10:30, 5:20] = True
        mask[8:32, 55:75] = True
        depth = np.full((40, 80), 10.0, dtype=np.float32)
        depth[10:30, 5:20] = 2.0
        depth[8:32, 55:75] = 0.8

        bbox = select_target_component_bbox(
            mask,
            relative_xy=(2.0, -1.0),
            image_width=80,
            previous_bbox=(5, 10, 19, 29),
            depth=depth,
            projected_uv=(65.0, 31.0),
        )

        self.assertEqual(bbox, (55, 8, 74, 31))


class OracleModularFollowTest(unittest.TestCase):
    def test_navmesh_diagnostic_distinguishes_human_and_spot_connectivity(self):
        raw_free = np.zeros((120, 120), dtype=np.uint8)
        raw_free[10:110, 10:50] = 1
        raw_free[10:110, 70:110] = 1
        raw_free[58:62, 50:70] = 1

        class FakePathfinder:
            nav_mesh_settings = SimpleNamespace(agent_radius=0.30)

            def get_topdown_view(self, meters_per_pixel, height):
                return raw_free

            def get_bounds(self):
                return np.array((0.0, 0.0, 0.0)), np.array((12.0, 1.0, 12.0))

        diagnostic = NavmeshConnectivityDiagnostic(
            meters_per_pixel=0.10, spot_radius_m=0.50
        )
        diagnostic.update(
            SimpleNamespace(pathfinder=FakePathfinder()),
            np.array((2.0, 0.0, 6.0)),
            np.array((10.0, 0.0, 6.0)),
        )

        self.assertTrue(diagnostic.last_connectivity["raw_connected"])
        self.assertFalse(diagnostic.last_connectivity["spot_connected"])
        self.assertIsNotNone(diagnostic.last_visualization)

    def test_reactive_controller_ignores_privileged_arguments(self):
        controller = ModularReactiveFollower()
        target = make_target(3.0, 0.0)

        decision = controller(None, None, object(), target)

        self.assertEqual(decision.mode, "reactive_approach")
        self.assertGreater(decision.action.forward, 0.0)

    def test_reactive_pointgoal_can_use_invisible_exact_target(self):
        controller = ModularReactiveFollower(use_invisible_pointgoal=True)
        target = make_target(3.0, 0.0, visible=False)

        decision = controller(None, None, object(), target)

        self.assertEqual(decision.mode, "reactive_pointgoal")
        self.assertGreater(decision.action.forward, 0.0)

    def test_reactive_controller_searches_without_target(self):
        controller = ModularReactiveFollower()
        target = make_target(2.0, 0.5, visible=False)

        decision = controller(None, None, object(), target)

        self.assertEqual(decision.mode, "reactive_search")
        self.assertEqual(decision.action.forward, 0.0)

    def test_c4_retreats_away_from_off_axis_person(self):
        controller = MapReactiveFollower(predictive_human_safety=True)
        controller(None, None, None, make_target(1.6, 0.4))

        decision = controller(None, None, None, make_target(1.0, 0.4))

        self.assertEqual(decision.mode, "map_predictive_human_retreat")
        self.assertLess(decision.action.forward, 0.0)
        self.assertLess(decision.action.lateral, 0.0)

    def test_c3_does_not_enable_predictive_human_retreat(self):
        controller = MapReactiveFollower(predictive_human_safety=False)
        controller(None, None, None, make_target(1.6, 0.4))

        decision = controller(None, None, None, make_target(1.0, 0.4))

        self.assertNotEqual(decision.mode, "map_predictive_human_retreat")

    def test_c4_yields_sideways_during_sustained_head_on_closure(self):
        controller = MapReactiveFollower(predictive_human_safety=True)
        controller(None, None, None, make_target(1.8, 0.0))
        controller(None, None, None, make_target(1.2, 0.0))

        decision = controller(None, None, None, make_target(1.0, 0.0))

        self.assertEqual(decision.mode, "map_predictive_human_yield")
        self.assertGreater(abs(decision.action.lateral), 0.5)

    def test_c4_proactively_yields_near_follow_band_for_incoming_person(self):
        controller = MapReactiveFollower(predictive_human_safety=True)
        controller(None, None, None, make_target(2.0, 0.0))

        retreat = controller(None, None, None, make_target(1.7, 0.0))
        yield_decision = controller(None, None, None, make_target(1.5, 0.0))

        self.assertEqual(retreat.mode, "map_predictive_human_retreat")
        self.assertLess(retreat.action.forward, 0.0)
        self.assertEqual(yield_decision.mode, "map_predictive_human_yield")
        self.assertGreater(abs(yield_decision.action.lateral), 0.5)

    def test_c4_does_not_yield_for_small_stationary_range_jitter(self):
        controller = MapReactiveFollower(predictive_human_safety=True)
        controller(None, None, None, make_target(1.75, 0.0))

        decision = controller(None, None, None, make_target(1.73, 0.0))

        self.assertFalse(decision.mode.startswith("map_predictive_human_"))

    def test_c4_keeps_yield_active_while_person_is_still_closing(self):
        controller = MapReactiveFollower(predictive_human_safety=True)
        controller(None, None, None, make_target(2.0, 0.0))
        controller(None, None, None, make_target(1.7, 0.0))

        decision = controller(None, None, None, make_target(1.9, 0.0))

        self.assertEqual(decision.mode, "map_predictive_human_yield")

    def test_c4_visibility_portal_bypasses_follow_band_hold(self):
        controller = MapReactiveFollower(
            predictive_human_safety=True,
            use_invisible_pointgoal=True,
        )
        controller._visibility_portal_episode = (1.0, 0.0)

        decision = controller(
            None, None, None, make_target(1.3, 0.0, visible=False)
        )

        self.assertEqual(decision.mode, "map_history_waypoint")
        self.assertGreater(decision.action.forward, 0.0)

    def test_c4_visibility_portal_ignores_occluded_euclidean_near_distance(self):
        controller = MapReactiveFollower(
            predictive_human_safety=True,
            use_invisible_pointgoal=True,
        )
        controller._visibility_portal_episode = (1.0, 0.0)

        decision = controller(
            None, None, None, make_target(1.0, 0.0, visible=False)
        )

        self.assertEqual(decision.mode, "map_history_waypoint")
        self.assertGreater(decision.action.forward, 0.0)

    def test_map_controller_turns_in_place_for_sharp_astar_carrot(self):
        controller = MapReactiveFollower(use_invisible_pointgoal=True)
        controller.obstacle_map.choose_waypoint = lambda *args, **kwargs: (
            0.1, 1.0, "map_astar"
        )

        decision = controller(
            None, None, None, make_target(3.0, 0.0, visible=False)
        )

        self.assertGreater(decision.action.forward, 0.0)
        self.assertEqual(decision.action.lateral, 0.0)
        self.assertGreater(decision.action.yaw, 0.0)

    def test_c4_invisible_safe_target_releases_predictive_yield(self):
        controller = MapReactiveFollower(
            predictive_human_safety=True,
            use_invisible_pointgoal=True,
        )
        controller._human_safety_active = True
        controller._human_safety_steps = 3

        decision = controller(
            None, None, None, make_target(1.8, 0.0, visible=False)
        )

        self.assertFalse(decision.mode.startswith("map_predictive_human_"))

    def test_c4_invisible_in_band_plans_into_target_room(self):
        controller = MapReactiveFollower(
            predictive_human_safety=True,
            use_invisible_pointgoal=True,
        )

        decision = controller(
            None, None, None, make_target(1.6, 0.5, visible=False)
        )

        self.assertIn(decision.mode, ("map_direct_waypoint", "map_astar_waypoint"))
        self.assertGreater(decision.action.forward, 0.0)

    def test_c4_incoming_occlusion_discards_stale_visibility_portal(self):
        controller = MapReactiveFollower(
            predictive_human_safety=True,
            use_invisible_pointgoal=True,
        )
        controller._visibility_portal_episode = (1.0, 0.0)
        controller._target_closing_ema = 0.08
        controller._human_safety_active = True
        controller._human_safety_steps = 3

        decision = controller(
            None, None, None, make_target(1.8, 0.0, visible=False)
        )

        self.assertIsNone(controller._visibility_portal_episode)
        self.assertTrue(decision.mode.startswith("map_predictive_human_"))
        self.assertLessEqual(decision.action.forward, -0.8)
        self.assertGreaterEqual(abs(decision.action.lateral), 0.7)

    def test_c4_finishes_occluded_pass_with_committed_reframe_turn(self):
        controller = MapReactiveFollower(
            predictive_human_safety=True,
            use_invisible_pointgoal=True,
        )
        controller._human_safety_active = True
        controller._occluded_pass_active = True
        controller._human_evasion_direction = -1.0
        controller._target_closing_ema = 0.0

        decision = controller(
            None, None, None, make_target(1.8, -0.5, visible=False)
        )

        self.assertEqual(decision.mode, "map_post_pass_reframe")
        self.assertLess(decision.action.forward, 0.0)
        self.assertGreater(decision.action.yaw, 0.0)

    def test_c4_does_not_treat_robot_approach_as_incoming_person(self):
        controller = MapReactiveFollower(predictive_human_safety=True)
        controller(None, None, None, make_target(1.8, 0.0))
        controller.obstacle_map.robot_pose = (0.5, 0.0, 0.0)

        decision = controller(None, None, None, make_target(1.3, 0.0))

        self.assertFalse(decision.mode.startswith("map_predictive_human_"))

    def test_c4_marks_direction_after_two_rejected_translations(self):
        controller = MapReactiveFollower(motion_blocked_feedback=True)
        action = ContinuousAction(0.7, 0.0, 0.0)

        controller.record_action(action, "map_direct_waypoint")
        controller._update_motion_feedback((0.0, 0.0, 0.0))
        self.assertFalse(controller.last_motion_block_added)
        controller.record_action(action, "map_direct_waypoint")
        controller._update_motion_feedback((0.0, 0.0, 0.0))

        self.assertTrue(controller.last_motion_block_added)
        self.assertGreater(controller.obstacle_map.motion_blocked_map.sum(), 0)

    def test_c3_does_not_add_motion_block_feedback(self):
        controller = MapReactiveFollower(motion_blocked_feedback=False)
        controller.record_action(
            ContinuousAction(0.7, 0.0, 0.0), "map_direct_waypoint"
        )
        controller._update_motion_feedback((0.0, 0.0, 0.0))

        self.assertEqual(controller.obstacle_map.motion_blocked_map.sum(), 0)

    def test_c4_does_not_learn_obstacle_while_turning_sharply(self):
        controller = MapReactiveFollower(motion_blocked_feedback=True)
        action = ContinuousAction(0.7, 0.0, -1.0)

        controller.record_action(action, "map_direct_waypoint")
        controller._update_motion_feedback((0.0, 0.0, -0.1))
        controller.record_action(action, "map_direct_waypoint")
        controller._update_motion_feedback((0.0, 0.0, -0.2))

        self.assertFalse(controller.last_motion_block_added)
        self.assertEqual(controller.obstacle_map.motion_blocked_map.sum(), 0)

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

    def test_goal_crop_appearance_selects_matching_person(self):
        image = np.zeros((40, 80, 3), dtype=np.uint8)
        image[:, :40] = (220, 20, 20)
        image[:, 40:] = (20, 20, 220)
        perception = RGBPersonPerception.__new__(RGBPersonPerception)
        perception.reid_model = None
        perception._bbox = None
        perception._goal_embedding = None
        perception._track_embedding = None
        perception._bbox_velocity = np.zeros(4, dtype=np.float32)
        perception.reid_threshold = 0.55
        perception.ambiguity_margin = 0.04
        perception._reference_hist = color_histogram(
            image[:, 40:], (0, 0, 40, 40)
        )
        selected = perception._select(
            image,
            [
                (np.array((0, 0, 40, 40), dtype=np.float32), 0.99),
                (np.array((40, 0, 80, 40), dtype=np.float32), 0.90),
            ],
        )
        np.testing.assert_array_equal(selected[0], (40, 0, 80, 40))

    def test_tracker_rejects_abrupt_large_occluder(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[10:35, 45:55] = (20, 20, 220)
        image[5:95, 5:80] = (220, 20, 20)
        perception = RGBPersonPerception.__new__(RGBPersonPerception)
        perception.reid_model = None
        perception._bbox = np.array((45, 10, 55, 35), dtype=np.float32)
        perception._bbox_velocity = np.zeros(4, dtype=np.float32)
        perception._goal_embedding = None
        perception._track_embedding = None
        perception._reference_hist = color_histogram(
            np.full((25, 10, 3), (20, 20, 220), dtype=np.uint8),
            (0, 0, 10, 25),
        )
        perception._track_hist = perception._reference_hist.copy()
        perception._missed_steps = 0
        perception.association_threshold = 0.20
        perception.reid_threshold = 0.55
        perception.ambiguity_margin = 0.04
        selected = perception._select(
            image,
            [(np.array((35, 5, 65, 95), dtype=np.float32), 0.99)],
        )
        self.assertIsNone(selected)

    def test_reacquisition_rejects_weak_goal_reid_match(self):
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        perception = RGBPersonPerception.__new__(RGBPersonPerception)
        perception.reid_model = object()
        perception._embed_crops = lambda crops: [
            np.array((0.6, 0.8), dtype=np.float32) for _ in crops
        ]
        perception._bbox = np.array((10, 5, 30, 35), dtype=np.float32)
        perception._bbox_velocity = np.zeros(4, dtype=np.float32)
        perception._goal_embedding = np.array((1.0, 0.0), dtype=np.float32)
        perception._track_embedding = perception._goal_embedding.copy()
        perception._reference_hist = None
        perception._track_hist = None
        perception._missed_steps = 1
        perception.association_threshold = 0.20
        perception.reid_threshold = 0.55
        perception.ambiguity_margin = 0.04
        selected = perception._select(
            image,
            [(np.array((10, 5, 30, 35), dtype=np.float32), 0.99)],
        )
        self.assertIsNone(selected)

    def test_global_reacquisition_keeps_goal_reid_threshold(self):
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        perception = RGBPersonPerception.__new__(RGBPersonPerception)
        perception.reid_model = object()
        perception._embed_crops = lambda crops: [
            np.array((0.6, 0.8), dtype=np.float32) for _ in crops
        ]
        perception._bbox = None
        perception._goal_embedding = np.array((1.0, 0.0), dtype=np.float32)
        perception._track_embedding = perception._goal_embedding.copy()
        perception._reference_hist = None
        perception._track_hist = None
        perception._missed_steps = 8
        perception.global_identity_threshold = 0.45
        perception.reid_threshold = 0.55
        perception.ambiguity_margin = 0.04
        selected = perception._select(
            image,
            [(np.array((10, 5, 30, 35), dtype=np.float32), 0.99)],
        )
        self.assertIsNone(selected)

    def test_global_reacquisition_ranks_identity_above_detector_score(self):
        image = np.zeros((40, 80, 3), dtype=np.uint8)
        perception = RGBPersonPerception.__new__(RGBPersonPerception)
        perception.reid_model = object()
        perception._embed_crops = lambda crops: [
            np.array((0.63, 0.776595), dtype=np.float32),
            np.array((0.702, 0.712176), dtype=np.float32),
        ]
        perception._bbox = None
        perception._goal_embedding = np.array((1.0, 0.0), dtype=np.float32)
        perception._track_embedding = perception._goal_embedding.copy()
        perception._reference_hist = None
        perception._track_hist = None
        perception._missed_steps = 8
        perception.global_identity_threshold = 0.45
        perception.reid_threshold = 0.55
        perception.ambiguity_margin = 0.04
        selected = perception._select(
            image,
            [
                (np.array((0, 5, 30, 35), dtype=np.float32), 0.99),
                (np.array((45, 5, 75, 35), dtype=np.float32), 0.55),
            ],
        )
        np.testing.assert_array_equal(selected[0], (45, 5, 75, 35))

    def test_confirmed_track_accepts_continuous_single_candidate_view_change(self):
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        perception = RGBPersonPerception.__new__(RGBPersonPerception)
        perception.reid_model = object()
        perception._embed_crops = lambda crops: [
            np.array((0.616, 0.787746), dtype=np.float32) for _ in crops
        ]
        perception._bbox = np.array((10, 5, 30, 35), dtype=np.float32)
        perception._bbox_velocity = np.zeros(4, dtype=np.float32)
        perception._goal_embedding = np.array((1.0, 0.0), dtype=np.float32)
        perception._track_embedding = perception._goal_embedding.copy()
        perception._reference_hist = None
        perception._track_hist = None
        perception._missed_steps = 0
        perception._recent_goal_similarities = [0.851, 0.867]
        perception._confirmed_track_steps = 2
        perception.association_threshold = 0.20
        perception.reid_threshold = 0.55
        perception.ambiguity_margin = 0.04
        selected = perception._select(
            image,
            [(np.array((10, 5, 30, 35), dtype=np.float32), 0.99)],
        )
        self.assertIsNotNone(selected)
        np.testing.assert_array_equal(selected[0], (10, 5, 30, 35))

    def test_new_track_accepts_second_frame_view_change(self):
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        perception = RGBPersonPerception.__new__(RGBPersonPerception)
        perception.reid_model = object()
        perception._embed_crops = lambda crops: [
            np.array((0.664, 0.747733), dtype=np.float32) for _ in crops
        ]
        perception._bbox = np.array((10, 5, 30, 35), dtype=np.float32)
        perception._bbox_velocity = np.zeros(4, dtype=np.float32)
        perception._goal_embedding = np.array((1.0, 0.0), dtype=np.float32)
        perception._track_embedding = perception._goal_embedding.copy()
        perception._reference_hist = None
        perception._track_hist = None
        perception._missed_steps = 0
        perception._recent_goal_similarities = [0.876]
        perception._confirmed_track_steps = 1
        perception.association_threshold = 0.20
        perception.reid_threshold = 0.55
        perception.ambiguity_margin = 0.04
        selected = perception._select(
            image,
            [(np.array((10, 5, 30, 35), dtype=np.float32), 0.99)],
        )
        self.assertIsNotNone(selected)
        np.testing.assert_array_equal(selected[0], (10, 5, 30, 35))

    def test_tracker_accepts_identity_consistent_rapid_scale_growth(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        perception = RGBPersonPerception.__new__(RGBPersonPerception)
        perception.reid_model = object()
        perception._embed_crops = lambda crops: [
            np.array((0.76, 0.649923), dtype=np.float32) for _ in crops
        ]
        perception._bbox = np.array((40, 30, 60, 70), dtype=np.float32)
        perception._bbox_velocity = np.zeros(4, dtype=np.float32)
        perception._goal_embedding = np.array((1.0, 0.0), dtype=np.float32)
        perception._track_embedding = perception._goal_embedding.copy()
        perception._reference_hist = None
        perception._track_hist = None
        perception._missed_steps = 0
        perception._recent_goal_similarities = [1.0]
        perception._confirmed_track_steps = 1
        perception.association_threshold = 0.20
        perception.reid_threshold = 0.55
        perception.ambiguity_margin = 0.04
        selected = perception._select(
            image,
            [(np.array((25, 5, 75, 95), dtype=np.float32), 0.99)],
        )
        self.assertIsNotNone(selected)
        np.testing.assert_array_equal(selected[0], (25, 5, 75, 95))

    def test_confirmed_track_allows_motion_to_disambiguate_multiple_candidates(self):
        image = np.zeros((40, 80, 3), dtype=np.uint8)
        perception = RGBPersonPerception.__new__(RGBPersonPerception)
        perception.reid_model = object()
        perception._embed_crops = lambda crops: [
            np.array((0.658, 0.753018), dtype=np.float32),
            np.array((0.5, 0.866025), dtype=np.float32),
        ]
        perception._bbox = np.array((5, 5, 30, 35), dtype=np.float32)
        perception._bbox_velocity = np.zeros(4, dtype=np.float32)
        perception._goal_embedding = np.array((1.0, 0.0), dtype=np.float32)
        perception._track_embedding = perception._goal_embedding.copy()
        perception._reference_hist = None
        perception._track_hist = None
        perception._missed_steps = 0
        perception._recent_goal_similarities = [0.881, 0.904]
        perception._confirmed_track_steps = 4
        perception.association_threshold = 0.20
        perception.reid_threshold = 0.55
        perception.ambiguity_margin = 0.04
        selected = perception._select(
            image,
            [
                (np.array((5, 5, 30, 35), dtype=np.float32), 0.95),
                (np.array((50, 5, 75, 35), dtype=np.float32), 0.99),
            ],
        )
        np.testing.assert_array_equal(selected[0], (5, 5, 30, 35))

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

    def test_stop_search_policy_brakes_then_rotates_without_gt_pose(self):
        controller = OracleNavmeshFollower(
            lost_target_policy="stop-search",
            lost_brake_steps=2,
            lost_search_yaw=0.3,
            lost_search_period_steps=1,
            lost_coast_steps=0,
        )
        controller._previous_forward = 1.0
        controller._previous_lateral = -0.8
        controller._last_seen_bearing = -0.2
        lost = make_target(2.0, 0.0, visible=False)

        first = controller(None, None, None, lost)
        second = controller(None, None, None, lost)
        third = controller(None, None, None, lost)
        fourth = controller(None, None, None, lost)

        self.assertEqual(first.mode, "lost_brake")
        self.assertAlmostEqual(first.action.forward, 0.4)
        self.assertAlmostEqual(first.action.lateral, -0.2)
        self.assertEqual(second.action.as_habitat(), [0.0, 0.0, 0.0])
        self.assertEqual(third.mode, "lost_search")
        self.assertAlmostEqual(third.action.yaw, -0.3)
        self.assertAlmostEqual(fourth.action.yaw, 0.3)

    def test_stop_search_policy_coasts_only_for_distant_lost_target(self):
        controller = OracleNavmeshFollower(
            lost_target_policy="stop-search",
            lost_coast_steps=2,
            lost_coast_min_range_m=2.0,
            lost_coast_max_translation=0.35,
        )
        controller._previous_forward = 0.7
        controller._previous_lateral = -0.2
        controller._last_seen_bearing = 0.1

        decision = controller(None, None, None, make_target(3.0, 0.0, False))

        self.assertEqual(decision.mode, "lost_coast")
        self.assertAlmostEqual(decision.action.forward, 0.35)
        self.assertAlmostEqual(decision.action.lateral, -0.2)
        self.assertAlmostEqual(decision.action.yaw, 0.12)

    def test_stop_search_policy_preserves_close_range_retreat(self):
        controller = OracleNavmeshFollower(
            lost_target_policy="stop-search", lost_retreat_steps=3
        )
        controller._previous_forward = -1.0
        controller._previous_lateral = 0.6
        controller._last_seen_bearing = 0.0

        decision = controller(None, None, None, make_target(0.5, 0.0, False))

        self.assertEqual(decision.mode, "lost_retreat")
        self.assertAlmostEqual(decision.action.forward, -1.0)
        self.assertAlmostEqual(decision.action.lateral, 0.6)

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

    def test_invisible_target_inside_hold_band_stops_with_coordinates(self):
        controller = OracleNavmeshFollower(min_distance_m=1.2, max_distance_m=1.5)
        controller._path_waypoint = lambda *args: None
        controller._local_xy = lambda robot, point: (
            float(point[0] - robot.base_pos[0]),
            float(-(point[2] - robot.base_pos[2])),
        )
        robot = SimpleNamespace(base_pos=np.zeros(3, dtype=np.float32))
        target_agent = SimpleNamespace(
            base_pos=np.array([1.4, 0.0, 0.0], dtype=np.float32)
        )
        decision = controller(
            None, robot, target_agent, make_target(1.4, 0.0, visible=False)
        )
        self.assertEqual(decision.mode, "hold_startup_coordinate")
        self.assertEqual(decision.action.forward, 0.0)

    def test_coordinate_takeover_reverses_from_stale_far_visual_range(self):
        controller = OracleNavmeshFollower(min_distance_m=1.2, max_distance_m=1.5)
        controller._target_has_moved = True
        controller._update_incoming = lambda *args: (False, 0.0)
        controller._local_xy = lambda robot, point: (
            float(point[0] - robot.base_pos[0]),
            float(-(point[2] - robot.base_pos[2])),
        )
        robot = SimpleNamespace(base_pos=np.zeros(3, dtype=np.float32))
        target_agent = SimpleNamespace(
            base_pos=np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )

        decision = controller(
            None, robot, target_agent, make_target(3.0, 0.0, visible=False)
        )

        self.assertEqual(decision.mode, "track_distance_coordinate")
        self.assertLess(decision.action.forward, 0.0)

    def test_coordinate_takeover_approaches_using_current_far_range(self):
        controller = OracleNavmeshFollower(min_distance_m=1.2, max_distance_m=1.5)
        controller._path_waypoint = lambda *args: np.array(
            [3.0, 0.0, 0.0], dtype=np.float32
        )
        controller._local_xy = lambda robot, point: (
            float(point[0] - robot.base_pos[0]),
            float(-(point[2] - robot.base_pos[2])),
        )
        robot = SimpleNamespace(base_pos=np.zeros(3, dtype=np.float32))
        target_agent = SimpleNamespace(
            base_pos=np.array([3.0, 0.0, 0.0], dtype=np.float32)
        )

        decision = controller(
            None, robot, target_agent, make_target(0.8, 0.0, visible=False)
        )

        self.assertEqual(decision.mode, "approach_coordinate")
        self.assertGreater(decision.action.forward, 0.0)

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
        self.assertEqual(controller.lost_target_policy, "coordinate")

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

    def test_v6_retreats_from_close_invisible_target_using_coordinates(self):
        controller = OracleNavmeshFollowerV6(
            min_distance_m=1.2, max_distance_m=1.5
        )
        controller._update_incoming = lambda *args: (True, 0.1)
        controller._retreat_goal = lambda *args: np.array(
            [-1.0, 0.0, 0.0], dtype=np.float32
        )
        controller._path_waypoint = lambda sim, robot, goal: goal
        controller._local_xy = lambda robot, point: (
            float(point[0] - robot.base_pos[0]),
            float(-(point[2] - robot.base_pos[2])),
        )
        robot = SimpleNamespace(base_pos=np.zeros(3, dtype=np.float32))
        target_agent = SimpleNamespace(
            base_pos=np.array([0.8, 0.0, 0.0], dtype=np.float32)
        )

        decision = controller(
            object(), robot, target_agent, make_target(0.8, 0.0, visible=False)
        )

        self.assertEqual(decision.mode, "retreat_safety_coordinate")
        self.assertLess(decision.action.forward, 0.0)


if __name__ == "__main__":
    unittest.main()
