import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from omtrackvla.evaluation.end_to_end_closed_loop import (
    ArchitectureV1PolicyWorker,
    WaypointActionAdapter,
    decision_api_audit,
    deployment_action,
    habitat_camera_calibration,
    normalized_bbox,
    atomic_json,
    partial_rollout_result,
    polar_reactive_physical_action,
    pose_derived_simulated_uwb,
)


class WaypointActionAdapterTest(unittest.TestCase):
    def test_uses_first_waypoint_reaching_lookahead(self):
        adapter = WaypointActionAdapter(
            lookahead_m=0.15,
            translation_gain=2.0,
            translation_slew=1.0,
            yaw_slew=1.0,
        )
        waypoints = np.asarray(
            [[0.0, 0.0], [0.05, 0.0], [0.12, 0.01], [0.18, 0.02]],
            dtype=np.float32,
        )
        action, mode, selected = adapter(waypoints, stop_probability=0.1)
        self.assertEqual(mode, "waypoint_follow")
        self.assertEqual(selected, 3)
        self.assertAlmostEqual(action.forward, 0.36, places=5)
        self.assertGreater(action.lateral, 0.0)

    def test_stop_logit_wins(self):
        adapter = WaypointActionAdapter()
        action, mode, selected = adapter(
            np.asarray([[0.0, 0.0], [1.0, 0.2]], dtype=np.float32),
            stop_probability=0.9,
        )
        self.assertEqual(mode, "policy_stop")
        self.assertEqual(selected, 0)
        self.assertEqual(action.as_habitat(), [0.0, 0.0, 0.0])

    def test_non_finite_prediction_fails_safe(self):
        adapter = WaypointActionAdapter()
        action, mode, _ = adapter(
            np.asarray([[0.0, 0.0], [np.nan, 0.0]], dtype=np.float32),
            stop_probability=0.1,
        )
        self.assertEqual(mode, "invalid_prediction_stop")
        self.assertEqual(action.as_habitat(), [0.0, 0.0, 0.0])

    def test_low_visual_confidence_without_uwb_stops(self):
        adapter = WaypointActionAdapter()
        action, mode, selected = deployment_action(
            adapter,
            np.asarray([[0.0, 0.0], [1.0, 0.2]], dtype=np.float32),
            stop_probability=0.0,
            visibility_probability=0.89,
            uwb_valid=False,
        )
        self.assertEqual(mode, "visual_uncertain_stop")
        self.assertEqual(selected, 0)
        self.assertEqual(action.as_habitat(), [0.0, 0.0, 0.0])

    def test_valid_uwb_can_continue_when_visual_confidence_is_low(self):
        adapter = WaypointActionAdapter(translation_slew=1.0, yaw_slew=1.0)
        action, mode, _ = deployment_action(
            adapter,
            np.asarray([[0.0, 0.0], [0.2, 0.0]], dtype=np.float32),
            stop_probability=0.0,
            visibility_probability=0.1,
            uwb_valid=True,
        )
        self.assertEqual(mode, "waypoint_follow")
        self.assertGreater(action.forward, 0.0)

    def test_visual_uncertainty_wins_over_large_bbox_without_uwb(self):
        adapter = WaypointActionAdapter(translation_slew=1.0, yaw_slew=1.0)
        action, mode, selected = deployment_action(
            adapter,
            np.asarray([[0.0, 0.0], [0.2, 0.1]], dtype=np.float32),
            stop_probability=0.0,
            visibility_probability=0.5,
            predicted_bbox_xyxy_norm=(0.3, 0.02, 0.7, 0.90),
            uwb_valid=False,
        )
        self.assertEqual(mode, "visual_uncertain_stop")
        self.assertEqual(selected, 0)
        self.assertEqual(action.as_habitat(), [0.0, 0.0, 0.0])

    def test_large_visual_bbox_holds_forward_but_keeps_steering(self):
        adapter = WaypointActionAdapter(translation_slew=1.0, yaw_slew=1.0)
        action, mode, selected = deployment_action(
            adapter,
            np.asarray([[0.0, 0.0], [0.2, 0.1]], dtype=np.float32),
            stop_probability=0.0,
            visibility_probability=0.99,
            predicted_bbox_xyxy_norm=(0.3, 0.02, 0.7, 0.90),
            uwb_valid=False,
        )
        self.assertEqual(mode, "visual_proximity_hold")
        self.assertEqual(selected, 1)
        self.assertEqual(action.forward, 0.0)
        self.assertGreater(action.lateral, 0.0)
        self.assertGreater(action.yaw, 0.0)


class PolarReactiveActionTest(unittest.TestCase):
    def test_approaches_a_distant_centered_target(self):
        action, mode = polar_reactive_physical_action(0.0, 2.2)
        self.assertEqual(mode, "polar_reactive_approach")
        np.testing.assert_allclose(action, [0.3, 0.0, 0.0], atol=1.0e-6)

    def test_turns_and_sidesteps_before_driving_forward(self):
        action, mode = polar_reactive_physical_action(np.pi / 2.0, 2.2)
        self.assertEqual(mode, "polar_reactive_approach")
        self.assertEqual(float(action[0]), 0.0)
        self.assertGreater(float(action[1]), 0.0)
        self.assertGreater(float(action[2]), 0.0)

    def test_retreats_only_inside_the_standoff_band(self):
        close, close_mode = polar_reactive_physical_action(0.0, 0.8)
        band, band_mode = polar_reactive_physical_action(0.0, 1.8)
        self.assertEqual(close_mode, "polar_reactive_retreat")
        self.assertLess(float(close[0]), 0.0)
        self.assertEqual(band_mode, "polar_reactive_hold")
        self.assertEqual(float(band[0]), 0.0)

    def test_rejects_nonfinite_polar_predictions(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            polar_reactive_physical_action(float("nan"), 1.5)


class ClosedLoopInputContractTest(unittest.TestCase):
    def test_pose_derived_uwb_is_sensor_shaped_and_base_relative(self):
        class Transform:
            def inverted(self):
                return self

            def transform_vector(self, value):
                np.testing.assert_allclose(value, [1.0, 0.0, -2.0])
                return SimpleNamespace(x=1.25, z=-0.4)

        robot = SimpleNamespace(
            base_pos=np.asarray([2.0, 0.0, 3.0]),
            sim_obj=SimpleNamespace(transformation=Transform()),
        )
        target = SimpleNamespace(base_pos=np.asarray([3.0, 0.0, 1.0]))
        sample = pose_derived_simulated_uwb(robot, target)
        self.assertEqual(sample.xy_m, (1.25, 0.4))
        self.assertEqual(sample.covariance_m2, ((0.0, 0.0), (0.0, 0.0)))
        self.assertEqual(sample.quality, 1.0)
        self.assertEqual(sample.age_s, 0.0)
        self.assertTrue(sample.valid)

    def test_decision_api_contains_only_rgb_and_optional_uwb(self):
        class Controller:
            def decide(self, rgb, uwb=None):
                del rgb, uwb

        report = decision_api_audit(Controller())
        self.assertTrue(report["passed"])
        self.assertFalse(report["gt_target_point_used"])
        self.assertFalse(report["later_bbox_used"])
        self.assertFalse(report["depth_used"])

        worker_parameters = list(
            __import__("inspect").signature(
                ArchitectureV1PolicyWorker.decide
            ).parameters
        )
        self.assertEqual(worker_parameters, ["self", "rgb", "uwb"])

    def test_privileged_input_is_rejected(self):
        class BadController:
            def decide(self, rgb, target_point=None):
                del rgb, target_point

        with self.assertRaisesRegex(ValueError, "privileged"):
            decision_api_audit(BadController())

    def test_bbox_and_camera_calibration(self):
        bbox = normalized_bbox((96, 48, 288, 336), (384, 384, 3))
        self.assertEqual(bbox, (0.25, 0.125, 0.75, 0.875))
        intrinsics, camera_from_base = habitat_camera_calibration(
            (384, 384, 3), 280, 504, 90.0
        )
        self.assertEqual(intrinsics.shape, (3, 3))
        self.assertEqual(camera_from_base.shape, (4, 4))
        self.assertAlmostEqual(float(intrinsics[0, 0]), 252.0, places=4)
        self.assertAlmostEqual(float(intrinsics[1, 1]), 140.0, places=4)
        np.testing.assert_allclose(
            camera_from_base[:3, :3],
            [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
        )


class CrashRecoverableResultTest(unittest.TestCase):
    def test_partial_result_keeps_a_contiguous_replay_prefix(self):
        base = {"split": "train", "dataset_index": 400}
        records = [{"step": 1}, {"step": 2}]
        result = partial_rollout_result(base, records)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["completed_steps"], 2)
        self.assertEqual(result["steps"], records)

    def test_partial_result_rejects_a_gap(self):
        with self.assertRaisesRegex(ValueError, "contiguous"):
            partial_rollout_result({}, [{"step": 1}, {"step": 3}])

    def test_atomic_json_replaces_previous_snapshot(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "result.partial.json"
            atomic_json(path, {"completed_steps": 1})
            atomic_json(path, {"completed_steps": 2})
            self.assertEqual(
                __import__("json").loads(path.read_text(encoding="utf-8")),
                {"completed_steps": 2},
            )
            self.assertEqual(list(path.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
