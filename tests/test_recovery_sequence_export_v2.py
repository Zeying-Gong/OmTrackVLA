import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from recovery_sequence_export_v2 import (
    replay_policy_calls, resample_expert_positions, write_candidate_sequence,
)


def expert_states(anchor=7.):
    # Deliberately irregular time intervals: no fixed-frequency assumption works.
    elapsed = [0., .063, .219, .487, .803]
    return [{
        "environment_step": 2 + index,
        "world_time_s": anchor + value,
        "robot_world_xyz_m": [1. + 2. * value, 0., 0.],
        "target_world_xyz_m": [4. + value, 0., 0.],
    } for index, value in enumerate(elapsed)]


def resample(states):
    return resample_expert_positions(
        states, anchor_world_time_s=7., anchor_position=[1., 0., 0.],
        forward_axis=[1., 0., 0.], left_axis=[0., 0., -1.],
    )


class RecoverySequenceExportTest(unittest.TestCase):
    def test_prefix_history_contains_only_available_frames(self):
        prefix = [{"environment_step": index, "world_time_s": value}
                  for index, value in enumerate([6.7, 6.82, 7.])]
        calls = replay_policy_calls(prefix, anchor_step=2, history_size=4)
        self.assertEqual([call["rgb_history_observation_indices"] for call in calls],
                         [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 2]])
        self.assertEqual([call["supervision_valid"] for call in calls], [False, False, True])
        with self.assertRaisesRegex(ValueError, "future frames"):
            replay_policy_calls(prefix + [{"environment_step": 3, "world_time_s": 8.}],
                                anchor_step=2, history_size=4)

    def test_irregular_observed_time_maps_to_physical_grid(self):
        result = resample(expert_states())
        np.testing.assert_allclose(result["waypoint_times_s"], np.arange(8) / 10.)
        np.testing.assert_allclose(np.asarray(result["waypoints_base_xy_m"])[:, 0], np.arange(8) * .2)
        self.assertFalse(result["fixed_frequency_assumed"])
        self.assertFalse(result["extrapolation_used"])

    def test_insufficient_or_nonmonotonic_time_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not cover"):
            resample(expert_states()[:-1])
        states = expert_states()
        states[2]["world_time_s"] = states[1]["world_time_s"]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            resample(states)

    def test_anchor_time_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "anchor world time"):
            resample(expert_states(anchor=7.02))

    def test_candidate_never_inherits_legacy_formal_status(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "rollout.json"
            source.write_text("{}", encoding="utf-8")
            prefix = [{"environment_step": index, "world_time_s": value}
                      for index, value in enumerate([6.7, 6.82, 7.])]
            frames = [(index, np.full((2, 3, 3), index, dtype=np.uint8)) for index in range(3)]
            path, _ = write_candidate_sequence(
                directory=root / "candidate", source_rollout=source, report_output=root / "report.json",
                result={"split": "train", "task": "stt", "episode_id": "0", "dataset_index": 0, "scene_id": "scene"},
                initial_rgb=frames[0][1], prefix_frames=frames, prefix_states=prefix,
                initial_bbox_xyxy_norm=[.1, .1, .9, .9], camera_intrinsics=np.eye(3), camera_from_base=np.eye(4),
                expert_states=expert_states(), expert_actions=[{
                    "expert_step": index,
                    "world_time_s_before": before["world_time_s"],
                    "world_time_s_after": after["world_time_s"],
                    "action": {"forward": .1, "lateral": 0., "yaw": 0.},
                } for index, (before, after) in enumerate(zip(expert_states()[:-1], expert_states()[1:]))],
                replay_actions=[[.1, 0., 0.]] * 2,
                anchor_position=[1., 0., 0.], forward_axis=[1., 0., 0.], left_axis=[0., 0., -1.],
                checkpoint_step=2, history_size=4, legacy_quality_passed=True,
            )
            sample = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(sample["schema_version"], 2)
            self.assertEqual(sample["stage"], "phase3_recovery_sequence_candidate_v2")
            self.assertFalse(sample["formal_training_eligible"])
            self.assertTrue(sample["audit_raw"]["legacy_step_stride_quality_passed"])
            self.assertFalse(sample["sequence_contract"]["expert_future_rgb_in_model_inputs"])
            self.assertEqual(sample["supervision"]["supervision_mask"], [False, False, True])
            self.assertEqual(len(list(path.parent.glob("prefix_*.png"))), 3)
            self.assertTrue(sample["source"]["evt_bench_used_for_waypoint_supervision"])
            self.assertNotIn("robot_world_xyz_m", sample["model_inputs"])


if __name__ == "__main__":
    unittest.main()
