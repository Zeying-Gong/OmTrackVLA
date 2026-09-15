import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from omtrackvla.data.recovery_sequence import (
    RecoveryValidationError, RecoverySequenceDataset, load_recovery_sequence,
    recovery_sequence_collate, validate_recovery_sample,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def make_fixture(root, anchor=2):
    """A straight, collision-free branch with deliberately irregular times."""
    folder = root / "candidate"
    folder.mkdir(parents=True)
    rollout_path, report_path, path = root/"rollout.json", root/"report.json", folder/"sample.json"
    initial = np.zeros((3, 4, 3), dtype=np.uint8)
    Image.fromarray(initial).save(folder/"initial_rgb.png")
    prefix = []
    def state(step, time, x=0.):
        transform = np.eye(4)
        transform[0, 3] = x
        return {"environment_step": step, "world_time_s": time, "robot_world_xyz_m": [x, 0., 0.],
                "target_world_xyz_m": [3., 0., 0.], "robot_transform_world": transform.tolist()}
    prefix_states = [state(i, i*.05) for i in range(anchor+1)]
    for i in range(anchor+1):
        image_path = folder/f"prefix_{i:04d}.png"
        Image.fromarray(initial+i).save(image_path)
        prefix.append({"environment_step": i, "world_time_s": i*.05, "rgb_path": image_path.name, "sha256": digest(image_path)})
    source = {"dataset": "EVT-Bench", "split": "train", "task": "stt", "dataset_index": 0, "episode_id": "4",
              "scene_id": "data/scene_datasets/hm3d/train/00083-16tymPtM7uS/16tymPtM7uS.basis.glb",
              "anchor_environment_step": anchor, "test_locked_used": False,
              "evt_bench_used_for_waypoint_supervision": True, "expert": "OracleNavmeshFollowerV6",
              "rollout_result": str(rollout_path), "relabel_report": str(report_path),
              "original_sample_reference": {"status": "not_supplied", "path": None, "sha256": None}}
    identity = {k: source[k] for k in ("split", "task", "dataset_index", "episode_id", "scene_id")}
    input_audit = {"decision_parameters": ["rgb", "uwb"], "depth_used": False, "gt_target_point_used": False,
                   "later_bbox_used": False, "perception_cache_used": False, "passed": True,
                   "model_input_keys": ["initial_rgb", "initial_bbox", "ego_rgb", "visual_initialization_valid", "rgb_valid", "binding_valid",
                       "uwb_xy", "uwb_covariance_xy", "uwb_quality", "uwb_age_s", "uwb_valid", "camera_intrinsics", "camera_from_base", "hidden_state", "_target_memory_override"]}
    rollout = {**identity, "initialization": {"environment_step": 0, "bbox_xyxy": [1., 0., 3., 3.]}, "input_audit": input_audit,
               "steps": [{"step": i+1, "policy": {"action": {"forward": .1, "lateral": 0., "yaw": 0.}},
                          "evaluation_only_after_action": {"gt_distance_m": 3.}} for i in range(anchor)]}
    write(rollout_path, rollout)
    source["rollout_result_sha256"] = digest(rollout_path)
    elapsed = [0., .06, .2, .32, .48, .6, .72, .84]
    expert_states = [state(anchor+i, anchor*.05+t, t*.5) for i, t in enumerate(elapsed)]
    records = []
    for i in range(7):
        before, after = expert_states[i:i+2]
        record = {"expert_step": i, "action": {"forward": .1, "lateral": 0., "yaw": 0.},
                  "robot_anchor_xy_m_before": [elapsed[i]*.5, 0.], "target_anchor_xy_m_before": [3., 0.]}
        for suffix, value in (("before", before), ("after", after)):
            for field in ("world_time_s", "robot_world_xyz_m", "target_world_xyz_m"):
                record[f"{field}_{suffix}"] = value[field]
        records.append(record)
    trajectory = {"waypoint_times_s": [i/10 for i in range(8)], "waypoints_base_xy_m": [[i/20, 0.] for i in range(8)],
                  "resampled_robot_world_xyz_m": [[i/20, 0., 0.] for i in range(8)], "resampled_target_world_xyz_m": [[3., 0., 0.]]*8,
                  "raw_observed_elapsed_s": elapsed, "observed_horizon_s": .84, "valid_mask": [True]*8,
                  "resampling": "piecewise_linear_world_xyz_at_recorded_world_time", "fixed_frequency_assumed": False, "extrapolation_used": False}
    sample = {"schema_version": 2, "stage": "phase3_recovery_sequence_candidate_v2", "status": "candidate_pending_independent_admission",
              "sample_id": f"evt_train/stt/4/post-action-{anchor:03d}/sequence-v2", "formal_training_eligible": False, "test_locked_used": False,
              "source": source,
              "sequence_contract": {"history_size": 4, "starts_at_episode_reset": True, "expert_future_rgb_in_model_inputs": False,
                  "fixed_frequency_assumed": False, "policy_call_indices": list(range(anchor+1)), "anchor_policy_call_index": anchor,
                  "startup_padding": "repeat_reset_rgb_on_left", "state_reconstruction": "replay_every_policy_call_with_current_training_weights",
                  "policy_interval_source": "observed_simulator_world_time", "world_time_reader": "env.sim.get_world_time",
                  "initial_state": {"hidden": "zero", "target_memory": "initialize_from_initial_rgb_bbox", "binding_valid": True}},
              "model_inputs": {"condition_mode": "visual_only", "initial_rgb": {"rgb_path": "initial_rgb.png", "sha256": digest(folder/"initial_rgb.png")},
                  "initial_bbox_xyxy_norm": [.25, 0., .75, 1.], "prefix_rgb": prefix,
                  "policy_calls": [{"policy_call_index": i, "observation_environment_step": i, "world_time_s": i*.05,
                      "rgb_history_observation_indices": [max(0, j) for j in range(i-3, i+1)], "supervision_valid": i == anchor,
                      "action_provenance": "expert_relabel_anchor" if i == anchor else "saved_policy_replay"} for i in range(anchor+1)],
                  "visual_initialization_valid": True, "rgb_valid": True, "binding_valid": True,
                  "uwb": {"valid": False, "relative_position_base_xy_m": [0., 0.], "covariance_base_xy_m2": [[1., 0.], [0., 1.]], "quality_01": 0., "age_s": 0.},
                  "camera_intrinsics": [[2., 0., 2.], [0., 1.5, 1.5], [0., 0., 1.]], "camera_from_base": np.eye(4).tolist()},
              "supervision": {"anchor_policy_call_index": anchor, "supervision_mask": [False]*anchor+[True], "expert_trajectory": trajectory,
                  "gt_used_only_on_label_or_audit_side": True, "later_bbox_used_by_model": False, "stop_label_available": False,
                  "visibility_label_available": False, "binding_label_available": False, "ego_motion_label_available": False},
              "audit_raw": {"prefix_states": prefix_states, "expert_states": expert_states, "expert_actions": records,
                            "saved_policy_replay_actions": [[.1, 0., 0.]]*anchor, "legacy_step_stride_quality_passed": True, "legacy_passed_is_not_formal_admission": True}}
    report = {"schema_version": 2, "stage": "next007_recovery_sequence_probe_v2", "status": "passed", "formal_training_eligible": False,
              **identity, "source_rollout": str(rollout_path), "model_visited_checkpoint_step": anchor,
              "expert_branch_steps": 7, "total_environment_steps": anchor+7, "maximum_safe_total_steps": 50,
              "phase3_recovery_sequence_candidate": {"manifest": str(path), "prefix_environment_steps": list(range(anchor+1)), "formal_training_eligible": False},
              "label_boundary": {"gt_pose_used_only_in_audit_and_supervision": True, "replay_actions_never_use_audit_pose": True,
                  "navmesh_used_only_by_expert": True, "expert_waypoint_entered_deployment_policy": False, "test_locked_used": False,
                  "future_target_trajectory_used_only_by_label_expert": False, "source_policy_input_audit": input_audit},
              "actual_time_audit": {"prefix_states": prefix_states, "expert_states": expert_states, "world_time_reader": "env.sim.get_world_time",
                  "evt_bench_train_used_for_waypoint_supervision": True, "legacy_passed_is_not_formal_admission": True, "fixed_frequency_assumed": False},
              "replay_audit": {"saved_policy_actions_replayed_without_gt": True, "matches_saved_model_visited_state": True,
                  "tolerance": 1e-4, "camera_transform_max_abs_error": 0., "target_distance_abs_error_m": 0.},
              "coordinate_audit": {"passed": True, "tolerance": 1e-4, "anchor_target_xy_m": [3., 0.], "habitat_local_target_xy_m": [3., 0.], "max_abs_error_m": 0.},
              "expert": {"controller": "OracleNavmeshFollowerV6", "waypoint_frame": "checkpoint robot base x-forward/y-left",
                  "records": records, "actuation_limits": {"max_forward": .1, "max_lateral": .1, "max_yaw": .75, "translation_slew_per_step": .1},
                  "target_forecast": {"lookahead_steps": 0, "deterministic_under_expert_actions": True, "same_time_target_max_error_m": 0.},
                  "waypoint_offsets_steps": list(range(8)), "legacy_step_stride_waypoints_are_not_v2_training_targets": True,
                  "waypoints_base_xy_m": [[t*.5, 0.] for t in elapsed], "legacy_step_stride_offsets_seconds_observed": elapsed,
                  "initial_target_anchor_xy_m": [3., 0.], "terminal_target_anchor_xy_m": [3., 0.],
                  "trajectory_statistics": {"path_length_m": .42, "terminal_radius_m": .42, "maximum_segment_m": .08},
                  "target_progress_statistics": {"stationary_anchor_final_target_distance_m": 3., "expert_final_target_distance_m": 2.58,
                      "distance_improvement_over_stationary_anchor_m": .42, "terminal_motion_target_alignment_cosine": 1.},
                  "trajectory_gate": {"passed": True, "minimum_path_m": .05, "maximum_path_m": 2.2, "maximum_terminal_radius_m": 1.8, "minimum_stationary_distance_improvement_m": .05},
                  "collision": 0.}}
    def save():
        write(path, sample)
        report["phase3_recovery_sequence_candidate"]["manifest_sha256"] = digest(path)
        write(report_path, report)
    save()
    return path, sample, report, save


class RecoveryValidationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path, self.sample, self.report, self.save = make_fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def validate(self):
        self.save()
        return validate_recovery_sample(self.path)

    def reject(self, phrase):
        with self.assertRaisesRegex(RecoveryValidationError, phrase):
            self.validate()

    def test_irregular_time_candidate_passes_without_mutating_eligibility(self):
        before = self.path.read_bytes()
        evidence = validate_recovery_sample(self.path)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(evidence["formal_training_eligible"])
        self.assertEqual(evidence["sequence_length"], 3)
        self.assertTrue(all(evidence["checks"].values()))
        self.assertEqual(len(evidence["artifacts"]["media"]), 4)

    def test_missing_prefix_frame(self):
        self.sample["model_inputs"]["prefix_rgb"].pop(1)
        self.reject("RGB prefix")

    def test_missing_rgb_file(self):
        (self.path.parent/"prefix_0001.png").unlink()
        self.reject("invalid recovery")

    def test_bad_hash(self):
        self.sample["model_inputs"]["prefix_rgb"][1]["sha256"] = "0"*64
        self.reject("SHA-256")

    def test_path_escape(self):
        Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(self.root/"outside.png")
        self.sample["model_inputs"]["prefix_rgb"][1]["rgb_path"] = "../outside.png"
        self.reject("escapes")

    def test_future_frame_in_policy_history(self):
        self.sample["model_inputs"]["policy_calls"][0]["rgb_history_observation_indices"][-1] = 1
        self.reject("future frames")

    def test_future_rgb_appended(self):
        self.sample["model_inputs"]["prefix_rgb"].append(copy.deepcopy(self.sample["model_inputs"]["prefix_rgb"][-1]))
        self.reject("RGB prefix")

    def test_nonmonotonic_prefix_time(self):
        self.sample["audit_raw"]["prefix_states"][1]["world_time_s"] = 0.
        self.reject("strictly increasing")

    def test_rgb_time_mismatch(self):
        self.sample["model_inputs"]["prefix_rgb"][1]["world_time_s"] += .01
        self.reject("RGB timestamps")

    def test_nonmonotonic_expert_time(self):
        self.sample["audit_raw"]["expert_states"][2]["world_time_s"] = self.sample["audit_raw"]["expert_states"][1]["world_time_s"]
        self.reject("strictly increasing")

    def test_fabricated_time_grid(self):
        self.sample["supervision"]["expert_trajectory"]["waypoint_times_s"][1] = .048
        self.reject("waypoint_times_s")

    def test_incorrect_interpolation(self):
        self.sample["supervision"]["expert_trajectory"]["waypoints_base_xy_m"][1][0] += .01
        self.reject("waypoints_base_xy_m")

    def test_claimed_no_extrapolation_does_not_override_coverage(self):
        for record in self.sample["audit_raw"]["expert_states"][1:]:
            record["world_time_s"] = .1 + (record["world_time_s"]-.1)*.5
        self.reject("without extrapolation")

    def test_gt_field_rejected_from_model_inputs(self):
        self.sample["model_inputs"]["robot_world_xyz_m"] = [1, 2, 3]
        self.reject("unapproved")

    def test_auxiliary_labels_are_not_inferred(self):
        self.sample["supervision"]["stop_label_available"] = True
        self.reject("stop_label_available")

    def test_prefix_supervision_rejected(self):
        self.sample["supervision"]["supervision_mask"][0] = True
        self.reject("anchor only")

    def test_train_only(self):
        self.sample["source"]["split"] = "val"
        self.reject("EVT-Bench train")

    def test_test_locked_rejected_before_artifact_read(self):
        self.sample["source"]["rollout_result"] = str(self.root/"test_locked"/"nonexistent.json")
        self.reject("test_locked")

    def test_report_failed(self):
        self.report["status"] = "failed"
        self.reject("status")

    def test_report_boolean_cannot_hide_replay_failure(self):
        self.report["replay_audit"]["camera_transform_max_abs_error"] = .001
        self.reject("replay .*failed")

    def test_report_boolean_cannot_hide_coordinate_failure(self):
        self.report["coordinate_audit"]["habitat_local_target_xy_m"][0] += .01
        self.reject("coordinate check failed")

    def test_report_legacy_statistics_recomputed(self):
        self.report["expert"]["trajectory_statistics"]["path_length_m"] = .43
        self.reject("legacy path_length_m")

    def test_report_collision_rejected(self):
        self.report["expert"]["collision"] = 1.
        self.reject("collision audit failed")

    def link_legacy(self, task="stt"):
        source = self.sample["source"]
        legacy = {"schema_version": 1, "sample_id": f"habitat/{task}/4/post-action-002",
                  "source": {key: source[key] for key in ("split", "scene_id", "episode_id", "dataset_index", "anchor_environment_step", "test_locked_used")}}
        legacy_path = self.root/"legacy.json"
        write(legacy_path, legacy)
        source["original_sample_reference"] = {"path": str(legacy_path), "sha256": digest(legacy_path), "status": "linked_only_not_admission"}

    def test_legacy_v1_task_identity_is_in_complete_sample_id(self):
        self.link_legacy()
        evidence = self.validate()
        self.assertIn("original_sample", evidence["artifacts"])
        self.assertFalse(evidence["formal_training_eligible"])

    def test_legacy_v1_wrong_task_is_rejected(self):
        self.link_legacy("dtst")
        self.reject("sample_id/task mismatch")

    def test_legacy_pass_does_not_admit_stationary_actual_time_prefix(self):
        positions = [0., 0., 0., 0., 0., 0., .04, .2]
        for state, x in zip(self.sample["audit_raw"]["expert_states"], positions):
            state["robot_world_xyz_m"][0] = x
            state["robot_transform_world"][0][3] = x
        for record, x in zip(self.sample["audit_raw"]["expert_actions"], positions):
            record["robot_anchor_xy_m_before"][0] = x
        trajectory = self.sample["supervision"]["expert_trajectory"]
        resampled = [0.]*7+[1/30]
        trajectory["waypoints_base_xy_m"] = [[x, 0.] for x in resampled]
        trajectory["resampled_robot_world_xyz_m"] = [[x, 0., 0.] for x in resampled]
        expert = self.report["expert"]
        expert["waypoints_base_xy_m"] = [[x, 0.] for x in positions]
        expert["trajectory_statistics"] = {"path_length_m": .2, "terminal_radius_m": .2, "maximum_segment_m": .16}
        expert["target_progress_statistics"].update({"expert_final_target_distance_m": 2.8, "distance_improvement_over_stationary_anchor_m": .2})
        self.reject("actual-time 0..0.7 s trajectory quality failed")

    def test_no_implicit_candidate_load(self):
        with self.assertRaisesRegex(RecoveryValidationError, "not formally admitted"):
            load_recovery_sequence(self.path)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is needed only by the tensor loader")
    def test_tensor_history_anchor_masks_and_input_boundary(self):
        import torch
        sample = load_recovery_sequence(self.path, image_height=3, image_width=4, allow_candidate_for_preflight=True)
        self.assertEqual(tuple(sample["ego_rgb"].shape), (3, 4, 3, 3, 4))
        self.assertEqual(sample["ego_rgb"].dtype, torch.float32)
        torch.testing.assert_close(sample["ego_rgb"][2, -1], torch.full((3, 3, 4), 2/255))
        self.assertEqual(sample["supervision_mask"].tolist(), [False, False, True])
        self.assertFalse(sample["waypoint_mask"][:-1].any())
        self.assertTrue(sample["waypoint_mask"][-1].all())
        self.assertFalse(sample["target_waypoints"][:-1].any())
        self.assertNotIn("audit_raw", sample)
        self.assertNotIn("stop_target", sample)
        self.assertFalse(sample["stop_label_valid"].any())
        batch = recovery_sequence_collate([sample, sample])
        self.assertEqual(tuple(batch["ego_rgb"].shape), (2, 3, 4, 3, 3, 4))
        altered = dict(sample, ego_rgb=sample["ego_rgb"][:-1])
        with self.assertRaisesRegex(RecoveryValidationError, "padding"):
            recovery_sequence_collate([sample, altered])

    def test_dataset_requires_explicit_candidate_scope_and_anchor(self):
        with self.assertRaisesRegex(RecoveryValidationError, "explicit"):
            RecoverySequenceDataset([self.path])
        with self.assertRaisesRegex(RecoveryValidationError, "required_anchor"):
            RecoverySequenceDataset([self.path], allow_candidate_for_preflight=True)
        data = RecoverySequenceDataset([self.path], required_anchor=2, allow_candidate_for_preflight=True)
        self.assertEqual(len(data), 1)


if __name__ == "__main__":
    unittest.main()
