"""Pure CPU tests with the real TeacherRecorder interpolation implementation."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from wa.wm.failure_state_recorder import (
    EXPERIMENT, PARTITION, RGB_KEY, FailureStateRecorder, canonical_sha256)
from wa.wm.recovery_replay import ReplayThenTeacher, dynamic_state, rgb_hash


class Transform:
    def __init__(self):
        self.matrix = np.eye(4)
    def __array__(self, dtype=None):
        return np.asarray(self.matrix, dtype=dtype)
    def rotation(self):
        return self.matrix[:3, :3]


class Teacher:
    def __init__(self):
        self.reply_error = None
        self.episode_fallback_count = 0
        self.last_trajectory = None  # Oracle is allowed not to predict waypoints.
        self.calls = 0
        self.environment = None
    def bind_environment(self, env):
        self.environment = env
    def act(self, *args, **kwargs):
        self.calls += 1
        return [.2, 0., 0.]
    def reset(self, *args, **kwargs):
        self.calls = 0


class Fixture:
    def __init__(self):
        self.transform = Transform()
        self.obj = SimpleNamespace(transformation=self.transform, joint_positions=np.zeros(2))
        self.robot = SimpleNamespace(sim_obj=self.obj, base_pos=np.zeros(3))
        self.human = SimpleNamespace(base_pos=np.ones(3))
        self.time = 0.
        self.env = SimpleNamespace(sim=SimpleNamespace(
            agents_mgr=[SimpleNamespace(articulated_agent=self.robot)],
            get_world_time=lambda: self.time))
        self.episode = SimpleNamespace(info={"main_human_semantic_id": 1})
    def set_step(self, step, dt=.13):
        self.time = 10. + step * dt
        self.robot.base_pos = np.asarray([step * .03, 0., 0.])
        self.transform.matrix[:3, 3] = self.robot.base_pos
        rgb = np.full((4, 5, 4), step % 255, dtype=np.uint8)
        return {RGB_KEY: rgb,
                "agent_1_articulated_agent_jaw_panoptic": np.ones((4, 5), dtype=np.int32)}
    @staticmethod
    def detector(zero=False):
        return {"agent_1_main_humanoid_detector_sensor":
                {"box": np.asarray([0, 0, 0, 0] if zero else [0, 0, 5, 4], dtype=float)}}


class FailureStateRecorderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.next_root = 0

    def root(self):
        self.next_root += 1
        return Path(self.tmp.name) / str(self.next_root)

    def metadata(self, teacher="student", **kwargs):
        result = dict(experiment=EXPERIMENT, partition=PARTITION, teacher=teacher,
                      task="stt", key="scene/1", seed=7, protocol_sha256="b"*64,
                      camera_alignment_verified=True)
        result.update(kwargs)
        return result

    def prefix(self, count=12):
        fixture = Fixture()
        rows = []
        for step in range(count):
            obs = fixture.set_step(step)
            rows.append(dict(step=step, rgb_sha256=rgb_hash(obs[RGB_KEY]),
                             dynamic_state=dynamic_state(fixture.env), action=[-.2, 0., 0.]))
        return rows

    def recorder(self, k=None, **kwargs):
        fixture = Fixture()
        prefix = self.prefix() if k is not None else None
        teacher = Teacher()
        agent = (ReplayThenTeacher(teacher, prefix, k, RGB_KEY) if k is not None
                 else SimpleNamespace(step=0))
        if k is not None:
            agent.bind_environment(fixture.env)
        metadata = self.metadata("oracle" if k is not None else "student",
                                 **({"prefix_sha256": canonical_sha256(prefix)} if prefix else {}),
                                 **kwargs)
        rec = FailureStateRecorder(self.root(), metadata, lambda: fixture.env, k,
                                   expected_prefix=prefix, agent=agent)
        return rec, fixture, agent, teacher

    def run_frames(self, rec, fixture, agent, count=12, zero=False):
        for step in range(count):
            obs = fixture.set_step(step)
            detector = fixture.detector(zero)
            rec.observe(obs, detector, fixture.robot, fixture.human, fixture.episode,
                        step, 10, fixture.time)
            action = (agent.act(obs, detector, "1") if rec.takeover_step is not None
                      else [-.2, 0., 0.])
            rec.record_action(step, action, None)
            if rec.takeover_step is None:
                agent.step += 1

    def load(self, rec, filename):
        return json.loads((rec.root / filename).read_text())

    @staticmethod
    def result(success=True, collision=False):
        return dict(success=success, collision=collision, following_rate=.9,
                    policy_init_valid=True)

    def test_student_success_never_supplies_labels(self):
        rec, fixture, agent, _ = self.recorder()
        self.run_frames(rec, fixture, agent)
        report = rec.finish(self.result())
        self.assertEqual(self.load(rec, "windows.json"), [])
        self.assertGreater(len(self.load(rec, "raw_windows.json")), 0)
        self.assertEqual(report["status"], "rerun_student_success")
        self.assertTrue(report["no_recovery_needed"])
        self.assertFalse(report["training_eligible"])
        self.assertFalse(report["replay_verified"])
        self.assertEqual(report["owned_suffix"]["action_count"], 0)
        self.assertEqual(report["agent_validation"]["agent_step_before_reset"], 12)
        self.assertFalse(any("post_state" in row for row in rec.replay))
        self.assertEqual(len(self.load(rec, "observations.json")), 12)

    def test_nonzero_teacher_prefix_and_full_actual_time_window(self):
        rec, fixture, agent, teacher = self.recorder(k=3)
        self.run_frames(rec, fixture, agent)
        report = rec.finish(self.result())
        windows = self.load(rec, "windows.json")
        self.assertTrue(windows)
        self.assertTrue(report["replay_verified"])
        self.assertEqual(report["agent_validation"]["verified_prefix_frames"], 4)
        self.assertEqual(report["agent_validation"]["agent_step_before_reset"], 12)
        self.assertEqual(teacher.calls, 12)  # Includes prefix warm-up.
        self.assertEqual(rec.replay[2]["action"], [-.2, 0., 0.])
        self.assertEqual(rec.replay[3]["action"], [.2, 0., 0.])
        self.assertEqual([r["owner"] for r in rec.replay[:4]],
                         ["student", "student", "student", "teacher"])
        self.assertEqual(report["takeover_observation_index"], 3)
        self.assertEqual(report["takeover_state_sha256"],
                         canonical_sha256(rec.expected_prefix[3]["dynamic_state"]))
        for window in windows:
            i = window["current_index"]
            end = window["label_endpoint_observation_index"]
            self.assertGreaterEqual(i, 3)
            self.assertEqual(window["teacher_owned_action_indices"], [i, end-1])
            self.assertTrue(all(rec.replay[j]["owner"] == "teacher" for j in range(i, end)))
            self.assertFalse(window["training_eligible"])
        self.assertFalse(report["training_released"])
        self.assertEqual(self.load(rec, "actions.json")[3]["teacher_predicted_waypoints"], None)

    def test_owner_audit_checks_entire_interpolation_span_not_only_current(self):
        rec, fixture, agent, _ = self.recorder(k=3)
        self.run_frames(rec, fixture, agent)
        rec.finish(self.result())
        window = self.load(rec, "windows.json")[0]
        end = window["label_endpoint_observation_index"]
        rec.actions[end-1]["owner"] = "student"
        admitted, reasons = rec._filter_windows([window])
        self.assertEqual(admitted, [])
        self.assertEqual(reasons, {"label_interval_not_teacher_owned": 1})

    def test_endpoint_action_not_in_label_interval(self):
        rec, fixture, agent, _ = self.recorder(k=3)
        self.run_frames(rec, fixture, agent)
        rec.finish(self.result())
        window = self.load(rec, "windows.json")[0]
        end = window["label_endpoint_observation_index"]
        rec.actions[end]["owner"] = "student"
        admitted, _ = rec._filter_windows([window])
        self.assertEqual(len(admitted), 1)

    def test_prefix_error_is_retained_after_teacher_clears_it(self):
        rec, fixture, agent, teacher = self.recorder(k=3)
        teacher.reply_error = "prefix transport error"
        # Observe captures the error before it is cleared; actions subsequently run.
        obs = fixture.set_step(0)
        detector = fixture.detector()
        rec.observe(obs, detector, fixture.robot, fixture.human, fixture.episode, 0, 10, fixture.time)
        teacher.reply_error = None
        rec.record_action(0, agent.act(obs, detector, "1"), None)
        for step in range(1, 12):
            obs = fixture.set_step(step)
            rec.observe(obs, detector, fixture.robot, fixture.human, fixture.episode, step, 10, fixture.time)
            rec.record_action(step, agent.act(obs, detector, "1"), None)
        report = rec.finish(self.result())
        self.assertTrue(report["transport_fallback"])
        self.assertEqual(self.load(rec, "windows.json"), [])
        self.assertTrue(self.load(rec, "fallback_events.json"))

    def test_nested_fallback_count_detected_on_finish(self):
        rec, fixture, agent, teacher = self.recorder(k=3)
        self.run_frames(rec, fixture, agent)
        teacher.episode_fallback_count = 1
        report = rec.finish(self.result())
        self.assertTrue(report["transport_fallback"])
        self.assertEqual(self.load(rec, "windows.json"), [])

    def test_failed_collision_invalid_init_and_repeat_do_not_admit(self):
        for result, verification in [(self.result(False), False), (self.result(True, True), False),
                                     ({**self.result(), "policy_init_valid": 0.}, False),
                                     (self.result(), True)]:
            with self.subTest(result=result, verification=verification):
                rec, fixture, agent, _ = self.recorder(k=0, verification_only=verification)
                self.run_frames(rec, fixture, agent)
                report = rec.finish(result)
                self.assertEqual(self.load(rec, "windows.json"), [])
                self.assertGreater(report["raw_windows"], 0)

    def test_missing_or_invalid_initialization_flag_rejected_before_artifacts(self):
        cases = [
            {k: v for k, v in self.result().items() if k != "policy_init_valid"},
            *[{**self.result(), "policy_init_valid": value}
              for value in (None, "true", "1", 2, -1, float("nan"),
                            float("inf"), [], {})],
        ]
        for result in cases:
            with self.subTest(result=result):
                rec, fixture, agent, _ = self.recorder(k=3)
                self.run_frames(rec, fixture, agent)
                with self.assertRaisesRegex(ValueError, "policy_init_valid"):
                    rec.finish(result)
                self.assertFalse(rec._finalized)
                self.assertFalse(any((rec.root / name).exists() for name in
                                     ["admission.json", "complete.json", "windows.json"]))

    def test_explicit_numeric_one_initialization_flag_is_valid(self):
        for value in (1, 1.0, np.bool_(True)):
            with self.subTest(value=value):
                rec, fixture, agent, _ = self.recorder(k=3)
                self.run_frames(rec, fixture, agent)
                result = rec.finish({**self.result(), "policy_init_valid": value})
                self.assertGreater(result["candidate_windows"], 0)

    def test_first_rgb_repair_preserves_original_box_not_takeover_box(self):
        fixture = Fixture()
        rgb = fixture.set_step(0)[RGB_KEY]
        repair = dict(bbox=[1, 1, 4, 3], rgb_raw_sha256=hashlib.sha256(
            np.ascontiguousarray(rgb[..., :3]).tobytes()).hexdigest())
        rec = FailureStateRecorder(self.root(), self.metadata(), lambda: fixture.env,
                                   repair=repair, agent=SimpleNamespace(step=0))
        self.run_frames(rec, fixture, rec.agent, zero=True)
        rec.finish(self.result())
        metadata = self.load(rec, "metadata.json")
        self.assertEqual(metadata["initial_bbox_sensor_xyxy_original"], [0, 0, 0, 0])
        self.assertEqual(metadata["initial_bbox_sensor_xyxy"], [0, 0, 0, 0])
        self.assertEqual(metadata["initial_bbox_rgb_xyxy"], repair["bbox"])
        self.assertEqual(metadata["initial_bbox_status"], "VERIFIED_FROZEN_FIRST_RGB_REPAIR")
        self.assertEqual(self.load(rec, "first_start.json")["observation_index"], 0)

    def test_development_is_only_partial_evidence_no_terminal_or_windows(self):
        rec, fixture, agent, _ = self.recorder(k=3)
        self.run_frames(rec, fixture, agent, count=5)
        report = rec.finish_development("bounded real-runtime interface check")
        self.assertEqual(report["agent_validation"]["agent_step_before_reset"], 5)
        self.assertEqual(report["agent_validation"]["verified_prefix_frames"], 4)
        self.assertTrue(report["no_success_rate"])
        self.assertTrue(report["no_terminal_result"])
        self.assertFalse(report["complete"])
        self.assertFalse(report["training_eligible"])
        self.assertEqual(report["takeover"]["step"], 3)
        self.assertEqual(len(report["replay"]), 5)
        self.assertFalse(any((rec.root / name).exists() for name in
                             ["complete.json", "result.json", "windows.json", "raw_windows.json"]))
        self.assertEqual(sorted(p.name for p in rec.root.glob("*.json")), ["development.json"])
        with self.assertRaisesRegex(RuntimeError, "already finalized"):
            rec.finish(self.result())

    def test_missing_final_action_has_no_fabricated_poststate_or_release(self):
        rec, fixture, agent, _ = self.recorder(k=3)
        self.run_frames(rec, fixture, agent, count=11)
        obs = fixture.set_step(11)
        rec.observe(obs, fixture.detector(), fixture.robot, fixture.human, fixture.episode, 11, 10, fixture.time)
        report = rec.finish(self.result(False))
        self.assertFalse(report["complete"])
        self.assertEqual(self.load(rec, "windows.json"), [])
        self.assertNotIn("action", self.load(rec, "replay.json")[-1])
        self.assertNotIn("post_state", self.load(rec, "replay.json")[-1])

    def test_invalid_action_shape_range_nan_and_duplicate_fail_closed(self):
        for invalid in ([1, 2], [2, 0, 0], [float("nan"), 0, 0]):
            rec, fixture, agent, _ = self.recorder()
            obs = fixture.set_step(0)
            rec.observe(obs, fixture.detector(), fixture.robot, fixture.human, fixture.episode, 0, 10, fixture.time)
            with self.assertRaisesRegex(ValueError, "normalized action"):
                rec.record_action(0, invalid, None)
            self.assertIn("INVALID_NORMALIZED_ACTION", rec.issues)
        rec, fixture, agent, _ = self.recorder()
        self.run_frames(rec, fixture, agent, count=1)
        with self.assertRaisesRegex(ValueError, "fresh preaction"):
            rec.record_action(0, [0, 0, 0], None)

    def test_takeover_must_be_observed_and_prefix_hash_must_match(self):
        fixture = Fixture()
        prefix = self.prefix(3)
        meta = self.metadata("oracle", prefix_sha256=canonical_sha256(prefix))
        for k in (-1, 3, True):
            with self.assertRaises(ValueError):
                FailureStateRecorder(self.root(), meta, lambda: fixture.env, k,
                                     expected_prefix=prefix)
        with self.assertRaisesRegex(ValueError, "prefix hash mismatch"):
            FailureStateRecorder(self.root(), {**meta, "prefix_sha256": "0"*64},
                                 lambda: fixture.env, 1, expected_prefix=prefix)
        self.assertEqual(canonical_sha256(prefix), hashlib.sha256(json.dumps(
            prefix, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest())

    def test_no_verified_takeover_when_act_did_not_return_at_k(self):
        rec, fixture, agent, _ = self.recorder(k=3)
        self.run_frames(rec, fixture, agent, count=3)
        obs = fixture.set_step(3)
        rec.observe(obs, fixture.detector(), fixture.robot, fixture.human, fixture.episode, 3, 10, fixture.time)
        result = rec.finish_development("teacher act failed at takeover")
        self.assertEqual(result["agent_validation"]["verified_prefix_frames"], 3)
        self.assertFalse(result["replay_verified"])

    def test_replay_rgb_and_dynamic_mismatch_still_fail_closed(self):
        for mutation in ("rgb", "state"):
            rec, fixture, agent, _ = self.recorder(k=3)
            obs = fixture.set_step(0)
            if mutation == "rgb":
                obs[RGB_KEY][0, 0, 3] += 1
            else:
                fixture.obj.joint_positions[0] += .2
            with self.assertRaisesRegex(RuntimeError, "REPLAY_"):
                agent.act(obs, fixture.detector(), "1")
            self.assertEqual(agent.step, 0)

    def test_schema_cannot_impersonate_old_collection(self):
        fixture = Fixture()
        for change in ({"experiment": "evaluation_set_adaptation_v1"},
                       {"partition": "train"}, {"teacher": "lightnav"}):
            with self.assertRaises(ValueError):
                FailureStateRecorder(self.root(), self.metadata(**change), lambda: fixture.env)

    def test_observed_environment_must_be_the_replay_bound_environment(self):
        rec, fixture, agent, _ = self.recorder(k=3)
        other = Fixture()
        agent.bind_environment(other.env)
        # Do not fake successful replay: use genuine observe evidence and inspect
        # the binding proof before any act. A wrong environment is not verified.
        obs = fixture.set_step(0)
        rec.observe(obs, fixture.detector(), fixture.robot, fixture.human,
                    fixture.episode, 0, 10, fixture.time)
        proof = rec.finish_development("wrong environment diagnostic")["agent_validation"]
        self.assertFalse(proof["environment_bound"])
        self.assertFalse(proof["replay_verified"])
        self.assertEqual(proof["verified_prefix_frames"], 0)

    def test_bad_future_brackets_are_not_candidates(self):
        rec, fixture, agent, _ = self.recorder(k=3)
        self.run_frames(rec, fixture, agent)
        rec.finish(self.result())
        window = self.load(rec, "windows.json")[0]
        window["future_bracket_indices"][-1][1] -= 1
        admitted, reasons = rec._filter_windows([window])
        self.assertEqual(admitted, [])
        self.assertEqual(reasons, {"noncausal_or_incorrect_brackets": 1})


if __name__ == "__main__":
    unittest.main()
