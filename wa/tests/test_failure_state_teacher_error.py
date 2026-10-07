"""Pure CPU tests for teacher error evidence; no simulator or model imports."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from wa.wm.failure_state_teacher_error import (
    EXPERIMENT, TeacherOutputInvalid, TeacherTransportError, build_teacher_error,
    canonical_sha, validate_teacher_error,
)


class TeacherErrorEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.replay_path = Path(self.tmp.name) / "partial_replay.json"
        self.actions_path = Path(self.tmp.name) / "partial_actions.json"
        self.k, self.teacher = 5, "lightnav"
        self.prefix = []
        for i in range(8):
            transform = [[1, 0, 0, i * 0.01], [0, 1, 0, 0],
                         [0, 0, 1, 0], [0, 0, 0, 1]]
            state = dict(timestamp=i * 0.05,
                         agents=[dict(transform=transform, joints=[0.1, 0.2])])
            command = [0.1, 0, i * 0.01]
            self.prefix.append(dict(
                step=i, observation_index=i, timestamp_s=i * 0.05,
                simulator_world_time_s=i * 0.05,
                rgb_sha256=canonical_sha(["rgb", i]), dynamic_state=state,
                dynamic_state_sha256=canonical_sha(state), action=command,
                normalized_action=command, owner="student",
                action_timing="before_env_step", post_state_recorded=False))
        self.pair = dict(
            experiment=EXPERIMENT, task="stt", key="rJhMRvNn4DS/8",
            takeover_step=self.k, seed=7, protocol_sha256="a" * 64,
            initial_rgb_sha256=self.prefix[0]["rgb_sha256"],
            takeover_state_sha256=self.prefix[self.k]["dynamic_state_sha256"],
            prefix_sha256=canonical_sha(self.prefix))
        self.error = TeacherOutputInvalid(
            rc=500, seq=3, message="Missing rvq act levels [0] in raw output")
        self.fixture(3)

    def fixture(self, failed_step):
        replay = copy.deepcopy(self.prefix[:failed_step + 1])
        actions = []
        for i, frame in enumerate(replay):
            if i == failed_step:
                for field in ("action", "normalized_action", "owner",
                              "action_timing", "post_state_recorded"):
                    frame.pop(field)
                continue
            owner = "student" if i < self.k else "teacher"
            cmd = frame["action"] if i < self.k else [0.2, 0.1, 0]
            frame.update(action=cmd, normalized_action=cmd, owner=owner)
            actions.append(dict(sim_step=i, normalized_action=cmd, owner=owner,
                                teacher=None if i < self.k else self.teacher))
        self.save(replay, actions)
        return replay, actions

    def save(self, replay, actions):
        self.replay_path.write_text(json.dumps(replay))
        self.actions_path.write_text(json.dumps(actions))

    def build(self, **kwargs):
        args = dict(error=self.error, teacher=self.teacher, expected_pair=self.pair,
                    prefix=self.prefix, partial_replay_path=self.replay_path,
                    partial_actions_path=self.actions_path, fallback_detected=True)
        args.update(kwargs)
        return build_teacher_error(**args)

    def valid(self, doc):
        self.assertEqual(validate_teacher_error(doc, prefix=self.prefix), doc)

    def invalid_doc(self, mutate):
        doc = self.build()
        mutate(doc)
        with self.assertRaises((ValueError, TypeError)):
            validate_teacher_error(doc, prefix=self.prefix)

    def test_before_takeover_has_no_actual_takeover(self):
        doc = self.build()
        obs = doc["observed_partial"]
        self.assertEqual((obs["observations"], obs["recorded_actions"]), (4, 3))
        self.assertEqual(obs["verified_prefix_frames"], 4)
        self.assertEqual(obs["failure_phase"], "before_takeover")
        self.assertIsNone(obs["actual_takeover"])
        self.assertIsNone(obs["actual_takeover_state_sha256"])
        self.assertIsNone(doc["result"])
        self.assertFalse(doc["replay_verified"])
        self.assertFalse(doc["complete"])
        self.assertFalse(doc["training_eligible"])
        self.assertEqual(doc["windows"], [])
        self.valid(doc)

    def test_at_takeover_uses_observed_frame(self):
        self.fixture(5)
        doc = self.build()
        obs = doc["observed_partial"]
        self.assertEqual(obs["failure_phase"], "at_takeover")
        self.assertEqual(obs["verified_prefix_frames"], 6)
        self.assertTrue(obs["prefix_validation_complete"])
        self.assertEqual(obs["actual_takeover"]["observation_index"], 5)
        self.assertFalse(doc["replay_verified"])
        self.valid(doc)

    def test_after_takeover_teacher_owned_commands(self):
        self.fixture(7)
        doc = self.build()
        self.assertEqual(doc["observed_partial"]["failure_phase"], "after_takeover")
        self.assertEqual(doc["observed_partial"]["recorded_actions"], 7)
        self.valid(doc)

    def test_empty_pre_observation_transport_evidence(self):
        self.save([], [])
        error = TeacherTransportError(exception_type="ReviewedConnectionError",
                                      message="connection closed before first observation")
        doc = self.build(error=error, fallback_detected=False)
        self.assertIsNone(doc["raw_error"]["rc"])
        self.assertEqual(doc["observed_partial"]["failure_phase"], "before_first_observation")
        self.assertIsNone(doc["observed_partial"]["initial_rgb_sha256"])
        self.valid(doc)

    def test_transport_preserves_actual_rc_seq(self):
        error = TeacherTransportError(exception_type="ReviewedTransportError",
                                      message="service unavailable", rc=503, seq=3)
        doc = self.build(error=error)
        self.assertEqual(doc["status"], "teacher_transport_error")
        self.assertEqual((doc["raw_error"]["rc"], doc["raw_error"]["seq"]), (503, 3))
        self.valid(doc)

    def test_repeat_error_explicitly_marked(self):
        doc = self.build(verification_only=True)
        self.assertTrue(doc["verification_only"])
        self.assertEqual(doc["candidate_windows"], 0)
        self.valid(doc)

    def test_oracle_explicit_transport_error_supported(self):
        self.teacher = "oracle"
        self.fixture(7)
        doc = self.build(error=TeacherTransportError(
            exception_type="ExplicitReviewedAdapterError", message="teacher boundary failure"))
        self.valid(doc)

    def test_no_generic_exception_conversion(self):
        for error in (RuntimeError("TEACHER_TRANSPORT_FALLBACK"), ValueError("invalid action")):
            with self.subTest(error=error):
                with self.assertRaises(TypeError):
                    self.build(error=error)

    def test_fallback_executed_rejected(self):
        with self.assertRaises(ValueError):
            self.build(fallback_executed=True)

    def test_fallback_flags_must_be_bool(self):
        with self.assertRaises(ValueError):
            self.build(fallback_detected=1)

    def test_terminal_result_or_rate_rejected(self):
        for field, value in (("result", {"success": 0, "following_rate": 0}),
                             ("following_rate", 0), ("success", False)):
            with self.subTest(field=field):
                self.invalid_doc(lambda d: d.update({field: value}))

    def test_release_complete_proof_or_windows_rejected(self):
        for field, value in (("complete", True), ("replay_verified", True),
                             ("training_released", True), ("training_eligible", True),
                             ("candidate_windows", 1), ("training_windows", 1),
                             ("windows", [{}])):
            with self.subTest(field=field):
                self.invalid_doc(lambda d: d.update({field: value}))

    def test_expected_takeover_cannot_masquerade_as_observed(self):
        self.invalid_doc(lambda d: d["observed_partial"].update(
            actual_takeover_state_sha256=self.pair["takeover_state_sha256"]))
        self.invalid_doc(lambda d: d["observed_partial"].update(
            actual_takeover=dict(dynamic_state_sha256=self.pair["takeover_state_sha256"])))

    def test_tolerant_state_still_hashes_actual_not_expected(self):
        replay, actions = self.fixture(5)
        replay[5]["dynamic_state"]["agents"][0]["transform"][0][3] += 0.0000005
        replay[5]["dynamic_state_sha256"] = canonical_sha(replay[5]["dynamic_state"])
        self.save(replay, actions)
        doc = self.build()
        actual = doc["observed_partial"]["actual_takeover_state_sha256"]
        self.assertNotEqual(actual, self.pair["takeover_state_sha256"])
        self.valid(doc)
        doc["observed_partial"]["actual_takeover_state_sha256"] = self.pair["takeover_state_sha256"]
        with self.assertRaises(ValueError):
            validate_teacher_error(doc, prefix=self.prefix)

    def test_state_divergence_stays_fatal(self):
        replay, actions = self.fixture(3)
        replay[2]["dynamic_state"]["timestamp"] += 0.1
        replay[2]["timestamp_s"] += 0.1
        replay[2]["simulator_world_time_s"] += 0.1
        replay[2]["dynamic_state_sha256"] = canonical_sha(replay[2]["dynamic_state"])
        self.save(replay, actions)
        with self.assertRaisesRegex(ValueError, "REPLAY_DYNAMIC_STATE_DIVERGED"):
            self.build()

    def test_rgb_divergence_stays_fatal(self):
        replay, actions = self.fixture(3)
        replay[2]["rgb_sha256"] = "f" * 64
        self.save(replay, actions)
        with self.assertRaisesRegex(ValueError, "REPLAY_RGB_DIVERGED"):
            self.build()

    def test_prefix_action_disagreement_stays_fatal(self):
        replay, actions = self.fixture(3)
        replay[1]["action"] = replay[1]["normalized_action"] = [0.9, 0, 0]
        actions[1]["normalized_action"] = [0.9, 0, 0]
        self.save(replay, actions)
        with self.assertRaisesRegex(ValueError, "REPLAY_PREFIX_ACTION_DIVERGED"):
            self.build()

    def test_wrong_owner_rejected(self):
        replay, actions = self.fixture(7)
        actions[5]["owner"] = "student"
        self.save(replay, actions)
        with self.assertRaises(ValueError):
            self.build()

    def test_failed_action_record_rejected(self):
        replay, actions = self.fixture(3)
        replay[-1]["action"] = [0.1, 0, 0]
        self.save(replay, actions)
        with self.assertRaises(ValueError):
            self.build()

    def test_partial_counts_invalid(self):
        replay, actions = self.fixture(3)
        self.save(replay[:-1], actions)
        with self.assertRaises(ValueError):
            self.build()

    def test_missing_frame_evidence(self):
        replay, actions = self.fixture(3)
        replay[2].pop("dynamic_state")
        self.save(replay, actions)
        with self.assertRaises(ValueError):
            self.build()

    def test_modified_file_hash_rejected_even_same_json(self):
        doc = self.build()
        self.replay_path.write_text(self.replay_path.read_text() + "\n")
        with self.assertRaises(ValueError):
            validate_teacher_error(doc, prefix=self.prefix)

    def test_forged_hash_count_and_status_rejected(self):
        self.invalid_doc(lambda d: d["observed_partial"].update(recorded_actions=32))
        self.invalid_doc(lambda d: d["observed_partial"]["evidence_files"]["partial_replay"].update(sha256="b"*64))
        self.invalid_doc(lambda d: d.update(status="unknown_error"))

    def test_missing_raw_error_and_unknown_fields_rejected(self):
        self.invalid_doc(lambda d: d["raw_error"].pop("message"))
        self.invalid_doc(lambda d: d.update(other="unexpected"))

    def test_wrong_expected_prefix_or_pair_rejected(self):
        pair = copy.deepcopy(self.pair)
        pair["prefix_sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            self.build(expected_pair=pair)
        pair = copy.deepcopy(self.pair)
        pair["experiment"] = "evaluation_adaptation_failure_state_v1"
        with self.assertRaises(ValueError):
            self.build(expected_pair=pair)

    def test_nonfinite_and_duplicate_json_rejected(self):
        self.replay_path.write_text("[NaN]")
        with self.assertRaises(ValueError):
            self.build()
        self.replay_path.write_text('[{"step":0,"step":1}]')
        with self.assertRaises(ValueError):
            self.build()

    def test_malformed_exception_metadata_rejected(self):
        for kwargs in (dict(rc=500, seq=None, message="bad"),
                       dict(rc=True, seq=3, message="bad"),
                       dict(rc=500, seq=3, message="")):
            with self.assertRaises(ValueError):
                TeacherOutputInvalid(**kwargs)


if __name__ == "__main__":
    unittest.main()
