"""No simulator/GPU/plan writes: v2 bounded selection and error-proof contracts."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from wa.wm import failure_state_protocol as v1
from wa.wm import failure_state_protocol_v2 as v2
from wa.wm.failure_state_teacher_error import (
    TeacherOutputInvalid, build_teacher_error, canonical_sha,
)


class ProtocolV2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.serial = 0
        self.prefix = []
        for i in range(70):
            state = dict(timestamp=i * 0.05, agents=[dict(
                transform=[[1, 0, 0, i * 0.01], [0, 1, 0, 0],
                           [0, 0, 1, 0], [0, 0, 0, 1]], joints=[0.1])])
            command = [0.1, 0.0, 0.01 * i]
            self.prefix.append(dict(
                step=i, observation_index=i, timestamp_s=i * 0.05,
                simulator_world_time_s=i * 0.05, rgb_sha256=canonical_sha(["rgb", i]),
                dynamic_state=state, dynamic_state_sha256=canonical_sha(state),
                action=command, normalized_action=command, owner="student",
                action_timing="before_env_step", post_state_recorded=False))
        self.student = dict(result=dict(success=0.0, collision=0.0,
                                       policy_init_valid=True, following_rate=0.1))

    def pair(self, k):
        return dict(
            experiment=v2.EXPERIMENT, task="stt", key="rJhMRvNn4DS/8", takeover_step=k,
            seed=7, protocol_sha256=v2.PROTOCOL_SHA,
            initial_rgb_sha256=self.prefix[0]["rgb_sha256"],
            takeover_state_sha256=canonical_sha(self.prefix[k]["dynamic_state"]),
            prefix_sha256=canonical_sha(self.prefix))

    def normal(self, name, k=5, repeat=False, tr=0.7, success=True,
               collision=False, windows=3):
        self.serial += 1
        return dict(
            self.pair(k), teacher=name, complete=True, replay_verified=True,
            transport_fallback=False, verification_only=repeat,
            artifact_root=str(self.root / f"{name}_{k}_{repeat}_{self.serial}"),
            candidate_windows=windows,
            result=dict(success=float(success), collision=float(collision),
                        policy_init_valid=True, following_rate=tr))

    def error(self, name, k=5, repeat=False):
        self.serial += 1
        folder = self.root / f"error_{name}_{k}_{repeat}_{self.serial}"
        folder.mkdir()
        failed_step = min(2, k)
        replay = copy.deepcopy(self.prefix[:failed_step+1])
        actions = []
        for i, frame in enumerate(replay):
            if i == failed_step:
                for field in ("action", "normalized_action", "owner",
                              "action_timing", "post_state_recorded"):
                    frame.pop(field)
            else:
                actions.append(dict(sim_step=i, normalized_action=frame["action"],
                                    owner="student", teacher=None))
        rp, ap = folder/"partial_replay.json", folder/"partial_actions.json"
        rp.write_text(json.dumps(replay)); ap.write_text(json.dumps(actions))
        return build_teacher_error(
            error=TeacherOutputInvalid(rc=500, seq=failed_step, message="missing rvq level"),
            teacher=name, expected_pair=self.pair(k), prefix=self.prefix,
            partial_replay_path=rp, partial_actions_path=ap, fallback_detected=True,
            verification_only=repeat)

    def select(self, ln, oc):
        return v2.select_recovery_teacher(ln, oc, prefix=self.prefix)

    def test_new_protocol_identity_and_unchanged_basics(self):
        self.assertNotEqual(v2.PROTOCOL_SHA, v1.PROTOCOL_SHA)
        self.assertEqual(v2.PROTOCOL["base_protocol_sha"], v1.PROTOCOL_SHA)
        for key in ("backoffs", "candidate_rule", "model_contract", "scope",
                    "expected_count", "lanes", "teacher_choice", "exact_tie"):
            self.assertEqual(v2.PROTOCOL[key], v1.PROTOCOL[key])
        self.assertEqual(v1.EXPERIMENT, "evaluation_adaptation_failure_state_v1")
        self.assertFalse(hasattr(v2, "load_plan"))

    def test_normal_higher_tr_and_exact_tie(self):
        for ln_tr, oc_tr, expected in ((0.7, 0.8, "oracle"),
                                        (0.8, 0.7, "lightnav"),
                                        (0.7, 0.7, "lightnav")):
            with self.subTest(ln=ln_tr, oc=oc_tr):
                choice = self.select(self.normal("lightnav", tr=ln_tr),
                                     self.normal("oracle", tr=oc_tr))
                self.assertEqual(choice["selected_teacher"], expected)
                self.assertEqual(choice["pair_comparability"], "full")

    def test_lightnav_error_oracle_success(self):
        choice = self.select(self.error("lightnav"), self.normal("oracle"))
        self.assertEqual(choice["selected_teacher"], "oracle")
        self.assertEqual(choice["reason"], "only_eligible_teacher")
        self.assertIsNone(choice["results"]["lightnav"])
        self.assertEqual(choice["pair_comparability"], "partial")
        self.assertTrue(choice["runtime_invalid"]["lightnav"])

    def test_normal_failure_is_not_runtime_error(self):
        choice = self.select(self.normal("lightnav", success=False),
                             self.normal("oracle"))
        self.assertFalse(choice["runtime_invalid"]["lightnav"])
        self.assertIsNotNone(choice["results"]["lightnav"])
        self.assertEqual(choice["pair_comparability"], "full")

    def test_normal_fallback_ineligible_and_partial(self):
        ln = self.normal("lightnav")
        ln["transport_fallback"] = True
        choice = self.select(ln, self.normal("oracle"))
        self.assertEqual(choice["selected_teacher"], "oracle")
        self.assertEqual(choice["pair_comparability"], "partial")

    def test_bad_flags_nan_and_inconsistent_result_rejected(self):
        for field, value in (("success", "0"), ("following_rate", float("nan")),
                             ("following_rate", 1.1), ("collision", True),
                             ("policy_init_valid", None)):
            branch = self.normal("lightnav")
            branch["result"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.select(branch, self.normal("oracle"))

    def test_foreign_pin_normal_and_error_rejected(self):
        ln = self.normal("lightnav")
        ln["protocol_sha256"] = v1.PROTOCOL_SHA
        with self.assertRaises(ValueError):
            self.select(ln, self.normal("oracle"))
        ln = self.error("lightnav")
        ln["expected_pair"]["protocol_sha256"] = v1.PROTOCOL_SHA
        with self.assertRaises(ValueError):
            self.select(ln, self.normal("oracle"))

    def test_expected_key_k_and_prefix_tamper_rejected(self):
        ln = self.normal("lightnav")
        ln["key"] = "AnotherScene/8"
        with self.assertRaises(ValueError):
            self.select(ln, self.normal("oracle"))
        with self.assertRaises(ValueError):
            self.select(self.normal("lightnav", k=5), self.normal("oracle", k=6))
        ln = self.normal("lightnav")
        ln["prefix_sha256"] = "c"*64
        with self.assertRaises(ValueError):
            self.select(ln, self.normal("oracle"))

    def test_error_file_hash_tamper_rejected(self):
        error = self.error("lightnav")
        path = Path(error["observed_partial"]["evidence_files"]["partial_replay"]["path"])
        path.write_text(path.read_text() + "\n")
        with self.assertRaises(ValueError):
            self.select(error, self.normal("oracle"))

    def test_unknown_error_status_and_fake_score_rejected(self):
        error = self.error("lightnav")
        error["status"] = "some_runtime_error"
        with self.assertRaises(ValueError):
            self.select(error, self.normal("oracle"))
        error = self.error("lightnav")
        error["result"] = dict(success=0, collision=0, policy_init_valid=True, following_rate=0)
        with self.assertRaises(ValueError):
            self.select(error, self.normal("oracle"))

    def test_normal_repeat_unchanged(self):
        original, repeat = self.normal("oracle"), self.normal("oracle", repeat=True)
        self.assertTrue(v2.repeat_valid(original, repeat, prefix=self.prefix))
        repeat["result"]["success"] = False
        self.assertFalse(v2.repeat_valid(original, repeat, prefix=self.prefix))

    def test_classified_repeat_error_false(self):
        self.assertFalse(v2.repeat_valid(
            self.normal("oracle"), self.error("oracle", repeat=True), prefix=self.prefix))

    def test_repeat_error_identity_and_directory_strict(self):
        original = self.normal("oracle")
        error = self.error("oracle", repeat=True)
        error["expected_pair"]["key"] = "AnotherScene/8"
        with self.assertRaises(ValueError):
            v2.repeat_valid(original, error, prefix=self.prefix)
        error = self.error("oracle", repeat=True)
        original["artifact_root"] = str(Path(
            error["observed_partial"]["evidence_files"]["partial_replay"]["path"]).parent)
        with self.assertRaises(ValueError):
            v2.repeat_valid(original, error, prefix=self.prefix)

    def test_repeat_role_and_teacher_strict(self):
        original = self.normal("oracle")
        for repeat in (self.error("oracle", repeat=False),
                       self.error("lightnav", repeat=True)):
            with self.assertRaises(ValueError):
                v2.repeat_valid(original, repeat, prefix=self.prefix)

    def test_verification_cannot_be_candidate(self):
        with self.assertRaises(ValueError):
            self.select(self.error("lightnav", repeat=True), self.normal("oracle"))

    def test_both_error_bounded_all_candidates(self):
        calls = []
        def run(name, k, repeat):
            calls.append((name, k, repeat))
            return self.error(name, k, repeat)
        report = v2.search_recovery(self.student, self.prefix, run)
        self.assertEqual([a["takeover_step"] for a in report["attempts"]], [65, 55, 40, 10, 0])
        self.assertEqual(len(calls), 10)
        self.assertTrue(all(not r for _, _, r in calls))
        self.assertIsNone(report["accepted"])
        self.assertEqual(report["outcome"], "no_valid_teacher_recovery")

    def test_error_repeat_goes_earlier_no_same_k_runner_up(self):
        calls = []
        def run(name, k, repeat):
            calls.append((name, k, repeat))
            if repeat and k == 65:
                return self.error(name, k, repeat)
            return self.normal(name, k, repeat, tr=0.9 if name=="lightnav" else 0.8)
        report = v2.search_recovery(self.student, self.prefix, run)
        self.assertEqual(report["accepted"]["takeover_step"], 55)
        self.assertEqual(calls, [
            ("lightnav", 65, False), ("oracle", 65, False), ("lightnav", 65, True),
            ("lightnav", 55, False), ("oracle", 55, False), ("lightnav", 55, True)])

    def test_error_lightnav_oracle_requires_single_repeat(self):
        calls = []
        def run(name, k, repeat):
            calls.append((name, k, repeat))
            return self.error(name, k, repeat) if name=="lightnav" else self.normal(name,k,repeat)
        report = v2.search_recovery(self.student, self.prefix, run)
        self.assertEqual(report["accepted"]["teacher"], "oracle")
        self.assertEqual(calls[-1], ("oracle", 65, True))
        self.assertEqual(len(calls), 3)
        self.assertFalse(report["training_released"])

    def test_zero_windows_earlier_after_single_repeat(self):
        calls = []
        def run(name, k, repeat):
            calls.append((name, k, repeat))
            return self.normal(name, k, repeat, windows=0 if k==65 else 1)
        report = v2.search_recovery(self.student, self.prefix, run)
        self.assertEqual(report["accepted"]["takeover_step"], 55)
        self.assertEqual(report["attempts"][0]["admission_reason"],
                         "selected_teacher_no_candidate_windows")
        self.assertEqual(len(calls), 6)

    def test_unknown_exception_propagates_without_oracle(self):
        calls = []
        def run(name, k, repeat):
            calls.append((name, k, repeat))
            raise RuntimeError("REPLAY_DYNAMIC_STATE_DIVERGED")
        with self.assertRaisesRegex(RuntimeError, "REPLAY_DYNAMIC_STATE_DIVERGED"):
            v2.search_recovery(self.student, self.prefix, run)
        self.assertEqual(calls, [("lightnav", 65, False)])

    def test_successful_student_no_calls_or_backfill(self):
        student = copy.deepcopy(self.student)
        student["result"]["success"] = True
        report = v2.search_recovery(student, self.prefix,
                                   lambda *args: self.fail("no teacher run expected"))
        self.assertEqual(report["outcome"], "rerun_student_success_no_recovery_needed")
        self.assertFalse(report["score_backfill_allowed"])
        self.assertEqual(report["attempts"], [])

    def test_invalid_student_init_no_calls(self):
        student = copy.deepcopy(self.student)
        student["result"]["policy_init_valid"] = False
        report = v2.search_recovery(student, self.prefix,
                                   lambda *args: self.fail("no teacher run expected"))
        self.assertEqual(report["outcome"], "student_invalid_initialization")

    def test_unrequested_k_return_rejected(self):
        def run(name, k, repeat):
            return self.normal(name, 5, repeat)
        with self.assertRaisesRegex(ValueError, "unrequested k"):
            v2.search_recovery(self.student, self.prefix, run)


if __name__ == "__main__":
    unittest.main()
