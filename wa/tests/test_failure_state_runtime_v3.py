"""CPU-only actual Runtime.run/recorder/replay integration; no Habitat rollout."""
import copy
import inspect
import json
import sys
import unittest
from unittest.mock import patch

from wa.tests import test_failure_state_runtime_v2 as fixtures
from wa.wm.failure_state_collect import Runtime
from wa.wm.failure_state_boundary_policy import (
    LEGACY_POLICY, CONTINUATION_POLICY, validate_boundary_policy,
)
from wa.wm import failure_state_protocol as v1
from wa.wm import failure_state_protocol_v2 as v2
from wa.wm.failure_state_teacher_error import validate_teacher_error


class CountTeacher(fixtures.RawTeacher):
    MESSAGE = ("got 2 act levels, expected 3 from "
               "'<apos_1273><opos_0><act_l0_167><act_l1_69><opos_26><|im_end|>'")

    def act(self, *args, **kwargs):
        action = super().act(*args, **kwargs)
        if self.reply_error is not None:
            self.reply_error["msg"] = self.MESSAGE
        return action


class RuntimeV3Tests(unittest.TestCase):
    setUp = fixtures.RuntimeV2Tests.setUp

    def setup_runtime(self, raw, *, policy=CONTINUATION_POLICY, experiment=v2.EXPERIMENT,
                      oracle_factory=None, corrupt_observation=False):
        runtime, modules = fixtures.RuntimeV2Tests.runtime_stub(self, raw, experiment)
        if policy is not None:
            runtime.teacher_boundary_policy = policy
        events = []

        def evaluate(config, subset, metrics, agent_factory, recorder):
            world = fixtures.ZeroWorldFixture()
            agent = agent_factory(None)
            agent.bind_environment(world.env)
            agent.reset()
            for i in range(50):
                obs = world.set_step(i)
                if corrupt_observation and i == 2:
                    obs[fixtures.RGB_KEY] = obs[fixtures.RGB_KEY] + 1
                recorder.observe(obs, world.detector(), world.robot, world.human,
                                 world.episode, i, 10, world.time)
                events.append(("observe", i))
                action = agent.act(obs, world.detector(), "1")
                recorder.record_action(i, action, None)
                events.append(("record_action", i))
                # Fake stepping seam follows exactly the benchmark's ordering.
                events.append(("env_step", i))
            recorder.finish(dict(success=True, collision=False, policy_init_valid=True,
                                 following_rate=.9, total_step=50, status="Success"))
            agent.reset()

        modules["trained_agent"].evaluate_agent = evaluate
        modules["wa.wm.oracle_teacher"].OracleTeacher = oracle_factory or (lambda: fixtures.RawTeacher())
        return runtime, modules, events

    def test_policy_constructor_and_identity_fail_closed(self):
        default = inspect.signature(Runtime.__init__).parameters["teacher_boundary_policy"].default
        self.assertEqual(default, LEGACY_POLICY)
        for experiment in (v1.EXPERIMENT, v2.EXPERIMENT):
            self.assertEqual(validate_boundary_policy(LEGACY_POLICY, experiment=experiment), LEGACY_POLICY)
        self.assertEqual(validate_boundary_policy(CONTINUATION_POLICY, experiment=v2.EXPERIMENT),
                         CONTINUATION_POLICY)
        for policy, experiment in ((CONTINUATION_POLICY, v1.EXPERIMENT),
                                  (LEGACY_POLICY, "foreign"), (CONTINUATION_POLICY, "foreign"),
                                  (True, v2.EXPERIMENT), (None, v2.EXPERIMENT),
                                  ("", v2.EXPERIMENT), ("other", v2.EXPERIMENT)):
            with self.subTest(policy=policy, experiment=experiment), self.assertRaises(ValueError):
                validate_boundary_policy(policy, experiment=experiment)
        # Failure happens before importing Habitat or opening output files.
        with self.assertRaisesRegex(ValueError, "explicit v2"):
            Runtime(self.plan, "a"*64, {}, self.root,
                    teacher_boundary_policy=CONTINUATION_POLICY)
        with self.assertRaisesRegex(ValueError, "unknown teacher boundary"):
            Runtime(self.plan, "a"*64, {}, self.root, experiment=v2.EXPERIMENT,
                    protocol_sha=v2.PROTOCOL_SHA, continuation_sha="b"*64,
                    teacher_boundary_policy="foreign")

    def test_new_count_error_is_before_record_or_step_and_closes_raw(self):
        raw = CountTeacher(37)
        runtime, modules, events = self.setup_runtime(raw)
        with patch.dict(sys.modules, modules):
            result = runtime.run(self.entry, self.root, "count", "unused",
                                 teacher_name="lightnav", prefix=self.prefix, takeover_step=35)
        self.assertTrue(raw.closed)
        self.assertEqual(result["status"], "teacher_output_invalid")
        self.assertEqual(result["candidate_windows"], 0)
        self.assertIsNone(result["result"])
        self.assertFalse(result["complete"])
        self.assertFalse(result["replay_verified"])
        self.assertFalse(result["fallback_executed"])
        self.assertEqual(validate_teacher_error(result, prefix=self.prefix), result)
        self.assertEqual(events[-1], ("observe", 37))
        self.assertEqual([i for kind, i in events if kind == "env_step"], list(range(37)))
        base = self.root/"count"
        replay = json.loads((base/"partial_replay.json").read_text())
        actions = json.loads((base/"partial_actions.json").read_text())
        self.assertEqual((len(replay), len(actions)), (38, 37))
        self.assertIsNone(replay[-1].get("action"))
        context = json.loads((base/"teacher_error_context.json").read_text())
        self.assertEqual(context["metadata"]["teacher_boundary_policy"], CONTINUATION_POLICY)
        for name in ("result.json", "complete.json", "admission.json", "windows.json", "ERROR.json"):
            self.assertFalse((base/name).exists(), name)

    def test_default_v1_and_v2_count_error_remains_fatal(self):
        for experiment, policy in ((v1.EXPERIMENT, None), (v2.EXPERIMENT, None),
                                   (v2.EXPERIMENT, LEGACY_POLICY)):
            raw = CountTeacher(37)
            runtime, modules, events = self.setup_runtime(raw, policy=policy, experiment=experiment)
            name = "legacy" + str(len(list(self.root.iterdir())))
            with patch.dict(sys.modules, modules), self.assertRaisesRegex(RuntimeError, "TEACHER_TRANSPORT_FALLBACK"):
                runtime.run(self.entry, self.root, name, "unused", teacher_name="lightnav",
                            prefix=self.prefix, takeover_step=35)
            self.assertTrue(raw.closed)
            self.assertEqual(events[-1], ("observe", 37))
            self.assertTrue((self.root/name/"ERROR.json").exists())
            self.assertFalse((self.root/name/"teacher_error.json").exists())

    def test_oracle_same_k_continues_after_invalid_ln_and_selection_is_unchanged(self):
        raw = CountTeacher(37)
        oracles = []
        def oracle_factory():
            teacher = fixtures.RawTeacher()
            oracles.append(teacher)
            return teacher
        runtime, modules, _ = self.setup_runtime(raw, oracle_factory=oracle_factory)
        calls = []
        def run(name, k, repeat):
            calls.append((name, k, repeat))
            return runtime.run(self.entry, self.root, f"{name}_{k}_{repeat}", "unused",
                               teacher_name=name, prefix=self.prefix, takeover_step=k,
                               verification_only=repeat)
        student = dict(result=dict(success=False, collision=False, policy_init_valid=True,
                                   following_rate=.2))
        with patch.dict(sys.modules, modules):
            report = v2.search_recovery(student, self.prefix, run)
        self.assertEqual(calls, [("lightnav", 35, False), ("oracle", 35, False), ("oracle", 35, True)])
        self.assertEqual(report["accepted"]["teacher"], "oracle")
        self.assertEqual(report["accepted"]["takeover_step"], 35)
        self.assertFalse(report["training_released"])
        self.assertEqual(report["attempts"][0]["selection"]["pair_comparability"], "partial")
        self.assertTrue(report["attempts"][0]["repeat_valid"])
        self.assertGreater(report["attempts"][0]["selected_candidate_windows"], 0)
        self.assertTrue(raw.closed)
        self.assertTrue(all(t.closed for t in oracles))

    def test_developer_invalid_cannot_be_a_pass(self):
        raw = CountTeacher(37)
        runtime, modules, _ = self.setup_runtime(raw)
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(RuntimeError, "not a passed"):
            runtime.run(self.entry, self.root, "developer", "unused", teacher_name="lightnav",
                        prefix=self.prefix, takeover_step=35, development_actions=45)
        self.assertTrue(raw.closed)
        self.assertTrue((self.root/"developer/teacher_error.json").exists())
        self.assertTrue((self.root/"developer/ERROR.json").exists())
        self.assertFalse((self.root/"developer/DEVELOPMENT_CHECK.json").exists())
        self.assertFalse((self.root/"developer/complete.json").exists())

    def test_unknown_oracle_and_replay_faults_still_fatal(self):
        for case in ("unknown", "oracle", "replay"):
            raw = CountTeacher(37, unknown=case == "unknown")
            runtime, modules, events = self.setup_runtime(
                raw, oracle_factory=lambda: raw, corrupt_observation=case == "replay")
            name = "fatal_" + case
            with patch.dict(sys.modules, modules), self.assertRaises(RuntimeError):
                runtime.run(self.entry, self.root, name, "unused",
                            teacher_name="oracle" if case == "oracle" else "lightnav",
                            prefix=self.prefix, takeover_step=35)
            self.assertTrue(raw.closed)
            self.assertTrue((self.root/name/"ERROR.json").exists())
            self.assertFalse((self.root/name/"teacher_error.json").exists())

    def test_default_metadata_remains_identical_and_new_policy_is_explicit(self):
        saved = []
        for policy in (None, LEGACY_POLICY, CONTINUATION_POLICY):
            raw = CountTeacher()
            runtime, modules, _ = self.setup_runtime(raw, policy=policy)
            name = "normal" + str(len(saved))
            with patch.dict(sys.modules, modules):
                runtime.run(self.entry, self.root, name, "unused", teacher_name="lightnav",
                            prefix=self.prefix, takeover_step=35)
            saved.append((self.root/name/"metadata.json").read_bytes())
            self.assertTrue(raw.closed)
        self.assertEqual(saved[0], saved[1])
        legacy, new = json.loads(saved[0]), json.loads(saved[2])
        self.assertNotIn("teacher_boundary_policy", legacy)
        self.assertEqual(new.pop("teacher_boundary_policy"), CONTINUATION_POLICY)
        self.assertEqual(new, legacy)


if __name__ == "__main__":
    unittest.main()
