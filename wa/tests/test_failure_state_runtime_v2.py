"""CPU integration tests: real recorder/replay/boundary, fake evaluator only."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from wa.tests.test_failure_state_recorder import Fixture, Teacher
from wa.wm.failure_state_collect import Runtime, runtime_contract
from wa.wm import failure_state_protocol as v1
from wa.wm import failure_state_protocol_v2 as v2
from wa.wm.failure_state_recorder import FailureStateRecorder, canonical_sha256, RGB_KEY
from wa.wm.failure_state_teacher_adapter import FailureStateTeacherBoundary
from wa.wm.failure_state_teacher_error import TeacherOutputInvalid, validate_teacher_error
from wa.wm.recovery_replay import ReplayThenTeacher, dynamic_state, rgb_hash
from wa.wm.failure_state_collect_v2 import load_workload, execute_entry


class ZeroWorldFixture(Fixture):
    def set_step(self, step, dt=.13):
        obs = super().set_step(step, dt)
        self.time -= 10
        return obs


class RawTeacher(Teacher):
    def __init__(self, fail_at=None, unknown=False):
        super().__init__()
        self.fail_at, self.unknown, self.seq, self.closed = fail_at, unknown, 0, False
    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.seq, self.episode_fallback_count, self.reply_error = 0, 0, None
    def act(self, *args, **kwargs):
        seq = self.seq
        self.seq += 1
        if seq == self.fail_at:
            if self.unknown:
                raise RuntimeError("REPLAY_TEST_UNKNOWN_FATAL")
            self.episode_fallback_count += 1
            self.reply_error = dict(rc=500, seq=seq,
                                    msg="Missing rvq act levels [0] in '<act_l1_77>'")
            return [0.9, 0, 0]
        return super().act(*args, **kwargs)
    def close(self):
        self.closed = True


class FakeBoundTeacher:
    def __init__(self, teacher, allow_released_fallback=False):
        self.teacher = teacher
        self.allow_released_fallback = allow_released_fallback
    def __getattr__(self, name):
        return getattr(self.teacher, name)
    def bind_environment(self, env):
        self.teacher.bind_environment(env)
    def act(self, *args, **kwargs):
        action = self.teacher.act(*args, **kwargs)
        if self.teacher.reply_error is not None:
            raise RuntimeError("TEACHER_TRANSPORT_FALLBACK")
        return action


class RuntimeV2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.prefix = []
        fixture = ZeroWorldFixture()
        for i in range(40):
            obs = fixture.set_step(i)
            state = dynamic_state(fixture.env)
            self.prefix.append(dict(
                step=i, observation_index=i, timestamp_s=fixture.time,
                simulator_world_time_s=fixture.time, rgb_sha256=rgb_hash(obs[RGB_KEY]),
                dynamic_state=state, dynamic_state_sha256=canonical_sha256(state),
                action=[-.2, 0, 0], normalized_action=[-.2, 0, 0], owner="student",
                action_timing="before_env_step", post_state_recorded=False))
        self.plan = dict(experiment=v1.EXPERIMENT, protocol_sha256=v1.PROTOCOL_SHA,
                         checkpoint=dict(sha256="c"*64, step=59716))
        self.entry = dict(task="stt", key="scene/1")

    def metadata(self, teacher="lightnav", repeat=False):
        return dict(
            experiment=v2.EXPERIMENT, partition="evaluation_adaptation",
            teacher=teacher, task="stt", key="scene/1", seed=7,
            protocol_sha256=v2.PROTOCOL_SHA, prefix_sha256=canonical_sha256(self.prefix),
            verification_only=repeat, camera_alignment_verified=True,
            plan_sha256="a"*64, base_plan_sha256="a"*64,
            base_protocol_sha256=v1.PROTOCOL_SHA, continuation_sha256="b"*64)

    def recorder(self, fail_at=32, k=35, repeat=False):
        fixture = ZeroWorldFixture()
        raw = RawTeacher(fail_at)
        boundary = FailureStateTeacherBoundary(FakeBoundTeacher(raw), teacher_name="lightnav")
        agent = ReplayThenTeacher(boundary, self.prefix, k, RGB_KEY)
        agent.bind_environment(fixture.env)
        rec = FailureStateRecorder(self.root/"branch", self.metadata(repeat=repeat),
                                   lambda: fixture.env, k, expected_prefix=self.prefix,
                                   agent=agent, experiment=v2.EXPERIMENT)
        return rec, fixture, agent, raw

    def drive_error(self, rec, fixture, agent, fail_at):
        for i in range(fail_at+1):
            obs = fixture.set_step(i)
            rec.observe(obs, fixture.detector(), fixture.robot, fixture.human,
                        fixture.episode, i, 10, fixture.time)
            try:
                action = agent.act(obs, fixture.detector(), "1")
            except TeacherOutputInvalid as error:
                return rec.finish_teacher_error(error)
            rec.record_action(i, action, None)
        self.fail("typed RVQ error expected")

    def test_runtime_identity_keeps_base_v1(self):
        self.assertEqual(runtime_contract(self.plan), (v1.EXPERIMENT, v1.PROTOCOL_SHA))
        original = copy.deepcopy(self.plan)
        self.assertEqual(runtime_contract(self.plan, experiment=v2.EXPERIMENT,
                                         protocol_sha=v2.PROTOCOL_SHA),
                         (v2.EXPERIMENT, v2.PROTOCOL_SHA))
        self.assertEqual(self.plan, original)
        for exp, sha in ((v2.EXPERIMENT, None), (v2.EXPERIMENT, v1.PROTOCOL_SHA),
                         ("foreign", v2.PROTOCOL_SHA)):
            with self.assertRaises(ValueError):
                runtime_contract(self.plan, experiment=exp, protocol_sha=sha)

    def test_relabelled_base_plan_rejected(self):
        plan = dict(self.plan, experiment=v2.EXPERIMENT, protocol_sha256=v2.PROTOCOL_SHA)
        with self.assertRaises(ValueError):
            runtime_contract(plan, experiment=v2.EXPERIMENT, protocol_sha=v2.PROTOCOL_SHA)

    def test_recorder_requires_explicit_v2_pin(self):
        with self.assertRaises(ValueError):
            FailureStateRecorder(self.root/"wrong", self.metadata(), lambda: None)
        meta = self.metadata(); meta["protocol_sha256"] = v1.PROTOCOL_SHA
        with self.assertRaises(ValueError):
            FailureStateRecorder(self.root/"wrong", meta, lambda: None, experiment=v2.EXPERIMENT)

    def test_actual_32_action_error_persists_no_terminal(self):
        rec, fixture, agent, raw = self.recorder()
        doc = self.drive_error(rec, fixture, agent, 32)
        self.assertEqual((len(rec.replay), len(rec.actions), agent.step), (33, 32, 32))
        self.assertEqual(doc["observed_partial"]["verified_prefix_frames"], 33)
        self.assertIsNone(doc["observed_partial"]["actual_takeover"])
        self.assertFalse(doc["replay_verified"])
        self.assertIsNone(doc["result"])
        self.assertTrue(raw.reply_error)
        self.assertEqual(validate_teacher_error(doc, prefix=self.prefix), doc)
        context = json.loads((rec.root/"teacher_error_context.json").read_text())
        self.assertEqual(context["metadata"]["experiment"], v2.EXPERIMENT)
        self.assertEqual(context["raw_reply"]["seq"], 32)
        for name in ("result.json", "complete.json", "windows.json", "admission.json"):
            self.assertFalse((rec.root/name).exists(), name)
        self.assertTrue((rec.root/"partial_observations.json").exists())

    def test_error_at_k_has_observed_not_faked_takeover(self):
        rec, fixture, agent, _ = self.recorder(fail_at=3, k=3)
        doc = self.drive_error(rec, fixture, agent, 3)
        self.assertEqual(doc["observed_partial"]["actual_takeover"]["step"], 3)
        self.assertFalse(doc["replay_verified"])

    def test_repeat_error_no_windows_and_explicit_repeat(self):
        rec, fixture, agent, _ = self.recorder(fail_at=5, k=3, repeat=True)
        doc = self.drive_error(rec, fixture, agent, 5)
        self.assertTrue(doc["verification_only"])
        self.assertEqual(doc["candidate_windows"], 0)
        self.assertEqual(doc["observed_partial"]["failure_phase"], "after_takeover")

    def test_typed_error_without_binding_rejected(self):
        rec, _, _, _ = self.recorder()
        with self.assertRaises(ValueError):
            rec.finish_teacher_error(TeacherOutputInvalid(rc=500, seq=0, message="fake"))

    def test_normal_v2_finish_preserves_proof_and_real_windows(self):
        rec, fixture, agent, _ = self.recorder(fail_at=None, k=3)
        for i in range(12):
            obs = fixture.set_step(i)
            rec.observe(obs, fixture.detector(), fixture.robot, fixture.human,
                        fixture.episode, i, 10, fixture.time)
            rec.record_action(i, agent.act(obs, fixture.detector(), "1"), None)
        doc = rec.finish(dict(success=True, collision=False,
                              policy_init_valid=True, following_rate=.9))
        self.assertEqual(doc["experiment"], v2.EXPERIMENT)
        self.assertTrue(doc["replay_verified"])
        self.assertGreater(doc["candidate_windows"], 0)
        complete = json.loads((rec.root/"complete.json").read_text())
        self.assertEqual(complete["schema"], v2.EXPERIMENT)
        self.assertFalse(complete["training_released"])

    def runtime_stub(self, raw, experiment=v2.EXPERIMENT):
        runtime = Runtime.__new__(Runtime)
        runtime.experiment = experiment
        runtime.protocol_sha = v2.PROTOCOL_SHA if experiment==v2.EXPERIMENT else v1.PROTOCOL_SHA
        runtime.continuation_sha = "b"*64
        runtime.plan, runtime.plan_sha = self.plan, "a"*64
        runtime.ready = dict(url="unused")
        runtime.manifest = dict(tasks=dict(stt=dict(sha256="d"*64)))
        runtime.repairs = {}
        runtime.config = SimpleNamespace(habitat=SimpleNamespace(task=SimpleNamespace(
            actions=SimpleNamespace(agent_1_base_velocity=None))))
        runtime.dataset = SimpleNamespace(episodes=[])
        runtime.bench = self.root
        episode = ZeroWorldFixture().episode
        episode.scene_id, episode.episode_id = "scene.glb", "1"
        runtime.check_entry = lambda entry: episode
        fixture = ZeroWorldFixture()
        def evaluate(config, subset, metrics, agent_factory, recorder):
            agent = agent_factory(None)
            agent.bind_environment(fixture.env)
            agent.reset()
            for i in range(33):
                obs = fixture.set_step(i)
                recorder.observe(obs, fixture.detector(), fixture.robot, fixture.human,
                                 fixture.episode, i, 10, fixture.time)
                action = agent.act(obs, fixture.detector(), "1")
                recorder.record_action(i, action, None)
        modules = {
            "torch": SimpleNamespace(manual_seed=lambda _: None,
                                     cuda=SimpleNamespace(manual_seed_all=lambda _: None)),
            "trained_agent": SimpleNamespace(evaluate_agent=evaluate),
            "lightnav_transport_20260927.agent": SimpleNamespace(Agent=lambda *args: raw),
            "wa.wm.diagnostic_agent": SimpleNamespace(DiagnosticAgent=None),
            "wa.wm.initial_bbox_repair_agent": SimpleNamespace(InitialBBoxRepairAgent=None),
            "wa.wm.dual_teacher_collect": SimpleNamespace(BoundTeacher=FakeBoundTeacher),
            "wa.wm.oracle_teacher": SimpleNamespace(OracleTeacher=None),
            "wa.wm.semantic_scene": SimpleNamespace(prepare_episode=lambda ep, bench: ep),
        }
        return runtime, modules

    def test_runtime_typed_error_returns_invalid_and_closes_teacher(self):
        raw = RawTeacher(32)
        runtime, modules = self.runtime_stub(raw)
        with patch.dict(sys.modules, modules):
            result = runtime.run(self.entry, self.root, "runtime", "unused",
                                 teacher_name="lightnav", prefix=self.prefix, takeover_step=35)
        self.assertEqual(result["status"], "teacher_output_invalid")
        self.assertTrue(raw.closed)
        self.assertFalse((self.root/"runtime/ERROR.json").exists())

    def test_runtime_unknown_error_stays_fatal(self):
        raw = RawTeacher(32, unknown=True)
        runtime, modules = self.runtime_stub(raw)
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(RuntimeError, "UNKNOWN_FATAL"):
            runtime.run(self.entry, self.root, "runtime", "unused",
                        teacher_name="lightnav", prefix=self.prefix, takeover_step=35)
        self.assertTrue(raw.closed)
        self.assertTrue((self.root/"runtime/ERROR.json").exists())
        self.assertFalse((self.root/"runtime/teacher_error.json").exists())

    def test_runtime_v1_rvq_stays_fatal(self):
        runtime, modules = self.runtime_stub(RawTeacher(32), experiment=v1.EXPERIMENT)
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(RuntimeError, "TEACHER_TRANSPORT_FALLBACK"):
            runtime.run(self.entry, self.root, "runtime", "unused",
                        teacher_name="lightnav", prefix=self.prefix, takeover_step=35)
        self.assertFalse((self.root/"runtime/teacher_error.json").exists())

    def test_developer_error_not_a_passed_check(self):
        runtime, modules = self.runtime_stub(RawTeacher(32))
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(RuntimeError, "not a passed"):
            runtime.run(self.entry, self.root, "runtime", "unused", teacher_name="lightnav",
                        prefix=self.prefix, takeover_step=35, development_actions=40)
        self.assertTrue((self.root/"runtime/teacher_error.json").exists())
        self.assertTrue((self.root/"runtime/ERROR.json").exists())


class EntryWorkloadTests(unittest.TestCase):
    def test_overlay_preserves_entries_and_does_not_relabel(self):
        entries = [dict(task="stt", key=f"scene/{i}") for i in range(126)]
        plan = dict(entries=entries)
        overlay = dict(base_plan=dict(path="/base.json", sha256="a"*64),
                       protocol_sha256=v1.PROTOCOL_SHA, expected_count=126, new_count=125,
                       reused_count=1, remaining_entries=entries[1:],
                       remaining_lanes=[entries[1:][i::8] for i in range(8)])
        original = copy.deepcopy(overlay)
        with patch("wa.wm.failure_state_collect_v2.load_plan", return_value=plan), \
             patch("wa.wm.failure_state_collect_v2.load_continuation", return_value=overlay):
            _, _, selected = load_workload("/base.json", "a"*64, "/overlay.json", "b"*64, 0)
            self.assertEqual(selected, overlay["remaining_lanes"][0])
            with self.assertRaises(ValueError):
                load_workload("/base.json", "a"*64, "/overlay.json", "b"*64, 0, "scene/0")
        self.assertEqual(overlay, original)

    def test_formal_entry_delegates_bounded_v2_search(self):
        calls = []
        prefix = [{"observed": True}]
        def run(entry, base, name, url, **kwargs):
            calls.append((name, kwargs))
            return dict(prefix=copy.deepcopy(prefix), result={"actual": True})
        runtime = SimpleNamespace(run=run)
        def search(student, actual_prefix, execute):
            self.assertEqual(actual_prefix, prefix)
            self.assertNotIn("prefix", student)
            return {"outcome": "mocked_cpu_only"}
        with patch("wa.wm.failure_state_collect_v2.search_recovery", side_effect=search):
            result = execute_entry(runtime, {"key":"scene/1"}, Path("/unused"), "unused")
        self.assertEqual(result["outcome"], "mocked_cpu_only")
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
