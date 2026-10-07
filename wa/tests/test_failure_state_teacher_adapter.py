"""CPU checks for the narrow boundary; real runtime validation is separate."""
import copy
import unittest
from types import SimpleNamespace

from wa.wm.failure_state_teacher_adapter import FailureStateTeacherBoundary
from wa.wm.failure_state_teacher_error import TeacherOutputInvalid

MESSAGE = "Missing rvq act levels [0] in '<apos_1249><opos_0>chu<act_l1_77><|im_end|>'"


class BoundFixture:
    allow_released_fallback = False
    def __init__(self, error=None, reply=None):
        self.teacher = SimpleNamespace(seq=32, episode_fallback_count=0, reply_error=None)
        self.error = error
        self.reply = reply
        self.calls = 0
        self.action = [0.1, 0.2, 0.3]
        self.environment = None

    def bind_environment(self, env):
        self.environment = env

    def reset(self, *args):
        self.teacher.seq = 0

    def act(self, *args, **kwargs):
        self.calls += 1
        if self.error is not None:
            self.teacher.reply_error = copy.deepcopy(self.reply)
            self.teacher.seq += 1
            self.teacher.episode_fallback_count += 1
            raise self.error
        return self.action


class BoundaryTests(unittest.TestCase):
    def bound(self, **overrides):
        args = dict(error=RuntimeError("TEACHER_TRANSPORT_FALLBACK"),
                    reply=dict(rc=500, seq=32, msg=MESSAGE))
        args.update(overrides)
        return BoundFixture(**args)

    def test_actual_observed_decode_error_classified_without_returned_action(self):
        b = self.bound()
        wrapper = FailureStateTeacherBoundary(b, teacher_name="lightnav")
        executed = []
        with self.assertRaises(TeacherOutputInvalid) as caught:
            executed.append(wrapper.act({}, None, "8", "instruction"))
        self.assertEqual(executed, [])
        self.assertEqual(b.calls, 1)
        self.assertEqual(caught.exception.evidence["seq"], 32)
        self.assertEqual(caught.exception.evidence["rc"], 500)
        self.assertEqual(caught.exception.evidence["message"], MESSAGE)
        self.assertIs(caught.exception.__cause__, b.error)

    def test_valid_output_identity_and_single_call_unchanged(self):
        for teacher in ("lightnav", "oracle"):
            b = BoundFixture()
            w = FailureStateTeacherBoundary(b, teacher_name=teacher)
            self.assertIs(w.act({}, None, "8"), b.action)
            self.assertEqual(b.calls, 1)
            env = object()
            w.bind_environment(env)
            self.assertIs(w.environment, env)
            w.reset(None)
            self.assertEqual(b.teacher.seq, 0)

    def test_unknown_error_not_reclassified(self):
        for e in (RuntimeError("REPLAY_DYNAMIC_STATE_DIVERGED"),
                  RuntimeError("Nonfinite released LightNav action"), ValueError("bad")):
            b = self.bound(error=e)
            w = FailureStateTeacherBoundary(b, teacher_name="lightnav")
            with self.assertRaises(type(e)) as got:
                w.act({}, None, "8")
            self.assertIs(got.exception, e)

    def test_oracle_error_not_classified_as_lightnav_response(self):
        b = self.bound()
        w = FailureStateTeacherBoundary(b, teacher_name="oracle")
        with self.assertRaises(RuntimeError) as got:
            w.act()
        self.assertIs(got.exception, b.error)

    def test_runtime_subclass_not_caught(self):
        class Unknown(RuntimeError):
            pass
        b = self.bound(error=Unknown("TEACHER_TRANSPORT_FALLBACK"))
        with self.assertRaises(Unknown):
            FailureStateTeacherBoundary(b, teacher_name="lightnav").act()

    def test_foreign_missing_or_unknown_response_fatal(self):
        for reply in (None, {}, {"rc":0,"seq":32,"msg":MESSAGE},
                      {"rc":500,"seq":31,"msg":MESSAGE}, {"rc":500,"seq":33,"msg":MESSAGE},
                      {"rc":500,"seq":True,"msg":MESSAGE},
                      {"rc":500,"seq":32,"msg":"CUDA OOM"},
                      {"rc":500,"seq":32,"msg":"Missing rvq act levels [] in x"}):
            b = self.bound(reply=reply)
            with self.subTest(reply=reply):
                with self.assertRaises(RuntimeError) as got:
                    FailureStateTeacherBoundary(b, teacher_name="lightnav").act()
                self.assertIs(got.exception, b.error)

    def test_unproven_fallback_increment_fatal(self):
        b = self.bound()
        b.teacher.episode_fallback_count = True
        with self.assertRaises(RuntimeError):
            FailureStateTeacherBoundary(b, teacher_name="lightnav").act()

    def test_unsafe_constructor_rejected(self):
        b = BoundFixture()
        b.allow_released_fallback = True
        with self.assertRaises(ValueError):
            FailureStateTeacherBoundary(b, teacher_name="lightnav")
        with self.assertRaises(ValueError):
            FailureStateTeacherBoundary(BoundFixture(), teacher_name="unknown")


if __name__ == "__main__":
    unittest.main()
