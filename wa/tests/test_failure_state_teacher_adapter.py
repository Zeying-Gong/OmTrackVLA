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


class MissingFinalLevelTests(unittest.TestCase):
    RAW = "<apos_1273><opos_0><act_l0_167><act_l1_69><opos_26><|im_end|>"
    PREFIX = "got 2 act levels, expected 3 from "

    def message(self, raw=None):
        return self.PREFIX + repr(self.RAW if raw is None else raw)

    def bound(self, message=None, **kwargs):
        return BoundFixture(error=kwargs.pop("error", RuntimeError("TEACHER_TRANSPORT_FALLBACK")),
                            reply=kwargs.pop("reply", dict(rc=500, seq=32, msg=self.message() if message is None else message)),
                            **kwargs)

    def reject(self, message, **kwargs):
        b = self.bound(message)
        w = FailureStateTeacherBoundary(b, teacher_name="lightnav",
                                       allow_missing_final_level=True, **kwargs)
        with self.assertRaises(RuntimeError) as got:
            w.act()
        self.assertIs(got.exception, b.error)
        self.assertEqual(b.calls, 1)

    def test_observed_message_requires_explicit_opt_in(self):
        for kwargs in ({}, {"allow_missing_final_level": False}):
            b = self.bound()
            with self.subTest(kwargs=kwargs), self.assertRaises(RuntimeError) as got:
                FailureStateTeacherBoundary(b, teacher_name="lightnav", **kwargs).act()
            self.assertIs(got.exception, b.error)
        b = self.bound()
        w = FailureStateTeacherBoundary(b, teacher_name="lightnav", allow_missing_final_level=True)
        executed = []
        with self.assertRaises(TeacherOutputInvalid) as got:
            executed.append(w.act())
        self.assertEqual(executed, [])
        self.assertEqual(b.calls, 1)
        self.assertEqual(got.exception.evidence["message"], self.message())
        self.assertEqual(got.exception.evidence["seq"], 32)
        self.assertIs(got.exception.__cause__, b.error)

    def test_opt_in_is_strict_bool(self):
        for value in (0, 1, "true", None, [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                FailureStateTeacherBoundary(BoundFixture(), teacher_name="lightnav",
                                            allow_missing_final_level=value)

    def test_canonical_single_and_double_quote_repr(self):
        for raw in (self.RAW, "it's " + self.RAW):
            with self.subTest(raw=raw), self.assertRaises(TeacherOutputInvalid):
                FailureStateTeacherBoundary(self.bound(self.message(raw)), teacher_name="lightnav",
                                            allow_missing_final_level=True).act()

    def test_wrong_count_text_or_extra_text_is_fatal(self):
        message = self.message()
        for msg in (message.replace("got 2", "got 1"), message.replace("got 2", "got 3"),
                    message.replace("got 2", "got 02"), message.replace("expected 3", "expected 2"),
                    message.replace("expected 3", "expected 4"), "prefix " + message,
                    message + " trailing", message + "\n"):
            with self.subTest(message=msg):
                self.reject(msg)

    def test_unsafe_non_string_or_noncanonical_repr_is_fatal(self):
        literals = ["unquoted", "'unterminated", repr(self.RAW.encode()),
                    repr([self.RAW]), repr({"raw": self.RAW}), "123", "True", "None",
                    "__import__('os').getcwd()", repr(self.RAW) + " + ''",
                    repr(self.RAW) + " ''", "(" + repr(self.RAW) + ")",
                    '"' + self.RAW + '"', repr(self.RAW) + " # comment",
                    repr(self.RAW + "x" * 4096)]
        for literal in literals:
            with self.subTest(literal=literal[:120]):
                self.reject(self.PREFIX + literal)

    def test_wrong_duplicate_gapped_or_malformed_tokens_are_fatal(self):
        values = ("", "<act_l0_1>", "<act_l1_1>", "<act_l0_1><act_l2_2>",
                  "<act_l1_1><act_l2_2>", "<act_l0_1><act_l1_2><act_l2_3>",
                  "<act_l0_1><act_l0_2><act_l1_3>", "<act_l0_1><act_l1_2><act_l1_3>",
                  "<act_l0_1><act_l0_2>", "<act_l00_1><act_l1_2>",
                  "<act_l0_1><act_l01_2>", "<act_l0_-1><act_l1_2>",
                  "<act_l0_1><act_l1_2><act_l2_bad>", "<act_l0_1><act_l1_2><act_l",
                  "<act_l0_1><act_l1_2><act_l2_3")
        for raw in values:
            with self.subTest(raw=raw):
                self.reject(self.message(raw))

    def test_all_existing_transport_guards_remain_required(self):
        missing = object()

        class GuardFixture(BoundFixture):
            def __init__(self, before, after, reply):
                super().__init__(RuntimeError("TEACHER_TRANSPORT_FALLBACK"), reply)
                self.after = after
                self.change(before)

            def change(self, values):
                for name, value in values.items():
                    if value is missing:
                        if hasattr(self.teacher, name):
                            delattr(self.teacher, name)
                    else:
                        setattr(self.teacher, name, value)

            def act(self, *args, **kwargs):
                self.calls += 1
                self.change(dict(seq=33, episode_fallback_count=1,
                                 reply_error=copy.deepcopy(self.reply)))
                self.change(self.after)
                raise self.error

        good = dict(rc=500, seq=32, msg=self.message())
        cases = []
        for rc in (0, 400, "500", 500.0, True, None):
            cases.append(({}, {}, dict(good, rc=rc)))
        for seq in (31, 33, -1, True, 32.0, "32", None):
            cases.append(({}, {}, dict(good, seq=seq)))
        for seq in (missing, None, True, 32.0, -1, 31):
            cases.append(({"seq": seq}, {}, good))
        for seq in (missing, None, True, 33.0, 32, 34):
            cases.append(({}, {"seq": seq}, good))
        for count in (None, True, 0.0, -1, 1):
            cases.append(({"episode_fallback_count": count}, {}, good))
        for count in (missing, None, True, 1.0, -1, 0, 2):
            cases.append(({}, {"episode_fallback_count": count}, good))
        for reply in (missing, None, [], {}):
            cases.append(({}, {"reply_error": reply}, good))
        for before, after, reply in cases:
            b = GuardFixture(before, after, reply)
            with self.subTest(before=before, after=after, reply=reply):
                with self.assertRaises(RuntimeError) as got:
                    FailureStateTeacherBoundary(b, teacher_name="lightnav",
                                                allow_missing_final_level=True).act()
                self.assertIs(got.exception, b.error)
                self.assertEqual(b.calls, 1)
        # The real client may have no counter before its first-ever act.
        b = GuardFixture({"episode_fallback_count": missing}, {}, good)
        with self.assertRaises(TeacherOutputInvalid):
            FailureStateTeacherBoundary(b, teacher_name="lightnav",
                                        allow_missing_final_level=True).act()

    def test_legacy_success_and_unknown_failures_unchanged_with_opt_in(self):
        for enabled in (False, True):
            for name in ("lightnav", "oracle"):
                b = BoundFixture()
                w = FailureStateTeacherBoundary(b, teacher_name=name, allow_missing_final_level=enabled)
                self.assertIs(w.act(), b.action)
                self.assertEqual(b.calls, 1)
            with self.assertRaises(TeacherOutputInvalid):
                FailureStateTeacherBoundary(self.bound(MESSAGE), teacher_name="lightnav",
                                            allow_missing_final_level=enabled).act()
        b = self.bound()
        with self.assertRaises(RuntimeError) as got:
            FailureStateTeacherBoundary(b, teacher_name="oracle", allow_missing_final_level=True).act()
        self.assertIs(got.exception, b.error)
        class Unknown(RuntimeError):
            pass
        for error in (RuntimeError("CUDA OOM"), RuntimeError("REPLAY_DYNAMIC_STATE_DIVERGED"),
                      ValueError("bad"), Unknown("TEACHER_TRANSPORT_FALLBACK")):
            b = self.bound(error=error)
            with self.subTest(error=error), self.assertRaises(type(error)) as got:
                FailureStateTeacherBoundary(b, teacher_name="lightnav",
                                            allow_missing_final_level=True).act()
            self.assertIs(got.exception, error)


if __name__ == "__main__":
    unittest.main()
