"""CPU-only regression tests for replay environment binding and invariants."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from wa.wm.recovery_replay import ReplayThenTeacher, rgb_hash


def state(step=0):
    return dict(timestamp=step * .05,
                agents=[dict(transform=np.eye(4).tolist(), joints=[0.])])


def fixture():
    images = [np.full((2, 2, 3), step, dtype=np.uint8) for step in range(3)]
    prefix = [dict(rgb_sha256=rgb_hash(rgb), action=[-.1 * (step + 1), .1, 0.],
                   dynamic_state=state(step))
              for step, rgb in enumerate(images)]
    return images, prefix


class LegacyTeacher:
    """Existing image teacher contract: no bind_environment method."""
    def __init__(self):
        self.calls = []
        self.last_trajectory = None

    def reset(self, *args, **kwargs):
        self.calls = []
        self.last_trajectory = None
        self.reset_arguments = (args, kwargs)
        return "reset-return"

    def act(self, observations, detector, episode_id, instruction=None):
        self.calls.append((episode_id, instruction))
        self.last_trajectory = [[0., 0.], [.2, .1]]
        return [.2, .3, .4]


class NeedsEnvironmentTeacher(LegacyTeacher):
    """Mock Oracle: binding is required and reset preserves it like OracleTeacher."""
    def __init__(self):
        super().__init__()
        self.environment = None
        self.bindings = []

    def bind_environment(self, env):
        self.environment = env
        self.bindings.append(env)

    def act(self, observations, detector, episode_id, instruction=None):
        if self.environment is None:
            raise RuntimeError("Oracle requires benchmark environment binding")
        return super().act(observations, detector, episode_id, instruction)


class RecoveryReplayBindingTests(unittest.TestCase):
    def test_required_oracle_environment_is_forwarded_by_identity(self):
        teacher = NeedsEnvironmentTeacher()
        agent = ReplayThenTeacher(teacher, [], 0, "rgb")
        with self.assertRaisesRegex(RuntimeError, "requires benchmark environment"):
            agent.act({"rgb": np.zeros((2, 2, 3), dtype=np.uint8)}, None, "ep")
        env = object()
        agent.bind_environment(env)
        self.assertIs(agent.environment, env)
        self.assertIs(teacher.environment, env)
        self.assertEqual(teacher.bindings, [env])
        self.assertEqual(agent.act({"rgb": None}, None, "ep"), [.2, .3, .4])

    def test_legacy_teacher_without_binder_remains_compatible(self):
        teacher = LegacyTeacher()
        agent = ReplayThenTeacher(teacher, [], 0, "rgb")
        env = object()
        agent.bind_environment(env)
        self.assertIs(agent.environment, env)
        self.assertFalse(hasattr(teacher, "bind_environment"))
        self.assertEqual(agent.act({"rgb": None}, None, 5), [.2, .3, .4])

    def test_reset_and_rebind_preserve_current_environment(self):
        teacher = NeedsEnvironmentTeacher()
        agent = ReplayThenTeacher(teacher, [], 0, "rgb")
        first, second = object(), object()
        agent.bind_environment(first)
        agent.act({"rgb": None}, None, 5)
        self.assertEqual(agent.reset("ep", flag=True), "reset-return")
        self.assertEqual(agent.step, 0)
        self.assertIsNone(agent.last_trajectory)
        self.assertIs(agent.environment, first)
        self.assertIs(teacher.environment, first)
        self.assertEqual(teacher.reset_arguments, (("ep",), {"flag": True}))
        agent.bind_environment(second)
        agent.reset()
        self.assertIs(agent.environment, second)
        self.assertIs(teacher.environment, second)
        self.assertEqual(teacher.bindings, [first, second])
        self.assertEqual(agent.act({"rgb": None}, None, 6), [.2, .3, .4])

    def test_prefix_executes_original_wa_actions_while_teacher_warms_up(self):
        images, prefix = fixture()
        saved = deepcopy(prefix)
        teacher = NeedsEnvironmentTeacher()
        agent = ReplayThenTeacher(teacher, prefix, 2, "rgb")
        env = SimpleNamespace(snapshot=state())
        agent.bind_environment(env)
        with patch("wa.wm.recovery_replay.dynamic_state", side_effect=lambda actual: actual.snapshot) as read:
            for step in range(3):
                env.snapshot = state(step)
                action = agent.act({"rgb": images[step]}, None, "ep", "target")
                self.assertEqual(action, prefix[step]["action"] if step < 2 else [.2, .3, .4])
                self.assertEqual(len(teacher.calls), step + 1)
                if step < 2:
                    self.assertIsNone(agent.last_trajectory)
            self.assertEqual(read.call_count, 3)
            self.assertTrue(all(call.args[0] is env for call in read.call_args_list))
        self.assertEqual(prefix, saved)
        self.assertEqual(teacher.calls, [("ep", "target")] * 3)
        self.assertEqual(agent.last_trajectory, teacher.last_trajectory)

    def test_rgb_mismatch_fails_before_teacher_call_or_step_advance(self):
        images, prefix = fixture()
        teacher = NeedsEnvironmentTeacher()
        agent = ReplayThenTeacher(teacher, prefix, 2, "rgb")
        agent.bind_environment(SimpleNamespace(snapshot=state()))
        with patch("wa.wm.recovery_replay.dynamic_state") as read:
            with self.assertRaisesRegex(RuntimeError, "REPLAY_DIVERGED"):
                agent.act({"rgb": images[0] + 1}, None, "ep")
            read.assert_not_called()
        self.assertEqual(teacher.calls, [])
        self.assertEqual(agent.step, 0)

    def test_dynamic_time_and_joint_mismatch_fail_closed(self):
        for mismatch in ("timestamp", "joints"):
            with self.subTest(mismatch=mismatch):
                images, prefix = fixture()
                teacher = NeedsEnvironmentTeacher()
                agent = ReplayThenTeacher(teacher, prefix, 2, "rgb")
                wrong = state()
                if mismatch == "timestamp":
                    wrong["timestamp"] = .1
                else:
                    wrong["agents"][0]["joints"] = [1.]
                agent.bind_environment(SimpleNamespace(snapshot=wrong))
                with patch("wa.wm.recovery_replay.dynamic_state", side_effect=lambda actual: actual.snapshot):
                    with self.assertRaisesRegex(RuntimeError, "REPLAY_(TIME|DYNAMIC_STATE)_DIVERGED"):
                        agent.act({"rgb": images[0]}, None, "ep")
                self.assertEqual(teacher.calls, [])
                self.assertEqual(agent.step, 0)

    def test_takeover_frame_is_still_checked_before_teacher_execution(self):
        images, prefix = fixture()
        teacher = NeedsEnvironmentTeacher()
        agent = ReplayThenTeacher(teacher, prefix, 2, "rgb")
        env = SimpleNamespace(snapshot=state())
        agent.bind_environment(env)
        with patch("wa.wm.recovery_replay.dynamic_state", side_effect=lambda actual: actual.snapshot):
            for step in range(2):
                env.snapshot = state(step)
                agent.act({"rgb": images[step]}, None, "ep")
            env.snapshot = state(2)
            with self.assertRaisesRegex(RuntimeError, "REPLAY_DIVERGED at step 2"):
                agent.act({"rgb": images[2] + 1}, None, "ep")
        self.assertEqual(len(teacher.calls), 2)
        self.assertEqual(agent.step, 2)

    def test_binder_failure_is_not_swallowed(self):
        class RejectingTeacher(LegacyTeacher):
            def bind_environment(self, env):
                raise RuntimeError("binding rejected")
        agent = ReplayThenTeacher(RejectingTeacher(), [], 0, "rgb")
        with self.assertRaisesRegex(RuntimeError, "binding rejected"):
            agent.bind_environment(object())


if __name__ == "__main__":
    unittest.main()
