"""Adversarial asynchronous-stream and TTL tests; CPU only, no I/O control."""
import importlib.util
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from active_search_fsm import ActiveSearchFSM, IdentityEvidence, MotionPermission, State, ZERO

spec = importlib.util.spec_from_file_location("active_search_review_fixture", ROOT/"tests/test_active_search_fsm.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
config, observation, lost = fixtures.config, fixtures.observation, fixtures.lost


def searching(**changes):
    fsm = ActiveSearchFSM(config(**changes))
    fsm.step(observation(0.))
    fsm.step(lost(.1))
    assert fsm.step(lost(.31)).state == State.SEARCH
    return fsm


class AsyncBoundaryTests(unittest.TestCase):
    def test_newer_deny_cannot_be_overridden_by_older_fresh_allow(self):
        fsm = searching()
        self.assertEqual(fsm.step(lost(.42, motion_permission=MotionPermission.DENY)).command, ZERO)
        decision = fsm.step(lost(.43, permission_timestamp_s=.40))
        self.assertEqual(decision.command, ZERO, "newer DENY must supersede old ALLOW within the age window")

    def test_newer_unknown_cannot_be_overridden_by_older_fresh_allow(self):
        fsm = searching()
        fsm.step(lost(.42, motion_permission=MotionPermission.UNKNOWN))
        decision = fsm.step(lost(.43, permission_timestamp_s=.40))
        self.assertEqual(decision.command, ZERO)

    def test_equal_permission_timestamp_cannot_change_deny_to_allow(self):
        fsm = searching()
        fsm.step(lost(.42, motion_permission=MotionPermission.DENY))
        decision = fsm.step(lost(.43, permission_timestamp_s=.42))
        self.assertEqual(decision.command, ZERO, "conflicting decisions at one permit timestamp cannot restore ALLOW")

    def test_deny_during_rgb_failure_still_rejects_later_older_allow(self):
        fsm = searching()
        fsm.step(lost(.42, rgb_available=False, motion_permission=MotionPermission.DENY))
        decision = fsm.step(lost(.43, permission_timestamp_s=.40))
        self.assertEqual(decision.command, ZERO, "an unrelated sensor guard must not discard a permit revocation")

    def test_deny_during_policy_stop_or_missing_initialization_is_remembered(self):
        for changes in ({"stop_probability": .9}, {"initialization_id": None}):
            with self.subTest(changes=changes):
                fsm = searching()
                fsm.step(lost(.42, motion_permission=MotionPermission.DENY, **changes))
                self.assertEqual(fsm.step(lost(.43, permission_timestamp_s=.40)).command, ZERO)

    def test_duplicate_control_tick_does_not_discard_same_stamp_revocation(self):
        fsm = searching()
        fsm.step(lost(.31, motion_permission=MotionPermission.DENY))
        self.assertEqual(fsm.step(lost(.32, permission_timestamp_s=.31)).command, ZERO)

    def test_newer_allow_resumes_without_refunding_budget(self):
        fsm = searching()
        held = fsm.step(lost(.42, motion_permission=MotionPermission.DENY))
        resumed = fsm.step(lost(.43))
        self.assertEqual(resumed.state, State.SEARCH)
        self.assertGreater(resumed.budget.search_reserved_s, held.budget.search_reserved_s)

    def test_same_stamp_conflict_requires_strictly_newer_permission(self):
        fsm = searching()
        fsm.step(lost(.42, motion_permission=MotionPermission.DENY))
        for t, permission in ((.43, MotionPermission.ALLOW), (.44, MotionPermission.DENY), (.45, MotionPermission.ALLOW)):
            self.assertEqual(fsm.step(lost(t, motion_permission=permission, permission_timestamp_s=.42)).command, ZERO)
        self.assertEqual(fsm.step(lost(.46)).state, State.SEARCH)

    def test_same_allow_timestamp_is_reusable_only_until_original_expiry(self):
        fsm = searching()
        reused = fsm.step(lost(.4, permission_timestamp_s=.31))
        self.assertEqual(reused.state, State.SEARCH)
        self.assertAlmostEqual(reused.command_valid_until_s, .46)
        self.assertEqual(fsm.step(lost(.46, permission_timestamp_s=.31)).command, ZERO)

    def test_invalid_future_deny_does_not_poison_permission_watermark(self):
        fsm = searching()
        self.assertEqual(fsm.step(lost(.42, motion_permission=MotionPermission.DENY, permission_timestamp_s=99.)).command, ZERO)
        self.assertEqual(fsm.step(lost(.43)).state, State.SEARCH)

    def test_buffered_near_simultaneous_frames_cannot_fake_confirmation_duration(self):
        fsm = searching()
        fsm.step(observation(.4))
        fsm.step(observation(.55, rgb_timestamp_s=.401, prediction_rgb_timestamp_s=.401))
        decision = fsm.step(observation(.71, rgb_timestamp_s=.402, prediction_rgb_timestamp_s=.402))
        self.assertFalse(decision.tracking_ready, "0.002 s of captured evidence cannot meet 0.3 s confirmation")
        self.assertEqual(decision.command, ZERO)

    def test_capture_gap_breaks_confirmation_even_if_callback_gap_is_small(self):
        fsm = searching()
        fsm.step(observation(1., rgb_timestamp_s=.6, prediction_rgb_timestamp_s=.6))
        fsm.step(observation(1.1, rgb_timestamp_s=.99, prediction_rgb_timestamp_s=.99))
        decision = fsm.step(observation(1.31, rgb_timestamp_s=1.1, prediction_rgb_timestamp_s=1.1))
        self.assertFalse(decision.tracking_ready, "captured-frame gap exceeded the 0.25 s continuity limit")
        self.assertEqual(decision.confirmation_frames, 2)

    def test_repeated_strong_capture_cannot_refresh_target_memory_clock(self):
        fsm = ActiveSearchFSM(config(max_target_memory_age_s=.4))
        fsm.step(observation(0.))
        fsm.step(observation(.2, rgb_timestamp_s=0., prediction_rgb_timestamp_s=0.))
        fsm.step(lost(.21))
        decision = fsm.step(lost(.42))
        self.assertEqual(decision.command, ZERO, "capture at 0.0 expires at 0.4 regardless of repeated callback")

    def test_delayed_initial_strong_capture_cannot_extend_memory_deadline(self):
        fsm = ActiveSearchFSM(config(max_target_memory_age_s=.4))
        fsm.step(observation(.3, rgb_timestamp_s=0., prediction_rgb_timestamp_s=0.))
        fsm.step(lost(.31))
        decision = fsm.step(lost(.52))
        self.assertEqual(decision.command, ZERO)

    def test_new_strong_capture_renews_memory_to_capture_deadline(self):
        fsm = ActiveSearchFSM(config(max_target_memory_age_s=.4, command_ttl_s=.2))
        fsm.step(observation(0.))
        fsm.step(observation(.25, rgb_timestamp_s=.2, prediction_rgb_timestamp_s=.2))
        fsm.step(lost(.26))
        decision = fsm.step(lost(.47))
        self.assertEqual(decision.state, State.SEARCH)
        self.assertAlmostEqual(decision.command_valid_until_s, .6)

    def test_delayed_real_duration_can_reacquire_but_uses_capture_memory_age(self):
        fsm = searching(max_target_memory_age_s=.4)
        for now, capture in ((.6, .4), (.76, .56), (.92, .72)):
            decision = fsm.step(observation(now, rgb_timestamp_s=capture, prediction_rgb_timestamp_s=capture))
        self.assertTrue(decision.tracking_ready)
        fsm.step(lost(.93))
        self.assertEqual(fsm.step(lost(1.14)).command, ZERO)

    def test_search_reservations_and_all_command_expiries_remain_bounded(self):
        cfg = config(max_search_reversals=8)
        fsm = ActiveSearchFSM(cfg)
        fsm.step(observation(0.))
        fsm.step(lost(.1))
        generator = random.Random(7104)
        now, summed_time, summed_yaw, issued = .3, 0., 0., 0
        for _ in range(100):
            now += generator.uniform(.001, .09)
            rgb = now-generator.uniform(0., .12)
            # Keep captures monotonic for this budget test.
            rgb = max(rgb, now-.001)
            stamp = now-generator.uniform(0., .08)
            decision = fsm.step(lost(now, rgb_timestamp_s=rgb, prediction_rgb_timestamp_s=rgb,
                                     permission_timestamp_s=stamp, suggested_yaw_rad_s=.3))
            if decision.command == ZERO:
                continue
            issued += 1
            ttl = decision.command_valid_until_s-now
            self.assertGreater(ttl, 0.)
            self.assertEqual((decision.command.forward_m_s, decision.command.lateral_m_s), (0., 0.))
            self.assertLessEqual(abs(decision.command.yaw_rad_s), cfg.max_yaw_rate_rad_s)
            for deadline in (now+cfg.command_ttl_s, rgb+cfg.max_rgb_age_s,
                             now+cfg.max_prediction_age_s, stamp+cfg.max_permission_age_s,
                             cfg.max_target_memory_age_s, now+decision.budget.lost_remaining_s):
                self.assertLessEqual(decision.command_valid_until_s, deadline+1e-12)
            summed_time += ttl
            summed_yaw += ttl*abs(decision.command.yaw_rad_s)
            self.assertLessEqual(summed_time, cfg.max_search_command_time_s+1e-12)
            self.assertLessEqual(summed_yaw, cfg.max_search_yaw_rad+1e-12)
        self.assertGreater(issued, 0)


if __name__ == "__main__":
    unittest.main()
