"""Behavioral tests of failure paths and cumulative search budgets (CPU only)."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omtrackvla.control.active_search_fsm import (
    ActiveSearchFSM, IdentityEvidence as Identity, MotionPermission as Permission,
    SearchConfig, SearchInput, State, ZERO,
)


def config(**changes):
    # Synthetic test values, deliberately not deployment recommendations.
    value = SearchConfig(
        lost_visibility_threshold=.5, reacquire_visibility_threshold=.9,
        stop_probability_threshold=.8, lost_confirm_s=.2,
        reacquire_confirm_s=.3, reacquire_min_frames=3, reacquire_max_gap_s=.25,
        max_rgb_age_s=.5, max_prediction_age_s=.5, max_permission_age_s=.15,
        max_target_memory_age_s=3., max_lost_time_s=2.,
        max_search_command_time_s=.8, max_search_yaw_rad=.3,
        max_search_reversals=1, max_yaw_rate_rad_s=.4, command_ttl_s=.1,
    )
    return replace(value, **changes)


def observation(t, **changes):
    value = SearchInput(
        now_s=t, initialization_id="user-initialization-1", rgb_available=True,
        rgb_timestamp_s=t, prediction_timestamp_s=t, prediction_rgb_timestamp_s=t,
        visibility_probability=.99, stop_probability=.01,
        identity=Identity.KNOWN_SAME, suggested_yaw_rad_s=1.,
        motion_permission=Permission.ALLOW, permission_timestamp_s=t,
    )
    return replace(value, **changes)


def lost(t, **changes):
    value = replace(observation(t), visibility_probability=.1, identity=Identity.UNKNOWN)
    return replace(value, **changes)


class ActiveSearchTests(unittest.TestCase):
    def setUp(self):
        self.fsm = ActiveSearchFSM(config())

    def start_search(self, fsm=None):
        fsm = fsm or self.fsm
        self.assertEqual(fsm.step(observation(0.)).state, State.TRACKING)
        self.assertEqual(fsm.step(lost(.1)).state, State.LOST_HOLD)
        result = fsm.step(lost(.31))
        self.assertEqual(result.state, State.SEARCH)
        return result

    def assert_stopped(self, result, reason=None):
        self.assertEqual(result.command, ZERO)
        self.assertIsNone(result.command_valid_until_s)
        self.assertFalse(result.tracking_ready)
        if reason:
            self.assertEqual(result.reason, reason)

    def test_config_has_no_defaults_or_nonfinite_values(self):
        with self.assertRaises(TypeError):
            SearchConfig()
        for field, value in (
            ("max_yaw_rate_rad_s", float("nan")), ("command_ttl_s", 0.),
            ("lost_visibility_threshold", .95), ("max_search_reversals", -1),
            ("reacquire_min_frames", 1), ("max_lost_time_s", True),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                config(**{field: value})

    def test_cold_no_target_never_searches_despite_motion_and_high_visibility(self):
        for t in (0., .4, 1., 5.):
            result = self.fsm.step(observation(t, initialization_id=None))
            self.assertEqual(result.state, State.UNINITIALIZED)
            self.assert_stopped(result, "cold_no_target")
            self.assertFalse(result.has_confirmed_tracking)
            self.assertEqual(result.budget.search_reserved_s, 0.)

    def test_default_unknown_permission_fails_closed(self):
        value = SearchInput(0., "init", True, 0., 0., 0., .99, .01,
                            Identity.KNOWN_SAME, 1.)
        self.assert_stopped(self.fsm.step(value), "motion_permission_unknown")

    def test_initialization_without_confirmed_tracking_is_not_search_authority(self):
        result = self.fsm.step(lost(0.))
        self.assert_stopped(result, "no_confirmed_tracking_history")
        self.assert_stopped(self.fsm.step(lost(.5)), "no_confirmed_tracking_history")

    def test_loss_immediately_stops_then_yaw_search_is_bounded_and_zero_translation(self):
        result = self.start_search()
        self.assertEqual(result.command.forward_m_s, 0.)
        self.assertEqual(result.command.lateral_m_s, 0.)
        self.assertEqual(result.command.yaw_rad_s, .4)
        self.assertEqual(result.command, result.suggestion)
        self.assertAlmostEqual(result.command_valid_until_s, .41)
        self.assertAlmostEqual(result.budget.search_reserved_s, .1)
        self.assertAlmostEqual(result.budget.yaw_reserved_rad, .04)

    def test_denied_unknown_and_stale_permissions_preserve_suggestion_but_send_stop(self):
        for changes, reason in (
            ({"motion_permission": Permission.DENY}, "motion_permission_deny"),
            ({"motion_permission": Permission.UNKNOWN}, "motion_permission_unknown"),
            ({"permission_timestamp_s": .1}, "stale_permission"),
            ({"permission_timestamp_s": None}, "missing_or_invalid_permission_timestamp"),
            ({"permission_timestamp_s": .5}, "future_permission_timestamp"),
        ):
            with self.subTest(reason=reason):
                fsm = ActiveSearchFSM(config())
                first = self.start_search(fsm)
                result = fsm.step(lost(.42, **changes))
                self.assert_stopped(result, reason)
                self.assertNotEqual(result.suggestion, ZERO)
                self.assertEqual(result.budget.search_reserved_s, first.budget.search_reserved_s)

    def test_missing_rgb_and_stale_predictions_outrank_reacquisition(self):
        for changes, reason in (
            ({"rgb_available": False}, "rgb_unavailable"),
            ({"rgb_timestamp_s": None}, "missing_or_invalid_rgb_timestamp"),
            ({"rgb_timestamp_s": -.2}, "stale_rgb"),
            ({"prediction_timestamp_s": -.2}, "stale_prediction"),
            ({"rgb_timestamp_s": .8}, "future_rgb_timestamp"),
            ({"prediction_timestamp_s": .8}, "future_prediction_timestamp"),
            ({"prediction_timestamp_s": .4}, "prediction_predates_rgb"),
        ):
            with self.subTest(reason=reason):
                fsm = ActiveSearchFSM(config())
                self.start_search(fsm)
                result = fsm.step(observation(.6, **changes))
                self.assert_stopped(result, reason)

    def test_nonfinite_and_out_of_range_predictions_stop_without_nonfinite_audit(self):
        for key, value, reason in (
            ("visibility_probability", float("nan"), "invalid_prediction_probability"),
            ("stop_probability", float("inf"), "invalid_prediction_probability"),
            ("visibility_probability", -1., "invalid_prediction_probability"),
            ("suggested_yaw_rad_s", float("nan"), "invalid_search_direction"),
        ):
            with self.subTest(key=key):
                fsm = ActiveSearchFSM(config())
                self.start_search(fsm)
                result = fsm.step(lost(.42, **{key: value}))
                self.assert_stopped(result, reason)
                json.dumps(asdict(result), allow_nan=False)

    def test_policy_stop_at_threshold_overrides_known_same_and_search(self):
        self.start_search()
        result = self.fsm.step(observation(.42, stop_probability=.8))
        self.assert_stopped(result, "policy_stop")

    def test_permission_at_exact_expiry_cannot_authorize_tracking_or_search(self):
        for value in (observation(.5, permission_timestamp_s=.25),
                      lost(.5, permission_timestamp_s=.25)):
            with self.subTest(visibility=value.visibility_probability):
                fsm = ActiveSearchFSM(config(max_permission_age_s=.25))
                self.start_search(fsm)
                self.assert_stopped(fsm.step(value), "stale_permission")

    def test_clock_rollback_and_nan_latch_until_explicit_task_reset(self):
        for bad_time, reason in ((.2, "monotonic_clock_regressed"),
                                 (float("nan"), "invalid_monotonic_clock")):
            with self.subTest(reason=reason):
                fsm = ActiveSearchFSM(config())
                self.start_search(fsm)
                result = fsm.step(lost(bad_time))
                self.assert_stopped(result, reason)
                json.dumps(asdict(result), allow_nan=False)
                self.assert_stopped(fsm.step(observation(1.)), reason)
                fsm.reset_task()
                self.assertEqual(fsm.step(observation(0.)).state, State.TRACKING)

    def test_identical_clock_cannot_emit_repeated_commands(self):
        first = self.start_search()
        result = self.fsm.step(lost(.31))
        self.assert_stopped(result, "monotonic_clock_did_not_advance")
        self.assertEqual(result.budget, first.budget)

    def test_sensor_timestamp_regression_fails_closed(self):
        self.start_search()
        result = self.fsm.step(lost(.42, rgb_timestamp_s=.2, prediction_rgb_timestamp_s=.2))
        self.assert_stopped(result, "rgb_timestamp_regressed")

    def test_newly_returned_prediction_for_old_rgb_is_rejected(self):
        self.start_search()
        result = self.fsm.step(lost(.42, prediction_rgb_timestamp_s=.31))
        self.assert_stopped(result, "prediction_rgb_mismatch")

    def test_initialization_cannot_change_or_clear_budget_implicitly(self):
        first = self.start_search()
        result = self.fsm.step(lost(.42, initialization_id="some-other-init"))
        self.assert_stopped(result, "initialization_changed_without_task_reset")
        self.assertEqual(result.initialization_id, "user-initialization-1")
        self.assertEqual(result.budget.search_reserved_s, first.budget.search_reserved_s)
        self.assert_stopped(self.fsm.step(observation(.5)), "initialization_changed_without_task_reset")

    def test_missing_initialization_preserves_target_and_does_not_become_cold(self):
        self.start_search()
        result = self.fsm.step(lost(.42, initialization_id=None))
        self.assert_stopped(result, "initialization_temporarily_unavailable")
        self.assertEqual(result.initialization_id, "user-initialization-1")
        self.assertTrue(result.has_confirmed_tracking)

    def test_reacquisition_requires_duration_and_distinct_continuous_same_target_frames(self):
        first = self.start_search()
        for t in (.4, .55):
            result = self.fsm.step(observation(t))
            self.assertEqual(result.state, State.REACQUIRE_CONFIRM)
            self.assert_stopped(result)
        result = self.fsm.step(observation(.71))
        self.assertEqual(result.state, State.TRACKING)
        self.assertTrue(result.tracking_ready)
        self.assertEqual(result.command, ZERO)  # Normal tracking is owned by dispatcher.
        self.assertEqual(result.budget.search_reserved_s, first.budget.search_reserved_s)
        self.assertGreater(result.budget.lost_elapsed_s, first.budget.lost_elapsed_s)

    def test_repeated_frame_cannot_complete_confirmation(self):
        fsm = ActiveSearchFSM(config(reacquire_min_frames=2))
        self.start_search(fsm)
        fsm.step(observation(.4))
        fsm.step(observation(.55))
        result = fsm.step(observation(.71, rgb_timestamp_s=.55, prediction_rgb_timestamp_s=.55))
        self.assertEqual(result.state, State.REACQUIRE_CONFIRM)
        self.assertEqual(result.confirmation_frames, 2)
        self.assert_stopped(result)

    def test_confirmation_gap_restarts_evidence_not_search_budget(self):
        first = self.start_search()
        self.fsm.step(observation(.4))
        self.fsm.step(observation(.55))
        result = self.fsm.step(observation(.9))
        self.assertEqual(result.state, State.REACQUIRE_CONFIRM)
        self.assertEqual(result.confirmation_frames, 1)
        self.assertEqual(result.budget.search_reserved_s, first.budget.search_reserved_s)

    def test_high_visibility_ambiguous_person_never_reacquires(self):
        first = self.start_search()
        result = self.fsm.step(observation(.42, identity=Identity.AMBIGUOUS))
        self.assertNotEqual(result.state, State.TRACKING)
        self.assertEqual(result.confirmation_frames, 0)
        self.assertEqual(result.initialization_id, first.initialization_id)

    def test_visibility_hysteresis_does_not_bounce_normal_tracking(self):
        self.fsm.step(observation(0.))
        result = self.fsm.step(observation(.1, visibility_probability=.7))
        self.assertEqual(result.state, State.TRACKING)
        result = self.fsm.step(observation(.2, visibility_probability=.49))
        self.assertEqual(result.state, State.LOST_HOLD)
        result = self.fsm.step(observation(.3, visibility_probability=.7))
        self.assertNotEqual(result.state, State.TRACKING)

    def test_false_reacquisition_and_direction_reversal_cannot_refund_budgets(self):
        first = self.start_search()
        pending = self.fsm.step(observation(.4))
        self.assertEqual(pending.state, State.REACQUIRE_CONFIRM)
        reversed_result = self.fsm.step(lost(.5, suggested_yaw_rad_s=-1.))
        self.assertEqual(reversed_result.state, State.SEARCH)
        self.assertEqual(reversed_result.budget.reversals_used, 1)
        self.assertGreater(reversed_result.budget.search_reserved_s, first.budget.search_reserved_s)
        denied = self.fsm.step(lost(.61, suggested_yaw_rad_s=1.))
        self.assert_stopped(denied, "search_reversal_budget_exhausted")
        self.assertEqual(denied.budget.reversals_used, 1)
        self.assertEqual(denied.budget.search_reserved_s, reversed_result.budget.search_reserved_s)

    def test_search_time_budget_shortens_last_command_then_stops(self):
        fsm = ActiveSearchFSM(config(max_search_command_time_s=.15))
        self.start_search(fsm)
        last = fsm.step(lost(.42))
        self.assertAlmostEqual(last.command_valid_until_s - .42, .05)
        result = fsm.step(lost(.5))
        self.assert_stopped(result, "search_time_budget_exhausted")
        self.assertAlmostEqual(result.budget.search_reserved_s, .15)

    def test_yaw_budget_shortens_last_command_then_stops(self):
        fsm = ActiveSearchFSM(config(max_search_yaw_rad=.05))
        self.start_search(fsm)
        last = fsm.step(lost(.42))
        self.assertAlmostEqual(last.command_valid_until_s - .42, .025)
        self.assert_stopped(fsm.step(lost(.5)), "search_yaw_budget_exhausted")

    def test_lost_time_budget_includes_confirmation_and_safe_holds(self):
        fsm = ActiveSearchFSM(config(max_lost_time_s=.6))
        self.start_search(fsm)
        fsm.step(observation(.4))
        fsm.step(observation(.55, motion_permission=Permission.DENY))
        result = fsm.step(lost(.71))
        self.assert_stopped(result, "lost_time_budget_exhausted")
        self.assertGreaterEqual(result.budget.lost_elapsed_s, .6)

    def test_spent_budget_allows_passive_same_target_confirmation_but_never_refunds_search(self):
        cases = (
            ({"max_lost_time_s": .25}, "lost_time_budget_exhausted"),
            ({"max_search_command_time_s": .1}, "search_time_budget_exhausted"),
            ({"max_search_yaw_rad": .04}, "search_yaw_budget_exhausted"),
        )
        for changes, reason in cases:
            with self.subTest(reason=reason):
                fsm = ActiveSearchFSM(config(**changes))
                self.start_search(fsm)
                spent = fsm.step(lost(.36))
                self.assert_stopped(spent, reason)
                for t in (.4, .55):
                    pending = fsm.step(observation(t))
                    self.assertEqual(pending.state, State.REACQUIRE_CONFIRM)
                    self.assert_stopped(pending, "same_target_confirmation_pending")
                recovered = fsm.step(observation(.71))
                self.assertEqual(recovered.state, State.TRACKING)
                self.assertEqual(recovered.reason, "same_target_reacquired")
                self.assertTrue(recovered.tracking_ready)
                self.assertEqual(recovered.command, ZERO)
                self.assertEqual(recovered.initialization_id, spent.initialization_id)
                self.assertEqual(recovered.budget.search_reserved_s, spent.budget.search_reserved_s)
                self.assertEqual(recovered.budget.yaw_reserved_rad, spent.budget.yaw_reserved_rad)
                self.assertEqual(recovered.budget.reversals_used, spent.budget.reversals_used)
                self.assertGreaterEqual(recovered.budget.lost_elapsed_s, spent.budget.lost_elapsed_s)
                # The original target may be followed, but no search entitlement
                # is created by confirmation or by subsequent target loss.
                again_lost = fsm.step(lost(.8))
                self.assert_stopped(again_lost, reason)
                self.assertEqual(again_lost.budget.search_reserved_s, spent.budget.search_reserved_s)
                self.assertEqual(again_lost.budget.yaw_reserved_rad, spent.budget.yaw_reserved_rad)

    def test_spent_budget_unknown_identity_and_single_candidate_never_restore_tracking(self):
        fsm = ActiveSearchFSM(config(max_search_command_time_s=.1))
        self.start_search(fsm)
        fsm.step(lost(.36))
        for t in (.4, .5, .6):
            result = fsm.step(observation(t, identity=Identity.UNKNOWN))
            self.assert_stopped(result, "search_time_budget_exhausted")
            self.assertEqual(result.confirmation_frames, 0)
        single = fsm.step(observation(.7))
        self.assert_stopped(single, "same_target_confirmation_pending")
        self.assertEqual(single.confirmation_frames, 1)
        interrupted = fsm.step(observation(.8, identity=Identity.AMBIGUOUS))
        self.assert_stopped(interrupted, "search_time_budget_exhausted")
        restarted = fsm.step(observation(.9))
        self.assert_stopped(restarted, "same_target_confirmation_pending")
        self.assertEqual(restarted.confirmation_frames, 1)

    def test_spent_budget_passive_confirmation_never_bypasses_global_protection(self):
        cases = (
            ({"motion_permission": Permission.UNKNOWN}, "motion_permission_unknown"),
            ({"motion_permission": Permission.DENY}, "motion_permission_deny"),
            ({"permission_timestamp_s": .1}, "stale_permission"),
            ({"stop_probability": .8}, "policy_stop"),
            ({"rgb_available": False}, "rgb_unavailable"),
            ({"prediction_timestamp_s": -.2}, "stale_prediction"),
            ({"now_s": .3}, "monotonic_clock_regressed"),
        )
        for changes, reason in cases:
            with self.subTest(reason=reason):
                fsm = ActiveSearchFSM(config(max_search_command_time_s=.1))
                self.start_search(fsm)
                fsm.step(lost(.36))
                fsm.step(observation(.4))
                result = fsm.step(observation(.55, **changes))
                self.assert_stopped(result, reason)
                self.assertEqual(result.confirmation_frames, 0)

    def test_passive_confirmation_preserves_reversal_count(self):
        self.start_search()
        reversed_result = self.fsm.step(lost(.42, suggested_yaw_rad_s=-1.))
        self.assertEqual(reversed_result.budget.reversals_used, 1)
        self.assert_stopped(self.fsm.step(lost(.53)), "search_reversal_budget_exhausted")
        for t in (.6, .75, .91):
            recovered = self.fsm.step(observation(t))
        self.assertTrue(recovered.tracking_ready)
        self.assertEqual(recovered.budget.reversals_used, 1)
        self.fsm.step(lost(1.))
        again = self.fsm.step(lost(1.21))
        self.assert_stopped(again, "search_reversal_budget_exhausted")
        self.assertEqual(again.budget.reversals_used, 1)

    def test_confirmed_reacquisition_does_not_reset_task_lifetime_budget(self):
        fsm = ActiveSearchFSM(config(max_search_command_time_s=.15))
        first = self.start_search(fsm)
        for t in (.4, .55, .71):
            fsm.step(observation(t))
        fsm.step(lost(.8))
        result = fsm.step(lost(1.01))
        self.assertEqual(result.state, State.SEARCH)
        self.assertAlmostEqual(result.command_valid_until_s - 1.01, .05)
        self.assertGreater(result.budget.lost_elapsed_s, first.budget.lost_elapsed_s)
        self.assert_stopped(fsm.step(lost(1.1)), "search_time_budget_exhausted")

    def test_command_expiry_cannot_outlive_permission_or_target_memory(self):
        self.start_search()
        # A delayed permit may shorten TTL but must be newer than the .31
        # watermark already accepted by start_search.
        result = self.fsm.step(lost(.42, permission_timestamp_s=.32))
        self.assertAlmostEqual(result.command_valid_until_s, .47)
        fsm = ActiveSearchFSM(config(max_target_memory_age_s=.4))
        result = self.start_search(fsm)
        self.assertAlmostEqual(result.command_valid_until_s, .4)
        self.assert_stopped(fsm.step(lost(.41)), "target_memory_expired")

    def test_no_direction_means_no_motion(self):
        self.fsm.step(observation(0., suggested_yaw_rad_s=None))
        self.fsm.step(lost(.1, suggested_yaw_rad_s=None))
        result = self.fsm.step(lost(.31, suggested_yaw_rad_s=None))
        self.assert_stopped(result, "no_search_direction")

    def test_only_explicit_reset_clears_target_faults_confirmation_and_budgets(self):
        before = self.start_search()
        self.fsm.step(observation(.4))
        self.fsm.reset_task()
        result = self.fsm.step(observation(0., initialization_id=None))
        self.assertEqual(result.task_generation, before.task_generation + 1)
        self.assertIsNone(result.initialization_id)
        self.assertEqual(result.confirmation_frames, 0)
        self.assertEqual(result.budget.search_reserved_s, 0.)
        self.assertEqual(result.budget.yaw_reserved_rad, 0.)
        self.assertEqual(result.budget.lost_elapsed_s, 0.)
        self.assertIsNone(result.latched_fault)


if __name__ == "__main__":
    unittest.main()
