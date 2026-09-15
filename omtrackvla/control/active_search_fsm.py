"""Pure, fail-closed search decisions; no sensor access, I/O or actuator control.

All timestamps use one caller-supplied monotonic seconds clock. ``command`` is
a proposed command for a future dispatcher, not evidence of physical execution.
TRACKING emits zero here and exposes ``tracking_ready``; normal waypoint control
is deliberately outside this module. Real execution requires its own watchdog.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class State(str, Enum):
    UNINITIALIZED = "UNINITIALIZED"
    TRACKING = "TRACKING"
    LOST_HOLD = "LOST_HOLD"
    SEARCH = "SEARCH"
    REACQUIRE_CONFIRM = "REACQUIRE_CONFIRM"
    SAFE_HOLD = "SAFE_HOLD"


class MotionPermission(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    UNKNOWN = "unknown"


class IdentityEvidence(str, Enum):
    KNOWN_SAME = "known_same"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class SearchConfig:
    """Required experimental values; no defaults imply calibrated settings."""

    lost_visibility_threshold: float
    reacquire_visibility_threshold: float
    stop_probability_threshold: float
    lost_confirm_s: float
    reacquire_confirm_s: float
    reacquire_min_frames: int
    reacquire_max_gap_s: float
    max_rgb_age_s: float
    max_prediction_age_s: float
    max_permission_age_s: float
    max_target_memory_age_s: float
    max_lost_time_s: float
    max_search_command_time_s: float
    max_search_yaw_rad: float
    max_search_reversals: int
    max_yaw_rate_rad_s: float
    command_ttl_s: float

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if name in ("reacquire_min_frames", "max_search_reversals"):
                if not isinstance(value, int) or isinstance(value, bool):
                    raise ValueError(f"{name} must be an integer")
            elif not _finite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 < self.lost_visibility_threshold < self.reacquire_visibility_threshold < 1:
            raise ValueError("visibility thresholds require 0 < lost < reacquire < 1")
        if not 0 < self.stop_probability_threshold < 1:
            raise ValueError("stop probability threshold must lie in (0, 1)")
        if self.reacquire_min_frames < 2 or self.max_search_reversals < 0:
            raise ValueError("at least two confirmation frames and nonnegative reversals required")
        if self.lost_confirm_s < 0:
            raise ValueError("lost_confirm_s must be nonnegative")
        for name in (
            "reacquire_confirm_s", "reacquire_max_gap_s", "max_rgb_age_s",
            "max_prediction_age_s", "max_permission_age_s", "max_target_memory_age_s",
            "max_lost_time_s", "max_search_command_time_s", "max_search_yaw_rad",
            "max_yaw_rate_rad_s", "command_ttl_s",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class SearchInput:
    now_s: float
    # Opaque user/task initialization nonce, never a benchmark GT identity.
    initialization_id: str | None
    rgb_available: bool
    rgb_timestamp_s: float | None
    prediction_timestamp_s: float | None
    prediction_rgb_timestamp_s: float | None
    visibility_probability: float
    stop_probability: float
    identity: IdentityEvidence
    # Caller derives this from legal predictions/history, never GT/navigation.
    suggested_yaw_rad_s: float | None
    motion_permission: MotionPermission = MotionPermission.UNKNOWN
    permission_timestamp_s: float | None = None


@dataclass(frozen=True)
class MotionCommand:
    forward_m_s: float = 0.0
    lateral_m_s: float = 0.0
    yaw_rad_s: float = 0.0


ZERO = MotionCommand()


@dataclass(frozen=True)
class BudgetAudit:
    lost_elapsed_s: float
    search_reserved_s: float
    yaw_reserved_rad: float
    reversals_used: int
    lost_remaining_s: float
    search_remaining_s: float
    yaw_remaining_rad: float


@dataclass(frozen=True)
class SearchDecision:
    evaluated_at_s: float | None
    task_generation: int
    initialization_id: str | None
    previous_state: State
    state: State
    reason: str
    suggestion: MotionCommand
    command: MotionCommand
    command_valid_until_s: float | None
    tracking_ready: bool
    has_confirmed_tracking: bool
    confirmation_frames: int
    budget: BudgetAudit
    latched_fault: str | None


class ActiveSearchFSM:
    """Bounded yaw-only search with cumulative, task-lifetime reservations.

    Each nonzero command reserves its complete TTL in the time and absolute-yaw
    budgets. Reissuing early consumes another reservation (conservative). Neither
    apparent nor confirmed reacquisition refunds budgets. Only reset_task clears
    them. An independent actuator watchdog must enforce command_valid_until_s.
    """

    def __init__(self, config: SearchConfig) -> None:
        self._config = config
        self._generation = 0
        self.reset_task()

    @property
    def config(self) -> SearchConfig:
        return self._config

    def reset_task(self) -> None:
        """Explicit task cancel/new-task boundary; caller must first send STOP."""
        self._generation += 1
        self.state = State.UNINITIALIZED
        self._initialization_id: str | None = None
        self._last_now: float | None = None
        self._last_rgb_stamp: float | None = None
        self._last_prediction_stamp: float | None = None
        self._last_permission_stamp: float | None = None
        self._last_permission_value: MotionPermission | None = None
        self._permission_stamp_conflicted = False
        self._last_trusted_time: float | None = None
        self._trusted_yaw_hint: float | None = None
        self._has_tracked = False
        self._loss_active = False
        self._loss_started: float | None = None
        self._lost_elapsed = 0.0
        self._search_reserved = 0.0
        self._yaw_reserved = 0.0
        self._reversals = 0
        self._last_search_sign = 0
        self._fault: str | None = None
        self._clear_confirmation()

    def _clear_confirmation(self) -> None:
        self._confirm_first_rgb: float | None = None
        self._confirm_last_now: float | None = None
        self._confirm_last_rgb: float | None = None
        self._confirm_frames = 0

    def _enter_loss(self, now: float) -> None:
        if not self._loss_active:
            self._loss_active = True
            self._loss_started = now

    def _budgets(self) -> BudgetAudit:
        return BudgetAudit(
            self._lost_elapsed, self._search_reserved, self._yaw_reserved,
            self._reversals,
            max(0.0, self.config.max_lost_time_s - self._lost_elapsed),
            max(0.0, self.config.max_search_command_time_s - self._search_reserved),
            max(0.0, self.config.max_search_yaw_rad - self._yaw_reserved),
        )

    def _emit(self, previous: State, reason: str, suggestion: MotionCommand,
              command: MotionCommand = ZERO, *, expiry: float | None = None,
              tracking_ready: bool = False) -> SearchDecision:
        return SearchDecision(
            self._last_now, self._generation, self._initialization_id,
            previous, self.state, reason, suggestion, command, expiry,
            tracking_ready, self._has_tracked, self._confirm_frames,
            self._budgets(), self._fault,
        )

    def _hold(self, previous: State, now: float, reason: str,
              suggestion: MotionCommand) -> SearchDecision:
        if self._initialization_id is not None:
            self._enter_loss(now)
        self.state = State.SAFE_HOLD
        self._clear_confirmation()
        return self._emit(previous, reason, suggestion)

    @staticmethod
    def _age_error(now: float, stamp: float | None, maximum: float, name: str) -> str | None:
        if not _finite(stamp):
            return f"missing_or_invalid_{name}_timestamp"
        if stamp > now:
            return f"future_{name}_timestamp"
        if now - stamp >= maximum:
            return f"stale_{name}"
        return None

    def _observe_permission(self, now: float, value: SearchInput) -> str | None:
        """Retain fresh permit watermarks even when another guard will hold.

        Repeated identical decisions may be reused until their original expiry.
        Conflicting values at one timestamp require a strictly newer decision;
        neither ALLOW nor a later replay can erase a same-stamp revocation.
        Invalid/future/stale timestamps never advance the watermark.
        """
        permission = value.motion_permission
        if not isinstance(permission, MotionPermission):
            return "invalid_motion_permission"
        stamp = value.permission_timestamp_s
        error = self._age_error(now, stamp, self.config.max_permission_age_s, "permission")
        if error is None:
            if self._last_permission_stamp is not None and stamp < self._last_permission_stamp:
                return "permission_timestamp_regressed"
            if stamp == self._last_permission_stamp:
                if permission != self._last_permission_value:
                    self._permission_stamp_conflicted = True
                if self._permission_stamp_conflicted:
                    return "permission_timestamp_conflict"
            else:
                self._last_permission_stamp = stamp
                self._last_permission_value = permission
                self._permission_stamp_conflicted = False
        if permission != MotionPermission.ALLOW:
            return f"motion_permission_{permission.value}"
        return error

    def _remember_strong_capture(self, value: SearchInput) -> None:
        """A processing callback cannot renew the age of an already seen image."""
        capture = value.rgb_timestamp_s
        if self._last_trusted_time is None or capture > self._last_trusted_time:
            self._last_trusted_time = capture
            self._trusted_yaw_hint = value.suggested_yaw_rad_s

    def step(self, value: SearchInput) -> SearchDecision:
        previous = self.state
        cfg = self.config
        if not _finite(value.now_s) or value.now_s < 0:
            self._fault = "invalid_monotonic_clock"
            self.state = State.SAFE_HOLD
            self._clear_confirmation()
            return self._emit(previous, self._fault, ZERO)
        now = float(value.now_s)
        if self._last_now is not None and now < self._last_now:
            self._fault = "monotonic_clock_regressed"
        if self._fault is not None:
            self.state = State.SAFE_HOLD
            self._clear_confirmation()
            return self._emit(previous, self._fault, ZERO)
        # Permission revocations remain meaningful even if a caller accidentally
        # repeats the control timestamp and this step must issue STOP.
        permission_error = self._observe_permission(now, value)
        if self._last_now is not None and now == self._last_now:
            return self._hold(previous, now, "monotonic_clock_did_not_advance", ZERO)
        if self._loss_active and self._last_now is not None:
            self._lost_elapsed += now - self._last_now
        self._last_now = now

        # Default and rejected outputs are finite and never inherit an old command.
        yaw = value.suggested_yaw_rad_s
        suggestion = ZERO
        if yaw is None:
            yaw = self._trusted_yaw_hint
        if _finite(yaw):
            suggestion = MotionCommand(yaw_rad_s=max(-cfg.max_yaw_rate_rad_s,
                                                      min(cfg.max_yaw_rate_rad_s, float(yaw))))

        token = value.initialization_id
        if token is not None and (not isinstance(token, str) or not token.strip()):
            return self._hold(previous, now, "invalid_initialization_id", suggestion)
        if self._initialization_id is None:
            if token is None:
                self.state = State.UNINITIALIZED
                self._clear_confirmation()
                return self._emit(previous, "cold_no_target", suggestion)
            self._initialization_id = token
        elif token != self._initialization_id:
            if token is None:
                return self._hold(previous, now, "initialization_temporarily_unavailable", suggestion)
            self._fault = "initialization_changed_without_task_reset"
            return self._hold(previous, now, self._fault, suggestion)

        if not isinstance(value.rgb_available, bool) or not value.rgb_available:
            return self._hold(previous, now, "rgb_unavailable", suggestion)
        for stamp, maximum, label in (
            (value.rgb_timestamp_s, cfg.max_rgb_age_s, "rgb"),
            (value.prediction_timestamp_s, cfg.max_prediction_age_s, "prediction"),
        ):
            error = self._age_error(now, stamp, maximum, label)
            if error:
                return self._hold(previous, now, error, suggestion)
        if value.prediction_timestamp_s < value.rgb_timestamp_s:
            return self._hold(previous, now, "prediction_predates_rgb", suggestion)
        if not _finite(value.prediction_rgb_timestamp_s) or value.prediction_rgb_timestamp_s != value.rgb_timestamp_s:
            return self._hold(previous, now, "prediction_rgb_mismatch", suggestion)
        if self._last_rgb_stamp is not None and value.rgb_timestamp_s < self._last_rgb_stamp:
            return self._hold(previous, now, "rgb_timestamp_regressed", suggestion)
        if self._last_prediction_stamp is not None and value.prediction_timestamp_s < self._last_prediction_stamp:
            return self._hold(previous, now, "prediction_timestamp_regressed", suggestion)
        self._last_rgb_stamp = value.rgb_timestamp_s
        self._last_prediction_stamp = value.prediction_timestamp_s
        if any(not _finite(p) or not 0 <= p <= 1
               for p in (value.visibility_probability, value.stop_probability)):
            return self._hold(previous, now, "invalid_prediction_probability", suggestion)
        if value.suggested_yaw_rad_s is not None and not _finite(value.suggested_yaw_rad_s):
            return self._hold(previous, now, "invalid_search_direction", ZERO)
        if not isinstance(value.identity, IdentityEvidence):
            return self._hold(previous, now, "invalid_identity_evidence", suggestion)
        if value.stop_probability >= cfg.stop_probability_threshold:
            return self._hold(previous, now, "policy_stop", suggestion)
        if permission_error:
            return self._hold(previous, now, permission_error, suggestion)

        same = value.identity == IdentityEvidence.KNOWN_SAME
        strong = same and value.visibility_probability >= cfg.reacquire_visibility_threshold
        tracking_evidence = same and value.visibility_probability >= cfg.lost_visibility_threshold
        if not self._has_tracked and strong and not self._loss_active:
            self._has_tracked = True
            self.state = State.TRACKING
            self._remember_strong_capture(value)
            return self._emit(previous, "initialized_tracking", suggestion, tracking_ready=True)
        if self.state == State.TRACKING and tracking_evidence:
            if strong:
                self._remember_strong_capture(value)
            return self._emit(previous, "tracking_evidence_retained", suggestion, tracking_ready=True)

        self._enter_loss(now)
        # Spent search budgets forbid new search motion, not passive confirmation
        # of the original target. Global input/clock/stop/permit guards above still
        # win; successful confirmation never refunds task-lifetime budgets.
        if strong:
            # Only distinct, consecutive, temporally close RGB observations count.
            if self._confirm_first_rgb is None or (
                self._confirm_last_now is not None and
                now - self._confirm_last_now > cfg.reacquire_max_gap_s
            ) or (
                self._confirm_last_rgb is not None and
                value.rgb_timestamp_s - self._confirm_last_rgb > cfg.reacquire_max_gap_s
            ):
                self._clear_confirmation()
                self._confirm_first_rgb = value.rgb_timestamp_s
            new_confirmation_frame = self._confirm_last_rgb is None or value.rgb_timestamp_s > self._confirm_last_rgb
            if new_confirmation_frame:
                self._confirm_frames += 1
                self._confirm_last_rgb = value.rgb_timestamp_s
                self._confirm_last_now = now
            self.state = State.REACQUIRE_CONFIRM
            if new_confirmation_frame and self._confirm_frames >= cfg.reacquire_min_frames and value.rgb_timestamp_s - self._confirm_first_rgb >= cfg.reacquire_confirm_s:
                self.state = State.TRACKING
                self._has_tracked = True
                self._loss_active = False
                self._loss_started = None
                self._remember_strong_capture(value)
                self._clear_confirmation()
                return self._emit(previous, "same_target_reacquired", suggestion, tracking_ready=True)
            return self._emit(previous, "same_target_confirmation_pending", suggestion)
        self._clear_confirmation()
        if self._lost_elapsed >= cfg.max_lost_time_s:
            return self._hold(previous, now, "lost_time_budget_exhausted", suggestion)
        if self._search_reserved >= cfg.max_search_command_time_s:
            return self._hold(previous, now, "search_time_budget_exhausted", suggestion)
        if self._yaw_reserved >= cfg.max_search_yaw_rad:
            return self._hold(previous, now, "search_yaw_budget_exhausted", suggestion)
        if not self._has_tracked or self._last_trusted_time is None:
            return self._hold(previous, now, "no_confirmed_tracking_history", suggestion)
        if now - self._last_trusted_time >= cfg.max_target_memory_age_s:
            return self._hold(previous, now, "target_memory_expired", suggestion)
        if now - self._loss_started < cfg.lost_confirm_s:
            self.state = State.LOST_HOLD
            return self._emit(previous, "loss_confirmation_pending", suggestion)
        if suggestion.yaw_rad_s == 0.0:
            return self._hold(previous, now, "no_search_direction", suggestion)

        sign = 1 if suggestion.yaw_rad_s > 0 else -1
        reversal = self._last_search_sign != 0 and sign != self._last_search_sign
        if reversal and self._reversals >= cfg.max_search_reversals:
            return self._hold(previous, now, "search_reversal_budget_exhausted", suggestion)
        # Command expires before any input, permit, memory or budget ceases to hold.
        remaining = self._budgets()
        ttl = min(
            cfg.command_ttl_s,
            value.rgb_timestamp_s + cfg.max_rgb_age_s - now,
            value.prediction_timestamp_s + cfg.max_prediction_age_s - now,
            value.permission_timestamp_s + cfg.max_permission_age_s - now,
            self._last_trusted_time + cfg.max_target_memory_age_s - now,
            remaining.lost_remaining_s,
            remaining.search_remaining_s,
            remaining.yaw_remaining_rad / abs(suggestion.yaw_rad_s),
        )
        if ttl <= 0 or not math.isfinite(now + ttl) or now + ttl <= now:
            return self._hold(previous, now, "no_valid_command_window", suggestion)
        self._search_reserved = min(cfg.max_search_command_time_s, self._search_reserved + ttl)
        self._yaw_reserved = min(cfg.max_search_yaw_rad,
                                 self._yaw_reserved + abs(suggestion.yaw_rad_s) * ttl)
        self._reversals += int(reversal)
        self._last_search_sign = sign
        self.state = State.SEARCH
        return self._emit(previous, "bounded_yaw_search", suggestion, suggestion, expiry=now + ttl)
