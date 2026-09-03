"""Reward utilities for the V27 IL -> online-PPO tracking experiment.

V22 used ``human_collision`` as an instantaneous collision signal.  In the
bundled EVT-Bench metric that field becomes sticky after the first
near-contact (distance < 0.5 m), so charging it at every later step corrupts
the return.  V27 prefers the independent ``did_multi_agents_collide`` event
and only falls back to a rising edge of the legacy field when that metric is
not available.

The policy never receives any value from this module.  These functions are
called after the environment step and only construct the PPO reward.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

try:
    from .online_rl_utils_v22 import metric_float
except ImportError:  # pragma: no cover - direct script execution
    from hybrid_flux.online_rl_utils_v22 import metric_float  # type: ignore


def _collision_value(metrics: Optional[Dict[str, Any]]) -> float:
    """Return the collision signal selected by the runner's metric snapshot."""

    if not metrics:
        return 0.0
    if metric_float(metrics, "collision_metric_available", 0.0) > 0.5:
        return float(metric_float(metrics, "did_multi_agents_collide", 0.0) > 0.5)
    # Compatibility for an older environment that does not expose the proper
    # contact measure.  The caller converts this sticky value to a rising
    # edge, so a single old-style near-contact is not charged for 300 frames.
    return float(metric_float(metrics, "human_collision", 0.0) > 0.5)


def tracking_reward_v27(
    previous_metrics: Optional[Dict[str, Any]],
    metrics: Dict[str, Any],
    action: Sequence[float],
    done: bool,
) -> Tuple[float, Dict[str, float]]:
    """Compute a benchmark-aligned dense reward with event-based safety cost.

    The official following condition is ``distance <= 3 m`` and detector
    facing.  The success interval is ``1--3 m``.  The smooth distance band is
    therefore centred at 2 m, while a separate near-contact term discourages
    entering the unsafe <0.85 m region.
    """

    current_distance = metric_float(metrics, "distance_to_leader", 10.0)
    previous_distance = (
        metric_float(previous_metrics, "distance_to_leader", current_distance)
        if previous_metrics is not None
        else current_distance
    )
    following = metric_float(metrics, "human_following", 0.0)
    previous_following = (
        metric_float(previous_metrics, "human_following", 0.0)
        if previous_metrics is not None
        else 0.0
    )
    success = metric_float(metrics, "human_following_success", 0.0)
    previous_success = (
        metric_float(previous_metrics, "human_following_success", 0.0)
        if previous_metrics is not None
        else 0.0
    )

    collision = _collision_value(metrics)
    previous_collision = _collision_value(previous_metrics)
    collision_event = float(collision > 0.5 and previous_collision <= 0.5)

    band = math.exp(-((current_distance - 2.0) / 0.85) ** 2)
    progress = float(
        np.clip((previous_distance - current_distance) / 0.5, -1.0, 1.0)
    )
    following_gain = float(max(0.0, following - previous_following))
    recovered = float(previous_following <= 0.5 and following > 0.5)
    lost = float(previous_following > 0.5 and following <= 0.5)
    near_contact = float(np.clip((0.85 - current_distance) / 0.85, 0.0, 1.0))

    command = np.asarray(action, dtype=np.float32).reshape(-1)[:3]
    command_cost = float(np.square(command).mean()) if command.size == 3 else 0.0

    # The outer runner keeps the historical reward_scale=0.10.  Coefficients
    # below are intentionally moderate: one contact event should matter, but
    # must not erase an entire 300-step episode as the sticky V22 signal did.
    reward = (
        0.80 * following
        + 0.35 * band
        + 0.20 * progress
        + 0.45 * following_gain
        + 0.35 * recovered
        - 0.80 * collision_event
        - 0.20 * near_contact
        - 0.01 * command_cost
    )
    if done and following > 0.5:
        reward += 0.80
    if done and success > 0.5 and previous_success <= 0.5:
        reward += 0.40

    components = {
        "distance": float(current_distance),
        "following": float(following),
        "success": float(success),
        # ``collision`` now means a collision event, matching the cost above.
        "collision": float(collision_event),
        "collision_event": float(collision_event),
        "band": float(band),
        "progress": float(progress),
        "success_bonus": float(following_gain),
        "recovered": float(recovered),
        "lost": float(lost),
        "near_contact": float(near_contact),
        "command_cost": float(command_cost),
    }
    return float(reward), components


__all__ = ["tracking_reward_v27"]
