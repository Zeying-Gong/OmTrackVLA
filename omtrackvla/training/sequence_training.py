"""Sequence training utilities; no fabricated recurrent state or labels.

Install as ``omtrackvla/training/sequence_training.py``. All step inputs use
the deployment policy's four-frame image history; that history axis is NEVER
treated as the sequence axis. Prefix replay must be built from actual policy
calls starting at reset. The caller owns that data/provenance validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


MODEL_INPUT_KEYS = (
    "initial_rgb", "initial_bbox", "ego_rgb", "visual_initialization_valid",
    "rgb_valid", "binding_valid", "uwb_xy", "uwb_covariance_xy", "uwb_quality",
    "uwb_age_s", "uwb_valid", "camera_intrinsics", "camera_from_base",
)
STEP_TRAILING_DIMS = {
    "rgb_valid": 0, "binding_valid": 0, "uwb_xy": 1,
    "uwb_covariance_xy": 2, "uwb_quality": 0, "uwb_age_s": 0,
    "uwb_valid": 0, "camera_intrinsics": 2, "camera_from_base": 2,
}


@dataclass(frozen=True)
class PolicySequenceState:
    """All recurrent values used by EndToEndFollowPolicy.forward_sequence."""

    hidden: torch.Tensor | None = None
    target_memory: torch.Tensor | None = None
    binding: torch.Tensor | None = None

    def detached(self) -> "PolicySequenceState":
        return PolicySequenceState(*(
            None if value is None else value.detach()
            for value in (self.hidden, self.target_memory, self.binding)
        ))


def sequence_model_inputs(batch: Mapping[str, torch.Tensor], device=None):
    """Keep every policy step when moving a clean/recovery batch to a device."""
    result = {name: batch[name] for name in MODEL_INPUT_KEYS}
    if result["ego_rgb"].ndim != 6:
        raise ValueError("ego_rgb must be [B,S,T,3,H,W]; four history frames are one policy step")
    if device is not None:
        result = {name: value.to(device, non_blocking=True) for name, value in result.items()}
    return result


def slice_sequence_inputs(inputs: Mapping[str, torch.Tensor], start: int, end: int):
    """Slice policy steps, preserving initialization and static calibration."""
    batch, steps = inputs["ego_rgb"].shape[:2]
    if not 0 <= start < end <= steps:
        raise ValueError("invalid policy-step slice")
    result = dict(inputs)
    result["ego_rgb"] = inputs["ego_rgb"][:, start:end]
    for key, trailing in STEP_TRAILING_DIMS.items():
        value = inputs[key]
        if value.ndim == trailing + 2:
            if value.shape[:2] != (batch, steps):
                raise ValueError(f"{key} sequence shape mismatch")
            result[key] = value[:, start:end]
    return result


def unroll_policy_sequence(
    policy: nn.Module,
    inputs: Mapping[str, torch.Tensor],
    *,
    state: PolicySequenceState | None = None,
    burn_in_steps: int = 0,
    tbptt_steps: int = 0,
) -> tuple[dict[str, torch.Tensor], PolicySequenceState]:
    """Replay real observations and return outputs AFTER the burn-in prefix.

    Burn-in reconstructs all state with current weights under no_grad. The
    caller aligns labels by dropping the same prefix. ``tbptt_steps`` detaches
    all three states between learning chunks. It bounds gradient history, not
    total output/activation storage when all chunks share one backward call.
    Never provide a state from another episode, or from an older checkpoint.
    """
    inputs = sequence_model_inputs(inputs)
    batch, steps = inputs["ego_rgb"].shape[:2]
    if not 0 <= burn_in_steps < steps:
        raise ValueError("burn-in must leave at least one learning policy step")
    if tbptt_steps < 0:
        raise ValueError("tbptt_steps must be nonnegative")
    state = state if state is not None else PolicySequenceState()

    def at(name: str, index: int) -> torch.Tensor:
        value = inputs[name]
        trailing = STEP_TRAILING_DIMS[name]
        if value.shape[0] != batch:
            raise ValueError(f"{name} batch dimension mismatch")
        if value.ndim == trailing + 2 and value.shape[1] == steps:
            return value[:, index]
        if value.ndim == trailing + 1:
            return value
        raise ValueError(f"{name} invalid static/sequence shape")

    outputs = []
    for index in range(steps):
        if index == burn_in_steps or (
            index > burn_in_steps and tbptt_steps
            and (index - burn_in_steps) % tbptt_steps == 0
        ):
            # A caller-supplied chunk state may intentionally retain a graph.
            # Only an actual burn-in prefix or a TBPTT boundary detaches it.
            if index > 0:
                state = state.detached()
                if index == burn_in_steps and torch.is_grad_enabled():
                    # An outer autocast may cache low-precision parameter
                    # casts made during no-grad burn-in. Clear these detached
                    # casts before learning so trainable weights reconnect to
                    # autograd, retaining the same dtype and recurrent state.
                    torch.clear_autocast_cache()
        step_inputs = {
            name: at(name, index) if name in STEP_TRAILING_DIMS else inputs[name]
            for name in MODEL_INPUT_KEYS if name != "ego_rgb"
        }
        step_inputs["ego_rgb"] = inputs["ego_rgb"][:, index]
        requested = step_inputs["binding_valid"]
        binding = requested if state.binding is None else torch.maximum(requested, state.binding)
        step_inputs["binding_valid"] = binding
        # Respect an outer no_grad/inference context during evaluation.
        with torch.set_grad_enabled(torch.is_grad_enabled() and index >= burn_in_steps):
            current = policy(
                **step_inputs, hidden_state=state.hidden,
                _target_memory_override=state.target_memory,
            )
            predicted_binding = (
                torch.sigmoid(current["binding_logit"]).squeeze(1)
                * step_inputs["uwb_valid"] * step_inputs["rgb_valid"]
            )
            state = PolicySequenceState(
                current["hidden_state"], current["target_memory_next"],
                torch.maximum(binding, predicted_binding),
            )
        if index >= burn_in_steps:
            outputs.append(current)
    return {
        key: torch.stack([output[key] for output in outputs], dim=1)
        for key in outputs[0]
    }, state


class SequenceTrainingPolicy(nn.Module):
    """Wrap this module with DDP, so DDP sees the entire sequence forward."""

    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, *, burn_in_steps=0, tbptt_steps=0, **inputs):
        return unroll_policy_sequence(
            self.policy, inputs, burn_in_steps=burn_in_steps,
            tbptt_steps=tbptt_steps,
        )[0]


def phase3_sequence_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise only real labels; clean replay can retain all auxiliary heads.

    Labels must already align with outputs after burn-in. ``supervision_mask``
    [B,S] selects labeled policy steps (e.g. only the recovery anchor). Specific
    masks further restrict each head. No stop/visibility/binding labels are
    inferred from RGB availability, target disappearance, or expert path size.
    ``visibility_label_valid`` and ``bbox_label_valid`` independently describe
    real perception labels. Only batches lacking these keys fall back to the
    legacy ``identity_label_valid`` mask; visibility does not label identity.
    Training-only world/action heads require the existing Phase-2 trainer.
    """
    from omtrackvla.models.end_to_end import (
        waypoint_only_loss, waypoint_delta_loss, waypoint_terminal_loss,
        waypoint_radial_progress_loss, waypoint_path_length_loss,
    )

    waypoint_functions = {
        "waypoint": waypoint_only_loss, "waypoint_delta": waypoint_delta_loss,
        "waypoint_terminal": waypoint_terminal_loss,
        "waypoint_radial_progress": waypoint_radial_progress_loss,
        "waypoint_path_length": waypoint_path_length_loss,
    }
    allowed = set(waypoint_functions) | {"stop", "bbox", "visibility", "identity_or_binding", "ego"}
    enabled = {name: float(value) for name, value in weights.items() if float(value) != 0.0}
    unknown = set(enabled) - allowed
    if unknown:
        raise ValueError(f"unsupported Phase-3 sequence losses: {sorted(unknown)}")
    shape = outputs["waypoints"].shape[:2]
    if outputs["waypoints"].ndim != 4:
        raise ValueError("sequence waypoints must be [B,S,H,2]")
    def checked_mask(value, name):
        if value.shape != shape:
            raise ValueError(f"{name} must have shape [B,S]")
        if not ((value == 0) | (value == 1)).all():
            raise ValueError(f"{name} must contain only finite 0/1 values")
        return value.bool()

    selected = checked_mask(batch.get("supervision_mask", torch.ones(
        shape, dtype=torch.bool, device=outputs["waypoints"].device)), "supervision_mask")
    if not selected.any():
        raise ValueError("supervision_mask must select at least one real labeled step")
    # Empty reductions preserve a zero-gradient graph without reading masked
    # NaN placeholders (NaN * 0 would still poison the objective).
    def head_zero(key):
        return outputs.get(key, outputs["waypoints"]).reshape(-1)[:0].sum()

    zero = head_zero("waypoints")
    losses = {}

    def mask(name, legacy=None):
        key = name if name in batch else legacy
        valid = checked_mask(batch.get(key, torch.ones_like(selected)), name)
        return selected & valid

    for name in enabled:
        if name in waypoint_functions:
            valid = selected & batch["waypoint_mask"][..., 1:].bool().any(-1)
            losses[name] = (
                waypoint_functions[name](
                    {"waypoints": outputs["waypoints"][valid]},
                    batch["target_waypoints"][valid], batch["waypoint_mask"][valid],
                ) if valid.any() else zero
            )
        elif name in {"stop", "identity_or_binding", "visibility"}:
            output_key, label_key, valid_key = {
                "stop": ("stop_logit", "stop_target", "stop_label_valid"),
                "identity_or_binding": ("binding_logit", "binding_target", "binding_label_valid"),
                "visibility": ("visibility_logit", "target_visible", "visibility_label_valid"),
            }[name]
            valid = mask(valid_key, "identity_label_valid" if name == "visibility" else None)
            # Outside an autocast context, BF16 sigmoid rounds large positive
            # logits to exactly one and erases the positive-label gradient.
            # Keep the BCE computation in FP32 regardless of forward dtype.
            losses[name] = F.binary_cross_entropy_with_logits(
                outputs[output_key].squeeze(-1)[valid].float(), batch[label_key][valid].float(),
            ) if valid.any() else head_zero(output_key)
        elif name == "bbox":
            valid = mask("bbox_label_valid", "identity_label_valid")
            if valid.any():
                visible = batch["target_visible"]
                if visible.shape != shape or not ((visible[valid] == 0) | (visible[valid] == 1)).all():
                    raise ValueError("bbox labels require finite 0/1 target_visible at valid steps")
                valid = valid & (visible == 1)
            losses[name] = F.smooth_l1_loss(
                outputs["bbox_pred"][valid], batch["target_bbox"][valid],
            ) if valid.any() else head_zero("bbox_pred")
        elif name == "ego":
            valid = mask("ego_label_valid")
            if valid.any():
                # BF16 vector normalization can round the cosine above one,
                # making a regression loss negative, especially near zero yaw.
                predicted, target = outputs["xi_hat"][valid].float(), batch["ego_motion_target"][valid].float()
                losses[name] = F.smooth_l1_loss(predicted[..., :2], target[..., :2]) + (
                    1.0 - (F.normalize(predicted[..., 2:4], dim=-1, eps=1e-6)
                    * F.normalize(target[..., 2:4], dim=-1, eps=1e-6)).sum(-1).clamp(-1.0, 1.0)
                ).mean()
            else:
                losses[name] = zero
    if not losses:
        raise ValueError("at least one Phase-3 loss must be enabled")
    return sum(enabled[name] * value for name, value in losses.items()), losses
