"""Training-only Architecture v1 heads and Phase 1 auxiliary objectives."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import roi_align

from omtrackvla.models.end_to_end import EndToEndFollowPolicy


def load_frozen_osnet_teacher(
    weights_path: str | Path,
    code_path: str | Path,
) -> tuple[nn.Module, dict[str, object]]:
    """Load the pinned OSNet teacher without constructing a person detector."""

    weights_path = Path(weights_path).expanduser().resolve(strict=True)
    code_path = Path(code_path).expanduser().resolve(strict=True)
    spec = importlib.util.spec_from_file_location("omtrackvla_phase1_osnet", code_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load OSNet module: {code_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or "classifier.weight" not in state:
        raise TypeError("OSNet teacher checkpoint has no classifier state")
    num_classes = int(state["classifier.weight"].shape[0])
    teacher = module.osnet_x0_25(num_classes=num_classes, pretrained=False)
    teacher.load_state_dict(state, strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    output_dim = int(state["classifier.weight"].shape[1])
    return teacher, {
        "kind": "frozen_osnet_x0_25_msmt17",
        "weights": str(weights_path),
        "code": str(code_path),
        "checkpoint_tensors": len(state),
        "checkpoint_parameters": sum(value.numel() for value in state.values()),
        "output_dim": output_dim,
        "trainable_parameters": 0,
        "deployment_input": False,
        "label_domain_crop_only": True,
    }


class ArchitectureV1Phase1Model(nn.Module):
    """Keep the deployment policy intact while attaching removable Phase 1 heads."""

    def __init__(
        self,
        policy: EndToEndFollowPolicy,
        *,
        action_input_mode: str = "correct",
        identity_teacher: nn.Module | None = None,
        identity_teacher_dim: int = 512,
    ) -> None:
        super().__init__()
        if action_input_mode not in {"correct", "zero", "shuffled"}:
            raise ValueError("action_input_mode must be correct, zero, or shuffled")
        self.policy = policy
        self.action_input_mode = action_input_mode
        self.identity_teacher = identity_teacher
        if self.identity_teacher is not None:
            self.identity_teacher.eval()
            for parameter in self.identity_teacher.parameters():
                parameter.requires_grad_(False)
            self.identity_teacher_projector = nn.Linear(
                policy.config.policy_dim, identity_teacher_dim
            )
            self.register_buffer(
                "identity_teacher_mean",
                torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "identity_teacher_std",
                torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
                persistent=False,
            )
        dim = policy.config.policy_dim
        self.action_embedding = nn.Sequential(
            nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.forward_dynamics = nn.Sequential(
            nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, dim), nn.GELU()
        )
        self.future_world_head = nn.Linear(dim, dim)
        self.future_target_xy_head = nn.Linear(dim, 2)
        self.future_visibility_head = nn.Linear(dim, 1)
        self.inverse_dynamics = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 4)
        )

    def train(self, mode: bool = True) -> "ArchitectureV1Phase1Model":
        super().train(mode)
        if self.identity_teacher is not None:
            self.identity_teacher.eval()
        return self

    def forward(
        self,
        *,
        transition_action: torch.Tensor,
        teacher_bbox: torch.Tensor | None = None,
        teacher_valid: torch.Tensor | None = None,
        **policy_inputs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        outputs = self.policy.forward_sequence(**policy_inputs)
        world = outputs["w_t"]
        target = outputs["z_target"]
        if world.shape[1] < 2:
            raise ValueError("Phase 1 World-Action training requires two policy steps")
        expected = (world.shape[0], world.shape[1] - 1, 3)
        if transition_action.shape != expected:
            raise ValueError(
                f"transition_action must have shape {expected}, got {tuple(transition_action.shape)}"
            )
        if self.action_input_mode == "zero":
            conditioned_action = torch.zeros_like(transition_action)
        elif self.action_input_mode == "shuffled":
            conditioned_action = torch.roll(transition_action, shifts=1, dims=0)
        else:
            conditioned_action = transition_action
        action = self.action_embedding(conditioned_action)
        hidden = self.forward_dynamics(
            torch.cat((world[:, :-1], target[:, :-1], action), dim=-1)
        )
        outputs.update(
            {
                "future_world_hat": self.future_world_head(hidden),
                "future_target_xy_hat": self.future_target_xy_head(hidden),
                "future_target_visibility_logit": self.future_visibility_head(
                    hidden
                ).squeeze(-1),
                "inverse_motion_hat": self.inverse_dynamics(
                    torch.cat((world[:, :-1], world[:, 1:]), dim=-1)
                ),
            }
        )
        if self.identity_teacher is not None:
            if teacher_bbox is None or teacher_valid is None:
                raise ValueError("OSNet teacher requires label-domain bbox and valid mask")
            if teacher_bbox.shape != (*world.shape[:2], 4):
                raise ValueError("teacher_bbox must be [B,S,4]")
            if teacher_valid.shape != world.shape[:2]:
                raise ValueError("teacher_valid must be [B,S]")
            rgb_history = policy_inputs["ego_rgb"]
            if rgb_history.ndim != 6 or rgb_history.shape[:2] != world.shape[:2]:
                raise ValueError("OSNet teacher requires sequence RGB history")
            frames = rgb_history[:, :, -1].flatten(0, 1)
            boxes_norm = teacher_bbox.flatten(0, 1).clamp(0.0, 1.0)
            valid = teacher_valid.flatten().bool()
            valid &= (boxes_norm[:, 2] > boxes_norm[:, 0]) & (
                boxes_norm[:, 3] > boxes_norm[:, 1]
            )
            safe_boxes = torch.where(
                valid[:, None],
                boxes_norm,
                boxes_norm.new_tensor([0.0, 0.0, 1.0, 1.0])[None],
            )
            boxes = torch.stack(
                (
                    torch.arange(
                        frames.shape[0], device=frames.device, dtype=frames.dtype
                    ),
                    safe_boxes[:, 0] * frames.shape[-1],
                    safe_boxes[:, 1] * frames.shape[-2],
                    safe_boxes[:, 2] * frames.shape[-1],
                    safe_boxes[:, 3] * frames.shape[-2],
                ),
                dim=1,
            )
            crops = roi_align(
                frames, boxes, output_size=(256, 128), spatial_scale=1.0, aligned=True
            )
            crops = (crops - self.identity_teacher_mean.to(crops)) / self.identity_teacher_std.to(crops)
            with torch.no_grad(), torch.autocast(
                device_type=crops.device.type, enabled=False
            ):
                teacher_embedding = self.identity_teacher(crops.float()).float()
            outputs["osnet_teacher_embedding"] = teacher_embedding.reshape(
                *world.shape[:2], -1
            )
            outputs["osnet_student_embedding"] = self.identity_teacher_projector(
                outputs["z_target"]
            )
            outputs["osnet_teacher_valid"] = valid.reshape(world.shape[:2])
        return outputs


def _motion_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    zero: torch.Tensor,
) -> torch.Tensor:
    mask = valid.bool()
    if not mask.any():
        return zero
    predicted = predicted[mask]
    target = target[mask]
    translation = F.smooth_l1_loss(predicted[..., :2], target[..., :2])
    predicted_angle = F.normalize(predicted[..., 2:4], dim=-1, eps=1.0e-6)
    target_angle = F.normalize(target[..., 2:4], dim=-1, eps=1.0e-6)
    angular = (1.0 - (predicted_angle * target_angle).sum(dim=-1)).mean()
    return translation + angular


def compute_phase1_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
    *,
    visibility_negative_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute only label-domain identity, ego and World-Action objectives."""

    zero = outputs["w_t"].sum() * 0.0
    identity_valid = batch["identity_label_valid"].bool()
    visible = identity_valid & batch["target_visible"].bool()
    bbox = (
        F.smooth_l1_loss(outputs["bbox_pred"][visible], batch["target_bbox"][visible])
        if visible.any()
        else zero
    )
    if identity_valid.any():
        visibility_raw = F.binary_cross_entropy_with_logits(
            outputs["visibility_logit"].squeeze(-1)[identity_valid],
            batch["target_visible"][identity_valid],
            reduction="none",
        )
        visibility_targets = batch["target_visible"][identity_valid]
        visibility_scale = torch.where(
            visibility_targets > 0.5,
            torch.ones_like(visibility_targets),
            torch.full_like(visibility_targets, visibility_negative_weight),
        )
        visibility = (visibility_raw * visibility_scale).mean()
        predicted_identity = F.normalize(
            outputs["z_target"][identity_valid], dim=-1, eps=1.0e-6
        )
        reference_identity = F.normalize(
            outputs["target_memory"][identity_valid].detach(), dim=-1, eps=1.0e-6
        )
        identity = (1.0 - (predicted_identity * reference_identity).sum(-1)).mean()
    else:
        visibility = zero
        identity = zero
    osnet_identity = zero
    teacher_keys = {
        "osnet_teacher_embedding",
        "osnet_student_embedding",
        "osnet_teacher_valid",
    }
    if teacher_keys.issubset(outputs):
        teacher_valid = outputs["osnet_teacher_valid"].bool()
        if teacher_valid.any():
            teacher_embedding = F.normalize(
                outputs["osnet_teacher_embedding"][teacher_valid].detach(),
                dim=-1,
                eps=1.0e-6,
            )
            student_embedding = F.normalize(
                outputs["osnet_student_embedding"][teacher_valid],
                dim=-1,
                eps=1.0e-6,
            )
            osnet_identity = (
                1.0 - (student_embedding * teacher_embedding).sum(dim=-1)
            ).mean()

    ego = _motion_loss(
        outputs["xi_hat"],
        batch["ego_motion_target"],
        batch["ego_motion_valid"],
        zero,
    )
    transition_valid = batch["transition_valid"].bool()
    if transition_valid.any():
        predicted_world = F.normalize(
            outputs["future_world_hat"][transition_valid], dim=-1, eps=1.0e-6
        )
        future_world = F.normalize(
            outputs["w_t"][:, 1:].detach()[transition_valid], dim=-1, eps=1.0e-6
        )
        future_latent = F.mse_loss(predicted_world, future_world)
    else:
        future_latent = zero

    target_xy_valid = batch["future_target_xy_valid"].bool()
    future_target_xy = (
        F.smooth_l1_loss(
            outputs["future_target_xy_hat"][target_xy_valid],
            batch["future_target_xy"][target_xy_valid],
        )
        if target_xy_valid.any()
        else zero
    )
    target_visibility_valid = batch["future_target_visibility_valid"].bool()
    future_visibility = (
        F.binary_cross_entropy_with_logits(
            outputs["future_target_visibility_logit"][target_visibility_valid],
            batch["future_target_visible"][target_visibility_valid],
        )
        if target_visibility_valid.any()
        else zero
    )
    world_action = future_latent + future_target_xy + future_visibility
    inverse = _motion_loss(
        outputs["inverse_motion_hat"],
        batch["ego_motion_target"][:, 1:],
        batch["transition_valid"],
        zero,
    )
    losses = {
        "bbox": bbox,
        "visibility": visibility,
        "identity_or_binding": identity,
        "osnet_identity": osnet_identity,
        "ego": ego,
        "future_latent": future_latent,
        "future_target_xy": future_target_xy,
        "future_visibility": future_visibility,
        "world_action": world_action,
        "inverse": inverse,
    }
    losses["total"] = sum(
        float(weights.get(name, 0.0)) * value
        for name, value in losses.items()
        if name not in {
            "total",
            "future_latent",
            "future_target_xy",
            "future_visibility",
        }
    )
    return losses["total"], losses


def _norm(parameters: Sequence[nn.Parameter]) -> float:
    values = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    return float(torch.stack(values).sum().sqrt().cpu()) if values else 0.0


def phase1_gradient_report(model: ArchitectureV1Phase1Model) -> dict[str, float]:
    named = list(model.named_parameters())
    adapters = [parameter for name, parameter in named if ".adapter." in name]
    policy = model.policy
    report = {
        "da3_adapter": _norm(adapters),
        "l11_projector": _norm(list(policy.l11_projector.parameters())),
        "target_attention": _norm(list(policy.target_attention.parameters())),
        "scene_attention": _norm(list(policy.scene_attention.parameters())),
        "motion_pair_head": _norm(list(policy.motion_pair_head.parameters())),
        "world_fusion": _norm(list(policy.world_fusion.parameters())),
        "bbox_head": _norm(list(policy.bbox_head.parameters())),
        "visibility_head": _norm(list(policy.visibility_head.parameters())),
        "action_embedding": _norm(list(model.action_embedding.parameters())),
        "forward_dynamics": _norm(list(model.forward_dynamics.parameters())),
        "future_world_head": _norm(list(model.future_world_head.parameters())),
        "future_target_xy_head": _norm(
            list(model.future_target_xy_head.parameters())
        ),
        "future_visibility_head": _norm(
            list(model.future_visibility_head.parameters())
        ),
        "inverse_dynamics": _norm(list(model.inverse_dynamics.parameters())),
    }
    report["identity_teacher_projector"] = (
        _norm(list(model.identity_teacher_projector.parameters()))
        if model.identity_teacher is not None
        else 0.0
    )
    return report
