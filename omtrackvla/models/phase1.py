"""A compact, real Phase 1 identity and world-dynamics baseline."""
from __future__ import annotations

from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


class ResNetFeatureEncoder(nn.Module):
    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        network = resnet18(weights=None)
        self.stem = nn.Sequential(
            network.conv1,
            network.bn1,
            network.relu,
            network.maxpool,
            network.layer1,
            network.layer2,
            network.layer3,
            network.layer4,
        )
        self.project = nn.Conv2d(512, feature_dim, kernel_size=1)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image = (image - image.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]) / image.new_tensor(
            [0.229, 0.224, 0.225]
        )[None, :, None, None]
        feature_map = self.project(self.stem(image))
        pooled = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
        return feature_map, pooled


class Phase1WorldIdentityModel(nn.Module):
    """Shared visual encoder with isolated identity and geometry losses.

    The identity path consumes an in-memory crop from the one legal
    initialization bbox and the current RGB. The geometry path consumes two
    ego frames and predicts realized SE(2). Neither source receives policy
    waypoint supervision in Phase 1.
    """

    def __init__(self, feature_dim: int = 128, correlation_temperature: float = 0.1) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.correlation_temperature = float(correlation_temperature)
        self.encoder = ResNetFeatureEncoder(self.feature_dim)
        self.identity_memory = nn.GRUCell(self.feature_dim, self.feature_dim)
        self.identity_size = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, 2),
        )
        self.visibility = nn.Sequential(
            nn.Linear(self.feature_dim * 2 + 1, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, 1),
        )
        self.inverse_dynamics = nn.Sequential(
            nn.Linear(self.feature_dim * 3, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, 3),
        )
        self.forward_dynamics = nn.Sequential(
            nn.Linear(self.feature_dim + 3, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.action_free_next_state = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, self.feature_dim),
        )

    @staticmethod
    def _soft_center(correlation: torch.Tensor, temperature: float) -> torch.Tensor:
        batch, height, width = correlation.shape
        probability = F.softmax(correlation.flatten(1) / temperature, dim=1).reshape(batch, height, width)
        xs = torch.linspace(0.5 / width, 1.0 - 0.5 / width, width, device=correlation.device)
        ys = torch.linspace(0.5 / height, 1.0 - 0.5 / height, height, device=correlation.device)
        center_x = (probability * xs[None, None, :]).sum(dim=(1, 2))
        center_y = (probability * ys[None, :, None]).sum(dim=(1, 2))
        return torch.stack((center_x, center_y), dim=1)

    def forward(
        self,
        frame0: torch.Tensor,
        frame1: torch.Tensor,
        motion_condition: torch.Tensor | None = None,
        history: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        map0, pooled0 = self.encoder(frame0)
        map1, pooled1 = self.encoder(frame1)

        if history is None:
            history = frame1[:, None]
            history_mask = torch.ones(
                frame1.shape[0], 1, dtype=torch.bool, device=frame1.device
            )
        if history_mask is None or history.ndim != 5 or history_mask.shape != history.shape[:2]:
            raise ValueError("history/history_mask must have shapes [B,T,C,H,W] and [B,T]")
        batch, steps = history.shape[:2]
        _, temporal_features = self.encoder(history.reshape(batch * steps, *history.shape[2:]))
        temporal_features = temporal_features.reshape(batch, steps, self.feature_dim)
        identity_state = pooled0
        for step in range(steps):
            updated = self.identity_memory(temporal_features[:, step], identity_state)
            identity_state = torch.where(history_mask[:, step, None], updated, identity_state)

        reference = F.normalize(identity_state, dim=1)
        search = F.normalize(map1, dim=1)
        correlation = torch.einsum("bc,bchw->bhw", reference, search)
        center = self._soft_center(correlation, self.correlation_temperature)
        size = torch.sigmoid(self.identity_size(torch.cat((identity_state, pooled1), dim=1)))
        size = 0.02 + size * 0.96
        bbox = torch.cat((center - size / 2.0, center + size / 2.0), dim=1).clamp(0.0, 1.0)
        correlation_peak = correlation.flatten(1).amax(dim=1, keepdim=True)
        visibility_logit = self.visibility(
            torch.cat((identity_state, pooled1, correlation_peak), dim=1)
        ).squeeze(1)

        pair = torch.cat((pooled0, pooled1, pooled1 - pooled0), dim=1)
        motion = self.inverse_dynamics(pair)
        condition = motion if motion_condition is None else motion_condition
        next_feature = self.forward_dynamics(torch.cat((pooled0, condition), dim=1))
        action_free_feature = self.action_free_next_state(pooled0)
        return {
            "bbox": bbox,
            "visibility_logit": visibility_logit,
            "correlation": correlation,
            "motion": motion,
            "feature0": pooled0,
            "feature1": pooled1,
            "identity_state": identity_state,
            "next_feature": next_feature,
            "action_free_feature": action_free_feature,
        }


def compute_phase1_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
    visibility_negative_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    task_id = batch["task_id"]
    identity = task_id == 0
    geometry = task_id == 1
    zero = outputs["motion"].sum() * 0.0

    visibility_loss = zero
    bbox_loss = zero
    if identity.any():
        if visibility_negative_weight <= 0.0:
            raise ValueError("visibility_negative_weight must be positive")
        visibility_targets = batch["target_visible"][identity]
        visibility_weights = torch.where(
            visibility_targets > 0.5,
            torch.ones_like(visibility_targets),
            torch.full_like(visibility_targets, float(visibility_negative_weight)),
        )
        visibility_loss = F.binary_cross_entropy_with_logits(
            outputs["visibility_logit"][identity],
            visibility_targets,
            weight=visibility_weights,
        )
        visible = identity & (batch["target_visible"] > 0.5)
        if visible.any():
            bbox_loss = F.smooth_l1_loss(outputs["bbox"][visible], batch["target_bbox"][visible])

    inverse_loss = zero
    forward_loss = zero
    next_state_loss = zero
    if geometry.any():
        inverse_loss = F.smooth_l1_loss(outputs["motion"][geometry], batch["motion"][geometry])
        target_feature = outputs["feature1"][geometry].detach()
        forward_loss = F.mse_loss(outputs["next_feature"][geometry], target_feature)
        next_state_loss = F.mse_loss(outputs["action_free_feature"][geometry], target_feature)

    losses = {
        "identity_visibility": visibility_loss,
        "identity_bbox": bbox_loss,
        "inverse_dynamics": inverse_loss,
        "forward_dynamics": forward_loss,
        "action_free_next_state": next_state_loss,
    }
    total = sum(float(weights.get(name, 1.0)) * value for name, value in losses.items())
    losses["total"] = total
    return total, losses
