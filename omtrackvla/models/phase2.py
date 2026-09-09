"""Small waypoint decoder over frozen visual identity and simulated-UWB signals."""
from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from omtrackvla.data.phase2 import CONDITION_MODES


class Phase2WaypointPolicy(nn.Module):
    def __init__(self, hidden_dim: int = 128, mode_dim: int = 16, horizon: int = 8) -> None:
        super().__init__()
        if horizon < 2:
            raise ValueError("Phase 2 horizon must contain at least two points")
        self.horizon = int(horizon)
        self.mode_embedding = nn.Embedding(len(CONDITION_MODES), int(mode_dim))
        input_dim = 2 + 1 + 1 + 2 + 1 + 1 + 1 + int(mode_dim)
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), (self.horizon - 1) * 2),
        )

    def forward(
        self,
        visual_xy: torch.Tensor,
        visual_confidence: torch.Tensor,
        visual_valid: torch.Tensor,
        uwb_xy: torch.Tensor,
        uwb_quality: torch.Tensor,
        uwb_valid: torch.Tensor,
        uwb_age_s: torch.Tensor,
        condition_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        scalar = lambda value: value.reshape(value.shape[0], 1)
        features = torch.cat(
            (
                visual_xy,
                scalar(visual_confidence),
                scalar(visual_valid),
                uwb_xy,
                scalar(uwb_quality),
                scalar(uwb_valid),
                scalar(uwb_age_s),
                self.mode_embedding(condition_index),
            ),
            dim=1,
        )
        future = self.decoder(features).reshape(-1, self.horizon - 1, 2)
        anchor = torch.zeros(future.shape[0], 1, 2, dtype=future.dtype, device=future.device)
        waypoints = torch.cat((anchor, future), dim=1)
        safe_stop = condition_index == CONDITION_MODES.index("safe_stop")
        waypoints = torch.where(safe_stop[:, None, None], torch.zeros_like(waypoints), waypoints)
        return {"waypoints": waypoints, "safe_stop": safe_stop}


def compute_phase2_loss(
    outputs: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    point_mask = batch["waypoint_mask"].bool().clone()
    point_mask[:, 0] = False
    mask = point_mask.unsqueeze(-1).expand_as(outputs["waypoints"])
    if not mask.any():
        raise ValueError("Phase 2 loss requires at least one valid future waypoint")
    errors = torch.nn.functional.smooth_l1_loss(
        outputs["waypoints"], batch["waypoints"], reduction="none"
    )
    return errors[mask].mean()
