import torch
import torch.nn as nn


class WaypointHead(nn.Module):
    """h_t -> a_hat in R^(horizon x action_dim). Deployable policy head."""

    def __init__(self, dim=384, horizon=8, action_dim=4):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, horizon * action_dim),
        )

    def forward(self, h_t):
        return self.mlp(h_t).reshape(-1, self.horizon, self.action_dim)


class StopHead(nn.Module):
    """h_t -> stop probability logit."""

    def __init__(self, dim=384):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, h_t):
        return self.mlp(h_t).squeeze(-1)


class ForwardDynamics(nn.Module):
    """Training-only: (h_t, F_t, a*) -> future residuals.

    Predicts per-patch DINO residual (dim), log-depth residual (1),
    free-space/occupancy logit (1), and a global target relative state (dx, dy, vis).
    """

    def __init__(self, dim=384, action_dim=4, horizon=8):
        super().__init__()
        self.act_embed = nn.Sequential(nn.Linear(action_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.cond_mlp = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.patch_mlp = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.dino_head = nn.Linear(dim, dim)
        self.depth_head = nn.Linear(dim, 1)
        self.free_head = nn.Linear(dim, 1)
        self.state_head = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 3))

    def forward(self, h_t, fused_patches, action_chunk):
        a = self.act_embed(action_chunk).mean(dim=1)                 # (B, dim)
        cond = self.cond_mlp(torch.cat([h_t, a], dim=-1))            # (B, dim)
        B, P, C = fused_patches.shape
        cond_p = cond[:, None, :].expand(B, P, -1)
        feat = self.patch_mlp(torch.cat([fused_patches, cond_p], dim=-1))
        return {
            "dino_residual": self.dino_head(feat),
            "depth_residual": self.depth_head(feat).squeeze(-1),
            "free_logit": self.free_head(feat).squeeze(-1),
            "target_state": self.state_head(torch.cat([h_t, cond], dim=-1)),
        }


class InverseDynamics(nn.Module):
    """Training-only: (h_t, stopgrad(h_t+H)) -> a_inv."""

    def __init__(self, dim=384, horizon=8, action_dim=4):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.GELU(),
            nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, horizon * action_dim),
        )

    def forward(self, h_t, h_future):
        return self.mlp(torch.cat([h_t, h_future], dim=-1)).reshape(-1, self.horizon, self.action_dim)
