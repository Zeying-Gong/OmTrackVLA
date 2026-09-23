import torch
import torch.nn as nn


class WaypointHead(nn.Module):
    """h_t -> a_hat in R^(horizon x action_dim). Deployable policy head.

    The last two dims are (sin(yaw), cos(yaw)); they are L2-normalized so the
    heading always stays on the unit circle.
    """

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
        a = self.mlp(h_t).reshape(-1, self.horizon, self.action_dim)
        if self.action_dim >= 4:
            s, c = a[..., 2], a[..., 3]
            n = torch.sqrt(s * s + c * c).clamp_min(1e-6)
            sc = torch.stack([s / n, c / n], dim=-1)
            a = torch.cat([a[..., :2], sc], dim=-1)
        return a


class StopHead(nn.Module):
    """h_t -> stop probability logit."""

    def __init__(self, dim=384):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, h_t):
        return self.mlp(h_t).squeeze(-1)


class TargetStateHead(nn.Module):
    """h_t -> target relative state (dx, dy, visibility logit). Identity supervision."""

    def __init__(self, dim=384):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 3))

    def forward(self, h_t):
        return self.mlp(h_t)


class ForwardDynamics(nn.Module):
    """Training-only: (h_t, F_t, a*) -> future residuals.

    Predicts per-patch DINO residual (dim), log-depth residual (1),
    free-space/occupancy logit (1), and a global target relative state (dx, dy, vis).

    The action chunk is encoded with a temporal GRU (per-step embedding +
    sequential conditioning), so order and timing within the 8-step chunk are
    preserved rather than collapsed into a mean.
    """

    def __init__(self, dim=384, action_dim=4, horizon=8):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.act_embed = nn.Linear(action_dim, dim)
        self.act_gru = nn.GRUCell(dim, dim)
        self.cond_mlp = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.patch_mlp = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.dino_head = nn.Linear(dim, dim)
        self.depth_head = nn.Linear(dim, 1)
        self.free_head = nn.Linear(dim, 1)
        self.state_head = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 3))

    def forward(self, h_t, fused_patches, action_chunk):
        # action_chunk: (B, horizon, action_dim) -> temporal GRU encoding
        h = torch.zeros(h_t.shape[0], h_t.shape[-1], device=h_t.device, dtype=h_t.dtype)
        for k in range(action_chunk.shape[1]):
            h = self.act_gru(self.act_embed(action_chunk[:, k]), h)
        a = h                                                     # (B, dim)
        cond = self.cond_mlp(torch.cat([h_t, a], dim=-1))         # (B, dim)
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
