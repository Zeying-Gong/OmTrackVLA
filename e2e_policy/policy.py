from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .dino_backbone import DinoEncoder
from .evggt import FrameGlobalBackbone
from .encoders import TargetEncoder, PointGoalEncoder, HistoryEncoder, TaskEncoder
from .heads import WaypointHead, StopHead, ForwardDynamics, InverseDynamics


@dataclass
class PolicyConfig:
    dino_ckpt: str = "/data/nfs/share/OmTrackVLA/models/dinov2/dinov2_vits14_pretrain.pth"
    dim: int = 384
    num_heads: int = 4
    mlp_ratio: float = 2.0
    evggt_depth: int = 4
    n_policy_queries: int = 4
    n_identity_tokens: int = 4
    horizon: int = 8
    action_dim: int = 4          # dx, dy, sin(yaw), cos(yaw) [+ optional speed]
    use_stop_head: bool = True
    use_forward_dyn: bool = True
    use_inverse_dyn: bool = True


class E2EFollowPolicy(nn.Module):
    """Unified target-conditioned navigation policy (architecture note 0814).

    Deployment path: DINOv2-S/14 (frozen) + eVGGT frame/global backbone
    + target/PointGoal/history/task encoders + waypoint head (+ stop head).

    Training-only: action-conditioned forward dynamics (future DINO/log-depth/
    free-space residuals), inverse dynamics, and frozen-teacher future features.
    """

    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.dino = DinoEncoder(cfg.dino_ckpt)
        self.target_enc = TargetEncoder(cfg.dim, cfg.n_identity_tokens)
        self.pointgoal_enc = PointGoalEncoder(cfg.dim)
        self.history_enc = HistoryEncoder(cfg.dim)
        self.task_enc = TaskEncoder(cfg.dim)
        self.backbone = FrameGlobalBackbone(
            cfg.dim, cfg.num_heads, cfg.mlp_ratio, cfg.evggt_depth, cfg.n_policy_queries
        )
        self.waypoint_head = WaypointHead(cfg.dim, cfg.horizon, cfg.action_dim)
        self.stop_head = StopHead(cfg.dim) if cfg.use_stop_head else None
        if cfg.use_forward_dyn:
            self.forward_dyn = ForwardDynamics(cfg.dim, cfg.action_dim, cfg.horizon)
        if cfg.use_inverse_dyn:
            self.inverse_dyn = InverseDynamics(cfg.dim, cfg.horizon, cfg.action_dim)
        self.future_pool = nn.Sequential(nn.Linear(cfg.dim, cfg.dim), nn.GELU(), nn.Linear(cfg.dim, cfg.dim))

    def forward(self, current_rgb,
                target_image=None, target_type=None, target_valid=None, target_confidence=None,
                pointgoal=None,
                history_rgb=None, history_motion=None,
                task_type=None, goal_spec=None,
                trajectory=None, future_rgb=None):
        """Returns a dict of outputs. `trajectory`/`future_rgb` only used in training."""
        cfg = self.cfg
        B = current_rgb.shape[0]
        C = cfg.dim

        cur_patches = self.dino.forward_patches(current_rgb)          # (B, P, C)

        if history_rgb is not None:
            K = history_rgb.shape[1]
            hp = self.dino.forward_patches(history_rgb.reshape(B * K, *history_rgb.shape[2:]))
            hp = hp.reshape(B, K, -1, C)
            cur_patches = torch.cat([hp, cur_patches[:, None]], dim=1)  # (B, T, P, C), current last
        else:
            cur_patches = cur_patches[:, None]                         # (B, 1, P, C)

        ctx = []
        if target_image is not None:
            tgt_patches = self.dino.forward_patches(target_image)     # (B, Pt, C)
            ctx.append(self.target_enc(tgt_patches, target_type, target_valid, target_confidence))
        else:
            ctx.append(current_rgb.new_zeros(B, 1, C))                # learnable-free null token

        if pointgoal is not None:
            ctx.append(self.pointgoal_enc(pointgoal).unsqueeze(1))

        if history_motion is not None:
            ctx.append(self.history_enc.motion_tokens(history_motion))

        if task_type is not None:
            ctx.append(self.task_enc(task_type, goal_spec).unsqueeze(1))

        context = torch.cat(ctx, dim=1) if ctx else current_rgb.new_zeros(B, 0, C)

        h_t, fused_patches = self.backbone(cur_patches, context)

        out = {
            "a_hat": self.waypoint_head(h_t),
            "h_t": h_t,
            "fused_patches": fused_patches,
            "cur_patches_teacher": cur_patches[:, -1].detach(),
        }
        if self.stop_head is not None:
            out["stop_logit"] = self.stop_head(h_t)

        if self.training:
            if future_rgb is not None:
                fut_patches = self.dino.forward_patches(future_rgb)
                out["future_patches_teacher"] = fut_patches.detach()
                h_future = self.future_pool(fut_patches.mean(dim=1))
                out["h_future"] = h_future
                if cfg.use_inverse_dyn and trajectory is not None:
                    out["a_inv"] = self.inverse_dyn(h_t, h_future.detach())
            if cfg.use_forward_dyn and trajectory is not None:
                out["forward"] = self.forward_dyn(h_t, fused_patches, trajectory)

        return out
