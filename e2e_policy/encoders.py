import torch
import torch.nn as nn


class TargetEncoder(nn.Module):
    """Target image (person/object/place crop) -> n_identity identity tokens.

    Cross-attention pooling over the target's DINO patch features, plus a
    target_type embedding and a projected (valid, confidence) conditioning.
    """

    def __init__(self, dim=384, n_identity_tokens=4, n_types=4):
        super().__init__()
        self.n_identity_tokens = n_identity_tokens
        self.queries = nn.Parameter(torch.zeros(1, n_identity_tokens, dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.scale = dim ** -0.5
        self.type_emb = nn.Embedding(n_types, dim)
        self.valid_proj = nn.Linear(2, dim)

    def forward(self, target_patches, target_type=None, target_valid=None, target_confidence=None):
        B, Pt, C = target_patches.shape
        q = self.queries.expand(B, -1, -1)
        q = self.q_proj(q)
        k = self.k_proj(target_patches)
        v = self.v_proj(target_patches)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = self.norm(attn @ v)
        if target_type is not None:
            t = self.type_emb(target_type.clamp(0, self.type_emb.num_embeddings - 1))
            out = out + t[:, None, :]
        if target_valid is not None:
            conf = target_confidence if target_confidence is not None else target_valid.float()
            cond = torch.stack([target_valid.float(), conf.float()], dim=-1)
            out = out + self.valid_proj(cond)[:, None, :]
        return out


class PointGoalEncoder(nn.Module):
    """(x, y, range, bearing, uncertainty, age, valid) -> 1 goal token."""

    def __init__(self, dim=384, in_dim=7):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, pointgoal):
        return self.norm(self.mlp(pointgoal))


class HistoryEncoder(nn.Module):
    """History Context Cache encoder.

    - scene memory tokens: mean-pooled DINO patch features per past frame.
    - motion tokens: per-step (v_x, v_y, omega) via MLP.
    A per-slot order embedding is added so the cache keeps temporal structure.
    """

    def __init__(self, dim=384, motion_dim=3, max_history=32):
        super().__init__()
        self.motion_mlp = nn.Sequential(nn.Linear(motion_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)
        self.pos_emb = nn.Embedding(max_history, dim)

    def scene_tokens(self, history_patches):
        """history_patches: (B, K, Ph, C) -> (B, K, C) scene memory tokens."""
        K = history_patches.shape[1]
        tok = history_patches.mean(dim=2)
        pos = self.pos_emb(torch.arange(K, device=history_patches.device))
        return self.norm(tok + pos[None])

    def motion_tokens(self, history_motion):
        """history_motion: (B, K, motion_dim) -> (B, K, C)."""
        K = history_motion.shape[1]
        tok = self.motion_mlp(history_motion)
        pos = self.pos_emb(torch.arange(K, device=history_motion.device))
        return self.norm(tok + pos[None])


class TaskEncoder(nn.Module):
    """Task Token + Goal Specification Token.

    task_type: person_follow | pointnav | imagegoal | objectnav ...
    goal_spec: [desired_distance, d_min, d_max, success_radius, terminal_mode].
    """

    def __init__(self, dim=384, n_tasks=8, spec_dim=5):
        super().__init__()
        self.task_emb = nn.Embedding(n_tasks, dim)
        self.spec_mlp = nn.Sequential(nn.Linear(spec_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, task_type, goal_spec=None):
        t = self.task_emb(task_type)
        if goal_spec is not None:
            t = t + self.spec_mlp(goal_spec)
        return self.norm(t)
