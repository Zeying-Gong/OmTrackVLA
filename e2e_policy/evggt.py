import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope3d import (
    get_3d_mrope_ids_text_tokens,
    get_3d_mrope_ids_vae_tokens,
    build_3d_mrope_cos_sin,
    apply_rotary,
)


class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, attn_mask=None, cos=None, sin=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if cos is not None:
            q, k = apply_rotary(q, k, cos, sin)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class EvggtBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio)

    def forward(self, x, attn_mask=None, cos=None, sin=None):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask, cos=cos, sin=sin)
        x = x + self.mlp(self.norm2(x))
        return x


class FrameGlobalBackbone(nn.Module):
    """eVGGT-style backbone: depth groups of [Frame Attention -> Global Attention].

    Sequence layout: [frame patches from a (T, H, W) vision grid (T*P tokens)]
                     [context tokens (Cctx)] [policy queries (Q)].
    - Frame Attention: attends only within the same frame group (and self for
      non-patch tokens).
    - Global Attention: full self-attention across all tokens.
    3D mRoPE (Cosmos3/Qwen3VL): vision patches get (t, h, w) position ids from
    the local 3D grid; context/query tokens get monotonic text-style ids
    continuing after the vision grid.
    """

    def __init__(self, dim=384, num_heads=4, mlp_ratio=2.0, depth=4, n_policy_queries=4,
                 rope_theta=10000.0, mrope_section=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.n_policy_queries = n_policy_queries
        self.rope_theta = rope_theta
        self.mrope_section = mrope_section
        self.policy_queries = nn.Parameter(torch.zeros(1, n_policy_queries, dim))
        nn.init.trunc_normal_(self.policy_queries, std=0.02)
        self.blocks = nn.ModuleList([EvggtBlock(dim, num_heads, mlp_ratio) for _ in range(2 * depth)])

    def forward(self, frame_tokens, context_tokens, grid=None):
        """frame_tokens: (B, T, P, dim) temporal stack of frame patch grids.

        grid: (Hp, Wp) real patch grid of the frames. If None, a square grid
        is inferred from P (backward-compatible fallback).

        Returns (h_t, fused_patches):
          h_t: (B, dim) policy state (mean of policy query tokens).
          fused_patches: (B, P, dim) current (last) frame's patch features.
        """
        B, T, P, C = frame_tokens.shape
        Q = self.n_policy_queries
        if context_tokens is None or context_tokens.shape[1] == 0:
            context_tokens = frame_tokens.new_zeros(B, 0, C)
        frame_tokens = frame_tokens.reshape(B, T * P, C)
        queries = self.policy_queries.expand(B, -1, -1)
        seq = torch.cat([frame_tokens, context_tokens, queries], dim=1)
        N = seq.shape[1]
        device = frame_tokens.device

        if grid is not None:
            Hp, Wp = grid
        else:
            Wp = int(round(P ** 0.5))
            Hp = P // Wp
        vis_ids, next_off = get_3d_mrope_ids_vae_tokens(T, Hp, Wp, 0, device=device)
        ctx_ids, _ = get_3d_mrope_ids_text_tokens(N - T * P, next_off, device=device)
        position_ids = torch.cat([vis_ids, ctx_ids], dim=1)  # (3, N)
        cos, sin = build_3d_mrope_cos_sin(
            position_ids, self.head_dim, self.rope_theta, self.mrope_section, dtype=seq.dtype
        )

        idx = torch.arange(N, device=device)
        frame_id = torch.arange(T * P, device=device).div(P, rounding_mode="floor")
        same_frame = frame_id[:, None] == frame_id[None, :]  # (TP, TP)
        frame_mask = torch.zeros(N, N, dtype=torch.bool, device=device)
        frame_mask[:T * P, :T * P] = same_frame
        frame_mask = frame_mask | (idx[:, None] == idx[None, :])
        frame_attn_mask = torch.where(frame_mask, 0.0, float("-inf"))[None, None].to(seq.dtype)

        for i in range(self.depth):
            seq = self.blocks[2 * i](seq, attn_mask=frame_attn_mask, cos=cos, sin=sin)
            seq = self.blocks[2 * i + 1](seq, cos=cos, sin=sin)

        h_t = seq[:, -Q:].mean(dim=1)
        fused_patches = seq[:, (T - 1) * P:T * P]
        return h_t, fused_patches
