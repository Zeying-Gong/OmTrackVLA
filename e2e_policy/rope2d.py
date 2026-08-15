import torch


def _rotate_half(x):
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x[..., 0], x[..., 1]
    return torch.stack((-x2, x1), dim=-1).reshape(*x.shape[:-2], -1)


def build_2d_rotary(pos_x, pos_y, head_dim, theta=10000.0):
    """InternVL-style 2D RoPE.

    pos_x, pos_y: (N,) float coordinates.
    Returns cos, sin of shape (N, head_dim).
    """
    half = head_dim // 4
    freqs = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32, device=pos_x.device) / half))
    ang_x = pos_x[:, None] * freqs[None, :]
    ang_y = pos_y[:, None] * freqs[None, :]
    ang = torch.cat([ang_x, ang_y, ang_x, ang_y], dim=-1)  # (N, head_dim)
    return ang.cos(), ang.sin()


def apply_rotary(q, k, cos, sin):
    # q, k: (B, H, N, head_dim); cos, sin: (N, head_dim)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k
