import math

import torch


def _rotate_half(x):
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x[..., 0], x[..., 1]
    return torch.stack((-x2, x1), dim=-1).reshape(*x.shape[:-2], -1)


def get_3d_mrope_ids_text_tokens(num_tokens, temporal_offset, device=None):
    """Text tokens: all three axes share the same monotonic position ids.

    Port of cosmos_framework .../sequence_packing/mrope.py
    `get_3d_mrope_ids_text_tokens`.

    Returns (ids (3, num_tokens), next_temporal_offset).
    """
    ids = torch.arange(num_tokens, dtype=torch.long, device=device) + int(temporal_offset)
    mrope_ids = ids.unsqueeze(0).expand(3, -1).contiguous()
    return mrope_ids, temporal_offset + num_tokens


def get_3d_mrope_ids_vae_tokens(
    grid_t,
    grid_h,
    grid_w,
    temporal_offset,
    reset_spatial_indices=True,
    device=None,
):
    """Vision tokens: local 3D grid (T, H, W), T-major flatten order.

    Port of cosmos_framework .../sequence_packing/mrope.py
    `get_3d_mrope_ids_vae_tokens` (integer, no FPS modulation).
    Flattening order is T-major: for each frame, iterate height then width.

    Returns (ids (3, grid_t*grid_h*grid_w), next_temporal_offset).
    """
    t_index = (
        torch.arange(grid_t, dtype=torch.long, device=device)
        .view(-1, 1)
        .expand(-1, grid_h * grid_w)
        .flatten()
        + int(temporal_offset)
    )
    h_index = (
        torch.arange(grid_h, dtype=torch.long, device=device)
        .view(1, -1, 1)
        .expand(grid_t, -1, grid_w)
        .flatten()
    )
    w_index = (
        torch.arange(grid_w, dtype=torch.long, device=device)
        .view(1, 1, -1)
        .expand(grid_t, grid_h, -1)
        .flatten()
    )
    if not reset_spatial_indices:
        spatial_offset = int(temporal_offset)
        h_index = h_index + spatial_offset
        w_index = w_index + spatial_offset
    mrope_ids = torch.stack([t_index, h_index, w_index], dim=0)
    next_temporal_offset = math.ceil(mrope_ids.max().item()) + 1
    return mrope_ids, next_temporal_offset


def build_3d_mrope_cos_sin(
    position_ids,
    head_dim,
    theta=10000.0,
    mrope_section=None,
    dtype=torch.float32,
):
    """Convert (3, N) t/h/w position ids into interleaved Qwen3VL mRoPE cos/sin.

    Port of Qwen3VLTextRotaryEmbedding.forward + apply_interleaved_mrope:
    - Each axis shares one inv_freq vector of length head_dim//2.
    - Frequency layout is reorganized from chunked [TT..HH..WW] to interleaved
      [T H W T H W ...] over the first 3*section[dim] slots (tail stays T-only).
    - emb = cat([freqs, freqs], -1) to reach head_dim, cos/sin from it.

    position_ids: (3, N) int64/float on any device.
    mrope_section: (t, h, w) counts summing to head_dim//2. Default splits
    head_dim//2 into three equal groups (e.g. [16,16,16] for head_dim=96).
    """
    half = head_dim // 2
    if mrope_section is None:
        base = half // 3
        mrope_section = [base, base, half - 2 * base]
    assert sum(mrope_section) == half, f"mrope_section {mrope_section} must sum to {half}"

    device = position_ids.device
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )  # (half,)

    freqs = inv_freq[None, None, :] * position_ids[:, :, None].float()  # (3, N, half)

    freqs_t = freqs[0].clone()  # (N, half), T axis
    for dim, offset in enumerate((1, 2), start=1):  # H, W
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]

    emb = torch.cat([freqs_t, freqs_t], dim=-1)  # (N, head_dim)
    cos = emb.cos()
    sin = emb.sin()
    return cos.to(dtype=dtype), sin.to(dtype=dtype)


def apply_rotary(q, k, cos, sin):
    """Interleaved mRoPE application, matching Qwen3VL.

    q, k: (B, H, N, head_dim); cos, sin: (N, head_dim).
    """
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    cos = torch.cat([cos[..., ::2], cos[..., 1::2]], dim=-1)
    sin = torch.cat([sin[..., ::2], sin[..., 1::2]], dim=-1)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k
