"""Architecture v1 end-to-end target-following policy.

The deployment graph contains only the pinned DA3-SMALL L11 backbone,
target/world/UWB fusion, a recurrent policy state, and policy/diagnostic heads.
The single-step path remains the NEXT-025 smoke contract; forward_sequence
unrolls the same graph for NEXT-026 without introducing a second temporal model.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align


@dataclass(frozen=True)
class ArchitectureV1Config:
    image_height: int = 280
    image_width: int = 504
    history_size: int = 4
    patch_size: int = 14
    backbone_dim: int = 768
    policy_dim: int = 256
    attention_heads: int = 8
    horizon: int = 8
    roi_size: int = 3
    adapter_layers: int = 2
    adapter_bottleneck: int = 64
    uwb_age_limit_s: float = 1.0
    uwb_age_decay_s: float = 0.5
    target_motion_sigma_m_s: float = 0.5
    tag_height_mean_m: float = 1.0
    tag_height_sigma_m: float = 0.35
    calibration_sigma_px: float = 2.0

    @property
    def grid_height(self) -> int:
        return self.image_height // self.patch_size

    @property
    def grid_width(self) -> int:
        return self.image_width // self.patch_size

    @property
    def patch_count(self) -> int:
        return self.grid_height * self.grid_width

    def validate(self) -> None:
        if self.image_height % self.patch_size or self.image_width % self.patch_size:
            raise ValueError("image dimensions must be divisible by the DA3 patch size")
        if self.history_size < 2:
            raise ValueError("camera-token pair requires at least two history frames")
        if self.policy_dim % self.attention_heads:
            raise ValueError("policy_dim must be divisible by attention_heads")
        if self.horizon != 8:
            raise ValueError("Architecture v1 fixes the waypoint horizon at eight points")
        if self.roi_size != 3:
            raise ValueError("Architecture v1 fixes RoIAlign output to 3x3")
        if self.adapter_layers < 1 or self.adapter_bottleneck < 1:
            raise ValueError("at least one positive-width DA3 residual adapter is required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArchitectureV1Ablation:
    """Explicit, opt-in controls for the frozen Architecture-v1 ablations.

    The defaults are exactly the deployment main method.  Alternative values
    are only for ABL-V1-01--06 and must be recorded in each run manifest.
    """

    backbone_weights: str = "da3"
    backbone_tuning: str = "adapter"
    feature_fusion: str = "l11"
    target_pooling: str = "roi_align"
    temporal_fusion: str = "gru"
    ego_representation: str = "se2"
    uwb_early_fusion: str = "geometric"

    def validate(self) -> None:
        choices = {
            "backbone_weights": {"da3", "dinov2"},
            "backbone_tuning": {"adapter", "frozen"},
            "feature_fusion": {"l11", "l5_l11"},
            "target_pooling": {"roi_align", "bbox_mean"},
            "temporal_fusion": {"gru", "single_step"},
            "ego_representation": {"se2", "raw_camera_difference", "none"},
            "uwb_early_fusion": {"geometric", "none", "learned"},
        }
        for name, allowed in choices.items():
            value = getattr(self, name)
            if value not in allowed:
                raise ValueError(f"invalid {name}={value!r}; expected one of {sorted(allowed)}")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class ResidualAdapterBlock(nn.Module):
    """Wrap one frozen DA3 transformer block with a trainable residual adapter."""

    def __init__(self, base_block: nn.Module, embed_dim: int, bottleneck: int) -> None:
        super().__init__()
        self.base_block = base_block
        self.adapter = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, embed_dim),
        )
        nn.init.normal_(self.adapter[-1].weight, std=1.0e-3)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, value: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        output = self.base_block(value, *args, **kwargs)
        return output + self.adapter(output)


def install_residual_adapters(
    official_backbone: nn.Module,
    *,
    layers: int,
    bottleneck: int,
) -> list[str]:
    """Freeze DA3 and insert adapters inside its last transformer blocks."""

    for parameter in official_backbone.parameters():
        parameter.requires_grad_(False)
    if layers == 0:
        return []
    transformer = getattr(official_backbone, "pretrained", None)
    blocks = getattr(transformer, "blocks", None)
    embed_dim = getattr(transformer, "embed_dim", None)
    if blocks is None or embed_dim is None:
        raise TypeError("expected the official DA3 DinoV2 backbone structure")
    if layers > len(blocks):
        raise ValueError(f"requested {layers} adapters for only {len(blocks)} DA3 blocks")
    installed = []
    for index in range(len(blocks) - layers, len(blocks)):
        if isinstance(blocks[index], ResidualAdapterBlock):
            raise ValueError(f"DA3 block {index} already has an adapter")
        blocks[index] = ResidualAdapterBlock(blocks[index], int(embed_dim), bottleneck)
        installed.append(f"pretrained.blocks.{index}.adapter")
    return installed


class DA3SmallL11Backbone(nn.Module):
    """Differentiable access to official DA3-SMALL L11 tokens without heads."""

    def __init__(
        self,
        official_backbone: nn.Module,
        config: ArchitectureV1Config,
        ablation: ArchitectureV1Ablation | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.ablation = ablation or ArchitectureV1Ablation()
        self.ablation.validate()
        self.backbone = official_backbone
        self.adapter_paths = install_residual_adapters(
            self.backbone,
            layers=(config.adapter_layers if self.ablation.backbone_tuning == "adapter" else 0),
            bottleneck=config.adapter_bottleneck,
        )
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True) -> "DA3SmallL11Backbone":
        super().train(mode)
        # The pretrained transformer remains deterministic/frozen; only the
        # inserted adapters follow the requested training mode.
        self.backbone.eval()
        for module in self.backbone.modules():
            if isinstance(module, ResidualAdapterBlock):
                module.adapter.train(mode)
        return self

    def forward(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if rgb.ndim != 5 or tuple(rgb.shape[2:]) != (
            3,
            self.config.image_height,
            self.config.image_width,
        ):
            raise ValueError(
                "DA3 input must be [B,T,3,H,W] with the configured H/W; "
                f"received {tuple(rgb.shape)}"
            )
        normalized = (rgb - self.image_mean.to(rgb)) / self.image_std.to(rgb)
        result = self.backbone(normalized, ref_view_strategy="middle")
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("official DA3 backbone returned an unexpected value")
        feature_levels, _ = result
        if len(feature_levels) != 4:
            raise RuntimeError("DA3-SMALL must expose the configured L5/L7/L9/L11 levels")
        selected = (
            (feature_levels[-1],)
            if self.ablation.feature_fusion == "l11"
            else (feature_levels[0], feature_levels[-1])
        )
        spatial = torch.cat([level[0] for level in selected], dim=-1)
        camera = torch.cat([level[1] for level in selected], dim=-1)
        feature_dim = self.config.backbone_dim * len(selected)
        expected_spatial = (
            rgb.shape[0],
            rgb.shape[1],
            self.config.patch_count,
            feature_dim,
        )
        expected_camera = (rgb.shape[0], rgb.shape[1], feature_dim)
        if tuple(spatial.shape) != expected_spatial or tuple(camera.shape) != expected_camera:
            raise RuntimeError(
                f"unexpected DA3 L11 shapes: spatial={tuple(spatial.shape)}, camera={tuple(camera.shape)}"
            )
        return spatial, camera


class StubDA3SmallL11Backbone(nn.Module):
    """Small differentiable stand-in used only by CPU tests and smoke."""

    def __init__(
        self,
        config: ArchitectureV1Config,
        ablation: ArchitectureV1Ablation | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.ablation = ablation or ArchitectureV1Ablation()
        self.ablation.validate()
        self.patch_embed = nn.Conv2d(
            3,
            config.backbone_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.camera_token = nn.Linear(config.backbone_dim, config.backbone_dim)
        self.adapter = nn.Sequential(
            nn.LayerNorm(config.backbone_dim),
            nn.Linear(config.backbone_dim, config.adapter_bottleneck),
            nn.GELU(),
            nn.Linear(config.adapter_bottleneck, config.backbone_dim),
        )
        for parameter in self.patch_embed.parameters():
            parameter.requires_grad_(False)
        for parameter in self.camera_token.parameters():
            parameter.requires_grad_(False)
        if self.ablation.backbone_tuning == "frozen":
            for parameter in self.adapter.parameters():
                parameter.requires_grad_(False)

    def forward(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if rgb.ndim != 5 or tuple(rgb.shape[2:]) != (
            3,
            self.config.image_height,
            self.config.image_width,
        ):
            raise ValueError("stub DA3 input shape mismatch")
        batch, steps = rgb.shape[:2]
        feature_map = self.patch_embed(rgb.reshape(batch * steps, *rgb.shape[2:]))
        early_tokens = feature_map.flatten(2).transpose(1, 2)
        tokens = early_tokens + self.adapter(early_tokens)
        camera = self.camera_token(tokens.mean(dim=1))
        if self.ablation.feature_fusion == "l5_l11":
            early_camera = self.camera_token(early_tokens.mean(dim=1))
            tokens = torch.cat((early_tokens, tokens), dim=-1)
            camera = torch.cat((early_camera, camera), dim=-1)
        return (
            tokens.reshape(batch, steps, self.config.patch_count, tokens.shape[-1]),
            camera.reshape(batch, steps, camera.shape[-1]),
        )


class QueryAttention(nn.Module):
    """Single-query cross-attention with an optional per-patch additive bias."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        self.norm_query = nn.LayerNorm(dim)
        self.norm_tokens = nn.LayerNorm(dim)

    def forward(
        self,
        query: torch.Tensor,
        tokens: torch.Tensor,
        patch_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, count, _ = tokens.shape
        q = self.query(self.norm_query(query)).reshape(batch, self.heads, self.head_dim)
        k = self.key(self.norm_tokens(tokens)).reshape(batch, count, self.heads, self.head_dim)
        v = self.value(self.norm_tokens(tokens)).reshape(batch, count, self.heads, self.head_dim)
        logits = torch.einsum("bhd,bnhd->bhn", q, k) / math.sqrt(self.head_dim)
        if patch_bias is not None:
            if patch_bias.shape != (batch, count):
                raise ValueError("patch bias must be [B,N]")
            logits = logits + patch_bias[:, None, :]
        weights = logits.softmax(dim=-1)
        context = torch.einsum("bhn,bnhd->bhd", weights, v).reshape(batch, self.dim)
        return self.output(context), weights.mean(dim=1)


class GeometricUWBProjector(nn.Module):
    """Project base-frame UWB uncertainty into a Gaussian patch prior."""

    def __init__(self, config: ArchitectureV1Config) -> None:
        super().__init__()
        self.config = config
        offsets_1d = torch.tensor([-1.0 / 3.0, 0.0, 1.0 / 3.0]) * config.patch_size
        offset_y, offset_x = torch.meshgrid(offsets_1d, offsets_1d, indexing="ij")
        offsets = torch.stack((offset_x.flatten(), offset_y.flatten()), dim=-1)
        y = (torch.arange(config.grid_height, dtype=torch.float32) + 0.5) * config.patch_size
        x = (torch.arange(config.grid_width, dtype=torch.float32) + 0.5) * config.patch_size
        center_y, center_x = torch.meshgrid(y, x, indexing="ij")
        centers = torch.stack((center_x.flatten(), center_y.flatten()), dim=-1)
        self.register_buffer(
            "quadrature_points",
            centers[:, None, :] + offsets[None, :, :],
            persistent=False,
        )

    def forward(
        self,
        uwb_xy: torch.Tensor,
        covariance_xy: torch.Tensor,
        quality: torch.Tensor,
        age_s: torch.Tensor,
        valid: torch.Tensor,
        intrinsics: torch.Tensor,
        camera_from_base: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch = uwb_xy.shape[0]
        if uwb_xy.shape != (batch, 2) or covariance_xy.shape != (batch, 2, 2):
            raise ValueError("UWB xy/covariance must have shapes [B,2] and [B,2,2]")
        if intrinsics.shape != (batch, 3, 3) or camera_from_base.shape != (batch, 4, 4):
            raise ValueError("camera intrinsics/extrinsics must have shapes [B,3,3] and [B,4,4]")
        quality = quality.reshape(batch).clamp(0.0, 1.0)
        age_s = age_s.reshape(batch).clamp_min(0.0)
        valid = valid.reshape(batch).bool()

        motion_variance = (self.config.target_motion_sigma_m_s * age_s).square()
        covariance_3d = covariance_xy.new_zeros(batch, 3, 3)
        covariance_3d[:, :2, :2] = covariance_xy
        covariance_3d[:, :2, :2] += (
            torch.eye(2, device=uwb_xy.device, dtype=uwb_xy.dtype)[None]
            * motion_variance[:, None, None]
        )
        covariance_3d[:, 2, 2] = self.config.tag_height_sigma_m**2
        base_point = torch.cat(
            (
                uwb_xy,
                uwb_xy.new_full((batch, 1), self.config.tag_height_mean_m),
                uwb_xy.new_ones(batch, 1),
            ),
            dim=1,
        )
        camera_point = torch.einsum("bij,bj->bi", camera_from_base, base_point)[:, :3]
        rotation = camera_from_base[:, :3, :3]
        covariance_camera = rotation @ covariance_3d @ rotation.transpose(1, 2)

        x, y, z = camera_point.unbind(dim=1)
        z_safe = z.clamp_min(1.0e-3)
        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]
        mean_uv = torch.stack((fx * x / z_safe + cx, fy * y / z_safe + cy), dim=1)
        jacobian = intrinsics.new_zeros(batch, 2, 3)
        jacobian[:, 0, 0] = fx / z_safe
        jacobian[:, 0, 2] = -fx * x / z_safe.square()
        jacobian[:, 1, 1] = fy / z_safe
        jacobian[:, 1, 2] = -fy * y / z_safe.square()
        covariance_uv = jacobian @ covariance_camera @ jacobian.transpose(1, 2)
        covariance_uv = covariance_uv + (
            torch.eye(2, device=uwb_xy.device, dtype=uwb_xy.dtype)[None]
            * self.config.calibration_sigma_px**2
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_uv)
        eigenvalues = eigenvalues.clamp_min(1.0)
        covariance_uv = eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.transpose(1, 2)
        inverse = torch.linalg.inv(covariance_uv)
        determinant = torch.linalg.det(covariance_uv).clamp_min(1.0e-8)

        points = self.quadrature_points.to(uwb_xy)
        delta = points[None] - mean_uv[:, None, None, :]
        mahalanobis = torch.einsum("bnqi,bij,bnqj->bnq", delta, inverse, delta)
        density = torch.exp(-0.5 * mahalanobis) / (
            2.0 * math.pi * determinant.sqrt()
        )[:, None, None]
        patch_mass = density.mean(dim=2) * float(self.config.patch_size**2)
        rho_fov = patch_mass.sum(dim=1).clamp(0.0, 1.0)
        projectable = z > 1.0e-3
        fresh = age_s <= self.config.uwb_age_limit_s
        enabled = valid & fresh & projectable
        gamma = (
            enabled.to(uwb_xy.dtype)
            * quality
            * torch.exp(-age_s / self.config.uwb_age_decay_s)
            * rho_fov
        ).clamp(0.0, 1.0)
        normalized_mass = patch_mass / rho_fov[:, None].clamp_min(1.0e-8)
        uniform = 1.0 / float(self.config.patch_count)
        probability = (1.0 - gamma[:, None]) * uniform + gamma[:, None] * normalized_mass
        patch_bias = torch.log(probability.clamp_min(1.0e-12)) - math.log(uniform)
        return patch_bias, {
            "uwb_mean_uv": mean_uv,
            "uwb_covariance_uv": covariance_uv,
            "uwb_rho_fov": rho_fov,
            "uwb_gamma": gamma,
            "uwb_projectable": projectable,
        }


class EndToEndFollowPolicy(nn.Module):
    """Frozen Architecture v1 deployment graph."""

    def __init__(
        self,
        da3: nn.Module,
        config: ArchitectureV1Config | None = None,
        ablation: ArchitectureV1Ablation | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ArchitectureV1Config()
        self.config.validate()
        self.ablation = ablation or ArchitectureV1Ablation()
        self.ablation.validate()
        self.da3 = da3
        dim = self.config.policy_dim
        backbone_feature_dim = self.config.backbone_dim * (
            2 if self.ablation.feature_fusion == "l5_l11" else 1
        )
        self.l11_projector = nn.Linear(backbone_feature_dim, dim)
        self.unknown_person = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.unknown_person, std=0.02)
        self.target_attention = QueryAttention(dim, self.config.attention_heads)
        self.scene_query = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.scene_query, std=0.02)
        self.scene_attention = QueryAttention(dim, self.config.attention_heads)
        self.motion_pair_norm = nn.LayerNorm(backbone_feature_dim * 2)
        self.motion_pair_head = nn.Sequential(
            nn.Linear(backbone_feature_dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 4),
        )
        self.motion_embedding = nn.Sequential(
            nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        if self.ablation.ego_representation == "raw_camera_difference":
            self.raw_motion_embedding = nn.Sequential(
                nn.LayerNorm(backbone_feature_dim),
                nn.Linear(backbone_feature_dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
        self.world_fusion = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.uwb_projector = GeometricUWBProjector(self.config)
        self.uwb_encoder = nn.Sequential(
            nn.Linear(8, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        if self.ablation.uwb_early_fusion == "learned":
            self.learned_uwb_projector = nn.Linear(8, self.config.patch_count)
        self.fusion = nn.Sequential(
            nn.Linear(dim * 3 + 4, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.gru = nn.GRUCell(dim, dim)
        self.waypoint_head = nn.Linear(dim, (self.config.horizon - 1) * 2)
        self.stop_head = nn.Linear(dim, 1)
        diagnostic_dim = dim * 2
        self.bbox_head = nn.Sequential(
            nn.Linear(diagnostic_dim, dim), nn.GELU(), nn.Linear(dim, 4)
        )
        self.visibility_head = nn.Sequential(
            nn.Linear(diagnostic_dim, dim), nn.GELU(), nn.Linear(dim, 1)
        )
        self.binding_head = nn.Sequential(
            nn.Linear(dim * 2 + 4, dim), nn.GELU(), nn.Linear(dim, 1)
        )

    def initialize_gru_near_identity(self) -> None:
        """Start recurrent fusion close to the proven single-step pathway.

        A default random GRU transforms even the first policy token from a zero
        state, creating an avoidable optimization bottleneck. The candidate
        block therefore starts as an identity map for the input and recurrent
        state, while a negative update-gate bias keeps the current Fusion token
        dominant. This changes initialization only: every GRU parameter stays
        trainable and recurrent gradients remain live during the two-step
        Architecture-v1 unroll.
        """

        if self.gru.input_size != self.gru.hidden_size:
            raise ValueError("near-identity GRU initialization requires equal dimensions")
        dim = self.gru.hidden_size
        with torch.no_grad():
            self.gru.weight_ih.zero_()
            self.gru.weight_hh.zero_()
            self.gru.bias_ih.zero_()
            self.gru.bias_hh.zero_()
            identity = torch.eye(
                dim,
                dtype=self.gru.weight_ih.dtype,
                device=self.gru.weight_ih.device,
            )
            # PyTorch GRU gate order is reset, update, new.
            self.gru.weight_ih[2 * dim : 3 * dim].copy_(identity)
            self.gru.weight_hh[2 * dim : 3 * dim].copy_(identity * 0.1)
            self.gru.bias_ih[dim : 2 * dim].fill_(-5.0)

    def _target_memory(
        self,
        initial_tokens: torch.Tensor,
        initial_bbox: torch.Tensor,
        visual_initialization_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch = initial_tokens.shape[0]
        dim = self.config.policy_dim
        feature_map = initial_tokens.reshape(
            batch, self.config.grid_height, self.config.grid_width, dim
        ).permute(0, 3, 1, 2)
        bbox = initial_bbox.clamp(0.0, 1.0)
        valid_box = (bbox[:, 2] > bbox[:, 0]) & (bbox[:, 3] > bbox[:, 1])
        valid = visual_initialization_valid.reshape(batch).bool() & valid_box
        safe_bbox = torch.where(
            valid[:, None],
            bbox,
            bbox.new_tensor([0.0, 0.0, 1.0, 1.0])[None],
        )
        boxes = torch.stack(
            (
                torch.arange(batch, device=bbox.device, dtype=bbox.dtype),
                safe_bbox[:, 0] * self.config.grid_width,
                safe_bbox[:, 1] * self.config.grid_height,
                safe_bbox[:, 2] * self.config.grid_width,
                safe_bbox[:, 3] * self.config.grid_height,
            ),
            dim=1,
        )
        if self.ablation.target_pooling == "roi_align":
            pooled = roi_align(
                feature_map,
                boxes,
                output_size=(self.config.roi_size, self.config.roi_size),
                spatial_scale=1.0,
                aligned=True,
            ).mean(dim=(2, 3))
        else:
            y = (
                torch.arange(self.config.grid_height, device=bbox.device, dtype=bbox.dtype)
                + 0.5
            ) / self.config.grid_height
            x = (
                torch.arange(self.config.grid_width, device=bbox.device, dtype=bbox.dtype)
                + 0.5
            ) / self.config.grid_width
            grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
            inside = (
                (grid_x[None] >= bbox[:, 0, None, None])
                & (grid_x[None] <= bbox[:, 2, None, None])
                & (grid_y[None] >= bbox[:, 1, None, None])
                & (grid_y[None] <= bbox[:, 3, None, None])
            ).flatten(1)
            # Tiny valid boxes can contain no patch centre.  In that case use
            # the nearest patch to the box centre instead of silently emitting
            # an all-zero target token.
            empty = ~inside.any(dim=1)
            if empty.any():
                centre_x = (bbox[:, 0] + bbox[:, 2]) * 0.5
                centre_y = (bbox[:, 1] + bbox[:, 3]) * 0.5
                distance = (
                    (grid_x.flatten()[None] - centre_x[:, None]).square()
                    + (grid_y.flatten()[None] - centre_y[:, None]).square()
                )
                nearest = F.one_hot(
                    distance.argmin(dim=1), num_classes=self.config.patch_count
                ).bool()
                inside = torch.where(empty[:, None], nearest, inside)
            weights = inside.to(initial_tokens.dtype)
            pooled = (initial_tokens * weights[..., None]).sum(dim=1) / weights.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)
        unknown = self.unknown_person[None].expand(batch, -1)
        return torch.where(valid[:, None], pooled, unknown)

    @staticmethod
    def _scalar(value: torch.Tensor, batch: int, name: str) -> torch.Tensor:
        if value.numel() != batch:
            raise ValueError(f"{name} must contain one scalar per batch item")
        return value.reshape(batch)

    def forward(
        self,
        *,
        initial_rgb: torch.Tensor,
        initial_bbox: torch.Tensor,
        ego_rgb: torch.Tensor,
        visual_initialization_valid: torch.Tensor,
        rgb_valid: torch.Tensor,
        binding_valid: torch.Tensor,
        uwb_xy: torch.Tensor,
        uwb_covariance_xy: torch.Tensor,
        uwb_quality: torch.Tensor,
        uwb_age_s: torch.Tensor,
        uwb_valid: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera_from_base: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
        _target_memory_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = initial_rgb.shape[0]
        if initial_rgb.shape != (
            batch,
            3,
            self.config.image_height,
            self.config.image_width,
        ):
            raise ValueError("initial_rgb shape mismatch")
        if ego_rgb.shape != (
            batch,
            self.config.history_size,
            3,
            self.config.image_height,
            self.config.image_width,
        ):
            raise ValueError("ego_rgb shape mismatch")
        if initial_bbox.shape != (batch, 4):
            raise ValueError("initial_bbox must be normalized xyxy with shape [B,4]")

        history_spatial, history_camera = self.da3(ego_rgb)
        current_tokens = self.l11_projector(history_spatial[:, -1])
        if _target_memory_override is None:
            initial_spatial, _ = self.da3(initial_rgb[:, None])
            initial_tokens = self.l11_projector(initial_spatial[:, 0])
            target_memory = self._target_memory(
                initial_tokens,
                initial_bbox,
                visual_initialization_valid,
            )
        else:
            if _target_memory_override.shape != (
                batch,
                self.config.policy_dim,
            ):
                raise ValueError("internal target memory override shape mismatch")
            target_memory = _target_memory_override

        uwb_valid_flat = self._scalar(uwb_valid, batch, "uwb_valid")
        valid_float = uwb_valid_flat.to(uwb_xy.dtype)
        covariance_features = torch.stack(
            (
                uwb_covariance_xy[:, 0, 0],
                uwb_covariance_xy[:, 0, 1],
                uwb_covariance_xy[:, 1, 1],
            ),
            dim=1,
        )
        uwb_continuous = torch.cat(
            (
                uwb_xy,
                covariance_features,
                self._scalar(uwb_quality, batch, "uwb_quality")[:, None],
                self._scalar(uwb_age_s, batch, "uwb_age_s")[:, None],
                valid_float[:, None],
            ),
            dim=1,
        )
        uwb_continuous = torch.cat(
            (
                uwb_continuous[:, :7] * valid_float[:, None],
                uwb_continuous[:, 7:],
            ),
            dim=1,
        )

        # Covariance eigendecomposition/inversion is intentionally kept in
        # float32 even when the visual path uses bfloat16 autocast.
        with torch.autocast(device_type=uwb_xy.device.type, enabled=False):
            geometric_uwb_bias, uwb_diagnostics = self.uwb_projector(
                uwb_xy.float(),
                uwb_covariance_xy.float(),
                uwb_quality.float(),
                uwb_age_s.float(),
                uwb_valid,
                camera_intrinsics.float(),
                camera_from_base.float(),
            )
        if self.ablation.uwb_early_fusion == "geometric":
            uwb_bias = geometric_uwb_bias.to(current_tokens)
        elif self.ablation.uwb_early_fusion == "none":
            uwb_bias = torch.zeros_like(geometric_uwb_bias).to(current_tokens)
        else:
            uwb_bias = self.learned_uwb_projector(uwb_continuous)
            uwb_bias = torch.tanh(uwb_bias) * valid_float[:, None]
            uwb_bias = uwb_bias.to(current_tokens)
        z_target, target_attention = self.target_attention(
            target_memory, current_tokens, uwb_bias
        )
        scene_query = self.scene_query[None].expand(batch, -1)
        z_scene, scene_attention = self.scene_attention(scene_query, current_tokens)

        camera_pair = torch.cat(
            (history_camera[:, -2], history_camera[:, -1]), dim=1
        )
        xi_hat = self.motion_pair_head(self.motion_pair_norm(camera_pair))
        if self.ablation.ego_representation == "se2":
            z_ego = self.motion_embedding(xi_hat)
        elif self.ablation.ego_representation == "raw_camera_difference":
            z_ego = self.raw_motion_embedding(
                history_camera[:, -1] - history_camera[:, -2]
            )
        else:
            z_ego = torch.zeros_like(z_scene)
        world = self.world_fusion(torch.cat((z_scene, z_ego), dim=1))

        z_uwb = self.uwb_encoder(uwb_continuous)

        masks = torch.stack(
            (
                self._scalar(
                    visual_initialization_valid,
                    batch,
                    "visual_initialization_valid",
                ),
                uwb_valid_flat,
                self._scalar(rgb_valid, batch, "rgb_valid"),
                self._scalar(binding_valid, batch, "binding_valid"),
            ),
            dim=1,
        ).to(uwb_xy.dtype)
        recurrent_input = self.fusion(
            torch.cat((z_target, world, z_uwb, masks), dim=1)
        )
        if hidden_state is None:
            hidden_state = recurrent_input.new_zeros(batch, self.config.policy_dim)
        if hidden_state.shape != (batch, self.config.policy_dim):
            raise ValueError("hidden_state shape mismatch")
        state = (
            self.gru(recurrent_input, hidden_state)
            if self.ablation.temporal_fusion == "gru"
            else recurrent_input
        )
        future = self.waypoint_head(state).reshape(
            batch, self.config.horizon - 1, 2
        )
        waypoints = torch.cat((future.new_zeros(batch, 1, 2), future), dim=1)
        diagnostic = torch.cat((z_target, current_tokens.mean(dim=1)), dim=1)
        box_parameters = torch.sigmoid(self.bbox_head(diagnostic))
        box_center = box_parameters[:, :2]
        box_size = 0.02 + 0.96 * box_parameters[:, 2:]
        bbox_pred = torch.cat(
            (box_center - box_size / 2.0, box_center + box_size / 2.0), dim=1
        ).clamp(0.0, 1.0)
        binding_logit = self.binding_head(
            torch.cat((z_target, z_uwb, masks), dim=1)
        )
        can_bind = (
            (1.0 - masks[:, 3]).clamp(0.0, 1.0)
            * masks[:, 1].clamp(0.0, 1.0)
            * masks[:, 2].clamp(0.0, 1.0)
        )
        binding_update = torch.sigmoid(binding_logit).squeeze(1) * can_bind
        target_memory_next = (
            target_memory * (1.0 - binding_update[:, None])
            + z_target * binding_update[:, None]
        )
        result = {
            "waypoints": waypoints,
            "stop_logit": self.stop_head(state),
            "bbox_pred": bbox_pred,
            "visibility_logit": self.visibility_head(diagnostic),
            "binding_logit": binding_logit,
            "hidden_state": state,
            "target_memory": target_memory,
            "target_memory_next": target_memory_next,
            "z_target": z_target,
            "z_scene": z_scene,
            "z_ego": z_ego,
            "w_t": world,
            "z_uwb": z_uwb,
            "xi_hat": xi_hat,
            "target_attention": target_attention,
            "scene_attention": scene_attention,
            "uwb_patch_bias": uwb_bias,
            "uwb_geometric_patch_bias": geometric_uwb_bias.to(current_tokens),
        }
        result.update(uwb_diagnostics)
        return result

    def forward_sequence(
        self,
        *,
        initial_rgb: torch.Tensor,
        initial_bbox: torch.Tensor,
        ego_rgb: torch.Tensor,
        visual_initialization_valid: torch.Tensor,
        rgb_valid: torch.Tensor,
        binding_valid: torch.Tensor,
        uwb_xy: torch.Tensor,
        uwb_covariance_xy: torch.Tensor,
        uwb_quality: torch.Tensor,
        uwb_age_s: torch.Tensor,
        uwb_valid: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera_from_base: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Unroll the frozen deployment step over consecutive policy states.

        The target initialization is kept fixed and the GRU hidden state is the
        only recurrent policy state passed between steps. This intentionally
        reuses forward so the training graph cannot diverge from deployment.
        """

        if ego_rgb.ndim != 6:
            raise ValueError("sequence ego_rgb must be [B,S,T,3,H,W]")
        batch, sequence_steps = ego_rgb.shape[:2]
        if sequence_steps < 1:
            raise ValueError("sequence must contain at least one policy step")

        def step_value(value: torch.Tensor, step: int, trailing_dims: int) -> torch.Tensor:
            if value.shape[0] != batch:
                raise ValueError("sequence tensor batch dimension mismatch")
            if value.ndim == trailing_dims + 2:
                if value.shape[1] != sequence_steps:
                    raise ValueError("sequence tensor step dimension mismatch")
                return value[:, step]
            if value.ndim == trailing_dims + 1:
                return value
            raise ValueError("invalid static/sequence tensor rank")

        outputs: list[dict[str, torch.Tensor]] = []
        state = hidden_state
        target_memory = None
        binding_state = None
        for step in range(sequence_steps):
            requested_binding = step_value(binding_valid, step, 0)
            current_binding = (
                requested_binding
                if binding_state is None
                else torch.maximum(requested_binding, binding_state)
            )
            current_uwb_valid = step_value(uwb_valid, step, 0)
            current_rgb_valid = step_value(rgb_valid, step, 0)
            current = self.forward(
                initial_rgb=initial_rgb,
                initial_bbox=initial_bbox,
                ego_rgb=ego_rgb[:, step],
                visual_initialization_valid=visual_initialization_valid,
                rgb_valid=current_rgb_valid,
                binding_valid=current_binding,
                uwb_xy=step_value(uwb_xy, step, 1),
                uwb_covariance_xy=step_value(uwb_covariance_xy, step, 2),
                uwb_quality=step_value(uwb_quality, step, 0),
                uwb_age_s=step_value(uwb_age_s, step, 0),
                uwb_valid=current_uwb_valid,
                camera_intrinsics=step_value(camera_intrinsics, step, 2),
                camera_from_base=step_value(camera_from_base, step, 2),
                hidden_state=state,
                _target_memory_override=target_memory,
            )
            state = current["hidden_state"]
            target_memory = current["target_memory_next"]
            predicted_binding = (
                torch.sigmoid(current["binding_logit"]).squeeze(1)
                * current_uwb_valid
                * current_rgb_valid
            )
            binding_state = torch.maximum(current_binding, predicted_binding)
            outputs.append(current)
        return {
            key: torch.stack([output[key] for output in outputs], dim=1)
            for key in outputs[0]
        }


def waypoint_only_loss(
    outputs: Mapping[str, torch.Tensor],
    target_waypoints: torch.Tensor,
    waypoint_mask: torch.Tensor,
) -> torch.Tensor:
    predicted = outputs["waypoints"]
    if target_waypoints.shape != predicted.shape or waypoint_mask.shape != predicted.shape[:-1]:
        raise ValueError("waypoint target/mask shape mismatch")
    mask = waypoint_mask.bool().clone()
    mask[..., 0] = False
    expanded = mask[..., None].expand_as(predicted)
    if not expanded.any():
        raise ValueError("waypoint-only loss requires at least one valid future point")
    loss = F.smooth_l1_loss(predicted, target_waypoints, reduction="none")
    return loss[expanded].mean()


def waypoint_delta_loss(
    outputs: Mapping[str, torch.Tensor],
    target_waypoints: torch.Tensor,
    waypoint_mask: torch.Tensor,
) -> torch.Tensor:
    """Supervise local waypoint increments so paths cannot shrink toward zero."""

    predicted = outputs["waypoints"]
    if target_waypoints.shape != predicted.shape or waypoint_mask.shape != predicted.shape[:-1]:
        raise ValueError("waypoint target/mask shape mismatch")
    segment_mask = waypoint_mask[..., 1:].bool() & waypoint_mask[..., :-1].bool()
    expanded = segment_mask[..., None].expand_as(predicted[..., 1:, :])
    if not expanded.any():
        return predicted.sum() * 0.0
    predicted_delta = predicted[..., 1:, :] - predicted[..., :-1, :]
    target_delta = target_waypoints[..., 1:, :] - target_waypoints[..., :-1, :]
    loss = F.smooth_l1_loss(predicted_delta, target_delta, reduction="none")
    return loss[expanded].mean()


def waypoint_terminal_loss(
    outputs: Mapping[str, torch.Tensor],
    target_waypoints: torch.Tensor,
    waypoint_mask: torch.Tensor,
) -> torch.Tensor:
    """Give the last valid future point direct weight against under-travel."""

    predicted = outputs["waypoints"]
    if target_waypoints.shape != predicted.shape or waypoint_mask.shape != predicted.shape[:-1]:
        raise ValueError("waypoint target/mask shape mismatch")
    future_mask = waypoint_mask.bool().clone()
    future_mask[..., 0] = False
    horizon_indices = torch.arange(
        predicted.shape[-2], device=predicted.device, dtype=torch.long
    )
    last_index = torch.where(future_mask, horizon_indices, -1).amax(dim=-1)
    valid = last_index >= 0
    if not valid.any():
        return predicted.sum() * 0.0
    gather_index = last_index.clamp_min(0)[..., None, None].expand(
        *last_index.shape, 1, predicted.shape[-1]
    )
    predicted_terminal = predicted.gather(-2, gather_index).squeeze(-2)
    target_terminal = target_waypoints.gather(-2, gather_index).squeeze(-2)
    return F.smooth_l1_loss(
        predicted_terminal[valid], target_terminal[valid]
    )


def waypoint_radial_progress_loss(
    outputs: Mapping[str, torch.Tensor],
    target_waypoints: torch.Tensor,
    waypoint_mask: torch.Tensor,
) -> torch.Tensor:
    """Supervise per-horizon radial progress without rewarding zig-zag distance."""

    predicted = outputs["waypoints"]
    if target_waypoints.shape != predicted.shape or waypoint_mask.shape != predicted.shape[:-1]:
        raise ValueError("waypoint target/mask shape mismatch")
    future_mask = waypoint_mask.bool().clone()
    future_mask[..., 0] = False
    if not future_mask.any():
        return predicted.sum() * 0.0
    predicted_delta = predicted - predicted[..., :1, :]
    target_delta = target_waypoints - target_waypoints[..., :1, :]
    target_radius = torch.linalg.vector_norm(target_delta, dim=-1)
    future_mask &= target_radius > 1.0e-6
    if not future_mask.any():
        return predicted.sum() * 0.0
    target_direction = target_delta / target_radius.clamp_min(1.0e-6)[..., None]
    predicted_radius = (predicted_delta * target_direction).sum(dim=-1)
    return F.smooth_l1_loss(
        predicted_radius[future_mask], target_radius[future_mask]
    )


def waypoint_path_length_loss(
    outputs: Mapping[str, torch.Tensor],
    target_waypoints: torch.Tensor,
    waypoint_mask: torch.Tensor,
) -> torch.Tensor:
    """Directly supervise travelled distance while position loss anchors shape."""

    predicted = outputs["waypoints"]
    if target_waypoints.shape != predicted.shape or waypoint_mask.shape != predicted.shape[:-1]:
        raise ValueError("waypoint target/mask shape mismatch")
    segment_mask = waypoint_mask[..., 1:].bool() & waypoint_mask[..., :-1].bool()
    valid = segment_mask.any(dim=-1)
    if not valid.any():
        return predicted.sum() * 0.0
    predicted_delta = predicted[..., 1:, :] - predicted[..., :-1, :]
    target_delta = target_waypoints[..., 1:, :] - target_waypoints[..., :-1, :]
    predicted_length = (
        torch.linalg.vector_norm(predicted_delta, dim=-1) * segment_mask
    ).sum(dim=-1)
    target_length = (
        torch.linalg.vector_norm(target_delta, dim=-1) * segment_mask
    ).sum(dim=-1)
    return F.smooth_l1_loss(predicted_length[valid], target_length[valid])


def compute_architecture_v1_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
    *,
    world_action_variant: str = "b",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute deployment objectives plus optional training-only dynamics losses."""

    if world_action_variant not in {"b", "latent_only"}:
        raise ValueError("world_action_variant must be 'b' or 'latent_only'")

    dynamics_enabled = any(
        float(weights.get(name, 0.0)) != 0.0
        for name in ("world_action", "inverse")
    )
    dynamics_keys = {
        "future_world_hat",
        "future_target_xy_hat",
        "future_target_visibility_logit",
        "inverse_motion_hat",
    }
    if dynamics_enabled and not dynamics_keys.issubset(outputs):
        missing = sorted(dynamics_keys.difference(outputs))
        raise ValueError(f"training-only dynamics outputs are missing: {missing}")
    zero = outputs["waypoints"].sum() * 0.0
    losses: dict[str, torch.Tensor] = {
        "waypoint": waypoint_only_loss(
            outputs, batch["target_waypoints"], batch["waypoint_mask"]
        ),
        "waypoint_delta": waypoint_delta_loss(
            outputs, batch["target_waypoints"], batch["waypoint_mask"]
        ),
        "waypoint_terminal": waypoint_terminal_loss(
            outputs, batch["target_waypoints"], batch["waypoint_mask"]
        ),
        "waypoint_radial_progress": waypoint_radial_progress_loss(
            outputs, batch["target_waypoints"], batch["waypoint_mask"]
        ),
        "waypoint_path_length": waypoint_path_length_loss(
            outputs, batch["target_waypoints"], batch["waypoint_mask"]
        ),
    }
    losses["stop"] = F.binary_cross_entropy_with_logits(
        outputs["stop_logit"].squeeze(-1), batch["stop_target"]
    )

    identity_valid = batch["identity_label_valid"].bool()
    visible = identity_valid & batch["target_visible"].bool()
    losses["bbox"] = (
        F.smooth_l1_loss(outputs["bbox_pred"][visible], batch["target_bbox"][visible])
        if visible.any()
        else zero
    )
    losses["visibility"] = (
        F.binary_cross_entropy_with_logits(
            outputs["visibility_logit"].squeeze(-1)[identity_valid],
            batch["target_visible"][identity_valid],
        )
        if identity_valid.any()
        else zero
    )
    losses["identity_or_binding"] = F.binary_cross_entropy_with_logits(
        outputs["binding_logit"].squeeze(-1), batch["binding_target"]
    )

    predicted_motion = outputs["xi_hat"]
    target_motion = batch["ego_motion_target"]
    if predicted_motion.shape != target_motion.shape or predicted_motion.shape[-1] != 4:
        raise ValueError("ego motion prediction/target must share [...,4] shape")
    translation_loss = F.smooth_l1_loss(
        predicted_motion[..., :2], target_motion[..., :2]
    )
    predicted_angle = F.normalize(predicted_motion[..., 2:4], dim=-1, eps=1.0e-6)
    target_angle = F.normalize(target_motion[..., 2:4], dim=-1, eps=1.0e-6)
    angle_loss = (1.0 - (predicted_angle * target_angle).sum(dim=-1)).mean()
    losses["ego"] = translation_loss + angle_loss
    losses["future_latent"] = zero
    losses["future_target_xy"] = zero
    losses["future_visibility"] = zero
    losses["world_action"] = zero
    losses["inverse"] = zero
    if dynamics_keys.issubset(outputs):
        transition_action = batch["transition_action"]
        transition_valid = batch["transition_valid"].bool()
        expected_action = (*outputs["w_t"].shape[:2], 3)
        expected_action = (
            expected_action[0], expected_action[1] - 1, expected_action[2]
        )
        if transition_action.shape != expected_action:
            raise ValueError(
                "transition_action must be [B,S-1,3], got "
                f"{tuple(transition_action.shape)} expected {expected_action}"
            )
        if transition_valid.shape != expected_action[:2]:
            raise ValueError("transition_valid must be [B,S-1]")
        if transition_valid.any():
            predicted_world = F.normalize(
                outputs["future_world_hat"][transition_valid], dim=-1, eps=1.0e-6
            )
            future_world = F.normalize(
                outputs["w_t"][:, 1:].detach()[transition_valid],
                dim=-1,
                eps=1.0e-6,
            )
            losses["future_latent"] = F.mse_loss(predicted_world, future_world)

        target_xy_valid = (
            batch["future_target_xy_valid"].bool() & transition_valid
        )
        if target_xy_valid.any():
            losses["future_target_xy"] = F.smooth_l1_loss(
                outputs["future_target_xy_hat"][target_xy_valid],
                batch["future_target_xy"][target_xy_valid],
            )
        target_visibility_valid = (
            batch["future_target_visibility_valid"].bool() & transition_valid
        )
        if target_visibility_valid.any():
            losses["future_visibility"] = F.binary_cross_entropy_with_logits(
                outputs["future_target_visibility_logit"][target_visibility_valid],
                batch["future_target_visible"][target_visibility_valid],
            )
        losses["world_action"] = losses["future_latent"]
        if world_action_variant == "b":
            losses["world_action"] = (
                losses["world_action"]
                + losses["future_target_xy"]
                + losses["future_visibility"]
            )
        if transition_valid.any():
            target_xi = torch.cat(
                (
                    transition_action[..., :2],
                    torch.sin(transition_action[..., 2:3]),
                    torch.cos(transition_action[..., 2:3]),
                ),
                dim=-1,
            )[transition_valid]
            predicted_xi = outputs["inverse_motion_hat"][transition_valid]
            translation = F.smooth_l1_loss(
                predicted_xi[..., :2], target_xi[..., :2]
            )
            predicted_angle = F.normalize(
                predicted_xi[..., 2:4], dim=-1, eps=1.0e-6
            )
            target_angle = F.normalize(target_xi[..., 2:4], dim=-1, eps=1.0e-6)
            angular = (
                1.0 - (predicted_angle * target_angle).sum(dim=-1)
            ).mean()
            losses["inverse"] = translation + angular
    losses["total"] = sum(
        float(weights.get(name, 0.0)) * value
        for name, value in losses.items()
        if name != "total"
    )
    return losses["total"], losses


def _norm(parameters: Sequence[nn.Parameter]) -> float:
    squares = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt().cpu())


def waypoint_gradient_report(model: EndToEndFollowPolicy) -> dict[str, float]:
    named = list(model.named_parameters())
    adapter_parameters = [
        parameter for name, parameter in named if ".adapter." in name
    ]
    return {
        "fusion": _norm(list(model.fusion.parameters())),
        "gru": _norm(list(model.gru.parameters())),
        "da3_adapter": _norm(adapter_parameters),
        "l11_projector": _norm(list(model.l11_projector.parameters())),
        "target_attention": _norm(list(model.target_attention.parameters())),
        "scene_attention": _norm(list(model.scene_attention.parameters())),
        "motion_pair_head": _norm(list(model.motion_pair_head.parameters())),
        "waypoint_head": _norm(list(model.waypoint_head.parameters())),
    }


def parameter_inventory(model: EndToEndFollowPolicy) -> dict[str, Any]:
    categories: dict[str, list[tuple[str, nn.Parameter]]] = {
        "frozen_da3": [],
        "da3_adapter": [],
        "new_policy": [],
    }
    for name, parameter in model.named_parameters():
        if ".adapter." in name:
            category = "da3_adapter"
        elif name.startswith("da3."):
            category = "frozen_da3"
        else:
            category = "new_policy"
        categories[category].append((name, parameter))
    report: dict[str, Any] = {}
    for category, values in categories.items():
        report[category] = {
            "parameter_tensors": len(values),
            "parameters": sum(parameter.numel() for _, parameter in values),
            "trainable_parameters": sum(
                parameter.numel() for _, parameter in values if parameter.requires_grad
            ),
        }
    incorrectly_trainable = [
        name
        for name, parameter in categories["frozen_da3"]
        if parameter.requires_grad
    ]
    if incorrectly_trainable:
        raise RuntimeError(
            f"non-adapter DA3 parameters are trainable: {incorrectly_trainable[:5]}"
        )
    return report


def load_official_da3_small_l11(
    model_path: str | Path,
    config: ArchitectureV1Config | None = None,
    ablation: ArchitectureV1Ablation | None = None,
    dinov2_model_path: str | Path | None = None,
) -> tuple[DA3SmallL11Backbone, dict[str, Any]]:
    """Load the pinned local DA3-SMALL checkpoint and report tensor coverage."""

    config = config or ArchitectureV1Config()
    config.validate()
    ablation = ablation or ArchitectureV1Ablation()
    ablation.validate()
    from depth_anything_3.api import DepthAnything3
    from safetensors import safe_open

    model_path = Path(model_path).expanduser().resolve(strict=True)
    weight_path = model_path / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(f"missing DA3 safetensors checkpoint: {weight_path}")
    api_model = DepthAnything3.from_pretrained(str(model_path))
    backbone = api_model.model.backbone
    if ablation.backbone_weights == "da3":
        loaded_state = api_model.state_dict()
        checkpoint_shapes: dict[str, tuple[int, ...]] = {}
        with safe_open(weight_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                checkpoint_shapes[key] = tuple(handle.get_slice(key).get_shape())
        matched = {
            key: shape
            for key, shape in checkpoint_shapes.items()
            if key in loaded_state and tuple(loaded_state[key].shape) == shape
        }
        checkpoint_numel = sum(math.prod(shape) for shape in checkpoint_shapes.values())
        matched_numel = sum(math.prod(shape) for shape in matched.values())
        report = {
            "checkpoint": str(weight_path),
            "checkpoint_parameter_tensors": len(checkpoint_shapes),
            "matched_parameter_tensors": len(matched),
            "tensor_coverage": len(matched) / max(1, len(checkpoint_shapes)),
            "checkpoint_parameters": checkpoint_numel,
            "matched_parameters": matched_numel,
            "parameter_coverage": matched_numel / max(1, checkpoint_numel),
            "missing_or_mismatched": sorted(set(checkpoint_shapes) - set(matched)),
            "newly_initialized_target_tensors": [],
        }
    else:
        if dinov2_model_path is None:
            raise FileNotFoundError(
                "ABL-V1-01 DINOv2 initialization requires dinov2_model in the config"
            )
        dinov2_path = Path(dinov2_model_path).expanduser().resolve(strict=True)
        source = torch.load(dinov2_path, map_location="cpu", weights_only=True)
        if not isinstance(source, Mapping) or not all(
            isinstance(key, str) and torch.is_tensor(value)
            for key, value in source.items()
        ):
            raise TypeError("DINOv2 checkpoint must be a tensor state dictionary")
        target_module = backbone.pretrained
        target = target_module.state_dict()
        compatible = {
            key: value
            for key, value in source.items()
            if key in target and tuple(value.shape) == tuple(target[key].shape)
        }
        missing_target = sorted(set(target) - set(compatible))
        unexpected_source = sorted(set(source) - set(compatible))
        target_module.load_state_dict(compatible, strict=False)
        named_tensors = dict(target_module.named_parameters())
        named_tensors.update(dict(target_module.named_buffers()))
        with torch.no_grad():
            for name in missing_target:
                value = named_tensors[name]
                if name == "camera_token":
                    nn.init.normal_(value, std=0.02)
                elif name.endswith("_norm.weight"):
                    value.fill_(1.0)
                elif name.endswith("_norm.bias"):
                    value.zero_()
                else:
                    raise RuntimeError(
                        f"no fail-closed DINOv2 initialization rule for {name}"
                    )
        source_numel = sum(value.numel() for value in source.values())
        matched_numel = sum(value.numel() for value in compatible.values())
        report = {
            "checkpoint": str(dinov2_path),
            "da3_structure_template": str(weight_path),
            "checkpoint_parameter_tensors": len(source),
            "matched_parameter_tensors": len(compatible),
            "tensor_coverage": len(compatible) / max(1, len(source)),
            "checkpoint_parameters": source_numel,
            "matched_parameters": matched_numel,
            "parameter_coverage": matched_numel / max(1, source_numel),
            "missing_or_mismatched": unexpected_source,
            "target_parameter_tensors": len(target),
            "target_parameters": sum(value.numel() for value in target.values()),
            "newly_initialized_target_tensors": missing_target,
        }
    report["backbone_weights"] = ablation.backbone_weights
    report["backbone_tuning"] = ablation.backbone_tuning
    report["feature_fusion"] = ablation.feature_fusion
    return DA3SmallL11Backbone(backbone, config, ablation), report
