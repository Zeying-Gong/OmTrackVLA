"""Architecture-v2 token-preserving target-following policy.

The deployment path consumes eight RGB frames sampled at 10 Hz, preserves
per-frame DA3 L11 evidence until Transformer trajectory decoding, and keeps
the geometric and continuous UWB paths separate.  Optional SE(2) yaw and
target-diagnostic heads are deployment-visible rather than training-only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn
from torchvision.ops import roi_align

from omtrackvla.models.end_to_end import ArchitectureV1Config, GeometricUWBProjector


@dataclass(frozen=True)
class ArchitectureV2DecoderConfig:
    policy_dim: int = 256
    attention_heads: int = 8
    grid_height: int = 20
    grid_width: int = 36
    history_size: int = 8
    history_stride_raw: int = 3
    scene_latents: int = 16
    fusion_layers: int = 2
    trajectory_layers: int = 2
    feedforward_ratio: int = 4
    horizon: int = 8
    max_segment_m: float = 0.30
    max_forward_m_s: float = 3.75
    max_lateral_m_s: float = 2.50
    max_yaw_rad_s: float = math.pi / 2.0
    max_yaw_segment_rad: float = math.pi / 6.0
    predict_se2_yaw: bool = False
    predict_target_diagnostics: bool = False
    predict_target_polar: bool = False

    @property
    def patch_count(self) -> int:
        return self.grid_height * self.grid_width

    def validate(self) -> None:
        if self.policy_dim <= 0 or self.policy_dim % self.attention_heads:
            raise ValueError("policy_dim must be positive and divisible by heads")
        if self.grid_height <= 0 or self.grid_width <= 0:
            raise ValueError("patch grid must be positive")
        if self.history_size < 2 or self.history_stride_raw < 1:
            raise ValueError("history must contain at least two positively spaced frames")
        if self.scene_latents <= 0 or self.fusion_layers <= 0:
            raise ValueError("scene/fusion layer counts must be positive")
        if self.trajectory_layers <= 0 or self.feedforward_ratio < 2:
            raise ValueError("trajectory layers/MLP ratio are invalid")
        if self.horizon < 2 or self.max_segment_m <= 0.0:
            raise ValueError("trajectory horizon/segment bound are invalid")
        if min(
            self.max_forward_m_s,
            self.max_lateral_m_s,
            self.max_yaw_rad_s,
            self.max_yaw_segment_rad,
        ) <= 0.0:
            raise ValueError("action and SE(2) yaw limits must be positive")


class ArchitectureV2TrajectoryDecoder(nn.Module):
    """Preserve spatial target/scene evidence until horizon-query decoding.

    Expected mask order is ``visual_initialization, uwb, rgb, binding``.  UWB
    remains dual-path: its explicit patch bias pools a geometry-conditioned
    visual token, while the independent continuous ``z_uwb`` is a separate
    context token.
    """

    SPECIAL_TOKENS = 6

    def __init__(self, config: ArchitectureV2DecoderConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        dim = config.policy_dim
        heads = config.attention_heads
        hidden = dim * config.feedforward_ratio

        self.patch_position = nn.Parameter(
            torch.empty(config.patch_count, dim).normal_(std=0.02)
        )
        self.time_position = nn.Parameter(
            torch.empty(config.history_size, dim).normal_(std=0.02)
        )
        self.scene_queries = nn.Parameter(
            torch.empty(config.scene_latents, dim).normal_(std=0.02)
        )
        self.scene_query_condition = nn.Linear(dim * 2, dim)
        self.scene_resampler = nn.MultiheadAttention(
            dim, heads, dropout=0.0, batch_first=True
        )
        self.target_query_norm = nn.LayerNorm(dim)
        self.patch_key_norm = nn.LayerNorm(dim)
        self.unknown_visual_target = nn.Parameter(torch.zeros(dim))
        self.unknown_geometry_target = nn.Parameter(torch.zeros(dim))
        self.mask_embedding = nn.Sequential(
            nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.special_type = nn.Parameter(
            torch.empty(self.SPECIAL_TOKENS, dim).normal_(std=0.02)
        )
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, config.grid_height),
            torch.linspace(-1.0, 1.0, config.grid_width),
            indexing="ij",
        )
        self.register_buffer(
            "patch_coordinates",
            torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1),
            persistent=False,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=hidden,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.fusion_layers, norm=nn.LayerNorm(dim)
        )
        self.stop_query = nn.Parameter(torch.empty(1, dim).normal_(std=0.02))
        self.action_query = nn.Parameter(torch.empty(1, dim).normal_(std=0.02))
        self.trajectory_queries = nn.Parameter(
            torch.empty(config.horizon - 1, dim).normal_(std=0.02)
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=hidden,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.trajectory_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.trajectory_layers,
            norm=nn.LayerNorm(dim),
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 2)
        )
        if config.predict_se2_yaw:
            self.yaw_delta_head = nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1)
            )
        self.stop_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.direct_action_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 3)
        )
        if config.predict_target_diagnostics:
            diagnostic_dim = dim * 2 + 10
            self.bbox_head = nn.Sequential(
                nn.LayerNorm(diagnostic_dim),
                nn.Linear(diagnostic_dim, dim),
                nn.GELU(),
                nn.Linear(dim, 4),
            )
            self.visibility_head = nn.Sequential(
                nn.LayerNorm(diagnostic_dim),
                nn.Linear(diagnostic_dim, dim),
                nn.GELU(),
                nn.Linear(dim, 1),
            )
            if config.predict_target_polar:
                self.target_polar_head = nn.Sequential(
                    nn.LayerNorm(diagnostic_dim),
                    nn.Linear(diagnostic_dim, dim),
                    nn.GELU(),
                    nn.Linear(dim, 3),
                )
        elif config.predict_target_polar:
            raise ValueError(
                "target polar prediction requires target diagnostics"
            )
        self.register_buffer(
            "direct_action_scale",
            torch.tensor(
                [
                    config.max_forward_m_s,
                    config.max_lateral_m_s,
                    config.max_yaw_rad_s,
                ],
                dtype=torch.float32,
            ),
            persistent=True,
        )

    @staticmethod
    def _check_vector(name: str, value: torch.Tensor, batch: int, dim: int) -> None:
        if value.shape != (batch, dim):
            raise ValueError(f"{name} must have shape [B,C]")

    def forward(
        self,
        *,
        history_tokens: torch.Tensor,
        target_memory: torch.Tensor,
        uwb_patch_bias: torch.Tensor,
        z_uwb: torch.Tensor,
        z_ego: torch.Tensor,
        masks: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if history_tokens.ndim != 4:
            raise ValueError("history_tokens must have shape [B,T,N,C]")
        batch, steps, patches, dim = history_tokens.shape
        if steps != self.config.history_size:
            raise ValueError("history_tokens do not match the frozen history size")
        if patches != self.config.patch_count or dim != self.config.policy_dim:
            raise ValueError("history_tokens do not match the frozen patch grid/dim")
        self._check_vector("target_memory", target_memory, batch, dim)
        self._check_vector("z_uwb", z_uwb, batch, dim)
        self._check_vector("z_ego", z_ego, batch, dim)
        if uwb_patch_bias.shape != (batch, patches):
            raise ValueError("uwb_patch_bias must have shape [B,N]")
        if masks.shape != (batch, 4):
            raise ValueError("masks must be [B,4]")

        rgb_valid = masks[:, 2].clamp(0.0, 1.0)
        visual_target_valid = (
            torch.maximum(masks[:, 0], masks[:, 3]).clamp(0.0, 1.0)
            * rgb_valid
        )
        uwb_valid = masks[:, 1].clamp(0.0, 1.0)
        positioned_history = (
            history_tokens
            + self.patch_position[None, None].to(history_tokens)
            + self.time_position[None, :, None].to(history_tokens)
        ) * rgb_valid[:, None, None, None]
        positioned_scene = positioned_history[:, -1]

        query = self.target_query_norm(target_memory)
        keys = self.patch_key_norm(positioned_scene)
        appearance_logits = torch.einsum("bc,bnc->bn", query, keys) / math.sqrt(dim)
        appearance_attention = torch.softmax(appearance_logits, dim=1)
        visual_target = torch.einsum(
            "bn,bnc->bc", appearance_attention, positioned_scene
        )
        visual_target = torch.where(
            visual_target_valid[:, None].bool(),
            visual_target,
            self.unknown_visual_target[None].to(visual_target),
        )

        geometry_attention = torch.softmax(uwb_patch_bias.float(), dim=1).to(
            positioned_scene
        )
        geometry_target = torch.einsum(
            "bn,bnc->bc", geometry_attention, positioned_scene
        )
        geometry_target = torch.where(
            (uwb_valid * rgb_valid)[:, None].bool(),
            geometry_target,
            self.unknown_geometry_target[None].to(geometry_target),
        )
        continuous_uwb = z_uwb * uwb_valid[:, None]

        conditioned_queries = self.scene_queries[None, None].expand(
            batch, steps, -1, -1
        )
        query_condition = self.scene_query_condition(
            torch.cat((visual_target, geometry_target), dim=1)
        )
        conditioned_queries = (
            conditioned_queries
            + query_condition[:, None, None]
            + self.time_position[None, :, None].to(conditioned_queries)
        )
        flat_queries = conditioned_queries.reshape(
            batch * steps, self.config.scene_latents, dim
        )
        flat_history = positioned_history.reshape(batch * steps, patches, dim)
        scene_latents, scene_attention = self.scene_resampler(
            flat_queries,
            flat_history,
            flat_history,
            need_weights=True,
            average_attn_weights=False,
        )
        scene_latents = scene_latents.reshape(
            batch, steps * self.config.scene_latents, dim
        )
        scene_attention = scene_attention.reshape(
            batch,
            steps,
            self.config.attention_heads,
            self.config.scene_latents,
            patches,
        )

        reference = torch.where(
            visual_target_valid[:, None].bool(),
            target_memory,
            self.unknown_visual_target[None].to(target_memory),
        )
        special = torch.stack(
            (
                reference,
                visual_target,
                geometry_target,
                continuous_uwb,
                z_ego * rgb_valid[:, None],
                self.mask_embedding(masks.to(history_tokens.dtype)),
            ),
            dim=1,
        )
        special = special + self.special_type[None].to(special)
        context = self.context_encoder(torch.cat((special, scene_latents), dim=1))

        # Query order is part of the checkpoint/deployment contract:
        # stop, direct EVT action, then the seven non-origin trajectory points.
        output_queries = torch.cat(
            (self.stop_query, self.action_query, self.trajectory_queries), dim=0
        )
        decoded = self.trajectory_decoder(
            output_queries[None].expand(batch, -1, -1), context
        )
        direct_action = torch.tanh(self.direct_action_head(decoded[:, 1])) * (
            self.direct_action_scale.to(decoded)
        )
        deltas = torch.tanh(self.delta_head(decoded[:, 2:])) * float(
            self.config.max_segment_m
        )
        future = torch.cumsum(deltas, dim=1)
        waypoints = torch.cat((future.new_zeros(batch, 1, 2), future), dim=1)
        output = {
            "waypoints": waypoints,
            "waypoint_deltas": deltas,
            "direct_action_m_s_rad_s": direct_action,
            "stop_logit": self.stop_head(decoded[:, :1]).squeeze(1),
            "context_tokens": context,
            "visual_target": visual_target,
            "geometry_target": geometry_target,
            "appearance_attention": appearance_attention,
            "uwb_geometry_attention": geometry_attention,
            "scene_resampler_attention": scene_attention,
        }
        if self.config.predict_se2_yaw:
            yaw_deltas = torch.tanh(self.yaw_delta_head(decoded[:, 2:]).squeeze(-1)) * (
                float(self.config.max_yaw_segment_rad)
            )
            future_yaw = torch.cumsum(yaw_deltas, dim=1)
            future_yaw = torch.atan2(torch.sin(future_yaw), torch.cos(future_yaw))
            waypoint_yaws = torch.cat(
                (future_yaw.new_zeros(batch, 1), future_yaw), dim=1
            )
            output.update(
                {
                    "waypoint_yaws": waypoint_yaws,
                    "waypoint_yaw_deltas": yaw_deltas,
                    "se2_waypoints": torch.cat(
                        (waypoints, waypoint_yaws[..., None]), dim=-1
                    ),
                }
            )
        if self.config.predict_target_diagnostics:
            attention_float = appearance_attention.float()
            coordinates = self.patch_coordinates.to(attention_float)
            center = torch.einsum("bn,nc->bc", attention_float, coordinates)
            centered = coordinates[None] - center[:, None]
            variance = torch.einsum(
                "bn,bnc->bc", attention_float, centered.square()
            ).clamp_min(0.0)
            covariance = torch.einsum(
                "bn,bn->b",
                attention_float,
                centered[..., 0] * centered[..., 1],
            )[:, None]
            top2 = attention_float.topk(k=2, dim=1).values
            normalized_entropy = -(
                attention_float.clamp_min(1.0e-8).log() * attention_float
            ).sum(dim=1, keepdim=True) / math.log(float(patches))
            target_similarity = torch.nn.functional.cosine_similarity(
                query.float(), visual_target.float(), dim=1
            )[:, None]
            attention_statistics = torch.cat(
                (
                    center,
                    variance.sqrt(),
                    covariance,
                    top2[:, :1],
                    normalized_entropy,
                    top2[:, :1] - top2[:, 1:2],
                    torch.tanh(appearance_logits.float().max(dim=1).values[:, None] / 10.0),
                    target_similarity,
                ),
                dim=1,
            ).to(visual_target)
            diagnostic = torch.cat(
                (visual_target, positioned_scene.mean(dim=1), attention_statistics),
                dim=1,
            )
            box_parameters = torch.sigmoid(self.bbox_head(diagnostic))
            box_center = box_parameters[:, :2]
            box_size = 0.02 + 0.96 * box_parameters[:, 2:]
            output.update(
                {
                    "bbox_pred": torch.cat(
                        (box_center - box_size / 2.0, box_center + box_size / 2.0),
                        dim=1,
                    ).clamp(0.0, 1.0),
                    "visibility_logit": self.visibility_head(diagnostic),
                }
            )
            if self.config.predict_target_polar:
                raw_polar = self.target_polar_head(diagnostic)
                angle_sincos = torch.nn.functional.normalize(
                    raw_polar[:, :2].float(), dim=1, eps=1.0e-6
                ).to(raw_polar)
                output.update(
                    {
                        # Canonical base frame: angle is atan2(left, forward).
                        "target_angle_sincos": angle_sincos,
                        "target_distance_m": torch.nn.functional.softplus(
                            raw_polar[:, 2:3]
                        ),
                        # Visibility is the deployment-visible valid/invalid gate.
                        "target_visual_valid_logit": output["visibility_logit"],
                    }
                )
        return output


class ArchitectureV2FollowPolicy(nn.Module):
    """End-to-end Architecture-v2 deployment graph from RGB to trajectory.

    ``initial_rgb`` and ``initial_bbox`` create an immutable target reference.
    ``ego_rgb`` is the frozen 8-frame/10-Hz history.  Only the current frame is
    used for target appearance and explicit UWB geometry pooling; all eight
    frames contribute scene tokens to the trajectory decoder.
    """

    def __init__(
        self,
        da3: nn.Module,
        sensor_config: ArchitectureV1Config | None = None,
        decoder_config: ArchitectureV2DecoderConfig | None = None,
    ) -> None:
        super().__init__()
        self.sensor_config = sensor_config or ArchitectureV1Config(history_size=8)
        self.sensor_config.validate()
        self.decoder_config = decoder_config or ArchitectureV2DecoderConfig(
            policy_dim=self.sensor_config.policy_dim,
            attention_heads=self.sensor_config.attention_heads,
            grid_height=self.sensor_config.grid_height,
            grid_width=self.sensor_config.grid_width,
            history_size=self.sensor_config.history_size,
            horizon=self.sensor_config.horizon,
        )
        self.decoder_config.validate()
        expected = (
            self.sensor_config.policy_dim,
            self.sensor_config.attention_heads,
            self.sensor_config.grid_height,
            self.sensor_config.grid_width,
            self.sensor_config.history_size,
            self.sensor_config.horizon,
        )
        received = (
            self.decoder_config.policy_dim,
            self.decoder_config.attention_heads,
            self.decoder_config.grid_height,
            self.decoder_config.grid_width,
            self.decoder_config.history_size,
            self.decoder_config.horizon,
        )
        if received != expected:
            raise ValueError(
                "sensor/decoder policy, grid, history, and horizon contracts differ: "
                f"expected={expected}, received={received}"
            )

        self.da3 = da3
        dim = self.sensor_config.policy_dim
        self.l11_projector = nn.Linear(self.sensor_config.backbone_dim, dim)
        self.unknown_person = nn.Parameter(torch.empty(dim).normal_(std=0.02))
        self.motion_pair_norm = nn.LayerNorm(self.sensor_config.backbone_dim * 2)
        self.motion_pair_head = nn.Sequential(
            nn.Linear(self.sensor_config.backbone_dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 4),
        )
        self.motion_embedding = nn.Sequential(
            nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.uwb_projector = GeometricUWBProjector(self.sensor_config)
        self.uwb_encoder = nn.Sequential(
            nn.Linear(8, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.trajectory = ArchitectureV2TrajectoryDecoder(self.decoder_config)

    @staticmethod
    def _scalar(value: torch.Tensor, batch: int, name: str) -> torch.Tensor:
        if value.numel() != batch:
            raise ValueError(f"{name} must contain one scalar per batch item")
        return value.reshape(batch)

    def _target_memory(
        self,
        initial_tokens: torch.Tensor,
        initial_bbox: torch.Tensor,
        visual_initialization_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch = initial_tokens.shape[0]
        dim = self.sensor_config.policy_dim
        feature_map = initial_tokens.reshape(
            batch,
            self.sensor_config.grid_height,
            self.sensor_config.grid_width,
            dim,
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
                safe_bbox[:, 0] * self.sensor_config.grid_width,
                safe_bbox[:, 1] * self.sensor_config.grid_height,
                safe_bbox[:, 2] * self.sensor_config.grid_width,
                safe_bbox[:, 3] * self.sensor_config.grid_height,
            ),
            dim=1,
        )
        pooled = roi_align(
            feature_map,
            boxes,
            output_size=(self.sensor_config.roi_size, self.sensor_config.roi_size),
            spatial_scale=1.0,
            aligned=True,
        ).mean(dim=(2, 3))
        return torch.where(
            valid[:, None], pooled, self.unknown_person[None].expand(batch, -1)
        )

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
    ) -> dict[str, torch.Tensor]:
        batch = initial_rgb.shape[0]
        expected_initial = (
            batch,
            3,
            self.sensor_config.image_height,
            self.sensor_config.image_width,
        )
        expected_history = (
            batch,
            self.sensor_config.history_size,
            3,
            self.sensor_config.image_height,
            self.sensor_config.image_width,
        )
        if initial_rgb.shape != expected_initial:
            raise ValueError("initial_rgb shape mismatch")
        if ego_rgb.shape != expected_history:
            raise ValueError("ego_rgb shape mismatch")
        if initial_bbox.shape != (batch, 4):
            raise ValueError("initial_bbox must be normalized xyxy with shape [B,4]")

        history_spatial, history_camera = self.da3(ego_rgb)
        history_tokens = self.l11_projector(history_spatial)
        initial_spatial, _ = self.da3(initial_rgb[:, None])
        initial_tokens = self.l11_projector(initial_spatial[:, 0])
        target_memory = self._target_memory(
            initial_tokens, initial_bbox, visual_initialization_valid
        )

        camera_pair = torch.cat(
            (history_camera[:, -2], history_camera[:, -1]), dim=1
        )
        xi_hat = self.motion_pair_head(self.motion_pair_norm(camera_pair))
        z_ego = self.motion_embedding(xi_hat)

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
        with torch.autocast(device_type=uwb_xy.device.type, enabled=False):
            uwb_patch_bias, uwb_diagnostics = self.uwb_projector(
                uwb_xy.float(),
                uwb_covariance_xy.float(),
                uwb_quality.float(),
                uwb_age_s.float(),
                uwb_valid,
                camera_intrinsics.float(),
                camera_from_base.float(),
            )
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
        output = self.trajectory(
            history_tokens=history_tokens,
            target_memory=target_memory,
            uwb_patch_bias=uwb_patch_bias.to(history_tokens),
            z_uwb=z_uwb,
            z_ego=z_ego,
            masks=masks,
        )
        output.update(
            {
                "target_memory": target_memory,
                "z_target": output["visual_target"],
                "z_ego": z_ego,
                "z_uwb": z_uwb,
                "xi_hat": xi_hat,
                "uwb_patch_bias": uwb_patch_bias.to(history_tokens),
            }
        )
        output.update(uwb_diagnostics)
        return output


def _gradient_norm(parameters: Sequence[nn.Parameter]) -> float:
    squares = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt().cpu())


def waypoint_gradient_report_v2(
    model: ArchitectureV2FollowPolicy,
) -> dict[str, float]:
    named = list(model.named_parameters())
    adapter_parameters = [
        parameter for name, parameter in named if ".adapter." in name
    ]
    return {
        "da3_adapter": _gradient_norm(adapter_parameters),
        "l11_projector": _gradient_norm(list(model.l11_projector.parameters())),
        "scene_resampler": _gradient_norm(
            list(model.trajectory.scene_resampler.parameters())
        ),
        "context_encoder": _gradient_norm(
            list(model.trajectory.context_encoder.parameters())
        ),
        "trajectory_decoder": _gradient_norm(
            list(model.trajectory.trajectory_decoder.parameters())
        ),
        "delta_head": _gradient_norm(list(model.trajectory.delta_head.parameters())),
        "uwb_encoder": _gradient_norm(list(model.uwb_encoder.parameters())),
        "motion_pair_head": _gradient_norm(list(model.motion_pair_head.parameters())),
    }


def parameter_inventory_v2(model: ArchitectureV2FollowPolicy) -> dict[str, Any]:
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
