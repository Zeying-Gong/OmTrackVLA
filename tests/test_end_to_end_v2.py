import math
import unittest

import torch

from omtrackvla.models.end_to_end import ArchitectureV1Config, StubDA3SmallL11Backbone
from omtrackvla.models.end_to_end_v2 import (
    ArchitectureV2DecoderConfig,
    ArchitectureV2FollowPolicy,
    ArchitectureV2TrajectoryDecoder,
    parameter_inventory_v2,
    waypoint_gradient_report_v2,
)


class ArchitectureV2TrajectoryDecoderTest(unittest.TestCase):
    def config(self) -> ArchitectureV2DecoderConfig:
        return ArchitectureV2DecoderConfig(
            policy_dim=32,
            attention_heads=4,
            grid_height=4,
            grid_width=6,
            history_size=3,
            scene_latents=4,
            fusion_layers=1,
            trajectory_layers=1,
            feedforward_ratio=2,
            horizon=8,
            max_segment_m=0.2,
            predict_se2_yaw=True,
            predict_target_diagnostics=True,
            predict_target_polar=True,
        )

    def inputs(self) -> dict[str, torch.Tensor]:
        return {
            "history_tokens": torch.randn(2, 3, 24, 32, requires_grad=True),
            "target_memory": torch.randn(2, 32, requires_grad=True),
            "uwb_patch_bias": torch.randn(2, 24),
            "z_uwb": torch.randn(2, 32, requires_grad=True),
            "z_ego": torch.randn(2, 32, requires_grad=True),
            "masks": torch.ones(2, 4),
        }

    def test_preserves_token_context_and_decodes_bounded_se2_path(self):
        model = ArchitectureV2TrajectoryDecoder(self.config())
        output = model(**self.inputs())

        self.assertEqual(tuple(output["waypoints"].shape), (2, 8, 2))
        self.assertEqual(tuple(output["waypoint_deltas"].shape), (2, 7, 2))
        self.assertEqual(tuple(output["waypoint_yaws"].shape), (2, 8))
        self.assertEqual(tuple(output["waypoint_yaw_deltas"].shape), (2, 7))
        self.assertEqual(tuple(output["se2_waypoints"].shape), (2, 8, 3))
        self.assertEqual(tuple(output["bbox_pred"].shape), (2, 4))
        self.assertEqual(tuple(output["visibility_logit"].shape), (2, 1))
        self.assertEqual(tuple(output["target_angle_sincos"].shape), (2, 2))
        self.assertEqual(tuple(output["target_distance_m"].shape), (2, 1))
        torch.testing.assert_close(
            output["target_angle_sincos"].float().norm(dim=1), torch.ones(2)
        )
        self.assertTrue(bool((output["target_distance_m"] >= 0.0).all()))
        self.assertTrue(
            torch.equal(
                output["target_visual_valid_logit"], output["visibility_logit"]
            )
        )
        self.assertEqual(tuple(output["direct_action_m_s_rad_s"].shape), (2, 3))
        self.assertLessEqual(
            float(output["direct_action_m_s_rad_s"][:, 0].abs().max()), 3.75
        )
        self.assertLessEqual(
            float(output["direct_action_m_s_rad_s"][:, 1].abs().max()), 2.50
        )
        self.assertLessEqual(
            float(output["direct_action_m_s_rad_s"][:, 2].abs().max()), math.pi / 2
        )
        self.assertEqual(tuple(output["stop_logit"].shape), (2, 1))
        self.assertEqual(tuple(output["context_tokens"].shape), (2, 18, 32))
        self.assertTrue(torch.equal(output["waypoints"][:, 0], torch.zeros(2, 2)))
        self.assertLessEqual(float(output["waypoint_deltas"].abs().max()), 0.2)
        self.assertLessEqual(
            float(output["waypoint_yaw_deltas"].abs().max()), math.pi / 6
        )
        self.assertFalse(any("gru" in name.lower() for name, _ in model.named_parameters()))

    def test_polar_head_requires_target_diagnostics(self):
        config = self.config()
        config = ArchitectureV2DecoderConfig(
            **{
                **config.__dict__,
                "predict_target_diagnostics": False,
                "predict_target_polar": True,
            }
        )
        with self.assertRaisesRegex(ValueError, "requires target diagnostics"):
            ArchitectureV2TrajectoryDecoder(config)

    def test_waypoint_loss_reaches_scene_target_uwb_and_ego_inputs(self):
        model = ArchitectureV2TrajectoryDecoder(self.config())
        inputs = self.inputs()
        output = model(**inputs)
        loss = output["waypoints"][:, 1:].square().mean()
        loss.backward()

        for name in ("history_tokens", "target_memory", "z_uwb", "z_ego"):
            gradient = inputs[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertGreater(float(gradient.norm()), 0.0, name)

    def test_uwb_geometry_bias_is_a_separate_spatial_path(self):
        model = ArchitectureV2TrajectoryDecoder(self.config())
        inputs = self.inputs()
        inputs["uwb_patch_bias"] = torch.full((2, 24), -20.0)
        inputs["uwb_patch_bias"][:, 7] = 20.0
        output = model(**inputs)

        self.assertTrue(
            torch.equal(output["uwb_geometry_attention"].argmax(dim=1), torch.full((2,), 7))
        )
        torch.testing.assert_close(
            output["geometry_target"],
            (
                inputs["history_tokens"][:, -1]
                + model.patch_position[None]
                + model.time_position[-1][None, None]
            )[:, 7],
            atol=1.0e-5,
            rtol=1.0e-5,
        )


class ArchitectureV2FollowPolicyTest(unittest.TestCase):
    def sensor_config(self) -> ArchitectureV1Config:
        return ArchitectureV1Config(
            image_height=28,
            image_width=42,
            history_size=3,
            patch_size=14,
            backbone_dim=48,
            policy_dim=32,
            attention_heads=4,
            adapter_layers=1,
            adapter_bottleneck=8,
        )

    def decoder_config(self) -> ArchitectureV2DecoderConfig:
        return ArchitectureV2DecoderConfig(
            policy_dim=32,
            attention_heads=4,
            grid_height=2,
            grid_width=3,
            history_size=3,
            scene_latents=2,
            fusion_layers=1,
            trajectory_layers=1,
            feedforward_ratio=2,
            horizon=8,
            max_segment_m=0.2,
        )

    def inputs(self, config: ArchitectureV1Config) -> dict[str, torch.Tensor]:
        batch = 2
        intrinsics = torch.tensor(
            [[20.0, 0.0, 21.0], [0.0, 20.0, 14.0], [0.0, 0.0, 1.0]]
        ).repeat(batch, 1, 1)
        camera_from_base = torch.eye(4).repeat(batch, 1, 1)
        camera_from_base[:, :3, :3] = torch.tensor(
            [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]
        )
        return {
            "initial_rgb": torch.rand(
                batch, 3, config.image_height, config.image_width
            ),
            "initial_bbox": torch.tensor([[0.2, 0.1, 0.7, 0.9]]).repeat(batch, 1),
            "ego_rgb": torch.rand(
                batch,
                config.history_size,
                3,
                config.image_height,
                config.image_width,
            ),
            "visual_initialization_valid": torch.ones(batch),
            "rgb_valid": torch.ones(batch),
            "binding_valid": torch.ones(batch),
            "uwb_xy": torch.tensor([[2.0, 0.0]]).repeat(batch, 1),
            "uwb_covariance_xy": torch.eye(2).mul(0.04).repeat(batch, 1, 1),
            "uwb_quality": torch.ones(batch),
            "uwb_age_s": torch.zeros(batch),
            "uwb_valid": torch.ones(batch),
            "camera_intrinsics": intrinsics,
            "camera_from_base": camera_from_base,
        }

    def test_waypoint_only_backward_reaches_every_v2_main_path(self):
        torch.manual_seed(17)
        sensor_config = self.sensor_config()
        model = ArchitectureV2FollowPolicy(
            StubDA3SmallL11Backbone(sensor_config),
            sensor_config,
            self.decoder_config(),
        )
        output = model(**self.inputs(sensor_config))
        loss = output["waypoints"][:, 1:].square().mean()
        loss.backward()

        gradients = waypoint_gradient_report_v2(model)
        for name, norm in gradients.items():
            self.assertGreater(norm, 0.0, name)
        self.assertEqual(tuple(output["waypoints"].shape), (2, 8, 2))
        self.assertEqual(tuple(output["uwb_patch_bias"].shape), (2, 6))
        self.assertFalse(
            any("gru" in name.lower() for name, _ in model.named_parameters())
        )

        inventory = parameter_inventory_v2(model)
        self.assertEqual(inventory["frozen_da3"]["trainable_parameters"], 0)
        self.assertGreater(inventory["da3_adapter"]["trainable_parameters"], 0)
        self.assertGreater(inventory["new_policy"]["trainable_parameters"], 0)


if __name__ == "__main__":
    unittest.main()
