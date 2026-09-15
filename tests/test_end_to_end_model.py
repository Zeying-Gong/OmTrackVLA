import unittest

import torch

from omtrackvla.models.end_to_end import (
    ArchitectureV1Ablation,
    ArchitectureV1Config,
    EndToEndFollowPolicy,
    GeometricUWBProjector,
    StubDA3SmallL11Backbone,
    compute_architecture_v1_loss,
    parameter_inventory,
    waypoint_gradient_report,
    waypoint_only_loss,
    waypoint_path_length_loss,
    waypoint_radial_progress_loss,
)
from omtrackvla.models.end_to_end_phase1 import ArchitectureV1Phase1Model


def tiny_config() -> ArchitectureV1Config:
    return ArchitectureV1Config(
        image_height=28,
        image_width=42,
        patch_size=14,
        backbone_dim=48,
        policy_dim=32,
        attention_heads=4,
        adapter_layers=1,
        adapter_bottleneck=8,
        tag_height_mean_m=1.0,
        tag_height_sigma_m=0.2,
        calibration_sigma_px=0.5,
    )


def camera_tensors(batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    intrinsics = torch.tensor(
        [[20.0, 0.0, 21.0], [0.0, 20.0, 14.0], [0.0, 0.0, 1.0]]
    ).repeat(batch, 1, 1)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]
    )
    transform = torch.eye(4).repeat(batch, 1, 1)
    transform[:, :3, :3] = rotation
    transform[:, :3, 3] = torch.tensor([0.0, 0.3, 0.0])
    return intrinsics, transform


def batch(config: ArchitectureV1Config, count: int = 2) -> dict[str, torch.Tensor]:
    intrinsics, transform = camera_tensors(count)
    target = torch.zeros(count, config.horizon, 2)
    target[:, 1:, 0] = torch.linspace(0.1, 0.7, config.horizon - 1)
    return {
        "initial_rgb": torch.rand(count, 3, config.image_height, config.image_width),
        "initial_bbox": torch.tensor([[0.2, 0.1, 0.7, 0.9]]).repeat(count, 1),
        "ego_rgb": torch.rand(
            count,
            config.history_size,
            3,
            config.image_height,
            config.image_width,
        ),
        "visual_initialization_valid": torch.ones(count),
        "rgb_valid": torch.ones(count),
        "binding_valid": torch.ones(count),
        "uwb_xy": torch.tensor([[2.0, 0.0]]).repeat(count, 1),
        "uwb_covariance_xy": torch.eye(2).mul(0.04).repeat(count, 1, 1),
        "uwb_quality": torch.ones(count),
        "uwb_age_s": torch.zeros(count),
        "uwb_valid": torch.ones(count),
        "camera_intrinsics": intrinsics,
        "camera_from_base": transform,
        "target_waypoints": target,
        "waypoint_mask": torch.ones(count, config.horizon, dtype=torch.bool),
    }


class EndToEndModelTest(unittest.TestCase):
    def test_frozen_architecture_defaults_are_20_by_36(self):
        config = ArchitectureV1Config()
        config.validate()
        self.assertEqual((config.grid_height, config.grid_width), (20, 36))
        self.assertEqual(config.patch_count, 720)
        self.assertEqual(config.history_size, 4)
        self.assertEqual(config.horizon, 8)

    def test_ablation_switches_are_explicit_and_shape_safe(self):
        torch.manual_seed(13)
        config = tiny_config()
        values = batch(config)
        model_inputs = {
            key: value
            for key, value in values.items()
            if key not in {"target_waypoints", "waypoint_mask"}
        }
        settings = (
            ArchitectureV1Ablation(backbone_tuning="frozen"),
            ArchitectureV1Ablation(feature_fusion="l5_l11"),
            ArchitectureV1Ablation(target_pooling="bbox_mean"),
            ArchitectureV1Ablation(temporal_fusion="single_step"),
            ArchitectureV1Ablation(ego_representation="raw_camera_difference"),
            ArchitectureV1Ablation(ego_representation="none"),
            ArchitectureV1Ablation(uwb_early_fusion="none"),
            ArchitectureV1Ablation(uwb_early_fusion="learned"),
        )
        for ablation in settings:
            with self.subTest(ablation=ablation.to_dict()):
                model = EndToEndFollowPolicy(
                    StubDA3SmallL11Backbone(config, ablation), config, ablation
                )
                outputs = model(**model_inputs)
                self.assertEqual(tuple(outputs["waypoints"].shape), (2, 8, 2))
                self.assertTrue(torch.isfinite(outputs["waypoints"]).all())
                self.assertEqual(
                    tuple(outputs["uwb_patch_bias"].shape),
                    (2, config.patch_count),
                )
                if ablation.backbone_tuning == "frozen":
                    self.assertEqual(
                        parameter_inventory(model)["da3_adapter"]["trainable_parameters"],
                        0,
                    )
                if ablation.ego_representation == "none":
                    self.assertTrue(torch.equal(outputs["z_ego"], torch.zeros_like(outputs["z_ego"])))
                if ablation.uwb_early_fusion == "none":
                    self.assertTrue(
                        torch.equal(
                            outputs["uwb_patch_bias"],
                            torch.zeros_like(outputs["uwb_patch_bias"]),
                        )
                    )

    def test_single_step_ablation_does_not_consume_gru_hidden_state(self):
        config = tiny_config()
        ablation = ArchitectureV1Ablation(temporal_fusion="single_step")
        model = EndToEndFollowPolicy(
            StubDA3SmallL11Backbone(config, ablation), config, ablation
        ).eval()
        values = batch(config)
        inputs = {
            key: value
            for key, value in values.items()
            if key not in {"target_waypoints", "waypoint_mask"}
        }
        with torch.no_grad():
            zero = model(**inputs, hidden_state=torch.zeros(2, config.policy_dim))
            random = model(**inputs, hidden_state=torch.randn(2, config.policy_dim))
        self.assertTrue(torch.equal(zero["waypoints"], random["waypoints"]))

    def test_gru_initialization_is_current_token_dominant_and_recurrent(self):
        config = tiny_config()
        model = EndToEndFollowPolicy(StubDA3SmallL11Backbone(config), config)
        model.initialize_gru_near_identity()
        gru = model.gru
        dim = gru.hidden_size
        identity = torch.eye(dim)
        self.assertTrue(
            torch.equal(gru.weight_ih[: 2 * dim], torch.zeros(2 * dim, dim))
        )
        self.assertTrue(
            torch.equal(gru.weight_hh[: 2 * dim], torch.zeros(2 * dim, dim))
        )
        self.assertTrue(torch.equal(gru.weight_ih[2 * dim :], identity))
        self.assertTrue(torch.equal(gru.weight_hh[2 * dim :], identity * 0.1))
        self.assertTrue(
            torch.equal(gru.bias_ih[dim : 2 * dim], torch.full((dim,), -5.0))
        )

        current = torch.full((1, dim), 0.05)
        previous = torch.full((1, dim), 0.02)
        state = gru(current, previous)
        state_without_history = gru(current, torch.zeros_like(previous))
        self.assertGreater(float((state - state_without_history).abs().sum()), 0.0)

    def test_waypoint_only_backward_reaches_fusion_gru_and_da3_adapter(self):
        torch.manual_seed(7)
        config = tiny_config()
        model = EndToEndFollowPolicy(StubDA3SmallL11Backbone(config), config)
        values = batch(config)
        outputs = model(
            **{
                key: value
                for key, value in values.items()
                if key not in {"target_waypoints", "waypoint_mask"}
            }
        )
        loss = waypoint_only_loss(
            outputs, values["target_waypoints"], values["waypoint_mask"]
        )
        loss.backward()
        gradients = waypoint_gradient_report(model)
        self.assertGreater(gradients["fusion"], 0.0)
        self.assertGreater(gradients["gru"], 0.0)
        self.assertGreater(gradients["da3_adapter"], 0.0)
        self.assertEqual(tuple(outputs["waypoints"].shape), (2, 8, 2))
        self.assertEqual(tuple(outputs["stop_logit"].shape), (2, 1))
        self.assertTrue(torch.equal(outputs["waypoints"][:, 0], torch.zeros(2, 2)))
        self.assertIsNone(model.stop_head.weight.grad)

    def test_radial_progress_detects_path_length_zigzag_loophole(self):
        target = torch.tensor([[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]])
        predicted = torch.tensor(
            [[[0.0, 0.0], [0.5, 0.0], [0.0, 0.0]]], requires_grad=True
        )
        mask = torch.ones(1, 3, dtype=torch.bool)
        outputs = {"waypoints": predicted}

        path_length = waypoint_path_length_loss(outputs, target, mask)
        radial_progress = waypoint_radial_progress_loss(outputs, target, mask)

        self.assertAlmostEqual(float(path_length), 0.0, places=7)
        self.assertGreater(float(radial_progress), 0.0)
        radial_progress.backward()
        self.assertGreater(float(predicted.grad.abs().sum()), 0.0)

    def test_parameter_inventory_separates_frozen_adapter_and_policy(self):
        config = tiny_config()
        model = EndToEndFollowPolicy(StubDA3SmallL11Backbone(config), config)
        report = parameter_inventory(model)
        self.assertEqual(report["frozen_da3"]["trainable_parameters"], 0)
        self.assertGreater(report["frozen_da3"]["parameters"], 0)
        self.assertGreater(report["da3_adapter"]["trainable_parameters"], 0)
        self.assertGreater(report["new_policy"]["trainable_parameters"], 0)

    def test_formal_sequence_unroll_and_connected_losses(self):
        torch.manual_seed(9)
        config = tiny_config()
        model = EndToEndFollowPolicy(StubDA3SmallL11Backbone(config), config)
        values = batch(config)
        steps = 2
        sequence_inputs = {
            "initial_rgb": values["initial_rgb"],
            "initial_bbox": values["initial_bbox"],
            "ego_rgb": values["ego_rgb"][:, None].repeat(1, steps, 1, 1, 1, 1),
            "visual_initialization_valid": values["visual_initialization_valid"],
            "rgb_valid": values["rgb_valid"][:, None].repeat(1, steps),
            "binding_valid": values["binding_valid"][:, None].repeat(1, steps),
            "uwb_xy": values["uwb_xy"][:, None].repeat(1, steps, 1),
            "uwb_covariance_xy": values["uwb_covariance_xy"][:, None].repeat(
                1, steps, 1, 1
            ),
            "uwb_quality": values["uwb_quality"][:, None].repeat(1, steps),
            "uwb_age_s": values["uwb_age_s"][:, None].repeat(1, steps),
            "uwb_valid": values["uwb_valid"][:, None].repeat(1, steps),
            "camera_intrinsics": values["camera_intrinsics"],
            "camera_from_base": values["camera_from_base"],
        }
        outputs = model.forward_sequence(**sequence_inputs)
        target_waypoints = values["target_waypoints"][:, None].repeat(
            1, steps, 1, 1
        )
        formal_batch = {
            "target_waypoints": target_waypoints,
            "waypoint_mask": torch.ones(
                values["initial_rgb"].shape[0], steps, config.horizon, dtype=torch.bool
            ),
            "stop_target": torch.zeros(values["initial_rgb"].shape[0], steps),
            "target_bbox": torch.tensor([0.2, 0.1, 0.7, 0.9]).repeat(
                values["initial_rgb"].shape[0], steps, 1
            ),
            "target_visible": torch.ones(values["initial_rgb"].shape[0], steps),
            "identity_label_valid": torch.ones(values["initial_rgb"].shape[0], steps),
            "binding_target": torch.ones(values["initial_rgb"].shape[0], steps),
            "ego_motion_target": torch.tensor([0.01, 0.0, 0.0, 1.0]).repeat(
                values["initial_rgb"].shape[0], steps, 1
            ),
        }
        weights = {
            "waypoint": 1.0,
            "waypoint_delta": 1.0,
            "waypoint_terminal": 1.0,
            "waypoint_radial_progress": 1.0,
            "waypoint_path_length": 1.0,
            "stop": 0.5,
            "bbox": 0.5,
            "visibility": 0.5,
            "identity_or_binding": 0.1,
            "ego": 0.1,
            "world_action": 0.0,
            "inverse": 0.0,
        }
        loss, losses = compute_architecture_v1_loss(outputs, formal_batch, weights)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(tuple(outputs["waypoints"].shape), (2, 2, 8, 2))
        self.assertEqual(tuple(outputs["bbox_pred"].shape), (2, 2, 4))
        self.assertGreater(float(losses["waypoint"]), 0.0)
        self.assertGreater(float(losses["waypoint_delta"]), 0.0)
        self.assertGreater(float(losses["waypoint_terminal"]), 0.0)
        self.assertGreater(float(losses["waypoint_radial_progress"]), 0.0)
        self.assertGreater(float(losses["waypoint_path_length"]), 0.0)
        self.assertIsNotNone(model.bbox_head[-1].weight.grad)
        self.assertIsNotNone(model.visibility_head[-1].weight.grad)
        self.assertIsNotNone(model.binding_head[-1].weight.grad)

    def test_phase2_curriculum_connects_world_action_and_inverse_heads(self):
        torch.manual_seed(10)
        config = tiny_config()
        policy = EndToEndFollowPolicy(StubDA3SmallL11Backbone(config), config)
        model = ArchitectureV1Phase1Model(policy)
        values = batch(config)
        steps = 2
        outputs = model(
            transition_action=torch.tensor(
                [[[0.1, 0.0, 0.02]], [[0.0, 0.1, -0.01]]]
            ),
            initial_rgb=values["initial_rgb"],
            initial_bbox=values["initial_bbox"],
            ego_rgb=values["ego_rgb"][:, None].repeat(1, steps, 1, 1, 1, 1),
            visual_initialization_valid=values["visual_initialization_valid"],
            rgb_valid=values["rgb_valid"][:, None].repeat(1, steps),
            binding_valid=values["binding_valid"][:, None].repeat(1, steps),
            uwb_xy=values["uwb_xy"][:, None].repeat(1, steps, 1),
            uwb_covariance_xy=values["uwb_covariance_xy"][:, None].repeat(
                1, steps, 1, 1
            ),
            uwb_quality=values["uwb_quality"][:, None].repeat(1, steps),
            uwb_age_s=values["uwb_age_s"][:, None].repeat(1, steps),
            uwb_valid=values["uwb_valid"][:, None].repeat(1, steps),
            camera_intrinsics=values["camera_intrinsics"],
            camera_from_base=values["camera_from_base"],
        )
        formal_batch = {
            "target_waypoints": values["target_waypoints"][:, None].repeat(
                1, steps, 1, 1
            ),
            "waypoint_mask": torch.ones(2, steps, config.horizon, dtype=torch.bool),
            "stop_target": torch.zeros(2, steps),
            "target_bbox": torch.tensor([0.2, 0.1, 0.7, 0.9]).repeat(2, steps, 1),
            "target_visible": torch.ones(2, steps),
            "identity_label_valid": torch.ones(2, steps),
            "binding_target": torch.ones(2, steps),
            "ego_motion_target": torch.tensor([0.01, 0.0, 0.0, 1.0]).repeat(
                2, steps, 1
            ),
            "transition_action": torch.tensor(
                [[[0.1, 0.0, 0.02]], [[0.0, 0.1, -0.01]]]
            ),
            "transition_valid": torch.ones(2, 1),
            "future_target_xy": torch.tensor([[[1.0, 0.1]], [[0.8, -0.1]]]),
            "future_target_xy_valid": torch.ones(2, 1),
            "future_target_visible": torch.ones(2, 1),
            "future_target_visibility_valid": torch.ones(2, 1),
        }
        weights = {
            "waypoint": 1.0,
            "waypoint_delta": 1.0,
            "waypoint_terminal": 1.0,
            "waypoint_radial_progress": 1.0,
            "waypoint_path_length": 1.0,
            "stop": 0.5,
            "bbox": 0.5,
            "visibility": 0.5,
            "identity_or_binding": 0.1,
            "ego": 0.1,
            "world_action": 0.1,
            "inverse": 0.1,
        }
        loss, losses = compute_architecture_v1_loss(outputs, formal_batch, weights)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(losses["world_action"]), 0.0)
        self.assertGreater(float(losses["inverse"]), 0.0)
        self.assertIsNotNone(model.forward_dynamics[0].weight.grad)
        self.assertIsNotNone(model.inverse_dynamics[0].weight.grad)
        self.assertIsNotNone(policy.fusion[0].weight.grad)

    def test_uwb_has_geometric_bias_and_independent_continuous_token(self):
        torch.manual_seed(11)
        config = tiny_config()
        model = EndToEndFollowPolicy(StubDA3SmallL11Backbone(config), config).eval()
        values = batch(config, count=1)
        inputs = {
            key: value
            for key, value in values.items()
            if key not in {"target_waypoints", "waypoint_mask"}
        }
        first = model(**inputs)
        shifted = dict(inputs)
        shifted["camera_from_base"] = inputs["camera_from_base"].clone()
        shifted["camera_from_base"][:, 0, 3] = 5.0
        second = model(**shifted)
        self.assertFalse(
            torch.allclose(first["uwb_patch_bias"], second["uwb_patch_bias"])
        )
        self.assertTrue(torch.allclose(first["z_uwb"], second["z_uwb"]))

    def test_uwb_bias_disables_invalid_stale_and_behind_camera_inputs(self):
        config = tiny_config()
        projector = GeometricUWBProjector(config)
        intrinsics, transform = camera_tensors(3)
        xy = torch.tensor([[2.0, 0.0], [2.0, 0.0], [-2.0, 0.0]])
        covariance = torch.eye(2).mul(0.04).repeat(3, 1, 1)
        bias, diagnostics = projector(
            xy,
            covariance,
            torch.ones(3),
            torch.tensor([0.0, config.uwb_age_limit_s + 0.1, 0.0]),
            torch.tensor([0.0, 1.0, 1.0]),
            intrinsics,
            transform,
        )
        self.assertTrue(torch.allclose(bias, torch.zeros_like(bias), atol=1e-6))
        self.assertTrue(torch.equal(diagnostics["uwb_projectable"], torch.tensor([True, True, False])))


if __name__ == "__main__":
    unittest.main()
