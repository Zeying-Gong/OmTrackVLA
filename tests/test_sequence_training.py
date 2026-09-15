"""CPU regression tests; install under tests after installing the helper."""
import unittest

import torch

from omtrackvla.models.end_to_end import (
    ArchitectureV1Config, EndToEndFollowPolicy, StubDA3SmallL11Backbone,
)
from omtrackvla.training.sequence_training import (
    phase3_sequence_loss, sequence_model_inputs, slice_sequence_inputs,
    unroll_policy_sequence,
)


def fixture(steps=5):
    torch.manual_seed(43)
    config = ArchitectureV1Config(
        image_height=28, image_width=42, patch_size=14, backbone_dim=48,
        policy_dim=32, attention_heads=4, adapter_layers=1, adapter_bottleneck=8,
    )
    policy = EndToEndFollowPolicy(StubDA3SmallL11Backbone(config), config).eval()
    rotation = torch.tensor([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
    transform = torch.eye(4)[None]
    transform[:, :3, :3] = rotation
    inputs = {
        "initial_rgb": torch.rand(1, 3, 28, 42),
        "initial_bbox": torch.tensor([[.2, .1, .7, .9]]),
        "ego_rgb": torch.rand(1, steps, config.history_size, 3, 28, 42, requires_grad=True),
        "visual_initialization_valid": torch.zeros(1),
        "rgb_valid": torch.ones(1, steps), "binding_valid": torch.zeros(1, steps),
        "uwb_xy": torch.tensor([[[2., 0.]]]).expand(1, steps, 2),
        "uwb_covariance_xy": torch.eye(2)[None, None].expand(1, steps, 2, 2) * .04,
        "uwb_quality": torch.ones(1, steps), "uwb_age_s": torch.zeros(1, steps),
        "uwb_valid": torch.ones(1, steps),
        "camera_intrinsics": torch.tensor([[[20., 0., 21.], [0., 20., 14.], [0., 0., 1.]]]),
        "camera_from_base": transform,
    }
    return policy, inputs


class SequenceTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_matches_existing_sequence_and_chunk_continuation(self):
        policy, inputs = fixture()
        with torch.no_grad():
            expected = policy.forward_sequence(**inputs)
            actual, _ = unroll_policy_sequence(policy, inputs)
            _, state = unroll_policy_sequence(policy, slice_sequence_inputs(inputs, 0, 3))
            tail, _ = unroll_policy_sequence(policy, slice_sequence_inputs(inputs, 3, 5), state=state)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])
            torch.testing.assert_close(tail[key], expected[key][:, 3:])

    def test_final_anchor_loss_reaches_earlier_observations(self):
        policy, inputs = fixture()
        outputs, _ = unroll_policy_sequence(policy, inputs)
        outputs["waypoints"][:, -1].square().sum().backward()
        self.assertGreater(float(inputs["ego_rgb"].grad[:, 0].abs().sum()), 0.)
        self.assertGreater(float(policy.gru.weight_hh.grad.abs().sum()), 0.)

    def test_burnin_reconstructs_state_but_has_no_prefix_gradient(self):
        policy, inputs = fixture()
        with torch.no_grad():
            expected = policy.forward_sequence(**inputs)
        outputs, _ = unroll_policy_sequence(policy, inputs, burn_in_steps=3)
        torch.testing.assert_close(outputs["waypoints"], expected["waypoints"][:, 3:])
        outputs["waypoints"][:, -1].square().sum().backward()
        self.assertEqual(float(inputs["ego_rgb"].grad[:, :3].abs().sum()), 0.)
        self.assertGreater(float(inputs["ego_rgb"].grad[:, 3].abs().sum()), 0.)

    def test_tbptt_cuts_all_recurrent_gradient_paths(self):
        policy, inputs = fixture(6)
        outputs, _ = unroll_policy_sequence(policy, inputs, tbptt_steps=3)
        outputs["waypoints"][:, -1].square().sum().backward()
        self.assertEqual(float(inputs["ego_rgb"].grad[:, :3].abs().sum()), 0.)
        self.assertGreater(float(inputs["ego_rgb"].grad[:, 3].abs().sum()), 0.)

    def test_only_real_anchor_label_contributes(self):
        predicted = torch.zeros(1, 3, 8, 2, requires_grad=True)
        predicted.data[..., 1:, :] = .2
        stop = torch.zeros(1, 3, 1, requires_grad=True)
        target = torch.zeros_like(predicted)
        target[:, :2] = float("nan")  # no labels in the prefix; must never enter a loss
        labels = {
            "target_waypoints": target, "waypoint_mask": torch.ones(1, 3, 8, dtype=torch.bool),
            "supervision_mask": torch.tensor([[False, False, True]]),
            "stop_target": torch.tensor([[float("nan"), float("nan"), 1.]]),
        }
        loss, _ = phase3_sequence_loss(
            {"waypoints": predicted, "stop_logit": stop}, labels,
            {"waypoint": 1., "stop": 1.},
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(float(predicted.grad[:, :2].abs().sum()), 0.)
        self.assertGreater(float(predicted.grad[:, 2].abs().sum()), 0.)
        self.assertEqual(float(stop.grad[:, :2].abs().sum()), 0.)
        self.assertGreater(float(stop.grad[:, 2].abs().sum()), 0.)

    def test_four_frame_anchor_is_not_a_policy_sequence(self):
        _, inputs = fixture()
        inputs["ego_rgb"] = inputs["ego_rgb"][:, 0]
        with self.assertRaisesRegex(ValueError, "four history frames are one policy step"):
            sequence_model_inputs(inputs)

    def test_bfloat16_positive_logits_keep_bce_gradient_outside_autocast(self):
        for name, output_key, label_key in (
            ("stop", "stop_logit", "stop_target"),
            ("visibility", "visibility_logit", "target_visible"),
            ("identity_or_binding", "binding_logit", "binding_target"),
        ):
            with self.subTest(loss=name):
                logit = torch.full((1, 1, 1), 10., dtype=torch.bfloat16, requires_grad=True)
                outputs = {"waypoints": torch.zeros(1, 1, 8, 2), output_key: logit}
                labels = {label_key: torch.ones(1, 1)}
                loss, _ = phase3_sequence_loss(outputs, labels, {name: 1.})
                self.assertEqual(loss.dtype, torch.float32)
                loss.backward()
                self.assertTrue(torch.isfinite(logit.grad).all())
                self.assertLess(float(logit.grad.item()), 0.)
                self.assertGreater(float(logit.grad.abs().item()), 1e-5)

    def test_bfloat16_ego_cosine_is_nonnegative_and_keeps_gradient(self):
        for yaw_component in (.01, .2):
            predicted = torch.tensor([[[0., 0., .01, 1.]]], dtype=torch.bfloat16, requires_grad=True)
            target = torch.tensor([[[0., 0., yaw_component, 1.]]])
            loss, _ = phase3_sequence_loss(
                {"waypoints": torch.zeros(1, 1, 8, 2), "xi_hat": predicted},
                {"ego_motion_target": target}, {"ego": 1.})
            self.assertEqual(loss.dtype, torch.float32)
            self.assertGreaterEqual(float(loss), 0.)
            loss.backward()
            self.assertTrue(torch.isfinite(predicted.grad).all())
            if yaw_component == .2:
                self.assertGreater(float(predicted.grad.abs().sum()), 0.)


if __name__ == "__main__":
    unittest.main()
