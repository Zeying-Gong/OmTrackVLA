import unittest
from types import SimpleNamespace

import torch
from torch import nn

from omtrackvla.data.end_to_end_phase1 import _left_padded_window
from omtrackvla.models.end_to_end_phase1 import (
    ArchitectureV1Phase1Model,
    compute_phase1_loss,
)


class _PolicyStub(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.config = SimpleNamespace(policy_dim=dim)
        self.scale = nn.Parameter(torch.ones(()))

    def forward_sequence(self, **inputs):
        batch = inputs["dummy"].shape[0]
        device = inputs["dummy"].device
        base = self.scale * torch.ones(batch, 2, self.config.policy_dim, device=device)
        return {
            "w_t": base,
            "z_target": base * 0.5,
            "target_memory": base * 0.25,
            "bbox_pred": torch.sigmoid(base[..., :4]),
            "visibility_logit": base[..., :1],
            "xi_hat": base[..., :4],
        }


class _TeacherStub(nn.Module):
    def __init__(self, output_dim: int = 6) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, output_dim, kernel_size=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.projection(value).mean(dim=(2, 3))


class EndToEndPhase1Test(unittest.TestCase):
    def test_left_padding_produces_exact_history(self):
        frames = [torch.full((3, 2, 2), float(index)) for index in range(2)]
        result = _left_padded_window(frames, 4)
        self.assertEqual(result.shape, (4, 3, 2, 2))
        self.assertTrue(torch.equal(result[0], frames[0]))
        self.assertTrue(torch.equal(result[-1], frames[1]))

    def test_training_only_heads_and_losses_are_connected(self):
        model = ArchitectureV1Phase1Model(_PolicyStub())
        outputs = model(
            transition_action=torch.tensor([[[0.1, 0.0, 0.02]], [[0.0, 0.1, -0.01]]]),
            dummy=torch.ones(2, 1),
        )
        self.assertEqual(outputs["future_world_hat"].shape, (2, 1, 8))
        self.assertEqual(outputs["inverse_motion_hat"].shape, (2, 1, 4))
        batch = {
            "identity_label_valid": torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
            "target_visible": torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
            "target_bbox": torch.zeros(2, 2, 4),
            "ego_motion_target": torch.tensor(
                [
                    [[0.0, 0.0, 0.0, 1.0], [0.1, 0.0, 0.02, 0.99]],
                    [[0.0, 0.0, 0.0, 1.0], [0.0, 0.1, -0.01, 0.99]],
                ]
            ),
            "ego_motion_valid": torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
            "transition_valid": torch.ones(2, 1),
            "future_target_xy": torch.tensor([[[1.0, 0.2]], [[0.0, 0.0]]]),
            "future_target_xy_valid": torch.tensor([[1.0], [0.0]]),
            "future_target_visible": torch.tensor([[1.0], [0.0]]),
            "future_target_visibility_valid": torch.tensor([[1.0], [0.0]]),
        }
        weights = {
            "bbox": 0.5,
            "visibility": 0.5,
            "identity_or_binding": 0.1,
            "ego": 0.1,
            "world_action": 0.1,
            "inverse": 0.1,
        }
        loss, losses = compute_phase1_loss(
            outputs, batch, weights, visibility_negative_weight=10.0
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0.0)
        self.assertGreater(float(losses["world_action"]), 0.0)
        loss.backward()
        self.assertIsNotNone(model.forward_dynamics[0].weight.grad)
        self.assertIsNotNone(model.inverse_dynamics[0].weight.grad)

    def test_osnet_teacher_is_frozen_label_only_and_updates_student(self):
        teacher = _TeacherStub()
        model = ArchitectureV1Phase1Model(
            _PolicyStub(), identity_teacher=teacher, identity_teacher_dim=6
        )
        model.train()
        self.assertFalse(teacher.training)
        self.assertFalse(any(parameter.requires_grad for parameter in teacher.parameters()))
        outputs = model(
            transition_action=torch.zeros(2, 1, 3),
            teacher_bbox=torch.tensor([0.1, 0.1, 0.9, 0.9]).repeat(2, 2, 1),
            teacher_valid=torch.tensor([[False, True], [False, True]]),
            ego_rgb=torch.rand(2, 2, 4, 3, 16, 8),
            dummy=torch.ones(2, 1),
        )
        self.assertEqual(outputs["osnet_teacher_embedding"].shape, (2, 2, 6))
        self.assertEqual(outputs["osnet_student_embedding"].shape, (2, 2, 6))
        self.assertEqual(int(outputs["osnet_teacher_valid"].sum()), 2)
        valid = outputs["osnet_teacher_valid"].bool()
        student = torch.nn.functional.normalize(
            outputs["osnet_student_embedding"][valid], dim=-1
        )
        target = torch.nn.functional.normalize(
            outputs["osnet_teacher_embedding"][valid], dim=-1
        )
        loss = (1.0 - (student * target).sum(dim=-1)).mean()
        loss.backward()
        self.assertIsNotNone(model.identity_teacher_projector.weight.grad)
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))


if __name__ == "__main__":
    unittest.main()
