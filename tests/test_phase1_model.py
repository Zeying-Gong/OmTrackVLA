import unittest

import torch

from omtrackvla.models.phase1 import Phase1WorldIdentityModel, compute_phase1_loss


class Phase1ModelTest(unittest.TestCase):
    def test_both_supervision_streams_are_finite(self):
        model = Phase1WorldIdentityModel(feature_dim=16)
        batch = {
            "task_id": torch.tensor([0, 1]),
            "frame0": torch.rand(2, 3, 64, 64),
            "frame1": torch.rand(2, 3, 64, 64),
            "history": torch.rand(2, 3, 3, 64, 64),
            "history_mask": torch.tensor([[True, True, True], [False, False, False]]),
            "target_bbox": torch.tensor([[0.2, 0.1, 0.7, 0.9], [0.0, 0.0, 0.0, 0.0]]),
            "target_visible": torch.tensor([1.0, 0.0]),
            "motion": torch.tensor([[0.0, 0.0, 0.0], [0.3, -0.1, 0.05]]),
        }
        output = model(
            batch["frame0"], batch["frame1"], batch["motion"], batch["history"], batch["history_mask"]
        )
        loss, losses = compute_phase1_loss(output, batch, {})
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(tuple(output["bbox"].shape), (2, 4))
        self.assertEqual(tuple(output["motion"].shape), (2, 3))
        self.assertEqual(set(losses), {
            "identity_visibility", "identity_bbox", "inverse_dynamics",
            "forward_dynamics", "action_free_next_state", "total",
        })
        loss.backward()

    def test_visibility_negative_weight_increases_absent_loss(self):
        outputs = {
            "visibility_logit": torch.tensor([2.0], requires_grad=True),
            "bbox": torch.zeros(1, 4),
            "motion": torch.zeros(1, 3, requires_grad=True),
        }
        batch = {
            "task_id": torch.tensor([0]),
            "target_visible": torch.tensor([0.0]),
            "target_bbox": torch.zeros(1, 4),
        }
        unweighted, _ = compute_phase1_loss(outputs, batch, {})
        weighted, _ = compute_phase1_loss(
            outputs, batch, {}, visibility_negative_weight=10.0
        )
        self.assertAlmostEqual(float(weighted), float(unweighted) * 10.0, places=5)


if __name__ == "__main__":
    unittest.main()
