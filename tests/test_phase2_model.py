import unittest

import torch

from omtrackvla.data.phase2 import CONDITION_MODES
from omtrackvla.models.phase2 import Phase2WaypointPolicy, compute_phase2_loss


class Phase2WaypointPolicyTest(unittest.TestCase):
    def _batch(self, modes):
        size = len(modes)
        return {
            "visual_xy": torch.zeros(size, 2),
            "visual_confidence": torch.ones(size),
            "visual_valid": torch.ones(size),
            "uwb_xy": torch.ones(size, 2),
            "uwb_quality": torch.ones(size),
            "uwb_valid": torch.ones(size),
            "uwb_age_s": torch.zeros(size),
            "condition_index": torch.tensor(
                [CONDITION_MODES.index(mode) for mode in modes], dtype=torch.long
            ),
            "waypoints": torch.zeros(size, 8, 2),
            "waypoint_mask": torch.ones(size, 8, dtype=torch.bool),
        }

    def test_forward_preserves_anchor_and_hard_stops_safe_mode(self):
        model = Phase2WaypointPolicy(hidden_dim=16, mode_dim=4)
        batch = self._batch(("visual_uwb", "safe_stop"))
        outputs = model(**{key: batch[key] for key in (
            "visual_xy", "visual_confidence", "visual_valid", "uwb_xy",
            "uwb_quality", "uwb_valid", "uwb_age_s", "condition_index",
        )})
        self.assertEqual(outputs["waypoints"].shape, (2, 8, 2))
        self.assertTrue(torch.equal(outputs["waypoints"][:, 0], torch.zeros(2, 2)))
        self.assertTrue(torch.equal(outputs["waypoints"][1], torch.zeros(8, 2)))
        self.assertTrue(outputs["safe_stop"][1])

    def test_loss_is_finite_and_backpropagates(self):
        model = Phase2WaypointPolicy(hidden_dim=16, mode_dim=4)
        batch = self._batch(("visual_only", "uwb_only"))
        outputs = model(**{key: batch[key] for key in (
            "visual_xy", "visual_confidence", "visual_valid", "uwb_xy",
            "uwb_quality", "uwb_valid", "uwb_age_s", "condition_index",
        )})
        loss = compute_phase2_loss(outputs, batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_loss_ignores_contract_anchor_point(self):
        model = Phase2WaypointPolicy(hidden_dim=16, mode_dim=4)
        batch = self._batch(("visual_uwb",))
        outputs = model(**{key: batch[key] for key in (
            "visual_xy", "visual_confidence", "visual_valid", "uwb_xy",
            "uwb_quality", "uwb_valid", "uwb_age_s", "condition_index",
        )})
        baseline = compute_phase2_loss(outputs, batch)
        batch["waypoints"][:, 0] = 1000.0
        torch.testing.assert_close(compute_phase2_loss(outputs, batch), baseline)


if __name__ == "__main__":
    unittest.main()
