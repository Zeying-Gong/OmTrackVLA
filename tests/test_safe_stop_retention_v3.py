"""Regression checks for changing only the safe-stop trajectory gradient."""
import unittest
import torch
from scripts.train_recovery_sequence_v3 import clean_training_objective
from omtrackvla.training.sequence_training import phase3_sequence_loss


class SafeStopRetentionTest(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(704)
        waypoint = (torch.rand(4, 2, 8, 2) * .1).requires_grad_()
        stop = torch.zeros(4, 2, 1, requires_grad=True)
        labels = {'target_waypoints': torch.zeros_like(waypoint),
                  'waypoint_mask': torch.ones(4, 2, 8, dtype=torch.bool),
                  'mode_index': torch.arange(4),
                  'stop_target': torch.tensor([[0., 0.]] * 3 + [[1., 1.]]),
                  'visual_initialization_valid': torch.tensor([1., 1., 0., 0.]),
                  'binding_valid': torch.tensor([[1., 1.], [1., 1.], [0., 0.], [0., 0.]]),
                  'rgb_valid': torch.ones(4, 2),
                  'uwb_valid': torch.tensor([[1., 1.], [0., 0.], [1., 1.], [0., 0.]])}
        return {'waypoints': waypoint, 'stop_logit': stop}, labels

    def test_only_safe_stop_waypoint_gradient_is_eight_times_base(self):
        outputs, labels = self.fixture()
        weights = {'waypoint': 1., 'stop': .5}
        base, _ = phase3_sequence_loss(outputs, labels, weights)
        ordinary = torch.autograd.grad(base, (outputs['waypoints'], outputs['stop_logit']), retain_graph=True)
        objective = clean_training_objective(outputs, labels, weights, phase3_sequence_loss)
        weighted = torch.autograd.grad(objective, (outputs['waypoints'], outputs['stop_logit']))
        torch.testing.assert_close(weighted[0][:3], ordinary[0][:3], rtol=0, atol=0)
        torch.testing.assert_close(weighted[0][3], ordinary[0][3] * 8)
        torch.testing.assert_close(weighted[1], ordinary[1], rtol=0, atol=0)

    def test_wrong_mode_or_nonzero_stop_target_is_rejected(self):
        for kind in ('duplicate_mode', 'moving_target', 'uwb_valid', 'unequal_mask'):
            outputs, labels = self.fixture()
            if kind == 'duplicate_mode':
                labels['mode_index'][0] = 3
            elif kind == 'moving_target':
                labels['target_waypoints'][3, -1, -1, 0] = .1
            elif kind == 'uwb_valid':
                labels['uwb_valid'][3] = 1
            else:
                labels['waypoint_mask'][3, 0, 1] = False
            with self.assertRaises(ValueError):
                clean_training_objective(outputs, labels, {'waypoint': 1.}, phase3_sequence_loss)

    def test_experiment_multiplier_cannot_silently_change(self):
        outputs, labels = self.fixture()
        with self.assertRaises(ValueError):
            clean_training_objective(outputs, labels, {'waypoint': 1.}, phase3_sequence_loss, multiplier=4.)


if __name__ == '__main__':
    unittest.main()
