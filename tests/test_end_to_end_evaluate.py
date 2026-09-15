import unittest

import torch

from omtrackvla.evaluation.end_to_end_evaluate import (
    _bbox_iou,
    _horizon_diagnostics,
    _path_length,
)


class EndToEndEvaluateTest(unittest.TestCase):
    def test_bbox_iou_preserves_existing_metric_contract(self):
        predicted = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.2, 0.2]])
        target = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.8, 0.8, 1.0, 1.0]])
        self.assertEqual(_bbox_iou(predicted, target).tolist(), [1.0, 0.0])

    def test_path_length_detects_short_predictions_without_rescaling(self):
        waypoints = torch.tensor(
            [
                [[0.0, 0.0], [3.0, 4.0], [3.0, 8.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
        lengths = _path_length(waypoints)
        self.assertEqual(lengths.tolist(), [9.0, 0.0])

    def test_path_length_rejects_invalid_waypoint_shape(self):
        with self.assertRaisesRegex(ValueError, "horizon,2"):
            _path_length(torch.zeros(2, 1, 3))

    def test_horizon_diagnostics_report_radius_and_mask_nonfinite_paths(self):
        predicted = torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [float("nan"), 0.0]]]
        )
        target = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
        error, predicted_radius, target_radius, valid = _horizon_diagnostics(
            predicted, target, torch.ones(1, 3, dtype=torch.bool)
        )
        self.assertEqual(error.shape, torch.Size([1, 3]))
        self.assertEqual(predicted_radius.shape, torch.Size([1, 3]))
        self.assertEqual(target_radius.tolist(), [[0.0, 2.0, 3.0]])
        self.assertEqual(valid.tolist(), [[False, False, False]])
        self.assertTrue(torch.isfinite(error.masked_fill(~valid, 0.0)).all())


if __name__ == "__main__":
    unittest.main()
