import unittest

import torch

from omtrackvla.evaluation.phase2_evaluate import _metrics, _wrapped_angle_error


class Phase2EvaluateTest(unittest.TestCase):
    def test_heading_error_wraps_at_pi(self):
        almost_left_down = torch.tensor([[-1.0, -0.01]])
        almost_left_up = torch.tensor([[-1.0, 0.01]])
        error = _wrapped_angle_error(almost_left_down, almost_left_up)
        self.assertLess(float(error[0]), 0.03)

    def test_metric_reduction_and_exact_stop(self):
        values = torch.tensor(
            [4.0, 4.0, 2.0, 4.0, 1.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=torch.float64,
        )
        metrics = _metrics(values, safety=True)
        self.assertEqual(metrics["sample_count"], 4)
        self.assertAlmostEqual(metrics["waypoint_ade_m"], 0.5)
        self.assertAlmostEqual(metrics["waypoint_fde_m"], 1.0)
        self.assertEqual(metrics["exact_stop_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
