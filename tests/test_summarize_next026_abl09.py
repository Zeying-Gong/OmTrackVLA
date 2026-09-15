import unittest

from scripts.summarize_next026_abl09 import _comparison, _policy_metrics


class SummarizeNext026Abl09Test(unittest.TestCase):
    def test_accepts_either_path_ratio_key(self):
        metrics = _policy_metrics(
            {
                "waypoint_ade_m": 0.2,
                "waypoint_fde_m": 0.3,
                "path_length_ratio_mean": 0.75,
                "finite_prediction_coverage": 1.0,
            }
        )
        self.assertEqual(metrics["path_length_ratio"], 0.75)

    def test_architecture_improvement_is_relative_to_baseline(self):
        comparison = _comparison(
            {"waypoint_ade_m": 0.2, "waypoint_fde_m": 0.4},
            {"waypoint_ade_m": 0.15, "waypoint_fde_m": 0.3},
        )
        self.assertAlmostEqual(
            comparison["waypoint_ade_m_relative_architecture_improvement"], 0.25
        )
        self.assertAlmostEqual(
            comparison["waypoint_fde_m_baseline_minus_architecture"], 0.1
        )


if __name__ == "__main__":
    unittest.main()
