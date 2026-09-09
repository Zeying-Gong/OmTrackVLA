import unittest

import numpy as np

from omtrackvla.models.candidate_fusion import (
    FEATURE_NAMES,
    LEGACY_FEATURE_NAMES,
    CandidateFusionModel,
    feature_vector,
)
from omtrackvla.training.train_candidate_fusion import _selection_metrics


class CandidateFusionTest(unittest.TestCase):
    def test_feature_contract_transforms_missed_steps(self):
        vector = feature_vector(
            {
                "detector_score": 0.9,
                "missed_steps": 3,
                "global_search": 1.0,
            }
        )

        self.assertEqual(vector.shape, (len(FEATURE_NAMES),))
        self.assertAlmostEqual(vector[0], 0.9)
        self.assertAlmostEqual(vector[-2], np.log(4.0))
        self.assertEqual(vector[-1], 1.0)

    def test_feature_contract_exposes_temporal_tracker_state(self):
        vector = feature_vector(
            {
                "association_score": 0.7,
                "identity_score": 0.8,
                "candidate_count": 4,
                "confirmed_track_steps": 7,
            }
        )

        self.assertAlmostEqual(
            vector[FEATURE_NAMES.index("association_score")], 0.7
        )
        self.assertAlmostEqual(
            vector[FEATURE_NAMES.index("identity_score")], 0.8
        )
        self.assertAlmostEqual(
            vector[FEATURE_NAMES.index("log1p_candidate_count")], np.log(5.0)
        )
        self.assertAlmostEqual(
            vector[FEATURE_NAMES.index("log1p_confirmed_track_steps")],
            np.log(8.0),
        )

    def test_legacy_fusion_payload_remains_loadable(self):
        payload = {
            "feature_names": list(LEGACY_FEATURE_NAMES),
            "feature_mean": [0.0] * len(LEGACY_FEATURE_NAMES),
            "feature_scale": [1.0] * len(LEGACY_FEATURE_NAMES),
            "weight1": [[0.0] * len(LEGACY_FEATURE_NAMES)],
            "bias1": [0.0],
            "weight2": [0.0],
            "bias2": 0.0,
            "score_threshold": 0.5,
            "margin_threshold": 0.0,
        }

        model = CandidateFusionModel(payload)

        self.assertEqual(model.feature_names, LEGACY_FEATURE_NAMES)
        self.assertEqual(model.predict({"detector_score": 1.0}), 0.5)

    def test_numpy_fusion_model_predicts_probability(self):
        hidden = 2
        payload = {
            "feature_names": list(FEATURE_NAMES),
            "feature_mean": [0.0] * len(FEATURE_NAMES),
            "feature_scale": [1.0] * len(FEATURE_NAMES),
            "weight1": [[1.0] + [0.0] * (len(FEATURE_NAMES) - 1)] * hidden,
            "bias1": [0.0] * hidden,
            "weight2": [1.0] * hidden,
            "bias2": 0.0,
            "score_threshold": 0.5,
            "margin_threshold": 0.1,
        }
        model = CandidateFusionModel(payload)

        low = model.predict({"detector_score": 0.0})
        high = model.predict({"detector_score": 1.0})

        self.assertAlmostEqual(low, 0.5)
        self.assertGreater(high, 0.8)
        self.assertEqual(model.thresholds(global_search=False), (0.5, 0.1))
        self.assertEqual(model.thresholds(global_search=True), (0.5, 0.1))

    def test_dual_operating_points_distinguish_tracking_and_reacquisition(self):
        payload = {
            "feature_names": list(FEATURE_NAMES),
            "feature_mean": [0.0] * len(FEATURE_NAMES),
            "feature_scale": [1.0] * len(FEATURE_NAMES),
            "weight1": [[0.0] * len(FEATURE_NAMES)],
            "bias1": [0.0],
            "weight2": [0.0],
            "bias2": 0.0,
            "score_threshold": 0.95,
            "margin_threshold": 0.06,
            "operating_points": {
                "tracking": {"score_threshold": 0.92, "margin_threshold": 0.02},
                "reacquisition": {
                    "score_threshold": 0.95,
                    "margin_threshold": 0.06,
                },
            },
        }
        model = CandidateFusionModel(payload)

        self.assertTrue(model.accepts(0.93, 0.03, global_search=False))
        self.assertFalse(model.accepts(0.93, 0.03, global_search=True))

    def test_selection_metric_penalizes_wrong_candidate_as_fp_and_fn(self):
        metrics = _selection_metrics(
            probabilities=np.array((0.8, 0.9), dtype=np.float32),
            labels=np.array((1.0, 0.0), dtype=np.float32),
            frame_rows=((0, 1),),
            score_threshold=0.5,
            margin_threshold=0.0,
        )

        self.assertEqual(metrics["true_positive"], 0)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["false_negative"], 1)


if __name__ == "__main__":
    unittest.main()
