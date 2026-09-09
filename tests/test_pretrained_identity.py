import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from omtrackvla.evaluation.pretrained_identity import (
    IdentitySequenceMetrics,
    _aggregate,
    _candidate_fusion_metadata,
)


class PretrainedIdentityMetricsTest(unittest.TestCase):
    def test_end_to_end_metrics_include_visibility_and_reacquisition(self):
        metrics = IdentitySequenceMetrics("sequence")
        target = (0.0, 0.0, 10.0, 20.0)

        metrics.update(
            True,
            target,
            target,
            candidate_count=2,
            target_candidate_present=True,
            memory_updated=True,
        )
        metrics.update(False, None, target, candidate_count=1)
        metrics.update(False, None, None, candidate_count=0)
        metrics.update(
            True,
            target,
            (20.0, 0.0, 30.0, 20.0),
            candidate_count=2,
            target_candidate_present=True,
        )
        metrics.update(
            True,
            target,
            target,
            candidate_count=1,
            target_candidate_present=True,
            memory_updated=True,
        )
        metrics.finish()
        summary = metrics.summary()

        self.assertEqual(summary["evaluated_frames"], 5)
        self.assertEqual(summary["gt_visible_frames"], 3)
        self.assertEqual(summary["gt_absent_frames"], 2)
        self.assertAlmostEqual(summary["visibility_recall"], 1.0)
        self.assertAlmostEqual(summary["absent_false_positive_rate"], 0.5)
        self.assertAlmostEqual(summary["end_to_end_success_iou_0_5"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["detector_candidate_recall_iou_0_5"], 1.0)
        self.assertAlmostEqual(
            summary["target_selection_accuracy_when_candidate_present"], 2.0 / 3.0
        )
        self.assertAlmostEqual(summary["output_precision_iou_0_5"], 0.5)
        self.assertEqual(summary["wrong_target_track_starts"], 1)
        self.assertEqual(summary["max_consecutive_false_follow_frames"], 1)
        self.assertEqual(summary["max_consecutive_visible_miss_frames"], 1)
        self.assertEqual(summary["reappearance_events"], 1)
        self.assertEqual(summary["reappearance_failures"], 0)
        self.assertEqual(summary["mean_reacquisition_latency_frames"], 1.0)
        self.assertEqual(summary["memory_updates"], 2)

    def test_unresolved_reappearance_is_failure(self):
        metrics = IdentitySequenceMetrics("sequence")
        target = (0.0, 0.0, 10.0, 20.0)
        metrics.update(False, None, None)
        metrics.update(True, target, None)
        metrics.finish()

        summary = metrics.summary()
        self.assertEqual(summary["reappearance_events"], 1)
        self.assertEqual(summary["reappearance_failures"], 1)
        self.assertEqual(summary["reappearance_success_rate"], 0.0)

    def test_aggregate_is_frame_weighted(self):
        first = IdentitySequenceMetrics("first")
        second = IdentitySequenceMetrics("second")
        target = (0.0, 0.0, 10.0, 20.0)
        first.update(True, target, target)
        second.update(True, target, None)
        second.update(True, target, None)
        first.finish()
        second.finish()

        aggregate = _aggregate((first.summary(), second.summary()))

        self.assertEqual(aggregate["sequence_count"], 2)
        self.assertAlmostEqual(aggregate["visibility_recall"], 1.0 / 3.0)
        self.assertAlmostEqual(aggregate["end_to_end_success_iou_0_5"], 1.0 / 3.0)

    def test_candidate_fusion_metadata_records_dual_operating_points(self):
        model = SimpleNamespace(
            operating_points={
                "tracking": {"score_threshold": 0.92, "margin_threshold": 0.02},
                "reacquisition": {
                    "score_threshold": 0.95,
                    "margin_threshold": 0.06,
                },
            }
        )
        with TemporaryDirectory() as directory:
            weights_path = Path(directory) / "fusion.json"
            weights_path.write_text("weights\n", encoding="utf-8")
            metadata = _candidate_fusion_metadata(weights_path, model)

        self.assertIsNotNone(metadata)
        self.assertEqual(len(metadata["weights_sha256"]), 64)
        self.assertEqual(metadata["operating_points"], model.operating_points)


if __name__ == "__main__":
    unittest.main()
