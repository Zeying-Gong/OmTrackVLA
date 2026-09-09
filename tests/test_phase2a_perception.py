import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from omtrackvla.evaluation.phase2a_perception_gate import evaluate_gate
from omtrackvla.evaluation.pretrained_identity import IdentitySequenceMetrics, _aggregate
from omtrackvla.models.candidate_fusion import FEATURE_NAMES
from omtrackvla.training.train_temporal_candidate_fusion import (
    _calibrate_or_raise,
    _hard_pairs,
    _load_rows,
)


class Phase2APerceptionTest(unittest.TestCase):
    def test_record_loader_adds_frame_context_and_hard_pairs(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sequence.jsonl"
            record = {
                "split": "train",
                "sequence_id": "0001",
                "frame_index": 4,
                "candidates": [
                    {
                        "detector_score": 0.9,
                        "association_score": 0.8,
                        "target_iou": 0.7,
                        "target_match_iou_0_5": True,
                    },
                    {
                        "detector_score": 0.8,
                        "association_score": 0.7,
                        "target_iou": 0.0,
                        "target_match_iou_0_5": False,
                    },
                ],
            }
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            rows, frames, sequences, paths = _load_rows((path,), "train")
            pairs = _hard_pairs(rows, frames, sequences, negatives_per_frame=1)

        self.assertEqual(paths, [path.resolve()])
        self.assertEqual(sequences, {"0001"})
        self.assertEqual(pairs, [(0, 1)])
        count_feature = rows[0].features[
            FEATURE_NAMES.index("log1p_candidate_count")
        ]
        self.assertAlmostEqual(count_feature, np.log(3.0))

    def test_record_loader_keeps_rollouts_with_same_frame_id_separate(self):
        with TemporaryDirectory() as directory:
            roots = []
            for rollout in ("legacy", "on_policy"):
                root = Path(directory) / rollout
                root.mkdir()
                path = root / "0001.jsonl"
                path.write_text(
                    json.dumps(
                        {
                            "split": "train",
                            "sequence_id": "0001",
                            "frame_index": 4,
                            "candidates": [
                                {
                                    "target_iou": 0.7,
                                    "target_match_iou_0_5": True,
                                },
                                {
                                    "target_iou": 0.0,
                                    "target_match_iou_0_5": False,
                                },
                            ],
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                roots.append(root)

            rows, frames, sequences, _ = _load_rows(tuple(roots), "train")
            pairs = _hard_pairs(rows, frames, sequences, negatives_per_frame=1)

        self.assertEqual(len(frames), 2)
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(rows[a].frame_key == rows[b].frame_key for a, b in pairs))

    def test_calibration_fails_closed_when_precision_floor_is_unreachable(self):
        probabilities = np.asarray([0.9, 0.8], dtype=np.float32)
        labels = np.asarray([0.0, 1.0], dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "cannot satisfy precision floor"):
            _calibrate_or_raise(
                probabilities,
                labels,
                ((0, 1),),
                "tracking",
                precision_floor=0.9,
                beta=0.5,
            )

    def test_gate_requires_improvement_and_safety(self):
        target = (0.0, 0.0, 10.0, 20.0)
        baseline_metric = IdentitySequenceMetrics("s")
        baseline_metric.update(True, target, None)
        baseline_metric.update(False, None, None)
        baseline_metric.update(True, target, target)
        baseline_metric.finish()
        candidate_metric = IdentitySequenceMetrics("s")
        candidate_metric.update(True, target, target)
        candidate_metric.update(False, None, None)
        candidate_metric.update(True, target, target)
        candidate_metric.finish()
        baseline = _aggregate((baseline_metric.summary(),))
        config = {
            "gate_id": "test",
            "split": "viz_val",
            "expected_sequences": ["s"],
            "expected_counts": {
                "evaluated_frames": 3,
                "gt_visible_frames": 2,
                "gt_absent_frames": 1,
            },
            "thresholds": {
                "end_to_end_absolute_improvement": {"min": 0.4},
                "output_precision_iou_0_5": {"min": 0.9},
                "absent_false_positive_rate": {"max": 0.1},
                "wrong_target_frames_iou_below_0_2": {"max": 0},
                "reappearance_success_rate": {"min": 1.0},
            },
        }

        result = evaluate_gate(
            config, baseline, (candidate_metric.summary(),)
        )

        self.assertTrue(result["passed"])
        self.assertAlmostEqual(
            result["candidate"]["end_to_end_success_iou_0_5"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
