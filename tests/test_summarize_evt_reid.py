import unittest

from scripts.summarize_evt_reid import aggregate_counts, episode_counts


class SummarizeEvtReidTest(unittest.TestCase):
    def test_counts_use_pre_action_perception_visibility(self):
        data = {
            "steps": [
                {
                    "visible": False,
                    "perception_gt_visible": True,
                    "perception_confidence": 0.9,
                    "target_selection_correct": True,
                    "target_bbox_iou": 0.75,
                    "perception_memory_updated": True,
                    "perception_candidates": [{"target_match": True}],
                },
                {
                    "visible": True,
                    "perception_gt_visible": False,
                    "perception_confidence": 0.8,
                    "target_selection_correct": False,
                    "target_bbox_iou": 0.0,
                    "perception_memory_updated": True,
                    "perception_candidates": [{"target_match": False}],
                },
            ]
        }
        aggregate = aggregate_counts([episode_counts(data)])
        self.assertEqual(aggregate["gt_visible"], 1)
        self.assertEqual(aggregate["detected"], 2)
        self.assertEqual(aggregate["correct"], 1)
        self.assertEqual(aggregate["perception_target_precision"], 0.5)
        self.assertEqual(aggregate["perception_target_recall"], 1.0)
        self.assertEqual(aggregate["perception_mean_target_iou"], 0.375)
        self.assertEqual(aggregate["perception_memory_update_precision"], 0.5)
        self.assertEqual(aggregate["detector_target_recall"], 1.0)

    def test_legacy_records_fall_back_to_post_action_visibility(self):
        aggregate = aggregate_counts(
            [episode_counts({"steps": [{"visible": True}]})]
        )
        self.assertEqual(aggregate["gt_visible"], 1)


if __name__ == "__main__":
    unittest.main()
