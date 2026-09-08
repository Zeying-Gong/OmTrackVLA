import unittest

from scripts.audit_sage3d_bboxes import (
    bbox_iou,
    choose_entries,
    clip_bbox,
    evenly_spaced,
    summarize,
)


class SageBBoxAuditTest(unittest.TestCase):
    def test_clip_bbox_reports_empty_projection(self):
        box, valid = clip_bbox([900.0, 0.0, 640.0, 480.0], 640, 480)
        self.assertEqual(box, [640, 0.0, 640, 480.0])
        self.assertFalse(valid)

    def test_bbox_iou_and_horizontal_mirror_geometry(self):
        source = [400.0, 20.0, 500.0, 220.0]
        mirrored = [640.0 - source[2], source[1], 640.0 - source[0], source[3]]
        person = [142.0, 20.0, 238.0, 220.0]
        self.assertEqual(bbox_iou(source, person), 0.0)
        self.assertGreater(bbox_iou(mirrored, person), 0.9)

    def test_evenly_spaced_includes_sequence_ends(self):
        self.assertEqual(evenly_spaced(list(range(10)), 3), [0, 4, 9])

    def test_entry_selection_round_robins_strata(self):
        entries = [
            {"mode": "at", "cam": "a", "path": f"a/{index}", "steps": 2}
            for index in range(3)
        ] + [
            {"mode": "stt", "cam": "b", "path": f"b/{index}", "steps": 2}
            for index in range(3)
        ]
        selected, population = choose_entries(entries, 4, seed=7)
        counts = {key: 0 for key in population}
        for entry in selected:
            counts[f"{entry['mode']}/{entry['cam']}"] += 1
        self.assertEqual(counts, {"at/a": 2, "stt/b": 2})

    def test_summary_reports_mirror_counterfactual(self):
        base = {
            "source_path": "run/stt/0/cam",
            "mode": "stt",
            "camera": "cam",
            "source_bbox_valid_after_clipping": True,
            "sample_position_name": "initial_visible",
        }
        records = [
            {
                **base,
                "status": "detector_disagreement",
                "best_iou": 0.0,
                "best_horizontal_mirror_iou": 0.75,
            },
            {
                **base,
                "status": "strong_agreement",
                "best_iou": 0.70,
                "best_horizontal_mirror_iou": 0.70,
            },
        ]
        summary = summarize(records, strong_iou=0.5, partial_iou=0.2)
        mirror = summary["horizontal_mirror_counterfactual"]
        self.assertEqual(mirror["original_strong_agreement_frames"], 1)
        self.assertEqual(mirror["mirrored_strong_agreement_frames"], 2)
        self.assertEqual(mirror["original_disagreement_frames"], 1)
        self.assertEqual(mirror["mirrored_disagreement_frames"], 0)


if __name__ == "__main__":
    unittest.main()
