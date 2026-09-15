"""CPU compatibility checks against the existing bbox convention and label boundary."""
import ast
import hashlib
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from semantic_labels import array_sha256, label_observation


def load_existing_bbox_normalizer():
    # Execute only the real pure normalization function, avoiding torch/model imports.
    candidates = [ROOT.parent/"cold_target_review/omtrackvla/data/end_to_end_training.py",
                  ROOT.parent.parent/"omtrackvla/data/end_to_end_training.py"]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError("the independently inspected source snapshot is missing")
    source = path.read_text(encoding="utf-8")
    if hashlib.sha256(path.read_bytes()).hexdigest() != "a52220a8728582bacb863ebe6a1aecb157c05d97fde5bdce1dd20016020eb68a":
        raise ValueError("the independently inspected source snapshot changed")
    parsed = ast.parse(source)
    function = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "_bbox_norm")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["_bbox_norm"]


class SemanticCompatibilityTests(unittest.TestCase):
    def label(self, panoptic, **changes):
        kwargs = {"rgb": np.zeros((*panoptic.shape[:2], 3), dtype=np.uint8), "panoptic": panoptic,
                  "assigned_humanoid_semantic_ids": {0: 100}, "environment_step": 0,
                  "world_time_s": 0., "terminal_observation": False}
        kwargs.update(changes)
        return label_observation(**kwargs)

    def test_all_nondegenerate_rectangles_match_real_training_normalizer(self):
        normalize = load_existing_bbox_normalizer()
        height, width = 4, 5
        for y1 in range(height-1):
            for y2 in range(y1+1, height):
                for x1 in range(width-1):
                    for x2 in range(x1+1, width):
                        panoptic = np.zeros((height, width), dtype=np.uint32)
                        panoptic[y1:y2+1, x1:x2+1] = 100
                        target = self.label(panoptic)["target"]
                        box = [x1, y1, x2, y2]
                        self.assertEqual(target["bbox_xyxy_inclusive"], box)
                        self.assertEqual(target["bbox_xyxy_norm"], normalize(box, width, height))
                        self.assertTrue(target["bbox_label_valid"])

    def test_full_frame_box_preserves_inclusive_endpoint_convention(self):
        target = self.label(np.full((4, 5), 100, dtype=np.uint32))["target"]
        self.assertEqual(target["bbox_xyxy_inclusive"], [0, 0, 4, 3])
        self.assertEqual(target["bbox_xyxy_norm"], [0., 0., .8, .75])
        self.assertEqual(target["mask_area_pixels"], 20)

    def test_single_pixel_single_row_and_single_column_have_visibility_only(self):
        normalize = load_existing_bbox_normalizer()
        for pixels in ([(1, 2)], [(1, 1), (1, 2), (1, 3)], [(0, 2), (1, 2), (2, 2)]):
            panoptic = np.zeros((4, 5), dtype=np.uint32)
            for y, x in pixels:
                panoptic[y, x] = 100
            row = self.label(panoptic)
            self.assertTrue(row["visibility_label_valid"])
            self.assertTrue(row["target"]["visible"])
            self.assertFalse(row["target"]["bbox_label_valid"])
            self.assertEqual(normalize(row["target"]["bbox_xyxy_inclusive"], 5, 4), [0., 0., 0., 0.])

    def test_disconnected_target_pixels_use_mask_union_not_distractor_extent(self):
        panoptic = np.zeros((5, 7), dtype=np.uint32)
        panoptic[1, 1] = panoptic[3, 4] = 100
        panoptic[0, 6] = 2002
        row = self.label(panoptic, assigned_humanoid_semantic_ids={0: 100, 2: 2002})
        self.assertEqual(row["target"]["bbox_xyxy_inclusive"], [1, 1, 4, 3])
        self.assertEqual(row["target"]["mask_area_pixels"], 2)
        self.assertEqual(row["distractors"][0]["bbox_xyxy_inclusive"], [6, 0, 6, 0])
        self.assertFalse(row["distractors"][0]["bbox_label_valid"])

    def test_reset_through_terminal_observation_does_not_invent_next_action(self):
        rows = []
        for step, world_time in enumerate((0., .06, .13, .21)):
            rgb = np.full((4, 5, 3), step, dtype=np.uint8)
            row = self.label(np.zeros((4, 5), dtype=np.uint32), rgb=rgb,
                             environment_step=step, world_time_s=world_time, terminal_observation=step == 3)
            self.assertEqual(row["rgb_array_sha256"], array_sha256(rgb))
            rows.append(row)
        self.assertEqual([row["policy_call_index"] for row in rows], [0, 1, 2, 3])
        self.assertEqual([row["after_source_action_step"] for row in rows], [None, 1, 2, 3])
        self.assertEqual([row["next_source_action_step"] for row in rows], [1, 2, 3, None])
        self.assertEqual(len({row["rgb_array_sha256"] for row in rows}), 4)

    def test_channel_squeeze_is_explicit_canonical_panoptic_hash(self):
        panoptic = np.arange(20, dtype=np.uint32).reshape(4, 5)
        row = self.label(panoptic[..., None])
        self.assertEqual(row["panoptic_array_sha256"], array_sha256(panoptic))
        self.assertNotEqual(row["panoptic_array_sha256"], array_sha256(panoptic[..., None]))

    def test_visible_invisible_and_terminal_rows_never_grant_control_or_identity(self):
        forbidden = ("stop_label_available", "motion_permission_label_available", "visual_identity_prediction_label_available",
                     "binding_label_available", "ego_label_available", "occluded_vs_out_of_view_label_available")
        for target_present in (False, True):
            panoptic = np.full((4, 5), 100 if target_present else 0, dtype=np.uint32)
            for terminal in (False, True):
                row = self.label(panoptic, terminal_observation=terminal)
                self.assertTrue(row["gt_used_only_on_label_or_audit_side"])
                self.assertFalse(row["later_bbox_used_by_model"])
                self.assertTrue(all(row[key] is False for key in forbidden))


if __name__ == "__main__":
    unittest.main()
