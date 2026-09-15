import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semantic_labels import label_observation, array_sha256


class SemanticLabelTests(unittest.TestCase):
    def label(self, **changes):
        rgb = np.zeros((6, 8, 3), dtype=np.uint8)
        panoptic = np.zeros((6, 8), dtype=np.uint32)
        panoptic[1:4, 2:6] = 100
        panoptic[4:6, 6:8] = 2002
        kwargs = dict(rgb=rgb, panoptic=panoptic,
                      assigned_humanoid_semantic_ids={0: 100, 2: 2002},
                      environment_step=7, world_time_s=.384,
                      terminal_observation=False)
        kwargs.update(changes)
        return label_observation(**kwargs)

    def test_target_and_distractor_boxes_are_distinct(self):
        row = self.label()
        self.assertEqual(row['target']['mask_area_pixels'], 12)
        self.assertEqual(row['target']['bbox_xyxy_inclusive'], [2, 1, 5, 3])
        self.assertEqual(row['target']['bbox_xyxy_norm'], [.25, 1/6, 5/8, .5])
        self.assertEqual(row['distractors'][0]['mask_area_pixels'], 4)
        self.assertTrue(row['target']['bbox_label_valid'])

    def test_no_target_pixels_is_valid_invisible_without_bbox(self):
        row = self.label(panoptic=np.zeros((6, 8), dtype=np.uint16))
        self.assertFalse(row['target']['visible'])
        self.assertIsNone(row['target']['bbox_xyxy_norm'])
        self.assertFalse(row['target']['bbox_label_valid'])
        self.assertTrue(row['visibility_label_valid'])

    def test_single_pixel_visible_without_degenerate_bbox_supervision(self):
        panoptic = np.zeros((6, 8), dtype=np.int32)
        panoptic[2, 3] = 100
        row = self.label(panoptic=panoptic)
        self.assertTrue(row['target']['visible'])
        self.assertFalse(row['target']['bbox_label_valid'])

    def test_after_action_observation_aligns_with_next_policy_call(self):
        row = self.label()
        self.assertEqual((row['after_source_action_step'], row['policy_call_index'], row['next_source_action_step']), (7, 7, 8))
        row = self.label(environment_step=0)
        self.assertIsNone(row['after_source_action_step'])
        self.assertEqual(row['next_source_action_step'], 1)
        row = self.label(terminal_observation=True)
        self.assertIsNone(row['next_source_action_step'])

    def test_labels_do_not_infer_stop_identity_or_motion_permission(self):
        row = self.label()
        for key in ('stop_label_available', 'visual_identity_prediction_label_available',
                    'binding_label_available', 'motion_permission_label_available',
                    'ego_label_available', 'occluded_vs_out_of_view_label_available'):
            self.assertIs(row[key], False)

    def test_misaligned_and_noninteger_panoptic_rejected(self):
        for panoptic in (np.zeros((5, 8), dtype=np.uint32), np.zeros((6, 8)),
                         -np.ones((6, 8), dtype=np.int32), np.zeros((6, 8, 2), dtype=np.uint32)):
            with self.subTest(shape=panoptic.shape, dtype=panoptic.dtype):
                with self.assertRaises(ValueError): self.label(panoptic=panoptic)
        self.assertTrue(self.label(panoptic=np.zeros((6, 8, 1), dtype=np.uint32))['visibility_label_valid'])

    def test_ambiguous_assignment_rejected(self):
        for assigned in ({0: 100, 2: 100}, {2: 100}, {0: 100, 1: 2001},
                         {'0': 100}, {0: True}, {0: -1}):
            with self.subTest(assigned=assigned):
                with self.assertRaises(ValueError): self.label(assigned_humanoid_semantic_ids=assigned)

    def test_invalid_time_and_index_rejected(self):
        for value in (float('nan'), float('inf'), -1, True, None):
            with self.assertRaises(ValueError): self.label(world_time_s=value)
        for value in (-1, True, 1.5):
            with self.assertRaises(ValueError): self.label(environment_step=value)

    def test_rgb_type_and_shape_rejected(self):
        for rgb in (np.zeros((6, 8, 4), dtype=np.uint8), np.zeros((6, 8, 3)),
                    np.zeros((0, 8, 3), dtype=np.uint8)):
            with self.assertRaises(ValueError): self.label(rgb=rgb)

    def test_hash_covers_pixels_dtype_shape_and_ignores_storage_layout(self):
        array = np.arange(24, dtype=np.uint32).reshape(4, 6)
        self.assertEqual(array_sha256(array), array_sha256(np.asfortranarray(array)))
        self.assertNotEqual(array_sha256(array), array_sha256(array.reshape(6, 4)))
        self.assertNotEqual(array_sha256(array), array_sha256(array.astype(np.int32)))
        other = array.copy(); other[0, 0] = 99
        self.assertNotEqual(array_sha256(array), array_sha256(other))


if __name__ == '__main__':
    unittest.main()
