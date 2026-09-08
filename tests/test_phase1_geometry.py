import math
import unittest

import numpy as np

from omtrackvla.geometry.se2 import (
    intern_pair_to_canonical_se2,
    relative_w2c_to_base_se2,
    robust_translation_scale,
)


class Phase1GeometryTest(unittest.TestCase):
    def test_da3_w2c_conversion(self):
        anchor_world_from_camera = np.eye(4)
        target_world_from_camera = np.eye(4)
        target_world_from_camera[0, 3] = 2.0
        target_world_from_camera[1, 3] = 0.5
        anchor_w2c = np.linalg.inv(anchor_world_from_camera)
        target_w2c = np.linalg.inv(target_world_from_camera)
        result = relative_w2c_to_base_se2(anchor_w2c, target_w2c, np.eye(4))
        np.testing.assert_allclose(result, [2.0, 0.5, 0.0], atol=1e-6)

    def test_intern_basis_change_is_explicit(self):
        anchor = np.eye(4)
        target = np.eye(4)
        target[1, 3] = 1.25
        angle = 0.2
        target[:2, :2] = [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        result = intern_pair_to_canonical_se2(anchor, target, np.eye(4))
        np.testing.assert_allclose(result, [1.25, 0.0, angle], atol=1e-6)

    def test_robust_scale_uses_moving_pairs(self):
        predicted = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        reference = np.asarray([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
        self.assertAlmostEqual(robust_translation_scale(predicted, reference), 2.0)

    def test_robust_scale_rejects_stationary_clip(self):
        with self.assertRaisesRegex(ValueError, "no displacement"):
            robust_translation_scale(np.zeros((3, 2)), np.zeros((3, 2)))


if __name__ == "__main__":
    unittest.main()
