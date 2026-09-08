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
        # Reproduce the real source conventions instead of testing with an
        # identity camera extrinsic, which hides the OpenCV/Habitat bridge.
        base_from_habitat_camera = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0, 0.35],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        opencv_from_habitat_camera = np.diag([1.0, -1.0, -1.0, 1.0])
        opencv_camera_from_base = opencv_from_habitat_camera @ np.linalg.inv(
            base_from_habitat_camera
        )

        anchor_world_from_base = np.eye(4)
        target_world_from_base = np.eye(4)
        angle = 0.2
        target_world_from_base[:2, :2] = [
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ]
        # Habitat planar coordinates are (right, forward).  This is canonical
        # x_forward=2.0, y_left=0.5.
        target_world_from_base[:2, 3] = [-0.5, 2.0]

        def da3_w2c(world_from_base):
            world_from_opencv_camera = world_from_base @ np.linalg.inv(
                opencv_camera_from_base
            )
            return np.linalg.inv(world_from_opencv_camera)

        result = relative_w2c_to_base_se2(
            da3_w2c(anchor_world_from_base),
            da3_w2c(target_world_from_base),
            base_from_habitat_camera,
        )
        np.testing.assert_allclose(result, [2.0, 0.5, angle], atol=1e-6)

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
