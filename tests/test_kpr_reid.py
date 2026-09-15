import math
import unittest

import numpy as np

from omtrackvla.kpr_reid import KPRReIDBackend
from omtrackvla.rgb_person_perception import RGBPersonPerception, TargetAppearanceMemory


def _backend(parts=3, dimension=2):
    backend = KPRReIDBackend.__new__(KPRReIDBackend)
    backend.parts = parts
    backend.dimension = dimension
    return backend


def _embedding(rows, visibility):
    return np.concatenate(
        (np.asarray(rows, dtype=np.float32), np.asarray(visibility)[:, None]),
        axis=1,
    )


class KPRReIDTest(unittest.TestCase):
    def test_similarity_uses_only_mutually_visible_parts(self):
        backend = _backend()
        left = _embedding(((1, 0), (0, 1), (1, 0)), (1, 1, 0))
        right = _embedding(((1, 0), (-1, 0), (0, 1)), (1, 0, 1))
        self.assertEqual(backend.similarity01(left, right), 1.0)

    def test_similarity_matches_official_normalized_euclidean_metric(self):
        backend = _backend(parts=1)
        left = _embedding(((1, 0),), (1,))
        right = _embedding(((0, 1),), (1,))
        self.assertTrue(
            math.isclose(
                backend.similarity01(left, right),
                1.0 - math.sqrt(2.0) / 2.0,
                rel_tol=1e-6,
            )
        )
        self.assertEqual(
            backend.similarity01(_embedding(((1, 0),), (0,)), right), 0.0
        )

    def test_blend_retains_previously_visible_occluded_parts(self):
        backend = _backend(parts=2)
        previous = _embedding(((1, 0), (0, 1)), (1, 1))
        current = _embedding(((0, 1), (1, 0)), (1, 0))
        blended = backend.blend(previous, current, 0.5)
        np.testing.assert_allclose(
            blended[0, :2],
            np.array((1, 1), dtype=np.float32) / math.sqrt(2.0),
        )
        np.testing.assert_allclose(blended[1], previous[1])

    def test_appearance_memory_accepts_part_structured_embeddings(self):
        backend = _backend(parts=2)
        anchor = _embedding(((1, 0), (0, 1)), (1, 1))
        memory = TargetAppearanceMemory(
            anchor,
            normalize_embedding=backend.normalize,
            similarity01=backend.similarity01,
        )
        self.assertEqual(memory.anchor_embedding.shape, (2, 3))
        self.assertEqual(memory.scores(anchor)["identity"], 1.0)

    def test_reacquisition_requires_a_consistent_multiframe_tracklet(self):
        backend = _backend(parts=1)
        tracker = RGBPersonPerception.__new__(RGBPersonPerception)
        tracker._reid_backend = backend
        tracker._missed_steps = 4
        tracker._pending_reacquisition = None
        tracker.reacquisition_confirm_frames = 3
        tracker.reacquisition_consistency_iou = 0.10
        tracker.reacquisition_consistency_reid = 0.50
        embedding = _embedding(((1, 0),), (1,))
        self.assertFalse(
            tracker._confirm_reacquisition((0, 0, 20, 40), embedding)
        )
        self.assertFalse(
            tracker._confirm_reacquisition((2, 0, 22, 40), embedding)
        )
        self.assertTrue(
            tracker._confirm_reacquisition((4, 0, 24, 40), embedding)
        )
        self.assertIsNone(tracker._pending_reacquisition)

    def test_kpr_memory_update_uses_backend_scaled_operating_point(self):
        backend = _backend(parts=1)
        anchor = _embedding(((1, 0),), (1,))
        current = _embedding(((0.8, 0.6),), (1,))
        tracker = RGBPersonPerception.__new__(RGBPersonPerception)
        tracker._reid_backend = backend
        tracker._goal_embedding = anchor
        tracker._track_embedding = anchor
        tracker._track_hist = None
        tracker._last_candidate_features = []
        tracker._appearance_memory = TargetAppearanceMemory(
            anchor,
            normalize_embedding=backend.normalize,
            similarity01=backend.similarity01,
        )
        tracker.memory_update_detector_threshold = 0.95
        tracker.memory_update_identity_threshold = 0.55
        tracker.memory_update_anchor_threshold = 0.55
        tracker.memory_update_association_threshold = 0.55
        tracker.memory_update_margin = 0.10
        tracker.memory_update_min_confirmed_steps = 3
        tracker.last_identity_margin = 0.20
        tracker._confirmed_track_steps = 3

        similarity = backend.similarity01(anchor, current)
        self.assertGreater(similarity, 0.55)
        self.assertTrue(
            tracker._update_appearance_memory(
                np.array((0, 0, 20, 40), dtype=np.float32),
                0.99,
                None,
                current,
                0.75,
                similarity,
                similarity,
                np.array((1, 0, 21, 40), dtype=np.float32),
                0,
            )
        )
        self.assertEqual(len(tracker._appearance_memory.positive_embeddings), 1)


if __name__ == "__main__":
    unittest.main()
