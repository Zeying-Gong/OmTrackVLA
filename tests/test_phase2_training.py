import unittest

from omtrackvla.training.phase2 import _learning_rate_scale, _selected_indices


class Phase2TrainingTest(unittest.TestCase):
    def test_epoch_selection_matches_end_to_end_policy(self):
        first = _selected_indices(1000, 128, seed=20260911, epoch=0)
        second = _selected_indices(1000, 128, seed=20260911, epoch=1)
        self.assertEqual(first, list(range(first[0], first[0] + 128)))
        self.assertEqual(second, list(range(second[0], second[0] + 128)))
        self.assertNotEqual(first, second)
        self.assertEqual(
            first[0], (20260911 * 1_000_003) % (1000 - 128 + 1)
        )

    def test_epoch_selection_rejects_oversampling(self):
        with self.assertRaisesRegex(ValueError, "exceeds dataset"):
            _selected_indices(10, 11, seed=1, epoch=0)

    def test_warmup_cosine_schedule(self):
        self.assertAlmostEqual(
            _learning_rate_scale(0, warmup_steps=200, maximum_steps=36864),
            1.0 / 200.0,
        )
        self.assertAlmostEqual(
            _learning_rate_scale(199, warmup_steps=200, maximum_steps=36864), 1.0
        )
        self.assertAlmostEqual(
            _learning_rate_scale(36864, warmup_steps=200, maximum_steps=36864),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
