"""CPU-only checks for the independent image-only training opt-in."""
import unittest

import torch

from wa.wm.train_input_mode import (
    audit_image_mode_counts, select_training_modes, validate_train_input_mode,
)


class TrainInputModeTests(unittest.TestCase):
    def test_default_sampled_mode_is_legacy_random_draw(self):
        torch.manual_seed(123)
        expected = torch.randint(0, 3, (128,), device='cpu')
        next_expected = torch.rand(4)
        torch.manual_seed(123)
        actual = select_training_modes(128, 'cpu', 'sampled')
        next_actual = torch.rand(4)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(next_actual, next_expected))

    def test_image_mode_is_all_zero_and_consumes_same_random_draw(self):
        torch.manual_seed(987)
        torch.randint(0, 3, (128,), device='cpu')
        next_expected = torch.rand(4)
        torch.manual_seed(987)
        actual = select_training_modes(128, 'cpu', 'image')
        next_actual = torch.rand(4)
        self.assertEqual(actual.dtype, torch.int64)
        self.assertEqual(actual.tolist(), [0] * 128)
        self.assertTrue(torch.equal(next_actual, next_expected))

    def test_opt_in_requires_fixed_failure_state_recipe(self):
        self.assertEqual(validate_train_input_mode('sampled', False), 'sampled')
        self.assertEqual(validate_train_input_mode('sampled', True), 'sampled')
        self.assertEqual(validate_train_input_mode('image', True), 'image')
        for failure_enabled in (False, None, 1, 'true'):
            with self.subTest(failure_enabled=failure_enabled), self.assertRaises(ValueError):
                validate_train_input_mode('image', failure_enabled)

    def test_unknown_modes_and_invalid_batch_sizes_fail_closed(self):
        for mode in (None, '', 'point', 'mixed', 'IMAGE', 0, False):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate_train_input_mode(mode, True)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                select_training_modes(2, 'cpu', mode)
        for size in (None, 0, -1, 1., True, '2'):
            with self.subTest(size=size), self.assertRaises(ValueError):
                select_training_modes(size, 'cpu', 'image')

    def test_exposure_count_must_match_actual_consumed_rows(self):
        report = audit_image_mode_counts(torch.tensor([16, 0, 0], dtype=torch.int64), 16)
        self.assertEqual(report, dict(image=16, point=0, mixed=0, total=16))
        for counts, total in (
            (torch.tensor([15, 0, 0]), 16),
            (torch.tensor([16, 1, 0]), 17),
            (torch.tensor([16, 0, 1]), 17),
            (torch.tensor([-1, 0, 0]), 16),
            (torch.tensor([16, 0]), 16),
            (torch.tensor([16, 0, 0], dtype=torch.float32), 16),
            (torch.tensor([16, 0, 0]), True),
        ):
            with self.subTest(counts=counts.tolist(), total=total), self.assertRaises(ValueError):
                audit_image_mode_counts(counts, total)


if __name__ == '__main__':
    unittest.main()
