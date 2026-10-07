import unittest
import numpy as np
from wa.tools.audit_stt_start_coverage import window_ages, group_summary, validate_row_indices

class StartCoverageTest(unittest.TestCase):
    def test_uses_actual_timestamps_not_nominal_steps(self):
        history = np.array([[0, 1], [1, 2], [2, 3]])
        age = window_ages([10, 10.03, 10.7, 12.1], history, np.array([0, 2]))
        np.testing.assert_allclose(age, [.03, 2.1])
        self.assertEqual(int((age <= 2).sum()), 1)
    def test_reject_bad_time_or_index(self):
        history = np.array([[0, 1]])
        for times in ([1, 1], [2, 1], [1, float('nan')], []):
            with self.assertRaises(ValueError): window_ages(times, history, np.array([0]))
        for index in (-1, 2):
            with self.assertRaises(ValueError): window_ages([1, 2], np.array([[0, index]]), np.array([0]))
    def test_zero_valid_is_kept_visible(self):
        history = np.array([[0, 1]])
        self.assertEqual(len(window_ages([1, 2], history, np.array([], dtype=int))), 0)
        report = group_summary([dict(key='x/1', valid=0, early_valid=0, early_candidate=4,
            early05_valid=0, early1_valid=0)])
        self.assertEqual(report['zero_early'], ['x/1'])
        self.assertEqual(report['early_candidate'], 4)
        self.assertEqual(report['early_windows'], 0)
        self.assertEqual(report['episodes'], 1)

    def test_reject_bad_valid_indices(self):
        history, episode = np.zeros((3, 4), dtype=int), np.array([0, 0, 1])
        for valid in (np.array([-1]), np.array([3]), np.array([1, 1]), np.array([2, 1]),
                      np.array([.5]), np.array([[1]])):
            with self.subTest(valid=valid), self.assertRaises(ValueError):
                validate_row_indices(history, episode, valid, 2)
        np.testing.assert_array_equal(validate_row_indices(history, episode, np.array([0, 2]), 2), [0, 1])
    def test_reject_bad_episode_or_history(self):
        history, valid = np.zeros((3, 4), dtype=int), np.array([0])
        for episode in (np.array([-1, 0, 1]), np.array([0, 1, 2]), np.array([0, 1, 0]), np.array([0., 0., 1.])):
            with self.assertRaises(ValueError): validate_row_indices(history, episode, valid, 2)
        with self.assertRaises(ValueError): validate_row_indices(history[:2], np.array([0, 0, 1]), valid, 2)

if __name__ == '__main__': unittest.main()
