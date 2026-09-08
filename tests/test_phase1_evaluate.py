import unittest

import torch

from omtrackvla.evaluation.evaluate import _standardize_features


class Phase1EvaluateTest(unittest.TestCase):
    def test_probe_standardization_uses_training_statistics(self):
        train = torch.tensor([[1.0, 5.0], [3.0, 5.0]])
        validation = torch.tensor([[5.0, 7.0]])

        train_standardized, validation_standardized = _standardize_features(
            train, validation
        )

        torch.testing.assert_close(
            train_standardized, torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
        )
        torch.testing.assert_close(
            validation_standardized, torch.tensor([[3.0, 2.0]])
        )


if __name__ == "__main__":
    unittest.main()
