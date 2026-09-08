import unittest

from omtrackvla.evaluation.gate import evaluate_thresholds


class Phase1GateTest(unittest.TestCase):
    def test_gate_pass_and_missing_metric_fail(self):
        config = {
            "gate_id": "test",
            "thresholds": {
                "benchmarks.B1-ID.accuracy": {"min": 0.5},
                "benchmarks.B1-GEO.error": {"max": 1.0},
            },
        }
        metrics = {"benchmarks": {"B1-ID": {"accuracy": 0.8}, "B1-GEO": {"error": 0.4}}}
        self.assertTrue(evaluate_thresholds(metrics, config)["passed"])
        del metrics["benchmarks"]["B1-GEO"]["error"]
        result = evaluate_thresholds(metrics, config)
        self.assertFalse(result["passed"])
        self.assertIsNone(result["checks"][1]["value"])


if __name__ == "__main__":
    unittest.main()
