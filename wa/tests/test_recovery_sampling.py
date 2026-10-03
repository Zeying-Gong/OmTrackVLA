"""Pure CPU tests for the standalone recovery exposure planner."""
import unittest
import numpy as np
from wa.wm.recovery_sampling import build_plan, episode_budgets, simulate_ddp


def windows(sizes, early_counts):
    result = []
    for ep, (size, early) in enumerate(zip(sizes, early_counts)):
        for j in range(size):
            k = len(result)
            result.append(dict(dataset_index=k, raw_row=k+10, episode=ep,
                               episode_uid=f"task:{ep:02}", task="stt" if ep % 2 else "dt",
                               category="mid_episode_recovery" if ep else "teacher_from_start",
                               takeover_age_s=.5 if j < early else 2.0))
    return result


class RecoverySamplingTests(unittest.TestCase):
    def test_exact_and_unique_with_caps(self):
        data = windows([2,5,9], [2,2,1])
        plan, report = build_plan(data, budget=60, max_repeat=4)
        counts = np.bincount(plan, minlength=len(data))
        self.assertEqual(len(plan), 60)
        self.assertEqual(len(np.unique(plan)), len(data))
        self.assertLessEqual(int(counts.max()), 4)
        self.assertGreaterEqual(int(counts.min()), 1)
        self.assertEqual([e["exposures"] for e in report["episodes"]], [8,20,32])

    def test_empty_late_explicit_reallocation(self):
        data = windows([3,3], [3,1])
        _, report = build_plan(data, budget=18, max_repeat=4)
        episode = report["episodes"][0]
        self.assertTrue(episode["empty_late"])
        self.assertEqual(episode["late_exposures"], 0)
        self.assertGreater(episode["layer_reallocation"], 0)

    def test_empty_early(self):
        _, report = build_plan(windows([4], [0]), budget=12, max_repeat=4)
        episode = report["episodes"][0]
        self.assertTrue(episode["empty_early"])
        self.assertEqual(episode["early_exposures"], 0)
        self.assertEqual(episode["late_exposures"], 12)

    def test_capacity_reallocation_not_unbounded_repeat(self):
        _, report = build_plan(windows([10], [1]), budget=30, max_repeat=4)
        episode = report["episodes"][0]
        self.assertEqual(episode["early_exposures"], 4)
        self.assertEqual(episode["late_exposures"], 26)
        self.assertEqual(episode["max_repeat"], 4)

    def test_deterministic_and_order_independent(self):
        data = windows([2,5,9], [2,2,1])
        first, report = build_plan(data, budget=60, max_repeat=4, seed=42)
        second, other = build_plan(list(reversed(data)), budget=60, max_repeat=4, seed=42)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(report, other)
        third, third_report = build_plan(data, budget=60, max_repeat=4, seed=43)
        self.assertFalse(np.array_equal(first, third))
        self.assertEqual(third_report["pre_ddp"]["exposures"], 60)

    def test_unique_floor_only(self):
        data = windows([3,4], [3,1])
        plan, report = build_plan(data, budget=7, max_repeat=1)
        np.testing.assert_array_equal(np.sort(plan), np.arange(7))
        self.assertEqual(report["max_repeat"], 1)

    def test_invalid_budgets_and_caps(self):
        for budget, cap in [(6,4),(29,4),(7,0),(-1,4),(7,True),(7,1.5)]:
            with self.subTest(budget=budget, cap=cap):
                with self.assertRaises(ValueError):
                    build_plan(windows([3,4],[2,1]), budget=budget, max_repeat=cap)
        with self.assertRaises(ValueError):
            episode_budgets([2,0], 2, 4, np.random.default_rng(42))

    def test_invalid_metadata(self):
        for field, value in [("takeover_age_s", -1.), ("takeover_age_s", float("nan")),
                             ("category", "failed_teacher")]:
            data = windows([3], [1]); data[0][field] = value
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    build_plan(data, 6, 4)
        data = windows([3], [1]); data[1]["raw_row"] = data[0]["raw_row"]
        with self.assertRaises(ValueError):
            build_plan(data, 6, 4)
        data = windows([3], [1]); data[1]["dataset_index"] = 20
        with self.assertRaises(ValueError):
            build_plan(data, 6, 4)

    def test_exact_ddp_semantics_and_coverage(self):
        data = windows([3,4], [2,1])
        plan, _ = build_plan(data, budget=28, max_repeat=4)
        report = simulate_ddp(plan, data, base_windows=103, world_size=8, batch_size=2)
        self.assertEqual(report["combined_pre_ddp"], 131)
        self.assertEqual(report["combined_actual"], 128)
        self.assertEqual(report["dropped_total"], 3)
        self.assertEqual(report["unique_covered"], 7)
        self.assertEqual(report["base_actual"] + report["exposures"], 128)
        self.assertEqual(report, simulate_ddp(plan, data, base_windows=103, world_size=8, batch_size=2))

    def test_within_layer_balanced_cycles(self):
        data = windows([17], [4])
        plan, _ = build_plan(data, budget=101, max_repeat=8)
        counts = np.bincount(plan, minlength=17)
        self.assertLessEqual(int(np.ptp(counts[:4])), 1)
        self.assertLessEqual(int(np.ptp(counts[4:])), 1)


if __name__ == "__main__":
    unittest.main()
