"""CPU-only independent sampler tests; no cache, training, or release created."""
import copy
import unittest

import numpy as np

from wa.wm.failure_state_sampling import build_plan, simulate_exposure

PINS = {name: str(i) * 64 for i, name in enumerate(
    ("old_plan", "old_actual_exposure", "recovery_admission", "dedup_report"), 1)}


def old_positions(n=16):
    # Different positions may reference the same old teacher window.
    rows = [("base", i) for i in range(n // 2)]
    rows += [("teacher", i % 4) for i in range(n - len(rows))]
    rows.insert(3, ("teacher", 7))
    counts = [1] * len(rows)
    counts[3] = 0
    return rows, counts


def windows(sizes=(2, 2), early_counts=(1, 1)):
    rows = []
    for ep, (size, early) in enumerate(zip(sizes, early_counts)):
        for j in range(size):
            index = len(rows)
            age = (j + 1) / (early + 1) if j < early else 1.1 + j
            rows.append(dict(dataset_index=100 + index * 2, raw_row=1000 + index,
                episode=f"stt:scene/{ep}", task="stt", key=f"scene/{ep}",
                teacher="oracle", current_index=20 + j, takeover_step=10,
                current_age_s=age, valid=True, exact_duplicate=False))
    return rows


def plan(data=None, *, n=16, budget=16, **kwargs):
    old, counts = old_positions(n)
    return build_plan(old, counts, windows() if data is None else data,
        source_pins=PINS, recovery_budget=budget, expected_dropped_positions=(3,),
        **kwargs)


class FailureStateSamplingTests(unittest.TestCase):
    def test_original_old_positions_preserved_not_reconstructed_dataset(self):
        value = plan()
        old, counts = old_positions()
        a = value["arrays"]
        self.assertEqual(a["old_plan_positions"][:16].tolist(),
                         [i for i, c in enumerate(counts) if c])
        self.assertEqual(a["dataset_indices"][:16].tolist(),
                         [old[i][1] for i, c in enumerate(counts) if c])
        self.assertEqual(a["old_teacher_counts"].tolist(), [2, 2, 2, 2, 0, 0, 0, 0])
        self.assertEqual(value["metadata"]["old_dropped_positions"], [3])
        self.assertNotIn(7, a["dataset_indices"][:16][a["source_codes"][:16] == 1])
        self.assertFalse(value["metadata"]["training_released"])

    def test_real_eight_rank_sampler_loader_and_old_window_exact(self):
        value = plan()
        actual = simulate_exposure(value)
        self.assertEqual(actual["actual_total"], 32)
        self.assertEqual(actual["dropped"], 0)
        self.assertEqual(actual["old_counts"]["base"]["exposures"], 8)
        self.assertEqual(actual["old_counts"]["teacher"]["exposures"], 8)
        self.assertEqual(actual["recovery_exposures"], 16)
        self.assertEqual(actual["recovery_unique"], 4)
        self.assertTrue(actual["old_counts"]["teacher"]["exact"])
        self.assertEqual([x["loader_positions"] for x in actual["ranks"]], [4] * 8)
        self.assertEqual([x["batches"] for x in actual["ranks"]], [2] * 8)
        self.assertEqual(actual, simulate_exposure(value))

    def test_exact_budget_and_unique_once_first(self):
        value = plan(windows((2, 3), (1, 1)), budget=32, window_cap=8)
        counts = value["arrays"]["recovery_counts"]
        self.assertEqual(int(counts.sum()), 32)
        self.assertGreaterEqual(int(counts.min()), 1)
        self.assertLessEqual(int(counts.max()), 8)
        for episode in value["metadata"]["episodes"]:
            self.assertLessEqual(episode["exposures"], 1024)

    def test_sparse_dataset_raw_row_mapping_and_recovery_trace(self):
        value = plan()
        mapping = {x["dataset_index"]: x["raw_row"] for x in windows()}
        a = value["arrays"]
        for i in range(16, len(a["source_codes"])):
            self.assertEqual(a["source_codes"][i], 2)
            self.assertEqual(a["old_plan_positions"][i], -1)
            self.assertEqual(a["recovery_raw_rows"][i], mapping[int(a["dataset_indices"][i])])
        self.assertTrue((a["recovery_raw_rows"][:16] == -1).all())

    def test_duplicate_and_invalid_rows_never_exposed(self):
        data = windows((2, 2, 1), (1, 1, 1))
        data[0]["exact_duplicate"] = True
        data[4]["valid"] = False
        value = plan(data, budget=16)
        self.assertEqual(value["arrays"]["recovery_counts"][[0, 4]].tolist(), [0, 0])
        self.assertNotIn(data[0]["dataset_index"], value["arrays"]["dataset_indices"][16:])
        self.assertNotIn(data[4]["dataset_index"], value["arrays"]["dataset_indices"][16:])
        self.assertEqual(value["metadata"]["zero_exposure_episodes"], ["stt:scene/2"])
        self.assertEqual(simulate_exposure(value)["recovery_unique"], 3)

    def test_deterministic_input_order_independent_and_seed_affects_order(self):
        data = windows((2, 3, 4), (1, 2, 1))
        a = plan(data, budget=48)
        b = plan(list(reversed(data)), budget=48)
        self.assertEqual(a["plan_sha256"], b["plan_sha256"])
        np.testing.assert_array_equal(a["arrays"]["dataset_indices"], b["arrays"]["dataset_indices"])
        c = plan(data, budget=48, seed=43)
        self.assertNotEqual(a["plan_sha256"], c["plan_sha256"])
        self.assertEqual(int(c["arrays"]["recovery_counts"].sum()), 48)

    def test_waterlevel_respects_short_and_long_episode_caps(self):
        value = plan(windows((1, 2, 5), (1, 1, 2)), budget=32,
                     window_cap=4, episode_cap=20)
        detail = value["metadata"]["episodes"]
        self.assertEqual([x["exposures"] for x in detail], [4, 8, 20])
        self.assertEqual([x["capacity"] for x in detail], [4, 8, 20])
        self.assertTrue(all(x["saturated"] for x in detail))

    def test_equal_capacity_waterfill(self):
        value = plan(windows((2, 2, 2), (1, 1, 1)), budget=32)
        totals = [x["exposures"] for x in value["metadata"]["episodes"]]
        self.assertLessEqual(max(totals) - min(totals), 1)

    def test_early_late_balance_applies_to_extras_not_unique_floor(self):
        value = plan(windows((8,), (2,)), budget=32)
        detail = value["metadata"]["episodes"][0]
        self.assertEqual(detail["allocated_extra_early"], 12)
        self.assertEqual(detail["allocated_extra_late"], 12)
        self.assertEqual(detail["early_exposures"], 14)
        self.assertEqual(detail["late_exposures"], 18)
        self.assertEqual(value["metadata"]["age_definition"],
                         "observation_time[current_index] - observation_time[takeover_step]")

    def test_empty_early_and_empty_late_capacity_return(self):
        for early in (0, 4):
            value = plan(windows((4,), (early,)), budget=16)
            detail = value["metadata"]["episodes"][0]
            self.assertGreater(detail["reallocated_extras"], 0)
            self.assertEqual(detail["early_exposures"], 16 if early else 0)
            self.assertEqual(detail["late_exposures"], 0 if early else 16)

    def test_rare_early_window_hard_cap_and_late_receives_overflow(self):
        value = plan(windows((8,), (1,)), budget=32, window_cap=4)
        detail = value["metadata"]["episodes"][0]
        self.assertEqual(detail["early_exposures"], 4)
        self.assertEqual(detail["late_exposures"], 28)
        self.assertGreater(detail["reallocated_extras"], 0)
        self.assertEqual(value["arrays"]["recovery_counts"].tolist(), [4] * 8)

    def test_cycle_counts_balanced_within_each_layer(self):
        value = plan(windows((11,), (3,)), budget=48)
        counts = value["arrays"]["recovery_counts"]
        self.assertLessEqual(int(np.ptp(counts[:3])), 1)
        self.assertLessEqual(int(np.ptp(counts[3:])), 1)

    def test_early_boundary_relative_and_tolerance_explicit(self):
        data = windows((3,), (0,))
        for row, age in zip(data, (1., 1. + .5e-8, 1. + 2e-8)):
            row["current_age_s"] = age
        value = plan(data, budget=16)
        detail = value["metadata"]["episodes"][0]
        self.assertEqual(detail["early_unique"], 2)
        self.assertEqual(detail["late_unique"], 1)
        self.assertEqual(value["metadata"]["early_tolerance_s"], 1e-8)

    def test_budget_capacity_alignment_and_unique_floor_fail_closed(self):
        for data, budget, options in [
            (windows(), 17, {}),
            (windows((20,), (4,)), 16, {}),
            (windows(), 80, {}),
            (windows((9,), (4,)), 16, {"episode_cap": 8}),
            (windows((2, 2), (1, 1)), 32, {"episode_cap": 8}),
        ]:
            with self.subTest(budget=budget, options=options):
                with self.assertRaises(ValueError):
                    plan(data, budget=budget, **options)
        data = windows()
        for row in data:
            row["exact_duplicate"] = True
        with self.assertRaises(ValueError):
            plan(data)

    def test_wrong_old_omission_repeat_and_bool_counts_rejected(self):
        old, counts = old_positions()
        for actual, omissions in [
            (counts, (4,)),
            ([2 if i == 0 else x for i, x in enumerate(counts)], (3,)),
            ([bool(x) for x in counts], (3,)),
            (counts[:-1], (3,)),
            (counts, (3, 3)),
        ]:
            with self.assertRaises(ValueError):
                build_plan(old, actual, windows(), source_pins=PINS, recovery_budget=16,
                           expected_dropped_positions=omissions)
        bad = list(old); bad[0] = ("recovery", 0)
        with self.assertRaises(ValueError):
            build_plan(bad, counts, windows(), source_pins=PINS, recovery_budget=16,
                       expected_dropped_positions=(3,))

    def test_pin_and_noninteger_config_rejected(self):
        for pins in ({}, PINS | {"unknown": "f" * 64},
                     PINS | {"dedup_report": "not-a-hash"}):
            old, counts = old_positions()
            with self.assertRaises(ValueError):
                build_plan(old, counts, windows(), source_pins=pins, recovery_budget=16,
                           expected_dropped_positions=(3,))
        for key, value in (("seed", True), ("window_cap", 0), ("episode_cap", 1.5),
                           ("window_cap", 17), ("episode_cap", 1025),
                           ("world_size", 0), ("batch_size", True)):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    plan(**{key: value})

    def test_recovery_bad_types_identity_and_future_age_rejected(self):
        for field, value in (("dataset_index", True), ("raw_row", -1),
            ("episode", "wrong"), ("task", "test"), ("teacher", "student"),
            ("valid", 1), ("exact_duplicate", "false"), ("current_age_s", float("nan")),
            ("current_age_s", True), ("current_age_s", -1.), ("current_age_s", 0.),
            ("takeover_step", 20), ("takeover_step", 40)):
            data = windows(); data[0][field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    plan(data)
        for field in ("dataset_index", "raw_row", "current_index"):
            data = windows(); data[1][field] = data[0][field]
            with self.assertRaises(ValueError):
                plan(data)
        data = windows(); data[1]["current_age_s"] = data[0]["current_age_s"]
        with self.assertRaises(ValueError):
            plan(data)

    def test_source_or_metadata_tampering_fails_before_simulation(self):
        value = plan()
        with self.assertRaises(ValueError):
            value["arrays"]["dataset_indices"][0] = 100
        tampered = copy.deepcopy(value)
        tampered["arrays"]["dataset_indices"][0] = 100
        with self.assertRaises(ValueError):
            simulate_exposure(tampered)
        tampered = copy.deepcopy(value)
        tampered["metadata"]["budget"] += 16
        with self.assertRaises(ValueError):
            simulate_exposure(tampered)
        with self.assertRaises(ValueError):
            simulate_exposure(value, seed=43)
        with self.assertRaises(ValueError):
            simulate_exposure(value, world_size=4)
        with self.assertRaises(ValueError):
            simulate_exposure(value, epoch=0)

    def test_inputs_not_mutated(self):
        data = windows()
        old, counts = old_positions()
        before = copy.deepcopy((old, counts, data, PINS))
        build_plan(old, counts, data, source_pins=PINS, recovery_budget=16,
                   expected_dropped_positions=(3,))
        self.assertEqual((old, counts, data, PINS), before)

    def test_production_default_omission_is_not_silently_ignored(self):
        old, counts = old_positions()
        with self.assertRaises(ValueError):
            build_plan(old, counts, windows(), source_pins=PINS, recovery_budget=16)


if __name__ == "__main__":
    unittest.main()
