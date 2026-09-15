"""Independent CPU checks for observation/source/prefix comparison boundaries."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collection_contract import compare_observation
from semantic_labels import array_sha256


class ObservationComparisonTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((3, 4, 3), dtype=np.uint8)
        self.camera = np.eye(4)
        after = self.camera.copy()
        after[0, 3] = 1.
        self.source = {"steps": [{"evaluation_only_after_action": {
            "gt_distance_m": 2., "gt_visible": True,
            "render_audit": {
                "camera_transform_before_action": self.camera.tolist(),
                "camera_transform_after_forced_render": after.tolist(),
            },
        }}]}
        self.after = after
        image_hash = array_sha256(self.rgb)
        self.prefixes = [{"sample": {"path": "frozen_sample.json"},
                          "initial_rgb": {"rgb_array_sha256": image_hash},
                          "prefix_rgb": [{"path": "rgb_000000.png", "world_time_s": 0.,
                                          "rgb_array_sha256": image_hash},
                                         {"path": "rgb_000001.png", "world_time_s": .1,
                                          "rgb_array_sha256": image_hash}]}]

    def compare(self, step=1, **changes):
        arguments = dict(source=self.source, step=step, rgb=self.rgb, camera=self.after,
                         distance=2., visible=True, world_time=.1,
                         prefix_samples=self.prefixes)
        arguments.update(changes)
        return compare_observation(**arguments)

    def test_reset_compares_before_action_camera_and_initial_rgb(self):
        checks = self.compare(0, camera=self.camera, world_time=0.)
        self.assertTrue(all(check["passed"] for check in checks))
        self.assertEqual({check["kind"] for check in checks},
                         {"source_camera", "stored_prefix_rgb_and_worldtime", "stored_initial_rgb"})

    def test_terminal_compares_last_post_action_metrics(self):
        checks = self.compare()
        self.assertTrue(all(check["passed"] for check in checks))
        self.assertEqual({check["kind"] for check in checks},
                         {"source_camera", "source_distance", "source_visible",
                          "stored_prefix_rgb_and_worldtime"})

    def test_pixel_mismatch_fails_even_when_world_time_matches(self):
        changed = self.rgb.copy()
        changed[2, 3, 1] = 1
        check = next(c for c in self.compare(rgb=changed)
                     if c["kind"] == "stored_prefix_rgb_and_worldtime")
        self.assertFalse(check["passed"])
        self.assertFalse(check["rgb_equal"])
        self.assertEqual(check["world_time_error_s"], 0.)

    def test_clock_mismatch_fails_even_when_pixels_match(self):
        check = next(c for c in self.compare(world_time=.11)
                     if c["kind"] == "stored_prefix_rgb_and_worldtime")
        self.assertFalse(check["passed"])
        self.assertTrue(check["rgb_equal"])

    def test_initial_rgb_has_independent_exact_check(self):
        prefixes = copy.deepcopy(self.prefixes)
        prefixes[0]["initial_rgb"]["rgb_array_sha256"] = "0" * 64
        checks = self.compare(0, camera=self.camera, world_time=0., prefix_samples=prefixes)
        self.assertFalse(next(c for c in checks if c["kind"] == "stored_initial_rgb")["passed"])
        self.assertTrue(next(c for c in checks if c["kind"] == "stored_prefix_rgb_and_worldtime")["passed"])

    def test_source_metric_and_camera_mismatches_are_independent(self):
        checks = self.compare(camera=self.camera, distance=3., visible=False)
        by_kind = {c["kind"]: c["passed"] for c in checks}
        self.assertEqual(by_kind, {"source_camera": False, "source_distance": False,
                                   "source_visible": False, "stored_prefix_rgb_and_worldtime": True})

    def test_nonfinite_actual_metrics_fail_before_comparison(self):
        for changes in ({"distance": float("nan")}, {"world_time": float("inf")},
                        {"camera": np.full((4, 4), float("nan"))}, {"visible": 1}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self.compare(**changes)


if __name__ == "__main__":
    unittest.main()
