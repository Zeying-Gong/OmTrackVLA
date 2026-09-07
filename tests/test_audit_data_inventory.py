import base64
import tempfile
import unittest
from pathlib import Path

from scripts.audit_data_inventory import (
    decode_image,
    elapsed_ratio,
    nested_language_keys,
    sage_is_canonically_accepted,
    sample_positions,
    stratified_samples,
    validate_output_path,
)


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class DataInventoryAuditTest(unittest.TestCase):
    def test_sample_positions_are_deterministic_and_cover_ends(self):
        self.assertEqual(sample_positions(10, 3), [0, 4, 9])
        self.assertEqual(sample_positions(3, 10), [0, 1, 2])
        self.assertEqual(sample_positions(0, 3), [])

    def test_stratified_samples_do_not_mix_strata(self):
        values = [("b", 3), ("a", 2), ("a", 1), ("b", 2), ("b", 1)]
        self.assertEqual(
            stratified_samples(values, lambda value: value[0], 2),
            [("a", 1), ("a", 2), ("b", 1), ("b", 3)],
        )

    def test_sage_acceptance_requires_all_three_signals(self):
        self.assertTrue(sage_is_canonically_accepted(True, True, "accepted"))
        self.assertFalse(sage_is_canonically_accepted(False, True, "accepted"))
        self.assertFalse(sage_is_canonically_accepted(True, False, "accepted"))
        self.assertFalse(sage_is_canonically_accepted(True, True, "rejected"))

    def test_elapsed_ratio_and_invalid_video_clock(self):
        self.assertEqual(elapsed_ratio([0.0, 100.0], [20.0, 470.0]), 4.5)
        self.assertIsNone(elapsed_ratio([0.0], [0.0]))
        self.assertIsNone(elapsed_ratio([10.0, 10.0], [0.0, 1.0]))

    def test_language_metadata_is_found_recursively(self):
        value = {"tasks": [{"sub_instruction": "do not use"}], "safe": 3}
        self.assertEqual(nested_language_keys(value), ["sub_instruction", "tasks"])

    def test_output_below_source_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            root.mkdir()
            with self.assertRaisesRegex(ValueError, "refusing to write"):
                validate_output_path(root / "audit.json", [root])
            validate_output_path(Path(temporary) / "reports/audit.json", [root])

    def test_png_is_actually_decoded(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.png"
            path.write_bytes(PNG_1X1)
            details = decode_image(path)
            self.assertEqual((details["width"], details["height"]), (1, 1))


if __name__ == "__main__":
    unittest.main()
