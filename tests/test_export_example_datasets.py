import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.export_example_datasets import (
    contiguous_indices,
    sha256,
    verify_no_symlinks,
    write_checksums,
)


class ExportExampleDatasetsTest(unittest.TestCase):
    def test_contiguous_indices_are_bounded(self):
        self.assertEqual(contiguous_indices(20, 3, 4), [3, 4, 5, 6])
        self.assertEqual(contiguous_indices(5, 3, 10), [3, 4])
        with self.assertRaises(ValueError):
            contiguous_indices(5, 5, 1)

    def test_sha256_and_checksum_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "nested/example.bin"
            path.parent.mkdir()
            path.write_bytes(b"example")
            expected = hashlib.sha256(b"example").hexdigest()
            self.assertEqual(sha256(path), expected)
            self.assertEqual(write_checksums(root), 1)
            self.assertEqual(
                (root / "checksums.sha256").read_text(),
                f"{expected}  nested/example.bin\n",
            )

    def test_symlink_is_rejected_when_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.write_text("x")
            link = root / "link"
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest("symbolic links are unavailable")
            with self.assertRaisesRegex(ValueError, "symbolic links"):
                verify_no_symlinks(root)


if __name__ == "__main__":
    unittest.main()
