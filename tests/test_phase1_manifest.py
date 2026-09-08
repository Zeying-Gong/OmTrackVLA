import json
import tempfile
import unittest
from pathlib import Path

from omtrackvla.data.phase1_manifest import build_phase1_manifest, write_phase1_manifest


class Phase1ManifestTest(unittest.TestCase):
    def _roots(self, base: Path):
        intern = base / "intern"
        sage = base / "sage"
        tpt = base / "tpt"
        for number in range(20):
            scene = intern / f"group-{number % 2}" / f"scene-{number:03d}" / "meta"
            scene.mkdir(parents=True)
            (scene / "episodes_stats.jsonl").write_text("{}\n", encoding="utf-8")
        sage.mkdir()
        (sage / "index.json").write_text(
            json.dumps({"eps": [{"run": f"run-{number:03d}"} for number in range(20)]}),
            encoding="utf-8",
        )
        for number in range(20):
            sequence = tpt / f"{number:04d}"
            sequence.mkdir(parents=True)
            (sequence / "frames.parquet").write_bytes(b"manifest-only")
        return {
            "intern_data_n1": intern,
            "sage3d_extracted": sage,
            "tpt_bench_clean_v2": tpt,
        }

    def test_manifest_is_deterministic_and_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = self._roots(Path(directory))
            left = build_phase1_manifest(roots, seed=7)
            right = build_phase1_manifest(roots, seed=7)
            self.assertEqual(left, right)
            self.assertFalse(left["media_scanned"])
            for entry in left["datasets"].values():
                all_units = []
                for units in entry["splits"].values():
                    all_units.extend(units)
                self.assertEqual(len(all_units), len(set(all_units)))
                self.assertEqual(len(all_units), entry["unit_count"])

    def test_writer_refuses_source_root(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = self._roots(Path(directory))
            manifest = build_phase1_manifest(roots)
            with self.assertRaisesRegex(ValueError, "inside source data"):
                write_phase1_manifest(roots["intern_data_n1"] / "manifest.json", manifest, roots.values())


if __name__ == "__main__":
    unittest.main()
