import gzip
import json
import tempfile
import unittest
from pathlib import Path

from monitor_oracle_progress import (
    EXPECTED_CONTROLLER_VERSION,
    completed_count,
    dataset_count,
    eligible_indices,
    progress_line,
)


class MonitorOracleProgressTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        dataset_path = self.root / "data/datasets/track/STT/val/val.json.gz"
        dataset_path.parent.mkdir(parents=True)
        with gzip.open(dataset_path, "wt") as handle:
            json.dump({"episodes": [{}, {}, {}, {}, {}]}, handle)
        self.output = self.root / "output"
        self.episodes = self.output / "stt/val/episodes"
        self.videos = self.output / "stt/val/videos"
        self.episodes.mkdir(parents=True)
        self.videos.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def write_result(self, index, success=True, video=True):
        path = self.episodes / f"episode_{index}.json"
        path.write_text(json.dumps({
            "controller_version": EXPECTED_CONTROLLER_VERSION,
            "task": "stt",
            "split": "val",
            "dataset_index": index,
            "summary": {"success": float(success)},
        }))
        if video:
            (self.videos / f"episode_{index}.mp4").write_bytes(b"video")

    def test_dataset_and_shard_limits_determine_total(self):
        self.assertEqual(dataset_count(self.root, "stt", "val"), 5)
        self.assertEqual(
            eligible_indices(5, 2, {1}, 2),
            {0, 2, 3},
        )

    def test_completed_count_honors_video_and_success_requirements(self):
        eligible = set(range(5))
        self.write_result(0)
        self.write_result(1, video=False)
        self.write_result(2, success=False)
        self.assertEqual(
            completed_count(self.output, "stt", "val", eligible, True, False),
            2,
        )
        self.assertEqual(
            completed_count(self.output, "stt", "val", eligible, True, True),
            1,
        )

    def test_progress_line_contains_counts_rate_and_eta(self):
        line = progress_line("stt", "val", 4, 10, 2, 60.0)
        self.assertIn("4/10 (40.00%)", line)
        self.assertIn("session=2", line)
        self.assertIn("rate=2.00 eps/min", line)
        self.assertIn("eta=03:00", line)


if __name__ == "__main__":
    unittest.main()
