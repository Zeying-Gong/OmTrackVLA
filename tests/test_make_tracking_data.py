import json
import tempfile
import unittest
from pathlib import Path

from make_tracking_data import should_keep_episode


class TestShouldKeepEpisode(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_status(self, **status):
        (self.run_dir / "episode.json").write_text(json.dumps(status))

    def test_only_success_keeps_successful_episode(self):
        self.write_status(success=1.0, finish=True, status="Normal")

        self.assertTrue(should_keep_episode(self.run_dir, "episode", only_success=True))

    def test_only_success_rejects_finished_unsuccessful_episode(self):
        self.write_status(success=0.0, finish=True, status="Normal")

        self.assertFalse(should_keep_episode(self.run_dir, "episode", only_success=True))

    def test_only_success_requires_numeric_or_boolean_success(self):
        self.write_status(success="success", finish=True, status="Success")

        self.assertFalse(should_keep_episode(self.run_dir, "episode", only_success=True))

    def test_only_success_rejects_missing_status(self):
        self.assertFalse(should_keep_episode(self.run_dir, "episode", only_success=True))

    def test_filter_disabled_keeps_episode_without_status(self):
        self.assertTrue(should_keep_episode(self.run_dir, "episode", only_success=False))


if __name__ == "__main__":
    unittest.main()
