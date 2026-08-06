import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from oracle_modular_batch import (
    CONTROLLER_VERSION,
    completed_result,
    compose_rgbd_video_frame,
    exhausted_success_attempts,
    episode_key,
    invisible_target,
    parse_dataset_indices,
    prior_evasion_side,
    target_goal_crop,
)


class OracleModularBatchTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.episodes = self.root / "episodes"
        self.videos = self.root / "videos"
        self.episodes.mkdir()
        self.videos.mkdir()
        self.episode = SimpleNamespace(
            scene_id="hm3d/train/example/example.basis.glb",
            episode_id="7",
            info={
                "robot_position": [0.0, 0.0, 0.0],
                "main_humanoid_name": "target",
            },
        )
        self.key = episode_key(3, self.episode)
        self.result_path = self.episodes / f"{self.key}.json"
        (self.videos / f"{self.key}.mp4").write_bytes(b"video")

    def tearDown(self):
        self.temporary.cleanup()

    def write_result(self, success, attempt=1, side=-1.0):
        self.result_path.write_text(json.dumps({
            "controller_version": CONTROLLER_VERSION,
            "success_attempt": attempt,
            "summary": {"success": float(success), "evasion_side": side},
        }))

    def test_require_success_keeps_failed_result_pending(self):
        self.write_result(False)
        self.assertTrue(completed_result(
            3, self.episode, self.episodes, self.videos, True
        ))
        self.assertFalse(completed_result(
            3, self.episode, self.episodes, self.videos, True,
            require_success=True,
        ))

    def test_failed_side_is_persisted_for_retry_flip(self):
        self.write_result(False, attempt=2, side=-1.0)
        self.assertEqual(prior_evasion_side(self.result_path), -1.0)

    def test_success_attempt_limit_only_applies_to_failures(self):
        self.write_result(False, attempt=5)
        self.assertTrue(exhausted_success_attempts(self.result_path, 5))
        self.write_result(True, attempt=5)
        self.assertFalse(exhausted_success_attempts(self.result_path, 5))

    def test_parse_dataset_indices_normalizes_values(self):
        self.assertEqual(parse_dataset_indices("186, 7,186"), (7, 186))

    def test_parse_dataset_indices_rejects_invalid_values(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_dataset_indices("186,bad")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_dataset_indices("-1")

    def test_rgbd_video_frame_keeps_both_first_person_views(self):
        rgb = np.full((12, 16, 3), 127, dtype=np.uint8)
        depth = np.full((6, 8, 1), 0.2, dtype=np.float32)

        frame = compose_rgbd_video_frame(rgb, depth)

        self.assertEqual(frame.shape, (12, 32, 3))
        np.testing.assert_array_equal(frame[:, :16], rgb)
        self.assertGreater(frame[:, 16:].max(), 0)

    def test_rgbd_video_frame_marks_invalid_depth_black(self):
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        depth = np.zeros((8, 8), dtype=np.float32)

        frame = compose_rgbd_video_frame(rgb, depth)

        np.testing.assert_array_equal(frame[:, 8:], 0)

    def test_goal_crop_can_be_deferred_until_target_is_visible(self):
        observations = {
            "agent_1_articulated_agent_jaw_rgb": np.arange(
                4 * 5 * 3, dtype=np.uint8
            ).reshape(4, 5, 3),
            "agent_1_articulated_agent_jaw_panoptic": np.zeros(
                (4, 5, 1), dtype=np.int32
            ),
        }
        self.assertIsNone(target_goal_crop(observations, 42))

        observations["agent_1_articulated_agent_jaw_panoptic"][1:3, 2:5] = 42
        crop = target_goal_crop(observations, 42)
        np.testing.assert_array_equal(
            crop,
            observations["agent_1_articulated_agent_jaw_rgb"][1:3, 2:5],
        )

    def test_deferred_goal_target_is_invisible_with_coordinate_range(self):
        target = invisible_target((3.0, 4.0))

        self.assertFalse(target.visible)
        self.assertEqual(target.relative_xy, (3.0, 4.0))
        self.assertEqual(target.range_m, 5.0)


if __name__ == "__main__":
    unittest.main()
