import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from scripts.audit_da3_multiscene import (
    DEFAULT_POLICY,
    arguments,
    load_json,
    select_split_clips,
    validate_policy,
)


MODEL_REVISION = "e08cab65ca0ec38e7826075418411ab90cab4da3"


class DA3MultiscenePolicyTest(unittest.TestCase):
    def arguments(self, *extra: str):
        return arguments(
            [
                "--model",
                "unused-model-path",
                "--model-revision",
                MODEL_REVISION,
                "--output-dir",
                "unused-output-path",
                *extra,
            ]
        )

    def test_default_development_arguments_match_policy(self):
        policy = load_json(DEFAULT_POLICY)
        self.assertEqual(
            validate_policy(self.arguments(), policy),
            "da3-small-intern-multiscene-v1",
        )

    def test_locked_split_requires_admission_mode(self):
        policy = load_json(DEFAULT_POLICY)
        with self.assertRaisesRegex(ValueError, "frozen policy purpose"):
            validate_policy(
                self.arguments("--evaluation-split", "test_locked"), policy
            )
        self.assertEqual(
            validate_policy(
                self.arguments(
                    "--evaluation-split", "test_locked", "--write-admission"
                ),
                policy,
            ),
            "da3-small-intern-multiscene-v1",
        )

    def test_parameter_drift_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "process_res"):
            validate_policy(
                self.arguments("--process-res", "518"), load_json(DEFAULT_POLICY)
            )


class DA3MultisceneSelectionTest(unittest.TestCase):
    def test_keyword_policy_arguments_select_a_moving_clip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unit = "hm3d_d435i/scene-a"
            scene = root / unit
            data = scene / "data" / "chunk-000"
            rgb = scene / "videos" / "chunk-000" / "observation.images.rgb"
            data.mkdir(parents=True)
            rgb.mkdir(parents=True)
            extrinsic = np.eye(4, dtype=np.float32)
            actions = []
            for index in range(13):
                action = np.eye(4, dtype=np.float32)
                action[1, 3] = 0.03 * index
                actions.append(action.reshape(-1).tolist())
                Image.new("RGB", (32, 24), (index, 40, 80)).save(
                    rgb / f"episode_000000_{index:03d}.jpg"
                )
            pq.write_table(
                pa.table(
                    {
                        "observation.camera_extrinsic": [extrinsic.reshape(-1).tolist()] * 13,
                        "action": actions,
                    }
                ),
                data / "episode_000000.parquet",
            )
            selected = select_split_clips(
                root,
                [unit],
                "val",
                offsets=[0, 4, 8, 12],
                clips_per_group=1,
                maximum_units_per_group=1,
                maximum_episodes_per_unit=1,
                maximum_starts_per_episode=1,
                minimum_motion_m=0.20,
                seed=7,
            )
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["source_split"], "val")
            self.assertEqual(selected[0]["indices"], [0, 4, 8, 12])


if __name__ == "__main__":
    unittest.main()
