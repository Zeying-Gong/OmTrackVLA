import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from omtrackvla.data.end_to_end import (
    load_raw_rgb_smoke_batch,
    load_sage3d_end_to_end_smoke_batch,
    reject_forbidden_deployment_inputs,
    sage_camera_calibration,
)
from omtrackvla.data.sage3d_policy import (
    CLOCK_SPEC_ID,
    POLICY_SPEC_ID,
    TRANSFORM_SPEC_ID,
    sha256_file,
)
from omtrackvla.data.sage3d_sidecar import GENERATION_SPEC_ID
from omtrackvla.models.end_to_end import ArchitectureV1Config
from omtrackvla.data.end_to_end_training import _realized_action


def camera_info():
    return {
        "camera": {
            "model": "pinhole",
            "axes": "ros",
            "width": 64,
            "height": 48,
            "intrinsics": {
                "k": [[40.0, 0.0, 32.0], [0.0, 40.0, 24.0], [0.0, 0.0, 1.0]]
            },
            "extrinsics_robot_to_camera": {
                "translation": [0.0, 0.0, 0.3],
                "camera_link": "base",
            },
        }
    }


class EndToEndDataTest(unittest.TestCase):
    def test_realized_action_spans_policy_states_and_keeps_raw_yaw(self):
        previous = {"robot_pos": [1.0, 2.0, 0.0], "robot_yaw": 0.5}
        current = {"robot_pos": [1.3, 2.2, 0.0], "robot_yaw": 0.7}
        action = torch.tensor(_realized_action(previous, current))
        self.assertEqual(tuple(action.shape), (3,))
        self.assertAlmostEqual(float(action[2]), 0.2, places=6)
        self.assertGreater(float(torch.linalg.vector_norm(action[:2])), 0.0)

    def test_camera_calibration_accepts_all_admitted_robot_link_names(self):
        config = ArchitectureV1Config(image_height=28, image_width=42)
        for camera_link in ("base", "base_link", "pelvis"):
            camera = camera_info()
            camera["camera"]["extrinsics_robot_to_camera"][
                "camera_link"
            ] = camera_link
            intrinsics, transform = sage_camera_calibration(camera, config)
            self.assertEqual(tuple(intrinsics.shape), (3, 3))
            self.assertEqual(tuple(transform.shape), (4, 4))

    def test_forbidden_cache_and_later_bbox_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "perception_cache"):
            reject_forbidden_deployment_inputs({"perception_cache": "cache.json"})
        with self.assertRaisesRegex(ValueError, "external_bbox"):
            reject_forbidden_deployment_inputs(
                {"rgb_history": [{"rgb": "frame.jpg", "external_bbox": [0, 0, 1, 1]}]}
            )

    def test_camera_calibration_scales_intrinsics_and_bridges_axes(self):
        config = ArchitectureV1Config(image_height=28, image_width=42)
        intrinsics, camera_from_base = sage_camera_calibration(camera_info(), config)
        self.assertAlmostEqual(float(intrinsics[0, 0]), 26.25)
        self.assertAlmostEqual(float(intrinsics[1, 1]), 40.0 * 28.0 / 48.0, places=5)
        self.assertEqual(camera_from_base[:3, :3].tolist(), [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])

    def test_raw_rgb_loader_emits_only_once_bbox_and_four_frame_window(self):
        config = ArchitectureV1Config(
            image_height=28,
            image_width=42,
            backbone_dim=48,
            policy_dim=32,
            attention_heads=4,
            adapter_layers=1,
            adapter_bottleneck=8,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for index in range(5):
                path = root / f"{index:05d}.jpg"
                Image.fromarray(
                    np.full((48, 64, 3), 20 * index, dtype=np.uint8), mode="RGB"
                ).save(path)
                paths.append(path)
            values = load_raw_rgb_smoke_batch(
                image_paths=paths,
                camera_info=camera_info(),
                initial_bbox_xyxy_norm=[0.2, 0.1, 0.7, 0.9],
                config=config,
            )
        self.assertEqual(tuple(values["initial_rgb"].shape), (1, 3, 28, 42))
        self.assertEqual(tuple(values["ego_rgb"].shape), (1, 4, 3, 28, 42))
        self.assertEqual(tuple(values["initial_bbox"].shape), (1, 4))
        self.assertNotIn("perception_cache", values)
        self.assertNotIn("target_bbox", values)
        self.assertEqual(tuple(values["target_waypoints"].shape), (1, 8, 2))

    def test_admitted_sage3d_loader_uses_train_split_and_expert_waypoints(self):
        config = ArchitectureV1Config(
            image_height=28,
            image_width=42,
            backbone_dim=48,
            policy_dim=32,
            attention_heads=4,
            adapter_layers=1,
            adapter_bottleneck=8,
        )
        episode_relative = "run_train/stt/0/go2_realsense_d435i"
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "sage3d"
            episode = root / episode_relative
            rgb = episode / "rgb"
            rgb.mkdir(parents=True)
            steps = []
            for index in range(30):
                Image.fromarray(
                    np.full((48, 64, 3), index, dtype=np.uint8), mode="RGB"
                ).save(rgb / f"{index:05d}.jpg")
                steps.append(
                    {
                        "robot_pos": [0.03 * index, 0.0, 0.0],
                        "robot_yaw": 0.0,
                        "target_pos": [2.0 + 0.03 * index, 0.1, 0.0],
                    }
                )
            camera_path = episode / "camera_info.json"
            camera_path.write_text(json.dumps(camera_info()), encoding="utf-8")
            derived_path = episode / "derived.json"
            derived_path.write_text(json.dumps({"steps": steps}), encoding="utf-8")
            (root / "index.json").write_text("{}\n", encoding="utf-8")

            sidecar_root = base / "sidecar"
            sidecar_path = sidecar_root / "episodes" / f"{episode_relative}.json"
            sidecar_path.parent.mkdir(parents=True)
            sidecar = {
                "schema_version": 1,
                "generation_spec_id": GENERATION_SPEC_ID,
                "source_path": episode_relative,
                "source_derived_sha256": sha256_file(derived_path),
                "source_camera_info_sha256": sha256_file(camera_path),
                "steps": [
                    {"visible": True, "bbox_xyxy": [12.0, 4.0, 44.0, 46.0]}
                    for _ in steps
                ],
            }
            sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
            manifest = {
                "schema_version": 1,
                "dataset_id": "sage3d_extracted",
                "generation_spec_id": GENERATION_SPEC_ID,
                "selection": "full",
                "episodes": {
                    episode_relative: {"sidecar_sha256": sha256_file(sidecar_path)}
                },
            }
            sidecar_manifest_path = sidecar_root / "manifest.json"
            sidecar_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (sidecar_root / "admission.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "manifest_sha256": sha256_file(sidecar_manifest_path),
                    }
                ),
                encoding="utf-8",
            )

            split = {
                "schema_version": 1,
                "datasets": {
                    "sage3d_extracted": {
                        "split_unit": "run",
                        "splits": {"train": ["run_train"], "test_locked": ["run_test"]},
                    }
                },
            }
            canonical = json.dumps(split, sort_keys=True, separators=(",", ":")).encode()
            split["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
            split_path = base / "phase1.json"
            split_path.write_text(json.dumps(split), encoding="utf-8")
            policy_path = base / "policy_admission.json"
            policy_path.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "policy_spec_id": POLICY_SPEC_ID,
                        "transform_spec_id": TRANSFORM_SPEC_ID,
                        "clock_spec_id": CLOCK_SPEC_ID,
                        "selection": {"test_locked_used": False},
                        "source_index_sha256": sha256_file(root / "index.json"),
                    }
                ),
                encoding="utf-8",
            )
            values = load_sage3d_end_to_end_smoke_batch(
                root=root,
                sidecar_root=sidecar_root,
                split_manifest=split_path,
                policy_admission=policy_path,
                episode_relative=episode_relative,
                initial_index=0,
                anchor_index=8,
                config=config,
            )
        self.assertEqual(tuple(values["ego_rgb"].shape), (1, 4, 3, 28, 42))
        self.assertEqual(tuple(values["target_waypoints"].shape), (1, 8, 2))
        self.assertEqual(values["target_waypoints"][0, 0].tolist(), [0.0, 0.0])
        self.assertAlmostEqual(float(values["target_waypoints"][0, -1, 0]), 0.63)
        self.assertAlmostEqual(float(values["uwb_xy"][0, 0]), 2.0)

    def test_admitted_sage3d_loader_rejects_test_locked_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split = {
                "schema_version": 1,
                "datasets": {
                    "sage3d_extracted": {
                        "split_unit": "run",
                        "splits": {"train": [], "test_locked": ["run_test"]},
                    }
                },
            }
            canonical = json.dumps(split, sort_keys=True, separators=(",", ":")).encode()
            split["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
            split_path = root / "phase1.json"
            split_path.write_text(json.dumps(split), encoding="utf-8")
            (root / "index.json").write_text("{}\n", encoding="utf-8")
            policy_path = root / "policy.json"
            policy_path.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "policy_spec_id": POLICY_SPEC_ID,
                        "transform_spec_id": TRANSFORM_SPEC_ID,
                        "clock_spec_id": CLOCK_SPEC_ID,
                        "selection": {"test_locked_used": False},
                        "source_index_sha256": sha256_file(root / "index.json"),
                    }
                ),
                encoding="utf-8",
            )
            sidecar = root / "sidecar"
            sidecar.mkdir()
            with self.assertRaisesRegex(ValueError, "train split"):
                load_sage3d_end_to_end_smoke_batch(
                    root=root,
                    sidecar_root=sidecar,
                    split_manifest=split_path,
                    policy_admission=policy_path,
                    episode_relative="run_test/stt/0/go2_realsense_d435i",
                    initial_index=0,
                    anchor_index=8,
                )


if __name__ == "__main__":
    unittest.main()
