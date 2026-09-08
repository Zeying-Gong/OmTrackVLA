import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from omtrackvla.data.sage3d_sidecar import (
    GENERATION_SPEC_ID,
    depth_visibility_evidence,
    episode_sidecar_path,
    load_admitted_sidecar,
    project_target_bbox,
    sha256_file,
)


def camera(width=640, height=480, translation=(0.0, 0.0, 0.3)):
    return {
        "camera": {
            "model": "pinhole",
            "axes": "ros",
            "width": width,
            "height": height,
            "intrinsics": {
                "fx": 386.0,
                "fy": 386.0,
                "cx": width / 2,
                "cy": height / 2,
            },
            "extrinsics_robot_to_camera": {"translation": list(translation)},
        }
    }


def step(target=(2.0, 0.0, 0.0), yaw=0.0):
    return {
        "robot_pos": [0.0, 0.0, 0.0],
        "robot_yaw": yaw,
        "target_pos": list(target),
    }


class Sage3DSidecarTest(unittest.TestCase):
    def test_robot_right_projects_to_image_right(self):
        result = project_target_bbox(step((2.0, -0.6, 0.0)), camera())
        self.assertEqual(result["projection_reason"], "projected")
        x0, _, x1, _ = result["bbox_xyxy"]
        self.assertGreater((x0 + x1) / 2.0, 320.0)

    def test_robot_left_projects_to_image_left(self):
        result = project_target_bbox(step((2.0, 0.6, 0.0)), camera())
        x0, _, x1, _ = result["bbox_xyxy"]
        self.assertLess((x0 + x1) / 2.0, 320.0)

    def test_camera_forward_translation_is_applied(self):
        no_offset = project_target_bbox(step(), camera(translation=(0.0, 0.0, 0.3)))
        forward_offset = project_target_bbox(step(), camera(translation=(0.2, 0.0, 0.3)))
        self.assertAlmostEqual(no_offset["expected_depth_m"], 2.0)
        self.assertAlmostEqual(forward_offset["expected_depth_m"], 1.8)
        no_width = no_offset["bbox_xyxy"][2] - no_offset["bbox_xyxy"][0]
        offset_width = forward_offset["bbox_xyxy"][2] - forward_offset["bbox_xyxy"][0]
        self.assertGreater(offset_width, no_width)

    def test_non_intersecting_projection_is_out_of_view(self):
        result = project_target_bbox(step((2.0, -10.0, 0.0)), camera())
        self.assertIsNone(result["bbox_xyxy"])
        self.assertEqual(result["projection_reason"], "out_of_view")

    def test_target_behind_camera_is_rejected(self):
        result = project_target_bbox(step((-1.0, 0.0, 0.0)), camera())
        self.assertIsNone(result["bbox_xyxy"])
        self.assertEqual(result["projection_reason"], "behind_camera")

    def test_depth_support_admits_visible_target(self):
        depth_mm = np.full((100, 100), 2000, dtype=np.uint16)
        evidence = depth_visibility_evidence(depth_mm, [20, 10, 80, 90], 2.0)
        self.assertTrue(evidence["visible"])
        self.assertEqual(evidence["visibility_reason"], "visible")
        self.assertAlmostEqual(evidence["support_fraction"], 1.0)

    def test_nearer_surface_marks_target_occluded(self):
        depth_mm = np.full((100, 100), 800, dtype=np.uint16)
        evidence = depth_visibility_evidence(depth_mm, [20, 10, 80, 90], 2.0)
        self.assertFalse(evidence["visible"])
        self.assertEqual(evidence["visibility_reason"], "occluded_by_nearer_surface")

    def test_wrong_depth_marks_target_inconsistent(self):
        depth_mm = np.full((100, 100), 4000, dtype=np.uint16)
        evidence = depth_visibility_evidence(depth_mm, [20, 10, 80, 90], 2.0)
        self.assertFalse(evidence["visible"])
        self.assertEqual(evidence["visibility_reason"], "depth_inconsistent")

    def test_sidecar_path_mirrors_episode_hierarchy(self):
        root = Path("sidecar")
        self.assertEqual(
            episode_sidecar_path(root, "run/stt/2/camera"),
            root / "episodes/run/stt/2/camera.json",
        )
        with self.assertRaises(ValueError):
            episode_sidecar_path(root, "../escape")

    def test_loader_requires_matching_passed_admission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {
                "schema_version": 1,
                "dataset_id": "sage3d_extracted",
                "generation_spec_id": GENERATION_SPEC_ID,
                "selection": "full",
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (root / "admission.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "manifest_sha256": sha256_file(manifest_path),
                    }
                ),
                encoding="utf-8",
            )
            loaded, admission = load_admitted_sidecar(root)
            self.assertEqual(loaded["generation_spec_id"], GENERATION_SPEC_ID)
            self.assertEqual(admission["status"], "passed")
            manifest_path.write_text(json.dumps({**manifest, "selection": "subset"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_admitted_sidecar(root)


if __name__ == "__main__":
    unittest.main()
