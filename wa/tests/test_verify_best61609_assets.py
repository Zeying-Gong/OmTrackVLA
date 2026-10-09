"""Small, asset-byte-free negative tests for the fixed WA61609 asset verifier."""

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPRO = Path(__file__).resolve().parents[1] / "repro"
VERIFIER = REPRO / "verify_best61609_assets.py"
MANIFEST = REPRO / "BEST61609_ASSET_HASHES_20261009.json"
spec = importlib.util.spec_from_file_location("verify_best61609_assets", VERIFIER)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class VerifyBest61609AssetsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(MANIFEST.read_text())

    def reject(self, change, message):
        report = copy.deepcopy(self.report)
        change(report)
        with self.assertRaisesRegex(ValueError, message):
            mod.validate_manifest(report)

    def test_fixed_manifest_has_expected_shape(self):
        self.assertEqual(len(mod.validate_manifest(self.report)), 827)

    def test_empty_manifest_rejected(self):
        self.reject(lambda r: r.__setitem__("files", []), "file count")

    def test_episode_count_rejected(self):
        self.reject(lambda r: r.__setitem__("episodes", 0), "episode count")

    def test_group_byte_count_rejected(self):
        self.reject(lambda r: r["bytes_by_group"].__setitem__("mp3d_habitat", 0), "counts or bytes")

    def test_row_byte_count_rejected(self):
        self.reject(lambda r: r["files"][0].__setitem__("bytes", 1), "disagree")

    def test_optional_role_substitution_rejected(self):
        self.reject(lambda r: r["files"][0].__setitem__("role", "use_unconfirmed"), "group/role")

    def test_duplicate_humanoid_role_rejected(self):
        def change(report):
            row = next(r for r in report["files"] if r["group"] == "habitat_humanoids_100" and r["role"] == "render_mesh")
            row["role"] = "source_mesh_not_proven_runtime_required"
        self.reject(change, "five distinct")

    def test_restore_path_traversal_rejected(self):
        self.reject(lambda r: r["files"][0].__setitem__("restore_path", "data/scene_datasets/../../secret"), "unsafe or duplicate")

    def test_scene_id_traversal_rejected(self):
        self.reject(lambda r: r["scene_ids"].__setitem__(0, "../secret.glb"), "unsafe scene")

    def test_scene_id_wrong_prefix_rejected(self):
        self.reject(lambda r: r["scene_ids"].__setitem__(0, "robot/val/x/x.basis.glb"), "unexpected scene prefix")

    def test_humanoid_name_traversal_rejected(self):
        self.reject(lambda r: r["humanoid_names"].__setitem__(0, "../../secret"), "unsafe humanoid")

    def test_checksum_pin_rejects_modified_manifest_before_asset_read(self):
        with tempfile.TemporaryDirectory() as work:
            work = Path(work)
            modified = copy.deepcopy(self.report)
            modified["files"] = []
            path = work / "modified.json"
            path.write_text(json.dumps(modified))
            result = subprocess.run(
                [sys.executable, str(VERIFIER), "--manifest", str(path), "--project-root", str(work)],
                text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("manifest SHA256 mismatch", result.stderr)

    def test_external_scene_symlink_requires_explicit_root(self):
        with tempfile.TemporaryDirectory() as work:
            work = Path(work)
            project = work / "project"
            external = work / "external"
            (project / "data").mkdir(parents=True)
            external.mkdir()
            (external / "scene.glb").write_bytes(b"x")
            (project / "data" / "scene_datasets").symlink_to(external, target_is_directory=True)
            path = project / "data" / "scene_datasets" / "scene.glb"
            with self.assertRaisesRegex(ValueError, "escapes allowed roots"):
                mod.contained_asset_path(path, project.resolve(), [], True)
            self.assertEqual(mod.contained_asset_path(path, project.resolve(), [external.resolve()], True),
                             (external / "scene.glb").resolve())

    def test_external_humanoid_robot_symlinks_never_use_scene_allowlist(self):
        with tempfile.TemporaryDirectory() as work:
            work = Path(work)
            project = work / "project"
            external = work / "external"
            (project / "data").mkdir(parents=True)
            external.mkdir()
            (external / "asset.bin").write_bytes(b"x")
            for leaf in ("humanoids", "robots"):
                (project / "data" / leaf).symlink_to(external, target_is_directory=True)
                with self.assertRaisesRegex(ValueError, "escapes allowed roots"):
                    mod.contained_asset_path(project / "data" / leaf / "asset.bin",
                                             project.resolve(), [external.resolve()], False)


if __name__ == "__main__":
    unittest.main()
