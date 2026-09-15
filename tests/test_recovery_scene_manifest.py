"""Unit checks for identity, split and provenance admission (stdlib only)."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_recovery_scene_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_recovery_scene_manifest", SCRIPT)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class SceneManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.parent = self.root / "phase2.ckpt"
        self.parent.write_bytes(b"preserved phase2 parent")
        self.protocol = self.root / "collection_protocol.json"
        self.protocol.write_text(json.dumps({"scope": "development_pilot", "measured_time": True}))
        self.paths = [self.add_sample(f"scene{index:05}", ("STT", "DT", "AT")[index % 3], index)
                      for index in range(6)]
        self.plan = self.root / "scene-plan.json"
        self.manifest = self.root / "manifest.json"

    def add_sample(self, scene, task, index, anchor=28):
        directory = self.root / f"candidate-{len(list(self.root.iterdir()))}"
        directory.mkdir()
        rollout = directory / "rollout.json"
        rollout.write_text(json.dumps({"split": "train", "scene_id": scene}))
        report = directory / "report.json"
        sample = directory / "sample.json"
        # Deliberately reuse the legacy sample_id across distinct scenes.
        value = {"schema_version": 2, "sample_id": f"evt_train/{task}/0/post-action-028/sequence-v2",
                 "formal_training_eligible": False, "test_locked_used": False,
                 "source_split": "train", "scene_id": scene, "task": task,
                 "dataset_index": index, "episode_id": "0", "anchor_environment_step": anchor,
                 "rollout": str(rollout), "report": str(report)}
        sample.write_text(json.dumps(value))
        report.write_text(json.dumps({"passed": True, "sample_sha256": m.sha256(sample)}))
        return sample

    def validator(self, path):
        value = json.loads(path.read_text())
        report = json.loads(Path(value["report"]).read_text())
        if report["sample_sha256"] != m.sha256(path) or report["passed"] is not True:
            raise ValueError("independent candidate validation failed")
        return {**{key: value[key] for key in ("schema_version", "sample_id", "formal_training_eligible",
                  "test_locked_used", "source_split", "scene_id", "task", "dataset_index", "episode_id",
                  "anchor_environment_step")}, "checks": {"independent_quality_passed": True},
                "artifacts": {"sample": m.artifact(path), "report": m.artifact(value["report"]),
                              "rollout": m.artifact(value["rollout"]), "media": []}}

    def freeze(self, **options):
        kwargs = dict(sample_paths=self.paths, output=self.plan, parent_checkpoint=self.parent,
                      collection_protocol=self.protocol, validator=self.validator)
        kwargs.update(options)
        return m.freeze_scene_plan(**kwargs)

    def build(self, **options):
        kwargs = dict(plan_path=self.plan, output=self.manifest, validator=self.validator)
        kwargs.update(options)
        return m.build_manifest(**kwargs)

    def rewrite_sealed(self, path, value, checksum="manifest_sha256"):
        value = copy.deepcopy(value)
        value.pop(checksum)
        path.write_text(json.dumps(m._seal(value, checksum)))

    def test_valid_pilot_preserves_candidate_restrictions_and_train_source_split(self):
        self.freeze()
        value = self.build()
        checked, training = m.verify_manifest(self.manifest, validator=self.validator, role="train", for_training=True)
        _, validation = m.verify_manifest(self.manifest, validator=self.validator, role="val")
        self.assertEqual(value, checked)
        self.assertTrue(training)
        self.assertTrue(validation)
        self.assertTrue(set(training).isdisjoint(validation))
        self.assertFalse(value["formal_training_eligible"])
        self.assertFalse(value["product_acceptance_evidence"])
        self.assertEqual(value["scope"], "development_pilot")
        self.assertEqual(value["coverage"]["global"]["tasks"], ["AT", "DT", "STT"])
        for record in value["samples"]:
            self.assertEqual(record["source_split"], "train")
            self.assertEqual(record["optimizer_input_allowed"], record["partition_role"] == "train")
            self.assertFalse(json.loads(Path(record["artifacts"]["sample"]["path"]).read_text())["formal_training_eligible"])

    def test_seed_split_deterministic_under_reordered_explicit_paths(self):
        first = self.freeze(seed=20260914)
        second = self.freeze(output=self.root / "plan-copy.json", sample_paths=reversed(self.paths), seed=20260914)
        self.assertEqual(first, second)

    def test_scene_alias_normalization(self):
        for scene in ("SCENE00742", "scene742", "scene_00742", "/dataset/STT/scene00742/scene00742.glb",
                      "D:\\dataset\\DT\\scene00742\\scene00742.basis.glb"):
            self.assertEqual(m.canonical_scene_id(scene), "scene00742")
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            m.canonical_scene_id("/scene00742/scene00200.glb")

    def test_same_scene_across_tasks_cannot_cross_partitions(self):
        self.paths.append(self.add_sample("/data/DT/scene00000/scene00000.glb", "DT", 10))
        self.paths.append(self.add_sample("D:\\data\\AT\\SCENE00000.glb", "AT", 11))
        self.freeze()
        value = self.build()
        roles = {record["partition_role"] for record in value["samples"]
                 if record["identity"]["scene_id"] == "scene00000"}
        self.assertEqual(len(roles), 1)

    def test_real_hm3d_directory_and_mesh_scene_aliases(self):
        aliases = ("data/scene_datasets/hm3d/train/00083-16tymPtM7uS/16tymPtM7uS.basis.glb",
                   "00083-16tymPtM7uS", "16tymPtM7uS", "16tymPtM7uS.glb",
                   "D:\\hm3d\\train\\00083-16tymPtM7uS\\16tymPtM7uS.basis.glb")
        self.assertEqual({m.canonical_scene_id(alias) for alias in aliases}, {"16tymptm7us"})

    def test_duplicate_identity_via_scene_alias_is_rejected(self):
        self.paths.append(self.add_sample("/aliases/STT/SCENE00000.glb", "STT", 0))
        with self.assertRaisesRegex(ValueError, "duplicate physical sample identity"):
            self.freeze()

    def test_reused_legacy_id_across_different_scenes_is_allowed(self):
        plan = self.freeze()
        self.assertEqual(len(plan["samples"]), 6)

    def test_bad_hash_from_validator_is_rejected(self):
        def wrong_hash(path):
            evidence = self.validator(path)
            evidence["artifacts"]["report"]["sha256"] = "0" * 64
            return evidence
        with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
            self.freeze(validator=wrong_hash)

    def test_independent_validator_must_pass_every_check(self):
        def failed_check(path):
            evidence = self.validator(path)
            evidence["checks"]["time_contract"] = False
            return evidence
        with self.assertRaisesRegex(ValueError, "independent"):
            self.freeze(validator=failed_check)

    def test_report_change_after_freeze_is_rejected_even_with_fresh_evidence(self):
        self.freeze()
        report_path = Path(json.loads(self.paths[0].read_text())["report"])
        report = json.loads(report_path.read_text())
        report["unrelated_but_hashed"] = "changed"
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "changed since"):
            self.build()

    def test_bound_parent_and_protocol_changes_are_rejected(self):
        self.freeze()
        self.parent.write_bytes(b"different checkpoint")
        with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
            self.build()
        self.parent.write_bytes(b"preserved phase2 parent")
        self.protocol.write_text("changed protocol")
        with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
            self.build()

    def test_existing_plan_and_manifest_are_not_overwritten(self):
        self.freeze()
        before = self.plan.read_bytes()
        with self.assertRaises(FileExistsError):
            self.freeze(seed=999)
        self.assertEqual(self.plan.read_bytes(), before)
        self.build()
        before = self.manifest.read_bytes()
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(self.manifest.read_bytes(), before)

    def test_tampered_manifest_checksum_is_rejected(self):
        self.freeze()
        value = self.build()
        value["sample_count"] = 100
        self.manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            m.verify_manifest(self.manifest, validator=self.validator)

    def test_resealed_scene_leakage_is_rejected_against_frozen_plan(self):
        self.paths.append(self.add_sample("/DT/scene00000.glb", "DT", 10))
        self.freeze()
        value = self.build()
        records = [record for record in value["samples"] if record["identity"]["scene_id"] == "scene00000"]
        records[0]["partition_role"] = "train" if records[1]["partition_role"] == "val" else "val"
        records[0]["optimizer_input_allowed"] = records[0]["partition_role"] == "train"
        self.rewrite_sealed(self.manifest, value)
        with self.assertRaisesRegex(ValueError, "scene leakage"):
            m.verify_manifest(self.manifest, validator=self.validator)

    def test_validation_paths_never_returned_as_optimizer_inputs(self):
        self.freeze()
        self.build()
        for role in (None, "val"):
            with self.assertRaisesRegex(ValueError, "explicit train role"):
                m.verify_manifest(self.manifest, validator=self.validator, role=role, for_training=True)

    def test_minimum_scene_counts_and_nonempty_partitions_enforced(self):
        with self.assertRaisesRegex(ValueError, "distinct physical scenes"):
            self.freeze(sample_paths=self.paths[:3])
        with self.assertRaisesRegex(ValueError, "at least one scene"):
            self.freeze(min_train_scenes=0)
        with self.assertRaisesRegex(ValueError, "at least one scene"):
            self.freeze(min_val_scenes=0)

    def test_missing_required_global_task_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "task coverage is incomplete"):
            self.freeze(sample_paths=[self.paths[i] for i in (0, 1, 3, 4)])

    def test_test_locked_path_rejected_before_validator_or_file_access(self):
        called = []
        def must_not_run(path):
            called.append(path)
            raise AssertionError("must not open locked data")
        with self.assertRaisesRegex(ValueError, "test_locked"):
            self.freeze(sample_paths=[self.root / "test_locked" / "sample.json"], validator=must_not_run)
        self.assertEqual(called, [])

    def test_frozen_validation_scenes_preserved_across_explicit_lineage(self):
        first = self.freeze()
        self.build()
        second = self.freeze(output=self.root / "next-plan.json", seed=999, prior_manifests=[self.manifest])
        self.assertEqual(first["scene_partitions"], second["scene_partitions"])
        self.assertEqual(second["prior_manifests"], [m.artifact(self.manifest)])


if __name__ == "__main__":
    unittest.main()
