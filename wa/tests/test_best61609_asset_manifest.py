"""CPU-only fixture tests: no real NAS payload, GPU, packing, or upload."""
import copy
import hashlib
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from wa.tools import best61609_asset_manifest as m


def fixture():
    audit = dict(status="PASS", episodes=4215, teacher_artifact_hashes={},
                 comparison="fixture", scope="fixture")
    manifest = dict(seed_each_episode=7, shards=8, tasks={})
    selections = []
    for task in ("stt", "dt", "at"):
        repair = sorted(key for t, key in m.REPAIR_KEYS if t == task)
        keys = repair + ["scene" + str(i) + "/1" for i in range(1405 - len(repair))]
        manifest["tasks"][task] = dict(path=m.VAL_ROOT + "/" + task.upper() + "/val/val.json.gz",
                                       sha256=m.VAL_SHA256[task], episodes=[dict(key=k) for k in keys])
        for i, key in enumerate(keys):
            identity = task + ":" + key
            row = dict(pair=dict(task=task, key=key, seed=7, takeover_step=0), branches={})
            teachers = {}
            for teacher in m.TEACHERS:
                root = m.TEACHER_ROOTS[0] + "/lane" + str(i % 8) + "/collection/" + task + "/" + key + "/" + teacher
                row["branches"][teacher] = dict(task=task, key=key, teacher=teacher, artifact_root=root)
                teachers[teacher] = {root + "/" + f: "1" * 64 for f in m.TEACHER_FILES}
            audit["teacher_artifact_hashes"][identity] = teachers
            selections.append(row)
    repairs = [dict(task=t, key=k, probe_report=m.R + "/artifacts/residual7_no_semantic_mesh_v3_" +
                    t + "_" + k.replace("/", "_") + "/report.json", probe_sha256="2" * 64)
               for t, k in sorted(m.REPAIR_KEYS)]
    plan = dict(version="initial_rgb_mesh_bbox_v1", baseline=m.BASELINE, baseline_sha256="3" * 64,
                repairs=repairs, lanes=[repairs[::2], repairs[1::2]])
    return audit, manifest, plan, selections


class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = fixture()
        cls.pins = m._collect_evidence(*cls.inputs)

    def mutated(self):
        return copy.deepcopy(self.inputs)

    def test_complete_fixed_counts_and_exact_schema_no_dino(self):
        def sizes(source):
            if source == m.CHECKPOINT:
                return m.CHECKPOINT_BYTES
            return m.EVIDENCE_BYTES - len(self.pins) + 1 if source == m.AUDIT else 1
        result = m._manifest_from_pins(self.pins, sizes)
        self.assertEqual(set(result), {"schema", "files"})
        self.assertEqual(result["schema"], "wa_best61609_asset_manifest_v1")
        self.assertEqual(len(result["files"]), 25306)
        self.assertEqual(sum(r["size"] for r in result["files"]), 6688667421)
        self.assertEqual(sum(r["package"] == "weights" for r in result["files"]), 1)
        self.assertEqual(result["files"][-1], dict(source=m.CHECKPOINT, sha256=m.CHECKPOINT_SHA256,
                                                size=4358277705, package="weights"))
        for row in result["files"]:
            self.assertEqual(set(row), {"source", "sha256", "size", "package"})
            self.assertNotIn("dinov2", row["source"])

    def test_production_entry_has_no_remapping_or_size_overrides(self):
        self.assertEqual(len(inspect.signature(m.build_manifest).parameters), 0)

    def test_audit_status_or_count_not_pass(self):
        for field, value in (("status", "PARTIAL"), ("episodes", 4214), ("episodes", True)):
            data = self.mutated(); data[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                m._collect_evidence(*data)

    def test_manifest_task_missing_duplicate_or_wrong_val(self):
        for kind in ("task", "duplicate", "path", "sha"):
            data = self.mutated(); spec = data[1]["tasks"]["stt"]
            if kind == "task": del data[1]["tasks"]["at"]
            elif kind == "duplicate": spec["episodes"][1] = spec["episodes"][0]
            elif kind == "path": spec["path"] = m.VAL_ROOT + "/STT/val/other.json.gz"
            else: spec["sha256"] = "0" * 64
            with self.subTest(kind=kind), self.assertRaises(ValueError):m._collect_evidence(*data)

    def test_selection_duplicate_missing_seed_or_noninitial(self):
        for kind in ("duplicate", "missing", "seed", "step"):
            data = self.mutated()
            if kind == "duplicate": data[3][1] = data[3][0]
            elif kind == "missing": data[3].pop()
            elif kind == "seed": data[3][0]["pair"]["seed"] = 8
            else: data[3][0]["pair"]["takeover_step"] = 1
            with self.subTest(kind=kind), self.assertRaises(ValueError):m._collect_evidence(*data)

    def test_audit_key_missing_or_teacher_missing(self):
        for kind in ("key", "teacher"):
            data = self.mutated(); key = next(iter(data[0]["teacher_artifact_hashes"]))
            if kind == "key": del data[0]["teacher_artifact_hashes"][key]
            else: del data[0]["teacher_artifact_hashes"][key]["oracle"]
            with self.subTest(kind=kind), self.assertRaises(ValueError):m._collect_evidence(*data)

    def test_teacher_extra_file_wrong_name_or_sha(self):
        for kind in ("extra", "rename", "sha"):
            data = self.mutated(); key = next(iter(data[0]["teacher_artifact_hashes"]))
            pins = data[0]["teacher_artifact_hashes"][key]["lightnav"]; source = next(iter(pins))
            if kind == "extra": pins[str(Path(source).parent / "actions.json")] = "1" * 64
            elif kind == "rename": pins[str(Path(source).parent / "rgb_0001.png")] = pins.pop(source)
            else: pins[source] = "not-a-hash"
            with self.subTest(kind=kind), self.assertRaises(ValueError):m._collect_evidence(*data)

    def test_teacher_other_job_lane_or_identity_rejected(self):
        for kind in ("job", "lane", "identity"):
            data = self.mutated(); branch = data[3][0]["branches"]["lightnav"]
            if kind == "job": branch["artifact_root"] = branch["artifact_root"].replace("job_61264", "job_1")
            elif kind == "lane": branch["artifact_root"] = branch["artifact_root"].replace("lane0", "lane8")
            else: branch["teacher"] = "oracle"
            with self.subTest(kind=kind), self.assertRaises(ValueError):m._collect_evidence(*data)

    def test_repairs_scope_lane_and_dependency_path(self):
        for kind in ("count", "duplicate", "lane", "baseline", "probe"):
            data = self.mutated(); plan = data[2]
            if kind == "count": plan["repairs"].pop()
            elif kind == "duplicate": plan["repairs"][1] = plan["repairs"][0]
            elif kind == "lane": plan["lanes"][0].pop()
            elif kind == "baseline": plan["baseline"] = "/data/nas_ray/other.jsonl"
            else: plan["repairs"][0]["probe_report"] = "/data/nas_ray/other/report.json"
            with self.subTest(kind=kind), self.assertRaises(ValueError):m._collect_evidence(*data)

    def test_total_and_checkpoint_size_mismatch(self):
        with self.assertRaisesRegex(ValueError, "evidence byte count"):
            m._manifest_from_pins(self.pins, lambda p: 1)
        def wrong_checkpoint(p):
            return m.EVIDENCE_BYTES - len(self.pins) + 1 if p == m.AUDIT else 1
        with self.assertRaisesRegex(ValueError, "checkpoint size"):
            m._manifest_from_pins(self.pins, wrong_checkpoint)

    def test_bad_size_and_pin_rejected(self):
        with self.assertRaisesRegex(ValueError, "file size"):
            m._manifest_from_pins(self.pins, lambda p: True)
        pins = dict(self.pins); pins[m.AUDIT] = "f" * 63
        with self.assertRaisesRegex(ValueError, "SHA256"):
            m._manifest_from_pins(pins, lambda p: 1)

    def test_noncanonical_paths(self):
        for value in ("relative/file", "/data/../other", "//data/x", "/data//x", "/data/./x", "/data/x\x00"):
            with self.subTest(value=value), self.assertRaises(ValueError):m._path(value)

    def test_json_rejects_duplicate_and_nonfinite(self):
        for value in ('{"a":1,"a":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.subTest(value=value), self.assertRaises(ValueError):m._json(value)

    def test_read_pin_regular_and_symlink_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / "file.json"; source.write_bytes(b'{"x":1}')
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            self.assertEqual(m._read_pinned(str(source), digest), b'{"x":1}')
            with self.assertRaisesRegex(ValueError, "anchor hash"):
                m._read_pinned(str(source), "0" * 64)
            link = root / "link"; link.symlink_to(source)
            with self.assertRaisesRegex(ValueError, "regular file"):m._checked_size(str(link))
            directory = root / "real"; directory.mkdir(); (directory / "x").write_bytes(b'x')
            linked = root / "linked"; linked.symlink_to(directory, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "real directory"):
                m._checked_size(str(linked / "x"))

    def test_builder_does_not_read_large_teacher_or_checkpoint_payload(self):
        audit, manifest, plan, selections = self.inputs
        import json
        payloads = {m.AUDIT: json.dumps(audit).encode(), m.MANIFEST: json.dumps(manifest).encode(),
                    m.PLAN: json.dumps(plan).encode(), m.SELECTIONS: b'\n'.join(json.dumps(r).encode() for r in selections)}
        def read_small(p, h):
            self.assertFalse(p.startswith(m.TEACHER_ROOTS))
            self.assertNotEqual(p, m.CHECKPOINT)
            return payloads.get(p, b'small definition')
        def sizes(p, checked=None):
            if p == m.CHECKPOINT:return m.CHECKPOINT_BYTES
            return m.EVIDENCE_BYTES - len(self.pins) + 1 if p == m.AUDIT else 1
        with patch.object(m, "_read_pinned", side_effect=read_small) as reads, patch.object(m, "_checked_size", side_effect=sizes):
            result = m.build_manifest()
        self.assertEqual(reads.call_count, 15)
        self.assertEqual(len(result["files"]), 25306)


if __name__ == "__main__":
    unittest.main()
