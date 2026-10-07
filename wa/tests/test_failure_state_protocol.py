"""CPU contract tests; no Habitat, GPU, rollout or source-result mutation."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wa.wm import failure_state_protocol as p
from wa.wm.full_mixed_contract import summarize
from wa.wm.initial_bbox_repair import KEYS, VERSION
from wa.tools.build_failure_state_plan import write_exclusive


def branch(teacher, rate=.8, success=1.0):
    return dict(experiment=p.EXPERIMENT, teacher=teacher, complete=True,
        replay_verified=True, transport_fallback=False, task="stt", key="Scene01/9",
        takeover_step=12, seed=7, protocol_sha256=p.PROTOCOL_SHA,
        initial_rgb_sha256="a"*64, takeover_state_sha256="b"*64, prefix_sha256="c"*64,
        result=dict(success=success, collision=0.0, policy_init_valid=True,
                    following_rate=rate))


def fixtures():
    manifest = dict(seed_each_episode=7, shards=8, reference_steps={}, tasks={})
    rows = []
    for task, failures in (("stt", 126), ("dt", 227), ("at", 198)):
        repair_keys = [key for t, key in sorted(KEYS) if t == task]
        keys = repair_keys+[f"Scene{i:04d}/1" for i in range(1405-len(repair_keys))]
        definitions = []
        for i, key in enumerate(keys):
            definitions.append(dict(index=i, key=key, scene_id=key.split("/")[0]+".glb",
                episode_id=key.split("/")[1], instruction="teacher-only", shard=i % 8))
            success = float(i >= failures)
            row = dict(task=task, key=key, mode="mixed", noise_mode="zero",
                controller="learned_yaw_guard_v1", policy_init_valid=True,
                checkpoint_sha256=p.CHECKPOINT_SHA, checkpoint_step=59716,
                semantic_protocol="mp3d_semantic_ply_v1", success=success, collision=0.,
                following_rate=success, following_step=int(success), total_step=1,
                finish=True, status="Normal", artifact_root="/frozen/shard_00")
            if (task, key) in KEYS:
                row.update(initialization_repair=VERSION,
                           initialization_repair_plan_sha256=p.REPAIR_SHA)
            rows.append(row)
        manifest["tasks"][task] = dict(episodes=definitions)
    summary = summarize(rows, manifest, checkpoint_sha=p.CHECKPOINT_SHA, step=59716)
    summary.update(experiment="evaluation_set_adaptation_v1", new_episodes=4215,
                   reused_baseline_episodes=0)
    repair = dict(version="initial_rgb_mesh_bbox_v1", repairs=[{} for _ in range(7)])
    paths = dict(manifest="/frozen/manifest.json", repair_plan="/frozen/bbox.json",
                 summary="/frozen/summary.json", combined="/frozen/combined_episodes.jsonl")
    return manifest, summary, rows, repair, paths, {}


class CandidateTests(unittest.TestCase):
    def test_exact_candidates(self):
        for n, expected in ((1, [0]), (5, [0]), (6, [1, 0]), (16, [11, 1, 0]),
                            (31, [26, 16, 1, 0]), (61, [56, 46, 31, 1, 0]),
                            (300, [295, 285, 270, 240, 0])):
            self.assertEqual(p.candidate_steps(n), expected)

    def test_exhaustive_bounds(self):
        for n in range(1, 401):
            out = p.candidate_steps(n)
            self.assertEqual(out, sorted(set(out), reverse=True))
            self.assertLessEqual(len(out), 5)
            self.assertTrue(all(0 <= k < n for k in out))
            self.assertEqual(out[-1], 0)

    def test_invalid_length(self):
        for n in (0, -1, True, 4., "9", None, float("nan")):
            with self.subTest(n=n), self.assertRaises(ValueError):
                p.candidate_steps(n)


class SelectionTests(unittest.TestCase):
    def pair(self):
        return branch("lightnav"), branch("oracle", .9)

    def test_higher_rate_and_exact_tie(self):
        ln, oc = self.pair()
        result = p.select_recovery_teacher(ln, oc)
        self.assertEqual(result["selected_teacher"], "oracle")
        self.assertFalse(result["training_released"])
        self.assertTrue(result["verification_required"])
        oc["result"]["following_rate"] = ln["result"]["following_rate"]
        self.assertEqual(p.select_recovery_teacher(ln, oc)["selected_teacher"], "lightnav")
        self.assertEqual(p.select_recovery_teacher(ln, oc)["reason"], "equal_rate_fixed_lightnav_tie")

    def test_bool_int_float_flags(self):
        for value in (True, 1, 1.):
            ln, oc = self.pair()
            ln["result"]["success"] = value
            oc["result"]["success"] = 0.
            self.assertEqual(p.select_recovery_teacher(ln, oc)["selected_teacher"], "lightnav")

    def test_failed_high_rate_is_ineligible(self):
        ln, oc = self.pair()
        oc["result"].update(success=0., following_rate=1.)
        self.assertEqual(p.select_recovery_teacher(ln, oc)["selected_teacher"], "lightnav")
        ln["result"]["success"] = False
        result = p.select_recovery_teacher(ln, oc)
        self.assertIsNone(result["selected_teacher"])
        self.assertFalse(result["demonstration_candidate"])

    def test_fallback_not_selected_even_if_higher_rate(self):
        ln, oc = self.pair()
        oc["transport_fallback"] = True
        self.assertEqual(p.select_recovery_teacher(ln, oc)["selected_teacher"], "lightnav")
        ln["transport_fallback"] = True
        self.assertIsNone(p.select_recovery_teacher(ln, oc)["selected_teacher"])

    def test_explicit_unverified_or_incomplete_not_selected(self):
        for field in ("complete", "replay_verified"):
            ln, oc = self.pair()
            oc[field] = False
            self.assertEqual(p.select_recovery_teacher(ln, oc)["selected_teacher"], "lightnav")

    def test_pair_mismatch_every_field(self):
        for field in p.PAIR_FIELDS:
            ln, oc = self.pair()
            oc[field] = {"task": "dt", "key": "Scene02/9", "takeover_step": 11,
                         "seed": 8}.get(field, "d"*64)
            with self.subTest(field=field), self.assertRaises(ValueError):
                p.select_recovery_teacher(ln, oc)

    def test_missing_pair_evidence_every_field(self):
        for field in p.PAIR_FIELDS:
            ln, oc = self.pair()
            del ln[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                p.select_recovery_teacher(ln, oc)

    def test_missing_and_nonbool_evidence_flags(self):
        for field in ("complete", "replay_verified", "transport_fallback"):
            for value in (None, 0, 1, "false"):
                ln, oc = self.pair()
                ln[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    p.select_recovery_teacher(ln, oc)

    def test_bad_result_flags(self):
        for field in ("success", "collision", "policy_init_valid"):
            for value in (None, "1", 2, -1, float("nan"), float("inf")):
                ln, oc = self.pair()
                ln["result"][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    p.select_recovery_teacher(ln, oc)

    def test_bad_rate(self):
        for value in (None, True, "0.9", -0.001, 1.001, float("nan"), float("inf")):
            ln, oc = self.pair()
            ln["result"]["following_rate"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                p.select_recovery_teacher(ln, oc)

    def test_inconsistent_success_and_repeated_verification_rejected(self):
        for update in (dict(collision=1.), dict(policy_init_valid=False)):
            ln, oc = self.pair()
            ln["result"].update(update)
            with self.assertRaises(ValueError):
                p.select_recovery_teacher(ln, oc)
        ln, oc = self.pair()
        ln["verification_only"] = True
        with self.assertRaises(ValueError):
            p.select_recovery_teacher(ln, oc)

    def test_pairing_allows_separate_actual_tolerance_hash(self):
        ln, oc = self.pair()
        ln["actual_takeover_state_sha256"] = "d"*64
        oc["actual_takeover_state_sha256"] = "e"*64
        self.assertEqual(p.select_recovery_teacher(ln, oc)["selected_teacher"], "oracle")

    def test_selection_does_not_mutate_inputs(self):
        ln, oc = self.pair()
        before = copy.deepcopy((ln, oc))
        p.select_recovery_teacher(ln, oc)
        self.assertEqual((ln, oc), before)


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = fixtures()
        cls.plan = p.make_plan(*cls.data)

    def test_exact_all_failures_and_order(self):
        plan = p.validate_plan_structure(copy.deepcopy(self.plan))
        self.assertEqual(len(plan["entries"]), 126)
        self.assertEqual([len(lane) for lane in plan["lanes"]], [16]*6+[15]*2)
        self.assertEqual([e["manifest_index"] for e in plan["entries"]], list(range(126)))
        self.assertEqual(plan["entries"][0]["baseline_row_sha256"], p.canonical_sha(self.data[2][0]))
        self.assertNotIn("instruction", plan["entries"][0])
        self.assertNotIn("initial_pair_evidence", plan["entries"][0])
        self.assertTrue(plan["evaluation_adaptation"])
        self.assertFalse(plan["untouched_test"])

    def test_dropped_or_duplicate_entries(self):
        for edit in ("drop", "duplicate"):
            plan = copy.deepcopy(self.plan)
            if edit == "drop":
                plan["entries"].pop()
            else:
                plan["entries"][1] = copy.deepcopy(plan["entries"][0])
            with self.assertRaises(ValueError):
                p.validate_plan_structure(plan)

    def test_lane_coverage_overlap_order_and_omission(self):
        for edit in ("overlap", "drop", "reorder", "omit_lane"):
            plan = copy.deepcopy(self.plan)
            if edit == "overlap":
                plan["lanes"][1][0] = plan["lanes"][0][0]
            elif edit == "drop":
                plan["lanes"][0].pop()
            elif edit == "reorder":
                plan["lanes"][0].reverse()
            else:
                plan["lanes"].pop()
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                p.validate_plan_structure(plan)

    def test_foreign_contract_or_missing_source(self):
        for field, value in (("experiment", "evaluation_set_adaptation_v1"),
                             ("expected_count", True), ("untouched_test", True),
                             ("checkpoint", dict(p.CHECKPOINT, step=59065)),
                             ("protocol_sha256", "f"*64)):
            plan = copy.deepcopy(self.plan)
            plan[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                p.validate_plan_structure(plan)
        plan = copy.deepcopy(self.plan)
        plan["source_hashes"].pop("/frozen/combined_episodes.jsonl")
        with self.assertRaises(ValueError):
            p.validate_plan_structure(plan)

    def test_original_success_cannot_enter_failure_plan(self):
        plan = copy.deepcopy(self.plan)
        plan["entries"][0]["baseline_result"]["success"] = 1.
        with self.assertRaises(ValueError):
            p.validate_plan_structure(plan)

    def test_bad_original_model_and_original_result(self):
        manifest, summary, rows, repair, _, _ = copy.deepcopy(self.data)
        rows[0]["checkpoint_step"] = 1
        with self.assertRaises(ValueError):
            p.derive_entries(manifest, summary, rows, repair)
        rows[0]["checkpoint_step"] = 59716
        rows[0]["following_rate"] = float("nan")
        with self.assertRaises(ValueError):
            p.derive_entries(manifest, summary, rows, repair)

    def test_changed_summary_or_failure_count(self):
        manifest, summary, rows, repair, _, _ = copy.deepcopy(self.data)
        summary["metrics_percent"]["stt"]["SR"] += 1
        with self.assertRaises(ValueError):
            p.derive_entries(manifest, summary, rows, repair)
        rows[0]["success"] = 1.
        summary = summarize(rows, manifest, checkpoint_sha=p.CHECKPOINT_SHA, step=59716)
        summary.update(experiment="evaluation_set_adaptation_v1", new_episodes=4215,
                       reused_baseline_episodes=0)
        with self.assertRaises(ValueError):
            p.derive_entries(manifest, summary, rows, repair)

    def test_exclusive_output_and_plan_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"plan.json"
            digest = write_exclusive(path, self.plan)
            original = path.read_bytes()
            self.assertEqual(digest, hashlib.sha256(original).hexdigest())
            with self.assertRaises(FileExistsError):
                write_exclusive(path, {})
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaises(ValueError):
                p.load_plan(path, "a"*64)
            with patch.object(p, "build_plan", return_value=self.plan):
                self.assertEqual(p.load_plan(path, digest), self.plan)
            changed = copy.deepcopy(self.plan)
            changed["entries"][0]["baseline_result"]["following_rate"] = .2
            with patch.object(p, "build_plan", return_value=changed), self.assertRaises(ValueError):
                p.load_plan(path, digest)

    def test_symlink_output_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"plan.json"
            path.symlink_to(Path(tmp)/"not_created")
            with self.assertRaises(FileExistsError):
                write_exclusive(path, self.plan)

    def test_strict_json_and_pinned_source(self):
        for text in ('{"x":NaN}', '{"x":Infinity}', '{"x":1,"x":2}'):
            with self.assertRaises(ValueError):
                p._document(text)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"source.json"
            path.write_text("{}")
            with self.assertRaises(ValueError):
                p._read_pinned(path, "0"*64)
            with self.assertRaises(ValueError):
                p._read_pinned("relative.json", "0"*64)


if __name__ == "__main__":
    unittest.main()
