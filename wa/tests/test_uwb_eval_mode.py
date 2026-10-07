"""CPU-only mode-boundary regression tests; fixtures are not model results."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from wa.tools import build_student_review as review

from wa.wm.full_mixed_contract import summarize, validate_ready
from wa.wm.initial_bbox_repair import KEYS, VERSION
from wa.wm.student_eval_contract import (
    REPAIR_SHA, evaluation_mode, model_contract, validate_student_rows,
)
from wa.wm.student_eval_finalize import validate_shard
from wa.wm.student_eval_partition import write_partition
from wa.tools.merge_student_partitions import read_partitions


CONTRACT = dict(checkpoint_sha="a" * 64, step=59716)


def environment(mode="image"):
    return dict(
        WA_EVAL_MODE=mode,
        WA_STUDENT_EVAL="evaluation_set_adaptation_v1",
        WA_EVAL_CHECKPOINT_SHA=CONTRACT["checkpoint_sha"],
        WA_EVAL_CHECKPOINT_STEP=str(CONTRACT["step"]),
        WA_SEMANTIC_PLY_FIX="mp3d_semantic_ply_v1",
        WA_INIT_REPAIR_PLAN="/synthetic/frozen-plan.json",
        WA_INIT_REPAIR_PLAN_SHA=REPAIR_SHA,
        WA_DIAG_CONTROLLER="learned_yaw_guard_v1",
    )


def ready(mode):
    return dict(
        checkpoint_sha256=CONTRACT["checkpoint_sha"], step=CONTRACT["step"],
        mode=mode, noise_mode="zero", sampling_steps=4,
        text_used=False, world_predictor_inference=False,
    )


def fixture(mode):
    manifest = dict(tasks={}, reference_steps={})
    rows = []
    for task in ("stt", "dt", "at"):
        keys = [key for t, key in sorted(KEYS) if t == task]
        keys += ["synthetic/" + str(i) for i in range(1405 - len(keys))]
        manifest["tasks"][task] = dict(
            episodes=[dict(key=k, shard=i % 8) for i, k in enumerate(keys)])
        for i, key in enumerate(keys):
            row = dict(
                task=task, key=key, mode=mode, noise_mode="zero",
                controller="learned_yaw_guard_v1",
                checkpoint_sha256=CONTRACT["checkpoint_sha"],
                checkpoint_step=CONTRACT["step"],
                semantic_protocol="mp3d_semantic_ply_v1",
                success=int(i % 5 != 0), collision=int(i % 7 == 0),
                following_rate=0.75, following_step=15, total_step=20,
                finish=True, policy_init_valid=True,
            )
            if (task, key) in KEYS:
                row.update(initialization_repair=VERSION,
                           initialization_repair_plan_sha256=REPAIR_SHA)
            rows.append(row)
    return manifest, rows


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def partition_fixture(directory, mode):
    manifest, rows = fixture(mode)
    roots = {}
    for task in ("stt", "dt", "at"):
        root = Path(directory) / task
        root.mkdir()
        roots[task] = root
        combined = []
        for index in range(8):
            dest = root / ("shard_%02d" % index)
            dest.mkdir()
            keys = {e["key"] for e in manifest["tasks"][task]["episodes"]
                    if e["shard"] == index}
            local = [r for r in rows if r["task"] == task and r["key"] in keys]
            (dest / "episodes.jsonl").write_text(
                "".join(json.dumps(r, allow_nan=False) + "\n" for r in local))
            write_json(dest / "COMPLETE.json", dict(episodes=len(local), mode=mode))
            write_json(dest / "server_ready.json", ready(mode))
            combined.extend(dict(r, artifact_root=str(dest)) for r in local)
        write_partition(root, combined, manifest, CONTRACT, task, mode=mode)
    return roots, manifest


class ModeSelectionTests(unittest.TestCase):
    def test_default_stays_mixed_and_identity_dict_stays_identity_only(self):
        self.assertEqual(evaluation_mode({}), "mixed")
        self.assertEqual(model_contract({}), {})
        for mode in ("mixed", "image"):
            self.assertEqual(evaluation_mode(environment(mode)), mode)
            self.assertEqual(model_contract(environment(mode)), CONTRACT)

    def test_image_requires_opt_in_and_exact_controller(self):
        for field, replacement in (
            ("WA_STUDENT_EVAL", None),
            ("WA_STUDENT_EVAL", "unknown"),
            ("WA_DIAG_CONTROLLER", None),
            ("WA_DIAG_CONTROLLER", "uwb_heading_v1"),
            ("WA_DIAG_CONTROLLER", "learned_target_guard_v3"),
        ):
            env = environment()
            if replacement is None:
                env.pop(field)
            else:
                env[field] = replacement
            with self.subTest(field=field, value=replacement):
                with self.assertRaises(ValueError):
                    evaluation_mode(env)
                with self.assertRaises(ValueError):
                    model_contract(env)

    def test_unknown_modes_are_not_silently_normalized(self):
        for value in ("point", "IMAGE", "image ", "", None, True):
            env = environment()
            env["WA_EVAL_MODE"] = value
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    evaluation_mode(env)
                with self.assertRaises(ValueError):
                    model_contract(env)

    def test_image_disallows_resume_and_targeted_reuse(self):
        for key in ("WA_RESUME_PLAN", "WA_TARGETED_PLAN"):
            env = environment()
            env[key] = "/synthetic/old-rows.json"
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    evaluation_mode(env)
                with self.assertRaises(ValueError):
                    model_contract(env)

    def test_image_keeps_existing_identity_semantic_and_repair_pins(self):
        for key, value in (
            ("WA_EVAL_CHECKPOINT_SHA", "bad"),
            ("WA_EVAL_CHECKPOINT_STEP", "0"),
            ("WA_SEMANTIC_PLY_FIX", "wrong"),
            ("WA_INIT_REPAIR_PLAN_SHA", "b" * 64),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                model_contract(dict(environment(), **{key: value}))


class ModeContractTests(unittest.TestCase):
    def test_ready_crossed_and_missing_modes_fail(self):
        for mode in ("mixed", "image"):
            validate_ready(ready(mode), **CONTRACT, mode=mode)
            wrong = ready("mixed" if mode == "image" else "image")
            with self.assertRaises(ValueError):
                validate_ready(wrong, **CONTRACT, mode=mode)
            missing = ready(mode)
            missing.pop("mode")
            with self.assertRaises(ValueError):
                validate_ready(missing, **CONTRACT, mode=mode)
        with self.assertRaises(ValueError):
            validate_ready(ready("image"), **CONTRACT)

    def test_metrics_identical_and_invalid_false_kept_in_full_denominator(self):
        reports = {}
        for mode in ("mixed", "image"):
            manifest, rows = fixture(mode)
            rows[1].update(policy_init_valid=False, success=False, collision=False,
                           following_rate=0, following_step=0,
                           policy_failure_reason="synthetic initialization failure")
            reports[mode] = summarize(rows, manifest, **CONTRACT, mode=mode)
            self.assertEqual(reports[mode]["mode"], mode)
            metrics = reports[mode]["metrics_percent"]["stt"]
            self.assertEqual(metrics["episodes"], 1405)
            self.assertEqual(metrics["invalid_init_count"], 1)
            expected = sum(r["success"] for r in rows if r["task"] == "stt")
            self.assertAlmostEqual(metrics["SR"], 100 * expected / 1405)
            self.assertAlmostEqual(metrics["CR"],
                                   100 * sum(r["collision"] for r in rows
                                             if r["task"] == "stt") / 1405)
        self.assertEqual(reports["mixed"]["metrics_percent"],
                         reports["image"]["metrics_percent"])

    def test_rows_must_explicitly_match_mode_in_both_validators(self):
        for mode in ("mixed", "image"):
            manifest, rows = fixture(mode)
            validate_student_rows(rows, CONTRACT, mode=mode)
            for replacement in (None, "mixed" if mode == "image" else "image"):
                changed = copy.deepcopy(rows)
                if replacement is None:
                    changed[0].pop("mode")
                else:
                    changed[0]["mode"] = replacement
                with self.subTest(mode=mode, replacement=replacement):
                    with self.assertRaises(ValueError):
                        validate_student_rows(changed, CONTRACT, mode=mode)
                    with self.assertRaises(ValueError):
                        summarize(changed, manifest, **CONTRACT, mode=mode)

    def test_image_does_not_relax_identity_or_frozen_repairs(self):
        _, rows = fixture("image")
        for key, value in (
            ("checkpoint_sha256", "b" * 64),
            ("checkpoint_step", 45900),
            ("semantic_protocol", "wrong"),
            ("initialization_repair_plan_sha256", "c" * 64),
        ):
            changed = copy.deepcopy(rows)
            changed[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_student_rows(changed, CONTRACT, mode="image")

    def test_image_shard_complete_requires_explicit_matching_mode(self):
        manifest, rows = fixture("image")
        keys = {e["key"] for e in manifest["tasks"]["stt"]["episodes"]
                if e["shard"] == 0}
        local = [r for r in rows if r["task"] == "stt" and r["key"] in keys]
        validate_shard(local, dict(episodes=len(local), mode="image"),
                       manifest, 0, ("stt",), mode="image")
        for marker in (dict(episodes=len(local)),
                       dict(episodes=len(local), mode="mixed")):
            with self.subTest(marker=marker), self.assertRaises(ValueError):
                validate_shard(local, marker, manifest, 0, ("stt",), mode="image")

    def test_legacy_mixed_shard_marker_allowed_but_crossed_explicit_mode_not(self):
        manifest, rows = fixture("mixed")
        keys = {e["key"] for e in manifest["tasks"]["stt"]["episodes"]
                if e["shard"] == 0}
        local = [r for r in rows if r["task"] == "stt" and r["key"] in keys]
        validate_shard(local, dict(episodes=len(local)),
                       manifest, 0, ("stt",))
        with self.assertRaises(ValueError):
            validate_shard(local, dict(episodes=len(local), mode="image"),
                           manifest, 0, ("stt",))

    def test_recorder_mode_labels_do_not_misrepresent_observer_gt(self):
        from wa.wm.review_recorder import method_label
        self.assertEqual(method_label(), "WA RGB+BBox+idealUWB")
        self.assertEqual(method_label("mixed"), "WA RGB+BBox+idealUWB")
        self.assertEqual(method_label("image"),
                         "WA RGB+BBox (no UWB; GT telemetry observer-only)")


class ModePartitionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_image_complete_merge_has_24_explicit_shards_and_78_hashes(self):
        roots, manifest = partition_fixture(self.tmp.name, "image")
        rows, hashes = read_partitions(roots, manifest, CONTRACT, mode="image")
        self.assertEqual(len(rows), 4215)
        self.assertEqual(len(hashes), 78)
        self.assertEqual({r["mode"] for r in rows}, {"image"})
        for root in roots.values():
            report = json.loads((root / "PARTITION_COMPLETE.json").read_text())
            self.assertEqual(report["mode"], "image")
            self.assertEqual(report["episodes"], 1405)
        with self.assertRaises(ValueError):
            read_partitions(roots, manifest, CONTRACT)

    def test_image_rejects_missing_or_crossed_partition_and_shard_markers(self):
        roots, manifest = partition_fixture(self.tmp.name, "image")
        targets = [
            roots["stt"] / "PARTITION_COMPLETE.json",
            roots["dt"] / "shard_03" / "COMPLETE.json",
            roots["at"] / "shard_07" / "server_ready.json",
        ]
        for target in targets:
            original = json.loads(target.read_text())
            for replacement in (None, "mixed"):
                changed = dict(original)
                if replacement is None:
                    changed.pop("mode")
                else:
                    changed["mode"] = replacement
                write_json(target, changed)
                with self.subTest(path=target.name, replacement=replacement):
                    with self.assertRaises(ValueError):
                        read_partitions(roots, manifest, CONTRACT, mode="image")
                write_json(target, original)

    def test_rehashed_crossed_rows_cannot_bypass_image_merge(self):
        roots, manifest = partition_fixture(self.tmp.name, "image")
        root = roots["stt"]
        combined_path = root / "combined_episodes.jsonl"
        combined = [json.loads(s) for s in combined_path.read_text().splitlines()]
        key = combined[0]["key"]
        combined[0]["mode"] = "mixed"
        combined_path.write_text("".join(json.dumps(r) + "\n" for r in combined))
        marker_path = root / "PARTITION_COMPLETE.json"
        marker = json.loads(marker_path.read_text())
        marker["combined_sha256"] = hashlib.sha256(combined_path.read_bytes()).hexdigest()
        write_json(marker_path, marker)
        shard_path = root / "shard_00" / "episodes.jsonl"
        local = [json.loads(s) for s in shard_path.read_text().splitlines()]
        for row in local:
            if row["key"] == key:
                row["mode"] = "mixed"
        shard_path.write_text("".join(json.dumps(r) + "\n" for r in local))
        with self.assertRaises(ValueError):
            read_partitions(roots, manifest, CONTRACT, mode="image")

    def test_old_mixed_markers_may_omit_mode_without_relabeling_rows(self):
        roots, manifest = partition_fixture(self.tmp.name, "mixed")
        for root in roots.values():
            paths = [root / "PARTITION_COMPLETE.json"]
            paths += [root / ("shard_%02d" % i) / "COMPLETE.json" for i in range(8)]
            for path in paths:
                marker = json.loads(path.read_text())
                marker.pop("mode", None)
                write_json(path, marker)
        rows, hashes = read_partitions(roots, manifest, CONTRACT)
        self.assertEqual(len(rows), 4215)
        self.assertEqual(len(hashes), 78)
        self.assertEqual({r["mode"] for r in rows}, {"mixed"})
        with self.assertRaises(ValueError):
            read_partitions(roots, manifest, CONTRACT, mode="image")


class ReviewCLIModeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.merged = self.root / "merged"
        self.merged.mkdir()
        self.source = self.root / "synthetic-media-source"
        self.source.mkdir()
        self.output = self.root / "review-output"
        self.manifest, self.rows = fixture("image")
        for row in self.rows:
            row["artifact_root"] = str(self.source)
        self.manifest_path = self.root / "manifest.json"
        write_json(self.manifest_path, self.manifest)
        (self.merged / "combined_episodes.jsonl").write_text(
            "".join(json.dumps(r, allow_nan=False) + "\n" for r in self.rows))
        metrics = summarize(self.rows, self.manifest, **CONTRACT, mode="image")
        self.superiority = dict(all_three_strictly_exceed=False,
                                tasks={}, scope="synthetic fixture only")
        self.saved = dict(
            mode="image", new_episodes=4215, reused_baseline_episodes=0,
            checkpoint_sha256=CONTRACT["checkpoint_sha"],
            checkpoint_step=CONTRACT["step"],
            partition_roots={"synthetic": str(self.root)},
            metrics_percent=metrics["metrics_percent"],
            superiority=self.superiority,
        )
        write_json(self.merged / "summary.json", self.saved)
        self.argv = [
            "build_student_review", "--merged", str(self.merged),
            "--manifest", str(self.manifest_path),
            "--teachers", str(self.root / "unused-teachers.jsonl"),
            "--ffprobe", "unused-ffprobe", "--output", str(self.output),
            "--mode", "image",
        ]

    def fake_digest(self, path):
        # Synthetic manifest intentionally does not impersonate the frozen bytes.
        if Path(path) == self.manifest_path:
            return review.MANIFEST_SHA
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def expect_rejected_before_audits_or_writes(self, saved, argv=None):
        write_json(self.merged / "summary.json", saved)
        with patch("sys.argv", self.argv if argv is None else argv), \
             patch.object(review, "digest", side_effect=self.fake_digest), \
             patch.object(review, "read_partitions") as partitions, \
             patch.object(review, "finalize") as finalize_mock, \
             patch.object(review, "audit_media") as media, \
             patch.object(Path, "write_text") as write:
            with self.assertRaises(ValueError):
                review.main()
            partitions.assert_not_called()
            finalize_mock.assert_not_called()
            media.assert_not_called()
            write.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_saved_checkpoint_sha_mismatch_or_missing_rejected_before_audits(self):
        for value in ("b" * 64, None):
            with self.subTest(value=value):
                self.expect_rejected_before_audits_or_writes(
                    dict(self.saved, checkpoint_sha256=value))

    def test_saved_checkpoint_step_mismatch_or_missing_rejected_before_audits(self):
        for value in (59065, None):
            with self.subTest(value=value):
                self.expect_rejected_before_audits_or_writes(
                    dict(self.saved, checkpoint_step=value))

    def test_saved_mode_mismatch_or_missing_rejected_before_audits(self):
        for value in ("mixed", None):
            with self.subTest(value=value):
                self.expect_rejected_before_audits_or_writes(
                    dict(self.saved, mode=value))

    def test_default_mixed_cli_cannot_publish_image_summary(self):
        self.expect_rejected_before_audits_or_writes(self.saved, self.argv[:-2])

    def test_image_html_discloses_inference_removal_not_retraining(self):
        # Pairing/media are mocked; this verifies publication wiring, not real artifacts.
        import threading
        media_count = [0]
        counter_lock = threading.Lock()
        def fake_media(row, ffprobe):
            # MagicMock.call_count is not an atomic counter across the four workers.
            with counter_lock:
                media_count[0] += 1
            return "synthetic-review.mp4", 1.0
        with patch("sys.argv", self.argv), \
             patch.object(review, "digest", side_effect=self.fake_digest), \
             patch.object(review, "read_partitions", return_value=(self.rows, {})) as partitions, \
             patch.object(review, "finalize",
                          return_value=(dict(superiority=self.superiority),
                                        dict(episodes=4215))), \
             patch.object(review, "audit_media", side_effect=fake_media), \
             patch("builtins.print"):
            review.main()
        self.assertEqual(partitions.call_args.kwargs, dict(mode="image"))
        self.assertEqual(media_count[0], 4215)
        page = (self.output / "index.html").read_text()
        self.assertIn("推理时无UWB", page)
        self.assertIn("不是无UWB重训", page)
        self.assertIn("首帧GT框", page)
        self.assertNotIn("WA使用RGB+首帧GT框+理想UWB", page)
        audit = json.loads((self.output / "audit.json").read_text())
        self.assertEqual((audit["mode"], audit["episodes"]), ("image", 4215))
        self.assertEqual(json.loads((self.output / "summary.json").read_text()),
                         self.saved)


if __name__ == "__main__":
    unittest.main()
