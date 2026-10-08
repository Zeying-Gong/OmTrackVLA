"""CPU-only v2 launcher tests: no GPU, scheduler or process launch."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from wa.tools import failure_state_launch as old
from wa.tools import failure_state_launch_v2 as launch


class V2LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.args = launch.arguments([
            str(self.root/"out"), "--plan", "/data/nas_ray/base.json",
            "--plan-sha", launch.BASE_SHA, "--continuation", "/data/nas_ray/overlay.json",
            "--continuation-sha", launch.CONTINUATION_SHA, "--development-key", "scene/1"])
        entries = [dict(key=f"scene/{i}", task="stt", artifact_root="/data/nas_ray/old")
                   for i in range(126)]
        entries[0]["key"] = launch.KEY
        self.plan = dict(entries=entries, lanes=[entries[i::8] for i in range(8)],
                         repair_plan=dict(path="/data/nas_ray/repair.json", sha256=old.REPAIR_SHA))
        self.overlay = dict(
            base_plan=dict(path=self.args.plan, sha256=self.args.plan_sha), expected_count=126,
            new_count=125, reused_count=1, reused_keys=[launch.KEY], training_released=False,
            remaining_entries=entries[1:],
            remaining_lanes=[[e for e in lane if e["key"] != launch.KEY] for lane in self.plan["lanes"]],
            experiment=old.EXPERIMENT, protocol_sha256=old.PROTOCOL_SHA,
            frozen_records=[dict(key=launch.KEY, source_protocol=old.PROTOCOL_SHA)])
        self.env = dict(CUDA_VISIBLE_DEVICES="6", PATH="/usr/bin")

    def formal(self):
        result = copy.copy(self.args)
        result.formal, result.development_key = True, None
        return result

    def marker(self, keys, args, lane=0):
        return dict(experiment=launch.EXPERIMENT, entries=len(keys), keys=keys, expected=len(keys),
                    shard=lane, development=not args.formal, training_released=False,
                    no_success_rate=True, plan_sha256=args.plan_sha, base_protocol_sha256=old.PROTOCOL_SHA,
                    protocol_sha256=launch.PROTOCOL_SHA, continuation_sha256=args.continuation_sha)

    def collection(self, args):
        root = self.root/"collection"
        root.mkdir()
        keys = ["scene/1"]
        marker = self.marker(keys, args)
        row = dict(experiment=launch.EXPERIMENT, key=keys[0], task="stt",
                   plan_sha256=args.plan_sha, base_protocol_sha256=old.PROTOCOL_SHA,
                   protocol_sha256=launch.PROTOCOL_SHA, continuation_sha256=args.continuation_sha,
                   outcome="no_valid_teacher_recovery")
        if not args.formal:
            row.update(status="DEVELOPER_NONZERO_REPLAY_CHECK_ONLY", prefix_actions=24,
                       takeover_step=20, branch_total_actions=24, teacher_owned_actions=4,
                       no_success_rate=True, training_eligible=False)
        name = "COMPLETE.json" if args.formal else "DEVELOPMENT_CHECK.json"
        (root/name).write_text(json.dumps(marker))
        (root/"records.jsonl").write_text(json.dumps(row)+"\n")
        return root, keys, marker, row

    def test_parser_requires_explicit_overlay(self):
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            launch.arguments(["/data/nas_ray/out", "--plan", "/data/nas_ray/base",
                              "--plan-sha", launch.BASE_SHA, "--formal"])

    def test_exact_route_and_old_guards_preserved(self):
        self.assertEqual(launch.validate_route(self.args, self.env), ["6"])
        args = self.formal()
        env = dict(MD_AK_JOB_ID="new", MD_AK_TASK_ID="new")
        self.assertEqual(launch.validate_route(args, env, 8), list(map(str, range(8))))
        for change in (dict(plan_sha="f"*64), dict(continuation_sha="e"*64),
                       dict(continuation="relative"), dict(continuation="/data/nas_ray/../overlay")):
            modified = copy.copy(self.args)
            for field, value in change.items():
                setattr(modified, field, value)
            with self.subTest(change=change), self.assertRaises(ValueError):
                launch.validate_route(modified, self.env)
        with self.assertRaises(ValueError):
            launch.validate_route(self.args, dict(self.env, WA_RESUME_PLAN="/old"))
        with self.assertRaises(ValueError):
            launch.validate_route(args, env, 2)

    def test_workset_preserves_original_lane_not_new_stride(self):
        saved = copy.deepcopy((self.plan, self.overlay))
        launch.validate_workset(self.plan, self.overlay, self.args)
        self.assertEqual([len(launch.planned_entries(self.overlay, self.formal(), i))
                          for i in range(8)], [15,16,16,16,16,16,15,15])
        self.assertEqual(self.overlay["remaining_lanes"][0][0]["key"], "scene/8")
        self.assertEqual((self.plan, self.overlay), saved)
        self.assertEqual(len(launch.planned_entries(self.overlay, self.args, 0)), 1)

    def test_bad_workset_rejected(self):
        for mutation in ("base", "count", "reused", "order", "duplicates", "restride", "released"):
            overlay = copy.deepcopy(self.overlay)
            if mutation == "base":
                overlay["base_plan"]["sha256"] = "a"*64
            elif mutation == "count":
                overlay["new_count"] = 126
            elif mutation == "reused":
                overlay["reused_keys"] = ["scene/2"]
            elif mutation == "order":
                overlay["remaining_entries"].reverse()
            elif mutation == "duplicates":
                overlay["remaining_entries"][0] = overlay["remaining_entries"][1]
            elif mutation == "restride":
                overlay["remaining_lanes"] = [overlay["remaining_entries"][i::8] for i in range(8)]
            else:
                overlay["training_released"] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                launch.validate_workset(self.plan, overlay, self.args)

    def test_developer_cannot_repeat_frozen_key(self):
        self.args.development_key = launch.KEY
        with self.assertRaises(ValueError):
            launch.planned_entries(self.overlay, self.args, 0)

    def test_only_worker_entry_changes_normal_server_semantics(self):
        before = old.make_commands(self.args, self.plan, 0, "6", self.root/"lane0", self.env, ["/cuda"])
        after = launch.make_commands(self.args, self.plan, 0, "6", self.root/"lane0", self.env, ["/cuda"])
        self.assertEqual(before[:2], after[:2])
        self.assertEqual(before[2]["env"], after[2]["env"])
        expected = before[2]["argv"].copy()
        expected[expected.index("wa.wm.failure_state_collect")] = "wa.wm.failure_state_collect_v2"
        expected += ["--continuation", self.args.continuation,
                     "--continuation-sha", self.args.continuation_sha]
        self.assertEqual(after[2]["argv"], expected)
        self.assertEqual(old.EXPERIMENT, self.overlay["experiment"])
        self.assertNotEqual(old.EXPERIMENT, launch.EXPERIMENT)

    def test_eight_lane_formal_commands_no_smoke(self):
        for lane in range(8):
            specs = launch.make_commands(self.formal(), self.plan, lane, str(lane),
                                          self.root/f"lane{lane}", self.env, ["/cuda"])
            self.assertNotIn("--development-key", specs[2]["argv"])
            self.assertEqual([s["env"]["CUDA_VISIBLE_DEVICES"] for s in specs], [str(lane)]*3)
            self.assertEqual(specs[2]["env"]["WA_DEVELOPMENT"], "0")

    def test_normal_completion_developer(self):
        root, keys, marker, _ = self.collection(self.args)
        self.assertEqual(launch.check_completion(root, keys, self.args, 0), marker)

    def test_normal_completion_formal(self):
        args = self.formal()
        root, keys, marker, row = self.collection(args)
        for outcome in ("rerun_student_success_no_recovery_needed", "student_invalid_initialization",
                        "repeated_teacher_recovery_candidate", "no_valid_teacher_recovery"):
            row["outcome"] = outcome
            (root/"records.jsonl").write_text(json.dumps(row)+"\n")
            self.assertEqual(launch.check_completion(root, keys, args, 0), marker)

    def test_completion_wrong_protocol_identity_and_partial(self):
        root, keys, marker, row = self.collection(self.args)
        for field, value in (("protocol_sha256", old.PROTOCOL_SHA),
                             ("base_protocol_sha256", launch.PROTOCOL_SHA),
                             ("experiment", old.EXPERIMENT),
                             ("continuation_sha256", "0"*64), ("plan_sha256", "0"*64),
                             ("key", launch.KEY), ("teacher_owned_actions", 0),
                             ("training_eligible", True)):
            changed = dict(row, **{field: value})
            (root/"records.jsonl").write_text(json.dumps(changed)+"\n")
            with self.subTest(field=field), self.assertRaises(ValueError):
                launch.check_completion(root, keys, self.args, 0)
        (root/"records.jsonl").write_text(json.dumps(row))
        with self.assertRaises(ValueError):
            launch.check_completion(root, keys, self.args, 0)
        (root/"records.jsonl").write_text(json.dumps(row)+"\n")
        marker["entries"] = True
        (root/"DEVELOPMENT_CHECK.json").write_text(json.dumps(marker))
        with self.assertRaises(ValueError):
            launch.check_completion(root, keys, self.args, 0)

    def test_full_coverage_keeps_old_protocol(self):
        args = self.formal()
        markers = [self.marker([e["key"] for e in lane], args, i)
                   for i, lane in enumerate(self.overlay["remaining_lanes"])]
        result = launch.coverage(self.plan, self.overlay, args, markers)
        self.assertEqual(result["new_count"], 125)
        self.assertEqual(result["reused_count"], 1)
        self.assertEqual(result["expected_total"], 126)
        self.assertEqual(result["reused_protocol_sha256"], old.PROTOCOL_SHA)
        self.assertEqual(result["new_protocol_sha256"], launch.PROTOCOL_SHA)
        self.assertFalse(result["training_released"])
        self.assertEqual(result["frozen_records"], self.overlay["frozen_records"])

    def test_coverage_missing_duplicate_or_old_key_fails(self):
        args = self.formal()
        markers = [self.marker([e["key"] for e in lane], args, i)
                   for i, lane in enumerate(self.overlay["remaining_lanes"])]
        for mutation in ("missing_lane", "missing_key", "duplicate", "old"):
            changed = copy.deepcopy(markers)
            if mutation == "missing_lane":
                changed.pop()
            elif mutation == "missing_key":
                changed[0]["keys"].pop()
            elif mutation == "duplicate":
                changed[0]["keys"][0] = changed[1]["keys"][0]
            else:
                changed[0]["keys"][0] = launch.KEY
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                launch.coverage(self.plan, self.overlay, args, changed)

    def test_developer_coverage_never_claims_frozen_merge(self):
        result = launch.coverage(self.plan, self.overlay, self.args,
                                 [self.marker(["scene/1"], self.args)])
        self.assertEqual(result["new_count"], 1)
        self.assertNotIn("frozen_records", result)
        self.assertNotIn("expected_total", result)


    def test_production_collector_marker_is_accepted(self):
        from wa.wm.failure_state_collect_v2 import completion_marker
        for formal in (False, True):
            args = self.formal() if formal else copy.copy(self.args)
            args.shard = 0
            path = self.root/("formal" if formal else "development")
            path.mkdir()
            keys = ["scene/1"]
            marker = completion_marker(args, keys, len(keys), not formal)
            (path/("COMPLETE.json" if formal else "DEVELOPMENT_CHECK.json")).write_text(json.dumps(marker))
            row = dict(experiment=launch.EXPERIMENT, task="stt", key=keys[0],
                       plan_sha256=args.plan_sha, base_protocol_sha256=old.PROTOCOL_SHA,
                       continuation_sha256=args.continuation_sha, protocol_sha256=launch.PROTOCOL_SHA,
                       outcome="no_valid_teacher_recovery")
            if not formal:
                row.update(status="DEVELOPER_NONZERO_REPLAY_CHECK_ONLY", prefix_actions=24,
                           takeover_step=20, branch_total_actions=24, teacher_owned_actions=4,
                           no_success_rate=True, training_eligible=False)
            (path/"records.jsonl").write_text(json.dumps(row)+"\n")
            self.assertEqual(launch.check_completion(path, keys, args, 0), marker)


if __name__ == "__main__":
    unittest.main()
