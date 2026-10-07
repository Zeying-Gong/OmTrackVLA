"""CPU-only command construction, routing and owned-process safety tests."""
import copy
import json
import os
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from wa.tools import failure_state_launch as launch


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.args = launch.arguments([
            str(self.root / "output"), "--plan", "/data/nas_ray/frozen_plan.json",
            "--plan-sha", "a" * 64, "--development-key", "VLzqgDo317F/238"])
        entries = [dict(key=f"scene/{i}", artifact_root="/data/nas_ray/old",
                        task="stt") for i in range(126)]
        entries[0]["key"] = self.args.development_key
        self.plan = dict(entries=entries, lanes=[entries[i::8] for i in range(8)],
                         repair_plan=dict(path="/data/nas_ray/repair.json",
                                          sha256=launch.REPAIR_SHA))
        self.env = dict(CUDA_VISIBLE_DEVICES="6", PATH="/usr/bin", SECRET_TOKEN="do-not-log")
        self.specs = launch.make_commands(self.args, self.plan, 0, "6",
                                         self.root / "lane0", self.env, ["/cuda/lib"])

    def formal(self):
        result = copy.copy(self.args)
        result.formal, result.development_key = True, None
        return result

    def ready(self):
        return dict(checkpoint=launch.CHECKPOINT_PATH, checkpoint_sha256=launch.CHECKPOINT_SHA,
                    step=59716, mode="mixed", noise_mode="zero", sampling_steps=4, seed="7+step",
                    text_used=False, world_predictor_inference=False, url="http://127.0.0.1:12345")

    def completion(self, development=True):
        collection = self.root / "collection"
        collection.mkdir()
        keys = [self.args.development_key]
        marker = dict(experiment=launch.EXPERIMENT, entries=1, keys=keys, expected=1, shard=0,
                      development=development, training_released=False, no_success_rate=True,
                      plan_sha256=self.args.plan_sha, protocol_sha256=launch.PROTOCOL_SHA)
        row = (dict(key=keys[0], status="DEVELOPER_NONZERO_REPLAY_CHECK_ONLY", prefix_actions=24,
                    takeover_step=20, branch_total_actions=24, teacher_owned_actions=4,
                    no_success_rate=True, training_eligible=False) if development else
               dict(key=keys[0], task="stt", plan_sha256=self.args.plan_sha,
                    protocol_sha256=launch.PROTOCOL_SHA,
                    outcome="rerun_student_success_no_recovery_needed"))
        marker_path = collection / ("DEVELOPMENT_CHECK.json" if development else "COMPLETE.json")
        marker_path.write_text(json.dumps(marker))
        (collection / "records.jsonl").write_text(json.dumps(row) + "\n")
        return collection, keys, marker, marker_path, row

    def test_parser_requires_exact_mode(self):
        base = ["/data/nas_ray/out", "--plan", "/data/nas_ray/p", "--plan-sha", "a"*64]
        with mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                launch.arguments(base)
            with self.assertRaises(SystemExit):
                launch.arguments(base+["--formal", "--development-key", "x/1"])

    def test_developer_only_explicit_single_gpu(self):
        self.assertEqual(launch.validate_route(self.args, self.env), ["6"])
        for env in ({}, dict(CUDA_VISIBLE_DEVICES="6,7"), dict(self.env, MD_AK_JOB_ID="1"),
                    dict(self.env, MD_AK_TASK_ID="2")):
            with self.subTest(env=env), self.assertRaises(ValueError):
                launch.validate_route(self.args, env)

    def test_formal_requires_scheduler_and_actual_eight_devices(self):
        a = self.formal()
        env = dict(MD_AK_JOB_ID="1", MD_AK_TASK_ID="2")
        self.assertEqual(launch.validate_route(a, env, 8), list(map(str, range(8))))
        for bad in (None, 1, 7, True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                launch.validate_route(a, env, bad)
        with self.assertRaises(ValueError):
            launch.validate_route(a, {}, 8)
        with self.assertRaises(ValueError):
            launch.validate_route(a, dict(env, CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,6"), 8)

    def test_no_reuse_and_explicit_plan(self):
        for field in ("WA_RESUME_PLAN", "WA_TARGETED_PLAN"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                launch.validate_route(self.args, dict(self.env, **{field: "/old"}))
        for field, value in (("plan", "relative"), ("plan_sha", "a"*63)):
            a = copy.copy(self.args); setattr(a, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                launch.validate_route(a, self.env)

    def test_reserved_ports_displays(self):
        for field, value in (("port_base", 18800), ("port_base", 80),
                             ("display_base", 530), ("display_base", 0)):
            a = copy.copy(self.args); setattr(a, field, value)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                launch.validate_route(a, self.env)

    def test_output_exclusive_and_protected(self):
        out = self.root / "fresh"
        self.assertEqual(launch.output_path(str(out), allowed_roots=[self.root]), out)
        out.mkdir()
        with self.assertRaises(ValueError):
            launch.output_path(str(out), allowed_roots=[self.root])
        link = self.root / "link"; link.symlink_to(self.root / "missing")
        with self.assertRaises(ValueError):
            launch.output_path(str(link), allowed_roots=[self.root])
        with self.assertRaises(ValueError):
            launch.output_path(str(self.root / "old/new"), [self.root/"old"], allowed_roots=[self.root])
        with self.assertRaises(ValueError):
            launch.output_path("/unapproved/new", allowed_roots=[self.root])
        with self.assertRaises(ValueError):
            launch.output_path(str(self.root / "../new"), allowed_roots=[self.root])

    def test_occupied_socket_fails_without_signaling(self):
        with mock.patch.object(launch.os.path, "lexists", return_value=True), \
                mock.patch.object(launch.os, "killpg") as kill:
            with self.assertRaises(ValueError):
                launch.check_address(19170, 570)
            kill.assert_not_called()
        probe = mock.MagicMock()
        probe.__enter__.return_value = probe
        probe.bind.side_effect = OSError("in use")
        with mock.patch.object(launch.os.path, "lexists", return_value=False), \
                mock.patch.object(launch.socket, "socket", return_value=probe):
            with self.assertRaises(OSError):
                launch.check_address(19170, 570)

    def test_fixed126_lane_selection_no_filtered_formal(self):
        a = self.formal()
        lanes = [launch.planned_entries(self.plan, a, i) for i in range(8)]
        self.assertEqual(list(map(len, lanes)), [16]*6+[15]*2)
        self.assertEqual(len({e["key"] for lane in lanes for e in lane}), 126)
        self.assertEqual(launch.planned_entries(self.plan, self.args, 0)[0]["key"],
                         self.args.development_key)
        self.args.development_key = "other/2"
        with self.assertRaises(ValueError):
            launch.planned_entries(self.plan, self.args, 0)

    def test_commands_use_exact_model_and_new_worker(self):
        wa, ln, worker = self.specs
        self.assertIn(launch.CHECKPOINT_PATH, wa["argv"])
        self.assertNotIn("--developer-check", wa["argv"])
        self.assertEqual(wa["argv"][-4:], ["--mode", "mixed", "--noise-mode", "zero"])
        self.assertIn("wa.wm.failure_state_collect", worker["argv"])
        self.assertIn("--development-key", worker["argv"])
        self.assertEqual(worker["env"]["WA_DIAG_CONTROLLER"], "learned_yaw_guard_v1")
        self.assertEqual(worker["env"]["WA_EVAL_CHECKPOINT_SHA"], launch.CHECKPOINT_SHA)
        self.assertEqual(worker["env"]["WA_EVAL_CHECKPOINT_STEP"], "59716")
        self.assertEqual(worker["env"]["WA_EVAL_MODE"], "mixed")
        self.assertEqual(worker["env"]["WA_INIT_REPAIR_PLAN_SHA"], launch.REPAIR_SHA)
        self.assertEqual(worker["env"]["WA_DEVELOPMENT"], "1")
        self.assertIn("ws://127.0.0.1:19170", worker["argv"])
        self.assertEqual(worker["env"]["OMTRACKVLA_XVFB_DISPLAY_NUM"], "570")
        self.assertTrue(any(x.endswith("lightnav_server_compat.py") for x in ln["argv"]))
        self.assertEqual(ln["env"]["WA_CUDA_MOUNT_COMPAT"], "1")
        self.assertEqual([p["env"]["CUDA_VISIBLE_DEVICES"] for p in self.specs], ["6"]*3)

    def test_formal_command_maps_all_lanes_without_smoke(self):
        for lane in range(8):
            specs = launch.make_commands(self.formal(), self.plan, lane, str(lane),
                                         self.root/f"lane{lane}", self.env, ["/cuda"])
            self.assertEqual([s["env"]["CUDA_VISIBLE_DEVICES"] for s in specs], [str(lane)]*3)
            self.assertNotIn("--development-key", specs[2]["argv"])
            self.assertEqual(specs[2]["env"]["WA_DEVELOPMENT"], "0")
            self.assertIn(f"ws://127.0.0.1:{19170+lane}", specs[2]["argv"])
            self.assertEqual(specs[2]["env"]["OMTRACKVLA_XVFB_DISPLAY_NUM"], str(570+lane))

    def test_command_records_do_not_save_credentials(self):
        for spec in self.specs:
            self.assertNotIn("SECRET_TOKEN", launch.command_record(spec)["env"])
            self.assertNotIn("do-not-log", json.dumps(launch.command_record(spec)))

    def test_ready_rejects_old_model_wrong_mode_and_bool_int(self):
        launch.validate_wa_ready(self.ready())
        for field, value in (("checkpoint_sha256", "b"*64), ("step", True),
                             ("step", 59715), ("mode", "image"), ("sampling_steps", 4.0),
                             ("text_used", 0), ("world_predictor_inference", True),
                             ("noise_mode", "random"), ("seed", "7"),
                             ("url", "http://example.org:80")):
            ready = self.ready(); ready[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                launch.validate_wa_ready(ready)

    def test_developer_completion_is_not_SR(self):
        c, keys, marker, _, _ = self.completion()
        self.assertEqual(launch.check_completion(c, keys, self.args.plan_sha, 0, True), marker)
        self.assertFalse(marker["training_released"])
        self.assertTrue(marker["no_success_rate"])

    def test_completion_strict_bool_int_and_partial_rows(self):
        c, keys, marker, path, row = self.completion()
        marker["entries"] = True; path.write_text(json.dumps(marker))
        with self.assertRaises(ValueError):
            launch.check_completion(c, keys, self.args.plan_sha, 0, True)
        marker["entries"] = 1; path.write_text(json.dumps(marker))
        (c/"records.jsonl").write_text(json.dumps(row))
        with self.assertRaises(ValueError):
            launch.check_completion(c, keys, self.args.plan_sha, 0, True)

    def test_completion_wrong_keys_or_zero_takeover(self):
        c, keys, _, _, row = self.completion()
        for key, value in (("key", "foreign/1"), ("takeover_step", 0),
                           ("training_eligible", True), ("teacher_owned_actions", 20)):
            changed = dict(row, **{key: value})
            (c/"records.jsonl").write_text(json.dumps(changed)+"\n")
            with self.subTest(key=key), self.assertRaises(ValueError):
                launch.check_completion(c, keys, self.args.plan_sha, 0, True)

    def test_formal_completion_all_outcomes_no_training_release(self):
        c, keys, marker, _, row = self.completion(False)
        for outcome in ("rerun_student_success_no_recovery_needed", "student_invalid_initialization",
                        "repeated_teacher_recovery_candidate", "no_valid_teacher_recovery"):
            (c/"records.jsonl").write_text(json.dumps(dict(row, outcome=outcome))+"\n")
            self.assertEqual(launch.check_completion(c, keys, self.args.plan_sha, 0, False), marker)
        (c/"records.jsonl").write_text(json.dumps(dict(row, outcome="legacy_success"))+"\n")
        with self.assertRaises(ValueError):
            launch.check_completion(c, keys, self.args.plan_sha, 0, False)

    def owned(self):
        owned = object.__new__(launch.OwnedProcess)
        owned.process = mock.Mock(pid=12345, returncode=None)
        owned.event = mock.Mock()
        return owned

    def test_spawn_journal_failure_cleans_just_created_process(self):
        spec = dict(role="test", argv=["not-executed"], env={}, cwd=str(self.root),
                    log=str(self.root/"owned.log"))
        with mock.patch.object(launch.subprocess, "Popen") as popen, \
                mock.patch.object(launch.OwnedProcess, "event", side_effect=OSError("disk full")), \
                mock.patch.object(launch.OwnedProcess, "stop") as stop:
            with self.assertRaises(OSError):
                launch.OwnedProcess(spec, self.root/"events.jsonl")
            stop.assert_called_once()
            self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_exit_status_uses_WNOWAIT_and_never_reaps(self):
        owned = self.owned()
        status = SimpleNamespace(si_status=0, si_code=os.CLD_EXITED)
        with mock.patch.object(launch.os, "waitid", return_value=status) as wait:
            self.assertEqual(owned.exit_code(), 0)
            wait.assert_called_once_with(os.P_PID, 12345, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        owned.process.poll.assert_not_called()
        owned.process.wait.assert_not_called()

    def test_cleanup_even_when_leader_exited_only_own_group(self):
        owned = self.owned(); owned.live_group = mock.Mock(return_value=False)
        owned.process.wait.return_value = 0
        with mock.patch.object(launch.os, "getpgid", return_value=12345), \
                mock.patch.object(launch.os, "killpg") as kill:
            owned.stop()
            kill.assert_called_once_with(12345, signal.SIGTERM)
        owned.process.poll.assert_not_called()
        owned.process.wait.assert_called_once_with(timeout=15)
        owned.event.assert_called_once_with("exit", pid=12345, returncode=0)

    def test_cleanup_refuses_foreign_or_reaped_leader(self):
        owned = self.owned()
        with mock.patch.object(launch.os, "getpgid", return_value=23456), \
                mock.patch.object(launch.os, "killpg") as kill:
            with self.assertRaises(RuntimeError):
                owned.stop()
            kill.assert_not_called()
        owned.process.returncode = 0
        with mock.patch.object(launch.os, "killpg") as kill:
            with self.assertRaises(RuntimeError):
                owned.stop()
            kill.assert_not_called()

    def test_formal_missing_scheduler_rejected_before_cuda_import_or_plan(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(launch, "load_plan") as load:
            with self.assertRaisesRegex(ValueError, "before CUDA import"):
                launch.main([str(self.root/"new"), "--plan", "/data/nas_ray/p",
                             "--plan-sha", "a"*64, "--formal"])
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
