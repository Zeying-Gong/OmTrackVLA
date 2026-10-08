"""Pure CPU checks for the independent fixed90 continuation shell wrapper.

These tests never run its formal path or import CUDA. The allocation snippet is
executed only with a fake torch module and fake nvidia-smi output. Hash checks
operate only on tiny TemporaryDirectory fixtures, not NAS experiment artifacts.
"""
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
OLD_PATH = ROOT / "wa/scripts/failure_state_collect_v2.sh"
PATH = ROOT / "wa/scripts/failure_state_collect_v3.sh"
OLD = OLD_PATH.read_text()
SCRIPT = PATH.read_text()
PLAN_SHA = "2eff9e83e89008ce1ee73632def406a27131fece6e77b01f6d4ca8c02624292b"
CONTINUATION_SHA = "06d098f30ec52d988dc707249e73b17d11642f44846a75015455315891182822"
MANIFESTS = ("dual_teacher_dependencies.sha256", "closed_loop_dependencies.sha256",
             "full_eval_dependencies.sha256", "failure_state_dependencies.sha256")


class WrapperTests(unittest.TestCase):
    def test_bash_syntax(self):
        result = subprocess.run(["bash", "-n", str(PATH)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unchanged_allocation_dependencies_output_and_cleanup_guards(self):
        for begin,end in (('test -x "$PYTHON"', 'check_dependencies() {'),
                          ('check_dependencies() {', '# Extract only'),
                          ('# Extract only', 'run_rc=0')):
            a=OLD.split(begin,1)[1].split(end,1)[0]
            b=SCRIPT.split(begin,1)[1].split(end,1)[0]
            self.assertEqual(b,a.replace("FAILURE_STATE_V2","FAILURE_STATE_V3"))
        self.assertIn('[ "$#" -ne 3 ]',SCRIPT)
        self.assertIn('CONTINUATION="$2"',SCRIPT)
        self.assertIn('CONTINUATION_SHA="$3"',SCRIPT)
        self.assertIn('failure_state_continuation_61836_v1.json',SCRIPT)
        self.assertIn('wa.tools.failure_state_launch_v3',SCRIPT)
        self.assertNotIn('wa.tools.failure_state_launch_v2',SCRIPT)

    def test_no_arguments_and_bad_scheduler_ids_fail_before_gpu(self):
        base = {k: v for k, v in os.environ.items()
                if k not in ("MD_AK_JOB_ID", "MD_AK_TASK_ID", "BASH_ENV", "ENV")}
        result = subprocess.run(["bash", str(PATH)], env=base, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        for job, task in ((None, None), ("0", "2"), ("1", "0"), ("-1", "2"),
                          ("1", "2;echo injected"), ("1.0", "2"), ("01", "2"),
                          (" 1", "2"), ("1", None)):
            env = dict(base)
            if job is not None:
                env["MD_AK_JOB_ID"] = job
            if task is not None:
                env["MD_AK_TASK_ID"] = task
            with self.subTest(job=job, task=task):
                result = subprocess.run(["bash", str(PATH), "/must/not/create", "/unused", "a"*64], env=env,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("requires a real scheduler", result.stderr)
                self.assertNotIn("WRAPPER_START", result.stdout)

    def allocation(self, rows=None, count=8, names=None):
        code = SCRIPT.split('"$PYTHON" -B - <<\'PY\'\n', 1)[1].split("\nPY\n", 1)[0]
        rows = (rows if rows is not None else
                [(str(i), f"GPU-{i}", "NVIDIA A800 80GB", "81920 MiB") for i in range(8)])
        names = names if names is not None else ["NVIDIA A800 80GB"] * 8
        gpu = SimpleNamespace(device_count=lambda: count, get_device_name=lambda i: names[i])
        output = "\n".join(", ".join(row) for row in rows)
        with mock.patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=gpu)}), \
                mock.patch("subprocess.check_output", return_value=output) as query, \
                mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
            exec(compile(code, str(PATH) + ":allocation", "exec"), {})
        query.assert_called_once()
        return stdout.getvalue()

    def test_exact_eight_unique_actual_a800_allocation(self):
        self.assertIn("GPU_ALLOCATION_VERIFIED", self.allocation())

    def test_wrong_gpu_count_or_duplicate_uuid_rejected(self):
        rows = [(str(i), f"GPU-{i}", "NVIDIA A800", "81920 MiB") for i in range(8)]
        for bad in (rows[:7], rows + [("8", "GPU-8", "NVIDIA A800", "81920 MiB")],
                    [rows[0]] + rows[1:7] + [rows[0]]):
            with self.subTest(rows=bad), self.assertRaises(ValueError):
                self.allocation(rows=bad)

    def test_smi_and_cuda_model_must_both_be_a800(self):
        rows = [(str(i), f"GPU-{i}", "NVIDIA A800", "81920 MiB") for i in range(8)]
        rows[3] = ("3", "GPU-3", "NVIDIA H100", "81920 MiB")
        with self.assertRaises(ValueError):
            self.allocation(rows=rows)
        with self.assertRaises(ValueError):
            self.allocation(names=["NVIDIA A800"] * 7 + ["NVIDIA RTX 4090"])

    def test_cuda_visibility_must_be_eight(self):
        for count in (0, 1, 7, 9):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.allocation(count=count)

    def checksum_fixture(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        manifest_dir = root / "wa/wm"
        manifest_dir.mkdir(parents=True)
        data = root / "pinned-data"
        data.write_bytes(b"small dependency fixture\n")
        digest = hashlib.sha256(data.read_bytes()).hexdigest()
        for name in MANIFESTS:
            (manifest_dir / name).write_text(f"{digest}  {data}\n")
        plan, overlay = root / "plan.json", root / "continuation.json"
        plan.write_text('{"expected_count":126}\n')
        overlay.write_text('{"new_count":90,"reused_count":36}\n')
        env = dict(os.environ, PLAN=str(plan), CONTINUATION=str(overlay),
                   PLAN_SHA=hashlib.sha256(plan.read_bytes()).hexdigest(),
                   CONTINUATION_SHA=hashlib.sha256(overlay.read_bytes()).hexdigest())
        env.pop("BASH_ENV", None)
        env.pop("ENV", None)
        return root, plan, overlay, env

    def check_dependencies(self, root, env):
        code = "check_dependencies() {" + SCRIPT.split("check_dependencies() {", 1)[1].split(
            "\n}\ncheck_dependencies", 1)[0] + "\n}\ncheck_dependencies\n"
        return subprocess.run(["bash", "-c", "set -euo pipefail\n" + code],
                              cwd=root, env=env, capture_output=True, text=True)

    def test_real_small_dependency_and_overlay_hash_checks_pass(self):
        root, _, _, env = self.checksum_fixture()
        result = self.check_dependencies(root, env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.count(": OK"), 6)

    def test_overlay_tampering_before_launch_fails_closed(self):
        root, _, overlay, env = self.checksum_fixture()
        overlay.write_text('{"new_count":124,"reused_count":2}\n')
        self.assertNotEqual(self.check_dependencies(root, env).returncode, 0)

    def test_overlay_tampering_after_initial_check_fails_postcheck(self):
        root, _, overlay, env = self.checksum_fixture()
        self.assertEqual(self.check_dependencies(root, env).returncode, 0)
        overlay.write_bytes(overlay.read_bytes() + b" ")
        self.assertNotEqual(self.check_dependencies(root, env).returncode, 0)
        self.assertEqual(SCRIPT.count("\ncheck_dependencies\n"), 1)
        self.assertEqual(SCRIPT.count("\ncheck_dependencies || post_rc=$?\n"), 1)

    def test_plan_or_dependency_tampering_still_rejected(self):
        for target in ("plan", "dependency"):
            root, plan, _, env = self.checksum_fixture()
            path = plan if target == "plan" else root / "pinned-data"
            path.write_bytes(path.read_bytes() + b"tamper")
            with self.subTest(target=target):
                self.assertNotEqual(self.check_dependencies(root, env).returncode, 0)

    def test_launcher_and_postcheck_exit_codes_not_hidden(self):
        tail = "run_rc=0\n" + SCRIPT.split("\nrun_rc=0\n", 1)[1]
        code = ('set -euo pipefail\n'
                'check_dependencies() { return "$POST_RC"; }\n' + tail)
        for executable, post, expected in (("/bin/true", 0, 0), ("/bin/true", 7, 7),
                                            ("/bin/false", 0, 1), ("/bin/false", 7, 1)):
            env = dict(os.environ, PYTHON=executable, OUT="/unused", PLAN="/unused-plan",
                       PLAN_SHA=PLAN_SHA, CONTINUATION="/unused-overlay",
                       CONTINUATION_SHA=CONTINUATION_SHA, POST_RC=str(post))
            env.pop("BASH_ENV", None)
            env.pop("ENV", None)
            with self.subTest(executable=executable, post=post):
                result = subprocess.run(["bash", "-c", code], env=env,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, expected, result.stderr)
                self.assertIn("FAILURE_STATE_V3_WRAPPER_END", result.stdout)

    def test_output_guard_rejects_existing_nested_and_escaping_paths(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        existing = root / "existing"
        existing.mkdir()
        dangling = root / "dangling"
        dangling.symlink_to(root / "missing")
        guard = SCRIPT.split('OUT="$1"\n', 1)[1].split("# Do not mkdir OUT:", 1)[0]
        for name, expected in (("fresh", 0), ("../escape", 2), ("nested/child", 2),
                               ("existing", 2), ("dangling", 2), (".hidden", 2), ("", 2)):
            env = dict(os.environ, OUT=str(root) + "/" + name, TASK_ROOT=str(root))
            env.pop("BASH_ENV", None)
            env.pop("ENV", None)
            with self.subTest(name=name):
                result = subprocess.run(["bash", "-c", "set -euo pipefail\n" + guard],
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, expected, result.stderr)
        self.assertFalse((root / "fresh").exists())


if __name__ == "__main__":
    unittest.main()
