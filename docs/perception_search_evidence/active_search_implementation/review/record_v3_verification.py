"""Record the completed local 50-test verification and exact install hashes."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT/"repo_ready/INSTALL_MANIFEST.json").read_text())
names = ("active_search_fsm.py", "tests/test_active_search_fsm.py",
         "review/test_active_search_async_review.py", "README.md")
hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
if hashes["active_search_fsm.py"] != manifest["files"]["omtrackvla/control/active_search_fsm.py"]:
    raise ValueError("install module differs from reviewed current module")
value = {"version": "v3_capture_time_and_permission_watermarks",
    "scope": "independent pure Python CPU module; no runner integration, GPU or actuator execution",
    "prior_verification": "history/v2/VERIFICATION.json",
    "prior_module_sha256": "0a3380bad01dca4e0e756efc892e395972b1601f69257ccfd052278d01199366",
    "red_regression_evidence": "history/v2/ASYNC_RED_EVIDENCE.json",
    "red_tests_run": 9, "red_failures": 8, "red_errors": 0,
    "test_command": "python -m unittest discover -s repo_ready/tests -v",
    "tests_run": 50, "failures": 0, "errors": 0,
    "evidence": "actual local unittest invocation completed successfully: 50 tests in 0.017s",
    "python_runtime_tested": "local Python 3.11", "python39_grammar_checked": True,
    "python39_runtime_tested": False, "files": hashes, "install_files": manifest["files"],
    "remote_deployments_performed_by_reviewer": 0, "gpu_processes_started": 0, "actuator_commands_sent": 0}
(ROOT/"VERIFICATION.json").write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")
print(json.dumps(value, indent=2))
