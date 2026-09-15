"""Preserve v2 source and actual failing regressions before a local-only fix."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import unittest

ROOT = Path(__file__).resolve().parents[1]
old = json.loads((ROOT/"VERIFICATION.json").read_text())
destination = (ROOT/"history/v2").resolve()
if not destination.is_relative_to(ROOT.resolve()) or destination.exists():
    raise ValueError("history/v2 must be a new directory inside this review")
for name, expected in old["files"].items():
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
        raise ValueError("v2 source changed before preservation: " + name)
names = [*old["files"], "VERIFICATION.json", "review/test_active_search_async_review.py"]
for name in names:
    target = destination/name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT/name, target)
stream = io.StringIO()
suite = unittest.defaultTestLoader.discover(str(ROOT/"review"), pattern="test_active_search_async_review.py")
result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
value = {"module_sha256": old["files"]["active_search_fsm.py"], "tests_run": result.testsRun,
         "failures": len(result.failures), "errors": len(result.errors), "output": stream.getvalue()}
for path in (destination/"ASYNC_RED_EVIDENCE.json", ROOT/"review/ASYNC_RED_EVIDENCE.json"):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
if (result.testsRun, len(result.failures), len(result.errors)) != (9, 8, 0):
    raise ValueError("unexpected regression baseline")
print(json.dumps({key: item for key, item in value.items() if key != "output"}))
