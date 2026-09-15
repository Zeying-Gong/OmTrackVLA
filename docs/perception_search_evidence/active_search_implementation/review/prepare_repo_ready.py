"""Build exclusive local install candidates; no remote operations."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
target = ROOT/"repo_ready"
if target.exists():
    raise ValueError("repo_ready already exists")
records = {}
for source_name, output_name in (
    ("active_search_fsm.py", "omtrackvla/control/active_search_fsm.py"),
    ("tests/test_active_search_fsm.py", "tests/test_active_search_fsm.py"),
    ("review/test_active_search_async_review.py", "tests/test_active_search_async.py"),
):
    source = ROOT/source_name
    data = source.read_bytes()
    if output_name.startswith("tests/"):
        text = data.decode()
        if text.count("from active_search_fsm import ") != 1:
            raise ValueError("unexpected test import layout")
        data = text.replace("from active_search_fsm import ", "from omtrackvla.control.active_search_fsm import ").encode()
    ast.parse(data.decode(), filename=output_name, feature_version=(3, 9))
    output = target/output_name
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(data)
    records[output_name] = hashlib.sha256(data).hexdigest()
with (target/"INSTALL_MANIFEST.json").open("x", encoding="utf-8") as handle:
    json.dump({"stage": "independent_active_search_fsm_v3_install_candidate", "files": records,
        "scope": "pure CPU module only; no runner integration or actuator deployment", "python39_grammar_checked": True}, handle, indent=2)
    handle.write("\n")
print(json.dumps(records, indent=2))
