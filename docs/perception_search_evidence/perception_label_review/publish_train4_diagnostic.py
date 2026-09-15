"""Run the same stdlib audit on remote frozen sources and verify output parity."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parent
NAMES = ("train4_visibility_diagnostic.json", "train4_visibility_diagnostic.md")
script = (ROOT / "diagnose_train4_visibility.py").read_bytes()
payload = {"script_base64": base64.b64encode(script).decode(),
           "script_sha256": hashlib.sha256(script).hexdigest(),
           "expected_report_hashes": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in NAMES}}
remote_code = '''
import base64, hashlib, json, pathlib, subprocess, sys
repo = pathlib.Path("/data/nfs/share/wam_tracking/OmTrackVLA")
root = repo / "outputs/takeover/train4_visibility_diagnostic_v1"
root.mkdir(exist_ok=True)
payload = json.load(sys.stdin)
script = base64.b64decode(payload["script_base64"], validate=True)
assert hashlib.sha256(script).hexdigest() == payload["script_sha256"]
target = root / "diagnose_train4_visibility.py"
if target.exists():
    assert not target.is_symlink() and target.read_bytes() == script
else:
    with target.open("xb") as stream: stream.write(script)
execution = subprocess.run([sys.executable, str(target), "--repository-root", str(repo), "--output-dir", str(root)],
                           capture_output=True, text=True, check=True)
summary = json.loads(execution.stdout)
artifacts = [{"path": str(target), "sha256": payload["script_sha256"], "bytes": len(script)}]
for name, expected in payload["expected_report_hashes"].items():
    path = root / name
    assert path.parent == root and not path.is_symlink()
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    assert actual == expected
    artifacts.append({"path": str(path), "sha256": actual, "bytes": len(data)})
print(json.dumps({"status": "remote_train4_source_hashes_verified_reports_identical", "artifacts": artifacts, "summary": summary}))
'''
spec = importlib.util.spec_from_file_location("diagnostic_ssh_access", r"C:\Users\59783\Desktop\OmTrackVLA\.codex_remote.py")
access = importlib.util.module_from_spec(spec)
spec.loader.exec_module(access)
transport = access.connect()
try:
    channel = transport.open_session(timeout=30)
    channel.settimeout(60)
    channel.exec_command("/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python -c " + shlex.quote(remote_code))
    channel.sendall(json.dumps(payload).encode())
    channel.shutdown_write()
    stdout = channel.makefile("rb").read()
    stderr = channel.makefile_stderr("rb").read()
    status = channel.recv_exit_status()
    if status != 0:
        raise RuntimeError(stderr.decode(errors="replace"))
    report = json.loads(stdout)
    assert report["status"] == "remote_train4_source_hashes_verified_reports_identical"
    (ROOT / "train4_visibility_publication.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
finally:
    transport.close()
