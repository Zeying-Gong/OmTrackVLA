"""Stage the independent verifier, run CPU tests, then audit completed raw sidecars."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parent
NAMES = ['verify_collected_sidecars.py', 'tests/test_verify_collected_sidecars.py',
         'semantic_labels.py', 'collection_contract.py', 'collect_perception_sidecar.py',
         'perception_sidecar_admission_policy_v1.json']
payload = {name: {'base64': base64.b64encode((ROOT/name).read_bytes()).decode(),
                  'sha256': hashlib.sha256((ROOT/name).read_bytes()).hexdigest()} for name in NAMES}
remote_code = '''
import base64, hashlib, json, pathlib, subprocess, sys
repo = pathlib.Path('/data/nfs/share/wam_tracking/OmTrackVLA')
stage = repo/'.codex_upload/perception_sidecar_verification_v1'
plan_path = repo/'outputs/takeover/perception_label_sidecar_train4_v1/frozen_plan.json'
plan_hash = 'dd9fcd6b00249a802410f1638216f56a64af51ef998fd1fc9d6566317e080780'
assert hashlib.sha256(plan_path.read_bytes()).hexdigest() == plan_hash
plan = json.loads(plan_path.read_text())
assert plan['plan_sha256'] == '789c1e4b5a668a1017e29d56aea174721ba08352921df22193877f9b96bd5697'
pins = {ref['path']: ref['sha256'] for ref in plan['artifacts']}
payload = json.load(sys.stdin)
stage.mkdir(exist_ok=False)
(stage/'tests').mkdir()
artifacts = []
for name, item in payload.items():
    target = stage/name
    assert stage.resolve() in target.resolve().parents
    data = base64.b64decode(item['base64'], validate=True)
    assert hashlib.sha256(data).hexdigest() == item['sha256']
    if name in ('semantic_labels.py', 'collection_contract.py', 'collect_perception_sidecar.py'):
        assert item['sha256'] == pins[str(pathlib.Path(plan['bundle'])/name)]
    with target.open('xb') as stream: stream.write(data)
    assert target.read_bytes() == data
    artifacts.append({'path': str(target), 'sha256': item['sha256'], 'bytes': len(data)})
tests = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(stage/'tests'),
                        '-p', 'test_verify_collected_sidecars.py', '-v'], capture_output=True, text=True)
test_result = {'returncode': tests.returncode, 'stdout': tests.stdout, 'stderr': tests.stderr}
with (stage/'CPU_TEST_RESULT.json').open('x') as stream: json.dump(test_result, stream, indent=2)
if tests.returncode:
    print(json.dumps({'status': 'cpu_tests_failed_no_admission_run', 'tests': test_result, 'artifacts': artifacts}))
    raise SystemExit(0)
batch = pathlib.Path(plan['output_root'])/'batch_status.json'
state = json.loads(batch.read_text())
assert all(row['status'] not in ('pending', 'running') for row in state['entries'])
output = repo/'outputs/takeover/perception_sidecar_independent_admission_v1'
execution = subprocess.run([sys.executable, '-B', str(stage/'verify_collected_sidecars.py'),
    '--plan', str(plan_path), '--plan-file-sha256', plan_hash, '--batch-status', str(batch),
    '--admission-policy', str(stage/'perception_sidecar_admission_policy_v1.json'),
    '--output-dir', str(output)], capture_output=True, text=True)
report_path = output/'verification.json'
data = report_path.read_bytes()
print(json.dumps({'status': 'independent_admission_run_complete', 'verifier_exit_code': execution.returncode,
    'stdout': execution.stdout, 'stderr': execution.stderr, 'tests': test_result, 'artifacts': artifacts,
    'report_path': str(report_path), 'report_sha256': hashlib.sha256(data).hexdigest(),
    'report_base64': base64.b64encode(data).decode(), 'report': json.loads(data)}))
'''
spec = importlib.util.spec_from_file_location('sidecar_verifier_ssh_access', r'C:\Users\59783\Desktop\OmTrackVLA\.codex_remote.py')
access = importlib.util.module_from_spec(spec)
spec.loader.exec_module(access)
transport = access.connect()
try:
    channel = transport.open_session(timeout=30)
    channel.settimeout(300)
    channel.exec_command('/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python -B -c ' + shlex.quote(remote_code))
    channel.sendall(json.dumps(payload).encode())
    channel.shutdown_write()
    stdout = channel.makefile('rb').read()
    stderr = channel.makefile_stderr('rb').read()
    status = channel.recv_exit_status()
    if status:
        raise RuntimeError(stderr.decode(errors='replace'))
    result = json.loads(stdout)
    if 'report_base64' in result:
        data = base64.b64decode(result.pop('report_base64'), validate=True)
        assert hashlib.sha256(data).hexdigest() == result['report_sha256']
        directory = ROOT/'sidecar_independent_admission_v1'
        directory.mkdir(exist_ok=False)
        (directory/'verification.json').write_bytes(data)
    (ROOT/'sidecar_verification_remote_execution.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))
finally:
    transport.close()
