"""Execute and retrieve the final bounded CPU diagnostic on hash-pinned inputs."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parent
data = (ROOT/'diagnose_bbox_localization.py').read_bytes()
payload = {'base64': base64.b64encode(data).decode(), 'sha256': hashlib.sha256(data).hexdigest()}
code = '''
import base64, hashlib, json, pathlib, subprocess, sys
root=pathlib.Path('/data/nfs/share/wam_tracking/OmTrackVLA')
output=root/'outputs/takeover/bbox_localization_diagnostic_v1'
payload=json.load(sys.stdin)
data=base64.b64decode(payload['base64'],validate=True)
assert hashlib.sha256(data).hexdigest()==payload['sha256']
output.mkdir(exist_ok=False)
script=output/'diagnose_bbox_localization.py'
with script.open('xb') as stream: stream.write(data)
run=subprocess.run([sys.executable,'-B',str(script),'--repository',str(root),'--output-dir',str(output)],capture_output=True,text=True)
if run.returncode:
    print(json.dumps({'status':'diagnostic_failed','exit_code':run.returncode,'stderr':run.stderr}))
else:
    artifacts=[]
    for name in ('bbox_localization_diagnostic.json','bbox_localization_diagnostic.md'):
        path=output/name
        data=path.read_bytes()
        artifacts.append({'path':str(path),'name':name,'sha256':hashlib.sha256(data).hexdigest(),'base64':base64.b64encode(data).decode()})
    print(json.dumps({'status':'diagnostic_complete','summary':json.loads(run.stdout),'artifacts':artifacts,'script_sha256':payload['sha256']}))
'''
spec = importlib.util.spec_from_file_location('bbox_diagnostic_access', r'C:\Users\59783\Desktop\OmTrackVLA\.codex_remote.py')
access = importlib.util.module_from_spec(spec); spec.loader.exec_module(access)
transport = access.connect()
try:
    channel = transport.open_session(timeout=30); channel.settimeout(60)
    channel.exec_command('/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python -B -c ' + shlex.quote(code))
    channel.sendall(json.dumps(payload).encode()); channel.shutdown_write()
    stdout = channel.makefile('rb').read(); stderr = channel.makefile_stderr('rb').read()
    if channel.recv_exit_status(): raise RuntimeError(stderr.decode(errors='replace'))
    result = json.loads(stdout)
    for item in result.get('artifacts', []):
        data = base64.b64decode(item.pop('base64'), validate=True)
        assert hashlib.sha256(data).hexdigest() == item['sha256']
        with (ROOT/item['name']).open('xb') as stream: stream.write(data)
    (ROOT/'bbox_localization_publication.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))
finally:
    transport.close()
