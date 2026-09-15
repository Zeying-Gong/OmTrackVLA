"""Transfer this review bundle and run only named CPU setup/verification steps."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import tarfile

HERE = Path(__file__).resolve().parent
REMOTE = '/data/nfs/share/wam_tracking/OmTrackVLA/.codex_upload/perception_label_collection_v1'
REPO = '/data/nfs/share/wam_tracking/OmTrackVLA'
PYTHON = '/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python'


def connection():
    path = HERE.parents[1]/'OmTrackVLA/.codex_remote.py'
    spec = importlib.util.spec_from_file_location('authorized_project_remote', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.connect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['upload', 'cpu-tests', 'freeze', 'preflight', 'download-plan'])
    args = parser.parse_args()
    payload = None
    if args.operation == 'upload':
        files = [HERE/name for name in ['semantic_labels.py', 'collection_contract.py',
            'freeze_perception_plan.py', 'collect_perception_sidecar.py',
            'source_config_inventory.json', 'perception_sidecar_admission_policy_v1.json']]
        files += sorted((HERE/'tests').glob('test_*.py'))
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w:gz') as archive:
            for path in files:
                archive.add(path, arcname=str(path.relative_to(HERE)).replace('\\', '/'))
        payload = stream.getvalue()
        (HERE/'bundle_upload.tar.gz').write_bytes(payload)
        manifest = {str(path.relative_to(HERE)):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
        (HERE/'bundle_upload_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        command = f'mkdir {shlex.quote(REMOTE)} && tar -xzf - -C {shlex.quote(REMOTE)}'
    elif args.operation == 'cpu-tests':
        command = f'cd {REPO} && CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B -m unittest discover -s {REMOTE}/tests -q'
    elif args.operation == 'freeze':
        command = f'cd {REPO} && CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B {REMOTE}/freeze_perception_plan.py --write-plan {REMOTE}/frozen_plan.json'
    elif args.operation == 'preflight':
        command = f'cd {REPO} && CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 {PYTHON} -B {REMOTE}/collect_perception_sidecar.py --plan {REMOTE}/frozen_plan.json'
    else:
        command = f'cat {REMOTE}/frozen_plan.json'
    transport = connection()
    try:
        channel = transport.open_session(timeout=30)
        channel.settimeout(300)
        channel.set_combine_stderr(True)
        channel.exec_command(command)
        if payload is not None:
            channel.sendall(payload)
            channel.shutdown_write()
        data = channel.makefile('rb').read()
        status = channel.recv_exit_status()
    finally:
        transport.close()
    if args.operation == 'download-plan' and status == 0:
        value = json.loads(data)
        (HERE/'frozen_plan.json').write_bytes(data)
        print(json.dumps({'local_plan_saved':True, 'plan_sha256':value['plan_sha256'],
                          'file_sha256':hashlib.sha256(data).hexdigest(), 'bytes':len(data)}))
    else:
        text = data.decode('utf-8', errors='replace')
        print(text)
        (HERE/(args.operation+'_remote.log')).write_text(text, encoding='utf-8')
    return status


if __name__ == '__main__':
    raise SystemExit(main())
