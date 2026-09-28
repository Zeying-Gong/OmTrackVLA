"""Portable, bounded result exchange; no training or scheduler submission."""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import zipfile
from pathlib import Path

FILES = ('environment.json', 'config.json', 'metrics.json')
LIMIT = 2 * 1024 * 1024

def digest(data):
    return hashlib.sha256(data).hexdigest()

def git(*args):
    root = Path(__file__).resolve().parents[2]
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()

def preflight(lane, output):
    info = dict(schema_version=1, kind='preflight_only', lane=lane,
                commit=git('rev-parse', 'HEAD'), dirty=git('status', '--porcelain'),
                python=sys.version, platform=platform.platform(), gpu_names=[])
    try:
        import torch
        info.update(torch=torch.__version__, cuda=torch.version.cuda,
                    gpu_names=[torch.cuda.get_device_name(i)
                               for i in range(torch.cuda.device_count())])
    except ImportError:
        info['torch'] = None
    names = info['gpu_names']
    info['gpu_ready'] = bool(names) if lane == 'managed' else (
        len(names) == 8 and all('H100' in name for name in names))
    output.mkdir(parents=True, exist_ok=False)
    (output / 'environment.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info, indent=2))
    return 0 if info['gpu_ready'] else 2

def pack(run, output):
    payload = {}
    for name in FILES:
        path = run / name
        if path.is_symlink():
            raise ValueError('Symlinks are not accepted')
        if path.exists():
            if path.stat().st_size > LIMIT:
                raise ValueError('File exceeds 2 MiB')
            data = path.read_bytes()
            json.loads(data)
            payload[name] = data
    if 'environment.json' not in payload:
        raise ValueError('environment.json required')
    manifest = dict(schema_version=1, kind='result_bundle' if 'metrics.json' in payload
                    else 'preflight_only', files={k: digest(v) for k, v in payload.items()})
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
        archive.writestr('manifest.json', json.dumps(manifest, indent=2))
    return manifest

def verify(bundle):
    with zipfile.ZipFile(bundle) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)) or not set(names) <= set(FILES) | {'manifest.json'}:
            raise ValueError('Unexpected or duplicate bundle entries')
        if any(entry.file_size > LIMIT for entry in entries):
            raise ValueError('Entry exceeds 2 MiB')
        manifest = json.loads(archive.read('manifest.json'))
        if manifest['schema_version'] != 1 or 'environment.json' not in manifest['files']:
            raise ValueError('Invalid manifest')
        if set(names) != set(manifest['files']) | {'manifest.json'}:
            raise ValueError('Manifest/file mismatch')
        for name, expected in manifest['files'].items():
            data = archive.read(name)
            if digest(data) != expected:
                raise ValueError('Checksum mismatch: ' + name)
            json.loads(data)
    return manifest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    pre = commands.add_parser('preflight')
    pre.add_argument('--lane', choices=['managed', 'external-h100'], required=True)
    pre.add_argument('--output', type=Path, required=True)
    make = commands.add_parser('pack')
    make.add_argument('--run', type=Path, required=True)
    make.add_argument('--output', type=Path, required=True)
    check = commands.add_parser('verify')
    check.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'preflight':
        return preflight(args.lane, args.output)
    result = pack(args.run, args.output) if args.command == 'pack' else verify(args.bundle)
    print(json.dumps(result, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
