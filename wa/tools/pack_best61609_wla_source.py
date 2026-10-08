"""Package the exact private WLA Python snapshot; no weights, upload or extraction."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path('/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928')
SOURCE = ROOT / 'dependencies/wla_v1'
ENVIRONMENT = Path('/data/nas_ray/project/md-ak/users/zeying.gong/job_61609/task_72803/wa_hard_stt_train_a800_v1/environment.json')
ENV_SHA = 'a76ba7506cac9a9a39db839fe554e7d0382c45ca76285fa58130f4bcba2546ad'
OUTPUT = ROOT / 'artifacts/best61609_wla_source_bundle_20261009_v1'
# Exact full pin; keep the helper unchanged because original bundles reference it.
PACKER_SHA = 'de437ffe79315583ec02269604df135b1d0d514080dc9954dd138759c3432876'
ALLOWLIST_SHA = '6027b97b0d6961f4a14ca26e07ea98f34cb1ef7cb0d793737a4a61fec6ef67b4'


def main():
    here = Path(__file__).resolve().parent
    for name, expected in [('pack_best61609_assets.py', PACKER_SHA),
                           ('best61609_asset_manifest.py', ALLOWLIST_SHA)]:
        path = here / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('packaging helper identity mismatch')
    from wa.tools.pack_best61609_assets import (
        open_source, open_directory, pack_archive, verify_archive, write_json_new)
    with open_source(str(ENVIRONMENT)) as stream:
        raw_environment = stream.read()
    if hashlib.sha256(raw_environment).hexdigest() != ENV_SHA:
        raise ValueError('best61609 environment identity mismatch')
    expected_files = json.loads(raw_environment)['wla']['source_files']
    if len(expected_files) != 288:
        raise ValueError('expected the original 288 Python files')
    actual_files = {str(p.relative_to(SOURCE)) for p in (SOURCE / 'src/md_wla').rglob('*.py')}
    if actual_files != set(expected_files):
        raise ValueError('WLA Python file set differs from original environment')
    entries = []
    for relative, expected in sorted(expected_files.items()):
        rel = Path(relative)
        if str(rel) != relative or rel.is_absolute() or '..' in rel.parts or not relative.startswith('src/md_wla/') or rel.suffix != '.py':
            raise ValueError('noncanonical source reference')
        path = SOURCE / rel
        with open_source(str(path)) as stream:
            data = stream.read()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected:
            raise ValueError('WLA source hash mismatch: ' + relative)
        # Low-level packer supports empty Python package files; no semantic edits.
        entries.append(dict(source=str(path), archive_member=str(path)[1:],
                            size=len(data), sha256=digest, package='private_wla_source'))
    entries.append(dict(source=str(ENVIRONMENT), archive_member=str(ENVIRONMENT)[1:],
                        size=len(raw_environment), sha256=ENV_SHA, package='provenance'))
    entries.sort(key=lambda item: item['archive_member'])
    if len({e['archive_member'] for e in entries}) != 289:
        raise ValueError('duplicate source')
    start = time.monotonic()
    with open_directory(ROOT / 'artifacts') as parent_fd:
        os.mkdir(OUTPUT.name, 0o700, dir_fd=parent_fd)
        out_fd = os.open(OUTPUT.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                         dir_fd=parent_fd)
    try:
        manifest = dict(schema='wa_best61609_private_wla_source_v1',
            status='PINNED_SOURCE_ONLY_NOT_RUNTIME', source=str(SOURCE),
            environment=str(ENVIRONMENT), environment_sha256=ENV_SHA, files=entries,
            python_files=288, source_bytes=sum(e['size'] for e in entries if e['package'] == 'private_wla_source'),
            uploaded=False, public_redistribution_authorized=False,
            license_note='Snapshot contains no LICENSE. Private authorized handoff only; do not infer an open-source license.',
            excluded=['weights', 'DINO/JEPA checkouts', 'environments', 'scenes',
                      '.git metadata', 'bytecode', 'training caches', 'credentials'])
        manifest_meta = write_json_new(out_fd, 'manifest.json', manifest)
        name = 'best61609_wla_source.tar'
        pack_archive(out_fd, name, entries)
        archive = verify_archive(out_fd, name, entries)
        result = dict(status='PASS_PRIVATE_WLA_SOURCE_BUNDLE_NOT_UPLOADED',
            created_utc=datetime.now(timezone.utc).isoformat(), elapsed_s=time.monotonic()-start,
            files=289, python_files=288, source_bytes=manifest['source_bytes'],
            source_hashes_equal_best61609=True, environment_sha256=ENV_SHA,
            manifest=manifest_meta, archive=archive, uploaded=False,
            h100_validated=False, contains_weights=False, contains_environment=False,
            contains_scenes=False, public_redistribution_authorized=False,
            builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            packer_sha256=PACKER_SHA,
            note='Original file bytes; relative archive paths; file modes0600. Python source only, not a full runtime.')
        os.fsync(out_fd)
        complete = write_json_new(out_fd, 'complete.json', result)
    except Exception as exc:
        write_json_new(out_fd, 'failed.json', dict(status='FAILED_PRESERVED', error=str(exc), uploaded=False))
        raise
    finally:
        os.close(out_fd)
    print(json.dumps(dict(status=result['status'], output=str(OUTPUT), complete=complete,
                          archive=archive)), flush=True)


if __name__ == '__main__':
    main()
