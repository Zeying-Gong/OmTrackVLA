"""NAS-only, allowlisted best61609 packaging; never authenticates or uploads.

Archives retain original bytes under relative data/nas_ray/... names. This is
not an extractor or a claim that environments, scenes or H100 are ready.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tarfile
import time
import uuid

from wa.tools.best61609_asset_manifest import build_manifest

ARTIFACTS = Path('/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts')
CHUNK = 1024 * 1024

def canonical_absolute(value):
    if not isinstance(value, str) or not value.startswith('/') or value.startswith('//'):
        raise ValueError('absolute source required')
    p = Path(value)
    if str(p) != value or '..' in p.parts or len(p.parts) < 2:
        raise ValueError('noncanonical source')
    return p

@contextmanager
def open_directory(path):
    """Walk through directory FDs: no symlink ancestors or path resolution race."""
    p = canonical_absolute(str(path)) if str(path) != '/' else Path('/')
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in p.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)

@contextmanager
def open_source(source):
    p = canonical_absolute(source)
    with open_directory(p.parent) as parent:
        fd = os.open(p.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError('source is not regular')
        stream = os.fdopen(fd, 'rb')
    except BaseException:
        os.close(fd)
        raise
    with stream:
        yield stream

class HashReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.count = 0
    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        self.count += len(data)
        return data

def normalize_entries(files):
    result, seen = [], set()
    for item in files:
        if set(item) != {'source', 'sha256', 'size', 'package'}:
            raise ValueError('unexpected entry schema')
        p = canonical_absolute(item['source'])
        if item['package'] not in ('weights', 'evidence'):
            raise ValueError('unknown package')
        if type(item['size']) is not int or item['size'] <= 0:
            raise ValueError('invalid file size')
        if not isinstance(item['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', item['sha256']):
            raise ValueError('invalid file hash')
        if str(p) in seen:
            raise ValueError('duplicate source')
        seen.add(str(p))
        result.append(dict(item, archive_member=str(p)[1:]))
    if not result:
        raise ValueError('empty package')
    return sorted(result, key=lambda x: x['archive_member'])

def signature(st):
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns

def pack_archive(out_fd, name, entries):
    """Read and hash each exact source while writing, leaving failures in place."""
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=out_fd)
    with os.fdopen(fd, 'wb') as raw:
        with tarfile.open(fileobj=raw, mode='w', format=tarfile.PAX_FORMAT, copybufsize=CHUNK) as archive:
            for i, entry in enumerate(entries):
                with open_source(entry['source']) as stream:
                    before = os.fstat(stream.fileno())
                    if before.st_size != entry['size']:
                        raise ValueError('source size differs: ' + entry['source'])
                    reader = HashReader(stream)
                    info = tarfile.TarInfo(entry['archive_member'])
                    info.size = entry['size']
                    info.mode = 0o600
                    archive.addfile(info, reader)
                    if reader.count != entry['size'] or reader.digest.hexdigest() != entry['sha256']:
                        raise ValueError('source hash differs: ' + entry['source'])
                    if stream.read(1) or signature(before) != signature(os.fstat(stream.fileno())):
                        raise ValueError('source changed during pack: ' + entry['source'])
                if i and i % 2000 == 0:
                    print(json.dumps({'phase': 'pack', 'archive': name, 'files': i}), flush=True)
        raw.flush()
        os.fsync(raw.fileno())

def verify_archive(out_fd, name, entries):
    expected = {e['archive_member']: e for e in entries}
    if len(expected) != len(entries):
        raise ValueError('duplicate expected archive name')
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=out_fd)
    with os.fdopen(fd, 'rb') as raw:
        before = os.fstat(raw.fileno())
        reader = HashReader(raw)
        payload_bytes = 0
        for entry in entries:
            info = tarfile.TarInfo(entry['archive_member'])
            info.size = entry['size']
            info.mode = 0o600
            header = info.tobuf(format=tarfile.PAX_FORMAT)
            if reader.read(len(header)) != header:
                raise ValueError('archive header differs: extra, renamed, duplicate or unsafe member')
            digest, remaining = hashlib.sha256(), entry['size']
            while remaining:
                chunk = reader.read(min(CHUNK, remaining))
                if not chunk:
                    raise ValueError('truncated archive payload')
                digest.update(chunk)
                remaining -= len(chunk)
            if digest.hexdigest() != entry['sha256']:
                raise ValueError('archive content hash mismatch')
            padding = (-entry['size']) % 512
            if reader.read(padding) != b'\0' * padding:
                raise ValueError('nonzero or missing member padding')
            payload_bytes += entry['size']
        footer = 1024 + (-(reader.count + 1024)) % tarfile.RECORDSIZE
        if reader.read(footer) != b'\0' * footer or reader.read(1):
            raise ValueError('extra, missing, or nonzero trailing archive data')
        if reader.count != before.st_size or signature(before) != signature(os.fstat(raw.fileno())):
            raise ValueError('archive changed or byte count mismatch')
        return {'name': name, 'files': len(entries), 'payload_bytes': payload_bytes,
                'bytes': before.st_size, 'sha256': reader.digest.hexdigest()}

def write_json_new(out_fd, name, value):
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()
    temp = '.' + name + '.' + uuid.uuid4().hex + '.tmp'
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=out_fd)
    published = False
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(out_fd)
        os.link(temp, name, src_dir_fd=out_fd, dst_dir_fd=out_fd, follow_symlinks=False)
        published = True
    finally:
        try:
            os.unlink(temp, dir_fd=out_fd)
        except OSError:
            if not published:
                raise
    return {'name': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

def build_bundle(output):
    output = canonical_absolute(str(output))
    if output.parent != ARTIFACTS or not re.fullmatch(r'best61609_private_bundle_[0-9]{8}_v[1-9][0-9]*', output.name):
        raise ValueError('output must be a new bounded best61609 NAS artifact directory')
    manifest = build_manifest()
    if manifest['schema'] != 'wa_best61609_asset_manifest_v1':
        raise ValueError('unexpected manifest schema')
    entries = normalize_entries(manifest['files'])
    if any(not e['source'].startswith('/data/nas_ray/') for e in entries):
        raise ValueError('outside NAS')
    start = time.monotonic()
    with open_directory(ARTIFACTS) as parent_fd:
        os.mkdir(output.name, mode=0o700, dir_fd=parent_fd)
        out_fd = os.open(output.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
    try:
        manifest_meta = write_json_new(out_fd, 'manifest.json', dict(manifest, files=entries))
        packages = []
        for group in ('weights', 'evidence'):
            selected = [e for e in entries if e['package'] == group]
            if not selected:
                raise ValueError('missing package ' + group)
            name = 'best61609_' + group + '.tar'
            print(json.dumps({'phase': 'start', 'archive': name, 'files': len(selected)}), flush=True)
            pack_archive(out_fd, name, selected)
            packages.append(verify_archive(out_fd, name, selected))
            print(json.dumps({'phase': 'verified', **packages[-1]}), flush=True)
        with open_source(str(Path(__file__).absolute())) as source:
            tool_sha = hashlib.file_digest(source, 'sha256').hexdigest()
        result = {'status': 'PASS_NAS_BUNDLE_NOT_UPLOADED', 'created_utc': datetime.now(timezone.utc).isoformat(),
                  'elapsed_s': time.monotonic() - start, 'tool_sha256': tool_sha,
                  'manifest': manifest_meta, 'archives': packages,
                  'files': len(entries), 'payload_bytes': sum(e['size'] for e in entries),
                  'uploaded': False, 'h100_validated': False, 'contains_environment': False,
                  'contains_scenes': False, 'contains_dino': False,
                  'contains_old_wla_or_jepa_initializers': False,
                  'note': 'Private handoff only; no re-encoding, extraction, authentication, or upload.'}
        os.fsync(out_fd)
        result_meta = write_json_new(out_fd, 'complete.json', result)
    except Exception as exc:
        write_json_new(out_fd, 'failed.json', {'status': 'FAILED_PRESERVED', 'error': str(exc),
                                              'uploaded': False, 'elapsed_s': time.monotonic() - start})
        raise
    finally:
        os.close(out_fd)
    # The complete marker is final; reporting/SSH errors cannot turn it into FAILED.
    print(json.dumps({'status': result['status'], 'output': str(output), 'complete': result_meta}), flush=True)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build_bundle(args.output)

if __name__ == '__main__':
    main()
