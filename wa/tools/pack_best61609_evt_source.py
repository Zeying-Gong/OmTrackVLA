"""Private current EVT source/config and Xvfb snapshot; no upload or extraction."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time

ROOT = Path('/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928')
REPOS = ROOT.parent
WLA = REPOS/'WLA-EVT-20260925'
BENCH = REPOS/'OmTrackVLA-da3-polar-20260924'
OUTPUT = ROOT/'artifacts/best61609_evt_source_bundle_20261009_v1'
HELPERS = {'pack_best61609_assets.py': 'de437ffe79315583ec02269604df135b1d0d514080dc9954dd138759c3432876',
           'best61609_asset_manifest.py': '6027b97b0d6961f4a14ca26e07ea98f34cb1ef7cb0d793737a4a61fec6ef67b4'}
PIN_LISTS = {'closed_loop_dependencies.sha256': '33a501d7ab3bbb4bcb248c625fcad8eee3f2cccab723db47c20697282899e991',
             'full_eval_dependencies.sha256': '5e870e9d6fdbe36bda9df47b134bce08a6fbc278bfa5a5990b55de2a6175db08'}
WLA_FILES = ('evt_text_action_v2/__init__.py', 'evt_text_action_v2/control.py',
    'evt_text_action_v3/__init__.py', 'evt_text_action_v3/control.py', 'evt_text_action_v3/eval_agent.py',
    'evt_full_20260926/__init__.py', 'evt_full_20260926/common.py', 'evt_full_20260926/manifest.json')
BENCH_FILES = ('trained_agent.py', 'humanoid_infos.json', 'model.py', 'cache_gridpool.py', 'hf_compat.py',
    'open_trackvla_hf/__init__.py', 'open_trackvla_hf/configuration_open_trackvla.py',
    'open_trackvla_hf/modeling_open_trackvla.py', 'scripts/da3/run_evt_ten.py',
    'scripts/runtime/install_xvfb_bundle.sh', 'scripts/runtime/run_xvfb.sh')

def tree_files(root):
    """Reject links/special files instead of silently skipping them."""
    from wa.tools.pack_best61609_assets import open_directory
    result = []
    def visit(fd, prefix):
        for name in sorted(os.listdir(fd)):
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            path = prefix/name
            if stat.S_ISLNK(info.st_mode):
                raise ValueError('symlink in source tree: '+str(path))
            if stat.S_ISDIR(info.st_mode):
                if name == '__pycache__':
                    continue
                child = os.open(name, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW, dir_fd=fd)
                try:
                    visit(child, path)
                finally:
                    os.close(child)
            elif stat.S_ISREG(info.st_mode):
                result.append(path)
            else:
                raise ValueError('special file in source tree: '+str(path))
    with open_directory(root) as fd:
        visit(fd, Path(root))
    return result

def selected_paths():
    groups = {
        'wla_eval': [WLA/p for p in WLA_FILES],
        'bench_bridge': [BENCH/p for p in BENCH_FILES],
        'evt_bench': [p for p in tree_files(BENCH/'evt_bench') if p.suffix == '.py'],
        'habitat_lab': tree_files(BENCH/'habitat-lab'),
        'xvfb_debs': [p for p in tree_files(BENCH/'artifacts/xvfb_bundle') if p.suffix == '.deb'],
    }
    expected = {'wla_eval':8, 'bench_bridge':11, 'evt_bench':6, 'habitat_lab':314, 'xvfb_debs':34}
    if {k:len(v) for k,v in groups.items()} != expected:
        raise ValueError('source inventory changed; review before packaging')
    return sorted((str(p), kind) for kind, paths in groups.items() for p in paths)

def pinned_dependencies():
    from wa.tools.pack_best61609_assets import open_source
    pins = {}
    for name, expected in PIN_LISTS.items():
        with open_source(str(ROOT/'checkout/wa/wm'/name)) as stream:
            data = stream.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('historical dependency list changed')
        for line in data.decode().splitlines():
            match = re.fullmatch(r'([0-9a-f]{64})  (/[^\n]+)', line)
            if not match or match[2] in pins:
                raise ValueError('invalid or duplicate historical dependency')
            pins[match[2]] = match[1]
    if len(pins) != 10:
        raise ValueError('expected ten historical dependencies')
    return pins

def entries_for(paths, pins):
    from wa.tools.pack_best61609_assets import canonical_absolute, open_source, signature
    result, seen = [], set()
    for source, kind in paths:
        canonical_absolute(source)
        if source in seen:
            raise ValueError('duplicate source')
        seen.add(source)
        with open_source(source) as stream:
            before = os.fstat(stream.fileno())
            data = stream.read()
            if len(data) != before.st_size or signature(before) != signature(os.fstat(stream.fileno())):
                raise ValueError('source changed during inventory')
        digest = hashlib.sha256(data).hexdigest()
        if source in pins and digest != pins[source]:
            raise ValueError('historical dependency hash mismatch')
        if kind != 'xvfb_debs' and re.search(rb'(?:-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|hf_[A-Za-z0-9]{24,}|ms-[a-f0-9]{8}-[a-f0-9-]{27,})', data):
            raise ValueError('potential credential material; manual review required')
        result.append(dict(source=source, archive_member=source[1:], size=len(data),
                           sha256=digest, package=kind, historical_pin=source in pins))
    if not set(pins).issubset(seen):
        raise ValueError('historical dependency missing from archive')
    return sorted(result, key=lambda item:item['archive_member'])

def main():
    here = Path(__file__).resolve().parent
    for name, expected in HELPERS.items():
        if (here/name).is_symlink() or hashlib.sha256((here/name).read_bytes()).hexdigest() != expected:
            raise ValueError('packaging helper identity mismatch')
    from wa.tools.pack_best61609_assets import open_directory, pack_archive, verify_archive, write_json_new
    start = time.monotonic()
    paths, pins = selected_paths(), pinned_dependencies()
    entries = entries_for(paths, pins)
    if len(entries) != 373 or sum(e['size'] for e in entries) != 16816185:
        raise ValueError('source bytes or count changed; review before packaging')
    with open_directory(ROOT/'artifacts') as parent_fd:
        os.mkdir(OUTPUT.name, 0o700, dir_fd=parent_fd)
        out_fd = os.open(OUTPUT.name, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW, dir_fd=parent_fd)
    try:
        manifest = dict(schema='wa_best61609_private_evt_source_v1', files=entries,
            source_identity='Current source snapshot; only ten existing dependency pins certify historical identity.',
            historical_pins=pins, historical_pin_lists=PIN_LISTS,
            public_redistribution_authorized=False, uploaded=False, h100_validated=False,
            archive_member_mode='0600',
            executable_restore_allowlist=[str(BENCH/'scripts/runtime/run_xvfb.sh')[1:]],
            restore_note='After verification and isolated extraction only, restore owner execute on run_xvfb.sh; launcher executes this file directly. No other executable restoration is inferred.',
            excluded=['Python/binary environments', 'weights', 'scenes', 'humanoids', 'robots',
                      'bytecode', 'credentials', 'training caches'],
            license_note='Preserve original headers/metadata. No independent LICENSE in this snapshot; do not infer public redistribution rights.',
            credential_scan='Bounded obvious text patterns only; not an exhaustive secret scan.')
        manifest_meta = write_json_new(out_fd, 'manifest.json', manifest)
        name = 'best61609_evt_source.tar'
        pack_archive(out_fd, name, entries)
        archive = verify_archive(out_fd, name, entries)
        if selected_paths() != paths or entries_for(paths, pinned_dependencies()) != entries:
            raise ValueError('source inventory changed after packaging')
        result = dict(status='PASS_PRIVATE_EVT_SOURCE_BUNDLE_NOT_UPLOADED',
            created_utc=datetime.now(timezone.utc).isoformat(), elapsed_s=time.monotonic()-start,
            files=373, source_config_files=339, xvfb_debs=34, payload_bytes=16816185,
            historical_pinned_files=10, entire_snapshot_historically_verified=False,
            archive=archive, manifest=manifest_meta, uploaded=False, h100_validated=False,
            contains_environment=False, contains_weights=False, contains_scenes=False,
            contains_humanoid_or_robot_assets=False, public_redistribution_authorized=False,
            builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), helper_sha256=HELPERS,
            note='Private original-byte snapshot, not a complete portable runtime; do not rerun or overwrite.')
        os.fsync(out_fd)
        complete = write_json_new(out_fd, 'complete.json', result)
    except Exception as exc:
        write_json_new(out_fd, 'failed.json', dict(status='FAILED_PRESERVED', error=str(exc), uploaded=False))
        raise
    finally:
        os.close(out_fd)
    print(json.dumps(dict(status=result['status'], output=str(OUTPUT), complete=complete, archive=archive)), flush=True)

if __name__ == '__main__':
    main()
