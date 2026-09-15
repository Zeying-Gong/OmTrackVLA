"""Extract selected text blobs using Git plumbing; never execute repo code."""
from pathlib import Path
import hashlib, json, subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE/'LightNav-0'
DEST = HERE/'source_snapshot'
revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO).decode().strip()
tree = subprocess.check_output(['git', 'ls-tree', '-r', '-z', 'HEAD'], cwd=REPO)
selected = []
for record in tree.split(b'\0'):
    if not record:
        continue
    meta, rawpath = record.split(b'\t', 1)
    mode, kind, oid = meta.decode().split()
    name = rawpath.decode('utf-8')
    if kind != 'blob' or mode not in ('100644', '100755'):
        continue
    if name.startswith(('mujoco_demo/vln_mujoco/assets/', 'docs/assets/')):
        continue
    if Path(name).suffix.lower() not in ('.py', '.md', '.toml', '.yaml', '.yml', '.sh', '.patch', '.cfg', '.xml', '.txt', '.json', '.js', '.html', '.css') and name not in ('LICENSE', 'Dockerfile', 'Makefile', '.gitignore', '.dockerignore'):
        continue
    selected.append((name, oid))
DEST.mkdir(exist_ok=True)
proc = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=REPO, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
files = {}
try:
    for name, oid in selected:
        proc.stdin.write((oid+'\n').encode())
        proc.stdin.flush()
        header = proc.stdout.readline().decode().strip().split()
        assert len(header) == 3 and header[0] == oid and header[1] == 'blob'
        data = proc.stdout.read(int(header[2]))
        assert len(data) == int(header[2]) and proc.stdout.read(1) == b'\n'
        target = DEST/name
        assert target.resolve().is_relative_to(DEST.resolve()) and ':' not in name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        files[name] = {'git_blob': oid, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
finally:
    proc.stdin.close()
    code = proc.wait(timeout=10)
assert code == 0
manifest = {'repository': 'https://github.com/lightorigins/LightNav-0', 'commit': revision,
            'method': 'git shallow clone objects + ls-tree + cat-file blob extraction; no repository code executed',
            'checkout_limitation': 'Windows checkout and archive rejected colon-containing MuJoCo asset names; those assets excluded from text source extraction',
            'files': files}
(HERE/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'commit': revision, 'text_files_extracted': len(files), 'destination': str(DEST)}))
