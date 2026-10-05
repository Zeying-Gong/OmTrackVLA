"""Hash-frozen complete-pair reuse; incomplete work is never counted."""
import hashlib,json
from pathlib import Path
from wa.wm.dual_teacher_selection import EXPERIMENT
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_resume(path, expected_sha, verify_artifacts=False):
    if not expected_sha or digest(path)!=expected_sha:raise ValueError('resume plan hash mismatch')
    p=json.loads(Path(path).read_text())
    if p['experiment']!=EXPERIMENT or p['expected_total']!=4215:raise ValueError('resume scope mismatch')
    if digest(p['manifest'])!=p['manifest_sha256']:raise ValueError('manifest changed')
    m=json.loads(Path(p['manifest']).read_text())
    allkeys={(t,e['key']) for t in ('stt','dt','at') for e in m['tasks'][t]['episodes']}
    old=[(r['pair']['task'],r['pair']['key']) for r in p['completed_rows']]
    new=[(r['task'],r['key']) for r in p['remaining']]
    lanes=[(r['task'],r['key']) for lane in p['lanes'] for r in lane]
    if len(old)!=len(set(old)) or len(new)!=len(set(new)) or len(lanes)!=len(set(lanes)):
        raise ValueError('duplicate planned pair')
    if set(old)&set(new) or set(old)|set(new)!=allkeys or len(allkeys)!=4215:
        raise ValueError('resume coverage mismatch')
    if len(p['lanes'])!=8 or set(lanes)!=set(new):raise ValueError('lane coverage mismatch')
    if len(old)!=p['completed_count'] or len(new)!=p['remaining_count']:raise ValueError('plan count mismatch')
    if verify_artifacts:
        for group in ('source_files','artifact_sha256'):
            for filename,h in p[group].items():
                if digest(filename)!=h:raise ValueError('frozen source changed: '+filename)
    return p
