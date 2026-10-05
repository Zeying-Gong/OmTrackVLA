"""Offline student/teacher start audit; never supplies state to a policy."""
import hashlib
import json
import math
from pathlib import Path
from wa.wm.dual_teacher_pair_evidence import verify_pair_evidence
from wa.wm.recovery_replay import check_dynamic_state

def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_tree(v) for v in value)
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

def audit_starts(students, teachers, expected_keys):
    """Require exact caller-supplied manifest coverage and persisted teacher evidence."""
    expected=list(expected_keys)
    if not expected or len(set(expected))!=len(expected):
        raise ValueError('empty or duplicate manifest')
    def index(rows, teacher=False):
        out={}
        for row in rows:
            identity=row['pair'] if teacher else row
            key=(identity['task'],identity['key'])
            if key in out:raise ValueError('duplicate start key')
            out[key]=row
        if set(out)!=set(expected):raise ValueError('start coverage differs from manifest')
        return out
    student=index(students);teacher=index(teachers,True);evidence={}
    for key in sorted(expected):
        s=student[key].get('initial_pair_evidence')
        if not isinstance(s,dict) or not finite_tree(s.get('state')):
            raise ValueError('missing or nonfinite student start state')
        t=teacher[key]
        if t['pair'].get('seed')!=7 or t['pair'].get('takeover_step')!=0:
            raise ValueError('teacher is not canonical seed7 start')
        hashes=verify_pair_evidence(t)
        if s.get('rgb')!=t['pair']['initial_rgb_sha256']:
            raise ValueError('student raw RGB differs from teachers')
        for name in ('lightnav','oracle'):
            path=Path(t['branches'][name]['artifact_root'])/'pair_start.json'
            data=path.read_bytes()
            if hashlib.sha256(data).hexdigest()!=hashes[name][str(path)]:
                raise ValueError('teacher start changed during student comparison')
            start=json.loads(data)
            if not finite_tree(start['state']):raise ValueError('nonfinite teacher state')
            check_dynamic_state(start['state'],s['state'])
        evidence[':'.join(key)]=hashes
    return dict(status='PASS',episodes=len(expected),teacher_artifact_hashes=evidence,
                comparison='raw sensor SHA with shape/dtype/all channels; dynamic state atol1e-6',
                scope='initial pairing only; not student performance or generalization')
