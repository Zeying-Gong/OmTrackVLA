"""Fail-closed full student pairing and per-task superiority report."""
import hashlib,json
from pathlib import Path
from wa.wm.student_pair_audit import audit_starts
from wa.wm.dual_teacher_metrics import summarize_teachers

TEACHER_SHA='f45f21d71b63d7f2a0898e1e762c393a95b0f3eaf8291bd1affbc827d23354d8'

def load_teachers(path):
    data=Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest()!=TEACHER_SHA:
        raise ValueError('teacher reference differs from frozen full collection')
    return [json.loads(line) for line in data.splitlines() if line]

def validate_shard(rows,complete,manifest,index,tasks=('stt','dt','at')):
    expected={(task,e['key']) for task in tasks
              for e in manifest['tasks'][task]['episodes'] if e['shard']==index}
    keys=[(r['task'],r['key']) for r in rows]
    if len(keys)!=len(expected) or set(keys)!=expected:
        raise ValueError('student shard differs from frozen assignment')
    if complete.get('episodes')!=len(keys):
        raise ValueError('student shard COMPLETE count differs')

def superiority(rows,teachers):
    comparisons={}
    for task in ('stt','dt','at'):
        s={r['key']:r for r in rows if r['task']==task}
        t={r['pair']['key']:r['results']['lightnav'] for r in teachers if r['pair']['task']==task}
        if len(s)!=1405 or s.keys()!=t.keys():raise ValueError('full task pairing required')
        student_count=sum(r['success'] for r in s.values())
        teacher_count=sum(r['success'] for r in t.values())
        comparisons[task]=dict(episodes=len(s),student_success=student_count,
            lightnav_success=teacher_count,strictly_exceeds=student_count>teacher_count,
            SR_delta_percentage_points=100*(student_count-teacher_count)/len(s),
            student_only_success=sum(bool(s[k]['success']) and not bool(t[k]['success']) for k in s),
            lightnav_only_success=sum(bool(t[k]['success']) and not bool(s[k]['success']) for k in s))
    return dict(all_three_strictly_exceed=all(r['strictly_exceeds'] for r in comparisons.values()),
                tasks=comparisons,
                scope='evaluation-set adaptation; different input systems; not unseen generalization')

def finalize(rows,manifest,teacher_path):
    teachers=load_teachers(teacher_path)
    expected=[(t,e['key']) for t in ('stt','dt','at') for e in manifest['tasks'][t]['episodes']]
    if len(expected)!=4215:raise ValueError('complete4215 manifest required')
    teacher_report=summarize_teachers(teachers,manifest)
    pair_audit=audit_starts(rows,teachers,expected)
    # Recheck the reference after all persisted evidence has been read.
    load_teachers(teacher_path)
    return dict(teacher_reference_sha256=TEACHER_SHA,
                teacher_metrics_percent=teacher_report['metrics_percent'],
                superiority=superiority(rows,teachers)),pair_audit
