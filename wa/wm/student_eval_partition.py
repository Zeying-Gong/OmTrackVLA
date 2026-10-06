"""Disjoint full task partitions: each of STT/DT/AT owns eight GPU shards."""
import hashlib,json
from pathlib import Path
from wa.wm.student_eval_contract import validate_student_rows

def task_scope(env):
    task=env.get('WA_EVAL_TASK')
    if task is None:return ('stt','dt','at')
    if env.get('WA_STUDENT_EVAL')!='evaluation_set_adaptation_v1' or task not in ('stt','dt','at'):
        raise ValueError('task partition requires declared student experiment and valid task')
    return (task,)

def write_partition(root,rows,manifest,contract,task):
    if task not in ('stt','dt','at') or not contract:raise ValueError('invalid student partition')
    validate_student_rows(rows,contract,(task,))
    expected={e['key'] for e in manifest['tasks'][task]['episodes']}
    if len(rows)!=1405 or len(expected)!=1405 or {r['key'] for r in rows}!=expected:
        raise ValueError('incomplete task partition')
    payload=''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows)
    root=Path(root)
    path=root/'combined_episodes.jsonl';path.write_text(payload)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    report=dict(status='COMPLETE_STUDENT_TASK_PARTITION_NOT_GLOBAL_AUDIT',task=task,
        episodes=len(rows),checkpoint_sha256=contract['checkpoint_sha'],checkpoint_step=contract['step'],
        combined_sha256=digest,global_pair_audit_pending=True)
    (root/'PARTITION_COMPLETE.json').write_text(json.dumps(report,indent=2,allow_nan=False))
