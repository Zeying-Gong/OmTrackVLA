"""Opt-in new-student full evaluation: no old rows and no implicit weights."""
from wa.wm.full_mixed_contract import CHECKPOINT_SHA,validate_checkpoint_identity
from wa.wm.initial_bbox_repair import KEYS,VERSION
REPAIR_SHA='6535f7b9a53e399ba7f2323c2d7f72d223146ee77f090223eb260f5885f6269a'

def model_contract(env):
    enabled=env.get('WA_STUDENT_EVAL')
    names=('WA_EVAL_CHECKPOINT_SHA','WA_EVAL_CHECKPOINT_STEP')
    if not enabled:
        if any(env.get(k) for k in names):
            raise ValueError('student identity requires explicit student evaluation')
        return {}
    if enabled!='evaluation_set_adaptation_v1':raise ValueError('unknown student experiment')
    if any(env.get(k) for k in ('WA_RESUME_PLAN','WA_TARGETED_PLAN')):
        raise ValueError('new student cannot reuse old model rows')
    h=env.get(names[0]);raw=env.get(names[1],'')
    if not raw.isdigit():raise ValueError('explicit student step required')
    step=int(raw);validate_checkpoint_identity(h,step)
    if h==CHECKPOINT_SHA:raise ValueError('old baseline is not a new student')
    if env.get('WA_SEMANTIC_PLY_FIX')!='mp3d_semantic_ply_v1':
        raise ValueError('student requires fixed semantic protocol')
    if not env.get('WA_INIT_REPAIR_PLAN') or env.get('WA_INIT_REPAIR_PLAN_SHA')!=REPAIR_SHA:
        raise ValueError('student requires frozen seven-bbox plan')
    return dict(checkpoint_sha=h,step=step)

def validate_student_rows(rows,contract,tasks=('stt','dt','at')):
    if not contract:return
    seen=set();repaired=set()
    for r in rows:
        if r['task'] not in tasks:raise ValueError('foreign task in student partition')
        key=(r['task'],r['key'])
        if key in seen:raise ValueError('duplicate student row')
        seen.add(key)
        if (r.get('checkpoint_sha256'),r.get('checkpoint_step'))!=(contract['checkpoint_sha'],contract['step']):
            raise ValueError('foreign or missing student identity')
        if r.get('semantic_protocol')!='mp3d_semantic_ply_v1':raise ValueError('wrong semantics')
        if key in KEYS:
            if r.get('initialization_repair')!=VERSION or r.get('initialization_repair_plan_sha256')!=REPAIR_SHA:
                raise ValueError('missing frozen first-frame repair')
            repaired.add(key)
        elif r.get('initialization_repair'):
            raise ValueError('unexpected repair outside seven keys')
    if repaired!={k for k in KEYS if k[0] in tasks}:raise ValueError('incomplete seven-key coverage')
