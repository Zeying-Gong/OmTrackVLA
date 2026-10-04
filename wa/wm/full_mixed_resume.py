"""CPU-only immutable completion snapshot and eight-lane remainder contract."""
import hashlib,json,math
from pathlib import Path
from wa.wm.full_mixed_contract import TASKS,MANIFEST_SHA,validate_ready

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_plan(plan,manifest,verify_files=False):
    assert plan['manifest_sha256']==MANIFEST_SHA
    expected={(t,e['key']) for t in TASKS for e in manifest['tasks'][t]['episodes']}
    assert len(expected)==4215
    prior=plan['completed_rows']; done=[(r['task'],r['key']) for r in prior]
    todo=[(e['task'],e['key']) for lane in plan['lanes'] for e in lane]
    assert len(plan['lanes'])==8 and len(set(done))==len(done) and len(set(todo))==len(todo)
    assert not set(done)&set(todo) and set(done)|set(todo)==expected
    for r in prior:
        assert (r['mode'],r['noise_mode'],r['controller'])==('mixed','zero','learned_yaw_guard_v1')
        assert type(r['policy_init_valid']) is bool
        assert r['success'] in (0,1) and r['collision'] in (0,1)
        assert r['policy_init_valid'] or r['success']==0
        assert len(r['initial_rgb_sha256'])==64
        for k in ('success','collision','following_rate','following_step','total_step'):
            assert math.isfinite(float(r[k]))
    if verify_files:
        for f in plan['source_files']: assert digest(f['path'])==f['sha256'],f['path']
    return plan

def build_plan(root,manifest):
    root=Path(root);rows=[];files=[]
    for p in sorted(root.glob('shard_*/episodes.jsonl')):
        raw=p.read_text()
        assert not raw or raw.endswith('\n'),'incomplete row: '+str(p)
        validate_ready(json.loads((p.parent/'server_ready.json').read_text()))
        files.append(dict(path=str(p),sha256=digest(p)))
        for line in raw.splitlines():
            row=json.loads(line);row['artifact_root']=str(p.parent)
            rows.append(row)
    done={(r['task'],r['key']) for r in rows};lanes=[[] for _ in range(8)]
    for t in TASKS:
        for e in manifest['tasks'][t]['episodes']:
            if (t,e['key']) not in done:
                lane=min(range(8),key=lambda i:len(lanes[i]))
                lanes[lane].append(dict(task=t,key=e['key']))
    plan=dict(version=1,parent_job=60885,parent_task=71808,parent_root=str(root),
        manifest_sha256=MANIFEST_SHA,completed_rows=rows,source_files=files,lanes=lanes)
    return validate_plan(plan,manifest,True)

def load_plan(path,manifest,expected_sha):
    assert digest(path)==expected_sha,'resume plan hash mismatch'
    return validate_plan(json.loads(Path(path).read_text()),manifest,True)

def verify_new_rows(rows,plan):
    expected={(e['task'],e['key']) for lane in plan['lanes'] for e in lane}
    actual=[(r['task'],r['key']) for r in rows]
    assert len(actual)==len(set(actual)) and set(actual)==expected,'new result missing/duplicate/unplanned'

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    assert digest(a.manifest)==MANIFEST_SHA
    result=build_plan(a.parent,json.loads(Path(a.manifest).read_text()))
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(completed=len(result['completed_rows']),remaining=sum(map(len,result['lanes'])),lanes=list(map(len,result['lanes'])),sha256=digest(a.output))))
