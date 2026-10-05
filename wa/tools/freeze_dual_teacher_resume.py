"""Freeze stopped collection's complete pairs; never reuse partial branches."""
import argparse,json,subprocess
from pathlib import Path
from wa.tools.audit_dual_teacher import audit,digest
from wa.wm.dual_teacher_selection import EXPERIMENT

def freeze(root, manifest, output, job, previous_plan=None, previous_sha=None):
    root,manifest,output=map(Path,(root,manifest,output))
    if f'job_{job}' not in root.parts:raise ValueError('job/root mismatch')
    tool='/data/nas_ray/home/zeying.gong/algorithm/envs/md_ai_kit_submit/bin/md_ai_kit'
    detail=subprocess.check_output([tool,'detail','--job',str(job)],text=True)
    if 'Status        : STOPPED' not in detail:raise ValueError('source job must be confirmed STOPPED')
    if bool(previous_plan)!=bool(previous_sha):raise ValueError('previous plan/hash required together')
    prior=None
    if previous_plan:
        from wa.wm.dual_teacher_resume import load_resume
        prior=load_resume(previous_plan,previous_sha,verify_artifacts=True)
        if digest(manifest)!=prior['manifest_sha256']:raise ValueError('previous manifest differs')
    sources=sorted(root.glob('lane*/collection/selections.jsonl'))
    before={str(p):digest(p) for p in sources}
    rows=[json.loads(line) for p in sources for line in p.read_text().splitlines() if line]
    evidence=audit(root,len(rows))
    new_count=len(rows)
    if prior:
        newkeys={(r['pair']['task'],r['pair']['key']) for r in rows}
        allowed={(r['task'],r['key']) for r in prior['remaining']}
        if not newkeys.issubset(allowed):raise ValueError('new completed rows outside prior remainder')
        rows=prior['completed_rows']+rows
        sources.append(Path(previous_plan));before[str(previous_plan)]=previous_sha
    if len({(r['pair']['task'],r['pair']['key']) for r in rows})!=len(rows):
        raise ValueError('duplicate chained completion')
    completed={(r['pair']['task'],r['pair']['key']) for r in rows}
    m=json.loads(manifest.read_text());allkeys=[]
    for task in ('stt','dt','at'):
        spec=m['tasks'][task]
        if digest(spec['path'])!=spec['sha256']:raise ValueError('dataset changed')
        if len(spec['episodes'])!=1405:raise ValueError('unexpected task size')
        allkeys += [(task,e['key']) for e in spec['episodes']]
    if len(set(allkeys))!=4215 or not completed.issubset(allkeys):raise ValueError('invalid completed keys')
    remaining=[dict(task=t,key=k) for t,k in allkeys if (t,k) not in completed]
    artifacts={}
    for r in rows:
        for branch in r['branches'].values():
            p=Path(branch['artifact_root'])
            for name in ('metadata.json','observations.json','actions.json','windows.json','result.json','pair_start.json'):
                artifacts[str(p/name)]=digest(p/name)
    if before!={str(p):digest(p) for p in sources}:raise ValueError('source changed while freezing')
    plan=dict(experiment=EXPERIMENT,source_job=job,source_root=str(root),source_status=detail,
        manifest=str(manifest),manifest_sha256=digest(manifest),
        completed_rows=rows,completed_count=len(rows),remaining=remaining,remaining_count=len(remaining),
        lanes=[remaining[i::8] for i in range(8)],source_files=before,artifact_sha256=artifacts,
        previous_audit=evidence,expected_total=4215,training_released=False,
        prior_plan=str(previous_plan) if prior else None,prior_plan_sha256=previous_sha,new_completed_count=new_count,
        note='Only complete paired selections reused. Partial branches and developer samples excluded. Released LightNav fallback scores retained by resumed protocol; selected fallback branches never demonstrations.')
    with output.open('x') as f:json.dump(plan,f,indent=2)
    return dict(completed=len(rows),remaining=len(remaining),lanes=[len(x) for x in plan['lanes']],sha256=digest(output))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','manifest','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--job',type=int,required=True)
    p.add_argument('--previous-plan');p.add_argument('--previous-sha');a=p.parse_args()
    print(json.dumps(freeze(a.root,a.manifest,a.output,a.job,a.previous_plan,a.previous_sha),indent=2))
