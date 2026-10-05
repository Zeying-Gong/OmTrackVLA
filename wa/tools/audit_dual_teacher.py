"""Audit all paired outcomes and select successful continuous demonstrations.

The release is explicitly evaluation-set adaptation, never heldout-clean data.
Training still requires SE2 cache conversion and student-input boundary validation.
"""
import argparse, hashlib, json, math
from collections import Counter
from pathlib import Path
from wa.wm.dual_teacher_selection import select_teacher, EXPERIMENT
from wa.wm.dual_teacher_pair_evidence import verify_pair_evidence

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def audit(root, expected, resume_plan=None, resume_sha=None, partition_index=0, partition_count=1, partition_roots=None):
    from wa.wm.dual_teacher_partitions import workload
    roots=[Path(p) for p in partition_roots] if partition_roots else [Path(root)]
    if len(set(p.resolve() for p in roots))!=len(roots):raise ValueError('duplicate partition roots')
    if partition_roots and (len(roots)!=partition_count or partition_count<2 or partition_index!=0):
        raise ValueError('partition root count/index mismatch')
    if (partition_count!=1 or partition_index!=0 or partition_roots) and not resume_plan:
        raise ValueError('partition audit requires resume plan')
    files=[f for p in roots for f in sorted(p.glob("lane*/collection/selections.jsonl"))]
    before={str(f):digest(f) for f in files}
    rows=[json.loads(line) for f in files for line in f.read_text().splitlines() if line]
    combined=False
    if resume_plan:
        from wa.wm.dual_teacher_resume import load_resume
        plan=load_resume(resume_plan,resume_sha,verify_artifacts=True)
        jobs=[workload(plan,i,partition_count) for i in range(partition_count)] if partition_roots else [workload(plan,partition_index,partition_count)]
        for dest,job in zip(roots,jobs):
            for i,lane in enumerate(job['lanes']):
                p=dest/f'lane{i}/collection'
                if json.loads((p/'COMPLETE.json').read_text())['pairs']!=len(lane):
                    raise ValueError('incomplete resumed lane')
                local=[json.loads(s) for s in (p/'selections.jsonl').read_text().splitlines() if s]
                localkeys=[(r['pair']['task'],r['pair']['key']) for r in local]
                if len(localkeys)!=len(lane) or set(localkeys)!={(r['task'],r['key']) for r in lane}:
                    raise ValueError('partition lane assignment mismatch')
        newkeys=[(r['pair']['task'],r['pair']['key']) for r in rows]
        needed={(r['task'],r['key']) for job in jobs for r in job['remaining']}
        if len(newkeys)!=len(needed) or set(newkeys)!=needed:raise ValueError('new rows differ from remainder')
        if partition_roots or partition_count==1:
            if needed!={(r['task'],r['key']) for r in plan['remaining']}:raise ValueError('global remainder missing')
            rows=plan['completed_rows']+rows
            combined=True
        else:
            expected=len(needed)
        files += [Path(resume_plan)]
        before[str(resume_plan)]=resume_sha
    keys=[(r["pair"]["task"],r["pair"]["key"]) for r in rows]
    if len(rows)!=expected or len(set(keys))!=expected:raise ValueError("incomplete or duplicate pairs")
    if expected==4215 and Counter(t for t,k in keys)!=dict(stt=1405,dt=1405,at=1405):
        raise ValueError("task counts differ")
    summaries={};accepted=[];rejected=[];choices=Counter();fallback_rejected=[]
    pair_evidence={}
    for task in ("stt","dt","at"):
        taskrows=[r for r in rows if r["pair"]["task"]==task]
        summaries[task]={}
        for teacher in ("lightnav","oracle"):
            results=[r["branches"][teacher]["result"] for r in taskrows]
            n=len(results)
            summaries[task][teacher]=dict(episodes=n,success=sum(bool(r["success"]) for r in results),
                collision=sum(bool(r["collision"]) for r in results),
                macro_following_rate=sum(r["following_rate"] for r in results)/n if n else None)
    for row in rows:
        pair_evidence[row['pair']['task']+':'+row['pair']['key']]=verify_pair_evidence(row)
        check=select_teacher(row["branches"]["lightnav"],row["branches"]["oracle"])
        if any(check[k]!=row[k] for k in ("pair","selected_teacher","reason","results")):
            raise ValueError("selection changed")
        choice=check["selected_teacher"];choices[choice or "neither"]+=1
        for name,b in row["branches"].items():
            p=Path(b["artifact_root"])
            stored=json.loads((p/"result.json").read_text())
            if stored!=b["result"]:raise ValueError("result artifact mismatch")
            if b.get("transport_fallback"):
                if json.loads((p/"fallback_events.json").read_text())!=b["fallback_events"]:
                    raise ValueError("fallback evidence mismatch")
                if json.loads((p/"windows.json").read_text()):raise ValueError("fallback branch contains admitted windows")
            if not stored["success"] and json.loads((p/"windows.json").read_text()):
                raise ValueError("failed teacher contains admitted windows")
        if choice is None:continue
        if not check["demonstration_candidate"]:
            fallback_rejected.append(check["pair"]);continue
        branch=Path(row["branches"][choice]["artifact_root"])
        meta=json.loads((branch/"metadata.json").read_text())
        if meta["experiment"]!=EXPERIMENT or meta["partition"]!="evaluation_adaptation":
            raise ValueError("data scope mismatch")
        if meta["initial_bbox_status"] not in ("VERIFIED_CONFIG_AND_SEMANTIC","VERIFIED_FROZEN_FIRST_RGB_REPAIR"):
            rejected.append(dict(pair=check["pair"],reason="unverified_initial_template"));continue
        obs=json.loads((branch/"observations.json").read_text())
        actions=json.loads((branch/"actions.json").read_text())
        windows=json.loads((branch/"windows.json").read_text())
        if len(obs)!=len(actions) or len(obs)<2:raise ValueError("record count mismatch")
        times=[r["timestamp_s"] for r in obs]
        if not all(math.isfinite(t) for t in times) or not all(b>a for a,b in zip(times,times[1:])):
            raise ValueError("bad recorded timebase")
        for w in windows:
            i=w["current_index"]
            if not 0<=i<len(obs) or times[i]+.7>times[-1]+1e-8:raise ValueError("future extrapolation")
        names=("metadata.json","observations.json","actions.json","windows.json","result.json","pair_start.json")
        accepted.append(dict(task=check["pair"]["task"],key=check["pair"]["key"],teacher=choice,
            branch=str(branch),takeover_step=0,window_indices=[w["current_index"] for w in windows],
            hashes={name:digest(branch/name) for name in names}))
    if before!={str(f):digest(f) for f in files}:raise ValueError('source changed during audit')
    if combined:
        with (Path(root)/'combined_selections.jsonl').open('x') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
    return dict(experiment=EXPERIMENT,paired_outcomes_validated=True,expected=expected,
        persisted_pair_evidence=pair_evidence,persisted_pair_evidence_count=len(pair_evidence),
        partition_index=partition_index,partition_count=partition_count,
        partial_partition=partition_count>1 and not bool(partition_roots),
        partition_roots=[str(p) for p in roots] if partition_roots else None,
        summaries=summaries,choices=dict(choices),teacher_demonstrations=accepted,
        rejected_templates=rejected,rejected_selected_fallback=fallback_rejected,training_released=False,
        pending="SE2 conversion, source-image hashes and real student-loader checks",
        source_files={str(f):digest(f) for f in files})

def main():
    p=argparse.ArgumentParser();p.add_argument("root");p.add_argument("--expected",type=int,default=4215)
    p.add_argument("--output",required=True);p.add_argument("--resume-plan");p.add_argument("--resume-sha")
    p.add_argument("--partition-index",type=int,default=0);p.add_argument("--partition-count",type=int,default=1)
    p.add_argument("--partition-root",action="append",default=None)
    a=p.parse_args()
    result=audit(a.root,a.expected,a.resume_plan,a.resume_sha,a.partition_index,a.partition_count,a.partition_root)
    with Path(a.output).open("x") as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(pairs=result['expected'],choices=result["choices"],demonstrations=len(result["teacher_demonstrations"]),rejected_templates=len(result["rejected_templates"]))))
if __name__=="__main__":main()
