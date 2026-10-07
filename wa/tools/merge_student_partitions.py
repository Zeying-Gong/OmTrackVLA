"""Merge three completed student task partitions without reusing baseline rows."""
import argparse,hashlib,json
from pathlib import Path
from wa.wm.full_mixed_contract import MANIFEST_SHA,validate_ready,summarize,validate_mode
from wa.wm.student_eval_contract import validate_student_rows
from wa.wm.student_eval_finalize import validate_shard,finalize

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_partitions(roots,manifest,contract,*,mode='mixed'):
    validate_mode(mode)
    if set(roots)!={'stt','dt','at'} or len({Path(p).resolve() for p in roots.values()})!=3:
        raise ValueError('three distinct task roots required')
    rows=[];hashes={}
    for task in ('stt','dt','at'):
        root=Path(roots[task])
        complete_path=root/'PARTITION_COMPLETE.json'
        report=json.loads(complete_path.read_text())
        if report.get('status')!='COMPLETE_STUDENT_TASK_PARTITION_NOT_GLOBAL_AUDIT' or report.get('task')!=task or report.get('episodes')!=1405:
            raise ValueError('incomplete or wrong task partition')
        if report.get('mode','mixed' if mode=='mixed' else None)!=mode:
            raise ValueError('wrong or missing partition input mode')
        if (report.get('checkpoint_sha256'),report.get('checkpoint_step'))!=(contract['checkpoint_sha'],contract['step']):
            raise ValueError('foreign partition checkpoint')
        path=root/'combined_episodes.jsonl'
        if digest(path)!=report['combined_sha256']:raise ValueError('partition rows changed')
        part=[json.loads(s) for s in path.read_text().splitlines() if s]
        validate_student_rows(part,contract,(task,),mode=mode)
        shards=[]
        for i in range(8):
            dest=root/f'shard_{i:02d}'
            complete=json.loads((dest/'COMPLETE.json').read_text())
            ready=json.loads((dest/'server_ready.json').read_text());validate_ready(ready,mode=mode,**contract)
            local=[json.loads(s) for s in (dest/'episodes.jsonl').read_text().splitlines() if s]
            validate_shard(local,complete,manifest,i,(task,),mode=mode)
            for r in local:
                r=dict(r,artifact_root=str(dest));shards.append(r)
            for f in (dest/'COMPLETE.json',dest/'server_ready.json',dest/'episodes.jsonl'):
                hashes[str(f)]=digest(f)
        canonical=lambda rs:{(r['task'],r['key']):r for r in rs}
        if len(shards)!=1405 or canonical(shards)!=canonical(part):
            raise ValueError('combined rows differ from original shards')
        rows.extend(part)
        hashes[str(path)]=digest(path);hashes[str(complete_path)]=digest(complete_path)
    validate_student_rows(rows,contract,mode=mode)
    for path,expected in hashes.items():
        if digest(path)!=expected:raise ValueError('source changed during merge')
    return rows,hashes

def main():
    p=argparse.ArgumentParser()
    for name in ('stt','dt','at','manifest','teacher-selections','checkpoint-sha','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--mode',choices=['mixed','image'],default='mixed')
    p.add_argument('--checkpoint-step',required=True,type=int);a=p.parse_args()
    if digest(a.manifest)!=MANIFEST_SHA:raise ValueError('manifest changed')
    manifest=json.loads(Path(a.manifest).read_text())
    contract=dict(checkpoint_sha=a.checkpoint_sha,step=a.checkpoint_step)
    roots={t:getattr(a,t) for t in ('stt','dt','at')}
    rows,hashes=read_partitions(roots,manifest,contract,mode=a.mode)
    report=summarize(rows,manifest,mode=a.mode,**contract)
    comparison,pairs=finalize(rows,manifest,a.teacher_selections);report.update(comparison)
    report.update(experiment='evaluation_set_adaptation_v1',partition_roots=roots,
                  source_hashes=hashes,new_episodes=4215,reused_baseline_episodes=0)
    report['limits'].append('Student trained on this evaluation set; not unseen generalization')
    for path,expected in hashes.items():
        if digest(path)!=expected:raise ValueError('source changed during final audit')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    (out/'combined_episodes.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    (out/'student_teacher_pair_audit.json').write_text(json.dumps(pairs,allow_nan=False))
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(dict(status='MERGED_AND_PAIRED',episodes=4215,superiority=report['superiority']),allow_nan=False))
if __name__=='__main__':main()
