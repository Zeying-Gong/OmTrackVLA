"""Bounded developer audit of recovery-label fit, not closed-loop efficacy."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from wa.wm.robot_data import RobotWorldData
from wa.wm.training import JointRobotModel
from wa.wm.loaders import sha

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.root);out=Path(a.output)
    if out.exists():raise ValueError('preserve existing audit')
    torch.set_num_threads(2);torch.cuda.set_device(0)
    cache=root/'artifacts/recovery_se2_cache_v2'
    ds=RobotWorldData(cache,'train',cache/'index')
    epids=np.asarray(ds.episode)[ds.rows]
    indices=[]
    for ep in sorted(set(epids)):
        candidates=np.flatnonzero(epids==ep)
        # Equal scene weight and include earliest teacher-owned takeover windows.
        picks=sorted(set([0,min(1,len(candidates)-1),len(candidates)//2,len(candidates)-1]))
        indices.extend(int(candidates[k]) for k in picks)
    base=RobotWorldData('/data/nas_ray/home/zeying.gong/datasets/wla_evt_se2_cache_20260925_v2',
       'heldout',root/'artifacts/robot_transition_audit_v2')
    held=list(map(int,np.linspace(0,len(base)-1,44,dtype=int)))
    selected={'recovery':(ds,indices),'heldout':(base,held)}
    model=JointRobotModel(str(root),'/data/nas_ray/home/zeying.gong/datasets_processed/threepanel_debug_v2/traversability_stepp_v1/_weights/dinov2_vits14_pretrain.pth',
      str(root/'dependencies/wla_v1'),'/data/nas_ray/project/md-ak/users/zeying.gong/job_58346/task_69086/wla_teacher_train/training/checkpoints/step-0043203.pt','jepa').cuda().eval()
    parents={'parent59866':'/data/nas_ray/project/md-ak/users/zeying.gong/job_59866/task_70728/wa_history_repeat_v1/checkpoint.pt',
      'recovery60502':'/data/nas_ray/project/md-ak/users/zeying.gong/job_60502/task_71381/wa_recovery_mix_a800_v1/checkpoint.pt'}
    result={'status':'DEVELOPER_PAIRED_LABEL_FIT_ONLY','selection':'equal4/episode recovery; evenly spaced44 heldout; mixed only; deterministic zero flow start','models':{}}
    start=time.monotonic()
    for name,cp in parents.items():
        ck=torch.load(cp,map_location='cpu',weights_only=True,mmap=True)
        load=model.load_state_dict(ck['model'],strict=False)
        assert not load.unexpected_keys and all(x.startswith('encoder.') for x in load.missing_keys)
        group={}
        for label,(data,rows) in selected.items():
            records=[]
            for i in range(0,len(rows),2):
                ids=rows[i:i+2];b={k:v.cuda() for k,v in default_collate([data[x] for x in ids]).items()}
                for repeat in (False,True):
                    bb=dict(b)
                    if repeat:bb['rgb']=b['rgb'][:,-1:].expand_as(b['rgb']);bb['times']=torch.zeros_like(b['times'])
                    with torch.no_grad():
                        pred,_=model.predict(bb,torch.full((len(ids),),2,device='cuda'),torch.zeros_like(bb['pose']))
                    error=torch.linalg.vector_norm(pred[...,:2]-b['pose'][...,:2],dim=-1)
                    yaw=torch.atan2(pred[...,2],pred[...,3])-torch.atan2(b['pose'][...,2],b['pose'][...,3])
                    ye=torch.atan2(yaw.sin(),yaw.cos()).abs()
                    for j,ix in enumerate(ids):
                        raw=int(data.rows[ix]);records.append({'index':ix,'episode':int(data.episode[raw]),'repeat_history':repeat,
                           'ADE':error[j].mean().item(),'FDE':error[j,-1].item(),'yaw_MAE':ye[j].mean().item(),
                           'label_first':b['pose'][j,0].tolist(),'pred_first':pred[j,0].tolist()})
            group[label]=records
            print(name,label,'windows',len(records),'elapsed',time.monotonic()-start,flush=True)
        result['models'][name]={'sha256':sha(cp),'step':ck['step'],'records':group}
        del ck
    result['elapsed_s']=time.monotonic()-start
    out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps(result,indent=2))
    for name,m in result['models'].items():
        for split,rs in m['records'].items():
            for repeat in (False,True):
                s=[r for r in rs if r['repeat_history']==repeat]
                print(name,split,repeat,{k:float(np.mean([r[k] for r in s])) for k in ('ADE','FDE','yaw_MAE')},flush=True)
if __name__=='__main__':main()
