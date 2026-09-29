import json, hashlib
from pathlib import Path
import numpy as np

base=Path('/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928')
cache=Path('/data/nas_ray/home/zeying.gong/datasets/wla_evt_se2_cache_20260925_v2')
out={'protocol':'64 episodes per split selected by SHA256 root; all audited rows in selected episodes; no RGB or training; not full distribution','splits':{}}
for part in ('train','heldout'):
    entries=json.loads((cache/f'{part}_episodes.json').read_text())
    valid=np.load(base/f'artifacts/robot_transition_audit_v2/{part}_valid.npy')
    ep=np.load(cache/f'{part}_episode.npy',mmap_mode='r')
    hist=np.load(cache/f'{part}_history.npy',mmap_mode='r')
    pose=np.load(cache/f'{part}_pose.npy',mmap_mode='r')
    eligible=np.unique(ep[valid]).tolist()
    chosen=sorted(eligible,key=lambda i:hashlib.sha256(entries[i]['root'].encode()).hexdigest())[:64]
    values=[]; errors=[]
    for i in chosen:
        rows=valid[ep[valid]==i]; now=hist[rows,-1]
        obs=json.loads((Path(entries[i]['root'])/'observations.json').read_text())
        pos=np.array([x['robot_position_world'] for x in obs])
        rot=np.array([x['robot_rotation_world_from_body'] for x in obs])
        target=np.array([x['target_position_world_label_only'] for x in obs])
        t=np.array([x['timestamp_s'] for x in obs])
        local=np.einsum('ni,nij->nj',target[now]-pos[now],rot[now])
        r=np.hypot(local[:,0],local[:,2]); bearing=np.arctan2(-local[:,2],local[:,0])
        q=t[now]+.1
        future=np.stack([np.interp(q,t,pos[:,j]) for j in range(3)],1)
        rel=np.einsum('ni,nij->nj',future-pos[now],rot[now])
        xy=np.stack([rel[:,0],-rel[:,2]],1)
        errors.extend(np.linalg.norm(xy-pose[rows,0,:2],axis=1).tolist())
        for k in range(len(rows)):
            values.append([r[k],bearing[k],float(pose[rows[k],0,0]),float(pose[rows[k],0,1])])
    a=np.array(values);bins={}
    for label,mask in {'range_lt1':a[:,0]<1,'range_1to1p5':(a[:,0]>=1)&(a[:,0]<1.5),'range_1p5to3':(a[:,0]>=1.5)&(a[:,0]<3),'range_ge3':a[:,0]>=3,'range_ge5':a[:,0]>=5,'bearing_gt0p4':abs(a[:,1])>.4}.items():
        b=a[mask];bins[label]={'n':len(b),'fraction':len(b)/len(a),'reverse_fraction':float((b[:,2]<-.03).mean()) if len(b) else None}
    out['splits'][part]={'episodes':chosen,'rows':len(a),'range_quantiles':np.quantile(a[:,0],[0,.1,.5,.9,1]).tolist(),'bins':bins,'raw_reconstructed_first_xy_error_m_max':max(errors),'raw_reconstructed_first_xy_error_m_mean':float(np.mean(errors))}
dest=base/'artifacts/coverage_audit_20260930_v1.json'
with dest.open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
