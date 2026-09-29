import json,time,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
base=Path('/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928')
cache=Path('/data/nas_ray/home/zeying.gong/datasets/wla_evt_se2_cache_20260925_v2')
report={'protocol':'all audited training and heldout windows; current simulated target geometry only; CPU metadata audit','splits':{}}
for part in ('train','heldout'):
    entries=json.loads((cache/f'{part}_episodes.json').read_text())
    valid=np.load(base/f'artifacts/robot_transition_audit_v2/{part}_valid.npy')
    ids=np.load(cache/f'{part}_episode.npy',mmap_mode='r')[valid]
    hist=np.load(cache/f'{part}_history.npy',mmap_mode='r')
    order=np.argsort(ids,kind='stable'); ordered=ids[order]
    boundaries=np.searchsorted(ordered,np.arange(len(entries)+1))
    def audit(i):
        rows=valid[order[boundaries[i]:boundaries[i+1]]]
        if not len(rows):return np.empty((0,2))
        obs=json.loads((Path(entries[i]['root'])/'observations.json').read_text())
        current=[obs[int(j)] for j in hist[rows,-1]]
        delta=np.array([np.array(o['target_position_world_label_only'])-o['robot_position_world'] for o in current])
        rotation=np.array([o['robot_rotation_world_from_body'] for o in current])
        local=np.einsum('ni,nij->nj',delta,rotation)
        return np.stack([np.hypot(local[:,0],local[:,2]),np.arctan2(-local[:,2],local[:,0])],1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        a=np.concatenate(list(pool.map(audit,range(len(entries)))))
    bins={key:int(mask.sum()) for key,mask in {'range_lt1':a[:,0]<1,'range_lt1p5':a[:,0]<1.5,'range_ge3':a[:,0]>=3,'range_ge4':a[:,0]>=4,'range_ge5':a[:,0]>=5,'bearing_abs_ge0p4':abs(a[:,1])>=.4}.items()}
    report['splits'][part]={'rows':len(a),'bins':bins,'range_quantiles':np.quantile(a[:,0],[0,.01,.1,.5,.9,.99,1]).tolist(),'valid_index_sha256':hashlib.sha256((base/f'artifacts/robot_transition_audit_v2/{part}_valid.npy').read_bytes()).hexdigest()}
    print(part,json.dumps(report['splits'][part]),flush=True)
with (base/'artifacts/full_coverage_20260930_v1.json').open('x') as f:json.dump(report,f,indent=2)
