import json
from pathlib import Path
import numpy as np
r=Path('/data/nas_ray/project/md-ak/users/zeying.gong/job_59846/task_70708/wa_lightnav_review_v1')
out={}
for method,pattern in [('wa','mixed_zero_shard*'),('lightnav','shard_*')]:
 rows=[]
 for lane in (r/method).glob(pattern):
  for l in (lane/'episodes.jsonl').read_text().splitlines():
   e=json.loads(l);p=lane/e['task']/'_review'/e['key'];d=[json.loads(x) for x in (p/'steps.jsonl').read_text().splitlines()];a=np.array([x['action'] for x in d]);polar=np.array([x['polar'] for x in d]);edge=abs(polar[:,1])>=.35
   e={k:e[k] for k in ['task','key','success','collision','status','initial_rgb_sha256']}
   e.update(n=len(d),video=str(p/'review.mp4'),min_range=float(polar[:,0].min()),max_range=float(polar[:,0].max()),reverse_frac=float((a[:,0]<-.05).mean()),lateral_mean_abs=float(abs(a[:,1]).mean()),yaw_mean_abs=float(abs(a[:,2]).mean()),edge_steps=int(edge.sum()),edge_yaw_mean_abs=float(abs(a[edge,2]).mean()) if edge.any() else None,first10_yaw_mean_abs=float(abs(a[:10,2]).mean()))
   rows.append(e)
 out[method]=rows
dest=Path('/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/paired_review_stats_59846.json')
with dest.open('x') as f:json.dump(out,f,indent=2)
for method,rows in out.items():
 for e in rows:
  if e['key']=='VLzqgDo317F/89':print(method,json.dumps(e))
old={}
for p in Path('/data/nas_ray/project/md-ak/users/zeying.gong/job_58433/task_69173/lightnav_transport/lightnav').glob('shard_*/episodes.jsonl'):
 for l in p.read_text().splitlines():
  e=json.loads(l);old[e['task'],e['key']]=e
for e in out['lightnav']:
 b=old[e['task'],e['key']]
 if b['success']!=e['success']:print('CHANGED',e['task'],e['key'],b['status'],e['status'])
