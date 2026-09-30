import json
from pathlib import Path
import numpy as np
base=Path('/data/nas_ray/project/md-ak/users/zeying.gong')
runs={'baseline':base/'job_59842/task_70704/wa_observer_v1','gain2':base/'job_59852/task_70714/wa_yaw_gain2_v1'}
out={'definition':'descriptive command oscillation; sign changes only adjacent yaw magnitudes >0.05; rates per100 transitions; per-episode macro means; lengths differ','runs':{}}
for name,root in runs.items():
 rows=[]
 for lane in root.glob('mixed_zero_shard*'):
  for l in (lane/'episodes.jsonl').read_text().splitlines():
   e=json.loads(l);trace=[json.loads(x) for x in (lane/e['task']/(e['key'].replace('/','_')+'.trace.jsonl')).read_text().splitlines()]
   a=np.array([x['action'] for x in trace]);yaw=a[:,2];delta=np.diff(yaw);n=max(1,len(delta));flips=(yaw[:-1]*yaw[1:]<0)&(abs(yaw[:-1])>.05)&(abs(yaw[1:])>.05)
   rows.append(dict(task=e['task'],key=e['key'],success=e['success'],following_rate=e['following_rate'],n=len(yaw),yaw_tv_per_step=float(abs(delta).sum()/n),yaw_flips_per100=float(flips.sum()/n*100),yaw_saturation_fraction=float((abs(yaw)>=.999).mean())))
 out['runs'][name]={'episodes':rows,'macro':{k:float(np.mean([e[k] for e in rows])) for k in ['yaw_tv_per_step','yaw_flips_per100','yaw_saturation_fraction']}}
 print(name,out['runs'][name]['macro'])
 for e in rows:
  if (e['task'],e['key']) in [('stt','SByzJLxpRGn/8'),('dt','auFeVz9Go4m/10')]:print(e)
dest=Path('/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/yaw_oscillation_59852.json')
with dest.open('x') as f:json.dump(out,f,indent=2)
