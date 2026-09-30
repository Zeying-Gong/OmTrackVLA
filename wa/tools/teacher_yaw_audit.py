import json
from pathlib import Path
import numpy as np
base=Path('/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928');cache=Path('/data/nas_ray/home/zeying.gong/datasets/wla_evt_se2_cache_20260925_v2')
selection=json.loads((base/'artifacts/coverage_audit_20260930_v1.json').read_text());out={'protocol':'same64 hash-selected episodes per split; oracle cached pose yaw decoded before target guard; differs in objective from instant expert action, not necessarily bug','splits':{}}
for part in ('train','heldout'):
 entries=json.loads((cache/f'{part}_episodes.json').read_text());ep=np.load(cache/f'{part}_episode.npy',mmap_mode='r');h=np.load(cache/f'{part}_history.npy',mmap_mode='r');pose=np.load(cache/f'{part}_pose.npy',mmap_mode='r');valid=np.load(base/f'artifacts/robot_transition_audit_v2/{part}_valid.npy');values=[]
 for i in selection['splits'][part]['episodes']:
  root=Path(entries[i]['root']);obs=json.loads((root/'observations.json').read_text());actions={int(a['sim_step']):a for a in json.loads((root/'actions.json').read_text())};times=np.array([o['timestamp_s'] for o in obs]);rot=np.array([o['robot_rotation_world_from_body'] for o in obs]);yaw=np.unwrap(np.arctan2(rot[:,0,2],rot[:,0,0]))
  for row in valid[ep[valid]==i]:
   now=int(h[row,-1]);command=float(np.clip(actions[int(obs[now]['sim_step'])]['normalized_action'][2],-1,1));label=float(np.arctan2(pose[row,0,2],pose[row,0,3]));dt=times[now]-times[now-1];decoded=float(np.clip(label/.1*dt/.025/6.28,-1,1));actual=(yaw[now+1]-yaw[now])/(6.28*.025)
   values.append([command,decoded,actual])
 a=np.array(values);active=abs(a[:,0])>=.5;v=a[active]
 result={'n':len(a),'teacher_command_vs_oracle_decoded_MAE':float(abs(a[:,0]-a[:,1]).mean()),'teacher_command_vs_observed_step_MAE':float(abs(a[:,0]-a[:,2]).mean()),'sharp_command_n':len(v),'sharp_median_abs_teacher':float(np.median(abs(v[:,0]))),'sharp_median_abs_decoded':float(np.median(abs(v[:,1]))),'sharp_decoded_less_than_half_fraction':float((abs(v[:,1])<.5*abs(v[:,0])).mean()),'sharp_decoded_opposite_fraction':float((v[:,0]*v[:,1]<0).mean())}
 out['splits'][part]=result;print(part,json.dumps(result))
with (base/'artifacts/teacher_yaw_audit_v1.json').open('x') as f:json.dump(out,f,indent=2)
