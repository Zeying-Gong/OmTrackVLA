import json
from pathlib import Path
import numpy as np
root=Path('/data/nas_ray/project/md-ak/users/zeying.gong/job_59842/task_70704/wa_observer_v1')
out={'note':'Measured previous-to-current displacement; command comparison uses previous body ray, not exact SE2 integration; no obstacle-contact attribution from positions alone','episodes':[]}
for lane in sorted(root.glob('mixed_zero_shard*')):
 for line in (lane/'episodes.jsonl').read_text().splitlines():
  e=json.loads(line); p=lane/e['task']/(e['key'].replace('/','_')+'.trace.jsonl')
  trace=[json.loads(x) for x in p.read_text().splitlines()];rows=[]
  assert all(x['observer_state']['usage']=='offline_diagnosis_only_not_policy_input' for x in trace)
  for previous,current in zip(trace,trace[1:]):
   a=previous['observer_state'];b=current['observer_state'];d=previous['diagnostics'];u=d['uwb'];dt=current['diagnostics']['uwb']['timestamp_s']-u['timestamp_s']
   rotation=np.array(a['robot_rotation_world_from_body']);delta=(np.array(b['robot_position_world'])-a['robot_position_world'])@rotation
   target_delta=(np.array(b['target_position_world'])-a['target_position_world'])@rotation
   ray=np.array([np.cos(u['polar'][1]),np.sin(u['polar'][1])]);actual=np.array([delta[0],-delta[2]]);target=np.array([target_delta[0],-target_delta[2]])
   command=np.array(previous['action'][:2])*[15,10]*.025
   rows.append({'step':previous['step'],'range':u['polar'][0],'command_radial_m':float(command@ray),'actual_robot_radial_m':float(actual@ray),'target_radial_m':float(target@ray),'actual_xy_m':actual.tolist(),'command_xy_m':command.tolist(),'dt':dt})
  e['motion']=rows;out['episodes'].append(e)
  if e['status']=='Collision':
   print(e['task'],e['key'],json.dumps(rows[-6:]))
dest=Path('/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/motion_audit_59842.json')
with dest.open('x') as f:json.dump(out,f,indent=2)
