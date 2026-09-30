import json
from pathlib import Path
base=Path('/data/nas_ray/project/md-ak/users/zeying.gong')
light={}
for p in (base/'job_58433/task_69173/lightnav_transport/lightnav').glob('shard_*/episodes.jsonl'):
 for l in p.read_text().splitlines():
  e=json.loads(l); k=(e['task'],e['key']);assert k not in light;light[k]=e
rows=[]
for p in (base/'job_59842/task_70704/wa_observer_v1').glob('mixed_zero_shard*/episodes.jsonl'):
 for l in p.read_text().splitlines():
  e=json.loads(l);b=light[(e['task'],e['key'])]
  rows.append(dict(task=e['task'],key=e['key'],wa_success=e['success'],wa_status=e['status'],lightnav_success=b['success'],lightnav_status=b['status'],lightnav_collision=b['collision'],lightnav_following_rate=b['following_rate'],initial_rgb_equal=e['initial_rgb_sha256']==b['initial_rgb_sha256'],lightnav_fallback=b.get('released_fallback_count')))
assert len(rows)==24
out={'note':'Historical LightNav RGB+text versus WA RGB+bbox+idealUWB; not modality controlled. Same task/key; verify initial RGB. No stored LightNav action traces or video found.','rows':rows,'summary':{}}
for task in ['stt','dt','at']:
 r=[e for e in rows if e['task']==task];fail=[e for e in r if not e['wa_success']]
 out['summary'][task]={'n':len(r),'wa_success':sum(e['wa_success'] for e in r),'lightnav_success':sum(e['lightnav_success'] for e in r),'wa_failed_lightnav_succeeded':sum(e['lightnav_success'] for e in fail),'wa_failures':len(fail),'initial_rgb_matched':sum(e['initial_rgb_equal'] for e in r)}
print(json.dumps(out,indent=2))
dest=Path('/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/lightnav_paired24_20260930.json')
with dest.open('x') as f:json.dump(out,f,indent=2)
