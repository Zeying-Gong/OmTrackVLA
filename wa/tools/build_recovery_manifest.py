import gzip,hashlib,json
from pathlib import Path
R=Path('/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928')
source=Path('/data/nas_ray/home/zeying.gong/algorithm/repos/WLA-EVT-20260925/intervention_ab_20260929/manifest.json')
m=json.loads(source.read_text());excluded=set()
def scene(s):return Path(s).name.split('.')[0]
for spec in m['sources'].values():
 for field,h in [('path','sha256'),('val_path','val_sha256')]:
  assert hashlib.sha256(Path(spec[field]).read_bytes()).hexdigest()==spec[h]
 with gzip.open(spec['val_path'],'rt') as f:
  excluded.update(scene(e['scene_id']) for e in json.load(f)['episodes'])
excluded.update(scene(e['scene_id']) for e in m['selection'] if e['partition']!='train')
selected=[]
for task in ('stt','dt','at'):
 rows=[e for e in m['selection'] if e['task']==task and e['partition']=='train' and scene(e['scene_id']) not in excluded]
 rows.sort(key=lambda e:hashlib.sha256(('wa-recovery-v1:'+e['task_episode_uid']).encode()).hexdigest())
 selected.extend(rows[:32])
assert len(selected)==96 and len({e['task_episode_uid'] for e in selected})==96
out=dict(version='wa_failure_backtrack_v1',selection=selected,sources=m['sources'],
 source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
 selection_rule='32 per task SHA-order from existing train partition; exclude all benchmark-val and source-heldout scenes; outcome blind',
 excluded_scenes=sorted(excluded),backtrack_gap_steps=5,
 positive_rule='full successful teacher continuation plus repeat success; no failed student prefix supervision',
 simulation_state_validation='prefix RGB exact and all articulated agents transform/joints/time tolerance1e-6',
 status='PREDECLARED_FIRST_COLLECTION_ROUND_NOT_RELEASED')
p=R/'checkout/wa/wm/recovery_manifest_v1.json'
with p.open('x') as f:json.dump(out,f,indent=2)
print('Selected',len(selected),'excluded scenes',len(excluded),'sha256',hashlib.sha256(p.read_bytes()).hexdigest())
