import json, html
from pathlib import Path
R=Path('/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928')
J=Path('/data/nas_ray/project/md-ak/users/zeying.gong')
out=R/'artifacts/closedloop_review_59960'
out.mkdir(exist_ok=False)
sources={'New WA':J/'job_59960/task_70822/wa_history_repeat_eval_v1','Old WA':J/'job_59846/task_70708/wa_lightnav_review_v1/wa','LightNav':J/'job_59846/task_70708/wa_lightnav_review_v1/lightnav'}
data={}
for i,(name,root) in enumerate(sources.items()):
 (out/f'media{i}').symlink_to(root,target_is_directory=True)
 rows={}
 for p in sorted(root.glob('*/episodes.jsonl')):
  for line in p.read_text().splitlines():
   e=json.loads(line); key=(e['task'],e['key']); assert key not in rows
   v=p.parent/e['task']/'_review'/e['key']/'review.mp4'
   assert v.is_file()
   e['video']=f'media{i}/'+str(v.relative_to(root)); rows[key]=e
 assert len(rows)==24
 data[name]=rows
assert all(set(x)==set(data['New WA']) for x in data.values())
parts=['<!doctype html><meta charset="utf-8"><title>WA 59960 closed-loop review</title><style>body{font:16px sans-serif;background:#101827;color:#eee;margin:24px}section{border-top:1px solid #567;padding:18px 0}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}video{width:100%}h1{font-size:26px}.bad{color:#ffadad}.good{color:#9eebad}select{padding:8px}</style><h1>WA mixed-mode closed-loop review · 59960</h1><p>New WA 14/24 · Old WA 14/24 · LightNav 17/24. Fixed development set; not product acceptance. WA: RGB+BBox+ideal simulated UWB, no text. LightNav: RGB+text. Different modalities. Videos 20fps; use overlaid simulation timestamps, not playback time, for timing comparisons.</p><p>New WA SR: STT 62.5%, DT 62.5%, AT 50%; CR 12.5% each; invalid 0. No claim of improvement.</p><label>Filter <select onchange="document.querySelectorAll(\'section\').forEach(x=>x.hidden=this.value!==\'all\'&&!x.dataset.tags.includes(this.value))"><option value="all">All 24</option><option value="failed">New WA failures</option><option value="door">Doorframe scene</option><option>stt</option><option>dt</option><option>at</option></select></label>']
paired=[]
for key in sorted(data['New WA']):
 n=data['New WA'][key]; tags=key[0]+(' failed' if not n['success'] else '')+(' door' if key[1]=='VLzqgDo317F/89' else '')
 assert len({rows[key]['initial_rgb_sha256'] for rows in data.values()})==1
 parts.append(f'<section data-tags="{tags}"><h2>{html.escape(" / ".join(key))}</h2><div class="grid">')
 for name,rows in data.items():
  e=rows[key];cls='good' if e['success'] else 'bad'
  parts.append(f'<div><h3>{name}</h3><p class="{cls}">{html.escape(e["status"])} · success {e["success"]} · collision {e["collision"]}</p><video controls preload="none" src="{html.escape(e["video"])}"></video><a href="{html.escape(e["video"])}" download>Download MP4</a></div>')
 parts.append('</div></section>');paired.append({'task':key[0],'key':key[1],'outcomes':{name:rows[key]['status'] for name,rows in data.items()}})
(out/'index.html').write_text(''.join(parts))
(out/'paired.json').write_text(json.dumps(paired,indent=2))
print(out)
