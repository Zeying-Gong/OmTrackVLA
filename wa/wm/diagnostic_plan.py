"""Outcome-blind scene-separated diagnostic selection, frozen before any runs."""
import argparse,hashlib,json
from pathlib import Path
def digest(x):return hashlib.sha256(x.encode()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    raw=Path(a.manifest).read_bytes();m=json.loads(raw)
    scenes=sorted({e['key'].split('/')[0] for e in m['tasks']['stt']['episodes']},key=lambda x:digest('wa-mixed-diagnostic-v1-scene-'+x))
    assert len(scenes)>=16
    plan=dict(manifest_sha256=hashlib.sha256(raw).hexdigest(),seed=7,selection='outcome-blind hash; 8 distinct scenes per split; no scene overlap',
        development={},confirmation={},variants=['image_random','mixed_random','image_zero','mixed_zero'],
        majority_gate=dict(SR_min_percent=80,require_each_task=True,report_CR_always=True),
        limitations=['ideal simulator UWB matches training; not real UWB','development reused for debugging; confirmation untouched until candidate fixed',
        'diagnostic samples must not enter training','no final-test claim; current complete image benchmark already uses validation split',
        'invalid first bbox retained as failure even with UWB for controlled input ablation'])
    for split,names in [('development',scenes[:8]),('confirmation',scenes[8:16])]:
        for task in ('stt','dt','at'):
            plan[split][task]=[]
            for name in names:
                choices=[e['key'] for e in m['tasks'][task]['episodes'] if e['key'].split('/')[0]==name]
                assert choices
                plan[split][task].append(min(choices,key=lambda x:digest('wa-mixed-diagnostic-v1-episode-'+x)))
    out=Path(a.output)
    with out.open('x') as f:json.dump(plan,f,indent=2)
    print('FIXED_PLAN',hashlib.sha256(out.read_bytes()).hexdigest(),flush=True)
if __name__=='__main__':main()
