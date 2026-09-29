import json
from pathlib import Path
base=Path('/data/nas_ray/project/md-ak/users/zeying.gong')
runs={'epoch1':base/'job_59726/task_70588/wa_mixed_diagnostic_v1','epoch2':base/'job_59826/task_70688/wa_epoch2_mixed_eval_v1'}
out={}
for name,root in runs.items():
    results=[]
    for lane in sorted(root.glob('mixed_zero_shard*')):
        for line in (lane/'episodes.jsonl').read_text().splitlines():
            e=json.loads(line);path=lane/e['task']/(e['key'].replace('/','_')+'.trace.jsonl')
            trace=[json.loads(x) for x in path.read_text().splitlines()]
            ds=[x['diagnostics'] for x in trace if x.get('diagnostics',{}).get('uwb')]
            def record(d):
                return {'step':d['prediction_step'],'range':d['uwb']['polar'][0],'bearing':d['uwb']['polar'][1],'xy0':d['xy'][0],'yaw0':d['yaw'][0],'action':d['normalized_action'],'blend':d['control_info'].get('heading_blend')}
            e['first']=record(ds[0]);e['last']=record(ds[-1]);e['crossings']={}
            for threshold in (3,4,5):
                crossing=next((i for i,d in enumerate(ds) if d['uwb']['polar'][0]>=threshold),None)
                e['crossings'][str(threshold)]=None if crossing is None else [record(d) for d in ds[max(0,crossing-3):crossing+1]]
            results.append(e)
            if not e['success']:print(name,e['task'],e['key'],e['status'],'first',e['first'],'last',e['last'],flush=True)
    assert len(results)==24
    out[name]=results
dest=Path('/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/failure_onset_20260930_v1.json')
with dest.open('x') as f:json.dump(out,f,indent=2)
