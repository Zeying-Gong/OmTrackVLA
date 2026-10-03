"""Pure CPU validation of full-evaluation model and metric contracts."""
import math
CHECKPOINT_SHA='20cc84b3f231ad4056e16b91c52c84cb14e18d886a80324b5f27ffd773be5331'
MANIFEST_SHA='a1f515534153ccaa720f787f01ea23b9097c174fe25020f756c7d7947eedf98f'
TASKS=('stt','dt','at')

def validate_ready(r):
    checks={'checkpoint_sha256':CHECKPOINT_SHA,'step':45900,'mode':'mixed','noise_mode':'zero','sampling_steps':4,'text_used':False,'world_predictor_inference':False}
    for key,value in checks.items():
        if key not in r or r[key]!=value: raise ValueError('wrong model contract '+key)

def summarize(rows, manifest):
    if len(rows)!=4215 or len({(r['task'],r['key']) for r in rows})!=4215:
        raise ValueError('missing or duplicate episode')
    results={}
    for task in TASKS:
        part=[r for r in rows if r['task']==task]
        expected={e['key'] for e in manifest['tasks'][task]['episodes']}
        if len(part)!=1405 or {r['key'] for r in part}!=expected: raise ValueError('wrong full dataset '+task)
        for r in part:
            if (r.get('mode'),r.get('noise_mode'),r.get('controller'))!=('mixed','zero','learned_yaw_guard_v1'): raise ValueError('wrong episode contract')
            if type(r.get('policy_init_valid')) is not bool: raise ValueError('missing init validity')
            for field in ('success','collision','following_rate','following_step','total_step'):
                if not math.isfinite(float(r[field])): raise ValueError('nonfinite metric')
            if r['success'] not in (0,1) or r['collision'] not in (0,1): raise ValueError('nonbinary outcome')
        ref=manifest['reference_steps'];den=sum(max(r['total_step'],ref.get(r['key'],0)) for r in part)
        results[task]=dict(episodes=1405,SR=100*sum(r['success'] for r in part)/1405,
            CR=100*sum(r['collision'] for r in part)/1405,
            TR=100*sum(r['following_step'] for r in part)/den if den else None,
            macro_TR=100*sum(r['following_rate'] for r in part)/1405,
            finish=100*sum(bool(r['finish']) for r in part)/1405,
            invalid_init_count=sum(not r['policy_init_valid'] for r in part),
            reference_missing=sum(r['key'] not in ref for r in part))
    return dict(status='COMPLETE_FULL_VALIDATION',mode='mixed',noise_mode='zero',controller='learned_yaw_guard_v1',
        episodes=4215,checkpoint_sha256=CHECKPOINT_SHA,checkpoint_step=45900,metrics_percent=results,
        limits=['ideal simulated polar UWB noise0 delay0; no text','existing validation split includes development/confirmation; not untouched test',
                'HumanCollision is target-person distance ever<0.5m not general obstacle contact','no real-UWB or edge-latency validation'])

def allocated_devices(raw, count):
    if not 1 <= count <= 8: raise ValueError("expected1..8 actually visible CUDA GPUs")
    devices = [str(i) for i in range(count)] if raw is None else raw.split(",")
    if len(devices) != count or len(set(devices)) != count or any(not d.strip() for d in devices):
        raise ValueError("CUDA_VISIBLE_DEVICES disagrees with actual device count")
    return devices
