"""Fail-closed final collection audit and nonduplicated teacher-suffix index."""
import contextlib,io,runpy,json,hashlib,math,struct
from pathlib import Path
repo=Path('/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/checkout')
with contextlib.redirect_stdout(io.StringIO()):
    state=runpy.run_path(str(repo/'wa/tools/audit_recovery_60322.py'))
s=state['summary'];root=state['root'];m=state['manifest']
for key in ('failed_branch_window_violations','teacher_missing_predictions','paired_validation_errors','label_errors','search_errors','split_scene_overlap'):
    assert not s[key],(key,s[key])
assert s['completed_searches']==96
assert json.loads((root/'COMPLETE.json').read_text())['episodes']==96
for lane in range(2):
    done=json.loads((root/f'lane{lane}/collection/COMPLETE.json').read_text())
    assert done['episodes']==done['expected']==48 and not done['audit_only']
    log=(root/f'lane{lane}/worker.log').read_text()
    assert not any(v in log for v in ('Traceback (most recent','LIGHTNAV_RELEASED_FALLBACK','REPLAY_DIVERGED','REPLAY_DYNAMIC_STATE_DIVERGED'))
reports=[json.loads(p.read_text()) for p in sorted(root.glob('lane*/collection/*/search.json'))]
assert sorted(x['episode']['task_episode_uid'] for x in reports)==sorted(x['task_episode_uid'] for x in m['selection'])
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inventory=[];frames_checked=0;failed_branches=0
for p in sorted(root.glob('lane*/collection/**/admission.json')):
    branch=p.parent;a=json.loads(p.read_text());meta=json.loads((branch/'metadata.json').read_text())
    obs=json.loads((branch/'observations.json').read_text());acts=json.loads((branch/'actions.json').read_text())
    replay=json.loads((branch/'replay.json').read_text());result=json.loads((branch/'result.json').read_text())
    assert len(obs)==len(acts)==len(replay)==result['total_step']
    assert meta['partition']=='train' and meta['initial_bbox_status']=='VERIFIED_CONFIG_AND_SEMANTIC'
    assert meta['sensor']=='ideal_simulated_uwb' and meta['uwb_noise']==meta['uwb_delay_s']==0
    assert meta['manifest_sha256']==digest(repo/'wa/wm/recovery_manifest_v1.json')
    for i,(o,act,r) in enumerate(zip(obs,acts,replay)):
        assert o['sim_step']==act['sim_step']==r['step']==i
        assert len(act['normalized_action'])==3 and all(math.isfinite(v) and abs(v)<=1 for v in act['normalized_action'])
        assert state['same'](act['normalized_action'],r['action'])
        data=(branch/o['frame']).read_bytes()
        assert data[:8]==b'\x89PNG\r\n\x1a\n'
        width,height=struct.unpack('>II',data[16:24]);assert [height,width,3]==meta['rgb_shape']
        frames_checked+=1
    if not a['episode_success']:failed_branches+=1
    inventory.append(dict(branch=str(branch),hashes={name:digest(branch/name) for name in ('metadata.json','observations.json','actions.json','replay.json','result.json','windows.json','admission.json')}))
accepted=[]
for report in reports:
    a=report['accepted']
    if not a:continue
    branch=Path(a['path']);windows=json.loads((branch/'windows.json').read_text())
    accepted.append(dict(uid=report['episode']['task_episode_uid'],task=report['episode']['task'],
        category='teacher_from_start' if a['step']==0 else 'mid_episode_recovery',takeover_step=a['step'],
        branch=str(branch),repeat_verification_only=a['repeat_path'],window_indices=[w['current_index'] for w in windows],
        windows_sha256=digest(branch/'windows.json')))
output=dict(version='recovery60322_collection_audit_v1',job=60322,task=71197,collection_validated=True,
    raw_training_flags_unchanged=True,automatic_training_authorized_by_this_manifest=False,
    scope='Validated collection and executed XY labels, not model improvement or a ready-made WA SE2 training cache.',
    summary=s,frames_checked=frames_checked,failed_branches=failed_branches,teacher_suffixes=accepted,
    teacher_suffix_windows=sum(len(x['window_indices']) for x in accepted),
    midpoint_recovery_count=sum(x['takeover_step']>0 for x in accepted),
    from_start_count=sum(x['takeover_step']==0 for x in accepted),inventory=inventory,
    exclusions='No failed branches; no failed WA prefixes; no repeat duplication; no student-success trajectories in teacher index.',
    limitations=['Ideal simulated UWB only; no physical UWB validation.','Teacher RGB+text vs student RGB+polar UWB; different modalities.',
      'Replay verifies observed RGB/articulated transforms/joints/time, not serialization of every hidden simulator state.',
      'One identical-seed repeat checks reproducibility, not robustness over random seeds.',
      'Gap5 search gives latest tested successful point, not exact optimal boundary.',
      'Teacher labels are executed XY at actual times; existing WA SE2 cache adaptation and retraining remain separate work.'])
dest=repo/'wa/results/recovery_60322_final_audit.json'
with dest.open('x') as f:json.dump(output,f,indent=2)
print(json.dumps({k:v for k,v in output.items() if k not in ('inventory','summary','teacher_suffixes')},indent=2))
