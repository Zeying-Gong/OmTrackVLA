"""Read-only recovery collection audit; does not release training data."""
import json,math,bisect,gzip,hashlib
from pathlib import Path
root=Path('/data/nas_ray/project/md-ak/users/zeying.gong/job_60322/task_71197/wa_recovery_collect_v1')
summary=dict(completed_searches=0,student_success=0,accepted_recoveries=0,branches=0,failed_branch_window_violations=[],teacher_missing_predictions=[],accepted=[],paired_validation_errors=[])
summary.update(label_errors=[],checked_windows=0,maximum_xy_error_m=0.)
repo=Path('/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/checkout')
manifest=json.loads((repo/'wa/wm/recovery_manifest_v1.json').read_text())
plan=json.loads((repo/'wa/wm/mixed_diagnostic_plan_v1.json').read_text())
scene=lambda s:Path(s).name.split('.')[0]
selected={scene(e['scene_id']) for e in manifest['selection']}
excluded=set(manifest['excluded_scenes'])
for phase in ('development','confirmation'):
    excluded.update(x.split('/')[0] for entries in plan[phase].values() for x in entries)
for spec in manifest['sources'].values():
    assert hashlib.sha256(Path(spec['val_path']).read_bytes()).hexdigest()==spec['val_sha256']
    with gzip.open(spec['val_path'],'rt') as stream:
        excluded.update(scene(e['scene_id']) for e in json.load(stream)['episodes'])
summary['split_scene_overlap']=sorted(selected & excluded)
summary['selected_entries']=len(manifest['selection'])
summary['selected_unique_scenes']=len(selected)
def same(a,b):
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,(int,float)):return math.isfinite(b) and abs(a-b)<=1e-6
    return a==b
for search in sorted(root.glob('lane*/collection/*/search.json')):
    x=json.loads(search.read_text());summary['completed_searches']+=1
    summary['student_success']+=int(bool(x['student'].get('success')))
    if x.get('accepted'):
        summary['accepted_recoveries']+=1
        summary['accepted'].append(dict(search=str(search),accepted=x['accepted']))
        try:
            accepted=x['accepted'];step=accepted['step']
            student=json.loads((search.parent/'student'/'replay.json').read_text())
            for key in ('path','repeat_path'):
                branch=Path(accepted[key]);assert branch.parent==search.parent
                result=json.loads((branch/'result.json').read_text())
                assert result['success'] and not result['collision'] and result['policy_init_valid']
                replay=json.loads((branch/'replay.json').read_text())
                for t in range(step+1):
                    assert student[t]['rgb_sha256']==replay[t]['rgb_sha256']
                    assert same(student[t]['dynamic_state'],replay[t]['dynamic_state'])
                    if t<step:assert same(student[t]['action'],replay[t]['action'])
                windows=json.loads((branch/'windows.json').read_text())
                assert all(w['current_index']>=step for w in windows)
        except Exception as exc:
            summary['paired_validation_errors'].append(dict(search=str(search),error=repr(exc)))
for admission in sorted(root.glob('lane*/collection/**/admission.json')):
    a=json.loads(admission.read_text());summary['branches']+=1
    windows=json.loads(admission.with_name('windows.json').read_text())
    try:
        frames=json.loads(admission.with_name('observations.json').read_text())
        times=[f['timestamp_s'] for f in frames]
        assert all(b>a for a,b in zip(times,times[1:]))
        for w in windows:
            cur=frames[w['current_index']];rot=cur['robot_rotation_world_from_body']
            assert same(w['future_times_s'],[i/10 for i in range(1,8)])
            for offset,xy in zip(w['future_times_s'],w['trajectory_xy_m']):
                t=cur['timestamp_s']+offset;assert t<=times[-1]+1e-8
                right=min(max(bisect.bisect_left(times,t),1),len(times)-1);left=right-1
                alpha=(t-times[left])/(times[right]-times[left])
                p=[frames[left]['robot_position_world'][j]*(1-alpha)+frames[right]['robot_position_world'][j]*alpha-cur['robot_position_world'][j] for j in range(3)]
                local=[sum(p[j]*rot[j][k] for j in range(3)) for k in range(3)]
                error=max(abs(xy[0]-local[0]),abs(xy[1]+local[2]));assert math.isfinite(error) and error<1e-6
                summary['maximum_xy_error_m']=max(summary['maximum_xy_error_m'],error)
            summary['checked_windows']+=1
    except Exception as exc:
        summary['label_errors'].append(dict(branch=str(admission.parent),error=repr(exc)))
    if not a['episode_success'] and windows:
        summary['failed_branch_window_violations'].append(str(admission))
    if a['takeover_step'] is not None:
        actions=json.loads(admission.with_name('actions.json').read_text())
        bad=[v['sim_step'] for v in actions if v.get('owner')=='teacher' and v.get('teacher_predicted_waypoints') is None]
        if bad:summary['teacher_missing_predictions'].append(dict(branch=str(admission.parent),steps=bad))
summary['training_release']=False
summary['scope']='Partial live audit: does not establish final repeat/state/split/label acceptance.'
print(json.dumps(summary,indent=2))
