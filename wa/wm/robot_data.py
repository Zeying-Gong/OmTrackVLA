"""Actual-time robot command transitions; never substitute future waypoints for commands."""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from wa.data import TrackingData,tensor_image,sha

CONTRACT='evt_normalized_command3_actual_dt_v1'

def transition_records(root,observations):
    records=json.loads((Path(root)/'actions.json').read_text())
    by_step={int(a['sim_step']):a for a in records}
    if len(by_step)!=len(records):raise ValueError('duplicate action steps')
    times=np.array([o['timestamp_s'] for o in observations],dtype=np.float64)
    commands=np.array([by_step[int(o['sim_step'])]['normalized_action'] for o in observations[:-1]],dtype=np.float32)
    if commands.shape!=(len(times)-1,3) or not np.isfinite(commands).all():
        raise ValueError('invalid normalized commands')
    clipped_steps=int((abs(commands)>1).any(axis=1).sum())
    commands=np.clip(commands,-1,1)  # Exact BaseVelNonCylinderAction.step saturation.
    dt=np.diff(times)
    if not np.isfinite(dt).all() or (dt<=0).any():raise ValueError('invalid actual timestamps')
    pos=np.array([o['robot_position_world'] for o in observations])
    rot=np.array([o['robot_rotation_world_from_body'] for o in observations])
    relative=np.einsum('nji,njk->nik',rot[:-1],rot[1:])
    yaw=np.arctan2(relative[:,0,2],relative[:,0,0])
    distance=np.linalg.norm(np.diff(pos[:,[0,2]],axis=0),axis=1)
    # Conservative single-control bounds: speed caps15/10m/s, yaw6.28rad/s,
    # control integration0.025s. Allow0.02m and0.01rad tolerance.
    # Outliers may be NavMesh/reset corrections: exclude, do not relabel them.
    bad=(distance>np.hypot(15,10)*.025+.02)|(abs(yaw)>6.28*.025+.01)|(dt<.02)|(dt>.15)
    cmd=np.concatenate([commands,(dt/.1).astype(np.float32)[:,None]],1)
    return cmd,bad,{'max_step_m':float(distance.max()),'max_speed_m_s':float((distance/dt).max()),'dt_min':float(dt.min()),'dt_max':float(dt.max()),'bad_steps':int(bad.sum()),'steps':len(dt),'clipped_command_steps':clipped_steps}

class RobotWorldData(TrackingData):
    def __init__(self,cache,part,index_root,limit=None,data_root=None,source_prefix=None):
        super().__init__(cache,part,data_root=data_root,source_prefix=source_prefix)
        self.rows=np.load(Path(index_root)/f'{part}_valid.npy')
        self.pose_labels=self.pose
        self.source_records={s['episode']:s['source_sha256'] for s in json.loads((Path(index_root)/f'{part}_episodes_audit.json').read_text())['stats']}
        if limit is not None:self.rows=self.rows[:limit]
        self.length=len(self.rows)
    @lru_cache(maxsize=32)
    def transitions(self,episode):
        entry,root,obs,times,template=self.episode_info(episode)
        for name,expected in self.source_records[episode].items():
            if sha(root/name)!=expected:raise ValueError('raw episode changed since audit')
        commands,bad,_=transition_records(root,obs)
        return commands,bad
    def __getitem__(self,index):
        row=int(self.rows[index]);ep=int(self.episode[row])
        entry,root,obs,times,template=self.episode_info(ep)
        history=self.history[row];now=int(history[-1]);commands,bad=self.transitions(ep)
        wi=np.arange(now-3,now+1)
        end=int(np.searchsorted(times,times[now]+.7))
        if now<4 or end>=len(times) or bad[int(history[0]):end].any():raise ValueError('audit/runtime validity mismatch')
        item=super().__getitem__(row)
        polar=item['point'][:2]
        geometry=torch.stack([polar[0].clamp_min(1e-6).log(),polar[1].cos(),polar[1].sin()])
        return dict(rgb=item['rgb'],template=template.clone(),polar=polar,times=item['times'],
                    pose=torch.from_numpy(np.array(self.pose_labels[row])),geometry=geometry,
                    wm_rgb=torch.stack([tensor_image(root/entry['frames'][int(i)]) for i in wi]),
                    commands=torch.from_numpy(commands[wi].copy()),proprio=torch.from_numpy(commands[wi-1].copy()),
                    future=tensor_image(root/entry['frames'][now+1]))

def audit_transitions(cache,output):
    cache=Path(cache);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    report={'contract':CONTRACT,'rules':'exclude windows spanning any oversized displacement/yaw or dt outside[.02,.15]; preserve original scene split and cached7x4 labels','cache_complete_sha256':sha(cache/'complete.json'),'splits':{}}
    for part in ('train','heldout'):
        data=TrackingData(cache,part)
        valid=np.zeros(len(data),dtype=bool);stats=[];errors=[]
        order=np.argsort(data.episode,kind='stable');eps=np.asarray(data.episode)[order]
        starts=np.searchsorted(eps,np.arange(len(data.episodes)+1))
        for ep in range(len(data.episodes)):
            rows=order[starts[ep]:starts[ep+1]]
            try:
                entry,root,obs,times,_=data.episode_info(ep)
                commands,bad,s=transition_records(root,obs)
                h=np.asarray(data.history[rows]);now=h[:,-1];end=np.searchsorted(times,times[now]+.7)
                prefix=np.r_[0,np.cumsum(bad)]
                supported=(now>=4)&(end<len(times))
                supported &= prefix[np.minimum(end,len(bad))]==prefix[h[:,0]]
                valid[rows]=supported
                s.update(episode=ep,retained=int(supported.sum()),rows=len(rows),source_sha256={name:sha(root/name) for name in ('metadata.json','observations.json','actions.json')})
                stats.append(s)
            except (ValueError,KeyError,IndexError) as exc:
                errors.append({'episode':ep,'error':str(exc)})
            if ep%1000==0:print(part,ep,'episodes audited',flush=True)
        np.save(output/f'{part}_valid.npy',np.flatnonzero(valid))
        (output/f'{part}_episodes_audit.json').write_text(json.dumps({'stats':stats,'errors':errors}))
        report['splits'][part]={'original':len(data),'retained':int(valid.sum()),'excluded':int((~valid).sum()),'bad_steps':sum(s['bad_steps'] for s in stats),'steps':sum(s['steps'] for s in stats),'episode_errors':errors,'index_sha256':sha(output/f'{part}_valid.npy')}
        print(part,{k:v for k,v in report['splits'][part].items() if k!='episode_errors'},'episode_errors',len(errors),flush=True)
    (output/'audit.json').write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();audit_transitions(a.cache,a.output)
