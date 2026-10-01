"""Convert accepted teacher suffixes to the unchanged WA robot training contract."""
import argparse,json
from pathlib import Path
import numpy as np
from wa.data import sha
from wa.wm.robot_data import transition_records,CONTRACT

def main():
 p=argparse.ArgumentParser()
 for key in ('release','base-cache','base-index','output'):p.add_argument('--'+key,required=True)
 a=p.parse_args();release=Path(a.release).resolve();m=json.loads(release.read_text())
 assert m['collection_validated'] and m['summary']['completed_searches']==96
 base=Path(a.base_cache).resolve();baseindex=Path(a.base_index).resolve();out=Path(a.output).resolve()
 out.mkdir(parents=True,exist_ok=False);idx=out/'index';idx.mkdir()
 inventory={x['branch']:x['hashes'] for x in m['inventory']}
 episodes=[];poses=[];histories=[];episodeids=[];valid=[];stats=[];row_start=0
 heldout=json.loads((base/'heldout_episodes.json').read_text())
 heldscenes={Path(x['scene']).name.split('.')[0] for x in heldout}
 for ep,entry in enumerate(m['teacher_suffixes']):
  root=Path(entry['branch'])
  for name,h in inventory[str(root)].items():assert sha(root/name)==h,(root,name)
  meta=json.loads((root/'metadata.json').read_text());obs=json.loads((root/'observations.json').read_text())
  wins=json.loads((root/'windows.json').read_text());indices=np.array([w['current_index'] for w in wins])
  assert indices.tolist()==entry['window_indices'] and (indices>=entry['takeover_step']).all()
  assert Path(meta['scene_id']).name.split('.')[0] not in heldscenes
  times=np.array([o['timestamp_s'] for o in obs]);rot=np.array([o['robot_rotation_world_from_body'] for o in obs])
  yaw=np.unwrap(np.arctan2(rot[:,0,2],rot[:,0,0]));query=times[indices,None]+np.arange(1,8)/10.
  futureyaw=np.interp(query.ravel(),times,yaw).reshape(-1,7)-yaw[indices,None]
  xy=np.array([w['trajectory_xy_m'] for w in wins],dtype=np.float32)
  pose=np.concatenate([xy,np.sin(futureyaw)[...,None],np.cos(futureyaw)[...,None]],-1).astype(np.float32)
  history=np.searchsorted(times,times[indices,None]+np.array([-1.5,-1.,-.5,0.]),side='right')-1
  history=np.minimum(np.maximum(history,0),indices[:,None]).astype(np.int32)
  assert pose.shape==(len(wins),7,4) and np.isfinite(pose).all()
  commands,bad,s=transition_records(root,obs)
  end=np.searchsorted(times,times[indices]+.7);prefix=np.r_[0,np.cumsum(bad)]
  supported=(indices>=4)&(end<len(times))
  supported &= prefix[np.minimum(end,len(bad))]==prefix[history[:,0]]
  valid.extend((row_start+np.flatnonzero(supported)).tolist());row_start+=len(wins)
  s.update(episode=ep,retained=int(supported.sum()),rows=len(wins),source_sha256={n:sha(root/n) for n in ('metadata.json','observations.json','actions.json')})
  stats.append(s);poses.append(pose);histories.append(history);episodeids.append(np.full(len(wins),ep,dtype=np.int32))
  episodes.append(dict(root=str(root),frames=[o['frame'] for o in obs],task=entry['task'],episode_uid=entry['uid'],scene=meta['scene_id'],takeover_step=entry['takeover_step'],category=entry['category']))
 for key,values in [('pose',poses),('history',histories),('episode',episodeids)]:np.save(out/f'train_{key}.npy',np.concatenate(values))
 (out/'train_episodes.json').write_text(json.dumps(episodes))
 for suffix in ('pose.npy','history.npy','episode.npy','episodes.json'):(out/f'heldout_{suffix}').symlink_to(base/f'heldout_{suffix}')
 complete=dict(source_release=str(release),source_sha256=sha(release),inputs='causal RGB/template/current simulated polar UWB; no text',files={f'{part}_{suffix}':sha(out/f'{part}_{suffix}') for part in ('train','heldout') for suffix in ('pose.npy','history.npy','episode.npy','episodes.json')})
 (out/'complete.json').write_text(json.dumps(complete,indent=2))
 np.save(idx/'train_valid.npy',np.array(valid,dtype=np.int64))
 (idx/'train_episodes_audit.json').write_text(json.dumps(dict(stats=stats,errors=[])))
 for name in ('heldout_valid.npy','heldout_episodes_audit.json'):(idx/name).symlink_to(baseindex/name)
 original=json.loads((baseindex/'audit.json').read_text())
 audit=dict(contract=CONTRACT,cache_complete_sha256=sha(out/'complete.json'),splits=dict(train=dict(original=row_start,retained=len(valid),excluded=row_start-len(valid),index_sha256=sha(idx/'train_valid.npy'),episode_errors=[]),heldout=original['splits']['heldout']))
 (idx/'audit.json').write_text(json.dumps(audit,indent=2))
 print(json.dumps(dict(episodes=len(episodes),candidate_windows=row_start,retained_windows=len(valid),excluded_windows=row_start-len(valid),cache=str(out),heldout_unchanged=True),indent=2))
if __name__=='__main__':main()
