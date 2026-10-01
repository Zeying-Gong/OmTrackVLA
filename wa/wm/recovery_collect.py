"""Training-only failure-backtracking collection. No evaluation scenes admitted."""
import argparse,copy,gc,json,random,hashlib
from pathlib import Path
import numpy as np
import torch,habitat,evt_bench,trained_agent
from omegaconf import OmegaConf
from evt_full_20260926.common import BENCH,SCENES
from lightnav_transport_20260927.agent import Agent as Teacher
from wa.wm.diagnostic_agent import DiagnosticAgent
from wa.wm.recovery_replay import ReplayThenTeacher,backward_candidates
from wa.wm.recovery_recorder import RecoveryRecorder

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser()
 for x in ('manifest','ready','teacher-url','output'):p.add_argument('--'+x,required=True)
 p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,default=8)
 p.add_argument('--audit-only',action='store_true');p.add_argument('--development-one',action='store_true')
 a=p.parse_args();assert 0<=a.shard<a.shards
 if a.development_one:
  import os
  assert os.environ.get('WA_DEVELOPMENT')=='1' and not os.environ.get('MD_AK_JOB_ID')
 m=json.loads(Path(a.manifest).read_text());assert m['version']=='wa_failure_backtrack_v1'
 entries=[e for i,e in enumerate(m['selection']) if i%a.shards==a.shard]
 if a.development_one:entries=entries[:1]
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 ready=None if a.audit_only else json.loads(Path(a.ready).read_text())
 if ready is not None:assert ready['mode']=='mixed'
 results=[]
 for task in ('stt','dt','at'):
  spec=m['sources'][task]
  assert sha(spec['path'])==spec['sha256'] and sha(spec['val_path'])==spec['val_sha256']
  cfg=habitat.get_config(str(BENCH/f'habitat-lab/habitat/config/benchmark/nav/track/track_infer_{task}.yaml'),[
   'habitat.dataset.split=train','habitat.dataset.data_path='+spec['path'],
   'habitat.environment.iterator_options.shuffle=false','habitat.simulator.scene_dataset='+SCENES,
   'habitat.simulator.habitat_sim_v0.gpu_device_id=0'])
  ds=habitat.make_dataset(id_dataset=cfg.habitat.dataset.type,config=cfg.habitat.dataset)
  sensors=OmegaConf.to_container(cfg.habitat.simulator.agents.agent_1.sim_sensors,resolve=True)
  fields=('height','width','position','orientation','hfov','sensor_subtype')
  assert all(sensors['jaw_rgb_sensor'][k]==sensors['jaw_panoptic_sensor'][k] for k in fields)
  for e in [e for e in entries if e['task']==task]:
   assert e['partition']=='train'
   ep=ds.episodes[e['dataset_index']]
   assert str(ep.episode_id)==e['episode_id'],(ep.episode_id,e['episode_id'])
   assert Path(ep.scene_id).as_posix().endswith(Path(e['scene_id']).as_posix()),(ep.scene_id,e['scene_id'])
   assert np.allclose(ep.start_position,e['source_start_position'],atol=1e-6,rtol=0)
   assert ep.info['instruction']==e['instruction']
   if a.audit_only:continue
   subset=copy.copy(ds);subset.episodes=[ep];base=out/e['task_episode_uid'].replace(':','_');base.mkdir()
   def run(name,takeover=None,prefix=None):
    random.seed(7);np.random.seed(7);torch.manual_seed(7);torch.cuda.manual_seed_all(7)
    teacher=None
    if takeover is None:
     agent=DiagnosticAgent(ready['url'],cfg.habitat.task.actions.agent_1_base_velocity,'mixed',base/(name+'.trace.jsonl'))
     environment=lambda:agent.diagnostic_env
    else:
     teacher=Teacher(str(base/(name+'_client')),a.teacher_url)
     agent=ReplayThenTeacher(teacher,prefix,takeover,'agent_1_articulated_agent_jaw_rgb')
     environment=lambda:agent.environment
    meta=dict(e,camera_alignment_verified=True,teacher='LightNav' if teacher else 'WA',
              sensor='ideal_simulated_uwb',uwb_noise=0,uwb_delay_s=0,
              manifest_sha256=sha(a.manifest),policy_inputs='WA RGB+BBox+polar only; teacher RGB+text only')
    rec=RecoveryRecorder(base/name,meta,environment,takeover)
    try:
     trained_agent.evaluate_agent(copy.deepcopy(cfg),subset,str(base/(name+'_metrics')),agent_factory=lambda _:agent,recorder=rec)
    finally:
     if teacher:teacher.close()
    result=json.loads((base/name/'result.json').read_text())
    return result,json.loads((base/name/'replay.json').read_text())
   original,prefix=run('student')
   report=dict(episode=e,student=original,attempts=[],accepted=None)
   if a.development_one:
    k=max(0,len(prefix)//2)
    probe,_=run('developer_replay_probe',k,prefix)
    report['developer_probe_only']=dict(step=k,result=probe,training_eligible=False)
   if not original.get('success',False) and original.get('policy_init_valid',True):
    # Descending grid; do not assume teacher recoverability is monotone in time.
    for k in backward_candidates(len(prefix),m['backtrack_gap_steps']):
     result,_=run(f'teacher_{k:04d}',k,prefix)
     report['attempts'].append(dict(step=k,result=result))
     if result.get('success',False) and not result.get('collision',False):
      repeat,_=run(f'teacher_{k:04d}_repeat',k,prefix)
      report['attempts'][-1]['repeat']=repeat
      if repeat.get('success',False) and not repeat.get('collision',False):
       report['accepted']=dict(step=k,path=str(base/f'teacher_{k:04d}'),repeat_path=str(base/f'teacher_{k:04d}_repeat'))
       break
   (base/'search.json').write_text(json.dumps(report,indent=2))
   results.append(report)
   (out/'progress.json').write_text(json.dumps(dict(completed=len(results),expected=len(entries))))
   gc.collect()
 (out/'COMPLETE.json').write_text(json.dumps(dict(audit_only=a.audit_only,episodes=len(results),expected=len(entries),training_release=False)))
if __name__=='__main__':main()
