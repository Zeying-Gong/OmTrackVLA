import argparse,copy,json,random
from pathlib import Path
import numpy as np
import torch,habitat,evt_bench,trained_agent
from lightnav_transport_20260927.agent import Agent
from evt_full_20260926.common import BENCH,SCENES,sha,scene,WLA
from wa.wm.review_recorder import ReviewRecorder

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--server',required=True);p.add_argument('--shard',type=int,choices=[0,1],required=True);p.add_argument('--plan',required=True);a=p.parse_args()
 manifest=WLA/'evt_full_20260926/manifest.json';m=json.loads(manifest.read_text());plan=json.loads(Path(a.plan).read_text());assert sha(manifest)==plan['manifest_sha256']
 out=Path(a.output);agent=Agent(str(out),a.server);rows=[]
 for task in ('stt','dt','at'):
  spec=m['tasks'][task];assert sha(spec['path'])==spec['sha256']
  cfg=habitat.get_config(str(BENCH/f'habitat-lab/habitat/config/benchmark/nav/track/track_infer_{task}.yaml'),['habitat.simulator.habitat_sim_v0.gpu_device_id=0','habitat.environment.iterator_options.shuffle=false','habitat.dataset.data_path='+spec['path'],'habitat.simulator.scene_dataset='+SCENES])
  ds=habitat.make_dataset(id_dataset=cfg.habitat.dataset.type,config=cfg.habitat.dataset);actual={scene(dict(scene_id=e.scene_id))+'/'+str(e.episode_id):e for e in ds.episodes};dest=out/task;dest.mkdir()
  for key in plan['development'][task][a.shard::2]:
   ep=actual[key];random.seed(7);np.random.seed(7);torch.manual_seed(7);torch.cuda.manual_seed_all(7);subset=copy.copy(ds);subset.episodes=[ep]
   agent.episode_action_count=0;agent.episode_fallback_count=0
   rec=ReviewRecorder(dest/'_review'/key,'LightNav RGB+text')
   trained_agent.evaluate_agent(copy.deepcopy(cfg),subset,str(dest),agent_factory=lambda _:agent,recorder=rec)
   row=json.loads((dest/(key+'.json')).read_text());row.pop('instruction',None);row.update(task=task,key=key,initial_rgb_sha256=sha(dest/'_live'/key/'step_0000.jpg'),released_fallback_count=agent.episode_fallback_count)
   rows.append(row)
   with (out/'episodes.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 assert len(rows)==12
 (out/'COMPLETE.json').write_text(json.dumps(dict(episodes=12)))
if __name__=='__main__':main()
