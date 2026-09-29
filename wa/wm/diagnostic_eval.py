"""Predeclared paired development diagnostic; distinct confirmation scenes reserved."""
import argparse,copy,hashlib,json,random
from pathlib import Path
import numpy as np
import torch,habitat,evt_bench,trained_agent
from wa.wm.diagnostic_agent import DiagnosticAgent
from evt_full_20260926.common import BENCH,SCENES,sha,scene

def main():
    p=argparse.ArgumentParser()
    for k in ('manifest','plan','ready','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--shard',type=int,choices=[0,1],required=True)
    p.add_argument('--split',choices=['development','confirmation'],default='development')
    a=p.parse_args();m=json.loads(Path(a.manifest).read_text());plan=json.loads(Path(a.plan).read_text())
    assert sha(a.manifest)==plan['manifest_sha256']
    ready=json.loads(Path(a.ready).read_text());out=Path(a.output);rows=[]
    for task in ('stt','dt','at'):
        spec=m['tasks'][task];assert sha(spec['path'])==spec['sha256']
        cfg=habitat.get_config(str(BENCH/f'habitat-lab/habitat/config/benchmark/nav/track/track_infer_{task}.yaml'),[
            'habitat.simulator.habitat_sim_v0.gpu_device_id=0','habitat.environment.iterator_options.shuffle=false',
            'habitat.dataset.data_path='+spec['path'],'habitat.simulator.scene_dataset='+SCENES])
        dataset=habitat.make_dataset(id_dataset=cfg.habitat.dataset.type,config=cfg.habitat.dataset)
        actual={scene(dict(scene_id=e.scene_id))+'/'+str(e.episode_id):e for e in dataset.episodes}
        keys=plan[a.split][task][a.shard::2]
        dest=out/task;dest.mkdir(exist_ok=False)
        for key in keys:
            ep=actual[key];random.seed(7);np.random.seed(7);torch.manual_seed(7);torch.cuda.manual_seed_all(7)
            subset=copy.copy(dataset);subset.episodes=[ep]
            trace=dest/(key.replace('/','_')+'.trace.jsonl')
            agent=DiagnosticAgent(ready['url'],cfg.habitat.task.actions.agent_1_base_velocity,ready['mode'],trace)
            print('EPISODE_START',task,key,flush=True)
            trained_agent.evaluate_agent(copy.deepcopy(cfg),subset,str(dest),agent_factory=lambda _:agent)
            row=json.loads((dest/(key+'.json')).read_text())
            row.pop('instruction',None)
            row.update(task=task,key=key,mode=ready['mode'],noise_mode=ready['noise_mode'],initial_rgb_sha256=sha(dest/'_live'/key/'step_0000.jpg'))
            rows.append(row)
            with (out/'episodes.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print('EPISODE_RESULT',json.dumps(row),flush=True)
    assert len(rows)==12 and len({(r['task'],r['key']) for r in rows})==12
    (out/'COMPLETE.json').write_text(json.dumps(dict(episodes=12,split=a.split)))
if __name__=='__main__':main()
