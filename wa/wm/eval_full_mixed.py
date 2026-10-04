"""Full STT/DT/AT mixed validation; fixed shards, current frozen learned-yaw controller."""
import argparse,copy,json,math,random,os
from pathlib import Path
import numpy as np
import torch,habitat,evt_bench,trained_agent
from wa.wm.diagnostic_agent import DiagnosticAgent
from wa.wm.full_mixed_contract import validate_ready
from evt_full_20260926.common import BENCH,SCENES,sha,scene,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--shard',required=True,type=int);p.add_argument('--ready');p.add_argument('--audit-only',action='store_true');a=p.parse_args()
    if not 0<=a.shard<8:raise ValueError('eight shards required')
    out=Path(a.output);m=json.loads(Path(a.manifest).read_text());assert m['shards']==8 and m['seed_each_episode']==7
    plan=None
    if os.environ.get('WA_RESUME_PLAN'):
        from wa.wm.full_mixed_resume import load_plan
        plan=load_plan(os.environ['WA_RESUME_PLAN'],m,os.environ['WA_RESUME_PLAN_SHA'])
    total=0
    if not a.audit_only:
        assert os.environ.get('WA_DIAG_CONTROLLER')=='learned_yaw_guard_v1'
        ready=json.loads(Path(a.ready).read_text());validate_ready(ready)
        if os.environ.get('WA_REVIEW_VIDEO')=='1':
            from wa.wm.review_recorder import install
            install()
    for task in ['stt','dt','at']:
        spec=m['tasks'][task];assert sha(spec['path'])==spec['sha256']
        config=habitat.get_config(str(BENCH/f'habitat-lab/habitat/config/benchmark/nav/track/track_infer_{task}.yaml'),[
            'habitat.simulator.habitat_sim_v0.gpu_device_id=0','habitat.environment.iterator_options.shuffle=false',
            'habitat.dataset.data_path='+spec['path'],'habitat.simulator.scene_dataset='+SCENES])
        ds=habitat.make_dataset(id_dataset=config.habitat.dataset.type,config=config.habitat.dataset)
        actual={scene(dict(scene_id=e.scene_id))+'/'+str(e.episode_id):e for e in ds.episodes}
        assert len(actual)==1405 and len(spec['episodes'])==1405
        assert set(actual)=={e['key'] for e in spec['episodes']}
        if plan is None:
            selected=[e for e in spec['episodes'] if e['shard']==a.shard]
        else:
            keys={e['key'] for e in plan['lanes'][a.shard] if e['task']==task}
            selected=[e for e in spec['episodes'] if e['key'] in keys]
            assert len(selected)==len(keys)
        for e in selected:assert actual[e['key']].info['instruction']==e['instruction']
        if a.audit_only:print('DATASET_AUDIT_PASS',task,len(actual),len(selected),flush=True);continue
        dest=out/task;dest.mkdir(parents=True,exist_ok=True)
        from omegaconf import OmegaConf
        (dest/'simulator_config.yaml').write_text(OmegaConf.to_yaml(config))
        ready=json.loads(Path(a.ready).read_text());write(out/'model.json',ready)
        for entry in selected:
            agent=DiagnosticAgent(ready['url'],config.habitat.task.actions.agent_1_base_velocity,'mixed',dest/(entry['key'].replace('/','_')+'.trace.jsonl'))
            ep=actual[entry['key']];random.seed(7);np.random.seed(7);torch.manual_seed(7);torch.cuda.manual_seed_all(7)
            subset=copy.copy(ds);subset.episodes=[ep];print('EPISODE_START',task,entry['key'],flush=True)
            trained_agent.evaluate_agent(copy.deepcopy(config),subset,str(dest),agent_factory=lambda _:agent)
            result=json.loads((dest/(entry['key']+'.json')).read_text())
            assert all(math.isfinite(float(result[k])) for k in ['success','collision','following_rate','following_step','total_step'])
            # Invalid first-frame bbox remains a recorded failure, never filtered/replaced.
            result.pop('instruction',None)
            result.update(task=task,key=entry['key'],mode='mixed',noise_mode='zero',controller='learned_yaw_guard_v1',initial_rgb_sha256=sha(dest/'_live'/entry['key']/'step_0000.jpg'))
            with (out/'episodes.jsonl').open('a') as f:f.write(json.dumps(result,allow_nan=False)+'\n')
            total+=1;write(out/'progress.json',dict(completed=total,task=task,last=entry['key']))
            print('EPISODE_RESULT',json.dumps(result),flush=True)
    write(out/('AUDIT_PASS.json' if a.audit_only else 'COMPLETE.json'),dict(episodes=total,shard=a.shard,mode='mixed',noise_mode='zero',controller='learned_yaw_guard_v1'))

if __name__=='__main__':main()
