"""Full STT/DT/AT validation; mixed default, opt-in image, fixed learned-yaw controller."""
import argparse,copy,json,math,random,os
from pathlib import Path
import numpy as np
import torch,habitat,evt_bench,trained_agent
from wa.wm.diagnostic_agent import DiagnosticAgent
from wa.wm.full_mixed_contract import validate_ready
from wa.wm.student_eval_contract import model_contract,evaluation_mode
from wa.wm.student_eval_partition import task_scope
from evt_full_20260926.common import BENCH,SCENES,sha,scene,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--shard',required=True,type=int);p.add_argument('--ready');p.add_argument('--audit-only',action='store_true');a=p.parse_args()
    if not 0<=a.shard<8:raise ValueError('eight shards required')
    out=Path(a.output);m=json.loads(Path(a.manifest).read_text());assert m['shards']==8 and m['seed_each_episode']==7
    plan=None
    contract=model_contract(os.environ);mode=evaluation_mode(os.environ);repairs={}
    if contract:
        from wa.wm.initial_bbox_repair import load_plan,VERSION as REPAIR_VERSION
        from wa.wm.initial_bbox_repair_agent import InitialBBoxRepairAgent
        repair_plan=load_plan(os.environ['WA_INIT_REPAIR_PLAN'],os.environ['WA_INIT_REPAIR_PLAN_SHA'])
        repairs={(r['task'],r['key']):r for r in repair_plan['repairs']}
    semantic_fix=os.environ.get('WA_SEMANTIC_PLY_FIX','')
    if semantic_fix:
        from wa.wm.semantic_scene import VERSION,prepare_episode,provenance,is_mp3d_ply_scene
        if semantic_fix!=VERSION:raise ValueError('unknown semantic repair version')
        if os.environ.get('WA_RESUME_PLAN'):raise ValueError('cannot merge old semantic protocol results')
        write(out/'semantic_protocol.json',provenance())
    if os.environ.get('WA_RESUME_PLAN'):
        from wa.wm.full_mixed_resume import load_plan
        plan=load_plan(os.environ['WA_RESUME_PLAN'],m,os.environ['WA_RESUME_PLAN_SHA'])
    targeted=bool(os.environ.get('WA_TARGETED_PLAN'))
    if targeted:
        if not semantic_fix or plan is not None:raise ValueError('targeted plan requires semantic-only protocol')
        from wa.wm.semantic_targeted import load_targeted,priority_barrier
        plan=load_targeted(os.environ['WA_TARGETED_PLAN'],m,os.environ['WA_TARGETED_PLAN_SHA'])
    total=0
    if not a.audit_only:
        assert os.environ.get('WA_DIAG_CONTROLLER')=='learned_yaw_guard_v1'
        ready=json.loads(Path(a.ready).read_text());validate_ready(ready,mode=mode,**contract)
        if os.environ.get('WA_REVIEW_VIDEO')=='1':
            from wa.wm.review_recorder import install
            install(mode=mode)
    schedule=[(phase,task) for phase in ([True,False] if targeted else [None]) for task in task_scope(os.environ)]
    for phase,task in schedule:
        if targeted and phase is False and task=='stt' and not a.audit_only:priority_barrier(out)
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
            keys={e['key'] for e in plan['lanes'][a.shard] if e['task']==task and (not targeted or e['priority']==phase)}
            selected=[e for e in spec['episodes'] if e['key'] in keys]
            assert len(selected)==len(keys)
        for e in selected:assert actual[e['key']].info['instruction']==e['instruction']
        if a.audit_only:
            total+=len(selected)
            print('DATASET_AUDIT_PASS',task,len(actual),len(selected),flush=True)
            continue
        dest=out/task;dest.mkdir(parents=True,exist_ok=True)
        from omegaconf import OmegaConf
        (dest/'simulator_config.yaml').write_text(OmegaConf.to_yaml(config))
        ready=json.loads(Path(a.ready).read_text());write(out/'model.json',ready)
        for entry in selected:
            repair=repairs.get((task,entry['key']))
            args=(ready['url'],config.habitat.task.actions.agent_1_base_velocity,mode,dest/(entry['key'].replace('/','_')+'.trace.jsonl'))
            agent=InitialBBoxRepairAgent(*args,repair=repair) if repair is not None else DiagnosticAgent(*args)
            ep=actual[entry['key']];random.seed(7);np.random.seed(7);torch.manual_seed(7);torch.cuda.manual_seed_all(7)
            if semantic_fix:ep=prepare_episode(ep,BENCH)
            subset=copy.copy(ds);subset.episodes=[ep];print('EPISODE_START',task,entry['key'],flush=True)
            trained_agent.evaluate_agent(copy.deepcopy(config),subset,str(dest),agent_factory=lambda _:agent)
            result=json.loads((dest/(entry['key']+'.json')).read_text())
            assert all(math.isfinite(float(result[k])) for k in ['success','collision','following_rate','following_step','total_step'])
            # Invalid first-frame bbox remains a recorded failure, never filtered/replaced.
            if contract:
                result.update(checkpoint_sha256=contract['checkpoint_sha'],checkpoint_step=contract['step'])
                if not hasattr(agent,'initial_pair_evidence'):raise ValueError('missing student paired-start evidence')
                result['initial_pair_evidence']=agent.initial_pair_evidence
            if repair is not None:
                if not agent.repair_applied:raise ValueError('frozen repair was not applied')
                result.update(initialization_repair=REPAIR_VERSION,
                    initialization_repair_plan_sha256=os.environ['WA_INIT_REPAIR_PLAN_SHA'])
            result.pop('instruction',None)
            result.update(task=task,key=entry['key'],mode=mode,noise_mode='zero',controller='learned_yaw_guard_v1',initial_rgb_sha256=sha(dest/'_live'/entry['key']/'step_0000.jpg'))
            if semantic_fix:
                result.update(semantic_protocol=semantic_fix,semantic_ply_repaired=is_mp3d_ply_scene(ep.scene_id))
            with (out/'episodes.jsonl').open('a') as f:f.write(json.dumps(result,allow_nan=False)+'\n')
            total+=1;write(out/'progress.json',dict(completed=total,task=task,last=entry['key']))
            print('EPISODE_RESULT',json.dumps(result),flush=True)
    write(out/('AUDIT_PASS.json' if a.audit_only else 'COMPLETE.json'),dict(episodes=total,shard=a.shard,mode=mode,noise_mode='zero',controller='learned_yaw_guard_v1'))

if __name__=='__main__':main()
