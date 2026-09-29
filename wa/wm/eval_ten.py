"""Entire requested diagnostic: fixed ten STT episodes; no filtering or automatic expansion."""
import argparse,copy,hashlib,json,random
from pathlib import Path
import numpy as np
import torch,habitat,evt_bench,trained_agent
from scripts.da3.run_evt_ten import IDS,SCENE
from wa.wm.eval_agent import WAAgent

def main():
    p=argparse.ArgumentParser()
    for k in ['benchmark','ready','output','baseline']:p.add_argument('--'+k,required=True)
    a=p.parse_args();root=Path(a.benchmark);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    config=habitat.get_config(str(root/'habitat-lab/habitat/config/benchmark/nav/track/track_infer_stt.yaml'),[
        'habitat.simulator.habitat_sim_v0.gpu_device_id=0','habitat.environment.iterator_options.shuffle=false',
        'habitat.simulator.scene_dataset=/data/nas_ray/home/zeying.gong/datasets/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json'])
    dataset=habitat.make_dataset(id_dataset=config.habitat.dataset.type,config=config.habitat.dataset)
    episodes=[]
    for eid in IDS:
        matches=[e for e in dataset.episodes if str(e.episode_id)==eid and SCENE in e.scene_id]
        if len(matches)!=1:raise ValueError('fixed episode unavailable')
        episodes.append(matches[0])
    ready=json.loads(Path(a.ready).read_text());baseline=Path(a.baseline)
    manifest=json.loads((baseline/'manifest.json').read_text())
    if manifest['episodes']!=IDS or manifest['scene']!=SCENE or manifest['seed_each_episode']!=7:raise ValueError('baseline mismatch')
    (out/'manifest.json').write_text(json.dumps(dict(manifest,method='WA_JEPA',mode='image',limits=['ten STT episodes in one scene only','first-frame target bbox; no text/UWB','different goal interface from text WLA baseline','invalid initialization retained in denominator']),indent=2))
    (out/'model.json').write_text(json.dumps(ready,indent=2))
    from omegaconf import OmegaConf
    (out/'simulator_config.yaml').write_text(OmegaConf.to_yaml(config))
    agent=WAAgent(ready['url'],config.habitat.task.actions.agent_1_base_velocity);rows=[]
    for ep in episodes:
        random.seed(7);np.random.seed(7);torch.manual_seed(7);torch.cuda.manual_seed_all(7)
        subset=copy.copy(dataset);subset.episodes=[ep];print('EPISODE_START',ep.episode_id,flush=True)
        trained_agent.evaluate_agent(copy.deepcopy(config),subset,str(out),agent_factory=lambda _:agent)
        row=dict(episode_id=str(ep.episode_id),**json.loads((out/SCENE/(str(ep.episode_id)+'.json')).read_text()));rows.append(row)
        print('EPISODE_RESULT',json.dumps(row),flush=True)
        (out/'partial_results.json').write_text(json.dumps(rows,indent=2))
    hashes={}
    for eid in IDS:
        paths=[out/'_live'/SCENE/eid/'step_0000.jpg',baseline/'_live'/SCENE/eid/'step_0000.jpg']
        h=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths];hashes[eid]={'wa':h[0],'baseline':h[1],'equal':h[0]==h[1]}
    summary=dict(status='COMPLETE_TEN_EPISODE_DIAGNOSTIC',episodes=len(rows),mode='image',metrics_percent={k:100*float(np.mean([float(r[k]) for r in rows])) for k in ['success','following_rate','collision','finish']},invalid_init_count=sum(r.get('policy_init_valid') is False for r in rows),initial_rgb_pairs=hashes)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print('SUMMARY',json.dumps(summary),flush=True)
    if not all(r['equal'] for r in hashes.values()):raise RuntimeError('initial RGB mismatch; paired comparison invalid')

if __name__=='__main__':main()
