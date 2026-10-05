"""Bounded paired static render regression. Not a closed-loop SR experiment."""
import argparse, copy, hashlib, json, random
from pathlib import Path
import numpy as np, torch, habitat, evt_bench
from omegaconf import open_dict
from PIL import Image, ImageDraw
from wa.wm.semantic_scene import prepare_episode, provenance, validate_runtime
R=Path('/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928')
B=R.parent/'OmTrackVLA-da3-polar-20260924'
def sha(array): return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument("scene");p.add_argument("episode")
    p.add_argument("--task",default="stt",choices=["stt","dt","at"]);a=p.parse_args()
    out=R/("artifacts/residual7_rgbstage_probe_v1_"+a.task+"_"+a.scene+"_"+a.episode)
    out.mkdir(exist_ok=False)
    manifest=json.loads((R.parent/'WLA-EVT-20260925/evt_full_20260926/manifest.json').read_text())
    config=habitat.get_config(str(B/f'habitat-lab/habitat/config/benchmark/nav/track/track_infer_{a.task}.yaml'),[
        'habitat.simulator.habitat_sim_v0.gpu_device_id=0','habitat.environment.iterator_options.shuffle=false',
        'habitat.dataset.data_path='+manifest['tasks'][a.task]['path']])
    ds=habitat.make_dataset(id_dataset=config.habitat.dataset.type,config=config.habitat.dataset)
    ep=next(e for e in ds.episodes if str(e.episode_id)==a.episode and Path(e.scene_id).name.split('.')[0]==a.scene)
    report={"key":a.scene+"/"+a.episode,"task":a.task,"repair":provenance(),"modes":{}}
    for mode in ("original","fixed"):
        with habitat.config.read_write(config):
            with open_dict(config.habitat.simulator.habitat_sim_v0):
                config.habitat.simulator.habitat_sim_v0.load_semantic_mesh = mode=="original"
        selected=prepare_episode(ep,B)
        if mode=="fixed":
            scene=(B/ep.scene_id).resolve()
            cfg={"stages":{"default_attributes":{"up":[0,0,1],"front":[0,1,0],
                 "semantic_up":[0,0,1],"semantic_front":[0,1,0],
                 "semantic_asset":str(scene)}}}
            config_path=out/"rgbstage.scene_dataset_config.json"
            config_path.write_text(json.dumps(cfg))
            selected.scene_dataset_config=str(config_path)
        subset=copy.copy(ds);subset.episodes=[selected]
        random.seed(7);np.random.seed(7);torch.manual_seed(7);torch.cuda.manual_seed_all(7)
        with habitat.TrackEnv(config=copy.deepcopy(config),dataset=subset) as env:
            env.reset();sim=env.sim
            print("SIMCONFIG",mode,str(sim.sim_config.sim_cfg),flush=True)
            obs=sim.get_sensor_observations();det=env.task._get_observations(selected)
            sem=np.asarray(obs['agent_1_articulated_agent_jaw_panoptic'])
            rgb=np.asarray(obs['agent_1_articulated_agent_jaw_rgb'])[...,:3]
            target=sem==int(ep.info['main_human_semantic_id'])
            attrs=sim.get_stage_initialization_template()
            print("TARGET_NODES", mode, [(str(n.semantic_id),str(n.absolute_transformation())) for n in sim.agents_mgr[0].articulated_agent.sim_obj.visual_scene_nodes],flush=True)
            record={"rgb_sha":sha(rgb),"depth_sha":sha(obs['agent_1_articulated_agent_jaw_depth']) if 'agent_1_articulated_agent_jaw_depth' in obs else None,
                    "target_pixels":int(target.sum()),"expected_id":int(ep.info["main_human_semantic_id"]),"id_counts":dict(zip(*[v.tolist() for v in np.unique(sem,return_counts=True)])),"bbox":det['agent_1_main_humanoid_detector_sensor']['box'].tolist(),
                    "facing":bool(det['agent_1_main_humanoid_detector_sensor']['facing']),
                    "poses":[np.asarray(sim.agents_mgr[i].articulated_agent.sim_obj.transformation).tolist() for i in range(len(sim.agents_mgr))],
                    "camera":np.asarray(sim._sensors['agent_1_articulated_agent_jaw_rgb']._sensor_object.node.absolute_transformation()).tolist(),
                    "stage":{k:str(getattr(attrs,k)) for k in dir(attrs) if any(s in k for s in ['orient','asset_handle','friction','restitution','scale','gravity'])}}
            report["modes"][mode]=record
            image=Image.fromarray(rgb);image.save(out/(mode+"_rgb.png"))
            overlay=rgb.copy();overlay[target]=(0.5*overlay[target]+np.array([0,127,0])).astype(np.uint8)
            view=Image.fromarray(overlay);draw=ImageDraw.Draw(view)
            if target.any():draw.rectangle(record["bbox"],outline="red",width=2)
            view.save(out/(mode+"_mask.png"));np.save(out/(mode+"_semantic.npy"),sem)
    old,new=report["modes"]["original"],report["modes"]["fixed"]
    same_stage={k:v for k,v in old["stage"].items() if not k.startswith("semantic_orient")}
    new_stage={k:v for k,v in new["stage"].items() if not k.startswith("semantic_orient")}
    report["checks"]={"rgb_identical":old["rgb_sha"]==new["rgb_sha"],
        "depth_identical":old["depth_sha"]==new["depth_sha"] if old["depth_sha"] is not None else None,"all_agent_poses_identical":old["poses"]==new["poses"],
        "camera_identical":old["camera"]==new["camera"],"nonsemantic_stage_identical":same_stage==new_stage,
        "fixed_bbox_valid":new["bbox"][2]>new["bbox"][0] and new["bbox"][3]>new["bbox"][1]}
    (out/"report.json").write_text(json.dumps(report,indent=2,default=float))
    print("PAIRED_CHECK",json.dumps(report),flush=True)
    assert all(v is not False for k,v in report["checks"].items() if k not in ("fixed_bbox_valid","nonsemantic_stage_identical")),report["checks"]
if __name__=="__main__":main()
