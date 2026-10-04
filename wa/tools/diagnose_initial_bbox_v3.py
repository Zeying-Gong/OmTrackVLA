import json,copy,random,os,sys
from pathlib import Path
import numpy as np,torch,habitat,evt_bench
from PIL import Image
R=Path('/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928')
B=R.parent/'OmTrackVLA-da3-polar-20260924'
M=json.loads((R.parent/'WLA-EVT-20260925/evt_full_20260926/manifest.json').read_text())
OUT=R/('artifacts/init_bbox_probe_20261004_v3_'+sys.argv[1]);OUT.mkdir(exist_ok=False)
config=habitat.get_config(str(B/'habitat-lab/habitat/config/benchmark/nav/track/track_infer_stt.yaml'),['habitat.simulator.habitat_sim_v0.gpu_device_id=0','habitat.environment.iterator_options.shuffle=false','habitat.dataset.data_path='+M['tasks']['stt']['path'],'habitat.simulator.scene_dataset=/data/nas_ray/home/zeying.gong/datasets/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json'])
ds=habitat.make_dataset(id_dataset=config.habitat.dataset.type,config=config.habitat.dataset)
ep=next(e for e in ds.episodes if e.episode_id==sys.argv[1] and 'oLBMNvg9in8' in e.scene_id)
ds.episodes=[ep];random.seed(7);np.random.seed(7);torch.manual_seed(7)
with habitat.TrackEnv(config=config,dataset=ds) as env:
    env.reset();sim=env.sim
    report={'episode':'stt/oLBMNvg9in8/'+sys.argv[1],'expected':ep.info['main_human_semantic_id'],'model':ep.info['main_humanoid_name']}
    target=sim.agents_mgr[0].articulated_agent.sim_obj
    report['target_nodes']=[{'id':n.semantic_id,'translation':list(n.translation)} for n in target.visual_scene_nodes]
    report['cameras']={k:np.asarray(sim._sensors[k]._sensor_object.node.absolute_transformation()).tolist() for k in ['agent_1_articulated_agent_jaw_rgb','agent_1_articulated_agent_jaw_panoptic']}
    report['semantic_object_160']=[{'id':o.id,'category':o.category.name(),'center':list(o.aabb.center),'sizes':list(o.aabb.sizes)} for o in sim.semantic_scene.objects if o is not None and o.semantic_id==160]
    report['frames']=[]
    for i in range(2):
        obs=sim.get_sensor_observations();det=env.task._get_observations(ep)
        sem=np.asarray(obs['agent_1_articulated_agent_jaw_panoptic']);rgb=np.asarray(obs['agent_1_articulated_agent_jaw_rgb'])[...,:3]
        ids,counts=np.unique(sem,return_counts=True)
        frame={'index':i,'semantic_counts':dict(zip(map(str,ids),map(int,counts))),'box':det['agent_1_main_humanoid_detector_sensor']['box'].tolist()}
        report['frames'].append(frame);np.save(OUT/f'semantic_{i}.npy',sem);Image.fromarray(rgb).save(OUT/f'rgb_{i}.png')
    (OUT/'report.json').write_text(json.dumps(report,indent=2,default=float));print('PROBE_REPORT',json.dumps(report,default=float),flush=True)
