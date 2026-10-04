"""Bounded, no-action render ablation. Never modifies shared runtime/assets."""
import json, random, sys, hashlib
from pathlib import Path
import numpy as np, torch, habitat, habitat_sim, evt_bench
from PIL import Image
R = Path('/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928')
B = R.parent / 'OmTrackVLA-da3-polar-20260924'
M = json.loads((R.parent/'WLA-EVT-20260925/evt_full_20260926/manifest.json').read_text())
eid = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else 'baseline'
out = R / ('artifacts/semantic_render_v7_' + eid + '_' + mode)
out.mkdir(exist_ok=False)
dataset_path = Path('/data/nas_ray/home/zeying.gong/datasets/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json')
if mode == 'ply_identity':
    dataset_config = json.loads(dataset_path.read_text())
    for section in dataset_config.values():
        if isinstance(section, dict):
            for extension, paths in section.get('paths', {}).items():
                section['paths'][extension] = [str(dataset_path.parent / p) for p in paths]
    dataset_config['stages']['default_attributes'].update(semantic_up=[0,1,0], semantic_front=[0,0,-1])
    dataset_path = out / 'probe.scene_dataset_config.json'
    dataset_path.write_text(json.dumps(dataset_config, indent=2))
config = habitat.get_config(str(B/'habitat-lab/habitat/config/benchmark/nav/track/track_infer_stt.yaml'), [
    'habitat.simulator.habitat_sim_v0.gpu_device_id=0',
    'habitat.environment.iterator_options.shuffle=false',
    'habitat.dataset.data_path=' + M['tasks']['stt']['path'],
    'habitat.simulator.scene_dataset=' + str(dataset_path)])
ds = habitat.make_dataset(id_dataset=config.habitat.dataset.type, config=config.habitat.dataset)
ep = next(e for e in ds.episodes if e.episode_id == eid and 'oLBMNvg9in8' in e.scene_id)
print('ORIGINAL_EPISODE_SCENE_DATASET', ep.scene_dataset_config, flush=True)
if mode == 'ply_identity':
    dataset_path.write_text(json.dumps({'stages': {'default_attributes': {'up': [0,0,1], 'front': [0,1,0], 'semantic_up': [0,1,0], 'semantic_front': [0,0,-1]}}}))
    ep.scene_dataset_config = str(dataset_path)
ds.episodes = [ep]
random.seed(7); np.random.seed(7); torch.manual_seed(7)
report = {'episode': 'stt/oLBMNvg9in8/' + eid, 'runtime': habitat_sim.__file__, 'frames': []}
with habitat.TrackEnv(config=config, dataset=ds) as env:
    env.reset()
    sim = env.sim
    attrs = sim.get_stage_initialization_template()
    report['stage'] = {k: str(getattr(attrs, k)) for k in dir(attrs) if any(s in k for s in ['orient', 'asset_handle'])}
    key = 'agent_1_articulated_agent_jaw_panoptic'
    sensor = sim._sensors[key]
    node = sensor._sensor_object.node
    root = sim.get_active_scene_graph().get_root_node()
    report['root'] = np.asarray(root.absolute_transformation()).tolist()
    report['semantic_root'] = np.asarray(sim.get_active_semantic_scene_graph().get_root_node().absolute_transformation()).tolist()
    report['parent'] = np.asarray(node.parent.absolute_transformation()).tolist()
    report['local'] = np.asarray(node.transformation).tolist()
    report['absolute'] = np.asarray(node.absolute_transformation()).tolist()
    report['expected'] = int(ep.info['main_human_semantic_id'])
    def capture(label):
        obs = sim.get_sensor_observations()
        sem = np.asarray(obs[key]).copy()
        rgb = np.asarray(obs['agent_1_articulated_agent_jaw_rgb'])[...,:3].copy()
        ids, counts = np.unique(sem, return_counts=True)
        ys, xs = np.where(sem == report['expected'])
        rec = {'label': label, 'counts': dict(zip(map(str, ids), map(int, counts))),
               'target_pixels': int(len(xs)), 'rgb_sha': hashlib.sha256(rgb.tobytes()).hexdigest(),
               'bbox': [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None}
        report['frames'].append(rec)
        np.save(out/(label+'.npy'), sem)
        Image.fromarray(rgb).save(out/(label+'.png'))
        print('CAPTURE', json.dumps(rec), flush=True)
    capture('baseline')
    old_cull = sim.frustum_culling
    sim.frustum_culling = False
    capture('no_frustum')
    sim.frustum_culling = old_cull
    parent, local = node.parent, node.transformation
    absolute = node.absolute_transformation()
    try:
        node.parent = root
        node.transformation = root.absolute_transformation().inverted() @ absolute
        capture('root_parent')
    finally:
        node.parent = parent
        node.transformation = local
    capture('restored')
    report['sensor_api'] = [k for k in dir(sensor._sensor_object) if 'draw' in k or 'render' in k]
    report['renderer_api'] = [k for k in dir(sim.renderer) if not k.startswith('_')]
(out/'report.json').write_text(json.dumps(report, indent=2, default=float))
print('REPORT', json.dumps(report, default=float), flush=True)
