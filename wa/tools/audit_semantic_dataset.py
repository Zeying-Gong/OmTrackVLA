"""Read-only real Habitat dataset preflight for all 4215 episode definitions."""
import collections, hashlib, json
from pathlib import Path
import habitat, evt_bench
from wa.wm.semantic_scene import prepare_episode, is_mp3d_ply_scene, provenance
R=Path('/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928')
B=R.parent/'OmTrackVLA-da3-polar-20260924'
M=R.parent/'WLA-EVT-20260925/evt_full_20260926/manifest.json'
manifest=json.loads(M.read_text())
report={"repair":provenance(),"tasks":{}}
for task,spec in manifest["tasks"].items():
    assert hashlib.sha256(Path(spec["path"]).read_bytes()).hexdigest()==spec["sha256"]
    cfg=habitat.get_config(str(B/f'habitat-lab/habitat/config/benchmark/nav/track/track_infer_{task}.yaml'),
                          ['habitat.dataset.data_path='+spec["path"]])
    ds=habitat.make_dataset(id_dataset=cfg.habitat.dataset.type,config=cfg.habitat.dataset)
    counts=collections.Counter()
    for ep in ds.episodes:
        fixed=prepare_episode(ep,B)
        for key in ("info","start_position","start_rotation","scene_id","episode_id"):
            assert getattr(fixed,key)==getattr(ep,key),key
        if is_mp3d_ply_scene(ep.scene_id):
            assert ep.scene_dataset_config=="default"
            counts["mp3d_fixed"]+=1
        else:
            assert ep.scene_dataset_config==fixed.scene_dataset_config
            counts["unchanged"]+=1
    assert len(ds.episodes)==1405
    assert counts=={"mp3d_fixed":721,"unchanged":684}
    report["tasks"][task]=dict(counts)
out=R/'artifacts/semantic_dataset_preflight_v1.json'
with out.open('x') as f:json.dump(report,f,indent=2)
print("SEMANTIC_DATASET_PREFLIGHT_PASS",json.dumps(report))
