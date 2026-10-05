"""Convert audited in-set teacher demonstrations to the unchanged robot contract."""
import argparse, json
from pathlib import Path
import numpy as np
from wa.data import sha
from wa.wm.robot_data import transition_records, CONTRACT
from wa.wm.dual_teacher_selection import EXPERIMENT

def convert(release, base, baseindex, out, development=False):
    release,base,baseindex,out=map(lambda p:Path(p).resolve(),(release,base,baseindex,out))
    m=json.loads(release.read_text())
    if m["experiment"]!=EXPERIMENT or not m["paired_outcomes_validated"]:
        raise ValueError("explicit audited in-set release required")
    if not development and m["expected"]!=4215:raise ValueError("full paired collection required")
    entries=m["teacher_demonstrations"]
    if not entries or len({(e["task"],e["key"]) for e in entries})!=len(entries):
        raise ValueError("empty or duplicate demonstrations")
    out.mkdir(parents=True,exist_ok=False);idx=out/"index";idx.mkdir()
    episodes=[];poses=[];histories=[];episodeids=[];valid=[];stats=[];row_start=0
    heldout=json.loads((base/"heldout_episodes.json").read_text())
    heldscenes={Path(x["scene"]).name.split(".")[0] for x in heldout}
    overlap=set();frame_inventory={}
    for ep,entry in enumerate(entries):
        root=Path(entry["branch"]).resolve()
        for name,h in entry["hashes"].items():
            if sha(root/name)!=h:raise ValueError("demonstration artifact changed")
        meta=json.loads((root/"metadata.json").read_text())
        if meta["experiment"]!=EXPERIMENT or meta["partition"]!="evaluation_adaptation":
            raise ValueError("unmarked data overlap")
        result=json.loads((root/"result.json").read_text())
        if not result["success"] or result["collision"]:raise ValueError("unsuccessful demonstration")
        obs=json.loads((root/"observations.json").read_text())
        wins=json.loads((root/"windows.json").read_text())
        indices=np.array([w["current_index"] for w in wins],dtype=np.int64)
        if indices.tolist()!=entry["window_indices"] or not len(indices):raise ValueError("window scope mismatch")
        scene=Path(meta["scene_id"]).name.split(".")[0]
        if scene in heldscenes:overlap.add(scene)
        times=np.array([o["timestamp_s"] for o in obs])
        rot=np.array([o["robot_rotation_world_from_body"] for o in obs])
        pos=np.array([o["robot_position_world"] for o in obs])
        query=times[indices,None]+np.arange(1,8)/10.
        if (np.diff(times)<=0).any() or query.max()>times[-1]+1e-8:raise ValueError("invalid interpolation time")
        futurepos=np.stack([np.interp(query.ravel(),times,pos[:,axis]).reshape(-1,7) for axis in range(3)],-1)
        local=np.einsum("nki,nij->nkj",futurepos-pos[indices,None],rot[indices])
        xy=np.stack([local[...,0],-local[...,2]],-1)
        if not np.allclose(xy,np.array([w["trajectory_xy_m"] for w in wins]),rtol=0,atol=1e-6):
            raise ValueError("executed pose labels disagree")
        yaw=np.unwrap(np.arctan2(rot[:,0,2],rot[:,0,0]))
        futureyaw=np.interp(query.ravel(),times,yaw).reshape(-1,7)-yaw[indices,None]
        pose=np.concatenate([xy,np.sin(futureyaw)[...,None],np.cos(futureyaw)[...,None]],-1).astype(np.float32)
        history=np.searchsorted(times,times[indices,None]+np.array([-1.5,-1.,-.5,0.]),side="right")-1
        history=np.minimum(np.maximum(history,0),indices[:,None]).astype(np.int32)
        if pose.shape!=(len(wins),7,4) or not np.isfinite(pose).all():raise ValueError("bad SE2 label")
        commands,bad,s=transition_records(root,obs)
        end=np.searchsorted(times,times[indices]+.7);prefix=np.r_[0,np.cumsum(bad)]
        supported=(indices>=4)&(end<len(times))
        supported &= prefix[np.minimum(end,len(bad))]==prefix[history[:,0]]
        valid.extend((row_start+np.flatnonzero(supported)).tolist());row_start+=len(wins)
        s.update(episode=ep,retained=int(supported.sum()),rows=len(wins),
                 source_sha256={n:sha(root/n) for n in ("metadata.json","observations.json","actions.json")})
        stats.append(s);poses.append(pose);histories.append(history);episodeids.append(np.full(len(wins),ep,dtype=np.int32))
        frames=[o["frame"] for o in obs]
        hashes={}
        for name in frames:
            image=(root/name).resolve()
            if image.parent!=root:raise ValueError("frame outside branch")
            hashes[name]=sha(image)
        frame_inventory[str(root)]=hashes
        episodes.append(dict(root=str(root),frames=frames,task=entry["task"],episode_uid=entry["task"]+":"+entry["key"],
            scene=meta["scene_id"],takeover_step=0,category="dual_teacher_inset",teacher=entry["teacher"]))
    for key,values in (("pose",poses),("history",histories),("episode",episodeids)):
        np.save(out/f"train_{key}.npy",np.concatenate(values))
    (out/"train_episodes.json").write_text(json.dumps(episodes))
    (out/"source_image_hashes.json").write_text(json.dumps(frame_inventory))
    for suffix in ("pose.npy","history.npy","episode.npy","episodes.json"):
        (out/f"heldout_{suffix}").symlink_to(base/f"heldout_{suffix}")
    complete=dict(experiment=EXPERIMENT,development_only=development,source_release=str(release),source_sha256=sha(release),
        inputs="causal RGB/template/current simulated polar UWB; no text; future only supervised targets",
        heldout_interpretation="Original files unchanged; NO clean generalization claim after evaluation-set adaptation",
        known_heldout_scene_overlap=sorted(overlap),source_image_inventory_sha256=sha(out/"source_image_hashes.json"),
        files={f"{part}_{suffix}":sha(out/f"{part}_{suffix}") for part in ("train","heldout") for suffix in ("pose.npy","history.npy","episode.npy","episodes.json")})
    (out/"complete.json").write_text(json.dumps(complete,indent=2))
    np.save(idx/"train_valid.npy",np.array(valid,dtype=np.int64))
    (idx/"train_episodes_audit.json").write_text(json.dumps(dict(stats=stats,errors=[])))
    for name in ("heldout_valid.npy","heldout_episodes_audit.json"):(idx/name).symlink_to(baseindex/name)
    original=json.loads((baseindex/"audit.json").read_text())
    audit=dict(contract=CONTRACT,experiment=EXPERIMENT,cache_complete_sha256=sha(out/"complete.json"),
        splits=dict(train=dict(original=row_start,retained=len(valid),excluded=row_start-len(valid),index_sha256=sha(idx/"train_valid.npy"),episode_errors=[]),heldout=original["splits"]["heldout"]))
    (idx/"audit.json").write_text(json.dumps(audit,indent=2))
    return dict(episodes=len(episodes),candidate_windows=row_start,retained_windows=len(valid),
                development_only=development,known_heldout_scene_overlap=len(overlap),cache=str(out))
def main():
    p=argparse.ArgumentParser()
    for key in ("release","base-cache","base-index","output"):p.add_argument("--"+key,required=True)
    p.add_argument("--development",action="store_true");a=p.parse_args()
    print(json.dumps(convert(a.release,a.base_cache,a.base_index,a.output,a.development),indent=2))
if __name__=="__main__":main()
