"""Scoped exact training-sample deduplication; does not edit a cache or sampling.

Compare new effective failure-state windows to EVERY old valid window sharing
task/key, regardless of teacher, absolute step, timestamp or source directory.
Float32 nonimage inputs/labels prefilter; all consumed image tensors decide the
remaining candidates. Cross-key/scene equality is explicitly outside scope.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import platform
import shlex
import sys
from PIL import __version__ as pillow_version
from pathlib import Path
import numpy as np
import torch
from wa.data import tensor_image
from wa.wm.robot_data import transition_records, CONTRACT
from wa.wm.failure_state_numeric import audit_numeric_windows
from wa.tools.build_failure_state_cache import Pins, require, same, path_file, path_dir, file_sha

SCHEMA="failure_state_exact_dedup_v1"
EXPECTED=dict(episodes=96,candidate_windows=7396,valid_windows=6864)
OLD_COMPLETE_SHA="79e5a7dc43b61a23bb64d5676c4c1cac17b459a5e390b308377012e582ba1a8d"
OLD_TRAINING_AUDIT=Path("/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/hard_stt_training_audit_61609_v1.json")
OLD_TRAINING_AUDIT_SHA="59d5b468eec4283db1a1d87ad7b9a9d1bc2123739cffc38cebfc9a1e14dcc727"
NONIMAGE=("polar","geometry","times","pose","commands","proprio")
IMAGES=("template","rgb","wm_rgb","future")


def tensor_digest(value):
    a=np.ascontiguousarray(value)
    require(a.dtype==np.dtype("float32") and np.isfinite(a).all(),"finite actual float32 tensor required")
    h=hashlib.sha256()
    h.update(json.dumps([a.dtype.str,list(a.shape)],separators=(",",":")).encode())
    h.update(b"\0");h.update(a.tobytes())
    return h.hexdigest()


def digest_parts(pieces):
    return hashlib.sha256(json.dumps(pieces,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def scalar_parts(obs,times,commands,history,pose):
    history=np.asarray(history);now=int(history[-1]);wi=np.arange(now-3,now+1)
    require(history.shape==(4,) and now>=4 and (history>=0).all()
            and (history<=now).all() and wi[-1]<len(commands),"invalid consumed history/commands")
    item=obs[now]
    delta=np.asarray(item["target_position_world_label_only"])-np.asarray(item["robot_position_world"])
    local=delta@np.asarray(item["robot_rotation_world_from_body"])
    polar=torch.tensor([np.hypot(local[0],local[2]),np.arctan2(-local[2],local[0])],dtype=torch.float32)
    geometry=torch.stack([polar[0].clamp_min(1e-6).log(),polar[1].cos(),polar[1].sin()])
    values=dict(polar=polar.numpy(),geometry=geometry.numpy(),
        times=torch.tensor(times[history]-times[now],dtype=torch.float32).numpy(),
        pose=np.array(pose,dtype=np.float32),commands=commands[wi].copy(),
        proprio=commands[wi-1].copy())
    return {k:tensor_digest(values[k]) for k in NONIMAGE}


class ImageTensors:
    """Only referenced candidate PNGs are read; identical PNG/crop is decoded once."""
    def __init__(self,pins):
        self.pins=pins;self.paths={};self.tensors={};self.decoded=0
    def one(self,root,name,inventory,box=None):
        require(type(name) is str and Path(name).name==name and name.endswith(".png"),"unsafe frame")
        p=path_file(root/name);require(name in inventory,"image absent from audited inventory")
        key=str(p)
        if key not in self.paths:
            self.paths[key]=self.pins.read(p,inventory[name])
        else:
            same(self.paths[key],inventory[name],"conflicting image SHA")
        crop=None if box is None else tuple(box)
        ck=(self.paths[key],crop)
        if ck not in self.tensors:
            self.tensors[ck]=tensor_digest(tensor_image(p,box).numpy());self.decoded+=1
        return self.tensors[ck]
    def parts(self,root,frames,meta,inventory,history):
        now=int(history[-1]);wi=np.arange(now-3,now+1)
        return dict(template=self.one(root,frames[0],inventory,meta["initial_bbox_rgb_xyxy"]),
            rgb=[self.one(root,frames[int(i)],inventory) for i in history],
            wm_rgb=[self.one(root,frames[int(i)],inventory) for i in wi],
            future=self.one(root,frames[now+1],inventory))


def load_raw(root,hashes,pins):
    root=path_dir(root)
    docs={n:pins.doc(root/n,hashes[n]) for n in ("metadata.json","observations.json","actions.json")}
    meta,obs,actions=(docs[n] for n in ("metadata.json","observations.json","actions.json"))
    require(isinstance(obs,list) and len(obs)>1,"missing full observed episode")
    times=np.array([o["timestamp_s"] for o in obs])
    require(times.dtype.kind in "fiu" and np.isfinite(times).all()
            and (np.diff(times)>0).all(),"invalid actual timebase")
    commands,bad,_=transition_records(root,obs)
    return dict(root=root,meta=meta,obs=obs,actions=actions,times=times,commands=commands,bad=bad)


def audit(release_path,release_sha,old_cache,old_index_audit_sha,*,progress=True):
    """Return report only. No cache/plan/sampling mutation and no media sweep."""
    torch.set_num_threads(1)
    pins=Pins();release=pins.doc(release_path,release_sha)
    require(release.get("schema")=="failure_state_collection_release_v1"
            and release.get("collection_validated") is True
            and release.get("completed_searches")==126 and release.get("expected")==126
            and release.get("training_released") is False
            and release.get("evaluation_adaptation") is True
            and release.get("untouched_test") is False
            and release.get("score_backfill_allowed") is False,"full nontraining collection release required")
    entries=release.get("teacher_demonstrations")
    require(isinstance(entries,list) and len(entries)==EXPECTED["episodes"],"original winner count")
    require(len({(e["task"],e["key"]) for e in entries})==len(entries),"duplicate new key")
    require(sum(e["candidate_windows"] for e in entries)==EXPECTED["candidate_windows"]
            and sum(e["expected_valid_count"] for e in entries)==EXPECTED["valid_windows"],"fixed new counts")
    b=path_dir(old_cache)
    oldtrust=pins.doc(OLD_TRAINING_AUDIT,OLD_TRAINING_AUDIT_SHA)
    require(oldtrust.get("status")=="TRAINING_ARTIFACT_AUDIT_PASS_OFFLINE_ONLY",
            "old training artifact audit missing")
    trusted=oldtrust["source_hashes"]
    same(trusted.get(str(b/"complete.json")),OLD_COMPLETE_SHA,"old cache trusted complete")
    same(trusted.get(str(b/"index/audit.json")),old_index_audit_sha,"old index trusted audit")
    episode_audit_sha=trusted.get(str(b/"index/train_episodes_audit.json"))
    require(type(episode_audit_sha) is str and len(episode_audit_sha)==64,"old episode audit not trusted")
    complete=pins.doc(b/"complete.json",OLD_COMPLETE_SHA)
    oldaudit=pins.doc(b/"index/audit.json",old_index_audit_sha)
    require(oldaudit.get("contract")==CONTRACT
            and oldaudit.get("cache_complete_sha256")==OLD_COMPLETE_SHA,"old cache contract")
    for n in ("train_episodes.json","train_episode.npy","train_history.npy","train_pose.npy"):
        pins.read(b/n,complete["files"][n])
    episodes=pins.doc(b/"train_episodes.json",complete["files"]["train_episodes.json"])
    image_inventory=pins.doc(b/"source_image_hashes.json",complete["source_image_inventory_sha256"])
    episode_ids=np.load(b/"train_episode.npy",mmap_mode="r",allow_pickle=False)
    histories=np.load(b/"train_history.npy",mmap_mode="r",allow_pickle=False)
    poses=np.load(b/"train_pose.npy",mmap_mode="r",allow_pickle=False)
    pins.read(b/"index/train_valid.npy",oldaudit["splits"]["train"]["index_sha256"])
    valid=np.load(b/"index/train_valid.npy",allow_pickle=False)
    ea=pins.doc(b/"index/train_episodes_audit.json",episode_audit_sha)
    require(ea.get("errors")==[],"old episode audit errors")
    stats={s["episode"]:s for s in ea["stats"]}
    require(len(stats)==len(ea["stats"]),"duplicate old episode stats")
    require(episode_ids.dtype==np.int32 and histories.dtype==np.int32 and poses.dtype==np.float32
            and poses.shape==(len(episode_ids),7,4) and histories.shape==(len(episode_ids),4)
            and valid.dtype==np.int64 and valid.ndim==1 and (valid>=0).all()
            and (valid<len(episode_ids)).all() and (np.diff(valid)>0).all(),"old cache shape/index")
    require(len(valid)==oldaudit["splits"]["train"]["retained"],"old valid count")
    bykey={}
    for eid,e in enumerate(episodes):
        key=e["episode_uid"]
        require(key not in bykey,"ambiguous old task/key")
        bykey[key]=(eid,e)
    grouped=defaultdict(list)
    for row,eid in zip(valid,episode_ids[valid]):grouped[int(eid)].append(int(row))
    images=ImageTensors(pins)
    all_rows=[];episode_reports=[];counter=Counter();source_meta={}
    code_paths=[Path(__file__).resolve(),Path(__import__("wa.data",fromlist=["x"]).__file__).resolve(),
        Path(__import__("wa.wm.robot_data",fromlist=["x"]).__file__).resolve(),
        Path(__import__("wa.wm.failure_state_numeric",fromlist=["x"]).__file__).resolve(),
        Path(__import__("wa.wm.failure_state_labels",fromlist=["x"]).__file__).resolve(),
        Path(__import__("wa.tools.build_failure_state_cache",fromlist=["x"]).__file__).resolve()]
    code={str(p):pins.read(p) for p in code_paths}
    offset=0
    for eid,e in enumerate(entries):
        require(e["task"]=="stt" and e["teacher"] in ("oracle","lightnav"),"unexpected original winner")
        root=path_dir(e["branch"])
        require(root!=Path(e["repeat_branch"]),"repeat cannot be original")
        hashes=e["hashes"]
        for name in ("metadata.json","observations.json","actions.json","windows.json"):
            same(release["source_files"].get(str(root/name)),hashes[name],"new source not release-pinned")
        new=load_raw(root,hashes,pins);m=new["meta"]
        for field,expected in dict(task=e["task"],key=e["key"],teacher=e["teacher"],
            takeover_step=e["takeover_step"],experiment=e["source_experiment"],
            protocol_sha256=e["source_protocol_sha256"]).items():same(m.get(field),expected,"new source identity")
        require(m.get("verification_only") is False and m.get("partition")=="evaluation_adaptation",
                "not an original demonstration")
        wins=pins.doc(root/"windows.json",hashes["windows.json"])
        numeric=audit_numeric_windows(new["obs"],new["actions"],wins,e["takeover_step"],per_window=False)
        same([w["current_index"] for w in wins],e["window_indices"],"candidate order changed")
        same(numeric["valid_window_indices"],e["valid_window_indices"],"effective set changed")
        same(numeric["excluded"],e["numeric_exclusions"],"effective exclusions changed")
        require(numeric["summary"]["valid_windows"]==e["expected_valid_count"],"effective count changed")
        d=numeric["derived"];frames=[o["frame"] for o in new["obs"]]
        inv={frame:hashes[frame] for frame in frames}
        for frame,h in inv.items():same(release["source_files"].get(str(root/frame)),h,"new image pin absent")
        source_meta[str(root)]={k:pins.files[str(root/k)] for k in
                              ("metadata.json","observations.json","actions.json","windows.json")}
        olditem=bykey.get(e["task"]+":"+e["key"]);old=None;nonimages=defaultdict(list);old_parts={}
        if olditem is not None:
            old_eid,oe=olditem;oroot=path_dir(oe["root"])
            old=load_raw(oroot,stats[old_eid]["source_sha256"],pins)
            require([o["frame"] for o in old["obs"]]==oe["frames"],"old observation frames changed")
            for row in grouped[old_eid]:
                h=np.asarray(histories[row]);now=int(h[-1]);end=int(np.searchsorted(old["times"],old["times"][now]+.7))
                require(now>=4 and end<len(old["times"]) and not old["bad"][int(h[0]):end].any(),
                        "old runtime eligibility mismatch")
                parts=scalar_parts(old["obs"],old["times"],old["commands"],h,np.asarray(poses[row]))
                old_parts[row]=parts;nonimages[digest_parts(parts)].append(row)
            counter["same_key_episodes"]+=1
            source_meta[str(oroot)]={k:pins.files[str(oroot/k)] for k in
                                    ("metadata.json","observations.json","actions.json")}
        epcount=Counter();samecurrent_first=None
        old_image_parts={}
        for j in np.flatnonzero(numeric["effective_mask"]):
            now=int(d["current_indices"][j]);h=d["history"][j]
            parts=scalar_parts(new["obs"],new["times"],new["commands"],h,d["pose"][j])
            digest=digest_parts(parts);candidates=nonimages.get(digest,[]);matches=[];image_mismatch={}
            ni=None
            if candidates:
                ni=images.parts(root,frames,m,inv,h)
                for row in candidates:
                    if row not in old_image_parts:
                        old_image_parts[row]=images.parts(old["root"],oe["frames"],old["meta"],
                            image_inventory[str(old["root"])],histories[row])
                    oi=old_image_parts[row];diff=[k for k in IMAGES if ni[k]!=oi[k]]
                    if not diff:matches.append(row)
                    else:image_mismatch[str(row)]=diff
            status="EXACT_DUPLICATE" if matches else ("NO_SAME_KEY_OLD_EPISODE" if old is None else
                   "NO_NONIMAGE_MATCH" if not candidates else "CONSUMED_IMAGE_DIFFERENCE")
            epcount[status]+=1;counter[status]+=1
            counter["nonimage_candidate_pairs"]+=len(candidates)
            if old is not None and samecurrent_first is None:
                samecurrent=[r for r in grouped[old_eid] if int(histories[r,-1])==now]
                if samecurrent:
                    r=samecurrent[0];diff=[k for k in NONIMAGE if parts[k]!=old_parts[r][k]]
                    if diff:samecurrent_first=dict(new_current=now,old_row=r,differing_fields=diff,
                        old_timestamp_s=float(old["times"][now]),new_timestamp_s=float(new["times"][now]))
            all_rows.append(dict(new_cache_row=offset+int(j),new_episode=eid,task=e["task"],key=e["key"],
                new_current_index=now,new_actual_timestamp_s=float(new["times"][now]),
                source_branch=str(root),source_teacher=e["teacher"],takeover_step=e["takeover_step"],
                status=status,nonimage_sha256=digest,nonimage_components=parts,
                old_nonimage_candidate_rows=candidates,old_matching_rows=matches,
                old_matches=[dict(row=r,episode=old_eid,current_index=int(histories[r,-1]),
                     actual_timestamp_s=float(old["times"][int(histories[r,-1])]),
                     source_branch=str(old["root"]),teacher=oe["teacher"]) for r in matches],
                consumed_image_components=ni,old_candidate_image_components={
                    str(r):old_image_parts[r] for r in candidates},
                image_mismatch_fields=image_mismatch))
        ep=dict(task=e["task"],key=e["key"],new_episode=eid,branch=e["branch"],teacher=e["teacher"],
            takeover_step=e["takeover_step"],candidate_row_start=offset,
            candidate_windows=len(wins),valid_windows=int(numeric["effective_mask"].sum()),
            old_same_key_present=old is not None,old_same_key_valid_rows=0 if old is None else len(grouped[old_eid]),
            counts=dict(epcount),first_samecurrent_nonimage_difference=samecurrent_first)
        episode_reports.append(ep);offset+=len(wins)
        if progress:print(json.dumps(dict(phase="SCOPED_DEDUP",episode=eid+1,key=e["key"],
                            valid=ep["valid_windows"],counts=ep["counts"])),flush=True)
    require(offset==EXPECTED["candidate_windows"] and len(all_rows)==EXPECTED["valid_windows"],
            "global new-row coverage")
    require(len({r["new_cache_row"] for r in all_rows})==len(all_rows),"duplicate new cache row")
    pins.finish()
    for p,h in code.items():require(file_sha(p)==h,"dedup source changed")
    duplicate=[r["new_cache_row"] for r in all_rows if r["status"]=="EXACT_DUPLICATE"]
    unique=[r["new_cache_row"] for r in all_rows if r["status"]!="EXACT_DUPLICATE"]
    return dict(schema=SCHEMA,status="PASS_SCOPED_EXACT_EQUIVALENCE",
        release=dict(path=str(path_file(release_path)),sha256=release_sha),
        old_cache=dict(path=str(b),complete_sha256=OLD_COMPLETE_SHA,index_audit_sha256=old_index_audit_sha),
        old_trust_anchor=dict(path=str(OLD_TRAINING_AUDIT),sha256=OLD_TRAINING_AUDIT_SHA,
                              train_episodes_audit_sha256=episode_audit_sha),
        source_files=pins.files,code_sha256=code,
        runtime=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,pillow=pillow_version),
        summary=dict(valid_windows=len(all_rows),
            candidate_windows=offset,original_episodes=len(entries),exact_duplicate_count=len(duplicate),
            deduplicated_new_count_within_same_key_scope=len(unique),counts=dict(counter),
            consumed_png_files_verified=len(images.paths),image_tensors_decoded=images.decoded),
        duplicate_new_cache_rows=duplicate,nonduplicate_new_cache_rows_within_scope=unique,
        episodes=episode_reports,windows=all_rows,sampling_changed=False,cache_modified=False,
        training_released=False,
        scope="All effective new windows versus all old valid windows with the SAME task/key",
        limitations=["Different task/key windows not compared; not global cross-scene pixel-collision dedup",
            "Remaining count is relative to the old cache; within-new-sample duplicates are not separately collapsed",
            "No teacher, directory, absolute step or absolute timestamp exclusion",
            "BBox metadata is evidence, not exclusion; actual cropped template tensor decides",
            "Source files include only consumed small/raw evidence and candidate media, not a repeated full raw audit",
            "Mapping is to release-ordered candidate cache rows and must match the eventual physical new cache",
            "This is an exposure diagnostic, not a sampling plan, cache release, or student SR"])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("release","release-sha","old-cache","old-index-audit-sha","output"):
        p.add_argument("--"+n,required=True)
    a=p.parse_args();out=Path(a.output)
    require(out.is_absolute() and not out.exists() and not out.is_symlink(),"fresh report path required")
    path_dir(out.parent)
    report=audit(a.release,a.release_sha,a.old_cache,a.old_index_audit_sha)
    report["reproduce_command"]=shlex.join([sys.executable,"-B","-m","wa.tools.audit_failure_state_dedup",
        "--release",a.release,"--release-sha",a.release_sha,"--old-cache",a.old_cache,
        "--old-index-audit-sha",a.old_index_audit_sha,"--output",a.output])
    with out.open("x",encoding="utf-8") as f:f.write(json.dumps(report,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps(dict(path=str(out),sha256=file_sha(out),summary=report["summary"])),flush=True)


if __name__=="__main__":main()
