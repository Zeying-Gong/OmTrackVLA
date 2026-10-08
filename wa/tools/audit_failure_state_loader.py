"""Independent CPU audit of an explicitly pinned, real failure-state cache.

This does not release training or measure student success. No model weights,
GPU, simulator, source mutation, or output overwrite is permitted.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image
import torch
from wa.wm.failure_state_data import FailureStateData, EXPECTED
from wa.wm.training import JointRobotModel

SHAPES = dict(rgb=(4,3,224,224), template=(3,224,224), polar=(2,),
              times=(4,), pose=(7,4), geometry=(3,), wm_rgb=(4,3,224,224),
              commands=(4,4), proprio=(4,4), future=(3,224,224))
STATUS = "CPU_LOADER_AUDIT_PASS_NOT_TRAINING_OR_MODEL_SR"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def raw_episode(entry):
    root = Path(entry["root"])
    obs = json.loads((root/"observations.json").read_text())
    actions = json.loads((root/"actions.json").read_text())
    meta = json.loads((root/"metadata.json").read_text())
    times = np.asarray([o["timestamp_s"] for o in obs], dtype=np.float64)
    return root, obs, actions, meta, times


def indices(times, now):
    """Independent causal search, continuous JEPA and predecessor commands."""
    history = np.asarray([max(0, min(now, int(np.searchsorted(
        times, times[now]+offset, side="right"))-1))
        for offset in (-1.5, -1., -.5, 0.)], dtype=np.int32)
    return history, np.arange(now-3,now+1), np.arange(now-4,now)


def select_windows(data):
    """All valid rows inspected; all prefix crossings plus fixed ordinal samples.

    "early" is the second valid window in chronological order (not a guessed
    elapsed-time threshold). If fewer than three windows exist, roles overlap
    and are explicitly recorded, never fabricated.
    """
    groups = {}
    for position, row in enumerate(data.rows):
        ep = int(data.episode[int(row)])
        groups.setdefault(ep, []).append((position,int(row)))
    selected, episodes = [], []
    for ep in range(len(data.episodes)):
        entry = data.episodes[ep]
        pairs = groups.get(ep, [])
        if not pairs:
            episodes.append(dict(episode=ep, episode_uid=entry["episode_uid"],
                                 valid_windows=0, selected_positions=[]))
            continue
        root, obs, actions, meta, times = raw_episode(entry)
        k = entry["takeover_step"]
        first, early, tail = pairs[0][0], pairs[min(1,len(pairs)-1)][0], pairs[-1][0]
        take = []
        previous = -1
        for position, row in pairs:
            now = int(data.history[row,-1])
            need(now>previous and now>=k, "valid rows must be chronological and teacher-owned")
            previous = now
            history, wm, proprio = indices(times,now)
            need(np.array_equal(history,data.history[row]), "manual sparse history differs")
            crossing = [name for name, seq in (("policy",history),("jepa_command",wm),
                                               ("proprio",proprio)) if (seq<k).any()]
            reasons = [name for name,p in (("first",first),("early_second",early),("tail",tail))
                       if position==p]
            if crossing:
                reasons.append("student_prefix")
            if reasons:
                record = dict(position=position, cache_row=row, episode=ep, now=now,
                              history=history.tolist(), jepa_indices=wm.tolist(),
                              proprio_indices=proprio.tolist(), reasons=reasons,
                              prefix_crossing=crossing)
                selected.append(record)
                take.append(position)
        episodes.append(dict(episode=ep,episode_uid=entry["episode_uid"],root=str(root),
            teacher=entry["teacher"],takeover_step=k,valid_windows=len(pairs),
            first_position=first,early_position=early,tail_position=tail,
            early_definition="second chronological valid window",
            early_elapsed_after_takeover_s=float(times[int(data.history[pairs[min(1,len(pairs)-1)][1],-1])]-times[k]),
            representative_unique_count=len({first,early,tail}),selected_positions=take))
    need(len(groups)==EXPECTED["episodes_with_valid_windows"],"nonzero episode count")
    need(len(data.rows)==EXPECTED["valid_windows"],"valid window count")
    return selected,episodes


def image_tensor(path,box=None):
    # Independent implementation; deliberately retain original PIL default
    # RGB resize resampling, not an unverified alternative interpolation.
    with Image.open(path) as image:
        image=image.convert("RGB")
        if box is not None:
            image=image.crop(tuple(box))
        array=np.asarray(image.resize((224,224)),dtype=np.float32).copy()/np.float32(255.)
    value=torch.from_numpy(array.transpose(2,0,1))
    return (value-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]


class ManualEpisode:
    """Raw-only arithmetic, independent of derive_labels/transition_records."""
    def __init__(self,entry):
        self.entry=entry
        self.root,self.obs,self.actions,self.meta,self.times=raw_episode(entry)
        self.positions=np.asarray([o["robot_position_world"] for o in self.obs],dtype=np.float64)
        self.rotations=np.asarray([o["robot_rotation_world_from_body"] for o in self.obs],dtype=np.float64)
        self.yaws=np.unwrap(np.arctan2(self.rotations[:,0,2],self.rotations[:,0,0]))
        self.by_step={a["sim_step"]:a for a in self.actions}
        need(len(self.by_step)==len(self.actions),"duplicate raw action")
        self.template=image_tensor(self.root/entry["frames"][0],self.meta["initial_bbox_rgb_xyxy"])
        self.picture=lru_cache(maxsize=40)(self._picture)

    def _picture(self,index):
        return image_tensor(self.root/self.entry["frames"][int(index)])

    def command(self,index):
        action=self.by_step[index]
        need(action["sim_step"]==index,"raw command step")
        command=np.asarray(action["normalized_action"],dtype=np.float32)
        need(command.shape==(3,) and np.isfinite(command).all() and (abs(command)<=1).all(),
             "raw normalized command")
        dt=np.float32((self.times[index+1]-self.times[index])/.1)
        return np.r_[command,dt].astype(np.float32)

    def expected(self,now):
        history,wm,previous=indices(self.times,now)
        origin=self.positions[now];rotation=self.rotations[now]
        pose=[]
        for offset in (.1,.2,.3,.4,.5,.6,.7):
            query=self.times[now]+offset
            need(query<=self.times[-1]+1e-8,"future extrapolation")
            future=np.asarray([np.interp(query,self.times,self.positions[:,a]) for a in range(3)])
            local=(future-origin)@rotation
            yaw=np.interp(query,self.times,self.yaws)-self.yaws[now]
            pose.append([local[0],-local[2],np.sin(yaw),np.cos(yaw)])
        endpoint=min(len(self.obs)-1,int(np.searchsorted(self.times,self.times[now]+.7,side="left")))
        need(all(self.by_step[i]["owner"]=="teacher" and
                 self.by_step[i]["teacher"]==self.entry["teacher"] for i in range(now,endpoint)),
             "future label transition is not owned by selected teacher")
        # Only this current observation is used for UWB.
        delta=np.asarray(self.obs[now]["target_position_world_label_only"],dtype=np.float64)-origin
        local=delta@rotation
        polar=torch.tensor([np.hypot(local[0],local[2]),np.arctan2(-local[2],local[0])],
                           dtype=torch.float32)
        geometry=torch.stack([polar[0].clamp_min(1e-6).log(),polar[1].cos(),polar[1].sin()])
        return dict(rgb=torch.stack([self.picture(int(i)) for i in history]),
            template=self.template,polar=polar,
            times=torch.tensor(self.times[history]-self.times[now],dtype=torch.float32),
            pose=torch.tensor(np.asarray(pose,dtype=np.float32)),geometry=geometry,
            wm_rgb=torch.stack([self.picture(int(i)) for i in wm]),
            commands=torch.from_numpy(np.stack([self.command(int(i)) for i in wm])),
            proprio=torch.from_numpy(np.stack([self.command(int(i)) for i in previous])),
            future=self.picture(now+1))


def compare_item(actual,expected,cache_pose):
    need(set(actual)==set(SHAPES),"exact ten input tensors")
    errors={}
    for key,shape in SHAPES.items():
        value=actual[key]
        need(isinstance(value,torch.Tensor) and value.device.type=="cpu" and value.dtype==torch.float32,
             key+": CPU float32 tensor required")
        need(tuple(value.shape)==shape and torch.isfinite(value).all().item(),key+": shape/finite")
        target=expected[key]
        error=float((value-target).abs().max())
        # Cached labels must be bit-exact. Independent matrix/interpolation
        # arithmetic may differ by float32 rounding: bounded separately.
        if key=="pose":
            need(np.array_equal(value.numpy(),np.asarray(cache_pose)),"pose not bit-exact cached label")
            need(error<=2e-6,"pose differs from independent raw SE2 calculation")
        else:
            need(torch.equal(value,target),key+": raw-derived tensor differs")
        errors[key]=error
    return errors


class CPUEncoderStandIn:
    """No encoder checkpoint is loaded; actual make_conditions routing only."""
    @staticmethod
    def encode(images):
        return images.mean(dim=(-2,-1)).unsqueeze(-2)


def check_conditions(item):
    batch={k:v.unsqueeze(0) for k,v in item.items()}
    modes=torch.tensor([2],dtype=torch.long)
    conditions=JointRobotModel.make_conditions(CPUEncoderStandIn(),batch,modes)
    expected={"visual","template","template_valid","polar","uwb_valid","age_s","times"}
    need(set(conditions)==expected,"condition key routing")
    need(conditions["template_valid"].tolist()==[True] and conditions["uwb_valid"].tolist()==[True],
         "mixed validity")
    need(torch.equal(conditions["polar"],batch["polar"]) and
         torch.equal(conditions["times"],batch["times"]) and conditions["age_s"].tolist()==[0.],
         "current polar/time routing")
    changed=dict(batch)
    for name in ("pose","geometry","wm_rgb","commands","proprio","future"):
        changed[name]=torch.full_like(batch[name],123.25)
    other=JointRobotModel.make_conditions(CPUEncoderStandIn(),changed,modes)
    need(all(torch.equal(value,other[name]) for name,value in conditions.items()),
         "supervised future/JEPA-only tensors leaked into policy conditions")
    return dict(status="ACTUAL_MAKE_CONDITIONS_ROUTING_PASS_CPU_ENCODER_STANDIN",
                loaded_encoder_weights=False,loaded_policy_weights=False,mode="mixed",
                perturbed_nonpolicy_fields=["pose","geometry","wm_rgb","commands","proprio","future"])


def distribution(values):
    a=np.asarray(values,dtype=float)
    return dict(count=len(values),minimum=float(a.min()),median=float(np.median(a)),
                p95=float(np.percentile(a,95)),maximum=float(a.max()),total=float(a.sum()))


def stat_gate_probe(data):
    """Nine read-only measurements of the existing, unchanged access gate.

    Select min/middle/max episode path counts, three calls each. This is a
    warmed-filesystem micro-measurement, not a substitute for actual getitem.
    """
    nonzero=sorted({int(data.episode[int(row)]) for row in data.rows},
                   key=lambda ep:(len(data._episode_paths[ep]),ep))
    chosen=[nonzero[0],nonzero[len(nonzero)//2],nonzero[-1]]
    probes=[]
    for ep in chosen:
        timings=[]
        for unused in range(3):
            started=time.perf_counter()
            for link,target in data._links.items():
                need(Path(link).is_symlink() and Path(link).resolve()==Path(target),
                     "heldout link changed during stat probe")
            data._reader.unchanged(data._cache_guard)
            data._reader.unchanged(data._episode_paths[ep])
            timings.append(time.perf_counter()-started)
        probes.append(dict(episode=ep,episode_uid=data.episodes[ep]["episode_uid"],
            png_paths=len(data.episodes[ep]["frames"]),
            episode_paths=len(data._episode_paths[ep]),
            fixed_paths=len(data._cache_guard),heldout_links=len(data._links),
            seconds=timings,median_seconds=float(np.median(timings))))
    return dict(scope="unchanged links+fixed-source+all-current-episode-path gate only",
                warm_filesystem=True,actual_getitem_not_replaced=True,
                samples=probes,total_measurements=9)

def audit(cache,admission_sha,*,progress=None):
    started=utc();wall=time.perf_counter()
    t=time.perf_counter()
    data=FailureStateData(cache,expected_admission_sha256=admission_sha)
    initialization=time.perf_counter()-t
    t=time.perf_counter();selected,episodes=select_windows(data);selection_seconds=time.perf_counter()-t
    rows=[];get_times=[];manual_times=[];condition_reports=[]
    current=-1;manual=None
    for number,record in enumerate(selected):
        if current!=record["episode"]:
            current=record["episode"];manual=ManualEpisode(data.episodes[current])
            conditions_done=False
        t=time.perf_counter();item=data[record["position"]];get_s=time.perf_counter()-t
        t=time.perf_counter();expected=manual.expected(record["now"])
        errors=compare_item(item,expected,data.pose_labels[record["cache_row"]])
        if not conditions_done:
            condition_reports.append(dict(episode=current,**check_conditions(item)))
            conditions_done=True
        manual_s=time.perf_counter()-t
        get_times.append(get_s);manual_times.append(manual_s)
        rows.append({**record,"max_absolute_errors":errors,
            "pose_tensor_sha256":tensor_sha(item["pose"]),
            "template_tensor_sha256":tensor_sha(item["template"]),
            "current_polar":item["polar"].tolist(),
            "getitem_seconds":get_s,"independent_verification_seconds":manual_s})
        if progress and (number+1)%50==0:
            progress(dict(stage="CHECKING_NOT_PASS",checked=number+1,selected=len(selected)))
    gate_probe=stat_gate_probe(data)
    # Final source state check includes all consumed inputs, even zero-valid eps.
    data._reader.unchanged()
    for link,target in data._links.items():
        need(Path(link).is_symlink() and Path(link).resolve()==Path(target),"heldout link changed")
    admission=json.loads((Path(cache)/"admission.json").read_text())
    need(admission["training_released"] is False,"audit must not release training on disk")
    sources=[Path(__file__).resolve(),Path(__import__("wa.wm.failure_state_data",fromlist=["x"]).__file__).resolve(),
             Path(__import__("wa.wm.robot_data",fromlist=["x"]).__file__).resolve(),
             Path(__import__("wa.data",fromlist=["x"]).__file__).resolve(),
             Path(__import__("wa.wm.training",fromlist=["x"]).__file__).resolve()]
    return dict(schema="failure_state_real_loader_audit_v1",status=STATUS,
        started_utc=started,finished_utc=utc(),cache=str(Path(cache).resolve()),
        admission_sha256=admission_sha,loader_admission=data.admission_report,
        tool_sources_sha256={str(p):sha(p) for p in sources},
        runtime=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
                     torch_cpu_threads=torch.get_num_threads(),cuda_used=False),
        selection=dict(rule="all prefix-crossing valid rows plus first/second/tail per nonzero episode",
            all_valid_rows_inspected=len(data),nonzero_episodes=sum(e["valid_windows"]>0 for e in episodes),
            zero_valid_episodes=sum(e["valid_windows"]==0 for e in episodes),
            selected_unique_windows=len(rows),all_prefix_crossing_windows=sum(bool(r["prefix_crossing"]) for r in rows),
            prefix_crossing_by_input=dict(Counter(name for r in rows for name in r["prefix_crossing"]))),
        shapes={k:list(v) for k,v in SHAPES.items()},dtype="torch.float32",
        pose_cache_comparison="bit-exact",raw_pose_max_absolute_tolerance=2e-6,
        episodes=episodes,windows=rows,condition_routing=condition_reports,
        timing=dict(initialization_seconds=initialization,selection_seconds=selection_seconds,
            getitem_seconds=distribution(get_times),independent_verification_seconds=distribution(manual_times),
            measured_getitems_per_second=len(rows)/sum(get_times),
            wall_seconds=time.perf_counter()-wall),
        stat_gate_probe=gate_probe,
        per_getitem_stat_scope=dict(fixed_source_paths=len(data._cache_guard),
            episode_paths=distribution([len(data._episode_paths[e["episode"]]) for e in episodes if e["valid_windows"]]),
            every_episode_png_statted=True,cryptographic_rehash_per_getitem=False),
        boundaries=["CPU loader admission and bounded actual getitem verification, not training or SR",
            "All consumed source files hashed at initialization; unconsumed historical weights not rescanned",
            "All valid rows considered for selection; nonselected getitems not executed",
            "Per-getitem inode/size/mtime checks include every PNG of the current episode",
            "Same-inode same-size adversarial rewrite with restored timestamp is not prevented by stat checks",
            "Actual make_conditions called with deterministic CPU encoder stand-in; no model inference",
            "Independent RGB checks use original PIL default RGB resizing, no image augmentation",
            "No changes to loss, architecture, controller, cached training flags, source data or heldout",
            "Evaluation-set adaptation cache, not untouched-test generalization",
            "Timing includes filesystem/cache state; not GPU throughput, rollout or device latency"],
        training_released=False,no_new_model_success_rate=True)


def run_to_file(cache,admission_sha,output,*,progress=None):
    output=Path(output)
    need(output.is_absolute() and output.parent.is_dir() and output.parent.resolve()==output.parent,
         "output requires an existing absolute nonsymlink parent")
    # Reserve before expensive admission; x/O_NOFOLLOW rejects existing files
    # and dangling symlinks. Failure is retained and cannot masquerade as PASS.
    fd=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0),0o644)
    with os.fdopen(fd,"w") as stream:
        try:
            result=audit(cache,admission_sha,progress=progress)
        except Exception as exc:
            failure=dict(schema="failure_state_real_loader_audit_v1",status="FAILED_NOT_RELEASED",
                         finished_utc=utc(),error_type=type(exc).__name__,error=str(exc),
                         cache=str(cache),admission_sha256=admission_sha,training_released=False)
            json.dump(failure,stream,indent=2,allow_nan=False);stream.write("\n")
            stream.flush();os.fsync(stream.fileno())
            raise
        json.dump(result,stream,indent=2,allow_nan=False);stream.write("\n")
        stream.flush();os.fsync(stream.fileno())
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",required=True)
    parser.add_argument("--admission-sha",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    result=run_to_file(args.cache,args.admission_sha,args.output,
                       progress=lambda value:print(json.dumps(value),flush=True))
    print(json.dumps(dict(status=result["status"],output=args.output,
                          sha256=sha(args.output),selection=result["selection"])),flush=True)


if __name__=="__main__":
    main()
