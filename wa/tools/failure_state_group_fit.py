"""Frozen 88+32+32 teacher-state label fitting; never training, rollout or SR.

select is CPU-only. run is developer-only, uses a previously pinned selection
and either both fixed baselines or one externally audited candidate. Repeated
history is synthetic policy-history replacement, not a different label/JEPA.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch

from wa.tools import replay_teacher_group_fit as replay
from wa.tools import audit_teacher_group_fit as original
from wa.tools.audit_failure_state_dedup import scalar_parts, digest_parts
from wa.wm.failure_state_data import FailureStateData, EXPECTED
from wa.wm.dual_teacher_data import DualTeacherData
from wa.wm.training import JointRobotModel
from wa.wm.robot_data import CONTRACT

DEDUP_SHA = "599d60ea2d40ff4f9ae69f87df03e3a5a192b749ce030504bf19177999262859"
BEST_SHA = "c510c04d987d141423401a25323ea353314107a16f4dac5d2901370ab199fa52"
BEST_STEP = 59716
BEST_AUDIT_SHA = "59d5b468eec4283db1a1d87ad7b9a9d1bc2123739cffc38cebfc9a1e14dcc727"
PARENT_SHA = original.MODELS["parent59866"][1]
SELECTION_SCHEMA = "failure_state_group_fit_selection_v1"
FIT_SCHEMA = "failure_state_group_fit_v1"
FIT_STATUS = "FIXED152_TEACHER_STATE_LABEL_FIT_ONLY_NOT_SR"
LOADER_STATUS = "CPU_LOADER_AUDIT_PASS_NOT_TRAINING_OR_MODEL_SR"
GROUPS = dict(original.LIMITS, recovery_early=32, recovery_late=32)
INPUTS = replay.INPUT_KEYS
SHAPES = dict(rgb=(4,3,224,224), template=(3,224,224), polar=(2,), times=(4,), pose=(7,4))
STATUSES = {"EXACT_DUPLICATE", "NO_NONIMAGE_MATCH", "NO_SAME_KEY_OLD_EPISODE",
            "CONSUMED_IMAGE_DIFFERENCE"}
need = replay.require
sha = replay.sha


def integer(x):
    return type(x) is int and x >= 0


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def document(path, expected, pins):
    p = Path(path)
    need(p.is_absolute() and not p.is_symlink() and p.resolve() == p, "unalias absolute file")
    replay.verify(p, expected, pins)
    return json.loads(p.read_text())


def source_modules():
    return [Path(__file__).resolve()] + [Path(m.__file__).resolve() for m in
        (replay, original, __import__("wa.tools.audit_failure_state_dedup", fromlist=["x"]),
         __import__("wa.wm.failure_state_data", fromlist=["x"]),
         __import__("wa.wm.robot_data", fromlist=["x"]),
         __import__("wa.wm.dual_teacher_data", fromlist=["x"]),
         __import__("wa.wm.training", fromlist=["x"]), __import__("wa.data", fromlist=["x"]))]


def validate_loader_gate(audit, cache, admission_sha):
    need(audit.get("schema") == "failure_state_real_loader_audit_v1"
         and audit.get("status") == LOADER_STATUS, "independent loader audit required")
    need(audit.get("cache") == str(Path(cache).resolve())
         and audit.get("admission_sha256") == admission_sha, "loader audit cache/admission binding")
    gate = audit.get("loader_admission", {})
    need(gate.get("status") == "LOADER_ADMISSION_PASS_NOT_NEW_MODEL_RESULT"
         and gate.get("summary") == EXPECTED
         and gate.get("admission_sha256") == admission_sha
         and gate.get("actual_consumed_sources_hashed") is True
         and gate.get("training_released_on_disk") is False, "loader admission contract")
    s = audit.get("selection", {})
    need(type(s.get("all_valid_rows_inspected")) is int and s["all_valid_rows_inspected"] == 6864
         and type(s.get("nonzero_episodes")) is int and s["nonzero_episodes"] == 90,
         "loader audit incomplete")


def validate_dedup(report, current, release_ref, old_cache):
    """All 6864 physical rows are bound, not just the 64 selected windows."""
    need(report.get("schema") == "failure_state_exact_dedup_v1"
         and report.get("status") == "PASS_SCOPED_EXACT_EQUIVALENCE", "dedup not admitted")
    need(report.get("release") == release_ref and
         report.get("old_cache", {}).get("path") == str(old_cache) and
         report["old_cache"].get("complete_sha256") == original.CACHE_SHA, "dedup source binding")
    for k in ("training_released", "cache_modified", "sampling_changed"):
        need(report.get(k) is False, "dedup must not change/release training")
    expected = dict(original_episodes=96, candidate_windows=7396, valid_windows=6864,
                    exact_duplicate_count=526, deduplicated_new_count_within_same_key_scope=6338)
    need(all(type(report.get("summary", {}).get(k)) is int and report["summary"][k] == v
             for k, v in expected.items()), "dedup fixed counts")
    windows = report.get("windows")
    need(isinstance(windows, list) and len(windows) == len(current) == 6864, "all valid dedup rows")
    need(len({r["new_cache_row"] for r in current}) == 6864, "duplicate physical row")
    table = {r["new_cache_row"]: r for r in current}
    seen, duplicate, nonduplicate = set(), [], []
    for row in windows:
        raw = row.get("new_cache_row")
        need(integer(raw) and raw in table and raw not in seen, "dedup missing/duplicate/foreign row")
        seen.add(raw)
        actual = table[raw]
        for k in ("new_episode", "new_current_index", "takeover_step"):
            need(integer(row.get(k)), "dedup index type: " + k)
        for k in ("new_cache_row", "new_episode", "new_current_index", "new_actual_timestamp_s",
                  "task", "key", "source_branch", "source_teacher", "takeover_step",
                  "nonimage_components", "nonimage_sha256"):
            need(row.get(k) == actual[k], "dedup physical mismatch: " + k)
        need(row.get("status") in STATUSES, "unknown dedup status")
        matches = row.get("old_matching_rows")
        need(isinstance(matches, list) and all(integer(v) for v in matches)
             and len(matches) == len(set(matches)), "old matching indices")
        need(bool(matches) == (row["status"] == "EXACT_DUPLICATE"), "dedup status/matches contradiction")
        (duplicate if matches else nonduplicate).append(raw)
    need(seen == set(table), "incomplete physical coverage")
    need(duplicate == report.get("duplicate_new_cache_rows") and len(duplicate) == 526
         and nonduplicate == report.get("nonduplicate_new_cache_rows_within_scope")
         and len(nonduplicate) == 6338, "dedup explicit sets differ")
    return [r for r in current if r["new_cache_row"] not in set(duplicate)]


def physical_rows(data):
    result = []
    for ep, entry in enumerate(data.episodes):
        _, _, obs, times, _ = data.episode_info(ep)
        commands, _ = data.transitions(ep)
        k = entry["takeover_step"]
        for position in np.flatnonzero(np.asarray(data.episode)[data.rows] == ep):
            row = int(data.rows[position]); now = int(data.history[row,-1])
            parts = scalar_parts(obs, times, commands, data.history[row], data.pose_labels[row])
            result.append(dict(position=int(position), new_cache_row=row, new_episode=ep,
                new_current_index=now, new_actual_timestamp_s=float(times[now]),
                elapsed_after_takeover_s=float(times[now] - times[k]), task=entry["task"],
                key=entry["episode_uid"].split(":",1)[1], source_branch=entry["root"],
                source_teacher=entry["teacher"], takeover_step=k,
                nonimage_components=parts, nonimage_sha256=digest_parts(parts)))
    return sorted(result, key=lambda r:r["new_cache_row"])


def recovery_selection(rows):
    """Fixed hash order by task/key, one per episode per stratum; no fit scores."""
    groups = []
    seen = set()
    for r in rows:
        need(integer(r.get("new_cache_row")) and integer(r.get("new_episode")) and
             r["new_cache_row"] not in seen, "ambiguous eligible row")
        seen.add(r["new_cache_row"])
        need(replay.finite_number(r.get("elapsed_after_takeover_s")) and
             r["elapsed_after_takeover_s"] >= 0, "invalid elapsed actual time")
    for group, bound, target in (("recovery_early", True, .5), ("recovery_late", False, 2.)):
        pools = {}
        for r in rows:
            age = r["elapsed_after_takeover_s"]
            if (age <= 1.) == bound:
                pools.setdefault((r["task"],r["key"]), []).append(r)
        need(len(pools) >= 32, "insufficient " + group + " eligible episodes")
        order = sorted(pools, key=lambda k:(hashlib.sha256(
            ("wa-failure-fit-v1:" + k[0] + ":" + k[1]).encode()).hexdigest(), k))
        for identity in order[:32]:
            chosen = min(pools[identity], key=lambda r:(abs(r["elapsed_after_takeover_s"]-target),
                                                       r["new_current_index"],r["new_cache_row"]))
            groups.append(dict(source="recovery", group=group, index=chosen["position"],
                raw_row=chosen["new_cache_row"], episode=chosen["new_episode"],
                episode_uid=chosen["task"]+":"+chosen["key"], teacher=chosen["source_teacher"],
                branch=chosen["source_branch"], current_index=chosen["new_current_index"],
                takeover_step=chosen["takeover_step"], elapsed_after_takeover_s=chosen["elapsed_after_takeover_s"],
                episode_order_sha256=hashlib.sha256(
                    ("wa-failure-fit-v1:"+":".join(identity)).encode()).hexdigest()))
    need(len({s["raw_row"] for s in groups}) == 64, "strata reused same raw window")
    for group in ("recovery_early","recovery_late"):
        need(len({s["episode_uid"] for s in groups if s["group"]==group})==32,
             "more than one window per episode/stratum")
    return groups


def sample(item, pose):
    got = {k:item[k] for k in INPUTS + ("pose",)}
    for k, shape in SHAPES.items():
        need(isinstance(got[k],torch.Tensor) and got[k].dtype==torch.float32
             and tuple(got[k].shape)==shape and torch.isfinite(got[k]).all(), "sample "+k)
    need(np.array_equal(got["pose"].numpy(),np.asarray(pose)), "label cache differs")
    return got


def fingerprint(item):
    return {k:replay.tensor_digest(item[k]) for k in INPUTS+("pose",)}


def add_window_sources(data, s, pins):
    entry=data.episodes[s["episode"]];root=Path(entry["root"]);now=int(data.history[s["raw_row"],-1])
    frames={0,now+1,*map(int,data.history[s["raw_row"]]),*range(now-3,now+1)}
    for name in ["metadata.json","observations.json","actions.json"] + [entry["frames"][i] for i in sorted(frames)]:
        path=root/name
        # New loader and old fixed reference have already admitted these exact bytes.
        h=sha(path);need(str(path) not in pins or pins[str(path)]==h,"conflicting sample pin")
        pins[str(path)]=h


def context(args):
    """Explicitly admitted real data only; never constructs/loads a model."""
    pins={str(p):sha(p) for p in source_modules()}
    ref, baseline = replay.read_reference(args.reference_fit,pins)
    replay.verify_reference_sources(ref,args.job_root,pins)
    old_cache=Path(args.root)/"artifacts/dual_teacher_se2_cache_20261006_v1"
    old=DualTeacherData(old_cache)
    loader_audit=document(args.loader_audit,args.loader_audit_sha256,pins)
    validate_loader_gate(loader_audit,args.cache,args.admission_sha256)
    for path,h in loader_audit["tool_sources_sha256"].items():
        replay.verify(path,h,pins)
    new=FailureStateData(args.cache,expected_admission_sha256=args.admission_sha256)
    admission=document(Path(args.cache)/"admission.json",args.admission_sha256,pins)
    need(new.admission_report["release_sha256"]==admission["collection_release"]["sha256"],
         "actual loader release mismatch")
    dedup=document(args.dedup_report,DEDUP_SHA,pins)
    current=physical_rows(new)
    allowed=validate_dedup(dedup,current,admission["collection_release"],old_cache)
    # Dedup is already independently SHA-pinned; preserve its historical code pins
    # without rerunning the old audit or repeatedly hashing unused historical weights.
    selected=[dict(s,source="old88") for s in ref["selected"]] + recovery_selection(allowed)
    samples=[]
    for i,s in enumerate(selected):
        if i<88:
            item=replay.fixed_sample(old,ref["selected"],baseline,i,pins);data=old
        else:
            item=new[s["index"]];data=new
        item=sample(item,data.pose_labels[s["raw_row"]])
        add_window_sources(data,s,pins);samples.append(item)
    for name,h in admission["files"].items():
        replay.verify(Path(args.cache)/name,h,pins)
    new._reader.unchanged()
    need(Counter(s["group"] for s in selected)==Counter(GROUPS),"fixed152 group counts")
    return selected,samples,pins,new.admission_report


CONTEXT_FIELDS=("root","job_root","reference_fit","cache","admission_sha256",
                "loader_audit","loader_audit_sha256","dedup_report")


def selection_report(args):
    selected,samples,pins,admission=context(args)
    return dict(schema=SELECTION_SCHEMA,status="FIXED152_INPUTS_ADMITTED_NO_MODEL_RESULT",
        context={k:str(getattr(args,k)) for k in CONTEXT_FIELDS},
        reference_sha256=replay.REFERENCE_SHA,dedup_sha256=DEDUP_SHA,
        selected=selected,input_fingerprints=[fingerprint(s) for s in samples],
        source_hashes=pins,loader_admission=admission,groups=GROUPS,
        selection_rule="Old88 verbatim; recovery fixed episode SHA order, early<=1s nearest0.5s and late>1s nearest2s; ties earlier current index; 32 each; exclude526 exact old duplicates",
        training_released=False,closed_loop=False,
        scope="Evaluation-set adaptation, original winning teacher states; not student SR or untouched-test performance",
        old_reference_missing_full_predictions_not_synthesized=True)


def read_selection(path, expected):
    pins={}
    report=document(path,expected,pins)
    need(report.get("schema")==SELECTION_SCHEMA and
         report.get("status")=="FIXED152_INPUTS_ADMITTED_NO_MODEL_RESULT", "wrong selection schema")
    need(report.get("reference_sha256")==replay.REFERENCE_SHA and report.get("dedup_sha256")==DEDUP_SHA,
         "selection fixed anchors")
    need(report.get("training_released") is False and report.get("closed_loop") is False,"selection scope")
    need(set(report.get("context",{}))==set(CONTEXT_FIELDS),"selection context")
    for p,h in report["source_hashes"].items():replay.verify(p,h,pins)
    args=SimpleNamespace(**report["context"])
    selected,samples,current,_=context(args)
    need(selected==report.get("selected") and report.get("groups")==GROUPS,
         "frozen selection changed, no reselection")
    need([fingerprint(s) for s in samples]==report.get("input_fingerprints"),"frozen input tensors changed")
    for p,h in current.items():
        need(report["source_hashes"].get(p)==h,"selection source graph changed")
    pins.update(current)
    return report,samples,pins


def summarize(records):
    return {g:{str(repeat):{m:float(np.mean([r[m] for r in records
            if r["group"]==g and r["repeat_history"]==repeat]))
            for m in replay.METRICS} for repeat in (False,True)} for g in GROUPS}


def validate_records(records, selected, fingerprints):
    need(isinstance(records,list) and len(records)==304,"exact152x2 predictions")
    seen={}
    for r in records:
        i=r.get("selection_index");rep=r.get("repeat_history")
        need(integer(i) and i<152 and type(rep) is bool and (i,rep) not in seen,"prediction pair identity")
        need(r.get("group")==selected[i]["group"],"prediction group")
        p=np.asarray(r.get("prediction_7x4"),dtype=np.float32)
        y=np.asarray(r.get("label_7x4"),dtype=np.float32)
        need(p.shape==y.shape==(7,4) and np.isfinite(p).all() and np.isfinite(y).all(),"full trajectories")
        need(replay.tensor_digest(torch.from_numpy(y))==fingerprints[i]["pose"]==
             r.get("label_tensor_sha256"),"prediction labels changed")
        recomputed=original.fit_metrics(torch.from_numpy(p)[None],torch.from_numpy(y)[None])
        for m,v in recomputed.items():
            need(replay.finite_number(r.get(m)) and math.isclose(r[m],float(v[0]),rel_tol=1e-6,abs_tol=1e-7),
                 "reported metric differs: "+m)
        need(set(r.get("input_tensor_sha256",{}))==set(INPUTS),"input identity fields")
        for k in ("template","polar") if rep else INPUTS:
            need(r["input_tensor_sha256"][k]==fingerprints[i][k],"input tensors changed")
        seen[i,rep]=r
    need(len(seen)==304,"paired coverage")
    for i in range(152):
        need(seen[i,False]["label_7x4"]==seen[i,True]["label_7x4"],"repeat changed label")
    return seen


def compare(records, baseline, selected, fingerprints):
    new=validate_records(records,selected,fingerprints)
    old=validate_records(baseline,selected,fingerprints)
    deltas=[]
    for key in sorted(new):
        need(new[key]["input_tensor_sha256"]==old[key]["input_tensor_sha256"],
             "cross-model inputs differ")
        deltas.append(dict(selection_index=key[0],repeat_history=key[1],
            group=new[key]["group"],**{m:new[key][m]-old[key][m] for m in replay.METRICS}))
    return dict(records=deltas,summaries=summarize(deltas))


def load_parent(model,path,loader=torch.load):
    ck=loader(path,map_location="cpu",weights_only=True,mmap=True)
    need(type(ck.get("step")) is int and ck["step"]==22707 and ck.get("kind")=="jepa"
         and ck.get("contract")==CONTRACT and type(ck.get("completed_epochs")) is int
         and ck["completed_epochs"]==1,"loaded parent identity")
    need(isinstance(ck.get("model"),dict) and ck["model"]
         and not any(k.startswith("encoder.") for k in ck["model"]),"parent saved weights")
    loaded=model.load_state_dict(ck["model"],strict=False)
    need(not loaded.unexpected_keys and all(k.startswith("encoder.") for k in loaded.missing_keys),
         "partial parent load")


def candidate_args(checkpoint, digest, step, audit_path, audit_sha):
    need(integer(step) and step>0,"explicit expected checkpoint step")
    need(isinstance(digest,str) and len(digest)==64 and all(c in "0123456789abcdef" for c in digest),
         "explicit expected checkpoint SHA")
    return SimpleNamespace(candidate_checkpoint=str(checkpoint),candidate_sha256=digest,candidate_step=step,
        training_audit=str(audit_path),training_audit_sha256=audit_sha)


def run(args):
    report,samples,pins=read_selection(args.selection,args.selection_sha256)
    root=Path(report["context"]["root"]);jobs=Path(report["context"]["job_root"])
    specs=[]
    if args.model=="pretrain":
        parent=jobs/original.MODELS["parent59866"][0]
        replay.verify(parent,PARENT_SHA,pins)
        best=candidate_args(jobs/"job_61609/task_72803/wa_hard_stt_train_a800_v1/checkpoint.pt",
            BEST_SHA,BEST_STEP,root/"artifacts/hard_stt_training_audit_61609_v1.json",BEST_AUDIT_SHA)
        audit,config,env=replay.candidate_inputs(best,pins)
        specs=[("parent59866",parent,PARENT_SHA,22707,None),
               ("best61609",Path(best.candidate_checkpoint),BEST_SHA,BEST_STEP,audit)]
        baseline=None
    else:
        required=("candidate_checkpoint","candidate_sha256","candidate_step","training_audit",
                  "training_audit_sha256","baseline_fit","baseline_fit_sha256")
        need(all(getattr(args,k,None) is not None for k in required),"candidate needs explicit final audit and baseline fit")
        need(args.candidate_sha256 not in (BEST_SHA,PARENT_SHA),"candidate is a fixed baseline")
        ca=candidate_args(args.candidate_checkpoint,args.candidate_sha256,args.candidate_step,
                          args.training_audit,args.training_audit_sha256)
        audit,config,env=replay.candidate_inputs(ca,pins)
        need(audit["parent_sha256"]==PARENT_SHA,"candidate not from audited parent59866")
        baseline=document(args.baseline_fit,args.baseline_fit_sha256,pins)
        need(baseline.get("schema")==FIT_SCHEMA and baseline.get("status")==FIT_STATUS and
             baseline.get("selection_sha256")==args.selection_sha256 and
             set(baseline.get("models",{}))=={"parent59866","best61609"},"baseline fit identity")
        for name,h,st in (("parent59866",PARENT_SHA,22707),("best61609",BEST_SHA,BEST_STEP)):
            b=baseline["models"][name]
            need(b.get("sha256")==h and type(b.get("step")) is int and b["step"]==st,"baseline model pin")
            validate_records(b["records"],report["selected"],report["input_fingerprints"])
            need(b.get("summaries")==summarize(b["records"]),"baseline summaries")
        specs=[("candidate",Path(ca.candidate_checkpoint),ca.candidate_sha256,ca.candidate_step,audit)]
    deps=replay.verify_dependencies(root,config,env,pins)
    need(torch.cuda.device_count()==1,"explicitly isolate one idle developer GPU")
    torch.set_num_threads(2);torch.manual_seed(42);torch.cuda.set_device(0)
    model=JointRobotModel(str(root),deps["encoder"],deps["wla_source"],deps["wla_checkpoint"],"jepa").cuda().eval()
    need(model.provenance==env["wla"],"constructed dependency differs from audited training")
    models={}
    for name,path,h,step,audit in specs:
        if audit is None:load_parent(model,path)
        else:replay.load_candidate(model,path,step,audit)
        records=replay.predict_records(model,samples,report["selected"])
        validate_records(records,report["selected"],report["input_fingerprints"])
        models[name]=dict(path=str(path),sha256=h,step=step,records=records,summaries=summarize(records))
    comparisons={}
    if baseline:
        for name,b in baseline["models"].items():
            comparisons["candidate_minus_"+name]=compare(models["candidate"]["records"],b["records"],
                                                       report["selected"],report["input_fingerprints"])
    else:
        comparisons["best61609_minus_parent59866"]=compare(models["best61609"]["records"],
            models["parent59866"]["records"],report["selected"],report["input_fingerprints"])
    for p,h in list(pins.items()):replay.verify(p,h,pins)
    return dict(schema=FIT_SCHEMA,status=FIT_STATUS,selection=str(Path(args.selection)),
        selection_sha256=args.selection_sha256,selected=report["selected"],models=models,
        comparisons=comparisons,source_hashes=pins,closed_loop=False,training_released=False,
        input="Only RGB/template/current polar/times; mode2 mixed; zero flow start; seed42; no pose/metadata/future in predict",
        history_diagnostic="normal and repeat_current: repeat only RGB policy history and zero relative times; labels unchanged",
        scope=report["scope"],gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__,
        peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30,gpu_bitwise_equivalence_claim=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="command",required=True)
    select=sub.add_parser("select")
    for k in CONTEXT_FIELDS:select.add_argument("--"+k.replace("_","-"),required=True)
    infer=sub.add_parser("run")
    for k in ("selection","selection_sha256","model"):
        infer.add_argument("--"+k.replace("_","-"),required=True,**({"choices":["pretrain","candidate"]} if k=="model" else {}))
    for k in ("candidate_checkpoint","candidate_sha256","training_audit","training_audit_sha256",
              "baseline_fit","baseline_fit_sha256"):infer.add_argument("--"+k.replace("_","-"))
    infer.add_argument("--candidate-step",type=int)
    for parser in (select,infer):parser.add_argument("--output",required=True)
    args=p.parse_args()
    need(not os.environ.get("MD_AK_JOB_ID") and not os.environ.get("MD_AK_TASK_ID"),"developer-only diagnostic")
    for k,v in vars(args).items():
        if v is not None and (k in CONTEXT_FIELDS or k in ("selection","output","candidate_checkpoint",
                  "training_audit","baseline_fit")) and not k.endswith("sha256"):
            q=Path(v)
            need(q.is_absolute() and str(q.resolve()).startswith("/data/nas_ray/")
                 and q.resolve()==q,"verified NAS paths only")
    replay.refuse_output(args.output)
    start=time.monotonic()
    output=selection_report(args) if args.command=="select" else run(args)
    output["elapsed_s"]=time.monotonic()-start
    replay.write_report(args.output,output)
    print(json.dumps(dict(status=output["status"],path=args.output,sha256=sha(args.output))),flush=True)


if __name__=="__main__":main()
