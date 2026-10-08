"""Read-only numeric audit of a completed failure-state continuation; no cache/release."""
import argparse
from collections import Counter, defaultdict
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time
import numpy as np
from wa.wm.failure_state_numeric import audit_numeric_windows, require
from wa.wm.robot_data import transition_records

SCHEMA = "failure_state_candidate_numeric_audit_61844_v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)


def equal(a,b,message):
    require(canonical(a)==canonical(b),message)


class Inputs:
    def __init__(self):
        self.pins = {}
    def read(self,path,*,expected=None,raw=False):
        p=Path(path)
        require(p.is_absolute() and p.is_file() and not p.is_symlink()
                and p.resolve()==p,"unsafe/missing input: "+str(p))
        data=p.read_bytes();h=hashlib.sha256(data).hexdigest()
        if expected is not None:require(h==expected,"input SHA mismatch: "+str(p))
        require(str(p) not in self.pins or self.pins[str(p)]==h,"input changed")
        self.pins[str(p)]=h
        if raw:return data
        def pairs(items):
            result={}
            for k,v in items:
                require(k not in result,"duplicate JSON key");result[k]=v
            return result
        return json.loads(data,object_pairs_hook=pairs,
                          parse_constant=lambda x:(_ for _ in ()).throw(ValueError("nonfinite JSON")))
    def verify(self):
        for p,h in self.pins.items():
            require(digest(p)==h,"input changed during audit: "+p)


def successful(ref,repeat):
    require(ref.get("complete") is True and ref.get("replay_verified") is True
            and ref.get("transport_fallback") is False
            and ref.get("verification_only") is repeat,"unqualified source winner/repeat")
    r=ref["result"]
    require(type(r.get("success")) in (bool,int,float) and r["success"]==1
            and type(r.get("collision")) in (bool,int,float) and r["collision"]==0
            and r.get("policy_init_valid") is True,"failed source winner/repeat")


def audit(root, overlay_path, overlay_sha, old_report_path, old_report_sha, *, progress=True):
    started=time.monotonic();root=Path(root);reader=Inputs()
    checkout=Path(__file__).resolve().parents[2]
    code_names=("wa/wm/failure_state_numeric.py","wa/tools/audit_failure_state_numeric.py",
                "wa/tests/test_failure_state_numeric.py","wa/wm/failure_state_labels.py",
                "wa/wm/robot_data.py","wa/tools/build_recovery_cache.py",
                "wa/tools/build_dual_teacher_cache.py")
    code_before={str(checkout/n):digest(checkout/n) for n in code_names}
    overlay=reader.read(overlay_path,expected=overlay_sha)
    require((overlay["expected_count"],overlay["new_count"],overlay["reused_count"])==(126,90,36),
            "wrong continuation scope")
    require(overlay["training_released"] is False,"overlay is not a release")
    old=reader.read(old_report_path,expected=old_report_sha)
    require(old["summary"]["valid_windows"]==1891 and old["summary"]["excluded_windows"]==148
            and old["summary"]["candidate_windows"]==2039 and old["training_released"] is False,
            "wrong independent36 reference")
    complete=reader.read(root/"COMPLETE.json");launch=reader.read(root/"launch.json")
    coverage=reader.read(root/"coverage.json")
    require(complete["status"]=="COLLECTION_PROCESSES_COMPLETE_NOT_TRAINING_RELEASE"
            and complete["formal"] is True and complete["training_released"] is False
            and complete["entries"]==90 and complete["lanes"]==8,"not complete90 process output")
    expected_keys=[e["key"] for e in overlay["remaining_entries"]]
    require(len(expected_keys)==len(set(expected_keys))==90,"bad remaining keys")
    equal(complete["keys"],expected_keys,"root completed keys")
    equal(coverage["new_keys"],expected_keys,"coverage new keys")
    equal(coverage["reused_keys"],overlay["reused_keys"],"coverage reused keys")
    require(not set(expected_keys)&set(overlay["reused_keys"]),"new/old overlap")
    for v in (complete,launch,coverage):
        require(v["continuation_sha256"]==overlay_sha
                and v["teacher_boundary_policy"]=="missing_rvq_or_final_l2_v1"
                and v["training_released"] is False,"source boundary identity")
    require(launch["source_git_status"]=="" and launch["expected_entries"]==90
            and launch["frozen_entries"]==36,"source launch scope")
    plan=reader.read(launch["plan"],expected=launch["plan_sha256"])
    require(launch["plan_sha256"]==overlay["base_plan"]["sha256"],"base plan identity")
    episodes=[];no_candidate=[];allkeys=[]
    for lane,entries in enumerate(overlay["remaining_lanes"]):
        collection=root/f"lane{lane}"/"collection"
        marker=reader.read(collection/"COMPLETE.json")
        identity=reader.read(collection/"plan_identity.json")
        wanted=[e["key"] for e in entries]
        equal(marker["keys"],wanted,"original lane order")
        require(marker["entries"]==marker["expected"]==len(wanted) and marker["shard"]==lane
                and marker["development"] is False and marker["training_released"] is False,
                "lane completion mismatch")
        for v in (marker,identity):
            require(v["continuation_sha256"]==overlay_sha
                    and v["teacher_boundary_policy"]==launch["teacher_boundary_policy"],
                    "lane source provenance")
        records_path=collection/"records.jsonl"
        raw=reader.read(records_path,raw=True)
        rows=[json.loads(s) for s in raw.splitlines() if s.strip()]
        equal([r["key"] for r in rows],wanted,"lane records do not match workset")
        for row,entry in zip(rows,entries):
            allkeys.append(row["key"])
            base=collection/entry["task"]/entry["key"]
            equal(reader.read(base/"search.json"),row,"records/search mismatch")
            require(row["task"]==entry["task"]=="stt"
                    and row["training_released"] is False and row["score_backfill_allowed"] is False
                    and row["teacher_boundary_policy"]==launch["teacher_boundary_policy"]
                    and row["continuation_sha256"]==overlay_sha,"row scope")
            equal(row["baseline_result_unchanged"],entry["baseline_result"],"baseline altered")
            accepted=row["accepted"]
            if accepted is None:
                require(row["outcome"]=="no_valid_teacher_recovery","unexpected noncandidate")
                no_candidate.append(dict(key=row["key"],lane=lane,outcome=row["outcome"]));continue
            require(row["outcome"]=="repeated_teacher_recovery_candidate"
                    and accepted["training_released"] is False and accepted["candidate_only"] is True,
                    "unsupported candidate")
            teacher=accepted["teacher"];k=accepted["takeover_step"]
            require(teacher in ("oracle","lightnav") and type(k) is int and k>=0,"bad winner")
            branch=base/f"{teacher}_{k:04d}";repeat=base/f"{teacher}_{k:04d}_repeat"
            equal(accepted["artifact_root"],str(branch),"original winner path")
            equal(accepted["repeat_artifact_root"],str(repeat),"repeat path")
            attempt=row["attempts"][-1]
            require(attempt["takeover_step"]==k and attempt["repeat_valid"] is True,"repeat source")
            ref=attempt["branches"][teacher];rep=attempt["repeat"]
            successful(ref,False);successful(rep,True)
            equal(attempt["selection"],accepted["selection"],"winner selection source")
            require(attempt["selection"]["selected_teacher"]==teacher,"wrong selected teacher")
            docs={n:reader.read(branch/n) for n in ("metadata.json","observations.json",
                "actions.json","windows.json","result.json","admission.json","complete.json",
                "first_start.json","takeover.json")}
            meta,obs,acts,wins=(docs[n] for n in ("metadata.json","observations.json","actions.json","windows.json"))
            for key,value in dict(experiment=row["experiment"],protocol_sha256=row["protocol_sha256"],
                task=row["task"],key=row["key"],teacher=teacher,takeover_step=k,verification_only=False,
                continuation_sha256=overlay_sha,teacher_boundary_policy=launch["teacher_boundary_policy"],
                partition="evaluation_adaptation",training_eligible=False,camera_alignment_verified=True).items():
                equal(meta.get(key),value,"winner metadata: "+key)
            require(len(wins)==ref["candidate_windows"]==attempt["selected_candidate_windows"]>0,
                    "candidate count differs")
            equal(docs["result.json"],ref["result"],"winner result differs")
            adm=docs["admission.json"]
            require(adm["complete"] is True and adm["replay_verified"] is True
                    and adm["transport_fallback"] is False and adm["verification_only"] is False
                    and adm["training_eligible"] is False and adm["training_released"] is False
                    and not adm["issues"],"winner admission source invalid")
            require(docs["complete.json"]["frames"]==len(obs)
                    and docs["complete.json"]["candidate_windows"]==len(wins),"raw completion count")
            repmeta=reader.read(repeat/"metadata.json")
            require(repmeta["verification_only"] is True and repmeta["takeover_step"]==k
                    and repmeta["teacher"]==teacher,"repeat identity")
            equal(reader.read(repeat/"result.json"),rep["result"],"repeat result differs")
            equal(reader.read(repeat/"windows.json"),[],"repeat windows duplicated")
            studentmeta=reader.read(base/"student/metadata.json")
            first=reader.read(base/"student/first_start.json")
            for field in ("initial_bbox_rgb_xyxy","initial_bbox_sensor_xyxy_original","initial_bbox_repair"):
                equal(meta.get(field),studentmeta.get(field),"takeover replaced template field")
            require(meta["initial_bbox_status"] in ("VERIFIED_CONFIG_AND_SEMANTIC",
                    "VERIFIED_FROZEN_FIRST_RGB_REPAIR"),"unverified template")
            equal(docs["first_start.json"]["rgb_sha256"],first["rgb_sha256"],"initial raw digest differs")
            equal(docs["takeover.json"]["initial_rgb_sha256"],first["rgb_sha256"],"takeover template changed")
            require(reader.read(branch/"rgb_0000.png",raw=True) ==
                    reader.read(base/"student/rgb_0000.png",raw=True),"episode-zero RGB bytes differ")
            require(all(o["frame"]==f"rgb_{i:04d}.png" for i,o in enumerate(obs)),"frame index mapping")
            numeric=audit_numeric_windows(obs,acts,wins,k)
            commands,bad,transition_stats=transition_records(branch,obs)
            require(np.array_equal(commands,numeric["derived"]["commands"])
                    and np.array_equal(bad,numeric["derived"]["transition_bad"]),
                    "actual RobotWorldData transition calculation differs")
            summary=numeric["summary"]
            episode=dict(lane=lane,task=row["task"],key=row["key"],teacher=teacher,
                takeover_step=k,category="start" if k==0 else "mid",branch=str(branch),
                repeat_branch=str(repeat),source_experiment=row["experiment"],
                source_protocol_sha256=row["protocol_sha256"],
                source_boundary_policy=row["teacher_boundary_policy"],
                observed_frames=len(obs),takeover_timestamp_s=obs[k]["timestamp_s"],
                **summary,valid_window_indices=numeric["valid_window_indices"],
                excluded=numeric["excluded"],
                old_mask_disagreement_indices=numeric["old_mask_disagreement_indices"],
                clipped_unclipped_endpoint_disagreement_indices=numeric["clipped_unclipped_endpoint_disagreement_indices"],
                transition_stats=transition_stats,template_index=0,causal_history_verified=True,
                future_teacher_ownership_verified=True,jepa_command_available_for_valid=True,
                previous_action_available_for_valid=True,raw_alpha_reconstruction=False,
                raw_media_audited_here=False,repeat_windows_counted=0)
            episodes.append(episode)
            if progress:
                print(json.dumps(dict(phase="NUMERIC_EPISODE",completed=len(episodes),key=row["key"],
                    candidate=summary["candidate_windows"],valid=summary["valid_windows"],
                    excluded=summary["excluded_windows"])),flush=True)
    require(len(allkeys)==len(set(allkeys))==90 and set(allkeys)==set(expected_keys),"incomplete90")
    totals=Counter();reasons=Counter();maxd=defaultdict(float);groups={}
    fields=("candidate_windows","derived_valid_windows","valid_windows","excluded_windows",
            "early_candidates","late_candidates","early_valid","late_valid",
            "valid_history_uses_student_prefix","valid_jepa_history_uses_student_prefix",
            "valid_proprio_uses_student_prefix","old_mask_disagreement_windows",
            "clipped_unclipped_endpoint_disagreement_windows","old_endpoint_out_of_range_windows",
            "future_label_student_action_count","jepa_next_target_student_action_count",
            "per_window_calls","rejection_reason_overlap_windows")
    for e in episodes:
        totals.update({f:e[f] for f in fields});reasons.update(e["rejection_reason_counts"])
        for k,v in e["max_deltas"].items():maxd[k]=max(maxd[k],v)
    for group in ("teacher","category","source_experiment"):
        g={}
        for e in episodes:
            t=g.setdefault(e[group],Counter(episodes=0));t.update(episodes=1)
            t.update({f:e[f] for f in ("candidate_windows","valid_windows","excluded_windows",
                                      "early_valid","late_valid")})
        groups[group]={k:dict(v) for k,v in g.items()}
    reader.verify()
    code_after={p:digest(p) for p in code_before};equal(code_after,code_before,"numeric code changed")
    return dict(schema=SCHEMA,root=str(root),job_id=61844,task_id=73066,
        source_collection=dict(experiment=launch["experiment"],protocol_sha256=launch["protocol_sha256"],
          boundary_policy=launch["teacher_boundary_policy"],commit=launch["source_commit"],source=launch["source"]),
        source_files=reader.pins,code_sha256_before=code_before,code_sha256_after=code_after,
        summary=dict(completed_searches=90,accepted_original_episodes=len(episodes),
            no_candidate_searches=len(no_candidate),episodes_with_valid_windows=sum(e["valid_windows"]>0 for e in episodes),
            **dict(totals),rejection_reason_counts=dict(reasons),max_deltas=dict(maxd),
            repeat_windows_counted=0,student_prefix_actions_used_as_future_labels=0,
            all_candidate_future_transitions_teacher_owned=True),
        groups=groups,episodes=episodes,no_candidate=no_candidate,
        zero_valid_episodes=[e["key"] for e in episodes if not e["valid_windows"]],
        old36_reference=dict(path=str(old_report_path),sha256=old_report_sha,read_only_not_reaudited=True,
            summary=old["summary"]),
        combined_counts_arithmetic_only=dict(candidate_windows=totals["candidate_windows"]+2039,
            valid_windows=totals["valid_windows"]+1891,excluded_windows=totals["excluded_windows"]+148,
            not_full126_release=True),
        finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic()-started,training_eligible=False,training_released=False,
        cache_generated=False,no_success_rate=True,full_search_media_audit=False,
        limitations=["Numerical audit only; separate complete raw/search/126 admission remains mandatory.",
          "Only original selected winner windows counted; repeat/student/failed branches contribute no labels.",
          "Student prefix is causal observation/command context only, never teacher action targets.",
          "Raw sensor alpha is not reconstructed; first PNG bytes and recorded raw digest compared only.",
          "No cache/training files changed; old36 numeric source identities remain unchanged."])


def main():
    p=argparse.ArgumentParser()
    for n in ("root","overlay","overlay-sha","old-report","old-report-sha","output"):
        p.add_argument("--"+n,required=True)
    a=p.parse_args()
    output=Path(a.output)
    require(output.is_absolute() and output.parent.is_dir() and not output.exists(),"exclusive report path required")
    report=audit(a.root,a.overlay,a.overlay_sha,a.old_report,a.old_report_sha)
    report["reproduce_command"]=shlex.join([sys.executable,"-B","-m",
        "wa.tools.audit_failure_state_numeric",*sys.argv[1:]])
    with output.open("x") as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write("\n")
    readback=json.loads(output.read_bytes());equal(readback,report,"report readback mismatch")
    print(json.dumps(dict(path=str(output),bytes=output.stat().st_size,sha256=digest(output),
                          summary=report["summary"],zero_valid_episodes=report["zero_valid_episodes"])),flush=True)


if __name__=="__main__":
    main()
