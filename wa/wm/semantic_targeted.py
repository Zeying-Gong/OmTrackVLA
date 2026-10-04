"""Audited semantic-scope continuation; prioritize old invalid starts, preserve provenance."""
import copy,json,time
from pathlib import Path
from wa.wm.full_mixed_contract import TASKS,MANIFEST_SHA,validate_ready
from wa.wm.full_mixed_resume import digest,validate_plan,verify_new_rows
from wa.wm.semantic_scene import VERSION,is_mp3d_ply_scene,provenance
PLAN_VERSION="semantic_targeted_v1"
METRICS=("finish","status","success","following_rate","following_step","total_step","collision","policy_init_valid","initial_rgb_sha256")
def key(r): return (r["task"],r["key"])
def validate_targeted(plan,manifest,verify_files=False):
    validate_plan(plan,manifest,verify_files)
    assert plan["version"]==PLAN_VERSION and plan["semantic_config_sha256"]==provenance()["config_sha256"]
    scopes={(t,e["key"]):is_mp3d_ply_scene(e["scene_id"]) for t in TASKS for e in manifest["tasks"][t]["episodes"]}
    assert sum(scopes.values())==2163 and len(scopes)-sum(scopes.values())==2052
    priorities={tuple(x) for x in plan["prior_invalid_keys"]}
    assert len(priorities)==165 and all(scopes[k] for k in priorities)
    for r in plan["completed_rows"]:
        if scopes[key(r)]:
            assert r["semantic_protocol"]==VERSION and r["semantic_ply_repaired"] is True
            assert r["reuse_reason"]=="already_completed_repaired"
        else:
            assert r["reuse_reason"] in ("unaffected_hm3d_baseline","already_completed_unaffected_hm3d")
        assert r.get("artifact_root")
    for lane in plan["lanes"]:
        for e in lane:
            assert scopes[key(e)] and e["priority"]==(key(e) in priorities)
        assert [e["priority"] for e in lane]==sorted([e["priority"] for e in lane],reverse=True)
    return plan
def build_targeted(parent,baseline,manifest):
    parent=Path(parent);baseline=Path(baseline)
    oldrows=[json.loads(l) for l in baseline.read_text().splitlines()]
    from wa.wm.full_mixed_contract import summarize
    summarize(oldrows,manifest)
    old={key(r):r for r in oldrows}; new={}; files=[dict(path=str(baseline),sha256=digest(baseline))]
    for p in sorted(parent.glob("shard_*/episodes.jsonl")):
        raw=p.read_text(); assert not raw or raw.endswith("\n"),"partial row: "+str(p)
        ready=p.parent/"server_ready.json";validate_ready(json.loads(ready.read_text()))
        for f in (p,ready):files.append(dict(path=str(f),sha256=digest(f)))
        for line in raw.splitlines():
            r=json.loads(line);k=key(r);assert k not in new and k in old
            assert r["semantic_protocol"]==VERSION
            assert r["initial_rgb_sha256"]==old[k]["initial_rgb_sha256"],("RGB changed",k)
            r["artifact_root"]=str(p.parent);new[k]=r
    for root in {r["artifact_root"] for r in oldrows}:
        f=Path(root)/"server_ready.json";validate_ready(json.loads(f.read_text()))
        files.append(dict(path=str(f),sha256=digest(f)))
    priorities={key(r) for r in oldrows if not r["policy_init_valid"]}
    rows=[];todo=[];counts={"new_mp3d":0,"new_hm3d":0,"old_hm3d":0}
    for t in TASKS:
        for e in manifest["tasks"][t]["episodes"]:
            k=(t,e["key"]);affected=is_mp3d_ply_scene(e["scene_id"])
            if k in new:
                r=copy.deepcopy(new[k]);assert r["semantic_ply_repaired"]==affected
                if not affected:
                    assert all(r[x]==old[k][x] for x in METRICS),("unaffected result differs",k)
                r["reuse_reason"]="already_completed_repaired" if affected else "already_completed_unaffected_hm3d"
                counts["new_mp3d" if affected else "new_hm3d"]+=1;rows.append(r)
            elif not affected:
                r=copy.deepcopy(old[k]);r["reuse_reason"]="unaffected_hm3d_baseline"
                r["source_semantic_protocol"]=r.get("semantic_protocol","original")
                r["semantic_protocol"]="equivalent_unaffected_hm3d";r["semantic_ply_repaired"]=False
                counts["old_hm3d"]+=1;rows.append(r)
            else:todo.append(dict(task=t,key=e["key"],priority=k in priorities))
    todo.sort(key=lambda e:not e["priority"]);lanes=[[] for _ in range(8)]
    for e in todo:lanes[min(range(8),key=lambda i:len(lanes[i]))].append(e)
    plan=dict(version=PLAN_VERSION,manifest_sha256=MANIFEST_SHA,semantic_config_sha256=provenance()["config_sha256"],
              parent_job=61144,parent_task=72164,parent_root=str(parent),baseline=str(baseline),
              completed_rows=rows,lanes=lanes,source_files=files,prior_invalid_keys=sorted(priorities),reuse_counts=counts,
              hm3d_equivalence="prepare_episode no-op; per-episode seed reset; ready contracts and all completed HM3D metrics/RGB exact")
    return validate_targeted(plan,manifest,True)
def load_targeted(path,manifest,expected_sha):
    assert digest(path)==expected_sha,"targeted plan SHA mismatch"
    return validate_targeted(json.loads(Path(path).read_text()),manifest,True)
def priority_barrier(out,timeout=7200):
    out=Path(out)
    with (out/"PRIORITY_COMPLETE.json").open("x") as f:json.dump({"stage":"prior_invalid_complete"},f)
    deadline=time.monotonic()+timeout
    while not all((out.parent/("shard_%02d"%i)/"PRIORITY_COMPLETE.json").is_file() for i in range(8)):
        if time.monotonic()>deadline:raise TimeoutError("priority barrier")
        time.sleep(3)
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--parent",required=True);p.add_argument("--baseline",required=True);p.add_argument("--manifest",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    assert digest(a.manifest)==MANIFEST_SHA
    result=build_targeted(a.parent,a.baseline,json.loads(Path(a.manifest).read_text()))
    with Path(a.output).open("x") as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(reused=len(result["completed_rows"]),remaining=sum(map(len,result["lanes"])),priority_remaining=sum(e["priority"] for lane in result["lanes"] for e in lane),reuse_counts=result["reuse_counts"],lanes=list(map(len,result["lanes"])),sha256=digest(a.output))))
