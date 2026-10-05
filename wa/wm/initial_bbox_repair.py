"""Seven-row first-frame annotation repair; never alter simulator semantics or metrics."""
import hashlib,json,math
from pathlib import Path
KEYS={(t,"pRbA3pwrgk9/"+e) for t,es in (("stt",["27","38"]),("dt",["27","38"]),("at",["27","38","71"])) for e in es}
VERSION="initial_rgb_mesh_bbox_v1"
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_plan(path,expected_sha):
    if digest(path)!=expected_sha: raise ValueError("repair plan hash mismatch")
    p=json.loads(Path(path).read_text())
    if p["version"]!=VERSION or len(p["repairs"])!=7: raise ValueError("wrong repair scope")
    if {(r["task"],r["key"]) for r in p["repairs"]}!=KEYS: raise ValueError("wrong repair keys")
    lanes=[r for lane in p["lanes"] for r in lane]
    if len(p["lanes"])!=2 or len(lanes)!=7 or {(r["task"],r["key"]) for r in lanes}!=KEYS: raise ValueError("wrong lanes")
    if digest(p["baseline"])!=p["baseline_sha256"]: raise ValueError("baseline changed")
    old=[json.loads(x) for x in Path(p["baseline"]).read_text().splitlines()]
    if len(old)!=4215 or len({(r["task"],r["key"]) for r in old})!=4215: raise ValueError("bad baseline")
    if {(r["task"],r["key"]) for r in old if not r["policy_init_valid"]}!=KEYS: raise ValueError("invalid scope changed")
    for r in p["repairs"]:
        if digest(r["probe_report"])!=r["probe_sha256"]: raise ValueError("probe changed")
        d=json.loads(Path(r["probe_report"]).read_text())
        if any(v is False for v in d["checks"].values()): raise ValueError("paired static check failed")
        a,b=d["modes"]["original"],d["modes"]["fixed"]
        if a["target_pixels"]!=0 or b["target_pixels"]<=0 or r["bbox"]!=b["bbox"] or r["rgb_raw_sha256"]!=a["rgb_sha"]: raise ValueError("invalid annotation evidence")
        x1,y1,x2,y2=r["bbox"]
        if not all(math.isfinite(x) for x in r["bbox"]) or not(0<=x1<x2 and 0<=y1<y2): raise ValueError("invalid box")
    return p
def repaired_detector(detector, rgb, repair):
    import copy,numpy as np
    h=hashlib.sha256(np.ascontiguousarray(rgb[...,:3]).tobytes()).hexdigest()
    if h!=repair["rgb_raw_sha256"]: raise ValueError("repair first RGB mismatch")
    sensor=detector.get("agent_1_main_humanoid_detector_sensor",detector)
    if list(sensor["box"])!=[0.,0.,0.,0.]: raise ValueError("refuse overriding a valid/nonzero box")
    box=repair["bbox"];x1,y1,x2,y2=box
    if not(0<=x1<x2<=rgb.shape[1] and 0<=y1<y2<=rgb.shape[0]): raise ValueError("bbox out of frame")
    result=copy.deepcopy(detector)
    result.get("agent_1_main_humanoid_detector_sensor",result)["box"]=np.asarray(box,dtype=np.float32)
    return result
def merge_rows(plan,new):
    if len(new)!=7 or {(r["task"],r["key"]) for r in new}!=KEYS: raise ValueError("missing/duplicate repaired row")
    old=[json.loads(x) for x in Path(plan["baseline"]).read_text().splitlines()]
    previous={(r["task"],r["key"]):r for r in old}
    for r in new:
        if r["initial_rgb_sha256"]!=previous[(r["task"],r["key"])]["initial_rgb_sha256"]: raise ValueError("initial JPEG mismatch")
        if r.get("initialization_repair")!=VERSION or r.get("semantic_protocol")!="mp3d_semantic_ply_v1": raise ValueError("wrong repair protocol")
    replacement={(r["task"],r["key"]):r for r in new}
    rows=[replacement.get((r["task"],r["key"]),r) for r in old]
    assert sum(r==o for r,o in zip(rows,old) if (o["task"],o["key"]) not in KEYS)==4208
    return rows
