"""Freeze seven first-frame GT annotations from paired no-action renderer checks."""
import json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
from wa.wm.initial_bbox_repair import KEYS,VERSION,load_plan,digest
R=Path("/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928")
out=R/"artifacts/initial_bbox_repair_v1";out.mkdir(exist_ok=False)
baseline=Path("/data/nas_ray/project/md-ak/users/zeying.gong/job_61171/task_72191/wa_semantic_targeted_v1/combined_episodes.jsonl")
repairs=[];canvas=Image.new("RGB",(384*4,404*2),"white")
for i,(task,key) in enumerate(sorted(KEYS)):
    scene,eid=key.split("/")
    probe=R/f"artifacts/residual7_no_semantic_mesh_v2_{task}_{scene}_{eid}/report.json"
    d=json.loads(probe.read_text());a,b=d["modes"]["original"],d["modes"]["fixed"]
    assert all(v is not False for v in d["checks"].values())
    assert a["target_pixels"]==0 and b["target_pixels"]>0
    rec=dict(task=task,key=key,bbox=b["bbox"],rgb_raw_sha256=a["rgb_sha"],probe_report=str(probe),probe_sha256=digest(probe),target_pixels=b["target_pixels"])
    repairs.append(rec)
    img=Image.open(probe.parent/"fixed_mask.png").convert("RGB")
    canvas.paste(img,(i%4*384,i//4*404))
    ImageDraw.Draw(canvas).text((i%4*384+5,i//4*404+384),f"{task.upper()} {eid} | {b['target_pixels']} pixels",fill="black")
canvas.save(out/"first_frame_review.png")
p=dict(version=VERSION,baseline=str(baseline),baseline_sha256=digest(baseline),repairs=repairs,lanes=[repairs[::2],repairs[1::2]],scope="seven zero initial bbox only; no subsequent semantic or metric change")
path=out/"plan.json";path.write_text(json.dumps(p,indent=2))
load_plan(path,digest(path))
(out/"index.html").write_text('<!doctype html><meta charset="utf-8"><h1>Seven first-frame GT bbox repairs — static, not success results</h1><img src="first_frame_review.png" style="max-width:100%">')
print(json.dumps({"path":str(path),"sha256":digest(path),"repairs":repairs},indent=2))
