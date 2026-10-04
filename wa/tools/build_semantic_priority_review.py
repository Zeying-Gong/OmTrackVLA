"""Immutable priority165 report and paired video page; never edits evaluation outputs."""
import collections,html,json,pathlib,subprocess
from wa.wm.semantic_targeted import load_targeted,key
R=pathlib.Path("/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928")
P=pathlib.Path("/data/nas_ray/project/md-ak/users/zeying.gong/job_61171/task_72191/wa_semantic_targeted_v1")
m=json.loads((R.parent/"WLA-EVT-20260925/evt_full_20260926/manifest.json").read_text())
plan=load_targeted(R/"artifacts/semantic_targeted_plan_20261004_v1.json",m,"7ad7697df1a29b6cd5589b784390a31c4483e4118a1dbff79a57e944a38be458")
assert len(list(P.glob("shard_*/PRIORITY_COMPLETE.json")))==8
wanted={tuple(k) for k in plan["prior_invalid_keys"]}
old={key(r):r for r in map(json.loads,pathlib.Path(plan["baseline"]).read_text().splitlines())}
rows=[dict(r) for r in plan["completed_rows"] if key(r) in wanted]
for p in P.glob("shard_*/episodes.jsonl"):
    for line in p.read_text().splitlines():
        r=json.loads(line)
        if key(r) in wanted:r["artifact_root"]=str(p.parent);rows.append(r)
assert len(rows)==165 and {key(r) for r in rows}==wanted
assert all(r["initial_rgb_sha256"]==old[key(r)]["initial_rgb_sha256"] for r in rows)
def stats(rs):
    return dict(n=len(rs),initialized=sum(r["policy_init_valid"] for r in rs),
        success=int(sum(r["success"] for r in rs)),invalid=sum(not r["policy_init_valid"] for r in rs),
        other_failure=sum(r["policy_init_valid"] and not r["success"] for r in rs),
        human_collision=int(sum(r["collision"] for r in rs)),
        valid_failure_status=dict(collections.Counter(r["status"] for r in rs if r["policy_init_valid"] and not r["success"])))
report=dict(scope="old165initialization_failures_only_not_full_SR",total=stats(rows),
    by_task={t:stats([r for r in rows if r["task"]==t]) for t in ("stt","dt","at")},
    initial_rgb_pairs=165,initial_rgb_mismatches=0,
    invalid_keys=[list(key(r)) for r in rows if not r["policy_init_valid"]])
out=R/"artifacts/semantic_priority_review_61171";out.mkdir(exist_ok=False);(out/"sources").mkdir()
roots={}
def video(r):
    root=r["artifact_root"]
    if root not in roots:
        label="r"+str(len(roots));roots[root]=label;(out/"sources"/label).symlink_to(root,target_is_directory=True)
    rel=r["task"]+"/_review/"+r["key"]+"/review.mp4";f=pathlib.Path(root)/rel
    assert f.is_file(),str(f)
    return "sources/"+roots[root]+"/"+rel,f
cards=[]
for r in sorted(rows,key=lambda r:(r["policy_init_valid"],r["success"],r["task"],r["key"])):
    nv,f=video(r);ov,_=video(old[key(r)])
    probe=subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","json",str(f)],capture_output=True,text=True,check=True)
    assert float(json.loads(probe.stdout)["format"]["duration"])>0
    category="invalid" if not r["policy_init_valid"] else ("success" if r["success"] else "other")
    title=html.escape(r["task"].upper()+" "+r["key"])
    cards.append('<article data-state="'+category+'"><h3>'+title+' — '+category+'</h3><p>状态 '+html.escape(r["status"])+'；HumanCollision '+str(r["collision"])+'</p><div class="pair"><section>旧协议（初始化失败）<video controls preload="none" src="'+ov+'"></video></section><section>修复后<video controls preload="none" src="'+nv+'"></video></section></div></article>')
report["video_metadata_checked"]=165
(out/"report.json").write_text(json.dumps(report,indent=2));(out/"episodes.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>WA 旧165初始化失败复评</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto;background:#eef1f4;color:#172333}article{background:white;padding:16px;margin:16px 0;border-radius:10px}.pair{display:flex;gap:18px}.pair section{width:50%}video{width:100%;display:block}button{padding:10px;margin:6px}pre{white-space:pre-wrap}</style><h1>旧165条初始化失败：配对复评</h1><p>158恢复初始化；138闭环成功；7仍初始化失败；20初始化后未成功。只代表该失败子集，不是全量成功率。相同权重、RGB、控制和判据；修复MP3D语义渲染。CR指目标人距离曾小于0.5m，不是门框碰撞。</p><p>左右视频独立播放；新视频165个ffprobe元数据检查通过，不代表逐帧人工审阅。</p><button onclick="pick('all')">全部165</button><button onclick="pick('invalid')">仍初始化失败7</button><button onclick="pick('other')">后续失败20</button><button onclick="pick('success')">成功138</button><p id="count"></p>'''+''.join(cards)+'''<script>function pick(s){let n=0;document.querySelectorAll('article').forEach(e=>{e.hidden=s!=='all'&&e.dataset.state!==s;if(!e.hidden)n++});document.getElementById('count').textContent='显示 '+n+' 条'}pick('all')</script></html>'''
(out/"index.html").write_text(page)
print(json.dumps(report))
