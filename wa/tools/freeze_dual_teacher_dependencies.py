"""Freeze external inference dependencies and model files before formal collection."""
import hashlib
from pathlib import Path
R=Path("/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928")
B=R.parent/"OmTrackVLA-da3-polar-20260924";W=R.parent/"WLA-EVT-20260925";L=R.parent/"LightNav-0"
S=Path(__file__).resolve().parents[2]
paths=set()
for manifest in ("closed_loop_dependencies.sha256","full_eval_dependencies.sha256"):
    for line in (S/"wa/wm"/manifest).read_text().splitlines():
        expected,name=line.split(maxsplit=1)
        p=Path(name.strip());h=hashlib.sha256(p.read_bytes()).hexdigest()
        if h!=expected:raise ValueError("existing dependency drift: "+str(p))
        paths.add(p)
paths.update([B/"oracle_modular_follow.py",B/"modular_obstacle_map.py",
              B/"scripts/da3/teacher_recorder.py",L/"evt_bench/trackvla_client_agent.py"])
paths.update((W/"lightnav_transport_20260927").glob("*.py"))
paths.update((L/"src").rglob("*.py"))
paths.update(p for p in (L/"checkpoints/LightNav-0").rglob("*") if p.is_file() and p.suffix not in (".png",".md"))
def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()
dest=S/"wa/wm/dual_teacher_dependencies.sha256"
with dest.open("x") as f:
    for p in sorted(paths):f.write(digest(p)+"  "+str(p)+"\n")
print("FROZEN_DEPENDENCIES",len(paths),str(dest))
