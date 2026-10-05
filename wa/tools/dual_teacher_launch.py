"""One bounded developer pair, or all allocated lanes of the full collection."""
import argparse, concurrent.futures, json, os, signal, subprocess, sys, time, threading
from pathlib import Path
R=Path("/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928")
S=Path(__file__).resolve().parents[2];B=R.parent/"OmTrackVLA-da3-polar-20260924"
W=R.parent/"WLA-EVT-20260925";E=R.parent.parent/"envs";L=R.parent/"LightNav-0"

def main():
 p=argparse.ArgumentParser();p.add_argument("output");p.add_argument("--formal",action="store_true")
 p.add_argument("--development-three",action="store_true")
 p.add_argument("--development-key")
 p.add_argument("--resume-plan");p.add_argument("--resume-sha")
 p.add_argument("--audit-only",action="store_true");a=p.parse_args()
 if not a.formal:assert not os.environ.get("MD_AK_JOB_ID")
 if a.development_key:assert not a.formal and not a.development_three
 resume=None
 if bool(a.resume_plan)!=bool(a.resume_sha):raise ValueError('resume plan/hash required together')
 if a.resume_plan:
  from wa.wm.dual_teacher_resume import load_resume
  resume=load_resume(a.resume_plan,a.resume_sha,verify_artifacts=True)
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 if a.formal:
  import torch
  count=torch.cuda.device_count()
  assert count==8,("formal collection requires8 actual GPUs",count)
  devices=os.environ.get("CUDA_VISIBLE_DEVICES","").split(",") if os.environ.get("CUDA_VISIBLE_DEVICES") else [str(i) for i in range(count)]
 else:devices=[os.environ.get("CUDA_VISIBLE_DEVICES","3")]
 stopping=threading.Event()
 def lane(i):
  dest=out/f"lane{i}";dest.mkdir()
  children=[]
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices[i],PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1",
   OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
   TRACKVLA_SAVE_VIDEO="0",SAVE_VIDEO="0",TRACKVLA_VERBOSE_STEPS="0",TRACKVLA_LIVE_FRAME_INTERVAL="0",
   HABITAT_SIM_LOG="quiet",MAGNUM_LOG="quiet")
  env.pop("DA3_EVT_CHECKPOINT",None)
  xb=Path(os.environ.get("XVFB_ROOT","/tmp/omtrackvla_xvfb_root_0"))
  assert (xb/"usr/bin/Xvfb").is_file()
  env.update(PATH=str(xb/"usr/bin")+":"+env["PATH"],XKB_CONFIG_ROOT=str(xb/"usr/share/X11/xkb"),
    OMTRACKVLA_GLX_LIB_DIRS="/usr/lib/x86_64-linux-gnu:/usr/lib64:/usr/local/nvidia/lib64:"+str(xb/"usr/lib/x86_64-linux-gnu")+":"+str(xb/"lib/x86_64-linux-gnu"))
  def spawn(cmd,en,log,cwd):
   with (dest/log).open("w") as f:
    proc=subprocess.Popen(cmd,env=en,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   children.append(proc);return proc
  try:
   port=19150+i
   if not a.audit_only:
    en=dict(env,PYTHONPATH=str(L/"src"),PYTHONSAFEPATH="1")
    en["LD_LIBRARY_PATH"]=":".join(map(str,(E/"lightnav/lib").glob("python*/site-packages/nvidia/*/lib")))+":"+env.get("LD_LIBRARY_PATH","")
    ready=dest/"teacher.ready"
    server=spawn([str(E/"lightnav/bin/python"),"-P",str(W/"lightnav_transport_20260927/server_transport.py"),
      "--task","tracking","--model_path",str(L/"checkpoints/LightNav-0"),"--backend","vllm_local",
      "--gpu_memory_utilization","0.45","--max_batch_size","1","--host","127.0.0.1","--port",str(port),
      "--ready_file",str(ready)],en,"teacher.log",B)
    deadline=time.monotonic()+600
    while not ready.exists():
     if stopping.is_set():raise RuntimeError("another collection lane failed")
     if server.poll() is not None:raise RuntimeError("LightNav server exited")
     if time.monotonic()>deadline:raise TimeoutError("LightNav startup")
     time.sleep(2)
   en=dict(env,WA_DEVELOPMENT="0" if a.formal else "1",
      PYTHONPATH=":".join(map(str,[B/"artifacts/lightnav_client_runtime",B/"artifacts/official_runtime",B/"torch_overlay",B,B/"habitat-lab",S,W])),
      OMTRACKVLA_XVFB_DISPLAY_NUM=str(530+i),OMTRACKVLA_HAB_SIM_GLX_ROOT=str(B),OMTRACKVLA_XVFB_LOG=str(dest/"xvfb.log"))
   cmd=[str(B/"scripts/runtime/run_xvfb.sh"),str(E/"habitat/bin/python"),"-u","-m","wa.wm.dual_teacher_collect",
     "--manifest",str(W/"evt_full_20260926/manifest.json"),"--output",str(dest/"collection"),
     "--repair-plan",str(R/"artifacts/initial_bbox_repair_v2/plan.json"),
     "--repair-sha","6535f7b9a53e399ba7f2323c2d7f72d223146ee77f090223eb260f5885f6269a",
     "--teacher-url",f"ws://127.0.0.1:{port}","--shard",str(i),"--shards",str(len(devices))]
   if a.audit_only:cmd+=["--audit-only"]
   elif not a.formal:cmd+=["--development-three" if a.development_three else "--development-one"]
   if a.development_key:cmd+=["--development-key",a.development_key]
   if resume:cmd+=["--resume-plan",a.resume_plan,"--resume-sha",a.resume_sha]
   worker=spawn(cmd,en,"worker.log",B)
   deadline=time.monotonic()+(170000 if a.formal else 600)
   while worker.poll() is None:
    if stopping.is_set():raise RuntimeError("another collection lane failed")
    if time.monotonic()>deadline:raise TimeoutError("collection worker timeout")
    time.sleep(1)
   rc=worker.returncode
   if rc:raise RuntimeError(f"worker failed: {rc}; {dest}")
  except BaseException:
   stopping.set();raise
  finally:
   for proc in reversed(children):
    if proc.poll() is None:
     os.killpg(proc.pid,signal.SIGTERM)
     try:proc.wait(timeout=15)
     except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
 with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:list(pool.map(lane,range(len(devices))))
 if not a.audit_only:
  counts=[json.loads((out/f"lane{i}/collection/COMPLETE.json").read_text())["pairs"] for i in range(len(devices))]
  assert sum(counts)==(resume['remaining_count'] if resume else (4215 if a.formal else (3 if a.development_three else 1))),counts
  if resume:assert counts==[len(x) for x in resume['lanes']],counts
  (out/"COMPLETE.json").write_text(json.dumps(dict(pairs=sum(counts),formal=a.formal,training_released=False)))
 print("LAUNCH_COMPLETE",str(out),flush=True)
if __name__=="__main__":main()
