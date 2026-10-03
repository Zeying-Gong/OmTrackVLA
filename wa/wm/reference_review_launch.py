"""Two full twelve-episode LightNav lanes; originals read-only."""
import os,sys,time,subprocess,signal,concurrent.futures,json
from pathlib import Path
ROOT=Path('/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928');SOURCE=Path.cwd();WLA=ROOT.parent/'WLA-EVT-20260925';BENCH=ROOT.parent/'OmTrackVLA-da3-polar-20260924';ENV=ROOT.parent.parent/'envs';LIGHT=ROOT.parent/'LightNav-0'
SPLIT=os.environ.get('WA_DIAG_SPLIT','development')
if SPLIT not in ('development','confirmation'):raise ValueError('WA_DIAG_SPLIT must be development or confirmation')
out=Path(sys.argv[1]);out.mkdir(exist_ok=False);devices=os.environ.get('CUDA_VISIBLE_DEVICES','0,1').split(',');assert len(devices)==2
def lane(shard):
 dest=out/f'shard_{shard}';dest.mkdir();ready=dest/'server.ready';port=18880+shard;children=[]
 def spawn(cmd,env,log):
  with (dest/log).open('w') as f:p=subprocess.Popen(cmd,cwd=BENCH,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  children.append(p);return p
 try:
  env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=devices[shard];env['PYTHONPATH']=str(LIGHT/'src');env['PYTHONSAFEPATH']='1'
  env['LD_LIBRARY_PATH']=':'.join(map(str,(ENV/'lightnav/lib').glob('python*/site-packages/nvidia/*/lib')))+':'+env.get('LD_LIBRARY_PATH','')
  server=spawn([str(ENV/'lightnav/bin/python'),'-P',str(WLA/'lightnav_transport_20260927/server_transport.py'),'--task','tracking','--model_path',str(LIGHT/'checkpoints/LightNav-0'),'--backend','vllm_local','--gpu_memory_utilization','0.60','--max_batch_size','1','--host','127.0.0.1','--port',str(port),'--ready_file',str(ready)],env,'server.log')
  deadline=time.monotonic()+1800
  while not ready.exists():
   if server.poll() is not None:raise RuntimeError('server exited')
   if time.monotonic()>deadline:raise TimeoutError('server startup')
   time.sleep(2)
  env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=devices[shard];env['PYTHONPATH']=':'.join(map(str,[BENCH/'artifacts/lightnav_client_runtime',BENCH/'artifacts/official_runtime',BENCH/'torch_overlay',BENCH,BENCH/'habitat-lab',SOURCE,WLA]));env['OMTRACKVLA_XVFB_DISPLAY_NUM']=str(460+shard);env['OMTRACKVLA_HAB_SIM_GLX_ROOT']=str(BENCH);env['OMTRACKVLA_XVFB_LOG']=str(dest/'xvfb.log')
  worker=spawn([str(BENCH/'scripts/runtime/run_xvfb.sh'),str(ENV/'habitat/bin/python'),'-u','-m','wa.wm.reference_review_eval','--output',str(dest),'--server',f'ws://127.0.0.1:{port}','--shard',str(shard),'--plan',str(SOURCE/'wa/wm/mixed_diagnostic_plan_v1.json'),'--split',SPLIT],env,'worker.log')
  while worker.poll() is None:
   if server.poll() is not None:raise RuntimeError('server died')
   time.sleep(2)
  if worker.returncode:raise RuntimeError('worker failed')
 finally:
  for p in reversed(children):
   if p.poll() is None:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=15)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lane,range(2)))
rows=[]
for d in sorted(out.glob('shard_*')):
 assert (d/'COMPLETE.json').exists()
 assert json.loads((d/'COMPLETE.json').read_text())['split']==SPLIT,'worker split mismatch'
 rows.extend(json.loads(l) for l in (d/'episodes.jsonl').read_text().splitlines())
assert len(rows)==24 and len({(r['task'],r['key']) for r in rows})==24
(out/'COMPLETE.json').write_text(json.dumps({'episodes':24,'split':SPLIT,'successes':sum(r['success'] for r in rows)}))
