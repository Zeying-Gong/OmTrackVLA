"""Full 96-episode paired diagnosis; flexible GPU count, no repeated queue smoke."""
import concurrent.futures,json,os,signal,subprocess,sys,threading,time
from pathlib import Path
import torch
ROOT=Path('/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928')
SOURCE=Path.cwd();WLA=ROOT.parent/'WLA-EVT-20260925';BENCH=ROOT.parent/'OmTrackVLA-da3-polar-20260924';ENV=ROOT.parent.parent/'envs'
OUT=Path(sys.argv[1]);OUT.mkdir(parents=True,exist_ok=False)
MANIFEST=WLA/'evt_full_20260926/manifest.json';PLAN=SOURCE/'wa/wm/mixed_diagnostic_plan_v1.json'
visible=os.environ.get('CUDA_VISIBLE_DEVICES',','.join(str(i) for i in range(torch.cuda.device_count()))).split(',')
if not 1<=len(visible)<=8:raise ValueError('expected 1..8 assigned devices')
children=[];lock=threading.Lock();cancel=threading.Event()
def stop(p):
    if p.poll() is None:
        try:os.killpg(p.pid,signal.SIGTERM)
        except ProcessLookupError:return
        try:p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            p.wait()
def spawn(cmd,env,log,cwd):
    with lock:
        if cancel.is_set():raise RuntimeError('cancelled')
        with log.open('w') as f:p=subprocess.Popen(cmd,env=env,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(p)
    return p
def lane(slot,variant,shard):
    mode,noise=variant.split('_');dest=OUT/f'{variant}_shard{shard}';dest.mkdir();ready=dest/'server_ready.json'
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=visible[slot];env['PYTHONPATH']=str(SOURCE)
    server=spawn([str(ROOT/'probe_env/bin/python'),'-u','-m','wa.wm.eval_server','--root',str(ROOT),
        '--encoder-weight','/data/nas_ray/home/zeying.gong/datasets_processed/threepanel_debug_v2/traversability_stepp_v1/_weights/dinov2_vits14_pretrain.pth',
        '--wla-source',str(ROOT/'dependencies/wla_v1'),'--wla-checkpoint','/data/nas_ray/project/md-ak/users/zeying.gong/job_58346/task_69086/wla_teacher_train/training/checkpoints/step-0043203.pt',
        '--checkpoint',os.environ.get('WA_DIAG_CHECKPOINT','/data/nas_ray/project/md-ak/users/zeying.gong/job_59566/task_70423/wm_jepa_robot_v1/checkpoint.pt'),
        '--ready',str(ready),'--mode',mode,'--noise-mode',noise],env,dest/'server.log',SOURCE)
    worker=None
    try:
        deadline=time.monotonic()+1800
        while not ready.exists():
            if cancel.is_set() or server.poll() is not None:raise RuntimeError('server failed/cancelled')
            if time.monotonic()>deadline:raise TimeoutError('model startup')
            time.sleep(2)
        env['PYTHONPATH']=':'.join(map(str,[BENCH/'artifacts/official_runtime',BENCH/'torch_overlay',BENCH,BENCH/'habitat-lab',SOURCE,WLA]))
        env['OMTRACKVLA_XVFB_DISPLAY_NUM']=str(410+slot);env['OMTRACKVLA_HAB_SIM_GLX_ROOT']=str(BENCH);env['OMTRACKVLA_XVFB_LOG']=str(dest/'xvfb.log')
        worker=spawn([str(BENCH/'scripts/runtime/run_xvfb.sh'),str(ENV/'habitat/bin/python'),'-u','-m','wa.wm.diagnostic_eval',
            '--manifest',str(MANIFEST),'--plan',str(PLAN),'--output',str(dest),'--shard',str(shard),'--ready',str(ready)],env,dest/'worker.log',BENCH)
        if worker.wait()!=0:raise RuntimeError('simulator failed; preserve logs')
    finally:
        if worker:stop(worker)
        stop(server)
jobs=[(v,s) for v in ['image_random','mixed_random','image_zero','mixed_zero'] for s in range(2)]
def slot_run(slot):
    for variant,shard in jobs[slot::len(visible)]:lane(slot,variant,shard)
def interrupt(*_):raise KeyboardInterrupt()
signal.signal(signal.SIGTERM,interrupt)
pool=concurrent.futures.ThreadPoolExecutor(max_workers=len(visible))
try:
    fs=[pool.submit(slot_run,i) for i in range(len(visible))]
    for f in concurrent.futures.as_completed(fs):f.result()
    results={};initial={}
    for variant in ['image_random','mixed_random','image_zero','mixed_zero']:
        rows=[]
        for s in range(2):
            dest=OUT/f'{variant}_shard{s}';assert (dest/'COMPLETE.json').exists()
            rows.extend(json.loads(line) for line in (dest/'episodes.jsonl').read_text().splitlines())
        assert len(rows)==24 and len({(r['task'],r['key']) for r in rows})==24
        results[variant]={}
        for r in rows:initial.setdefault((r['task'],r['key']),set()).add(r['initial_rgb_sha256'])
        for task in ['stt','dt','at']:
            part=[r for r in rows if r['task']==task];assert len(part)==8
            results[variant][task]=dict(n=8,SR=100*sum(r['success'] for r in part)/8,CR=100*sum(r['collision'] for r in part)/8,
                macro_TR=100*sum(r['following_rate'] for r in part)/8,invalid=sum(r.get('policy_init_valid') is False for r in part))
    if not all(len(h)==1 for h in initial.values()):raise ValueError('paired initial RGB mismatch')
    (OUT/'summary.json').write_text(json.dumps(dict(status='COMPLETE_PAIRED_DEVELOPMENT_DIAGNOSTIC',episodes=96,results=results,
        initial_rgb_pairs_equal=True,uwb='ideal_simulated_uwb',real_uwb_verified=False),indent=2))
    print('DIAGNOSTIC_COMPLETE',json.dumps(results),flush=True)
finally:
    with lock:cancel.set();snapshot=list(children)
    for p in snapshot:stop(p)
    pool.shutdown(wait=True,cancel_futures=True)
