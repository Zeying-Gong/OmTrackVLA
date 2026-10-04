"""Full4215 mixed evaluation; multiplex eight fixed shards over the allocated GPUs."""
import concurrent.futures,json,os,signal,subprocess,sys,time,threading
from pathlib import Path
from wa.wm.full_mixed_contract import validate_ready,summarize,MANIFEST_SHA,allocated_devices
from wa.wm.loaders import sha
ROOT=Path('/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928')
SOURCE=Path.cwd();WLA=ROOT.parent/'WLA-EVT-20260925';BENCH=ROOT.parent/'OmTrackVLA-da3-polar-20260924';ENV=ROOT.parent.parent/'envs'
MANIFEST=WLA/'evt_full_20260926/manifest.json'
assert sha(MANIFEST)==MANIFEST_SHA
manifest=json.loads(MANIFEST.read_text());plan=None
semantic_fix=os.environ.get('WA_SEMANTIC_PLY_FIX','')
if semantic_fix:
    from wa.wm.semantic_scene import VERSION,provenance
    if semantic_fix!=VERSION:raise ValueError('unknown semantic repair version')
    if os.environ.get('WA_RESUME_PLAN'):raise ValueError('fresh semantic protocol requires all episodes')
if os.environ.get('WA_RESUME_PLAN'):
    from wa.wm.full_mixed_resume import load_plan,verify_new_rows
    plan=load_plan(os.environ['WA_RESUME_PLAN'],manifest,os.environ['WA_RESUME_PLAN_SHA'])
targeted=bool(os.environ.get('WA_TARGETED_PLAN'))
if targeted:
    if not semantic_fix or plan is not None:raise ValueError('targeted plan requires semantic-only protocol')
    from wa.wm.semantic_targeted import load_targeted
    from wa.wm.full_mixed_resume import verify_new_rows
    plan=load_targeted(os.environ['WA_TARGETED_PLAN'],manifest,os.environ['WA_TARGETED_PLAN_SHA'])
assert os.environ.get('WA_DIAG_CONTROLLER')=='learned_yaw_guard_v1'
OUT=Path(sys.argv[1]);OUT.mkdir(parents=True,exist_ok=False)
if plan is not None:
    (OUT/'resume_plan.json').write_text(json.dumps(plan,indent=2,allow_nan=False))
import torch
visible=allocated_devices(os.environ.get('CUDA_VISIBLE_DEVICES'),torch.cuda.device_count())
print('GPU_ALLOCATION',json.dumps(dict(devices=visible,names=[torch.cuda.get_device_name(i) for i in range(len(visible))])),flush=True)
children=[];lock=threading.Lock();cancel=threading.Event()
def stop(p):
    if p.poll() is None:
        try:os.killpg(p.pid,signal.SIGTERM)
        except ProcessLookupError:return
        try:p.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
def spawn(cmd,env,log,cwd):
    with lock:
        if cancel.is_set():raise RuntimeError('evaluation cancelled')
        with log.open('w') as f:p=subprocess.Popen(cmd,env=env,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(p)
    return p
def lane(index,slot):
    dest=OUT/f'shard_{index:02d}';dest.mkdir(exist_ok=False);ready=dest/'server_ready.json'
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=visible[slot];env['PYTHONPATH']=str(SOURCE)
    server=spawn([str(ROOT/'probe_env/bin/python'),'-u','-m','wa.wm.eval_server','--root',str(ROOT),
        '--encoder-weight','/data/nas_ray/home/zeying.gong/datasets_processed/threepanel_debug_v2/traversability_stepp_v1/_weights/dinov2_vits14_pretrain.pth',
        '--wla-source',str(ROOT/'dependencies/wla_v1'),'--wla-checkpoint','/data/nas_ray/project/md-ak/users/zeying.gong/job_58346/task_69086/wla_teacher_train/training/checkpoints/step-0043203.pt',
        '--checkpoint',os.environ['WA_DIAG_CHECKPOINT'],'--mode','mixed','--noise-mode','zero','--ready',str(ready)],env,dest/'server.log',SOURCE)
    try:
        deadline=time.monotonic()+1800
        while not ready.exists():
            if cancel.is_set() or server.poll() is not None:raise RuntimeError(f'server{index} failed')
            if time.monotonic()>deadline:raise TimeoutError('server startup')
            time.sleep(3)
        validate_ready(json.loads(ready.read_text()))
        env['PYTHONPATH']=':'.join(map(str,[BENCH/'artifacts/official_runtime',BENCH/'torch_overlay',BENCH,BENCH/'habitat-lab',SOURCE,WLA]))
        env['OMTRACKVLA_XVFB_DISPLAY_NUM']=str(510+slot);env['OMTRACKVLA_HAB_SIM_GLX_ROOT']=str(BENCH);env['OMTRACKVLA_XVFB_LOG']=str(dest/'xvfb.log')
        worker=spawn([str(BENCH/'scripts/runtime/run_xvfb.sh'),str(ENV/'habitat/bin/python'),'-u','-m','wa.wm.eval_full_mixed','--manifest',str(MANIFEST),'--output',str(dest),'--shard',str(index),'--ready',str(ready)],env,dest/'worker.log',BENCH)
        if worker.wait()!=0:raise RuntimeError(f'shard{index} failed; retain logs')
    finally:stop(server)
def slot_run(slot):
    for index in range(slot,8,len(visible)):lane(index,slot)
def interrupt(*args):raise KeyboardInterrupt()
signal.signal(signal.SIGTERM,interrupt)
pool=concurrent.futures.ThreadPoolExecutor(max_workers=len(visible))
try:
    futures=[pool.submit(slot_run,i) for i in range(len(visible))]
    for f in concurrent.futures.as_completed(futures):f.result()
    manifest=json.loads(MANIFEST.read_text());rows=[]
    for i in range(8):
        dest=OUT/f'shard_{i:02d}';assert (dest/'COMPLETE.json').exists()
        part=[json.loads(x) for x in (dest/'episodes.jsonl').read_text().splitlines()]
        if plan is not None:
            assert {(r['task'],r['key']) for r in part}=={(e['task'],e['key']) for e in plan['lanes'][i]}
            assert len(part)==len(plan['lanes'][i])
            assert json.loads((dest/'COMPLETE.json').read_text())['episodes']==len(part)
            for r in part:r['artifact_root']=str(dest)
        if semantic_fix:
            for r in part:r['artifact_root']=str(dest)
        rows.extend(part)
    if plan is not None:
        verify_new_rows(rows,plan)
        if targeted:load_targeted(os.environ['WA_TARGETED_PLAN'],manifest,os.environ['WA_TARGETED_PLAN_SHA'])
        else:load_plan(os.environ['WA_RESUME_PLAN'],manifest,os.environ['WA_RESUME_PLAN_SHA'])
        rows=plan['completed_rows']+rows
    report=summarize(rows,manifest)
    if semantic_fix:
        report['semantic_protocol']=provenance()
        (OUT/'combined_episodes.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    if plan is not None:
        report['continuation']=dict(parent_job=plan['parent_job'],parent_task=plan['parent_task'],reused=len(plan['completed_rows']),new=len(rows)-len(plan['completed_rows']),plan_sha256=os.environ['WA_TARGETED_PLAN_SHA'] if targeted else os.environ['WA_RESUME_PLAN_SHA'])
        if targeted:report['continuation'].update(scope='MP3D re-evaluation plus audited unaffected HM3D reuse',reuse_counts=plan['reuse_counts'])
        (OUT/'combined_episodes.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    (OUT/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('FULL_MIXED_EVALUATION_COMPLETE',json.dumps(report),flush=True)
finally:
    with lock:cancel.set();snapshot=list(children)
    for child in snapshot:stop(child)
    pool.shutdown(wait=True,cancel_futures=True)
