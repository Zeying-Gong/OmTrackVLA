"""Reserve all eight GPUs once; independent model+simulator lane per GPU."""
import concurrent.futures,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path('/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928')
SOURCE=Path.cwd();WLA=ROOT.parent/'WLA-EVT-20260925';BENCH=ROOT.parent/'OmTrackVLA-da3-polar-20260924';ENV=ROOT.parent.parent/'envs'
MANIFEST=WLA/'evt_full_20260926/manifest.json'
OUT=Path(sys.argv[1]);OUT.mkdir(parents=True,exist_ok=False)
visible=os.environ.get('CUDA_VISIBLE_DEVICES','0,1,2,3,4,5,6,7').split(',')
if len(visible)!=8:raise ValueError('full evaluation requires eight allocated GPUs')
children=[]
def stop(p):
    if p.poll() is None:
        try:os.killpg(p.pid,signal.SIGTERM)
        except ProcessLookupError:return
        try:p.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
def spawn(cmd,env,log,cwd):
    with log.open('w') as f:p=subprocess.Popen(cmd,env=env,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    children.append(p);return p
def lane(index):
    dest=OUT/f'shard_{index:02d}';dest.mkdir(exist_ok=False);ready=dest/'server_ready.json'
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=visible[index];env['PYTHONPATH']=str(SOURCE)
    server=spawn([str(ROOT/'probe_env/bin/python'),'-u','-m','wa.wm.eval_server','--root',str(ROOT),
        '--encoder-weight','/data/nas_ray/home/zeying.gong/datasets_processed/threepanel_debug_v2/traversability_stepp_v1/_weights/dinov2_vits14_pretrain.pth',
        '--wla-source',str(ROOT/'dependencies/wla_v1'),'--wla-checkpoint','/data/nas_ray/project/md-ak/users/zeying.gong/job_58346/task_69086/wla_teacher_train/training/checkpoints/step-0043203.pt',
        '--checkpoint','/data/nas_ray/project/md-ak/users/zeying.gong/job_59566/task_70423/wm_jepa_robot_v1/checkpoint.pt','--ready',str(ready)],env,dest/'server.log',SOURCE)
    try:
        deadline=time.monotonic()+1800
        while not ready.exists():
            if server.poll() is not None:raise RuntimeError(f'server{index} failed')
            if time.monotonic()>deadline:raise TimeoutError('server startup')
            time.sleep(3)
        env['PYTHONPATH']=':'.join(map(str,[BENCH/'artifacts/official_runtime',BENCH/'torch_overlay',BENCH,BENCH/'habitat-lab',SOURCE,WLA]))
        env['OMTRACKVLA_XVFB_DISPLAY_NUM']=str(310+index);env['OMTRACKVLA_HAB_SIM_GLX_ROOT']=str(BENCH);env['OMTRACKVLA_XVFB_LOG']=str(dest/'xvfb.log')
        worker=spawn([str(BENCH/'scripts/runtime/run_xvfb.sh'),str(ENV/'habitat/bin/python'),'-u','-m','wa.wm.eval_full','--manifest',str(MANIFEST),'--output',str(dest),'--shard',str(index),'--ready',str(ready)],env,dest/'worker.log',BENCH)
        if worker.wait()!=0:raise RuntimeError(f'shard{index} failed; retain logs')
    finally:stop(server)
def interrupt(*args):raise KeyboardInterrupt()
signal.signal(signal.SIGTERM,interrupt)
pool=concurrent.futures.ThreadPoolExecutor(max_workers=8)
try:
    futures=[pool.submit(lane,i) for i in range(8)]
    for f in concurrent.futures.as_completed(futures):f.result()
    manifest=json.loads(MANIFEST.read_text());rows=[]
    for i in range(8):
        dest=OUT/f'shard_{i:02d}';assert (dest/'COMPLETE.json').exists()
        rows.extend(json.loads(x) for x in (dest/'episodes.jsonl').read_text().splitlines())
    if len(rows)!=4215 or len({(r['task'],r['key']) for r in rows})!=4215:raise ValueError('missing/duplicate episodes')
    results={}
    for task in ['stt','dt','at']:
        part=[r for r in rows if r['task']==task];assert len(part)==1405
        assert {r['key'] for r in part}=={e['key'] for e in manifest['tasks'][task]['episodes']}
        ref=manifest['reference_steps'];den=sum(max(r['total_step'],ref.get(r['key'],0)) for r in part)
        results[task]=dict(episodes=len(part),SR=100*sum(bool(r['success']) for r in part)/len(part),
            TR=100*sum(r['following_step'] for r in part)/den,CR=100*sum(float(r['collision']) for r in part)/len(part),
            macro_TR=100*sum(float(r['following_rate']) for r in part)/len(part),finish=100*sum(bool(r['finish']) for r in part)/len(part),
            invalid_init_count=sum(r.get('policy_init_valid') is False for r in part),reference_missing=sum(r['key'] not in ref for r in part))
    (OUT/'summary.json').write_text(json.dumps(dict(status='COMPLETE_FULL_VALIDATION',mode='image',episodes=4215,metrics_percent=results,limits=['simulation only','first-frame bbox; no text/UWB','no edge latency verification']),indent=2))
    print('FULL_EVALUATION_COMPLETE',json.dumps(results),flush=True)
finally:
    for child in list(children):stop(child)
    pool.shutdown(wait=True,cancel_futures=True)
