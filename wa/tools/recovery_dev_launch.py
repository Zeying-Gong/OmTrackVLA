"""Bounded developer integration check, not a formal collection job."""
import os,sys,time,subprocess,signal
from pathlib import Path
R=Path('/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928');S=R/'checkout';B=R.parent/'OmTrackVLA-da3-polar-20260924';W=R.parent/'WLA-EVT-20260925';E=R.parent.parent/'envs';L=R.parent/'LightNav-0'
assert not os.environ.get('MD_AK_JOB_ID')
out=Path(sys.argv[1]);out.mkdir(exist_ok=False);children=[]
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='2',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
xb=Path('/tmp/omtrackvla_xvfb_root_0');assert (xb/'usr/bin/Xvfb').is_file()
env['PATH']=str(xb/'usr/bin')+':'+env['PATH']
env['XKB_CONFIG_ROOT']=str(xb/'usr/share/X11/xkb')
env['OMTRACKVLA_GLX_LIB_DIRS']='/usr/lib/x86_64-linux-gnu:/usr/lib64:/usr/local/nvidia/lib64:'+str(xb/'usr/lib/x86_64-linux-gnu')+':'+str(xb/'lib/x86_64-linux-gnu')
def spawn(cmd,en,log,cwd):
 with (out/log).open('w') as f:p=subprocess.Popen(cmd,env=en,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 children.append(p);return p
try:
 en=dict(env,PYTHONPATH=str(S));ready=out/'wa_ready.json'
 wa=spawn([str(R/'probe_env/bin/python'),'-u','-m','wa.wm.eval_server','--root',str(R),'--encoder-weight','/data/nas_ray/home/zeying.gong/datasets_processed/threepanel_debug_v2/traversability_stepp_v1/_weights/dinov2_vits14_pretrain.pth','--wla-source',str(R/'dependencies/wla_v1'),'--wla-checkpoint','/data/nas_ray/project/md-ak/users/zeying.gong/job_58346/task_69086/wla_teacher_train/training/checkpoints/step-0043203.pt','--checkpoint','/data/nas_ray/project/md-ak/users/zeying.gong/job_59866/task_70728/wa_history_repeat_v1/checkpoint.pt','--ready',str(ready),'--mode','mixed','--noise-mode','zero'],en,'wa.log',S)
 en=dict(env,PYTHONPATH=str(L/'src'),PYTHONSAFEPATH='1')
 en['LD_LIBRARY_PATH']=':'.join(map(str,(E/'lightnav/lib').glob('python*/site-packages/nvidia/*/lib')))+':'+env.get('LD_LIBRARY_PATH','')
 tr=out/'teacher.ready'
 teacher=spawn([str(E/'lightnav/bin/python'),'-P',str(W/'lightnav_transport_20260927/server_transport.py'),'--task','tracking','--model_path',str(L/'checkpoints/LightNav-0'),'--backend','vllm_local','--gpu_memory_utilization','0.45','--max_batch_size','1','--host','127.0.0.1','--port','18989','--ready_file',str(tr)],en,'teacher.log',B)
 deadline=time.monotonic()+600
 while not(ready.exists() and tr.exists()):
  if wa.poll() is not None or teacher.poll() is not None:raise RuntimeError('server exited')
  if time.monotonic()>deadline:raise TimeoutError('startup')
  time.sleep(2)
 en=dict(env,WA_DEVELOPMENT='1',WA_DIAG_CONTROLLER='learned_target_guard_v3',PYTHONPATH=':'.join(map(str,[B/'artifacts/lightnav_client_runtime',B/'artifacts/official_runtime',B/'torch_overlay',B,B/'habitat-lab',S,W])),OMTRACKVLA_XVFB_DISPLAY_NUM='489',OMTRACKVLA_HAB_SIM_GLX_ROOT=str(B),OMTRACKVLA_XVFB_LOG=str(out/'xvfb.log'))
 worker=spawn([str(B/'scripts/runtime/run_xvfb.sh'),str(E/'habitat/bin/python'),'-u','-m','wa.wm.recovery_collect','--manifest',str(S/'wa/wm/recovery_manifest_v1.json'),'--ready',str(ready),'--teacher-url','ws://127.0.0.1:18989','--output',str(out/'collection'),'--shard','0','--shards','8','--development-one'],en,'worker.log',B)
 rc=worker.wait(timeout=600)
 if rc:raise RuntimeError(f'worker exit {rc}')
 print('DEVELOPER_RUN_COMPLETED',flush=True)
finally:
 for p in reversed(children):
  if p.poll() is None:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
