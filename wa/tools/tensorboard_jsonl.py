"""Read-only conversion of existing JSONL metrics into a live TensorBoard run."""
import argparse,json,time
from collections import deque
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True)
a=p.parse_args();run=Path(a.run);out=Path(a.output)
if out.exists():raise SystemExit('use a new event directory; do not duplicate prior scalars')
writer=SummaryWriter(str(out));seen=set();window=deque(maxlen=100)
writer.add_text('readme','Training losses are rank-0 LAST microbatch samples every25 optimizer steps, NOT DDP-averaged train loss or validation loss. Rolling100 averages100 logged samples (~2500steps). Validation ADE/FDE/yaw is emitted only when metrics.json exists. No closed-loop SR/collision results are available. Gradient norm is before clipping.')
keys=['loss','flow','geometry','world']
while True:
    for line in (run/'train.jsonl').read_text().splitlines():
        try:r=json.loads(line)
        except json.JSONDecodeError:continue
        step=r['step']
        if step in seen:continue
        seen.add(step);window.append(r)
        for key in keys:
            writer.add_scalar('train_rank0_last_microbatch/'+key,r[key],step)
            writer.add_scalar('train_rolling100_logged_samples/'+key,sum(x[key] for x in window)/len(window),step)
        for key in ['grad_norm','peak_allocated_gib','elapsed_s']:
            writer.add_scalar('diagnostics/'+key,r[key],step)
    path=run/'metrics.json'
    if path.exists():
        result=json.loads(path.read_text())
        for mode,metrics in result['metrics'].items():
            for key,value in metrics.items():
                if isinstance(value,(int,float)) and value is not None:
                    writer.add_scalar('offline_validation/'+mode+'/'+key,value,result['steps'])
        writer.add_text('offline_validation/status',json.dumps(result,indent=2),result['steps'])
        writer.flush();writer.close();print('FINAL_METRICS_EXPORTED',flush=True);break
    writer.flush();print('TRAIN_EVENTS',len(seen),'LAST_STEP',max(seen,default=0),'WAITING_FOR_VALIDATION',flush=True)
    time.sleep(30)
