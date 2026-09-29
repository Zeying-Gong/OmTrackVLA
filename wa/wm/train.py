"""Full epoch DDP world-action training; bounded diagnostics only on developer GPU."""
import argparse,contextlib,json,math,os,random,subprocess,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler,Subset
from wa.data import audit,sha
from wa.wm.robot_data import RobotWorldData,CONTRACT
from wa.wm.training import JointRobotModel,optimizer_groups

def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','encoder-weight','wla-source','wla-checkpoint','cache','index-root','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--kind',choices=['jepa','dino'],required=True)
    p.add_argument('--batch-size',type=int,default=2);p.add_argument('--accumulation',type=int,default=2)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--world-weight',type=float,default=.1)
    p.add_argument('--diagnostic',action='store_true');p.add_argument('--lane',choices=['managed','external-h100'],default='managed')
    p.add_argument('--data-root');p.add_argument('--source-prefix')
    return p.parse_args()

def moved(batch,device):return {k:v.to(device,non_blocking=True) for k,v in batch.items()}

def evaluate(model,data,batch_size,device,rank,world):
    loader=DataLoader(Subset(data,range(rank,len(data),world)),batch_size=batch_size,num_workers=0)
    results={};model.eval()
    for name,mode in [('image',0),('point',1),('mixed',2)]:
        sums=torch.zeros(4,device=device,dtype=torch.float64)
        for batch in loader:
            batch=moved(batch,device);n=len(batch['pose'])
            # Fixed zero integration start gives deterministic offline comparisons.
            pred,_=model.predict(batch,torch.full((n,),mode,device=device),torch.zeros_like(batch['pose']))
            err=torch.linalg.vector_norm(pred[...,:2]-batch['pose'][...,:2],dim=-1)
            yaw=torch.atan2(pred[...,2],pred[...,3])-torch.atan2(batch['pose'][...,2],batch['pose'][...,3])
            wrapped=torch.atan2(yaw.sin(),yaw.cos()).abs()
            sums+=torch.stack([err.mean(1).sum(),err[:,-1].sum(),wrapped.mean(1).sum(),err.new_tensor(n)]).double()
        if world>1:dist.all_reduce(sums)
        results[name]=dict(ADE_m=float(sums[0]/sums[3]),FDE_m=float(sums[1]/sums[3]),yaw_MAE_rad=float(sums[2]/sums[3]),windows=int(sums[3]),SR=None,collision_rate=None)
    return results

def main():
    a=arguments()
    if min(a.batch_size,a.accumulation,a.epochs)<1 or not math.isfinite(a.world_weight) or a.world_weight<=0:raise ValueError('invalid training configuration')
    if a.diagnostic and os.environ.get('MD_AK_JOB_ID'):raise ValueError('developer diagnostic forbidden in formal cluster job')
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    if not a.diagnostic and world!=8:raise ValueError('full recipe requires8 ranks')
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    if a.lane=='external-h100' and (world!=8 or not all('H100' in torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))):raise ValueError('external lane requires8H100')
    if world>1:dist.init_process_group('nccl')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(2)
    os.environ['MD_WLA_ACTION_ACTIVATION_CHECKPOINTING']='1'
    output=Path(a.output)
    if rank==0:output.mkdir(parents=True,exist_ok=False)
    if world>1:dist.barrier()
    # Verify original cache identity and the independently audited row selection.
    cache_sha=audit(a.cache)
    selection=json.loads((Path(a.index_root)/'audit.json').read_text())
    if selection['contract']!=CONTRACT or selection['cache_complete_sha256']!=cache_sha:raise ValueError('audit/cache mismatch')
    for part in ('train','heldout'):
        if sha(Path(a.index_root)/f'{part}_valid.npy')!=selection['splits'][part]['index_sha256']:raise ValueError('index hash mismatch')
    limit=world*a.batch_size*a.accumulation*2 if a.diagnostic else None
    train=RobotWorldData(a.cache,'train',a.index_root,limit,a.data_root,a.source_prefix)
    val=RobotWorldData(a.cache,'heldout',a.index_root,world*a.batch_size if a.diagnostic else None,a.data_root,a.source_prefix)
    sampler=DistributedSampler(train,num_replicas=world,rank=rank,shuffle=True,seed=a.seed,drop_last=True)
    loader=DataLoader(train,batch_size=a.batch_size,sampler=sampler,drop_last=True,num_workers=a.workers,pin_memory=True)
    model=JointRobotModel(a.root,a.encoder_weight,a.wla_source,a.wla_checkpoint,a.kind).to(device)
    wrapped=DDP(model,device_ids=[local],find_unused_parameters=True,broadcast_buffers=False) if world>1 else model
    groups=optimizer_groups(model);base_lrs=[g['lr'] for g in groups]
    optimizer=torch.optim.AdamW(groups,weight_decay=.01)
    if rank==0:
        config=vars(a)|dict(contract=CONTRACT,cache_sha256=cache_sha,index_audit_sha256=sha(Path(a.index_root)/'audit.json'),effective_batch=world*a.batch_size*a.accumulation,train_rows=len(train),heldout_rows=len(val),world_size=world,precision='fp32 training / bf16 frozen encoder',base_lrs=base_lrs)
        root=Path(__file__).resolve().parents[2]
        env=dict(commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),dirty=subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True).strip(),torch=torch.__version__,gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],source_sha256={str(f.relative_to(root)):sha(f) for f in sorted((root/'wa').rglob('*.py'))},wla=model.provenance)
        (output/'config.json').write_text(json.dumps(config,indent=2));(output/'environment.json').write_text(json.dumps(env,indent=2))
    # Random streams diverge after identical model initialization/DDP synchronization.
    torch.manual_seed(a.seed+rank)
    start=time.monotonic();step=0;total_steps=math.ceil(len(loader)/a.accumulation)*a.epochs
    for epoch in range(a.epochs):
        sampler.set_epoch(epoch);wrapped.train();optimizer.zero_grad(set_to_none=True)
        for i,batch in enumerate(loader):
            batch=moved(batch,device);modes=torch.randint(0,3,(len(batch['pose']),),device=device)
            group_start=i//a.accumulation*a.accumulation
            group_count=min(a.accumulation,len(loader)-group_start)
            sync=(i+1)%a.accumulation==0 or i+1==len(loader)
            scope=wrapped.no_sync() if world>1 and not sync else contextlib.nullcontext()
            with scope:
                values=wrapped(batch,modes,a.world_weight)
                if not torch.isfinite(values['loss']):raise FloatingPointError('nonfinite loss')
                (values['loss']/group_count).backward()
            if sync:
                multiplier=min(1.,(step+1)/100)*(.1+.9*.5*(1+math.cos(math.pi*step/max(total_steps,1))))
                for g,lr in zip(optimizer.param_groups,base_lrs):g['lr']=lr*multiplier
                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
                optimizer.step();optimizer.zero_grad(set_to_none=True);step+=1
                if rank==0 and (step==1 or step%25==0 or a.diagnostic):
                    row=dict(step=step,total_steps=total_steps,loss=float(values['loss'].detach()),flow=float(values['flow']),geometry=float(values['geometry']),world=float(values['world']),grad_norm=float(norm),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,elapsed_s=time.monotonic()-start)
                    with (output/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    print(json.dumps(row),flush=True)
                if rank==0 and not a.diagnostic and step%2000==0:
                    torch.save({'model':{k:v for k,v in model.state_dict().items() if not k.startswith('encoder.')},'optimizer':optimizer.state_dict(),'step':step,'kind':a.kind,'contract':CONTRACT},output/f'step-{step:07d}.pt')
    if rank==0 and not a.diagnostic:
        torch.save({'model':{k:v for k,v in model.state_dict().items() if not k.startswith('encoder.')},'optimizer':optimizer.state_dict(),'step':step,'kind':a.kind,'contract':CONTRACT},output/'checkpoint.pt')
    if world>1:dist.barrier()
    metrics=evaluate(model,val,a.batch_size,device,rank,world)
    if rank==0:
        (output/'metrics.json').write_text(json.dumps(dict(status='DIAGNOSTIC_PASS' if a.diagnostic else 'OFFLINE_ONLY',kind=a.kind,steps=step,metrics=metrics,closed_loop=False,edge_latency_verified=False,point_source='simulated_uwb',elapsed_s=time.monotonic()-start),indent=2))
        print('COMPLETE',json.dumps(metrics),flush=True)
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
