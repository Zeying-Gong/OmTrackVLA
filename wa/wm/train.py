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
from wa.wm.failure_state_train_args import add_arguments as failure_arguments,validate_training_recipe as failure_recipe,source_pins_from_args

def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','encoder-weight','wla-source','wla-checkpoint','cache','index-root','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--kind',choices=['jepa','dino'],required=True)
    p.add_argument('--batch-size',type=int,default=2);p.add_argument('--accumulation',type=int,default=2)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--world-weight',type=float,default=.1)
    p.add_argument('--diagnostic',action='store_true');p.add_argument('--lane',choices=['managed','external-h100'],default='managed')
    p.add_argument('--data-root');p.add_argument('--source-prefix')
    p.add_argument('--resume');p.add_argument('--resume-sha256')
    p.add_argument('--completed-epochs',type=int,default=0)
    p.add_argument('--history-repeat-probability',type=float,default=0.)
    p.add_argument('--recovery-cache')
    p.add_argument('--recovery-index')
    p.add_argument('--recovery-repeats',type=int,default=16)
    p.add_argument('--dual-teacher-cache')
    p.add_argument('--teacher-window-plan')
    p.add_argument('--teacher-plan-report-sha256')
    p.add_argument('--dual-teacher-repeats',type=int)
    p.add_argument('--evaluation-set-adaptation',action='store_true')
    failure_arguments(p)
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
    failure_enabled=failure_recipe(a)
    if min(a.batch_size,a.accumulation,a.epochs)<1 or not math.isfinite(a.world_weight) or a.world_weight<=0:raise ValueError('invalid training configuration')
    if not math.isfinite(a.history_repeat_probability) or not 0<=a.history_repeat_probability<=1:raise ValueError('invalid history repeat probability')
    if a.diagnostic and os.environ.get('MD_AK_JOB_ID'):raise ValueError('developer diagnostic forbidden in formal cluster job')
    if bool(a.resume)!=bool(a.completed_epochs) or a.completed_epochs<0 or a.completed_epochs+a.epochs>2:raise ValueError('explicit epoch provenance and maximum2 epochs required')
    if a.resume and (not a.resume_sha256 or sha(a.resume)!=a.resume_sha256):raise ValueError('resume hash mismatch')
    if bool(a.dual_teacher_cache)!=a.evaluation_set_adaptation:raise ValueError('explicit evaluation-set adaptation flag/cache pair required')
    if a.dual_teacher_cache:
        if a.recovery_cache or not a.resume:raise ValueError('independent teacher stage requires parent and excludes legacy recovery mix')
        if a.dual_teacher_repeats is None or not 1<=a.dual_teacher_repeats<=32:raise ValueError('explicit bounded teacher exposure required')
    elif a.dual_teacher_repeats is not None:raise ValueError('teacher repeats without cache')
    if bool(a.teacher_window_plan)!=bool(a.teacher_plan_report_sha256):raise ValueError('candidate path/hash pair required')
    if a.teacher_window_plan and (not a.dual_teacher_cache or a.dual_teacher_repeats!=1 or
        a.completed_epochs!=1 or a.epochs!=1 or a.seed!=42 or a.batch_size!=2 or a.accumulation!=2):
        raise ValueError('hard-STT candidate fixes base/teacher once, one independent continuation, seed42/batch2/accum2')
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    if not a.diagnostic and world!=8:raise ValueError('full recipe requires8 ranks')
    failure_recipe(a,world=world)
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    if a.lane=='external-h100' and (world!=8 or not all('H100' in torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))):raise ValueError('external lane requires8H100')
    # Ray TorchTrainer initializes NCCL before invoking this entry point.
    # Standalone torchrun still needs initialization here.
    if world>1 and not dist.is_initialized():dist.init_process_group('nccl')
    if world>1 and (dist.get_world_size()!=world or dist.get_rank()!=rank):
        raise ValueError('launcher/process-group rank mismatch')
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
    train=RobotWorldData(a.cache,'train',a.index_root,None if a.teacher_window_plan else limit,a.data_root,a.source_prefix)
    val=RobotWorldData(a.cache,'heldout',a.index_root,world*a.batch_size if a.diagnostic else None,a.data_root,a.source_prefix)
    recovery_exposure=None
    if bool(a.recovery_cache)!=bool(a.recovery_index):raise ValueError('recovery cache/index pair required')
    planned_mix=plan_records=plan_windows=None
    failure_mix=None
    runtime_exposure=None
    if a.recovery_cache:
        if not a.resume:raise ValueError('recovery stage requires explicit parent checkpoint')
        from wa.wm.recovery_mix import RecoveryMix
        recovery_sha=audit(a.recovery_cache)
        recovery_selection=json.loads((Path(a.recovery_index)/'audit.json').read_text())
        if recovery_selection['contract']!=CONTRACT or recovery_selection['cache_complete_sha256']!=recovery_sha:raise ValueError('recovery audit mismatch')
        for part in ('train','heldout'):
            if sha(Path(a.recovery_index)/f'{part}_valid.npy')!=recovery_selection['splits'][part]['index_sha256']:raise ValueError('recovery index mismatch')
        for suffix in ('pose.npy','history.npy','episode.npy','episodes.json'):
            if sha(Path(a.cache)/f'heldout_{suffix}')!=sha(Path(a.recovery_cache)/f'heldout_{suffix}'):raise ValueError('heldout changed')
        recovery=RobotWorldData(a.recovery_cache,'train',a.recovery_index,limit)
        train=RecoveryMix(train,recovery,a.recovery_repeats)
        recovery_exposure=train.exposure|dict(cache_sha256=recovery_sha,index_sha256=sha(Path(a.recovery_index)/'audit.json'))
        if rank==0:(output/'recovery_exposure.json').write_text(json.dumps(recovery_exposure,indent=2))
        # Parent model AND optimizer are resumed below; no silent reset or new objective.
    if a.dual_teacher_cache:
        from wa.wm.dual_teacher_data import DualTeacherData
        from wa.wm.recovery_mix import RecoveryMix
        from wa.wm.dual_teacher_selection import EXPERIMENT
        teachers=DualTeacherData(a.dual_teacher_cache,development=a.diagnostic)
        if a.diagnostic and not a.teacher_window_plan:
            teachers=Subset(teachers,np.linspace(0,len(teachers)-1,min(limit,len(teachers)),dtype=int).tolist())
        for suffix in ('pose.npy','history.npy','episode.npy','episodes.json'):
            if sha(Path(a.cache)/f'heldout_{suffix}')!=sha(Path(a.dual_teacher_cache)/f'heldout_{suffix}'):raise ValueError('original heldout files changed')
        if a.teacher_window_plan:
            from wa.wm.teacher_plan_runtime import load_candidate,diagnostic_positions
            from wa.wm.teacher_window_plan import ExposureTaggedData,strip_exposure_tag
            from wa.wm.runtime_exposure import RuntimeExposure
            candidate_root=Path(a.root)/'artifacts'
            planned_mix,plan_records,plan_windows,plan_report=load_candidate(
                a.teacher_window_plan,a.teacher_plan_report_sha256,train,teachers,
                base_index=a.index_root,student_rows=candidate_root/'student61377_full_audit_20261007_v1/combined_episodes.jsonl',
                teacher_selections=candidate_root/'dual_teacher_complete_audit_20261006_v1/combined_selections.jsonl',
                project_root=a.root)
            if plan_report.get('candidate_kind') == 'stt_anchor_v1':
                from wa.wm.teacher_plan_runtime_v2 import validate_training_recipe
                validate_training_recipe(plan_report,a,world=world)
            mixed=planned_mix
            train=ExposureTaggedData(mixed)
            if a.diagnostic and not failure_enabled:
                train=Subset(train,diagnostic_positions(mixed,plan_records,plan_windows,limit*2))
        else:
            mixed=RecoveryMix(train,teachers,a.dual_teacher_repeats)
            train=mixed
        teacher_exposures=len(mixed)-len(mixed.base)
        teacher_exposure=dict(experiment=EXPERIMENT,base_windows=len(mixed.base),teacher_unique_windows=len(teachers),
            teacher_repeats=a.dual_teacher_repeats,teacher_exposures=teacher_exposures,
            hard_stt_plan_sha256=planned_mix.plan_sha256 if planned_mix is not None else None,
            total_windows=len(mixed),teacher_fraction=teacher_exposures/len(mixed),
            cache_sha256=sha(Path(a.dual_teacher_cache)/'complete.json'),
            evaluation_interpretation='in-set adaptation; unchanged heldout files do NOT establish unseen generalization',
            note='Pre-DDP exposure; sampler and batch drop_last discard final rows; audit actual rank exposure before submission.')
        if planned_mix is not None and plan_report.get('candidate_kind') == 'stt_anchor_v1':
            teacher_exposure.update(teacher_plan_schema=planned_mix.schema,
                teacher_candidate_kind=plan_report['candidate_kind'],
                selection_source_hashes=plan_report['selection_source_hashes'])
        if rank==0:(output/'dual_teacher_exposure.json').write_text(json.dumps(teacher_exposure,indent=2))
    if failure_enabled:
        from wa.wm.failure_state_train_gate import verify_loader_audit
        from wa.wm.failure_state_data import FailureStateData
        from wa.wm.failure_state_sampling_runtime import load_candidate as load_failure_candidate,diagnostic_positions as failure_diagnostic_positions,RuntimeExposure as FailureRuntimeExposure
        # Independently checked real tensors and routing, then actual loader admission.
        # The three-source runtime independently rebuilds all 6864 row identities and
        # retains exact best61609 consumed counts; no metadata enters the policy.
        loader_gate=verify_loader_audit(a.failure_state_loader_audit,
            a.failure_state_loader_audit_sha256,a.failure_state_cache,a.failure_state_admission_sha256)
        failure_data=FailureStateData(a.failure_state_cache,expected_admission_sha256=a.failure_state_admission_sha256)
        failure_mix,failure_provenance,failure_simulation=load_failure_candidate(
            a.failure_state_plan,a.failure_state_plan_admission_sha256,planned_mix,failure_data,
            old_plan_root=a.teacher_window_plan,old_run_root=a.failure_state_old_run,
            dedup_path=a.failure_state_dedup_report,source_pins=source_pins_from_args(a))
        train=ExposureTaggedData(failure_mix)
        if a.diagnostic:
            train=Subset(train,failure_diagnostic_positions(failure_mix,plan_records,plan_windows,limit*2))
        if rank==0:
            failure_info=dict(experiment='evaluation_adaptation_failure_state_mix_v1',
                plan_sha256=failure_mix.plan_sha256,provenance=failure_provenance,
                simulation=failure_simulation,loader_audit_sha256=loader_gate['report_sha256'],
                total_planned_positions=len(failure_mix),loader_positions=len(train),
                diagnostic=a.diagnostic,initialization='59866 model AND optimizer; one new epoch, cumulative two',
                interpretation='Same old per-window exposure, not same ordering, RNG or optimizer trajectory; not unseen-test generalization.')
            with (output/'failure_state_exposure.json').open('x') as f:json.dump(failure_info,f,indent=2)
    sampler=DistributedSampler(train,num_replicas=world,rank=rank,shuffle=True,seed=a.seed,drop_last=True)
    loader=DataLoader(train,batch_size=a.batch_size,sampler=sampler,drop_last=True,num_workers=a.workers,pin_memory=True)
    model=JointRobotModel(a.root,a.encoder_weight,a.wla_source,a.wla_checkpoint,a.kind).to(device)
    wrapped=DDP(model,device_ids=[local],find_unused_parameters=True,broadcast_buffers=False) if world>1 else model
    groups=optimizer_groups(model);base_lrs=[g['lr'] for g in groups]
    optimizer=torch.optim.AdamW(groups,weight_decay=.01)
    resume_step=0
    if a.resume:
        checkpoint=torch.load(a.resume,map_location='cpu',weights_only=True,mmap=True)
        if checkpoint['kind']!=a.kind or checkpoint['contract']!=CONTRACT:raise ValueError('resume contract mismatch')
        expected={k for k in model.state_dict() if not k.startswith('encoder.')}
        if set(checkpoint['model'])!=expected:raise ValueError('resume state mismatch')
        result=model.load_state_dict(checkpoint['model'],strict=False)
        if result.unexpected_keys or any(not k.startswith('encoder.') for k in result.missing_keys):raise ValueError('partial resume')
        optimizer.load_state_dict(checkpoint['optimizer']);resume_step=int(checkpoint['step'])
        if resume_step!=22707 or a.completed_epochs!=1:raise ValueError('this continuation is exactly original1epoch to2epochs')
        base_lrs=[g['lr'] for g in optimizer.param_groups]  # Continue from last LR, do not restart at peak.
        del checkpoint
    if rank==0:
        config=vars(a)|dict(contract=CONTRACT,cache_sha256=cache_sha,index_audit_sha256=sha(Path(a.index_root)/'audit.json'),effective_batch=world*a.batch_size*a.accumulation,train_rows=len(train),heldout_rows=len(val),world_size=world,precision='fp32 training / bf16 frozen encoder',base_lrs=base_lrs)
        root=Path(__file__).resolve().parents[2]
        env=dict(commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),dirty=subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True).strip(),torch=torch.__version__,gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],source_sha256={str(f.relative_to(root)):sha(f) for f in sorted((root/'wa').rglob('*.py'))},wla=model.provenance)
        (output/'config.json').write_text(json.dumps(config,indent=2));(output/'environment.json').write_text(json.dumps(env,indent=2))
    # Random streams diverge after identical model initialization/DDP synchronization.
    torch.manual_seed(a.seed+rank+10000*a.completed_epochs)
    start=time.monotonic();step=resume_step;phase_steps=math.ceil(len(loader)/a.accumulation)*a.epochs;total_steps=resume_step+phase_steps
    from wa.wm.history_augmentation import repeat_current_history
    history_rng=torch.Generator(device=device).manual_seed(a.seed+rank+7919)
    for epoch in range(a.epochs):
        sampler.set_epoch(epoch+a.completed_epochs);wrapped.train();optimizer.zero_grad(set_to_none=True)
        group_log=torch.zeros(4,device=device)
        if failure_mix is not None:
            runtime_exposure=FailureRuntimeExposure(failure_mix)
        elif planned_mix is not None:
            runtime_exposure=RuntimeExposure(planned_mix,episode_index=plan_windows['episode_index'],
                episodes=plan_records,hard=plan_windows['hard'],early=plan_windows['early'])
        for i,batch in enumerate(loader):
            exposure_ids=None
            if runtime_exposure is not None:
                batch,exposure_ids=strip_exposure_tag(batch)  # CPU bookkeeping never enters moved/model.
            batch=moved(batch,device);modes=torch.randint(0,3,(len(batch['pose']),),device=device)
            batch=repeat_current_history(batch,a.history_repeat_probability,history_rng)
            group_start=i//a.accumulation*a.accumulation
            group_count=min(a.accumulation,len(loader)-group_start)
            sync=(i+1)%a.accumulation==0 or i+1==len(loader)
            scope=wrapped.no_sync() if world>1 and not sync else contextlib.nullcontext()
            with scope:
                values=wrapped(batch,modes,a.world_weight)
                if not torch.isfinite(values['loss']):raise FloatingPointError('nonfinite loss')
                (values['loss']/group_count).backward()
            if runtime_exposure is not None:runtime_exposure.consume(exposure_ids)
            group_log+=torch.stack([values[k].detach().float() for k in ('loss','flow','geometry','world')])/group_count
            if sync:
                phase_step=step-resume_step
                warmup=1. if a.resume else min(1.,(phase_step+1)/100)
                multiplier=warmup*(.1+.9*.5*(1+math.cos(math.pi*phase_step/max(phase_steps,1))))
                for g,lr in zip(optimizer.param_groups,base_lrs):g['lr']=lr*multiplier
                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
                optimizer.step();optimizer.zero_grad(set_to_none=True);step+=1
                log_now=step==resume_step+1 or step%25==0 or a.diagnostic
                if log_now and world>1:dist.all_reduce(group_log);group_log/=world
                if rank==0 and log_now:
                    row=dict(step=step,total_steps=total_steps,phase_step=step-resume_step,epoch=epoch+a.completed_epochs,
                        **{k:float(group_log[j]) for j,k in enumerate(('loss','flow','geometry','world'))},
                        loss_aggregation='DDP_mean_over_accumulation_group',loss_rank0_last_microbatch=float(values['loss'].detach()),
                        lr=[g['lr'] for g in optimizer.param_groups],grad_norm=float(norm),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,elapsed_s=time.monotonic()-start)
                    with (output/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    print(json.dumps(row),flush=True)
                group_log.zero_()
                if rank==0 and not a.diagnostic and step%2000==0:
                    torch.save({'model':{k:v for k,v in model.state_dict().items() if not k.startswith('encoder.')},'optimizer':optimizer.state_dict(),'step':step,'kind':a.kind,'contract':CONTRACT},output/f'step-{step:07d}.pt')
        if runtime_exposure is not None:
            state=runtime_exposure.state()
            if world>1:
                contexts=[None]*world
                dist.all_gather_object(contexts,runtime_exposure.context_sha256)
                runtime_exposure.validate_contexts(contexts)
                actual=state['position_counts'].to(device)
                dist.all_reduce(actual,op=dist.ReduceOp.SUM)
                state['position_counts']=actual.cpu()
                del actual
            if rank==0:
                exposure=(runtime_exposure.summarize(state) if a.diagnostic else
                    runtime_exposure.finalize(state=state,world=world,batch=a.batch_size,
                        seed=a.seed,epoch=epoch+a.completed_epochs))
                if a.diagnostic:exposure['status']='DIAGNOSTIC_PARTIAL_EXPOSURE_ONLY'
                exposure['optimizer_steps_completed']=step-resume_step
                exposure['diagnostic']=a.diagnostic
                if failure_mix is not None:
                    base_counts,teacher_counts,recovery_counts=runtime_exposure.source_counts(state)
                    count_arrays=dict(base_counts=base_counts.numpy(),teacher_counts=teacher_counts.numpy(),
                        recovery_counts=recovery_counts.numpy())
                else:
                    base_counts,teacher_counts=runtime_exposure.source_counts(state)
                    count_arrays=dict(base_counts=base_counts.numpy(),teacher_counts=teacher_counts.numpy())
                np.savez_compressed(output/f'actual_exposure_epoch{epoch+a.completed_epochs}.npz',
                    position_counts=state['position_counts'].numpy(),**count_arrays)
                with (output/f'actual_exposure_epoch{epoch+a.completed_epochs}.json').open('x') as f:
                    json.dump(exposure,f,indent=2)
            if world>1:dist.barrier()
    if rank==0 and not a.diagnostic:
        torch.save({'model':{k:v for k,v in model.state_dict().items() if not k.startswith('encoder.')},'optimizer':optimizer.state_dict(),'step':step,'kind':a.kind,'contract':CONTRACT,'completed_epochs':a.completed_epochs+a.epochs,'parent_checkpoint':a.resume,'parent_sha256':a.resume_sha256},output/'checkpoint.pt')
    if world>1:dist.barrier()
    metrics=evaluate(model,val,a.batch_size,device,rank,world)
    if rank==0:
        (output/'metrics.json').write_text(json.dumps(dict(status='DIAGNOSTIC_PASS' if a.diagnostic else 'OFFLINE_ONLY',kind=a.kind,steps=step,metrics=metrics,closed_loop=False,edge_latency_verified=False,point_source='simulated_uwb',elapsed_s=time.monotonic()-start),indent=2))
        print('COMPLETE',json.dumps(metrics),flush=True)
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
