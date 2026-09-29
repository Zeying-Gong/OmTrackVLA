"""Short developer diagnostic: official WM + restored WLA heads, no training job."""
import argparse,json,sys,time,traceback
from pathlib import Path
import torch

p=argparse.ArgumentParser()
for key in ('root','wla-source','wla-checkpoint','encoder-weight','cache','output'):
    p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from wa.wm.adapters import GoalMetaQueryAdapter,WorldActionPolicy
from wa.wm.loaders import OfficialWorld,load_encoder,load_wla_heads,sha,PINS,HASHES
from wa.data import TrackingData

if a.output.exists():raise FileExistsError(a.output)
a.output.parent.mkdir(parents=True,exist_ok=True)
torch.manual_seed(11);torch.set_num_threads(4)
report={'status':'RUNNING','warning':'Interface/gradient diagnostic only. Robot actions NOT mapped to PointMaze. Fusion adapters untrained; no SR/CR or quality claim. Not a formal GPU task.', 'python':sys.version,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'upstream_pins':PINS,'weight_hashes':HASHES,'models':{}}
report['adapter_sources']={str(x.relative_to(Path(__file__).resolve().parents[2])):sha(x) for x in (Path(__file__).resolve(),Path(__file__).resolve().parents[1]/'wm/adapters.py',Path(__file__).resolve().parents[1]/'wm/loaders.py')}
def save():a.output.write_text(json.dumps(report,indent=2)+'\n')
def gradnorm(module):
    return sum(float(x.grad.float().norm()) for x in module.parameters() if x.grad is not None)
def check(cond,msg):
    if not cond:raise AssertionError(msg)
save()
try:
    encoder=load_encoder(a.root,a.encoder_weight).cuda()
    sample=TrackingData(a.cache,'train',limit=101)[100]
    with torch.no_grad():
        visual=encoder.forward_features(sample['rgb'].cuda())['x_norm_patchtokens'][None]
        template=encoder.forward_features(sample['template'][None].cuda())['x_norm_patchtokens']
        future=encoder.forward_features(sample['future'][None].cuda())['x_norm_patchtokens']
    report['sample']={'split':'train','index':100,'times':sample['times'].tolist(),'uwb_source':'simulator target label ONLY for explicitly simulated UWB input; not real sensor','polar':sample['point'][:2].tolist()}
    for kind in ('jepa','dino'):
        query,expert,target,provenance=load_wla_heads(a.wla_source,a.wla_checkpoint)
        report['wla']=provenance
        model=WorldActionPolicy(GoalMetaQueryAdapter(query),expert,target,OfficialWorld(a.root,kind)).cuda().eval()
        # Backprop through unchanged heads while avoiding allocating their parameter gradients.
        model.action_expert.requires_grad_(False);model.target_head.requires_grad_(False)
        batch=dict(visual=visual,template=template,template_valid=torch.tensor([True],device='cuda'),polar=sample['point'][None,:2].cuda(),uwb_valid=torch.tensor([True],device='cuda'),age_s=torch.zeros(1,device='cuda'),times=sample['times'][None].cuda())
        taps,q=model.conditions(batch)
        check(len(taps)==28 and taps[-1].shape==(1,64,2560),'WLA tap contract')
        wm_loss=model.dynamics_loss(visual,q,torch.zeros(1,4,10,device='cuda'),torch.zeros(1,4,4,device='cuda'),future,action_contract='pointmaze_pretrained_probe_only')
        wm_loss.backward()
        wm_grads={'metaquery':gradnorm(model.adapter.metaquery),'polar_encoder':gradnorm(model.adapter.polar_encoder),'fusion':gradnorm(model.adapter.fusion),'bridge':gradnorm(model.world_bridge),'official_predictor':gradnorm(model.world)}
        check(all(v>0 and torch.isfinite(torch.tensor(v)) for v in wm_grads.values()),'WM gradient path missing')
        model.zero_grad(set_to_none=True)
        # Action gradient diagnostic uses shape-correct synthetic SE2 labels; not an optimizer step.
        labels=torch.zeros(1,7,4,device='cuda');labels[...,3]=1
        flow_loss=model.action_training_loss(batch,labels,torch.zeros(1,3,device='cuda'),torch.ones(1,device='cuda'))
        flow_loss.backward()
        flow_grad=gradnorm(model.adapter.metaquery)
        check(flow_grad>0 and torch.isfinite(torch.tensor(flow_grad)),'action gradient path missing')
        model.zero_grad(set_to_none=True)
        # Inference must not invoke the world predictor, even accidentally.
        def forbidden(*args):raise AssertionError('world predictor called during policy inference')
        hook=model.world.register_forward_pre_hook(forbidden)
        noise=torch.randn(1,7,4,device='cuda'); outputs={}
        for mode in ('mixed','image','point'):
            b=dict(batch)
            if mode=='image':b['uwb_valid']=torch.tensor([False],device='cuda')
            if mode=='point':b['template_valid']=torch.tensor([False],device='cuda')
            start=time.perf_counter();pose,geometry=model.predict(b,noise)
            torch.cuda.synchronize()
            check(torch.isfinite(pose).all() and torch.isfinite(geometry).all(),'nonfinite output')
            outputs[mode]={'shape':list(pose.shape),'one_call_ms':(time.perf_counter()-start)*1000,'pose':pose.cpu().tolist(),'target_geometry':geometry.cpu().tolist()}
        hook.remove()
        report['models'][kind]={'status':'DIAGNOSTIC_PASS','parameters_total':sum(x.numel() for x in model.parameters())+sum(x.numel() for x in encoder.parameters()),'wm_loss':float(wm_loss.detach()),'wm_gradient_norms':wm_grads,'flow_loss':float(flow_loss.detach()),'flow_metaquery_gradient_norm':flow_grad,'world_off_inference_verified':True,'modes':outputs}
        print(kind,json.dumps({k:v for k,v in report['models'][kind].items() if k!='modes'}),flush=True)
        save()
        del model,expert,query,target,taps,q,wm_loss,flow_loss
        torch.cuda.empty_cache()
    report['status']='DIAGNOSTIC_PASS';save()
except Exception:
    report['status']='FAILED';report['error']=traceback.format_exc();save();raise
