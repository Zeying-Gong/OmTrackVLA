"""Official pretrained PointMaze WM interface probe, NOT a tracking evaluation."""
import argparse, hashlib, json, sys, time, subprocess
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--encoder-weight', type=Path, required=True)
p.add_argument('--rgb-root', type=Path, required=True)
a = p.parse_args()
sys.path.insert(0, str(a.root / 'upstream_audit/jepa-wms'))
from app.plan_common.models.AdaLN_vit import vit_predictor_AdaLN
from app.plan_common.models.vit import ViTPredictor
from app.plan_common.models.prop_embedding import ProprioceptiveEmbedding

torch.manual_seed(42)
torch.set_num_threads(4)
device = 'cuda'
def count(m): return sum(p.numel() for p in m.parameters())
def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()
def load(m, state):
    m.load_state_dict({k.removeprefix('module.'): v for k,v in state.items()}, strict=True)
    return m.eval().to(device)
def timing(fn):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(20):
        start = time.perf_counter(); fn(); torch.cuda.synchronize()
        samples.append((time.perf_counter()-start)*1000)
    return {'p50_ms': float(np.percentile(samples,50)), 'p95_ms': float(np.percentile(samples,95)), 'iterations':20}

weight = a.encoder_weight
encoder = torch.hub.load(str(a.root/'upstream_audit/dinov2'), 'dinov2_vits14', source='local', pretrained=False)
load(encoder, torch.load(weight,map_location='cpu',weights_only=True))
rgbroot = a.rgb_root
paths = sorted(rgbroot.glob('rgb_*.png'))[:3]
assert len(paths)==3
transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
images = torch.stack([transform(Image.open(f).convert('RGB')) for f in paths]).to(device)
report = {'status':'interface_probe_only','warning':'Official Meta PointMaze checkpoints; real EVT images but synthetic normalized action/proprio probes, not tracking or prediction-quality evaluation. FP32 developer A800 timing excludes I/O, preprocessing, planning and control; not Thor latency.', 'torch':torch.__version__, 'gpu':torch.cuda.get_device_name(), 'encoder_parameters':count(encoder),'encoder_sha256':digest(weight),'images':[str(f) for f in paths], 'models':{}}
report['source_commits'] = {name: subprocess.check_output(['git','-C',str(a.root/'upstream_audit'/name),'rev-parse','HEAD'],text=True).strip() for name in ['dinov2','jepa-wms']}
report['script_sha256'] = digest(__file__)
report['python'] = sys.version
a.output.parent.mkdir(parents=True,exist_ok=True)
if a.output.exists(): raise FileExistsError(a.output)
with torch.inference_mode():
    encode = lambda: encoder.forward_features(images)['x_norm_patchtokens'].reshape(1,3,256,384)
    features = encode()
    report['encoder_timing'] = timing(encode)
    actions = torch.zeros(1,3,10,device=device)
    proprio = torch.zeros(1,3,4,device=device)
    for name in ['jepa','dino']:
        path = a.root/'models/jepa_wms'/f'mz_{name}-wm.pth.tar'
        ckpt = torch.load(path,map_location='cpu',weights_only=True)
        prop = ProprioceptiveEmbedding(num_frames=4,tubelet_size=1,in_chans=4,embed_dim=16 if name=='jepa' else 20,shift_input=False)
        load(prop,ckpt['proprio_encoder'])
        if name=='jepa':
            pred = vit_predictor_AdaLN(img_size=224,patch_size=14,num_frames=4,tubelet_size=1,embed_dim=384,predictor_embed_dim=384,depth=6,num_heads=16,use_rope=True,local_window=(3,-1,-1),action_dim=10,proprio_dim=4,proprio_emb_dim=16,proprio_encoder_inpred=False)
            def forward(z, act): return pred(z.reshape(1,3,1,16,16,384),act,prop(proprio).expand(-1,-1,256,-1))[0]
            extra = 0
        else:
            pred = ViTPredictor(num_patches=256,num_frames=4,dim=414,depth=6,heads=16,mlp_dim=2048,dropout=.1,use_sdpa=True)
            act_enc = ProprioceptiveEmbedding(num_frames=4,tubelet_size=1,in_chans=10,embed_dim=10,shift_input=False)
            load(act_enc,ckpt['action_encoder']); extra = count(act_enc)
            def forward(z, act):
                x = torch.cat([z,prop(proprio).expand(-1,-1,256,-1),act_enc(act).expand(-1,-1,256,-1)],dim=-1)
                return pred(x.reshape(1,768,414)).reshape(1,3,256,414)[...,:384]
        load(pred,ckpt['predictor'])
        del ckpt
        torch.cuda.reset_peak_memory_stats()
        result = forward(features,actions)
        altered = features.clone(); altered[:,-1] += .1
        row = {'checkpoint_sha256':digest(path),'strict_load':True,'predictor_parameters':count(pred),'total_parameters':count(encoder)+count(pred)+count(prop)+extra,'output_shape':list(result.shape),'finite':bool(result.isfinite().all()),'action_response_mean_abs':float((forward(features,actions+.1)-result).abs().mean()),'causal_future_leak_max':float((forward(altered,actions)[:,:-1]-result[:,:-1]).abs().max()),'predictor_timing':timing(lambda:forward(features,actions)), 'encoder_plus_predictor_timing':timing(lambda:forward(encode(),actions)), 'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20}
        report['models'][name] = row
        a.output.write_text(json.dumps(report,indent=2)+'\n')
        print(name,json.dumps(row),flush=True)
        del pred, prop, result
        if name=='dino': del act_enc
        torch.cuda.empty_cache()
