"""Strict local-only upstream loaders. No downloads and no silent partial loads."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import torch
from torch import nn

PINS={'jepa-wms':'13cf1d9c7e476f53c17714d2e0f1dc239a883ce0','dinov2':'7764ea0f912e53c92e82eb78a2a1631e92725fc8'}
HASHES={'jepa':'a01d99c4592fbedf44af076cf4c339de230c56f9f377c7559f584b97569b59bc','dino':'b4dc463b5c7a1546c1e5edf2ef813005c136523fa373ba98bc207782860b3803','encoder':'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'}

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()

def verify_source(path,name):
    if subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()!=PINS[name]:
        raise ValueError('upstream revision mismatch: '+name)
    if subprocess.check_output(['git','-C',str(path),'status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('modified upstream source: '+name)

def state(module, weights):
    module.load_state_dict({k.removeprefix('module.'):v for k,v in weights.items()},strict=True)

def load_encoder(root,weight):
    root=Path(root);verify_source(root/'upstream_audit/dinov2','dinov2')
    if sha(weight)!=HASHES['encoder']:raise ValueError('encoder hash mismatch')
    model=torch.hub.load(str(root/'upstream_audit/dinov2'),'dinov2_vits14',source='local',pretrained=False)
    state(model,torch.load(weight,map_location='cpu',weights_only=True))
    return model.eval().requires_grad_(False)

class OfficialWorld(nn.Module):
    def __init__(self,root,kind):
        super().__init__()
        if kind not in ('jepa','dino'):raise ValueError(kind)
        root=Path(root); verify_source(root/'upstream_audit/jepa-wms','jepa-wms')
        sys.path.insert(0,str(root/'upstream_audit/jepa-wms'))
        from app.plan_common.models.AdaLN_vit import vit_predictor_AdaLN
        from app.plan_common.models.vit import ViTPredictor
        from app.plan_common.models.prop_embedding import ProprioceptiveEmbedding
        path=root/'models/jepa_wms'/f'mz_{kind}-wm.pth.tar'
        if sha(path)!=HASHES[kind]:raise ValueError('world checkpoint hash mismatch')
        ckpt=torch.load(path,map_location='cpu',weights_only=True)
        self.kind=kind
        self.prop=ProprioceptiveEmbedding(num_frames=4,tubelet_size=1,in_chans=4,embed_dim=16 if kind=='jepa' else 20,shift_input=False)
        state(self.prop,ckpt['proprio_encoder'])
        if kind=='jepa':
            self.predictor=vit_predictor_AdaLN(img_size=224,patch_size=14,num_frames=4,tubelet_size=1,embed_dim=384,predictor_embed_dim=384,depth=6,num_heads=16,use_rope=True,local_window=(3,-1,-1),action_dim=10,proprio_dim=4,proprio_emb_dim=16,proprio_encoder_inpred=False)
        else:
            self.predictor=ViTPredictor(num_patches=256,num_frames=4,dim=414,depth=6,heads=16,mlp_dim=2048,dropout=.1,use_sdpa=True)
            self.act=ProprioceptiveEmbedding(num_frames=4,tubelet_size=1,in_chans=10,embed_dim=10,shift_input=False)
            state(self.act,ckpt['action_encoder'])
        state(self.predictor,ckpt['predictor'])

    def forward(self,z,actions,proprio):
        b,t,p,d=z.shape
        if not 1<=t<=4 or (p,d)!=(256,384) or actions.shape!=(b,t,10) or proprio.shape!=(b,t,4):
            raise ValueError('official PointMaze interface requires <=4 frames / 256 patches / action10 / state4')
        if not all(torch.isfinite(x).all() for x in (z,actions,proprio)):raise ValueError('nonfinite dynamics input')
        prop=self.prop(proprio).expand(-1,-1,256,-1)
        if self.kind=='jepa':return self.predictor(z.reshape(b,t,1,16,16,384),actions,prop)[0]
        x=torch.cat([z,prop,self.act(actions).expand(-1,-1,256,-1)],-1)
        return self.predictor(x.reshape(b,t*256,414)).reshape(b,t,256,414)[...,:384]

def load_wla_heads(source,checkpoint):
    """Read-only dependency on user's WLA snapshot; do not redistribute it silently."""
    source=Path(source);sys.path.insert(0,str(source/'src'))
    from md_wla.models.action.expert import ActionExpertConfig,LayerwiseActionExpert
    from md_wla.models.queries import MetaQueryTokens
    # Exact checkpoint construction flags from the existing WLA EVT loader.
    from md_wla.deploy.rx_model import EXACT_MODEL_ENVIRONMENT
    previous={k:v for k,v in os.environ.items() if k.startswith('MD_WLA_DIAGNOSTIC_') or k in EXACT_MODEL_ENVIRONMENT}
    try:
        for k in previous:os.environ.pop(k,None)
        os.environ.update(EXACT_MODEL_ENVIRONMENT)
        expert=LayerwiseActionExpert(ActionExpertConfig(action_dim=4,horizon=7,backbone_dim=2560,model_dim=1024,num_blocks=16,num_heads=32,max_state_dim=0,state_history_frames=4,tap_indices=tuple(range(12,28))))
    finally:
        for k in EXACT_MODEL_ENVIRONMENT:os.environ.pop(k,None)
        os.environ.update(previous)
    query=MetaQueryTokens(hidden_size=2560)
    target=nn.Sequential(nn.LayerNorm(2560),nn.Linear(2560,512),nn.SiLU(),nn.Linear(512,3))
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    for name,module in [('action_expert',expert),('metaquery',query),('target_head',target)]:
        module.load_state_dict(ckpt[name],strict=True)
    manifest={str(p.relative_to(source)):sha(p) for p in sorted((source/'src/md_wla').rglob('*.py'))}
    return query,expert,target,{'checkpoint_sha256':sha(checkpoint),'checkpoint_step':ckpt.get('step'),'source_files':manifest,'strict_loaded':['action_expert','metaquery','target_head'],'omitted':['Qwen backbone','language LoRA'],'action_contract':'7x4 XY/sin(yaw)/cos(yaw)'}
