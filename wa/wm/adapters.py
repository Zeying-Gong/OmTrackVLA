"""WA interface adapters, not a replacement visual or dynamics backbone.

USS-inspired read/write/read fusion; WLA owns MetaQuery and ActionExpert.
Inputs are causal, caller-owned windows. No hidden cross-episode cache.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def polar_features(polar, valid, age_s, *, range_scale=10., max_age_s=.5):
    """Robot-frame planar metres/radians; forward=0, left positive.

    Missing/stale measurements may contain NaN and are masked BEFORE arithmetic.
    Zero distance is valid and is never a missing-value sentinel.
    """
    if polar.ndim != 2 or polar.shape[-1] != 2 or not polar.is_floating_point():
        raise ValueError('polar must be floating [B,2] = [r metres, theta radians]')
    if valid.dtype != torch.bool or valid.shape != polar.shape[:1] or age_s.shape != valid.shape:
        raise ValueError('valid bool[B] and age_s[B] required')
    if not math.isfinite(range_scale) or range_scale <= 0 or not math.isfinite(max_age_s) or max_age_s <= 0:
        raise ValueError('invalid range scale or freshness limit')
    if not torch.isfinite(age_s[valid]).all() or (age_s[valid] < 0).any():
        raise ValueError('valid measurement requires finite nonnegative age')
    usable = valid & (age_s <= max_age_s)
    if not torch.isfinite(polar[usable]).all() or (polar[usable,0] < 0).any():
        raise ValueError('usable UWB requires finite r>=0 and theta')
    p = torch.where(usable[:,None], polar, torch.zeros_like(polar))
    age = torch.where(usable, age_s, torch.zeros_like(age_s))
    features = torch.stack([p[:,0]/range_scale,p[:,1].sin(),p[:,1].cos(),age/max_age_s,usable.to(p.dtype)],-1)
    return features * usable[:,None], usable


class ReadWriteRead(nn.Module):
    """Adapter implementation of USS-style fusion, not official USS code."""
    def __init__(self, dim=384, heads=6):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim,heads,batch_first=True)
        self.read = nn.MultiheadAttention(dim,heads,batch_first=True)
        self.write = nn.MultiheadAttention(dim,heads,batch_first=True)
        self.reread = nn.MultiheadAttention(dim,heads,batch_first=True)
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(4))

    def forward(self, query, prompt, prompt_valid, visual):
        u = torch.cat([query,prompt],1)
        mask = torch.cat([torch.zeros(query.shape[:2],dtype=torch.bool,device=query.device),~prompt_valid],1)
        u = self.norms[0](u+self.self_attn(u,u,u,key_padding_mask=mask,need_weights=False)[0])
        u = self.norms[1](u+self.read(u,visual,visual,need_weights=False)[0])
        z = self.norms[2](visual+self.write(visual,u,u,key_padding_mask=mask,need_weights=False)[0])
        u = self.norms[3](u+self.reread(u,z,z,need_weights=False)[0])
        return u[:,:query.shape[1]],z


class GoalMetaQueryAdapter(nn.Module):
    """Explicit 384 -> 2560 WLA interface; no fictitious Qwen layer outputs.

    Three fusion states map monotonically to 16 existing action conditioning slots.
    Slot-specific norms and one shared projection are newly trained. Unused WLA
    slots 0..11 are placeholders, not claimed pretrained visual layers.
    """
    def __init__(self, metaquery, dim=384, backbone_dim=2560, slots=16):
        super().__init__()
        self.metaquery = metaquery
        self.query_in = nn.Linear(backbone_dim,dim)
        self.polar_encoder = nn.Sequential(nn.Linear(5,dim),nn.SiLU(),nn.Linear(dim,dim))
        self.time_encoder = nn.Sequential(nn.Linear(1,dim),nn.SiLU(),nn.Linear(dim,dim))
        self.kind = nn.Parameter(torch.randn(2,dim)*.02)
        self.fusion = nn.ModuleList(ReadWriteRead(dim) for _ in range(3))
        self.slot_norm = nn.ModuleList(nn.LayerNorm(dim) for _ in range(slots))
        self.to_wla = nn.Linear(dim,backbone_dim)
        self.mapping = tuple(i*2//(slots-1) for i in range(slots))

    def forward(self, visual, template, template_valid, polar, uwb_valid, age_s, times):
        if visual.ndim != 4 or visual.shape[-1] != 384 or visual.shape[2] != 256:
            raise ValueError('visual must be [B,T,256,384]')
        b,t,_,d = visual.shape
        if times.shape != (b,t) or not torch.isfinite(times).all() or (times>0).any() or (times[:,1:]<times[:,:-1]).any() or (times[:,-1]!=0).any():
            raise ValueError('times must be ordered causal offsets ending at zero')
        if not torch.isfinite(visual).all(): raise ValueError('nonfinite visual features')
        if template.shape != (b,256,d) or template_valid.shape != (b,) or template_valid.dtype != torch.bool:
            raise ValueError('template [B,256,384] and template_valid bool[B] required')
        if not torch.isfinite(template[template_valid]).all(): raise ValueError('invalid template')
        pf, usable = polar_features(polar,uwb_valid,age_s)
        if not (template_valid | usable).all():
            raise ValueError('no usable goal; caller must hold/stop or supply explicit goal memory')
        safe_template = torch.where(template_valid[:,None,None],template,torch.zeros_like(template))
        # Fixed spatial pooling keeps a 4x4 target appearance token grid.
        crop = F.adaptive_avg_pool2d(safe_template.transpose(1,2).reshape(b,d,16,16),(4,4)).flatten(2).transpose(1,2)
        prompt = torch.cat([crop+self.kind[0], self.polar_encoder(pf)[:,None]+self.kind[1]],1)
        prompt_valid = torch.cat([template_valid[:,None].expand(-1,16),usable[:,None]],1)
        z = (visual+self.time_encoder(times[:,:,None])[:,:,None]).flatten(1,2)
        q = self.query_in(self.metaquery())[None].expand(b,-1,-1)
        states=[]
        for block in self.fusion:
            q,z = block(q,prompt,prompt_valid,z); states.append(q)
        slots = [self.to_wla(norm(states[index])) for norm,index in zip(self.slot_norm,self.mapping)]
        taps = tuple([slots[0]]*12+slots)
        return taps,q


class QueryToWorld(nn.Module):
    """Conditions the last dense grid without changing official token layout."""
    def __init__(self, dim=384):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim,6,batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.gain = nn.Parameter(torch.tensor(.01))

    def forward(self, visual, query):
        z = visual[:,-1]
        delta = self.attn(self.norm(z),query,query,need_weights=False)[0]
        return torch.cat([visual[:,:-1],(z+self.gain*delta)[:,None]],1)


class WorldActionPolicy(nn.Module):
    """Existing WLA action/target heads + explicit goal adapter + official WM.

    World actions remain explicitly in the checkpoint's domain for interface tests.
    No silent robot-command -> PointMaze mapping is provided.
    """
    def __init__(self, adapter, action_expert, target_head, world):
        super().__init__()
        self.adapter=adapter; self.action_expert=action_expert
        self.target_head=target_head; self.world=world; self.world_bridge=QueryToWorld()

    def conditions(self, batch):
        return self.adapter(**batch)

    def dynamics_loss(self, visual, query, actions, proprio, future, *, action_contract):
        if action_contract != 'pointmaze_pretrained_probe_only':
            raise ValueError('Robot dynamics action/time contract is not yet validated')
        z=self.world_bridge(visual,query)
        predicted=self.world(z,actions,proprio)[:,-1]
        if future.shape != predicted.shape or not torch.isfinite(future).all():
            raise ValueError('future target must be finite [B,256,384]')
        return F.smooth_l1_loss(F.layer_norm(predicted,(384,)),F.layer_norm(future.detach(),(384,)))

    def action_training_loss(self, batch, pose, target_geometry, weights):
        """Keep existing TargetPolicy's flow + 0.5 geometry objective unchanged.

        Dynamics is separate until robot control/time semantics are validated;
        this method is not a production trainer or authorization to skip that gate.
        """
        from md_wla.models.action.flow import sample_flow_training_pair
        taps,_=self.conditions(batch)
        b=taps[0].shape[0]
        if pose.shape!=(b,7,4) or target_geometry.shape!=(b,3) or weights.shape!=(b,):
            raise ValueError('Expected pose[B,7,4], geometry[B,3], weights[B]')
        if not all(torch.isfinite(x).all() for x in (pose,target_geometry,weights)) or (weights<0).any():
            raise ValueError('Nonfinite labels or invalid weights')
        pair=sample_flow_training_pair(pose.float())
        mask=torch.ones_like(pose,dtype=torch.bool)
        pred=self.action_expert(taps,pair.noisy,pair.t,action_dim_mask=mask,time_mask=mask[...,0])
        flow=(pred.float()-pair.velocity.float()).square().mean((1,2))
        estimate=self.target_head(taps[-1].float().mean(1))
        geometry=F.smooth_l1_loss(estimate,target_geometry.float(),reduction='none').mean(-1)
        return ((flow+.5*geometry)*weights).mean()

    @torch.no_grad()
    def predict(self, batch, noise, steps=4):
        from md_wla.models.action.flow import euler_sample
        taps,_=self.conditions(batch)
        if noise.shape != (taps[0].shape[0],7,4): raise ValueError('Expected WLA SE2 noise [B,7,4]')
        mask=torch.ones_like(noise,dtype=torch.bool)
        pose=euler_sample(lambda x,t:self.action_expert(taps,x,t,action_dim_mask=mask,time_mask=mask[...,0]),noise=noise,steps=steps)
        return pose,self.target_head(taps[-1].float().mean(1))
