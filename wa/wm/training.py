"""Joint WLA action + official latent dynamics training on audited robot commands."""
import torch
from torch import nn
from torch.nn import functional as F
from wa.wm.adapters import GoalMetaQueryAdapter,WorldActionPolicy
from wa.wm.loaders import load_encoder,load_wla_heads,OfficialWorld
from wa.wm.robot_data import CONTRACT

class RobotConditionAdapter(nn.Module):
    """New trainable domain interface; no claim of PointMaze action equivalence.

    Input [normalized_forward, normalized_left, normalized_yaw, actual_dt/0.1].
    State is the preceding recorded command plus its interval, not privileged pose.
    Official world-model interior weights are retained; these interfaces are new.
    """
    def __init__(self,out_dim):
        super().__init__();self.layers=nn.Sequential(nn.Linear(4,64),nn.SiLU(),nn.Linear(64,out_dim))
    def forward(self,x):
        if x.shape[-1]!=4 or not torch.isfinite(x).all() or (x[...,3]<=0).any():
            raise ValueError('invalid robot command/dt')
        return self.layers(x)

class JointRobotModel(nn.Module):
    def __init__(self,root,encoder_weight,wla_source,wla_checkpoint,kind):
        super().__init__()
        self.encoder=load_encoder(root,encoder_weight)
        query,expert,target,self.provenance=load_wla_heads(wla_source,wla_checkpoint)
        self.policy=WorldActionPolicy(GoalMetaQueryAdapter(query),expert,target,OfficialWorld(root,kind))
        self.robot_action=RobotConditionAdapter(10);self.robot_state=RobotConditionAdapter(4)
        self.kind=kind
    def train(self,mode=True):
        super().train(mode);self.encoder.eval();return self
    @torch.no_grad()
    def encode(self,images):
        # Frozen image encoder; memory-bounded chunks and cached output tensors.
        shape=images.shape[:-3];flat=images.reshape(-1,*images.shape[-3:]);parts=[]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for chunk in flat.split(4):parts.append(self.encoder.forward_features(chunk)['x_norm_patchtokens'].float())
        return torch.cat(parts).reshape(*shape,256,384)
    def make_conditions(self,batch,mode_ids):
        return dict(visual=self.encode(batch['rgb']),template=self.encode(batch['template']),
                    template_valid=batch.get('template_valid',mode_ids!=1),polar=batch['polar'],uwb_valid=mode_ids!=0,
                    age_s=batch.get('uwb_age_s',torch.zeros_like(batch['polar'][:,0])),times=batch['times'])
    def forward(self,batch,mode_ids,world_weight=.1):
        from md_wla.models.action.flow import sample_flow_training_pair
        cond=self.make_conditions(batch,mode_ids);taps,query=self.policy.conditions(cond)
        pose=batch['pose'];pair=sample_flow_training_pair(pose);mask=torch.ones_like(pose,dtype=torch.bool)
        prediction=self.policy.action_expert(taps,pair.noisy,pair.t,action_dim_mask=mask,time_mask=mask[...,0])
        flow=(prediction.float()-pair.velocity.float()).square().mean()
        geometry=F.smooth_l1_loss(self.policy.target_head(taps[-1].float().mean(1)),batch['geometry'])
        z=self.policy.world_bridge(self.encode(batch['wm_rgb']),query)
        future=self.encode(batch['future'])
        pred=self.policy.world(z,self.robot_action(batch['commands']),self.robot_state(batch['proprio']))[:,-1]
        wm=F.smooth_l1_loss(F.layer_norm(pred,(384,)),F.layer_norm(future.detach(),(384,)))
        loss=flow+.5*geometry+world_weight*wm
        return {'loss':loss,'flow':flow.detach(),'geometry':geometry.detach(),'world':wm.detach()}
    @torch.no_grad()
    def predict(self,batch,mode_ids,noise):
        return self.policy.predict(self.make_conditions(batch,mode_ids),noise)

def optimizer_groups(model):
    query=model.policy.adapter.metaquery
    qids={id(p) for p in query.parameters()}
    new=[p for p in model.policy.adapter.parameters() if id(p) not in qids]
    new+=list(model.policy.world_bridge.parameters())+list(model.robot_action.parameters())+list(model.robot_state.parameters())
    return [dict(params=list(model.policy.action_expert.parameters()),lr=2.5e-6),
            dict(params=list(query.parameters()),lr=5e-6),
            dict(params=list(model.policy.target_head.parameters()),lr=5e-5),
            dict(params=list(model.policy.world.parameters()),lr=1e-5),
            dict(params=new,lr=1e-4)]
