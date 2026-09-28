"""Small prompt-conditioned causal policy; baseline, not an USS reproduction."""
import torch
from torch import nn
from torchvision.models import resnet18

class Policy(nn.Module):
    def __init__(self, weights=None, width=128):
        super().__init__()
        encoder = resnet18(weights=None)
        if weights:
            encoder.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.encoder = nn.Sequential(*list(encoder.children())[:-2])
        self.encoder.requires_grad_(False)
        self.visual = nn.Linear(512, width)
        self.position = nn.Parameter(torch.randn(1, 49, width) * .01)
        self.identity = nn.Linear(512, width)
        self.geometry = nn.Sequential(nn.Linear(5, width), nn.GELU(), nn.Linear(width, width))
        self.missing_id = nn.Parameter(torch.zeros(width))
        self.missing_point = nn.Parameter(torch.zeros(width))
        self.fusion = nn.MultiheadAttention(width, 4, dropout=0., batch_first=True)
        self.time = nn.Sequential(nn.Linear(1, width), nn.Tanh())
        layer = nn.TransformerEncoderLayer(width, 4, width * 4, dropout=0.,
                                          batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.action = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 14))
        self.dynamics = nn.Sequential(nn.Linear(width + 14, width), nn.GELU(), nn.Linear(width, 512))

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()  # frozen BN: no future-frame leakage via batch statistics
        return self

    def scene(self, images):
        with torch.no_grad():
            self.encoder.eval()
            return self.encoder(images).flatten(2).transpose(1, 2)

    def forward(self, rgb, template, point, times, image_valid, point_valid, future=None, labels=None, world_weight=0.):
        # point: range_m, bearing_rad, age_s; invalid entries may contain NaN.
        b, t = rgb.shape[:2]
        features = self.scene(rgb.flatten(0, 1))
        visual = self.visual(features) + self.position
        ident = self.identity(self.scene(template).mean(1))
        ident = torch.where(image_valid[:, None], ident, self.missing_id)
        safe = torch.where(point_valid[:, None], point, torch.zeros_like(point))
        r, angle, age = safe.unbind(-1)
        numeric = torch.stack((torch.log1p(r.clamp_min(0)), angle.sin(), angle.cos(),
                               age.clamp_min(0), point_valid.float()), -1)
        geo = self.geometry(numeric)
        geo = torch.where(point_valid[:, None], geo, self.missing_point)
        query = torch.stack((ident, geo), 1)[:, None].expand(-1, t, -1, -1).reshape(b*t, 2, -1)
        fused = self.fusion(query, visual, visual, need_weights=False)[0].mean(1).reshape(b, t, -1)
        fused = fused + self.time(times[..., None])
        mask = torch.triu(torch.ones(t, t, device=rgb.device, dtype=torch.bool), diagonal=1)
        states = self.temporal(fused, mask=mask)
        result = {"xy": self.action(states[:, -1]).reshape(b, 7, 2), "states": states}
        if labels is not None:
            result["trajectory_loss"] = nn.functional.smooth_l1_loss(result["xy"], labels)
            aux = self.world_loss(states[:, -1], labels, future) if world_weight else result["xy"].sum()*0
            result["loss"] = result["trajectory_loss"] + world_weight*aux
        return result

    def world_loss(self, state, trajectory, future):
        # Training-only auxiliary. Deployment forward never accepts future observations.
        target = self.scene(future).mean(1).detach()
        predicted = self.dynamics(torch.cat((state, trajectory.flatten(1)), -1))
        return (1 - nn.functional.cosine_similarity(predicted, target, dim=-1)).mean()
