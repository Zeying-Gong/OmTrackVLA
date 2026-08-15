import torch
import torch.nn as nn

from . import dino_vit

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoEncoder(nn.Module):
    """Frozen DINOv2-S/14 backbone producing patch tokens.

    Shared across current RGB, target image, history frames and (training-only)
    future RGB. Initialized from facebookresearch dinov2_vits14_pretrain.pth.
    """

    def __init__(self, ckpt_path):
        super().__init__()
        self.model = dino_vit.DINOv2("vits")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        ckpt = {k: v for k, v in ckpt.items() if not k.startswith("head.")}
        missing, unexpected = self.model.load_state_dict(ckpt, strict=False)
        unexpected = [k for k in unexpected if k != "mask_token"]
        assert not missing, f"missing keys: {missing}"
        assert not unexpected, f"unexpected keys: {unexpected}"
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dim = 384
        self.patch_size = 14
        self.register_buffer("_mean", torch.tensor(_IMAGENET_MEAN)[None, :, None, None])
        self.register_buffer("_std", torch.tensor(_IMAGENET_STD)[None, :, None, None])

    @torch.no_grad()
    def forward_patches(self, rgb):
        """rgb: (B,3,H,W) float [0,1] or uint8 -> (B, P, 384) patch tokens."""
        if rgb.dtype == torch.uint8:
            rgb = rgb.float() / 255.0
        elif rgb.numel() and rgb.max() > 1.5:
            rgb = rgb / 255.0
        rgb = (rgb - self._mean) / self._std
        toks = self.model.get_intermediate_layers(rgb, n=1, norm=True)[0]
        return toks

    @torch.no_grad()
    def grid_size(self, rgb):
        """Returns the real patch grid (Hp, Wp) for an image of rgb's spatial size."""
        h, w = rgb.shape[-2:]
        return h // self.patch_size, w // self.patch_size
