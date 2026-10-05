"""Read audited existing EVT caches; never consume target text."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

PROTOCOL = "wa_evt_xy7_dt0.1_bbox_simuwb_v1"
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def tensor_image(path, box=None):
    with Image.open(path) as source:
        im = source.convert("RGB")
        if box is not None:
            x1,y1,x2,y2 = box
            if not (0 <= x1 < x2 <= im.width and 0 <= y1 < y2 <= im.height):
                raise ValueError("Invalid target bbox")
            im = im.crop(box)
        # Preserve the entire observation; resize is a deliberate baseline choice.
        array = np.asarray(im.resize((224,224)), dtype=np.float32).copy()/255.
    value = torch.from_numpy(array).permute(2,0,1)
    return (value - torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]

class TrackingData(Dataset):
    def __init__(self, cache, part, data_root=None, source_prefix=None, limit=None):
        self.cache = Path(cache)
        self.part = part
        self.data_root = Path(data_root) if data_root else None
        self.source_prefix = Path(source_prefix) if source_prefix else None
        if bool(self.data_root) != bool(self.source_prefix):
            raise ValueError("data_root and source_prefix must be supplied together")
        self.episodes = json.loads((self.cache/f"{part}_episodes.json").read_text())
        self.pose = np.load(self.cache/f"{part}_pose.npy", mmap_mode="r")
        self.history = np.load(self.cache/f"{part}_history.npy", mmap_mode="r")
        self.episode = np.load(self.cache/f"{part}_episode.npy", mmap_mode="r")
        if self.pose.shape != (len(self.episode),7,4) or self.history.shape != (len(self.episode),4):
            raise ValueError("Expected audited XY+sin/cos-yaw, seven future points, four history frames")
        self.length = len(self.pose) if limit is None else min(limit,len(self.pose))
        if self.length < 1:
            raise ValueError("Empty split")

    def __len__(self):
        return self.length

    def validate_identity(self, meta, root):
        if not meta.get("camera_alignment_verified") or meta.get("initial_bbox_status") != "VERIFIED_CONFIG_AND_SEMANTIC":
            raise ValueError("Unverified first-frame identity: "+str(root))

    @lru_cache(maxsize=16)
    def episode_info(self, index):
        entry = self.episodes[index]
        root = Path(entry["root"])
        if self.data_root:
            root = self.data_root / root.relative_to(self.source_prefix)
        meta = json.loads((root/"metadata.json").read_text())
        self.validate_identity(meta, root)
        obs = json.loads((root/"observations.json").read_text())
        times = np.array([o["timestamp_s"] for o in obs])
        if not np.isfinite(times).all() or not np.all(np.diff(times)>0):
            raise ValueError("Invalid timebase")
        box = meta["initial_bbox_rgb_xyxy"]
        template = tensor_image(root/entry["frames"][0], box)
        return entry, root, obs, times, template

    def __getitem__(self, index):
        entry, root, obs, timestamps, template = self.episode_info(int(self.episode[index]))
        history = self.history[index]
        now = int(history[-1])
        if np.any(history > now) or np.any(history < 0):
            raise ValueError("Noncausal history")
        future = int(np.searchsorted(timestamps, timestamps[now]+.7, side="left"))
        if future >= len(obs):
            raise ValueError("No future observation for dynamics supervision")
        # Explicit simulator-only UWB approximation. No text is used as a policy input.
        item = obs[now]
        delta = np.asarray(item["target_position_world_label_only"]) - np.asarray(item["robot_position_world"])
        local = delta @ np.asarray(item["robot_rotation_world_from_body"])
        point = torch.tensor([np.hypot(local[0],local[2]),np.arctan2(-local[2],local[0]),0.],dtype=torch.float32)
        xy = torch.from_numpy(np.array(self.pose[index,:,:2]))
        if not torch.isfinite(xy).all() or not torch.isfinite(point).all():
            raise ValueError("Nonfinite geometry")
        return dict(rgb=torch.stack([tensor_image(root/entry["frames"][int(j)]) for j in history]),
                    template=template.clone(), point=point,
                    times=torch.tensor(timestamps[history]-timestamps[now],dtype=torch.float32),
                    xy=xy, future=tensor_image(root/entry["frames"][future]))

def audit(cache):
    cache = Path(cache)
    meta = json.loads((cache/"complete.json").read_text())
    for name in ("train","heldout"):
        for suffix in ("pose.npy","history.npy","episode.npy","episodes.json"):
            filename = f"{name}_{suffix}"
            if sha(cache/filename) != meta["files"][filename]:
                raise ValueError("Cache checksum mismatch: "+filename)
    splits = {p:json.loads((cache/f"{p}_episodes.json").read_text()) for p in ("train","heldout")}
    if {e["scene"] for e in splits["train"]} & {e["scene"] for e in splits["heldout"]}:
        raise ValueError("Scene leakage")
    return sha(cache/"complete.json")
