"""Offline one-step inference from a trusted checkpoint and a JSON observation."""
import argparse
import json
from pathlib import Path
import torch
from wa.data import tensor_image
from wa.model import Policy

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--request",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    args=p.parse_args()
    torch.set_num_threads(2)
    item=json.loads(Path(args.request).read_text())
    mode=item["mode"]
    if mode not in ("image","point","mixed"):
        raise ValueError("Unknown mode")
    rgb=torch.stack([tensor_image(path) for path in item["rgb"]])[None].to(args.device)
    times=torch.tensor([item["times_s"]],dtype=torch.float32,device=args.device)
    if times.shape != rgb.shape[:2] or not torch.isfinite(times).all() or (times>0).any():
        raise ValueError("times_s must match causal RGB timestamps relative to now")
    if times[0,-1] != 0 or (times[:,1:] < times[:,:-1]).any():
        raise ValueError("times_s must be ordered and end at zero")
    iv=mode!="point"
    pv=mode!="image" and item.get("point_valid",False)
    template=tensor_image(item["target_image"],item["bbox"]) if iv else torch.zeros(3,224,224)
    point=torch.tensor([item["point"] if pv else [0,0,0]],dtype=torch.float32,device=args.device)
    if point.shape!=(1,3) or not torch.isfinite(point).all() or point[0,0]<0 or point[0,2]<0:
        raise ValueError("point must be finite nonnegative range, bearing radians, nonnegative age")
    checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    model=Policy().to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.inference_mode():
        result=model(rgb,template[None].to(args.device),point,times,
                     torch.tensor([iv],device=args.device),torch.tensor([pv],device=args.device))
    output=dict(xy_m=result["xy"][0].cpu().tolist(),times_s=[i/10 for i in range(1,8)],
                coordinate_frame="current_robot_x_forward_y_left",mode=mode,
                safety_checked=False,source_status="offline_policy_prediction")
    with open(args.output,"x") as stream:
        json.dump(output,stream,indent=2)
    print(json.dumps(output))

if __name__=="__main__":
    main()
