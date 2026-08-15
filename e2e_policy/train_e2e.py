"""Training entry for the e2e follow policy (architecture note 0814).

Default: waypoint + stop + target-state supervision only (auxiliary forward /
inverse dynamics disabled, per implementation-review step 2). Enable them with
--use-forward-dyn --use-inverse-dyn once the base follower learns.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "data_loader"))

from e2e_policy import E2EFollowPolicy, PolicyConfig
from e2e_policy import losses as L
from e2e_policy.data_collate import collate_batch

from mixed import MixedOmniDataset


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sage-root", default="/data/nfs/share/OmTrackVLA/example_datasets/sage3d_extracted")
    ap.add_argument("--tpt-root", default="/data/nfs/share/OmTrackVLA/example_datasets/tpt_bench")
    ap.add_argument("--navdp-root", default="/data/nfs/share/OmTrackVLA/example_datasets/traj_data")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--steps", type=int, default=0, help=">0 overrides epochs (debug)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--history", type=int, default=8)
    ap.add_argument("--image-steps", type=int, default=0, help=">0: cache N real images to /tmp")
    ap.add_argument("--use-forward-dyn", action="store_true")
    ap.add_argument("--use-inverse-dyn", action="store_true")
    ap.add_argument("--no-target-state", action="store_true", help="disable target-state head/loss (default on)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-interval", type=int, default=5)
    ap.add_argument("--out", default="/data/nfs/share/OmTrackVLA/e2e_policy/ckpt_e2e.pt")
    return ap.parse_args()


def make_model_cfg(args):
    cfg = PolicyConfig()
    cfg.use_forward_dyn = args.use_forward_dyn
    cfg.use_inverse_dyn = args.use_inverse_dyn
    cfg.use_target_state_head = not args.no_target_state
    return cfg


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds = MixedOmniDataset(args.sage_root, args.tpt_root, args.navdp_root,
                          image_size=args.size, memory_size=args.history)
    print(f"dataset len={len(ds)} (pf_units={len(ds.pf_units)}, navdp_eps={ds.episodes})")

    cfg = make_model_cfg(args)
    model = E2EFollowPolicy(cfg).to(args.device)
    model.train()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train / 1e6:.2f}M; fwd_dyn={cfg.use_forward_dyn} inv_dyn={cfg.use_inverse_dyn}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)

    total_steps = args.steps if args.steps > 0 else ds.__len__() // args.batch * args.epochs
    t0 = time.time()
    for step in range(total_steps):
        samples = [ds.get_sample(as_arrays=True) for _ in range(args.batch)]
        samples = [s for s in samples if s is not None]
        if not samples:
            continue
        mkw, batch = collate_batch(samples, image_size=args.size, max_history=args.history,
                                   horizon=cfg.horizon, device=args.device)

        opt.zero_grad()
        mkw["trajectory"] = batch["trajectory"]
        out = model(**mkw)
        total, logs = L.compute_total_loss(out, batch, cfg)
        if not torch.isfinite(total):
            print(f"step {step}: non-finite loss {total.item()}, skip")
            continue
        total.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 10.0)
        opt.step()

        if step % args.log_interval == 0:
            dt = time.time() - t0
            rate = (step + 1) / dt
            print(f"step {step}/{total_steps} | loss={total.item():.4f} | {rate:.2f} step/s | "
                  f"L_wp={logs.get('L_waypoint', float('nan')):.4f} | {logs}", flush=True)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        torch.save(model.state_dict(), args.out)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
