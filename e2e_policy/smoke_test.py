"""Smoke test: build the policy, run a training forward/backward on dummy data."""
import argparse

import torch

from e2e_policy import E2EFollowPolicy, PolicyConfig
from e2e_policy import losses as L

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--size", type=int, default=224)     # current/future RGB
    ap.add_argument("--hsize", type=int, default=224)    # history RGB
    ap.add_argument("--tsize", type=int, default=224)    # target crop
    ap.add_argument("--history", type=int, default=3)
    ap.add_argument("--iter", type=int, default=2)
    args = ap.parse_args()

    torch.manual_seed(0)
    cfg = PolicyConfig()
    model = E2EFollowPolicy(cfg).to(args.device)
    model.train()

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_trainable/1e6:.2f}M, total params: {n_total/1e6:.2f}M")

    B = args.batch
    cur = torch.rand(B, 3, args.size, args.size, device=args.device)
    tgt = torch.rand(B, 3, args.tsize, args.tsize, device=args.device)
    hist = torch.rand(B, args.history, 3, args.hsize, args.hsize, device=args.device)
    motion = torch.randn(B, args.history, 3, device=args.device)
    pg = torch.rand(B, 7, device=args.device)
    task = torch.zeros(B, dtype=torch.long, device=args.device)
    gspec = torch.rand(B, 5, device=args.device)
    traj = torch.rand(B, cfg.horizon, cfg.action_dim, device=args.device) * 0.1
    fut = torch.rand(B, 3, args.size, args.size, device=args.device)
    av = torch.ones(B, cfg.horizon, device=args.device)
    ac = torch.ones(B, cfg.horizon, device=args.device)
    depth = torch.rand(B, (args.size // 14) ** 2, device=args.device)
    free = (torch.rand(B, (args.size // 14) ** 2, device=args.device) > 0.5).float()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    for it in range(args.iter):
        opt.zero_grad()
        out = model(current_rgb=cur, target_image=tgt, pointgoal=pg,
                    history_rgb=hist, history_motion=motion,
                    task_type=task, goal_spec=gspec,
                    trajectory=traj, future_rgb=fut)
        batch = {
            "trajectory": traj, "action_valid": av, "action_confidence": ac,
            "depth_residual": depth, "free_target": free,
            "patch_weight": L.build_patch_weight(free, alpha_target=2.0),
            "target_state": torch.rand(B, 3, device=args.device),
            "target_state_valid": torch.ones(B, 3, device=args.device),
            "stop_label": (torch.rand(B, device=args.device) > 0.5).float(),
            "stop_task_mask": torch.ones(B, dtype=torch.long, device=args.device),
            "grid": (args.size // 14, args.size // 14),
        }
        total, logs = L.compute_total_loss(out, batch, cfg)
        total.backward()
        gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 10.0)
        opt.step()
        shapes = {k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)}
        print(f"iter {it}: loss={total.item():.4f} grad_norm={gn:.3f}")
        print(f"  out shapes: {shapes}")
        print(f"  logs: {logs}")

    print("SMOKE OK")


if __name__ == "__main__":
    main()
