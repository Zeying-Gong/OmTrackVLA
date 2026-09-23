#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duty", type=float, default=0.1, help="Duty cycle as 0..1, or percent as 0..100.")
    parser.add_argument("--period", type=float, default=10.0)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--sync-every", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available for gpu_burn_duty.py")

    raw_duty = args.duty / 100.0 if args.duty > 1.0 else args.duty
    duty = max(0.0, min(1.0, raw_duty))
    active_s = args.period * duty
    sleep_s = max(0.0, args.period - active_s)
    sync_every = max(1, args.sync_every)

    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.dtype]
    print(
        f"Starting duty GPU burn on {torch.cuda.get_device_name(0)}: "
        f"duty={duty:.3f}, period={args.period:.1f}s, size={args.size}, dtype={args.dtype}, "
        f"sync_every={sync_every}",
        flush=True,
    )
    a = torch.randn(args.size, args.size, device="cuda", dtype=dtype)
    b = torch.randn(args.size, args.size, device="cuda", dtype=dtype)
    c = torch.empty_like(a)
    while True:
        end = time.monotonic() + active_s
        op_count = 0
        while time.monotonic() < end:
            torch.mm(a, b, out=c)
            op_count += 1
            if op_count % sync_every == 0:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        if sleep_s > 0:
            time.sleep(sleep_s)


if __name__ == "__main__":
    main()
