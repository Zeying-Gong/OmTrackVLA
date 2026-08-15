"""Self-test for the unified loader: build datasets, pull samples, print shapes."""
from __future__ import annotations

import sys

from sage3d_ds import Sage3DDataset
from tpt_ds import TpTDataset
from navdp_ds import NavDPDataset
from mixed import MixedOmniDataset
from unified import TASK_PERSON_FOLLOW, TASK_POINTNAV, TASK_IMAGEGOAL


def _describe(s, n):
    if s is None:
        return "None"
    out = [f"task={s['task']} ep={s['ep_id'][:60]} valid={s['valid']}"]
    if s.get("current"):
        out.append(f"current={s['current']}")
    out.append(f"window={len(s['window'])}")
    if s.get("target_img"):
        out.append(f"target={s['target_img']}")
    if s.get("traj"):
        out.append(f"traj={len(s['traj'])} pts first={[round(x,2) for x in s['traj'][0]]}")
    if s.get("actions"):
        out.append(f"actions={len(s['actions'])}")
    if s.get("pointgoal"):
        out.append(f"pg={[round(x,3) for x in s['pointgoal']]} valid={s['pointgoal_valid']}")
    if s.get("future_rgb"):
        out.append(f"future={s['future_rgb']}")
    if s.get("depth"):
        out.append(f"depth={s['depth']}")
    if s.get("bbox"):
        out.append(f"bbox={[round(x) for x in s['bbox']]}")
    return " | ".join(out)


def main():
    import os
    sage_root = os.environ.get("OMTVL_SAGE_ROOT", "/data/nfs/share/OmTrackVLA/data/sage3d_extracted")
    tpt_root = os.environ.get("OMTVL_TPT_ROOT", "/data/nfs/share/OmTrackVLA/data/tpt_bench")
    navdp_root = os.environ.get("OMTVL_NAVDP_ROOT", "/h100-2/vln_n1/traj_data")

    if "--sage-only" in sys.argv:
        ds = Sage3DDataset(sage_root)
        print(f"sage eps: {len(ds)}")
        for i in range(min(4, len(ds))):
            n = ds.step_count(i)
            for k in [0, n // 2, min(n - 1, 299)]:
                print(" ", _describe(ds.get(i, k), None))
        return

    if "--tpt-only" in sys.argv:
        ds = TpTDataset(tpt_root)
        print(f"tpt seqs: {len(ds)}")
        for i in range(min(4, len(ds))):
            seq = ds.seqs[i]["seq"]
            t = ds._table(seq)
            for r in [0, t.num_rows // 2, min(t.num_rows - 1, 400)]:
                print(" ", _describe(ds.get(seq, r), None))
        return

    if "--navdp-only" in sys.argv:
        for cond in [TASK_POINTNAV, TASK_IMAGEGOAL]:
            ds = NavDPDataset(navdp_root, condition=cond, cap_episodes=6)
            print(f"navdp[{cond}] eps(cap): {len(ds)}")
            for i in range(4):
                print(" ", _describe(ds.get(i, cond), None))
        return

    ds = MixedOmniDataset(sage_root, tpt_root, navdp_root, seed=7)
    print(f"mixed len={len(ds)} pf_units={len(ds.pf_units)} navdp_eps={ds.episodes}")
    for i in range(12):
        s = ds.get_sample(index=i, as_arrays=True)
        arr = f"current_arr={s['current_arr'].shape} window_arr={s['window_arr'].shape}"
        target = f"target_arr={s['target_arr'].shape}" if s.get("target_arr") is not None else "target_arr=None"
        print(f"[{i}] {_describe(s, None)} | {arr} {target}")


if __name__ == "__main__":
    main()