"""Task-balanced mixed dataset over Sage3D(person_follow+waypoint) + TpT
(person_follow, no waypoint) + NavDP(pointnav/imagegoal).

Sampling is stratified by (task, supervision type) so that person-follow
waypoint supervision (Sage3D) keeps a guaranteed share instead of being diluted
by the ~141k no-waypoint TpT rows.
"""
from __future__ import annotations

import numpy as np

from sage3d_ds import Sage3DDataset
from tpt_ds import TpTDataset
from navdp_ds import NavDPDataset

from unified import TASK_PERSON_FOLLOW, TASK_POINTNAV, TASK_IMAGEGOAL, load_sample_arrays


class MixedOmniDataset:
    def __init__(
        self,
        sage_root,
        tpt_root,
        navdp_root,
        weights=(1.0, 0.5, 1.0, 1.0),  # (sage waypoint, tpt no-wp, pointnav, imagegoal)
        seed=0,
        image_size=224,
        memory_size=8,
        navdp_condition_random=True,
        start_offset=2,
    ):
        self.image_size = image_size
        self.memory_size = memory_size
        self.rng = np.random.RandomState(seed)
        self.weights = np.array(weights, dtype=np.float64)
        self.weights = self.weights / self.weights.sum()

        self.sage = Sage3DDataset(sage_root)
        self.tpt = TpTDataset(tpt_root)
        self.navdp = NavDPDataset(navdp_root, memory_size=memory_size, seed=seed)

        # Sage3D person-follow steps WITH waypoint supervision.
        self.sage_units = []  # (src, id, key, n)
        for i in range(len(self.sage)):
            n = self.sage.step_count(i)
            for k in range(start_offset, n):
                self.sage_units.append(("sage", i, k, n))
        # TpT person-follow frames WITHOUT waypoint supervision.
        self.tpt_units = []
        for i in range(len(self.tpt)):
            t = self.tpt._table(self.tpt.seqs[i]["seq"])
            nrow = t.num_rows
            for r in range(start_offset, nrow):
                self.tpt_units.append(("tpt", i, r, nrow))

        self.pf_units = self.sage_units + self.tpt_units   # compat alias
        self.episodes = len(self.navdp)

    def __len__(self):
        return len(self.sage_units) + len(self.tpt_units) + self.episodes * 2

    def _pf_sample(self, unit):
        src, id_, key, _ = unit
        if src == "sage":
            return self.sage.get(id_, key)
        return self.tpt.get(self.tpt.seqs[id_]["seq"], key)

    def get_sample(self, index=None, as_arrays=False):
        n_sage = len(self.sage_units)
        n_tpt = len(self.tpt_units)
        n_nav = self.episodes
        if index is None:
            c = self.rng.rand()
            if c < self.weights[0]:
                s = self._pf_sample(self.sage_units[int(self.rng.randint(0, n_sage))])
            elif c < self.weights[0] + self.weights[1]:
                s = self._pf_sample(self.tpt_units[int(self.rng.randint(0, n_tpt))])
            elif c < self.weights[0] + self.weights[1] + self.weights[2]:
                s = self.navdp.sample(TASK_POINTNAV)
            else:
                s = self.navdp.sample(TASK_IMAGEGOAL)
        else:
            # deterministic: bucket by index across the four groups
            idx = int(index) % len(self)
            if idx < n_sage:
                s = self._pf_sample(self.sage_units[idx])
            elif idx < n_sage + n_tpt:
                s = self._pf_sample(self.tpt_units[idx - n_sage])
            elif idx < n_sage + n_tpt + n_nav:
                s = self.navdp.get(idx - n_sage - n_tpt, TASK_POINTNAV)
            else:
                s = self.navdp.get(idx - n_sage - n_tpt - n_nav, TASK_IMAGEGOAL)
        if s is None:
            return None
        if as_arrays:
            arrs = load_sample_arrays(s, img_size=self.image_size, memory_size=self.memory_size)
            s = dict(s)
            s["current_arr"] = arrs["current"]
            s["window_arr"] = arrs["window"]
            s["target_arr"] = arrs["target"]
            s["depth_arr"] = arrs.get("depth")
            s["future_depth"] = arrs.get("future_depth")
        return s
