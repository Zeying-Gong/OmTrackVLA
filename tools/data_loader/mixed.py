"""Task-balanced mixed dataset over Sage3D(person_follow) + TpT(person_follow)
+ NavDP(pointnav/imagegoal), sampling sources by weight 2:1:1."""
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
        weights=(2.0, 1.0, 1.0),
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

        # Build lazily-addressable person_follow units (sage: per derived step;
        # tpt: per video-frame row).
        self.pf_units = []  # (src, id, key, n)
        for i in range(len(self.sage)):
            n = self.sage.step_count(i)
            for k in range(start_offset, n):
                self.pf_units.append(("sage", None if False else i, k, n))
        for i in range(len(self.tpt)):
            t = self.tpt._table(self.tpt.seqs[i]["seq"])
            nrow = t.num_rows
            for r in range(start_offset, nrow):
                self.pf_units.append(("tpt", i, r, nrow))

        self.episodes = len(self.navdp)

    def __len__(self):
        return len(self.pf_units) + self.episodes * 2

    def _pf_sample(self, i):
        src, id_, key, _ = self.pf_units[i]
        if src == "sage":
            return self.sage.get(id_, key)
        return self.tpt.get(self.tpt.seqs[id_]["seq"], key)

    def get_sample(self, index=None, as_arrays=False):
        if index is None:
            c = self.rng.rand()
            if c < self.weights[0]:
                s = self._pf_sample(int(self.rng.randint(0, len(self.pf_units))))
            elif c < self.weights[0] + self.weights[1]:
                s = self.navdp.sample(TASK_POINTNAV)
            else:
                s = self.navdp.sample(TASK_IMAGEGOAL)
        else:
            # deterministic: map index buckets by weights across units
            idx = int(index) % len(self)
            n_pf = len(self.pf_units)
            n_nav = self.episodes
            if idx < n_pf:
                s = self._pf_sample(idx)
            elif idx < n_pf + n_nav:
                s = self.navdp.get(idx - n_pf, TASK_POINTNAV)
            else:
                s = self.navdp.get(idx - n_pf - n_nav, TASK_IMAGEGOAL)
        if s is None:
            return None
        if as_arrays:
            arrs = load_sample_arrays(s, img_size=self.image_size, memory_size=self.memory_size)
            s = dict(s)
            s["current_arr"] = arrs["current"]
            s["window_arr"] = arrs["window"]
            s["target_arr"] = arrs["target"]
            s["depth_arr"] = arrs.get("depth")
        return s