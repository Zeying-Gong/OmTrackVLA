"""Reproduce actual DDP and batch tail drops without loading images."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from torch.utils.data import DistributedSampler
from wa.data import sha
from wa.wm.recovery_mix import RecoveryMix

def exposure(base_count, teacher_count, repeats, world=8, batch=2, seed=42, epoch=1):
    if min(base_count, teacher_count, world, batch) < 1:
        raise ValueError('positive sizes required')
    mix = RecoveryMix(range(base_count), range(teacher_count), repeats)
    seen = np.zeros(len(mix), dtype=np.uint8)
    teachers = np.zeros(teacher_count, dtype=np.int64)
    ranks = []
    for rank in range(world):
        sampler = DistributedSampler(mix, num_replicas=world, rank=rank,
                                     shuffle=True, seed=seed, drop_last=True)
        sampler.set_epoch(epoch)
        indices = np.asarray(list(sampler), dtype=np.int64)
        indices = indices[:len(indices)//batch*batch]
        if seen[indices].any() or len(np.unique(indices)) != len(indices):
            raise ValueError('rank overlap or duplicate mixed index')
        seen[indices] = 1
        selected = indices[indices >= base_count] - base_count
        np.add.at(teachers, selected % teacher_count, 1)
        ranks.append(dict(rank=rank, total=len(indices), teacher=len(selected),
                          base=int((indices < base_count).sum())))
    report = dict(seed=seed, epoch=epoch, world_size=world, batch_size=batch,
        repeats=repeats, base_unique=base_count, teacher_unique=teacher_count,
        pre_ddp_total=len(mix), actual_total=int(seen.sum()),
        actual_base=int(seen[:base_count].sum()), actual_teacher=int(teachers.sum()),
        dropped=int(len(mix)-seen.sum()), ranks=ranks,
        per_window_exposure_histogram={str(k):v for k,v in Counter(teachers.tolist()).items()},
        sampler='DistributedSampler shuffle/drop_last; DataLoader drop_last',
        scope='Exact index simulation, not evidence that training executed')
    return report, teachers

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True)
    p.add_argument('--base-index', required=True)
    p.add_argument('--repeats', type=int, required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    cache = Path(a.cache)
    rows = np.load(cache/'index/train_valid.npy')
    base = np.load(Path(a.base_index)/'train_valid.npy')
    episodes = json.loads((cache/'train_episodes.json').read_text())
    episode_ids = np.load(cache/'train_episode.npy', mmap_mode='r')[rows]
    report, counts = exposure(len(base), len(rows), a.repeats)
    by_episode = np.bincount(episode_ids, weights=counts, minlength=len(episodes)).astype(np.int64)
    groups = Counter()
    per_episode = []
    for i, entry in enumerate(episodes):
        value = int(by_episode[i])
        groups[entry['task']+':'+entry['teacher']] += value
        per_episode.append(dict(episode_uid=entry['episode_uid'], teacher=entry['teacher'],
                                exposures=value))
    report.update(cache=str(cache), cache_sha256=sha(cache/'complete.json'),
        index_sha256=sha(cache/'index/train_valid.npy'),
        base_index_sha256=sha(Path(a.base_index)/'train_valid.npy'),
        groups=dict(groups), per_episode=per_episode,
        teacher_fraction=report['actual_teacher']/report['actual_total'])
    with Path(a.output).open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k:v for k,v in report.items() if k != 'per_episode'}, indent=2))

if __name__ == '__main__':
    main()
