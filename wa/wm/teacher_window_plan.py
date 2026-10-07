"""Opt-in bounded teacher re-exposure; no model/task-ID inputs or loss changes.

This module does not select successful demonstrations. The caller must derive an
eligible window whitelist from the frozen student/teacher evidence, verify its
source hashes, and pass that whitelist to validate_plan before use.
"""
from collections import Counter
import hashlib
import json
import re

import numpy as np
import torch
from torch.utils.data import Dataset, DistributedSampler

SCHEMA = 'wa.teacher-window-plan.v1'
SOURCE_KEYS = frozenset(('cache_complete', 'teacher_index', 'base_index',
                         'student_rows', 'teacher_selections'))
SAMPLE_INDEX = '_wa_exposure_index'

def canonical_sha(plan):
    payload = json.dumps(plan, sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()

def positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0

def validate_plan(plan, *, base_count, teacher_count, source_hashes,
                  eligible_teacher_indices):
    if not positive_int(base_count) or not positive_int(teacher_count):
        raise ValueError('positive source counts required')
    required = {'schema', 'base_count', 'teacher_count', 'extra_repeats',
                'extra_teacher_indices', 'source_hashes'}
    if set(plan) != required or plan['schema'] != SCHEMA:
        raise ValueError('unknown plan schema/fields')
    if not positive_int(plan['base_count']) or not positive_int(plan['teacher_count']):
        raise ValueError('invalid plan counts')
    if (plan['base_count'], plan['teacher_count']) != (base_count, teacher_count):
        raise ValueError('source count mismatch')
    for hashes in (source_hashes, plan['source_hashes']):
        if not isinstance(hashes, dict) or set(hashes) != SOURCE_KEYS:
            raise ValueError('all source hashes required')
        if any(not isinstance(h, str) or not re.fullmatch('[0-9a-f]{64}', h)
               for h in hashes.values()):
            raise ValueError('invalid source hash')
    if plan['source_hashes'] != source_hashes:
        raise ValueError('source hash mismatch')
    repeats = plan['extra_repeats']
    if type(repeats) is not int or not 0 <= repeats <= 2:
        raise ValueError('at most two extra exposures per eligible window')
    ids = plan['extra_teacher_indices']
    if not isinstance(ids, list) or any(type(i) is not int for i in ids):
        raise ValueError('integer window indices required')
    if ids != sorted(set(ids)) or any(i < 0 or i >= teacher_count for i in ids):
        raise ValueError('indices must be unique sorted and in bounds')
    eligible = list(eligible_teacher_indices)
    if any(type(i) is not int or i < 0 or i >= teacher_count for i in eligible):
        raise ValueError('invalid eligibility whitelist')
    if len(eligible) != len(set(eligible)) or not set(ids).issubset(eligible):
        raise ValueError('foreign or duplicate eligibility')
    if bool(ids) != bool(repeats):
        raise ValueError('empty/no-op plan must have zero extra repeats')
    return tuple(ids), repeats

class PlannedTeacherMix(Dataset):
    """Each base/teacher once, plus a bounded, externally verified teacher plan."""
    def __init__(self, base, teacher, plan, *, source_hashes,
                 eligible_teacher_indices):
        self.extra, self.repeats = validate_plan(plan, base_count=len(base),
            teacher_count=len(teacher), source_hashes=source_hashes,
            eligible_teacher_indices=eligible_teacher_indices)
        self.base, self.teacher = base, teacher
        self.plan_sha256 = canonical_sha(plan)

    def __len__(self):
        return len(self.base) + len(self.teacher) + len(self.extra) * self.repeats

    def locate(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        if index < len(self.base):
            return 'base', index
        index -= len(self.base)
        if index < len(self.teacher):
            return 'teacher', index
        return 'teacher', self.extra[(index - len(self.teacher)) % len(self.extra)]

    def __getitem__(self, index):
        source, index = self.locate(index)
        return (self.base if source == 'base' else self.teacher)[index]

class ExposureTaggedData(Dataset):
    """Keep sampler-position evidence out of the model via strip_exposure_tag."""
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        item = self.dataset[index]
        if not isinstance(item, dict) or SAMPLE_INDEX in item:
            raise ValueError('expected untagged dict sample')
        return dict(item, **{SAMPLE_INDEX: torch.tensor(index, dtype=torch.long)})

def strip_exposure_tag(batch):
    indices = batch.get(SAMPLE_INDEX)
    if not isinstance(indices, torch.Tensor) or indices.dtype != torch.long or indices.ndim != 1:
        raise ValueError('missing/invalid batched exposure tag')
    return {k: v for k, v in batch.items() if k != SAMPLE_INDEX}, indices

def simulate_exposure(mix, *, world=8, batch=2, seed=42, epoch=1):
    if not positive_int(world) or not positive_int(batch):
        raise ValueError('positive world/batch required')
    seen = np.zeros(len(mix), dtype=np.uint8)
    teacher_counts = np.zeros(len(mix.teacher), dtype=np.int64)
    ranks = []
    for rank in range(world):
        sampler = DistributedSampler(mix, num_replicas=world, rank=rank,
            shuffle=True, seed=seed, drop_last=True)
        sampler.set_epoch(epoch)
        ids = list(sampler)
        ids = ids[:len(ids) // batch * batch]
        if len(set(ids)) != len(ids) or seen[ids].any():
            raise ValueError('overlapping sampler positions')
        seen[ids] = 1
        teacher_n = 0
        for index in ids:
            source, item = mix.locate(index)
            if source == 'teacher':
                teacher_counts[item] += 1
                teacher_n += 1
        ranks.append(dict(rank=rank, total=len(ids), teacher=teacher_n,
                          base=len(ids) - teacher_n))
    actual_teacher = int(teacher_counts.sum())
    report = dict(plan_sha256=mix.plan_sha256, seed=seed, epoch=epoch,
        world_size=world, batch_size=batch, pre_ddp_total=len(mix),
        actual_total=int(seen.sum()), actual_teacher=actual_teacher,
        actual_base=int(seen.sum()) - actual_teacher, dropped=int((seen == 0).sum()),
        ranks=ranks, per_window_exposure_histogram=dict(Counter(teacher_counts.tolist())),
        scope='Exact index simulation only; not executed training exposure')
    return report, teacher_counts
