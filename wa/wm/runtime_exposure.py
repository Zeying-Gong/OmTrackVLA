"""Actual DataLoader consumption evidence; never model inputs or GPU collectives.

Call consume with the CPU index tensor removed by strip_exposure_tag, only after
the batch has been successfully processed. Counts prove consumption, not an
optimizer update or improved performance. Use a new counter for every epoch.
A distributed caller may verify all context hashes, SUM the CPU position_counts
(or transport a copy to its collective device), and pass the CPU result back to
finalize. This module never initializes a process group or touches CUDA.
"""
from collections import Counter
import hashlib
import json

import numpy as np
import torch
from torch.utils.data import DistributedSampler

from wa.wm.teacher_window_plan import PlannedTeacherMix, positive_int, simulate_exposure

SCHEMA = 'wa.runtime-exposure.v1'

def _array(value, count, name, *, boolean=False):
    if isinstance(value, torch.Tensor):
        if value.device.type != 'cpu':
            raise ValueError(name + ' must be CPU metadata')
        value = value.detach().numpy()
    result = np.asarray(value)
    kind = result.dtype == np.bool_ if boolean else np.issubdtype(result.dtype, np.integer)
    if result.ndim != 1 or len(result) != count or not kind:
        raise ValueError('invalid ' + name)
    return result.astype(np.bool_ if boolean else np.int64, copy=True)

def _counts(value, count):
    if not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or value.dtype != torch.long:
        raise ValueError('CPU int64 position counts required')
    if value.ndim != 1 or value.numel() != count or ((value < 0) | (value > 1)).any().item():
        raise ValueError('invalid or repeated sampler positions')
    return value

class RuntimeExposure:
    def __init__(self, mix, *, episode_index, episodes, hard, early):
        if not isinstance(mix, PlannedTeacherMix):
            raise ValueError('verified PlannedTeacherMix required')
        self.mix = mix
        self.base_count, self.teacher_count = len(mix.base), len(mix.teacher)
        self.total = len(mix)
        self.episode_index = _array(episode_index, self.teacher_count, 'episode_index')
        self.hard = _array(hard, self.teacher_count, 'hard', boolean=True)
        self.early = _array(early, self.teacher_count, 'early', boolean=True)
        if not isinstance(episodes, list) or not episodes:
            raise ValueError('episode metadata required')
        self.episodes = []
        for row in episodes:
            if not isinstance(row, dict):
                raise ValueError('invalid episode metadata')
            values = {k: row.get(k) for k in ('episode_uid', 'task', 'teacher')}
            if values['task'] not in ('stt', 'dt', 'at') or values['teacher'] not in ('lightnav', 'oracle'):
                raise ValueError('invalid task/teacher metadata')
            uid = values['episode_uid']
            if not isinstance(uid, str) or not uid.startswith(values['task'] + ':') or not uid.split(':', 1)[1]:
                raise ValueError('invalid episode identity')
            self.episodes.append(values)
        if len({x['episode_uid'] for x in self.episodes}) != len(self.episodes):
            raise ValueError('duplicate episode identity')
        if (self.episode_index < 0).any() or (self.episode_index >= len(self.episodes)).any():
            raise ValueError('episode index out of bounds')
        tasks = np.asarray([x['task'] for x in self.episodes])
        if (self.hard & (tasks[self.episode_index] != 'stt')).any():
            raise ValueError('hard-STT metadata includes another task')
        digest = hashlib.sha256()
        for value in (SCHEMA, mix.plan_sha256, self.episodes):
            digest.update(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
            digest.update(b'\0')
        for value in (self.episode_index, self.hard, self.early):
            digest.update(value.dtype.str.encode()); digest.update(value.tobytes())
        self.context_sha256 = digest.hexdigest()
        self._position_counts = torch.zeros(self.total, dtype=torch.long)
        # Source indices deliberately repeat only in the extra planned positions.
        self._teacher_sources = torch.tensor(list(range(self.teacher_count)) +
            list(mix.extra) * mix.repeats, dtype=torch.long)

    def consume(self, indices):
        if not isinstance(indices, torch.Tensor) or indices.device.type != 'cpu' or indices.dtype != torch.long:
            raise ValueError('CPU int64 consumed indices required')
        if indices.ndim != 1 or not indices.numel():
            raise ValueError('nonempty one-dimensional consumed indices required')
        if (indices < 0).any().item() or (indices >= self.total).any().item():
            raise ValueError('consumed index out of bounds')
        if indices.unique().numel() != indices.numel() or self._position_counts[indices].any().item():
            raise ValueError('sampler position consumed more than once')
        # Validate the entire batch before mutation: failures cannot half-count it.
        self._position_counts[indices] = 1

    def state(self):
        return dict(schema=SCHEMA, context_sha256=self.context_sha256,
                    position_counts=self._position_counts.clone())

    def validate_contexts(self, contexts):
        if not isinstance(contexts, (list, tuple)) or not contexts or any(x != self.context_sha256 for x in contexts):
            raise ValueError('rank exposure contexts differ')

    def _state_counts(self, state):
        if not isinstance(state, dict) or set(state) != {'schema', 'context_sha256', 'position_counts'}:
            raise ValueError('invalid exposure state')
        if state['schema'] != SCHEMA or state['context_sha256'] != self.context_sha256:
            raise ValueError('foreign exposure state')
        return _counts(state['position_counts'], self.total)

    def merge_states(self, states):
        if not isinstance(states, (list, tuple)) or not states:
            raise ValueError('rank states required')
        merged = torch.zeros(self.total, dtype=torch.long)
        for state in states:
            merged += self._state_counts(state)
        _counts(merged, self.total)  # Cross-rank duplicate positions are invalid.
        return dict(schema=SCHEMA, context_sha256=self.context_sha256, position_counts=merged)

    def source_counts(self, state=None):
        counts = self._position_counts if state is None else self._state_counts(state)
        teacher = torch.zeros(self.teacher_count, dtype=torch.long)
        teacher.index_add_(0, self._teacher_sources, counts[self.base_count:])
        return counts[:self.base_count].clone(), teacher

    def summarize(self, state=None):
        counts = self._position_counts if state is None else self._state_counts(state)
        base, teacher = self.source_counts(state)
        by_episode = torch.zeros(len(self.episodes), dtype=torch.long)
        by_episode.index_add_(0, torch.from_numpy(self.episode_index), teacher)
        groups = Counter()
        per_episode = []
        for ep, n in zip(self.episodes, by_episode.tolist()):
            groups[ep['task'] + ':' + ep['teacher']] += n
            per_episode.append(dict(ep, exposures=n))
        hard = torch.from_numpy(self.hard)
        early = torch.from_numpy(self.early)
        total = int(counts.sum())
        return dict(schema=SCHEMA, context_sha256=self.context_sha256,
            plan_sha256=self.mix.plan_sha256, pre_ddp_total=self.total,
            actual_total=total, actual_base=int(base.sum()), actual_teacher=int(teacher.sum()),
            unconsumed_positions=self.total-total, groups=dict(groups),
            hard_teacher_exposures=int(teacher[hard].sum()),
            early_teacher_exposures=int(teacher[early].sum()),
            hard_early_teacher_exposures=int(teacher[hard & early].sum()),
            base_window_exposure_histogram=dict(Counter(base.tolist())),
            per_window_exposure_histogram=dict(Counter(teacher.tolist())),
            per_episode=per_episode,
            scope='Recorded DataLoader consumption; not optimizer-step or performance evidence')

    def finalize(self, *, state=None, world=8, batch=2, seed=42, epoch=1):
        """Require complete exact DDP+DataLoader consumption, including tail drops."""
        if not positive_int(world) or not positive_int(batch):
            raise ValueError('positive world/batch required')
        if type(seed) is not int or type(epoch) is not int or seed < 0 or epoch < 0:
            raise ValueError('nonnegative integer seed/epoch required')
        counts = self._position_counts if state is None else self._state_counts(state)
        expected_positions = torch.zeros(self.total, dtype=torch.long)
        for rank in range(world):
            sampler = DistributedSampler(self.mix, num_replicas=world, rank=rank,
                shuffle=True, seed=seed, drop_last=True)
            sampler.set_epoch(epoch)
            ids = list(sampler)
            ids = ids[:len(ids)//batch*batch]
            expected_positions[ids] += 1
        _counts(expected_positions, self.total)
        if not torch.equal(counts, expected_positions):
            raise ValueError('actual consumption differs from expected sampler positions')
        expected, expected_teacher = simulate_exposure(self.mix, world=world,
            batch=batch, seed=seed, epoch=epoch)
        report = self.summarize(state)
        _, teacher = self.source_counts(state)
        if not np.array_equal(teacher.numpy(), expected_teacher):
            raise ValueError('teacher window counts differ from simulation')
        for key in ('actual_total', 'actual_base', 'actual_teacher', 'pre_ddp_total'):
            if report[key] != expected[key]:
                raise ValueError('exposure total differs from simulation: ' + key)
        if report['unconsumed_positions'] != expected['dropped']:
            raise ValueError('tail-drop mismatch')
        report.update(status='ACTUAL_EXPOSURE_MATCHES_SIMULATION', dropped=expected['dropped'],
            world_size=world, batch_size=batch, seed=seed, epoch=epoch,
            simulation_ranks=expected['ranks'])
        return report
