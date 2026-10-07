"""Explicit v2 teacher-window schedule; v1 plans remain independently validated.

Extra entries are sampler positions, not distinct teacher windows. Only the
runtime admission layer may supply the independently derived expected multiset
and group whitelist. No sampling metadata is returned as a model input.
"""
from collections import Counter
import re

from torch.utils.data import Dataset

from wa.wm.teacher_window_plan import SOURCE_KEYS, canonical_sha, positive_int

SCHEMA = 'wa.teacher-window-schedule.v2'
SELECTION_SOURCE_KEYS = frozenset(('old_candidate_report', 'candidate_rows',
    'candidate_outcomes', 'fixed88_replay'))
GROUPS = ('hard_stt_early', 'hard_stt_late', 'new_regression_stt',
          'successful_stt_anchor')


def _hashes(value, keys, label):
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError('all ' + label + ' hashes required')
    if any(not isinstance(h, str) or not re.fullmatch('[0-9a-f]{64}', h)
           for h in value.values()):
        raise ValueError('invalid ' + label + ' hash')


def _indices(value, teacher_count, *, unique=False):
    if not isinstance(value, (list, tuple)) or any(type(i) is not int for i in value):
        raise ValueError('integer teacher indices required')
    if any(i < 0 or i >= teacher_count for i in value):
        raise ValueError('teacher index out of bounds')
    result = list(value)
    if result != sorted(result) or (unique and len(result) != len(set(result))):
        raise ValueError('teacher indices must be sorted' + (' and unique' if unique else ''))
    return tuple(result)


def validate_schedule(plan, *, base_count, teacher_count, source_hashes,
                      selection_source_hashes, expected_extra_teacher_indices,
                      sampling_groups):
    required = {'schema', 'base_count', 'teacher_count', 'extra_teacher_indices',
                'source_hashes', 'selection_source_hashes'}
    if not isinstance(plan, dict) or set(plan) != required or plan['schema'] != SCHEMA:
        raise ValueError('unknown schedule schema/fields')
    if not positive_int(base_count) or not positive_int(teacher_count):
        raise ValueError('positive live source counts required')
    if (not positive_int(plan['base_count']) or not positive_int(plan['teacher_count'])
            or (plan['base_count'], plan['teacher_count']) != (base_count, teacher_count)):
        raise ValueError('source count mismatch')
    for actual, expected, keys, label in (
            (plan['source_hashes'], source_hashes, SOURCE_KEYS, 'source'),
            (plan['selection_source_hashes'], selection_source_hashes,
             SELECTION_SOURCE_KEYS, 'selection source')):
        _hashes(actual, keys, label); _hashes(expected, keys, label)
        if actual != expected: raise ValueError(label + ' hash mismatch')
    if not isinstance(plan['extra_teacher_indices'], list):
        raise ValueError('explicit JSON multiset list required')
    extra = _indices(plan['extra_teacher_indices'], teacher_count)
    expected = _indices(expected_extra_teacher_indices, teacher_count)
    if not extra or max(Counter(extra).values()) > 2 or extra != expected:
        raise ValueError('schedule differs from exact bounded eligible multiset')
    if not isinstance(sampling_groups, dict) or set(sampling_groups) != set(GROUPS):
        raise ValueError('all independently derived sampling groups required')
    groups = {name: _indices(sampling_groups[name], teacher_count, unique=True) for name in GROUPS}
    flat = [i for indices in groups.values() for i in indices]
    if len(flat) != len(set(flat)):
        raise ValueError('sampling groups overlap')
    group_extra = sorted(i for name, indices in groups.items() for i in indices
                         for _ in range(2 if name == 'hard_stt_early' else 1))
    if tuple(group_extra) != extra:
        raise ValueError('group qualifications do not reproduce the multiset')
    return extra, groups


class ScheduledTeacherMix(Dataset):
    """Each base/teacher once followed by the v2 explicit extra-position map."""
    def __init__(self, base, teacher, plan, *, source_hashes,
                 selection_source_hashes, expected_extra_teacher_indices,
                 sampling_groups):
        self.extra_positions, self.sampling_groups = validate_schedule(
            plan, base_count=len(base), teacher_count=len(teacher),
            source_hashes=source_hashes, selection_source_hashes=selection_source_hashes,
            expected_extra_teacher_indices=expected_extra_teacher_indices,
            sampling_groups=sampling_groups)
        self.base, self.teacher = base, teacher
        self.plan_sha256 = canonical_sha(plan)
        self.schema = SCHEMA

    def __len__(self):
        return len(self.base) + len(self.teacher) + len(self.extra_positions)

    def locate(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        if index < len(self.base): return 'base', index
        index -= len(self.base)
        if index < len(self.teacher): return 'teacher', index
        return 'teacher', self.extra_positions[index - len(self.teacher)]

    def __getitem__(self, index):
        source, index = self.locate(index)
        return (self.base if source == 'base' else self.teacher)[index]
