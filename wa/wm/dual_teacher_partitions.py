"""Deterministic disjoint work lists, not permission to submit or reuse live rows.
Call only after validating a frozen resume plan. Runtime/merge integration is separate.
"""
from collections import Counter

def key(row):
    if set(row) != {'task', 'key'} or row['task'] not in ('stt', 'dt', 'at'):
        raise ValueError('invalid work item')
    if not isinstance(row['key'], str) or not row['key']:
        raise ValueError('invalid episode key')
    return row['task'], row['key']

def validate(remaining, groups):
    expected = [key(r) for r in remaining]
    if len(set(expected)) != len(expected):
        raise ValueError('duplicate remainder')
    if not groups or len(groups) > 8:
        raise ValueError('invalid group count')
    observed = []
    for group in groups:
        if len(group) != 8:
            raise ValueError('each group requires eight lanes')
        observed.extend(key(r) for lane in group for r in lane)
    if len(set(observed)) != len(observed):
        raise ValueError('duplicate assignment')
    if Counter(observed) != Counter(expected):
        raise ValueError('missing or unexpected assignment')
    return True

def partition(remaining, count):
    if type(count) is not int or not 1 <= count <= 8:
        raise ValueError('group count must be integer1..8')
    keys = [key(r) for r in remaining]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate remainder')
    if len(keys) < count * 8:
        raise ValueError('do not allocate idle GPU lanes')
    # Sort independently of file append order; task order preserved for monitoring.
    order = {'stt': 0, 'dt': 1, 'at': 2}
    rows = [dict(task=t, key=k) for t, k in sorted(keys, key=lambda x:(order[x[0]], x[1]))]
    groups = []
    for i in range(count):
        assigned = rows[i::count]
        groups.append([assigned[j::8] for j in range(8)])
    validate(remaining, groups)
    return groups
