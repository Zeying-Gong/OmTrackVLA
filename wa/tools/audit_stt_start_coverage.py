"""Audit actual timestamp coverage without loading images or changing data."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def window_ages(times, history, rows):
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or len(times) == 0 or not np.isfinite(times).all() or not (np.diff(times) > 0).all():
        raise ValueError('nonfinite or nonmonotonic observation times')
    current = np.asarray(history[rows, -1])
    if not np.issubdtype(current.dtype, np.integer) or (current < 0).any() or (current >= len(times)).any():
        raise ValueError('invalid current observation index')
    return times[current] - times[0]

def validate_row_indices(history, episode, valid, episode_count):
    if history.ndim != 2 or history.shape[1] < 1 or episode.ndim != 1 or len(history) != len(episode):
        raise ValueError('history/episode shape mismatch')
    if not np.issubdtype(episode.dtype, np.integer) or (episode < 0).any() or (episode >= episode_count).any():
        raise ValueError('invalid episode identity')
    if valid.ndim != 1 or not np.issubdtype(valid.dtype, np.integer):
        raise ValueError('one-dimensional integer valid indices required')
    if (valid < 0).any() or (valid >= len(episode)).any() or not (np.diff(valid) > 0).all():
        raise ValueError('valid indices must be sorted unique and in range')
    if not (np.diff(episode) >= 0).all():
        raise ValueError('episode row order changed')
    return episode[valid]

def group_summary(records):
    return dict(episodes=len(records), valid_windows=sum(r['valid'] for r in records),
        early_windows=sum(r['early_valid'] for r in records),
        early_candidate=sum(r['early_candidate'] for r in records),
        zero_early=[r['key'] for r in records if not r['early_valid']],
        no_window_before_05s=sum(r['early05_valid'] == 0 for r in records),
        no_window_before_1s=sum(r['early1_valid'] == 0 for r in records))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'student-root', 'cache-sha256', 'checkpoint-sha256',
                 'index-audit-sha256', 'episode-audit-sha256', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--checkpoint-step', type=int, required=True)
    a = p.parse_args()
    cache, root, output = Path(a.cache), Path(a.student_root), Path(a.output)
    if output.exists(): raise ValueError('refusing to overwrite coverage report')
    hashes = {}
    def verify(path, expected=None):
        actual = sha(path)
        if expected is not None and actual != expected: raise ValueError('changed source: ' + str(path))
        hashes[str(path)] = actual
    verify(cache/'complete.json', a.cache_sha256)
    complete = json.loads((cache/'complete.json').read_text())
    verify(cache/'index/audit.json', a.index_audit_sha256)
    audit = json.loads((cache/'index/audit.json').read_text())
    if audit['cache_complete_sha256'] != a.cache_sha256: raise ValueError('cache/index mismatch')
    for name in ('train_history.npy', 'train_episode.npy', 'train_episodes.json'):
        verify(cache/name, complete['files'][name])
    verify(cache/'index/train_valid.npy', audit['splits']['train']['index_sha256'])
    verify(cache/'index/train_episodes_audit.json', a.episode_audit_sha256)
    verify(root/'PARTITION_COMPLETE.json')
    partition = json.loads((root/'PARTITION_COMPLETE.json').read_text())
    if partition['task'] != 'stt' or partition['episodes'] != 1405 or partition['status'] != 'COMPLETE_STUDENT_TASK_PARTITION_NOT_GLOBAL_AUDIT':
        raise ValueError('complete STT partition required')
    verify(root/'combined_episodes.jsonl', partition['combined_sha256'])
    rows = [json.loads(s) for s in (root/'combined_episodes.jsonl').read_text().splitlines() if s]
    student = {r['key']: r for r in rows}
    if len(rows) != 1405 or len(student) != 1405: raise ValueError('duplicate/missing student rows')
    if any(r['task'] != 'stt' or r['checkpoint_sha256'] != a.checkpoint_sha256 or r['checkpoint_step'] != a.checkpoint_step for r in rows):
        raise ValueError('foreign student contract')
    entries = json.loads((cache/'train_episodes.json').read_text())
    sources = {x['episode']: x['source_sha256'] for x in json.loads((cache/'index/train_episodes_audit.json').read_text())['stats']}
    history = np.load(cache/'train_history.npy', mmap_mode='r')
    episode = np.load(cache/'train_episode.npy', mmap_mode='r')
    valid = np.load(cache/'index/train_valid.npy')
    valid_episode = validate_row_indices(history, episode, valid, len(entries))
    records = []
    for ep, entry in enumerate(entries):
        if entry['task'] != 'stt': continue
        key = entry['episode_uid'].split(':', 1)[1]
        outcome = student[key]
        obs_path = Path(entry['root'])/'observations.json'
        verify(obs_path, sources[ep]['observations.json'])
        times = [x['timestamp_s'] for x in json.loads(obs_path.read_text())]
        left, right = np.searchsorted(episode, [ep, ep + 1])
        vl, vr = np.searchsorted(valid_episode, [ep, ep + 1])
        raw, kept = np.arange(left, right), valid[vl:vr]
        raw_age, age = window_ages(times, history, raw), window_ages(times, history, kept)
        records.append(dict(key=key, teacher=entry['teacher'],
            student_success=bool(outcome['success']), status=outcome['status'],
            student_steps=outcome['total_step'], valid=len(kept),
            early_candidate=int((raw_age <= 2).sum()), early_valid=int((age <= 2).sum()),
            first_valid_s=float(age.min()) if len(age) else None,
            early05_valid=int((age <= .5).sum()), early1_valid=int((age <= 1).sum())))
        if len(records) % 200 == 0: print(json.dumps(dict(episodes_checked=len(records))), flush=True)
    if len({r['key'] for r in records}) != len(records): raise ValueError('duplicate STT demonstration')
    hard = [r for r in records if not r['student_success']]
    # This group is defined by stored step count, not a claim of measured student duration.
    early_keys = {k for k, r in student.items() if r['status'] == 'Collision' and r['total_step'] <= 40}
    groups = dict(all_STT_with_demo=records,
        successful_STT_with_demo=[r for r in records if r['student_success']],
        failed_STT_with_demo=hard, failed_STT_nonzero=[r for r in hard if r['valid']],
        collision_within40steps_with_demo=[r for r in records if r['key'] in early_keys])
    report = dict(status='COVERAGE_AUDIT_PASS', checkpoint_sha256=a.checkpoint_sha256,
        checkpoint_step=a.checkpoint_step, source_hashes=hashes, tool_sha256=sha(__file__),
        groups={name: group_summary(group) for name, group in groups.items()},
        collision_within40steps_without_demo=sorted(early_keys-{r['key'] for r in records}),
        hard_details=hard, scope='Cached selected teacher coverage only; not fitting or closed-loop gain',
        early_definition='teacher observations timestamp_s minus first timestamp_s <=2 seconds')
    for path, expected in hashes.items():
        if sha(path) != expected: raise ValueError('source changed during audit: ' + path)
    with output.open('x') as f: json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(dict(status=report['status'], groups=report['groups'], output=str(output))), flush=True)

if __name__ == '__main__': main()
