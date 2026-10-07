"""Verify frozen hard-STT candidate at opt-in training startup."""
import json
from pathlib import Path
import numpy as np
from wa.wm.loaders import sha
from wa.tools.build_hard_stt_plan import PINNED, CHECKPOINT_SHA, derive_admissions
from wa.wm.teacher_window_plan import PlannedTeacherMix, canonical_sha
from wa.wm.dual_teacher_selection import EXPERIMENT

def verify_metadata(records, windows, expected_records, teacher):
    required = {'dataset_index', 'raw_row', 'episode_index', 'current_observation_index',
                'age_s', 'hard', 'early'}
    if set(windows) != required or len(records) != len(expected_records):
        raise ValueError('metadata schema/count mismatch')
    n = len(teacher)
    if any(v.ndim != 1 or len(v) != n for v in windows.values()):
        raise ValueError('window metadata shape mismatch')
    for key in ('dataset_index', 'raw_row', 'episode_index', 'current_observation_index'):
        if windows[key].dtype.kind not in 'iu': raise ValueError('integer metadata required')
    if windows['hard'].dtype.kind != 'b' or windows['early'].dtype.kind != 'b':
        raise ValueError('boolean group metadata required')
    ep = np.asarray(teacher.episode)[teacher.rows]
    expected = dict(dataset_index=np.arange(n), raw_row=teacher.rows, episode_index=ep,
        current_observation_index=np.asarray(teacher.history)[teacher.rows, -1])
    for key, value in expected.items():
        if not np.array_equal(windows[key], value): raise ValueError('changed window identity: ' + key)
    if not np.isfinite(windows['age_s']).all() or (windows['age_s'] < 0).any():
        raise ValueError('invalid time metadata')
    if not np.array_equal(windows['early'], windows['age_s'] <= 2):
        raise ValueError('early-window definition changed')
    for record, expected_record in zip(records, expected_records):
        if any(type(record.get(k)) is not type(v) or record.get(k) != v
               for k, v in expected_record.items()):
            raise ValueError('episode qualification changed')
    hard = np.array([expected_records[int(e)]['hard'] for e in ep], dtype=bool)
    if not np.array_equal(hard, windows['hard']):
        raise ValueError('hard grouping changed')
    counts = np.bincount(ep, minlength=len(records))
    early_counts = np.bincount(ep[windows['early']], minlength=len(records))
    for i, record in enumerate(records):
        if any(type(record.get(k)) is not int for k in ('valid_windows', 'early_windows')):
            raise ValueError('integer per-episode metadata count required')
        if record['valid_windows'] != counts[i] or record['early_windows'] != early_counts[i]:
            raise ValueError('per-episode metadata count changed')
    return np.flatnonzero(hard).tolist()

def load_candidate(directory, report_sha256, base, teacher, *, base_index,
                   student_rows, teacher_selections):
    directory = Path(directory)
    if sha(directory/'report.json') != report_sha256:
        raise ValueError('candidate report hash mismatch')
    report = json.loads((directory/'report.json').read_text())
    if report['status'] != 'CANDIDATE_ONLY_NOT_TRAINING_RELEASE' or report['experiment'] != EXPERIMENT:
        raise ValueError('unexpected candidate protocol')
    if report['checkpoint_sha256'] != CHECKPOINT_SHA or report['checkpoint_step'] != 59065:
        raise ValueError('wrong student failure source')
    artifacts = report['artifacts']
    if set(artifacts) != {'plan.json', 'episodes.json', 'teacher_windows.npz', 'exposure.json'}:
        raise ValueError('candidate artifact inventory mismatch')
    for name, expected in artifacts.items():
        if sha(directory/name) != expected: raise ValueError('candidate artifact changed: ' + name)
    # DualTeacherData has already validated the immutable cache/release/image inventory.
    cache = Path(teacher.cache)
    paths = dict(cache_complete=cache/'complete.json', teacher_index=cache/'index/train_valid.npy',
        base_index=Path(base_index)/'train_valid.npy', student_rows=Path(student_rows),
        teacher_selections=Path(teacher_selections))
    source_hashes = {}
    for key, path in paths.items():
        actual = sha(path)
        if actual != PINNED[key] or report['source_hashes'].get(str(path.resolve())) != actual:
            raise ValueError('candidate source changed: ' + key)
        source_hashes[key] = actual
    students = [json.loads(s) for s in Path(student_rows).read_text().splitlines() if s]
    selections = [json.loads(s) for s in Path(teacher_selections).read_text().splitlines() if s]
    complete = json.loads((cache/'complete.json').read_text())
    if sha(complete['source_release']) != PINNED['teacher_release']:
        raise ValueError('release changed')
    release = json.loads(Path(complete['source_release']).read_text())
    expected_records, _ = derive_admissions(students, selections,
        release['teacher_demonstrations'], teacher.episodes)
    plan = json.loads((directory/'plan.json').read_text())
    if canonical_sha(plan) != report['plan_canonical_sha256']:
        raise ValueError('candidate plan canonical hash mismatch')
    records = json.loads((directory/'episodes.json').read_text())
    with np.load(directory/'teacher_windows.npz', allow_pickle=False) as z:
        windows = {k: z[k] for k in z.files}
    extras = verify_metadata(records, windows, expected_records, teacher)
    if plan['extra_teacher_indices'] != extras or len(extras) != report['eligible_teacher_windows']:
        raise ValueError('plan does not exactly match admitted hard windows')
    mix = PlannedTeacherMix(base, teacher, plan, source_hashes=source_hashes,
        eligible_teacher_indices=extras)
    return mix, records, windows, report

def diagnostic_positions(mix, records, windows, count=16):
    """Bounded developer check intentionally covers all sources plus extra positions."""
    if type(count) is not int or count < 8 or count % 2:
        raise ValueError('even integer diagnostic count >=8 required')
    positions = [0, len(mix.base)-1]
    for task in ('stt', 'dt', 'at'):
        candidates = [i for i, ep in enumerate(windows['episode_index'])
                      if records[int(ep)]['task'] == task and not windows['hard'][i]]
        if not candidates: raise ValueError('missing diagnostic task')
        positions.append(len(mix.base) + candidates[len(candidates)//2])
    hard = np.flatnonzero(windows['hard'])
    if len(hard) < 2: raise ValueError('missing hard diagnostic windows')
    positions.extend([len(mix.base)+int(hard[0]), len(mix.base)+int(hard[-1]),
        len(mix.base)+len(mix.teacher)])
    for pos in np.linspace(0, len(mix)-1, count*2, dtype=int).tolist():
        if pos not in positions: positions.append(pos)
        if len(positions) == count: break
    if len(set(positions)) != count or max(positions) >= len(mix):
        raise ValueError('invalid diagnostic positions')
    return positions
