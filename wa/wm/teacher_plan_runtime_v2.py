"""Strict opt-in admission for the STT anchor v2 schedule, without model IO."""
from collections import Counter
import json
from pathlib import Path
import numpy as np

from wa.tools import build_stt_anchor_candidate as recipe
from wa.wm.loaders import sha
from wa.wm.teacher_window_plan import canonical_sha, simulate_exposure
from wa.wm.teacher_window_schedule import ScheduledTeacherMix, SCHEMA

PARENT_PATH = '/data/nas_ray/project/md-ak/users/zeying.gong/job_59866/task_70728/wa_history_repeat_v1/checkpoint.pt'
PARENT_SHA = 'ab39144b0490937ce43c4ab28e2d90cdb0c560f700d3364a29ce1dc9a08b999d'
BUDGET = dict(base_once=726631, teacher_once=436816, hard_extra=13683,
    regression_extra=886, balanced_anchor_extra=6257, all_extra=20826,
    pre_ddp=1184273, simulated_post_drop=1184272)
TRAINING_RECIPE = dict(parent_checkpoint=PARENT_PATH, parent_sha256=PARENT_SHA,
    model_and_optimizer=True, new_epochs=1, cumulative_epochs=2, world_size=8,
    seed=42, batch_size=2, accumulation=2, history_repeat_probability=.25,
    model_loss_controller_physics='unchanged; resumed parent LR, no LR increase')


def exact(actual, expected, label):
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(label)


def exact_dict(actual, expected, label):
    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise ValueError(label + ' fields')
    for key, value in expected.items(): exact(actual[key], value, label + ': ' + key)


def selection_paths(root):
    artifacts = Path(root).resolve() / 'artifacts'
    return dict(
        old_candidate_report=artifacts/'hard_stt_candidate_61377_20261007_v1/report.json',
        candidate_rows=artifacts/'student61609_full_audit_20261007_v1/combined_episodes.jsonl',
        candidate_outcomes=artifacts/'student61609_hard_outcomes_20261007_v1/hard_stt_outcomes.json',
        fixed88_replay=artifacts/'teacher_group_fit_61609_replay_20261007_v1.json')


def derive_schedule(records, windows, candidate):
    """Reconstruct eligibility and multiplicities, never accept caller group lists."""
    extras, allocation = recipe.allocate(records, windows, candidate)
    pieces = allocation['pieces']
    exact_dict({k: len(v) for k, v in pieces.items()},
        dict(hard_once=10413, hard_early_once=3270, regression_once=886,
             balanced_anchor_once=6257), 'fixed extra pieces')
    for name, count in (('regression_episode_indices', 10), ('anchor_episode_indices', 1248),
                       ('original_gain_episode_indices', 15), ('original_remaining_episode_indices', 78)):
        exact(len(allocation[name]), count, 'fixed cohort: ' + name)
    if Counter(x['requested'] for x in allocation['anchor_details']) != {5: 1231, 6: 17}:
        raise ValueError('anchor per-episode allocation changed')
    groups = dict(hard_stt_early=sorted(pieces['hard_early_once']),
        hard_stt_late=sorted(set(pieces['hard_once']) - set(pieces['hard_early_once'])),
        new_regression_stt=sorted(pieces['regression_once']),
        successful_stt_anchor=sorted(pieces['balanced_anchor_once']))
    exact(len(extras), BUDGET['all_extra'], 'fixed extra budget')
    return extras, groups, allocation


def verify_base_source(base, base_index):
    """Bind live base rows/cache to the unchanged fully audited valid split."""
    base_index = Path(base_index)
    if sha(base_index/'audit.json') != recipe.old.PINNED['base_index_audit']:
        raise ValueError('base audit source changed')
    audit = json.loads((base_index/'audit.json').read_text())
    if (getattr(base, 'part', None) != 'train' or getattr(base, 'data_root', None) is not None
            or getattr(base, 'source_prefix', None) is not None):
        raise ValueError('v2 requires the original unredirected base train split')
    if sha(Path(base.cache)/'complete.json') != audit['cache_complete_sha256']:
        raise ValueError('base cache does not match the pinned valid-index audit')
    with (base_index/'train_valid.npy').open('rb') as stream:
        valid = np.load(stream, allow_pickle=False)
    rows = np.asarray(base.rows)
    if (rows.ndim != 1 or rows.dtype.kind not in 'iu'
            or not np.array_equal(rows, valid) or len(base) != len(valid)):
        raise ValueError('live base dataset is not the full valid source')


def validate_training_recipe(report, arguments, *, world):
    """Selection checkpoint61609 must never become the initialization checkpoint."""
    exact_dict(report.get('training_recipe'), TRAINING_RECIPE, 'candidate initialization recipe')
    expected = dict(resume=PARENT_PATH, resume_sha256=PARENT_SHA, completed_epochs=1,
        epochs=1, seed=42, batch_size=2, accumulation=2, history_repeat_probability=.25,
        dual_teacher_repeats=1, evaluation_set_adaptation=True, kind='jepa', world_weight=.1)
    for key, value in expected.items():
        exact(getattr(arguments, key, None), value, 'actual v2 training recipe: ' + key)
    if any(getattr(arguments, key, None) is not None for key in
           ('recovery_cache', 'recovery_index', 'data_root', 'source_prefix')):
        raise ValueError('v2 excludes legacy recovery data and source redirection')
    if type(world) is not int or world < 1 or (not arguments.diagnostic and world != 8):
        raise ValueError('v2 formal training requires eight ranks')


def load_v2(directory, report, base, teacher, *, base_index, student_rows,
            teacher_selections, project_root):
    from wa.wm.teacher_plan_runtime import load_candidate, verify_metadata
    if project_root is None: raise ValueError('explicit project root required for v2 evidence')
    root = Path(project_root).resolve()
    directory = Path(directory).resolve()
    if directory.parent != root/'artifacts':
        raise ValueError('v2 candidate must be a direct project artifacts child')
    for key, expected in dict(status='CANDIDATE_ONLY_NOT_TRAINING_RELEASE',
            candidate_kind=recipe.KIND, runtime_integration_required=True,
            experiment=recipe.old.EXPERIMENT, checkpoint_sha256=recipe.old.CHECKPOINT_SHA,
            checkpoint_step=59065, candidate_checkpoint_sha256=recipe.CHECKPOINT_SHA,
            candidate_checkpoint_step=59716).items():
        exact(report.get(key), expected, 'v2 report identity: ' + key)
    exact_dict(report.get('budget'), BUDGET, 'v2 budget')
    exact_dict(report.get('training_recipe'), TRAINING_RECIPE, 'v2 parent recipe')
    if report.get('tool_sha256') != sha(recipe.__file__):
        raise ValueError('candidate selection implementation changed')
    exact_dict(report.get('selection_source_hashes'), recipe.SOURCE_SHA, 'selection hashes')
    artifacts = report.get('artifacts')
    names = {'plan.json', 'episodes.json', 'teacher_windows.npz', 'exposure.json', 'simulated_counts.npz'}
    if not isinstance(artifacts, dict) or set(artifacts) != names:
        raise ValueError('v2 candidate artifact inventory mismatch')
    for name, expected in artifacts.items():
        if sha(directory/name) != expected: raise ValueError('candidate artifact changed: ' + name)
    paths = selection_paths(root)
    if not isinstance(report.get('source_hashes'), dict):
        raise ValueError('source inventory required')
    original_paths = dict(cache_complete=Path(teacher.cache)/'complete.json',
        teacher_index=Path(teacher.cache)/'index/train_valid.npy',
        base_index=Path(base_index)/'train_valid.npy', student_rows=Path(student_rows),
        teacher_selections=Path(teacher_selections))
    for key, path in original_paths.items():
        if report['source_hashes'].get(str(path.resolve())) != recipe.old.PINNED[key]:
            raise ValueError('new report omitted or changed original source: ' + key)
    verify_base_source(base, base_index)
    for key, path in paths.items():
        actual = sha(path)
        if actual != recipe.SOURCE_SHA[key] or report['source_hashes'].get(str(path.resolve())) != actual:
            raise ValueError('selection source changed: ' + key)
    # Reuse strict immutable-cache and original-admission validation, not v1 extras.
    _, original_records, original_windows, _ = load_candidate(
        paths['old_candidate_report'].parent, recipe.SOURCE_SHA['old_candidate_report'],
        base, teacher, base_index=base_index, student_rows=student_rows,
        teacher_selections=teacher_selections)
    records = json.loads((directory/'episodes.json').read_text())
    with np.load(directory/'teacher_windows.npz', allow_pickle=False) as z:
        windows = {k: z[k] for k in z.files}
    if records != original_records or set(windows) != set(original_windows):
        raise ValueError('v2 changed original episode/window metadata')
    if any(not np.array_equal(windows[k], original_windows[k]) for k in windows):
        raise ValueError('v2 changed original window identities/timestamps')
    verify_metadata(records, windows, original_records, teacher)
    candidate_rows = recipe.read_rows(paths['candidate_rows'])
    recipe.validate_student_rows(candidate_rows,
        dict(checkpoint_sha=recipe.CHECKPOINT_SHA, step=59716))
    candidate = recipe.unique_rows(candidate_rows)
    outcome = json.loads(paths['candidate_outcomes'].read_text())
    for key, expected in dict(audit_status='PASS', episodes=4215,
            candidate_checkpoint_sha256=recipe.CHECKPOINT_SHA,
            candidate_checkpoint_step=59716).items():
        exact(outcome.get(key), expected, 'candidate outcome evidence: ' + key)
    replay = json.loads(paths['fixed88_replay'].read_text())
    exact(replay.get('status'), 'FIXED88_CANDIDATE_LABEL_FIT_ONLY', 'fixed88 status')
    exact(replay.get('candidate', {}).get('sha256'), recipe.CHECKPOINT_SHA, 'fixed88 candidate SHA')
    exact(replay.get('candidate', {}).get('step'), 59716, 'fixed88 candidate step')
    extras, groups, allocation = derive_schedule(records, windows, candidate)
    if (report.get('anchor_selection') != allocation['anchor_details']
            or report.get('regression_episode_uids') !=
                [records[i]['episode_uid'] for i in allocation['regression_episode_indices']]):
        raise ValueError('stored candidate qualifications differ from derivation')
    if report.get('all12_regression_keys') != outcome['groups']['regressions']['by_task']['stt']['keys']:
        raise ValueError('regression outcome inventory changed')
    plan = json.loads((directory/'plan.json').read_text())
    if canonical_sha(plan) != report.get('plan_canonical_sha256'):
        raise ValueError('v2 plan canonical hash mismatch')
    exact(len(base), BUDGET['base_once'], 'full base data required')
    exact(len(teacher), BUDGET['teacher_once'], 'full valid teacher data required')
    source_hashes = {key: recipe.old.PINNED[key] for key in recipe.SOURCE_KEYS}
    mix = ScheduledTeacherMix(base, teacher, plan, source_hashes=source_hashes,
        selection_source_hashes=recipe.SOURCE_SHA, expected_extra_teacher_indices=extras,
        sampling_groups=groups)
    # Verify the emitted candidate simulation against the actual runtime mapping.
    simulated, counts = simulate_exposure(mix, world=8, batch=2, seed=42, epoch=1)
    exposure = json.loads((directory/'exposure.json').read_text())
    normalize = lambda value: json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    for key, expected in simulated.items():
        if key != 'scope' and normalize(exposure.get(key)) != normalize(expected):
            raise ValueError('v2 runtime simulation differs: ' + key)
    with np.load(directory/'simulated_counts.npz', allow_pickle=False) as z:
        if set(z.files) != {'planned_teacher_counts', 'simulated_teacher_counts',
                'extra_counts', 'simulated_base_counts', 'rank_teacher_counts'}:
            raise ValueError('v2 simulation array inventory changed')
        shapes = dict(planned_teacher_counts=(len(teacher),),
            simulated_teacher_counts=(len(teacher),), extra_counts=(len(teacher),),
            simulated_base_counts=(len(base),), rank_teacher_counts=(8, len(teacher)))
        if any(z[key].dtype != np.int64 or z[key].shape != shape or (z[key] < 0).any()
               for key, shape in shapes.items()):
            raise ValueError('invalid simulation array shape/type/count')
        if (not np.array_equal(z['rank_teacher_counts'].sum(0), counts)
                or z['rank_teacher_counts'].sum(1).tolist() != [r['teacher'] for r in simulated['ranks']]
                or not (z['simulated_base_counts'] <= 1).all()
                or int(z['simulated_base_counts'].sum()) != simulated['actual_base']):
            raise ValueError('rank/base simulation counts are inconsistent')
        extra_counts = np.bincount(extras, minlength=len(teacher))
        if (not np.array_equal(z['simulated_teacher_counts'], counts)
                or not np.array_equal(z['extra_counts'], extra_counts)
                or not np.array_equal(z['planned_teacher_counts'], 1 + extra_counts)):
            raise ValueError('v2 per-window simulation differs from runtime')
    for key, path in paths.items():
        if sha(path) != recipe.SOURCE_SHA[key]: raise ValueError('selection changed during admission')
    return mix, records, windows, report


def diagnostic_positions_v2(mix, records, windows, count):
    if type(count) is not int or count < 12 or count % 2:
        raise ValueError('v2 diagnostic requires an even count >=12')
    positions = [0, len(mix.base)-1]
    for task in ('stt', 'dt', 'at'):
        candidates = [i for i, ep in enumerate(windows['episode_index'])
                      if records[int(ep)]['task'] == task and not windows['hard'][i]]
        if not candidates: raise ValueError('missing diagnostic task')
        positions.append(len(mix.base) + candidates[len(candidates)//2])
    offset = len(mix.base) + len(mix.teacher)
    extra = np.asarray(mix.extra_positions)
    for name, indices in mix.sampling_groups.items():
        if not indices: raise ValueError('missing v2 diagnostic group: ' + name)
        index = indices[0]
        positions.append(len(mix.base) + index)
        positions.extend((offset + np.flatnonzero(extra == index)).tolist())
    positions = list(dict.fromkeys(positions))
    if len(positions) > count: raise ValueError('diagnostic count cannot cover v2 sources')
    for pos in np.linspace(0, len(mix)-1, count*2, dtype=int).tolist():
        if len(positions) == count: break
        if pos not in positions: positions.append(pos)
    if len(positions) != count or len(set(positions)) != count or max(positions) >= len(mix):
        raise ValueError('invalid v2 diagnostic positions')
    return positions
