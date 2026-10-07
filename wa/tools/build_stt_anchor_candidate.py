"""Build exactly one CPU-only STT anchor candidate; never train or change inputs.

The old runtime intentionally rejects this v2 schedule. A separate reviewed
runtime integration is REQUIRED before any formal training release.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from torch.utils.data import DistributedSampler

from wa.tools import build_hard_stt_plan as old
from wa.tools import audit_hard_stt_outcomes as outcomes
from wa.tools import replay_teacher_group_fit as fit
from wa.wm.student_eval_contract import validate_student_rows
from wa.wm.student_eval_finalize import load_teachers
from wa.wm.teacher_window_plan import SOURCE_KEYS, canonical_sha, simulate_exposure

SCHEMA = 'wa.teacher-window-schedule.v2'
KIND = 'stt_anchor_v1'
SOURCE_SHA = {
    'old_candidate_report': '663c66b16f1784e1a5a9962501fc4b66c32c32792dc0483be95fad2c16680c66',
    'candidate_rows': '0ab45e1b35bb0b8809fcc77fcaabca59b35bab0839507d716c371ce2d5a6f358',
    'candidate_outcomes': 'b03ec11a2932ed317d9ebedbd420342f9ac26551b7a39872f4f99e072e681736',
    'fixed88_replay': '3db61f67a648258593382dd2adbfd508a2d8dbe0aeb077648b0e8b51f54d7f73',
}
CHECKPOINT_SHA = 'c510c04d987d141423401a25323ea353314107a16f4dac5d2901370ab199fa52'
PARENT_SHA = 'ab39144b0490937ce43c4ab28e2d90cdb0c560f700d3364a29ce1dc9a08b999d'
ANCHOR_SALT = 'wa-stt-anchor-v1:'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def unique_rows(rows):
    result = old.unique_map(rows, lambda row: (row['task'], row['key']))
    require(len(result) == 4215 and Counter(k[0] for k in result) ==
            dict(stt=1405, dt=1405, at=1405), 'full three-task rows required')
    return result


def refuse_output(output, protected=()):
    output = Path(output)
    require(not output.exists() and not output.is_symlink(), 'refusing existing candidate or symlink')
    require(output.parent.is_dir(), 'candidate parent must already exist')
    resolved = output.resolve()
    for source in protected:
        source = Path(source).resolve()
        require(resolved != source and source not in resolved.parents, 'candidate inside immutable source')


def time_quantiles(indices, ages, count):
    """Unique valid-window quantiles in actual timestamp order, not RGB guesses."""
    indices = np.asarray(indices)
    ages = np.asarray(ages, dtype=np.float64)
    require(indices.ndim == 1 and indices.dtype.kind in 'iu' and
            ages.shape == indices.shape and len(set(indices.tolist())) == len(indices),
            'invalid candidate window identities')
    require(type(count) is int and count > 0 and len(indices) >= count,
            'too few valid windows: no replacement or relaxed cap allowed')
    require(np.isfinite(ages).all() and (ages >= 0).all(), 'invalid actual ages')
    order = np.argsort(ages, kind='stable')
    require((np.diff(ages[order]) > 0).all(), 'episode ages must be strictly increasing')
    positions = np.rint(np.linspace(0, len(indices)-1, count)).astype(np.int64)
    picked = indices[order[positions]].astype(np.int64)
    require(len(set(picked.tolist())) == count, 'quantile selection repeated a window')
    return picked.tolist()


def allocate(records, windows, candidate, *, sixth_count=17):
    """Pure allocation. Formal source and exact count gates are in build()."""
    n = len(windows['dataset_index'])
    ep = windows['episode_index']
    require(np.array_equal(windows['dataset_index'], np.arange(n)), 'teacher index identity changed')
    require(ep.shape == (n,) and ep.dtype.kind in 'iu' and
            (ep >= 0).all() and (ep < len(records)).all(), 'invalid episode map')
    require(windows['age_s'].shape == (n,) and np.isfinite(windows['age_s']).all()
            and (windows['age_s'] >= 0).all(), 'invalid time metadata')
    require(windows['hard'].dtype.kind == 'b' and windows['early'].dtype.kind == 'b' and
            np.array_equal(windows['early'], windows['age_s'] <= 2), 'early metadata changed')
    require(len({r['episode_uid'] for r in records}) == len(records), 'duplicate episode UID')
    hard = np.array([records[int(i)]['hard'] for i in ep], dtype=bool)
    require(np.array_equal(hard, windows['hard']), 'original hard identities changed')
    valid_count = np.bincount(ep, minlength=len(records))
    early_count = np.bincount(ep[windows['early']], minlength=len(records))
    regression_episodes, anchor_episodes, gain_episodes, remaining_episodes = [], [], [], []
    for i, record in enumerate(records):
        require(record['episode_uid'] == record['task'] + ':' + record['key'], 'episode UID mismatch')
        require(record['selected_teacher_success'] is True and record['selected_teacher_fallback'] is False,
                'unsuccessful/fallback teacher is never an anchor')
        require(type(record['hard']) is bool and type(record['student_success']) is bool and
                record['hard'] == (record['task'] == 'stt' and not record['student_success']),
                'hard must remain defined by old61377')
        require(record['valid_windows'] == valid_count[i] and record['early_windows'] == early_count[i],
                'per-episode window count mismatch')
        after = candidate[(record['task'], record['key'])]
        require(type(after['success']) in (bool, int, float) and after['success'] in (0, 1), 'invalid candidate outcome')
        if record['task'] != 'stt':
            continue
        if record['hard']:
            (gain_episodes if after['success'] else remaining_episodes).append(i)
        elif valid_count[i]:
            (anchor_episodes if after['success'] else regression_episodes).append(i)
    require(type(sixth_count) is int and 0 <= sixth_count <= len(anchor_episodes), 'invalid sixth budget')
    anchor_episodes.sort(key=lambda i: hashlib.sha256(
        (ANCHOR_SALT + records[i]['episode_uid']).encode()).hexdigest())
    sixth = set(anchor_episodes[:sixth_count])
    hard_ids = np.flatnonzero(hard).tolist()
    hard_early = np.flatnonzero(hard & windows['early']).tolist()
    regression_ids = np.flatnonzero(np.isin(ep, regression_episodes)).tolist()
    anchor_ids, anchor_details = [], []
    for i in anchor_episodes:
        ids = np.flatnonzero(ep == i)
        count = 6 if i in sixth else 5
        picked = time_quantiles(ids, windows['age_s'][ids], count)
        anchor_ids.extend(picked)
        anchor_details.append(dict(episode_index=i, episode_uid=records[i]['episode_uid'],
            requested=count, indices=picked, ages_s=windows['age_s'][picked].tolist(),
            valid_windows=int(valid_count[i])))
    pieces = dict(hard_once=hard_ids, hard_early_once=hard_early,
                  regression_once=regression_ids, balanced_anchor_once=anchor_ids)
    extras = sorted(index for ids in pieces.values() for index in ids)
    frequency = np.bincount(extras, minlength=n).astype(np.int64)
    require(frequency.max(initial=0) <= 2, 'more than two extras per teacher window')
    require(len(set(anchor_ids)) == len(anchor_ids) and
            not set(anchor_ids) & set(regression_ids) and
            not (set(anchor_ids) | set(regression_ids)) & set(hard_ids), 'extra cohorts overlap')
    require(all(records[int(ep[i])]['task'] == 'stt' for i in extras), 'DT/AT extras forbidden')
    return extras, dict(pieces=pieces, anchor_details=anchor_details,
        regression_episode_indices=regression_episodes, anchor_episode_indices=anchor_episodes,
        original_gain_episode_indices=gain_episodes, original_remaining_episode_indices=remaining_episodes)


class IndexSimulationView:
    """CPU-only schedule view; deliberately not a training Dataset implementation."""
    def __init__(self, plan):
        required = {'schema', 'base_count', 'teacher_count', 'extra_teacher_indices',
                    'source_hashes', 'selection_source_hashes'}
        require(set(plan) == required and plan['schema'] == SCHEMA, 'unknown candidate schedule')
        for key in ('base_count', 'teacher_count'):
            require(type(plan[key]) is int and plan[key] > 0, 'invalid source count')
        require(set(plan['source_hashes']) == SOURCE_KEYS, 'original five source pins required')
        require(set(plan['selection_source_hashes']) == set(SOURCE_SHA), 'selection source pins required')
        for hashes in (plan['source_hashes'], plan['selection_source_hashes']):
            require(all(isinstance(v, str) and len(v) == 64 and
                        all(c in '0123456789abcdef' for c in v) for v in hashes.values()), 'invalid SHA256')
        ids = plan['extra_teacher_indices']
        require(isinstance(ids, list) and ids and all(type(i) is int for i in ids)
                and ids == sorted(ids) and min(ids) >= 0 and max(ids) < plan['teacher_count'],
                'sorted in-bounds explicit extra multiset required')
        require(max(Counter(ids).values()) <= 2, 'extra multiplicity exceeds two')
        self.base, self.teacher = range(plan['base_count']), range(plan['teacher_count'])
        self.extra = tuple(ids)
        self.plan_sha256 = canonical_sha(plan)

    def __len__(self):
        return len(self.base) + len(self.teacher) + len(self.extra)

    def locate(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        if index < len(self.base):
            return 'base', index
        index -= len(self.base)
        if index < len(self.teacher):
            return 'teacher', index
        return 'teacher', self.extra[index-len(self.teacher)]


def masks_for(records, windows, allocation):
    ep = windows['episode_index']
    masks = {'original_hard': windows['hard'], 'early': windows['early'],
             'original_hard_early': windows['hard'] & windows['early'],
             'original_hard_late': windows['hard'] & ~windows['early']}
    cohorts = {
        'old15_gains': allocation['original_gain_episode_indices'],
        'old76_hard_still_failed': allocation['original_remaining_episode_indices'],
        'old12_regressions_with_demo': allocation['regression_episode_indices'],
        'balanced_success_anchor_pool': allocation['anchor_episode_indices'],
    }
    for name, ids in cohorts.items():
        masks[name] = np.isin(ep, ids)
    return masks


def grouped_counts(records, windows, counts, masks):
    ep = windows['episode_index']
    by_episode = np.bincount(ep, weights=counts, minlength=len(records)).astype(np.int64)
    early = np.bincount(ep[windows['early']], weights=counts[windows['early']],
                        minlength=len(records)).astype(np.int64)
    task_teacher, scenes = Counter(), Counter()
    for i, record in enumerate(records):
        task_teacher[record['task'] + ':' + record['teacher']] += int(by_episode[i])
        scenes[record['task'] + ':' + record['key'].split('/', 1)[0]] += int(by_episode[i])
    cohorts = {name: dict(total=int(counts[mask].sum()),
                         early=int(counts[mask & windows['early']].sum()),
                         late=int(counts[mask & ~windows['early']].sum()))
               for name, mask in masks.items()}
    return dict(task_teacher=dict(sorted(task_teacher.items())),
                scene=dict(sorted(scenes.items())), cohorts=cohorts)


def rank_counts(mix, records, windows, masks, expected):
    """Second independent vectorized enumeration, reconciled with existing simulator."""
    teacher = np.zeros((8, len(mix.teacher)), dtype=np.int64)
    base = np.zeros(len(mix.base), dtype=np.int64)
    seen = np.zeros(len(mix), dtype=bool)
    ranks = []
    extras = np.asarray(mix.extra, dtype=np.int64)
    for rank in range(8):
        sampler = DistributedSampler(mix, num_replicas=8, rank=rank, shuffle=True, seed=42, drop_last=True)
        sampler.set_epoch(1)
        ids = np.asarray(list(sampler), dtype=np.int64)
        ids = ids[:len(ids)//2*2]
        require(len(set(ids.tolist())) == len(ids) and not seen[ids].any(), 'overlapping rank positions')
        seen[ids] = True
        original = ids[ids < len(mix.base)]
        base[original] += 1
        ti = ids[ids >= len(mix.base)] - len(mix.base)
        ti = np.where(ti < len(mix.teacher), ti,
                      extras[np.clip(ti-len(mix.teacher), 0, len(extras)-1)])
        teacher[rank] = np.bincount(ti, minlength=len(mix.teacher))
        counted = dict(rank=rank, total=len(ids), base=len(original), teacher=len(ti))
        require(counted == expected['ranks'][rank], 'rank totals differ from existing simulator')
        ranks.append(dict(**counted, **grouped_counts(records, windows, teacher[rank], masks)))
    return base, teacher, ranks, np.flatnonzero(~seen)


def load_context(root):
    artifact = root / 'artifacts'
    old_dir = artifact / 'hard_stt_candidate_61377_20261007_v1'
    sources = old.Sources()
    old_report = sources.read(old_dir / 'report.json', SOURCE_SHA['old_candidate_report'])
    for name, expected in old_report['artifacts'].items():
        sources.verify(old_dir / name, expected)
    manifest_paths = [p for p, h in old_report['source_hashes'].items() if h == old.MANIFEST_SHA]
    require(len(manifest_paths) == 1, 'unique original manifest required')
    manifest = Path(manifest_paths[0])
    cache = artifact / 'dual_teacher_se2_cache_20261006_v1'
    base_index = artifact / 'robot_transition_audit_v2'
    baseline = artifact / 'student61377_full_audit_20261007_v1'
    teacher = artifact / 'dual_teacher_complete_audit_20261006_v1/combined_selections.jsonl'
    verified, records, windows, old_extras, source_pins, base_count = old.load_verified(
        cache, base_index, baseline, teacher, manifest)
    sources.hashes.update(verified.hashes)
    require(records == json.loads((old_dir / 'episodes.json').read_text()), 'frozen episode metadata changed')
    with np.load(old_dir / 'teacher_windows.npz', allow_pickle=False) as z:
        require(set(z.files) == set(windows) and all(np.array_equal(z[k], windows[k]) for k in z.files),
                'rederived cache timestamps/identities differ from frozen plan')
    old_plan = json.loads((old_dir / 'plan.json').read_text())
    require(old_plan['extra_teacher_indices'] == old_extras and old_plan['extra_repeats'] == 2,
            'previous plan changed')

    candidate_root = artifact / 'student61609_full_audit_20261007_v1'
    candidate_path = candidate_root / 'combined_episodes.jsonl'
    sources.verify(candidate_path, SOURCE_SHA['candidate_rows'])
    candidate_rows = read_rows(candidate_path)
    validate_student_rows(candidate_rows, dict(checkpoint_sha=CHECKPOINT_SHA, step=59716))
    candidate = unique_rows(candidate_rows)
    outcome_path = artifact / 'student61609_hard_outcomes_20261007_v1/hard_stt_outcomes.json'
    outcome = sources.read(outcome_path, SOURCE_SHA['candidate_outcomes'])
    require(outcome['audit_status'] == 'PASS' and outcome['episodes'] == 4215 and
            outcome['candidate_checkpoint_sha256'] == CHECKPOINT_SHA and
            outcome['candidate_checkpoint_step'] == 59716, 'wrong full outcome audit')
    for path, expected in outcome['source_hashes'].items():
        sources.verify(path, expected)
    derived = outcomes.classify(candidate_rows, read_rows(baseline / 'combined_episodes.jsonl'),
                                load_teachers(teacher), records, dict(tasks=outcome['tasks']))
    require(derived == outcome['groups'], 'recomputed gains/regressions differ from frozen outcome audit')
    replay_path = artifact / 'teacher_group_fit_61609_replay_20261007_v1.json'
    replay = sources.read(replay_path, SOURCE_SHA['fixed88_replay'])
    original_path = artifact / 'teacher_group_fit_61377_20261007_v1.json'
    original_fit = sources.read(original_path, fit.REFERENCE_SHA)
    reference = fit.validate_reference(original_fit)
    require(replay['status'] == 'FIXED88_CANDIDATE_LABEL_FIT_ONLY' and
            replay['reference_sha256'] == fit.REFERENCE_SHA and
            replay['selected'] == original_fit['selected'] and
            replay['baseline'] == original_fit['models'][fit.BASELINE] and
            replay['candidate']['sha256'] == CHECKPOINT_SHA and replay['candidate']['step'] == 59716,
            'fixed88 replay identity changed')
    deltas, groups = fit.paired_deltas(replay['candidate']['records'], replay['selected'], reference)
    require(deltas == replay['paired_metric_deltas'] and groups == replay['paired_summary'],
            'fixed88 replay derived deltas changed')
    sources.verify(__file__, old.sha(__file__))
    return sources, records, windows, candidate, outcome, source_pins, base_count, old_dir


def build(root, output):
    root, output = Path(root), Path(output)
    require(root.is_absolute() and str(root.resolve()).startswith('/data/nas_ray/'), 'verified NAS root required')
    artifact = root / 'artifacts'
    require(output.parent.resolve() == artifact.resolve(), 'candidate must be a new direct artifacts child')
    refuse_output(output)
    sources, records, windows, candidate, outcome, source_pins, base_count, old_dir = load_context(root)
    extras, allocation = allocate(records, windows, candidate)
    pieces = allocation['pieces']
    require({k: len(v) for k, v in pieces.items()} == dict(hard_once=10413,
        hard_early_once=3270, regression_once=886, balanced_anchor_once=6257), 'fixed budget components changed')
    require(len(allocation['regression_episode_indices']) == 10 and
            len(allocation['anchor_episode_indices']) == 1248 and
            len(allocation['original_gain_episode_indices']) == 15 and
            len(allocation['original_remaining_episode_indices']) == 78,
            'fixed 15/76+2/10/1248 cohorts changed')
    require(len(outcome['groups']['regressions']['by_task']['stt']['keys']) == 12, 'expected all12 regressions')
    require(Counter(x['requested'] for x in allocation['anchor_details']) == {5:1231, 6:17},
            'anchor per-episode allocation changed')
    require(base_count == 726631 and len(windows['dataset_index']) == 436816 and
            len(extras) == 20826, 'fixed source/extra totals changed')
    plan = dict(schema=SCHEMA, base_count=base_count, teacher_count=len(windows['dataset_index']),
                extra_teacher_indices=extras, source_hashes=source_pins, selection_source_hashes=SOURCE_SHA)
    mix = IndexSimulationView(plan)
    exposure, counts = simulate_exposure(mix, world=8, batch=2, seed=42, epoch=1)
    masks = masks_for(records, windows, allocation)
    base_counts, rank_teacher_counts, ranks, dropped = rank_counts(mix, records, windows, masks, exposure)
    require(np.array_equal(rank_teacher_counts.sum(0), counts), 'two independent count enumerations disagree')
    require(len(mix) == 1184273 and exposure['actual_total'] == 1184272 and
            exposure['actual_base'] == 726631 and exposure['actual_teacher'] == 457641 and
            len(dropped) == 1 and (base_counts == 1).all(), 'pre-DDP/post-drop budget changed')
    extra_counts = np.bincount(extras, minlength=len(counts)).astype(np.int64)
    planned = 1 + extra_counts
    require(np.array_equal(planned[windows['hard'] & windows['early']],
                          np.full(3270, 3)) and
            np.array_equal(planned[windows['hard'] & ~windows['early']], np.full(7143, 2)),
            'hard early/late planned counts changed')
    teacher_difference = planned-counts
    require((teacher_difference >= 0).all() and teacher_difference.sum() == 1,
            'unexpected source-window drop pattern')
    exposure.update(candidate_kind=KIND, status='EXACT8RANK_INDEX_SIMULATION_ONLY_NOT_EXECUTED',
        grouped_planned=grouped_counts(records, windows, planned, masks),
        grouped_simulated=grouped_counts(records, windows, counts, masks), rank_groups=ranks,
        dropped_positions=dropped.tolist(), dropped_teacher_indices=np.flatnonzero(teacher_difference).tolist(),
        per_episode=[dict(episode_uid=row['episode_uid'], task=row['task'], teacher=row['teacher'],
            original_hard=row['hard'], planned=int(planned[windows['episode_index'] == i].sum()),
            simulated=int(counts[windows['episode_index'] == i].sum()))
            for i, row in enumerate(records)])
    sources.recheck()
    refuse_output(output, (old_dir,))
    output.mkdir(exist_ok=False)
    for name, value in (('plan.json', plan), ('episodes.json', records), ('exposure.json', exposure)):
        with (output / name).open('x') as f:
            json.dump(value, f, indent=2 if name != 'episodes.json' else None, allow_nan=False)
    # Identity metadata remains byte-equivalent in values to the old seven-field contract.
    np.savez_compressed(output / 'teacher_windows.npz', **windows)
    np.savez_compressed(output / 'simulated_counts.npz', planned_teacher_counts=planned,
        simulated_teacher_counts=counts, extra_counts=extra_counts,
        simulated_base_counts=base_counts, rank_teacher_counts=rank_teacher_counts)
    report = dict(status='CANDIDATE_ONLY_NOT_TRAINING_RELEASE', candidate_kind=KIND,
        runtime_integration_required=True, experiment=old.EXPERIMENT,
        plan_canonical_sha256=canonical_sha(plan), checkpoint_sha256=old.CHECKPOINT_SHA,
        checkpoint_step=59065, candidate_checkpoint_sha256=CHECKPOINT_SHA, candidate_checkpoint_step=59716,
        source_hashes=sources.hashes, selection_source_hashes=SOURCE_SHA, tool_sha256=old.sha(__file__),
        budget=dict(base_once=726631, teacher_once=436816, hard_extra=13683,
            regression_extra=886, balanced_anchor_extra=6257, all_extra=20826,
            pre_ddp=1184273, simulated_post_drop=1184272),
        selection_rules=dict(hard='original61377 hard identities unchanged; <=2s total3 else total2',
            regressions='all valid windows of10 old-success/new-failure admitted STT teachers gain one extra',
            anchors='remaining1248 old-success/current-success STT with valid windows:1231x5,17x6',
            sixth_order='SHA256(' + ANCHOR_SALT + 'episode_uid), ascending',
            quantiles='round-to-even linspace(0,n-1,k) positions after strict actual-age sort; unique or reject',
            policy_metadata='No task/teacher/group/time bucket tags enter the policy'),
        anchor_selection=allocation['anchor_details'],
        regression_episode_uids=[records[i]['episode_uid'] for i in allocation['regression_episode_indices']],
        all12_regression_keys=outcome['groups']['regressions']['by_task']['stt']['keys'],
        absent2_regression_keys=[r['key'] for r in outcome['groups']['regressions']['records']
            if r['task']=='stt' and r['demonstration_class']=='neither_teacher_success'],
        original15_gain_keys=outcome['groups']['original91_recovered']['by_task']['stt']['keys'],
        original76_still_failed_keys=outcome['groups']['original91_still_failed']['by_task']['stt']['keys'],
        training_recipe=dict(parent_checkpoint='/data/nas_ray/project/md-ak/users/zeying.gong/'
            'job_59866/task_70728/wa_history_repeat_v1/checkpoint.pt', parent_sha256=PARENT_SHA,
            model_and_optimizer=True, new_epochs=1, cumulative_epochs=2, world_size=8,
            seed=42, batch_size=2, accumulation=2, history_repeat_probability=.25,
            model_loss_controller_physics='unchanged; resumed parent LR, no LR increase'),
        limits=['Offline candidate only; old v1 runtime must reject this schema.',
                'Exact simulated counts are not actual training consumption or new model performance.',
                'Fixed88 labels and in-set outcomes informed the recipe; no untouched-test claim.',
                'Lower late-hard exposure can regress existing gains; full paired4215 required after any training.',
                'No filter changes or demonstrations created for zero-window or both-teacher-failed cases.'],
        artifacts={p.name: old.sha(p) for p in output.iterdir() if p.is_file()})
    with (output / 'report.json').open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(dict(status=report['status'], output=str(output),
        report_sha256=old.sha(output/'report.json'), plan_sha256=canonical_sha(plan),
        budget=report['budget'], simulated=exposure['grouped_simulated']['cohorts'])), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    build(args.root, args.output)


if __name__ == '__main__':
    main()
