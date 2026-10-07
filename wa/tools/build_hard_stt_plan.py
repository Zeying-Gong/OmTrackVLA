"""Freeze a bounded hard-STT sampling candidate; never train or modify sources.

Eligibility is derived from the pinned, fully audited 61377 outcomes and admitted
successful no-fallback teacher branches. No caller-supplied eligibility list is
accepted. Metadata is for sampling/diagnostics only, never policy conditioning.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from wa.tools.audit_stt_start_coverage import validate_row_indices, window_ages
from wa.tools.merge_student_partitions import read_partitions
from wa.wm.dual_teacher_selection import EXPERIMENT, select_teacher
from wa.wm.full_mixed_contract import MANIFEST_SHA, summarize
from wa.wm.robot_data import CONTRACT
from wa.wm.student_eval_contract import validate_student_rows
from wa.wm.student_eval_finalize import TEACHER_SHA, load_teachers, superiority
from wa.wm.teacher_window_plan import (
    SCHEMA, PlannedTeacherMix, canonical_sha, simulate_exposure,
)

CHECKPOINT_SHA = 'b5236a21f2d2695780029503c97e339c8350dc4f7337d05ec7e8692b9b9c327f'
PINNED = {
    'student_rows': '62ac4c5a2bb28e8db89dbf1f67c968725977ca2bb321abcc78536b16e233d14b',
    'student_summary': '30dfdbd9263ca0ba3615cb1845938de235c7e1b15b2141edb5ac584622f6bbf3',
    'student_pair_audit': 'dc98149bdcfb6baf05d1fae167de572b476679f48384dd91ea92e57851645844',
    'teacher_selections': TEACHER_SHA,
    'teacher_release': '7d49433f23b3379059359ae77dcc8b7136b5ff88a6afb2635232d3319c300459',
    'cache_complete': '79e5a7dc43b61a23bb64d5676c4c1cac17b459a5e390b308377012e582ba1a8d',
    'teacher_index': '6c6357ba0e1cf1a9d1eab56b3d90d0c8de6df6df8d6514d9ff73fdb906603b21',
    'teacher_index_audit': '5aa41388f10be824101588eb064b7392e6538db0183b5402a736dfce5c450027',
    'teacher_episode_audit': '89d06351fbcbe575a46b373c85e693c94081e192be2db15174c6f93acf8a6a74',
    'base_index': '17ecf5abe76583759fd338d58117896343a26ebc86b55b1e88c956457a763538',
    'base_index_audit': '3138364dea80543fc83be476e1af8191a4eca3711a9b353591311fc8daf5c2d3',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class Sources:
    def __init__(self):
        self.hashes = {}

    def verify(self, path, expected):
        path = str(Path(path).resolve())
        actual = self.hashes.get(path)
        if actual is None:
            actual = sha(path)
        if actual != expected:
            raise ValueError('changed source: ' + path)
        self.hashes[path] = actual

    def read(self, path, expected):
        self.verify(path, expected)
        return json.loads(Path(path).read_text())

    def recheck(self):
        for path, expected in self.hashes.items():
            if sha(path) != expected:
                raise ValueError('source changed during plan build: ' + path)


def unique_map(rows, key):
    result = {}
    for row in rows:
        identity = key(row)
        if identity in result:
            raise ValueError('duplicate evidence identity')
        result[identity] = row
    return result


def derive_admissions(student_rows, selections, demonstrations, entries):
    """Pure identity/selection check; formal completeness/pins checked by load."""
    students = unique_map(student_rows, lambda r: (r['task'], r['key']))
    teachers = unique_map(selections, lambda r: (r['pair']['task'], r['pair']['key']))
    if students.keys() != teachers.keys():
        raise ValueError('student/teacher keysets differ')
    eligible = {}
    for identity, row in teachers.items():
        selected = select_teacher(row['branches']['lightnav'], row['branches']['oracle'])
        for field in ('experiment', 'pair', 'selected_teacher', 'reason',
                      'results', 'demonstration_candidate'):
            if row.get(field) != selected[field]:
                raise ValueError('stored teacher selection differs: ' + str(identity))
        if selected['demonstration_candidate']:
            teacher = selected['selected_teacher']
            branch = row['branches'][teacher]
            if branch.get('transport_fallback') or not branch['result']['success']:
                raise ValueError('failed/fallback branch cannot be demonstrated')
            eligible[identity] = (teacher, str(Path(branch['artifact_root']).resolve()))

    admitted = unique_map(demonstrations, lambda r: (r['task'], r['key']))
    if admitted.keys() != eligible.keys():
        raise ValueError('release does not match successful no-fallback selections')
    cached = unique_map(entries, lambda r: (r['task'], r['episode_uid'].split(':', 1)[1]))
    if cached.keys() != admitted.keys():
        raise ValueError('cache/release identities differ')
    records = []
    for entry in entries:
        task, key = entry['episode_uid'].split(':', 1)
        identity = (task, key)
        if task != entry['task']:
            raise ValueError('cached task identity mismatch')
        admission = admitted[identity]
        teacher, branch = eligible[identity]
        if (entry['teacher'], str(Path(entry['root']).resolve())) != (teacher, branch):
            raise ValueError('cache chose a different teacher branch')
        if (admission['teacher'], str(Path(admission['branch']).resolve())) != (teacher, branch):
            raise ValueError('release chose a different teacher branch')
        if entry.get('takeover_step') != 0 or admission.get('takeover_step') != 0:
            raise ValueError('expected paired from-start demonstrations')
        outcome = students[identity]
        if outcome.get('success') not in (0, 1):
            raise ValueError('nonbinary student outcome')
        records.append(dict(
            episode_uid=entry['episode_uid'], task=task, key=key, teacher=teacher,
            hard=task == 'stt' and not bool(outcome['success']),
            student_success=bool(outcome['success']), student_status=outcome['status'],
            branch=branch, selected_teacher_success=True, selected_teacher_fallback=False))
    return records, admitted


def derive_windows(history, episode, valid, records, times_by_episode):
    """Map valid local teacher indices to verified episode/time diagnostic metadata."""
    valid_episode = validate_row_indices(history, episode, valid, len(records))
    age = np.zeros(len(valid), dtype=np.float64)
    hard = np.array([records[int(ep)]['hard'] for ep in valid_episode], dtype=bool)
    for ep, record in enumerate(records):
        positions = np.flatnonzero(valid_episode == ep)
        age[positions] = window_ages(times_by_episode[ep], history, valid[positions])
        record['valid_windows'] = len(positions)
        record['early_windows'] = int((age[positions] <= 2.).sum())
        record['first_valid_age_s'] = float(age[positions].min()) if len(positions) else None
    extras = np.flatnonzero(hard).tolist()
    return extras, dict(dataset_index=np.arange(len(valid), dtype=np.int64),
        raw_row=valid, episode_index=valid_episode,
        current_observation_index=np.asarray(history[valid, -1]),
        age_s=age, hard=hard, early=age <= 2.)


def make_plan(base_count, teacher_count, extras, source_hashes, extra_repeats):
    if type(extra_repeats) is not int or extra_repeats not in (1, 2):
        raise ValueError('candidate requires one or two bounded extra exposures')
    if not extras:
        raise ValueError('no eligible hard-STT windows')
    plan = dict(schema=SCHEMA, base_count=base_count, teacher_count=teacher_count,
        extra_repeats=extra_repeats, extra_teacher_indices=list(extras),
        source_hashes=source_hashes)
    mix = PlannedTeacherMix(range(base_count), range(teacher_count), plan,
        source_hashes=source_hashes, eligible_teacher_indices=extras)
    return plan, mix


def summarize_exposure(records, windows, counts):
    groups, episodes = Counter(), []
    for ep, record in enumerate(records):
        positions = np.flatnonzero(windows['episode_index'] == ep)
        value = int(counts[positions].sum())
        early = int(counts[positions][windows['early'][positions]].sum())
        groups[record['task'] + ':' + record['teacher']] += value
        groups['hard_stt' if record['hard'] else 'nonhard_teacher'] += value
        groups['hard_stt_early' if record['hard'] else 'nonhard_teacher_early'] += early
        episodes.append(dict(episode_uid=record['episode_uid'], exposures=value,
                             early_exposures=early, hard=record['hard']))
    return dict(groups=dict(groups), per_episode=episodes,
        early_definition='teacher timestamp_s minus first timestamp_s <= 2 seconds')


def load_verified(cache, base_index, student_root, teacher_selections, manifest_path):
    cache, base_index, student_root = map(Path, (cache, base_index, student_root))
    sources = Sources()
    manifest = sources.read(manifest_path, MANIFEST_SHA)
    summary = sources.read(student_root/'summary.json', PINNED['student_summary'])
    pairs = sources.read(student_root/'student_teacher_pair_audit.json', PINNED['student_pair_audit'])
    sources.verify(student_root/'combined_episodes.jsonl', PINNED['student_rows'])
    rows = [json.loads(s) for s in (student_root/'combined_episodes.jsonl').read_text().splitlines() if s]
    contract = dict(checkpoint_sha=CHECKPOINT_SHA, step=59065)
    validate_student_rows(rows, contract)
    recomputed = summarize(rows, manifest, **contract)
    for key, value in recomputed.items():
        if key != 'limits' and summary.get(key) != value:
            raise ValueError('student summary differs: ' + key)
    if summary.get('experiment') != EXPERIMENT or summary.get('new_episodes') != 4215 or summary.get('reused_baseline_episodes') != 0:
        raise ValueError('complete new in-set student audit required')
    if pairs.get('status') != 'PASS' or pairs.get('episodes') != 4215:
        raise ValueError('full student paired-start audit required')
    if set(pairs['teacher_artifact_hashes']) != {r['task'] + ':' + r['key'] for r in rows}:
        raise ValueError('incomplete student pair evidence')
    original, original_hashes = read_partitions(summary['partition_roots'], manifest, contract)
    if original_hashes != summary['source_hashes'] or len(original_hashes) != 78:
        raise ValueError('original student source inventory changed')
    if unique_map(original, lambda r: (r['task'], r['key'])) != unique_map(rows, lambda r: (r['task'], r['key'])):
        raise ValueError('merged student rows differ from original partitions')
    for name, expected in original_hashes.items():
        sources.verify(name, expected)
    sources.verify(teacher_selections, TEACHER_SHA)
    teachers = load_teachers(teacher_selections)
    if summary['superiority'] != superiority(rows, teachers) or summary['teacher_reference_sha256'] != TEACHER_SHA:
        raise ValueError('student teacher comparison changed')

    complete = sources.read(cache/'complete.json', PINNED['cache_complete'])
    if complete['experiment'] != EXPERIMENT or complete['development_only'] is not False:
        raise ValueError('formal explicit in-set cache required')
    if complete['source_sha256'] != PINNED['teacher_release']:
        raise ValueError('cache release differs')
    release = sources.read(complete['source_release'], PINNED['teacher_release'])
    if release['experiment'] != EXPERIMENT or release['paired_outcomes_validated'] is not True or release['expected'] != 4215 or release['partial_partition']:
        raise ValueError('full validated teacher release required')
    if release['persisted_pair_evidence_count'] != 4215 or release['persisted_pair_evidence'] != pairs['teacher_artifact_hashes']:
        raise ValueError('teacher release and student pair evidence differ')
    for name, expected in release['source_files'].items():
        sources.verify(name, expected)
    for name, expected in complete['files'].items():
        sources.verify(cache/name, expected)
    sources.verify(cache/'source_image_hashes.json', complete['source_image_inventory_sha256'])
    index_audit = sources.read(cache/'index/audit.json', PINNED['teacher_index_audit'])
    episode_audit = sources.read(cache/'index/train_episodes_audit.json', PINNED['teacher_episode_audit'])
    if index_audit['contract'] != CONTRACT or index_audit['cache_complete_sha256'] != PINNED['cache_complete'] or index_audit['splits']['train']['index_sha256'] != PINNED['teacher_index']:
        raise ValueError('teacher index/cache mismatch')
    sources.verify(cache/'index/train_valid.npy', PINNED['teacher_index'])
    base_audit = sources.read(base_index/'audit.json', PINNED['base_index_audit'])
    if base_audit['contract'] != CONTRACT or base_audit['splits']['train']['index_sha256'] != PINNED['base_index']:
        raise ValueError('base index contract changed')
    sources.verify(base_index/'train_valid.npy', PINNED['base_index'])
    base = np.load(base_index/'train_valid.npy')
    if base.ndim != 1 or base.dtype.kind not in 'iu' or len(base) != 726631 or not (np.diff(base) > 0).all():
        raise ValueError('base index shape/order/count changed')

    entries = json.loads((cache/'train_episodes.json').read_text())
    records, admitted = derive_admissions(rows, teachers, release['teacher_demonstrations'], entries)
    stats = unique_map(episode_audit['stats'], lambda s: s['episode'])
    if set(stats) != set(range(len(entries))) or episode_audit['errors']:
        raise ValueError('incomplete teacher episode audit')
    times = {}
    for ep, entry in enumerate(entries):
        admission = admitted[(records[ep]['task'], records[ep]['key'])]
        root = Path(entry['root'])
        # Only already released selected branches are read; never failed prefixes.
        for name, expected in admission['hashes'].items():
            sources.verify(root/name, expected)
        for name, expected in stats[ep]['source_sha256'].items():
            if admission['hashes'].get(name) != expected:
                raise ValueError('episode/index admission hash differs')
        result = json.loads((root/'result.json').read_text())
        if not result['success'] or result['collision'] or not result['policy_init_valid']:
            raise ValueError('unsuccessful stored demonstration')
        observations = json.loads((root/'observations.json').read_text())
        times[ep] = np.array([o['timestamp_s'] for o in observations], dtype=np.float64)
        if ep % 500 == 0:
            print(json.dumps(dict(verified_teacher_episodes=ep)), flush=True)
    history = np.load(cache/'train_history.npy', mmap_mode='r')
    episode = np.load(cache/'train_episode.npy', mmap_mode='r')
    valid = np.load(cache/'index/train_valid.npy')
    extras, windows = derive_windows(history, episode, valid, records, times)
    if len(valid) != 436816 or len(records) != 3985 or len(extras) != 10413:
        raise ValueError('frozen teacher/window qualification counts changed')
    hard = [r for r in records if r['hard']]
    if len(hard) != 93 or sum(r['valid_windows'] > 0 for r in hard) != 91:
        raise ValueError('expected 93 admitted/91 nonempty hard-STT episodes')
    source_hashes = {key: PINNED[key] for key in
        ('cache_complete', 'teacher_index', 'base_index', 'student_rows', 'teacher_selections')}
    return sources, records, windows, extras, source_hashes, len(base)


def build(cache, base_index, student_root, teacher_selections, manifest, output,
          extra_repeats=2):
    output = Path(output)
    if output.exists():
        raise ValueError('refusing to overwrite candidate directory')
    sources, records, windows, extras, hashes, base_count = load_verified(
        cache, base_index, student_root, teacher_selections, manifest)
    plan, mix = make_plan(base_count, len(windows['raw_row']), extras, hashes, extra_repeats)
    exposure, counts = simulate_exposure(mix, world=8, batch=2, seed=42, epoch=1)
    exposure.update(summarize_exposure(records, windows, counts))
    before = np.ones(len(counts), dtype=np.int64)
    before[extras] += extra_repeats
    exposure['pre_ddp_groups'] = summarize_exposure(records, windows, before)['groups']
    exposure['teacher_fraction'] = exposure['actual_teacher'] / exposure['actual_total']
    sources.recheck()
    output.mkdir(parents=True, exist_ok=False)
    with (output/'plan.json').open('x') as f:
        json.dump(plan, f, indent=2, allow_nan=False)
    with (output/'episodes.json').open('x') as f:
        json.dump(records, f, allow_nan=False)
    np.savez_compressed(output/'teacher_windows.npz', **windows)
    with (output/'exposure.json').open('x') as f:
        json.dump(exposure, f, indent=2, allow_nan=False)
    report = dict(status='CANDIDATE_ONLY_NOT_TRAINING_RELEASE',
        experiment=EXPERIMENT, plan_canonical_sha256=canonical_sha(plan),
        checkpoint_sha256=CHECKPOINT_SHA, checkpoint_step=59065,
        hard_episode_count=93, hard_nonempty_episode_count=91,
        eligible_teacher_windows=len(extras), extra_repeats=extra_repeats,
        source_hashes=sources.hashes, tool_sha256=sha(__file__),
        metadata_scope='Sampling and diagnostics only; no task/teacher/failure/time group tags enter policy',
        validation_scope='Pinned full student/teacher paired audits and media provenance reused; actual source JSON/cache/index hashes rechecked; no new rollout or repeated media decode',
        artifacts={p.name: sha(p) for p in output.iterdir() if p.is_file()})
    with (output/'report.json').open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(dict(status=report['status'], output=str(output),
        eligible_windows=len(extras), plan_sha256=canonical_sha(plan),
        actual_total=exposure['actual_total'], actual_teacher=exposure['actual_teacher'],
        groups=exposure['groups'])), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'base-index', 'student-root', 'teacher-selections',
                 'manifest', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--extra-repeats', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    build(args.cache, args.base_index, args.student_root, args.teacher_selections,
          args.manifest, args.output, args.extra_repeats)


if __name__ == '__main__':
    main()
