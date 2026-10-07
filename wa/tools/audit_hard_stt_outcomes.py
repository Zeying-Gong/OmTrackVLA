"""CPU-only outcome diagnosis against the frozen 61377 hard-STT sampling set.

No media, checkpoint, model, rollout or training is loaded. Complete candidate
and baseline bundles are independently revalidated by the existing goal audit.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from wa.tools import audit_student_goal as goal
from wa.wm import dual_teacher_selection as selection
from wa.wm.dual_teacher_metrics import summarize_teachers
from wa.wm.dual_teacher_selection import select_teacher
from wa.wm.student_eval_finalize import load_teachers

SCHEMA = 'wa.hard-stt-outcomes.v1'
PLAN_EPISODES_SHA = '4a7baac03e8ca8871fe55dc2dc5444283a468bd61f4dc1f8fcf138156187a9fc'
DEFAULT_PLAN = goal.PROJECT/'artifacts/hard_stt_candidate_61377_20261007_v1/episodes.json'


def unique(rows, identity):
    result = {}
    for row in rows:
        key = identity(row)
        if key in result:
            raise ValueError('duplicate outcome/teacher/plan identity: ' + str(key))
        result[key] = row
    return result


def load_plan(path):
    if goal.file_sha(path) != PLAN_EPISODES_SHA:
        raise ValueError('not the frozen 61377 hard sampling episodes')
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list) or len(rows) != 3985:
        raise ValueError('frozen plan episode inventory changed')
    return rows


def validate_plan(plan, baseline, teachers):
    """Recompute teacher admission and hard identity, not merely a PASS flag."""
    admitted = {}
    selected = {}
    if baseline.keys() != teachers.keys():
        raise ValueError('teacher/baseline keysets differ')
    for identity, row in teachers.items():
        choice = select_teacher(row['branches']['lightnav'], row['branches']['oracle'])
        for field in ('experiment', 'pair', 'selected_teacher', 'reason', 'results',
                      'demonstration_candidate'):
            if row.get(field) != choice[field]:
                raise ValueError('teacher selection changed: ' + str(identity))
        selected[identity] = choice
        if choice['demonstration_candidate']:
            admitted[identity] = row['branches'][choice['selected_teacher']]
    metadata = unique(plan, lambda r: (r['task'], r['key']))
    if metadata.keys() != admitted.keys():
        raise ValueError('plan does not exactly match admitted teacher identities')
    for identity, row in metadata.items():
        task, key = identity
        old = baseline[identity]
        branch = admitted[identity]
        if row['episode_uid'] != task + ':' + key:
            raise ValueError('plan episode UID mismatch')
        for field in ('hard', 'student_success', 'selected_teacher_success', 'selected_teacher_fallback'):
            if type(row[field]) is not bool:
                raise ValueError('plan boolean metadata required')
        if (row['student_success'] != bool(old['success']) or
                row['hard'] != (task == 'stt' and not bool(old['success'])) or
                row['student_status'] != old['status']):
            raise ValueError('hard metadata is not defined by the fixed baseline')
        if (row['selected_teacher_success'] is not True or row['selected_teacher_fallback'] is not False or
                row['teacher'] != selected[identity]['selected_teacher'] or
                str(Path(row['branch']).resolve()) != str(Path(branch['artifact_root']).resolve())):
            raise ValueError('plan teacher admission differs')
        for field in ('valid_windows', 'early_windows'):
            if type(row[field]) is not int or row[field] < 0:
                raise ValueError('nonnegative integer window metadata required')
        if row['early_windows'] > row['valid_windows']:
            raise ValueError('early windows exceed valid windows')
    hard = {k for k, r in metadata.items() if r['hard']}
    nonzero = {k for k in hard if metadata[k]['valid_windows'] > 0}
    if len(hard) != 93 or len(nonzero) != 91 or sum(metadata[k]['valid_windows'] for k in hard) != 10413:
        raise ValueError('fixed 93/91/10413 hard qualification changed')
    return metadata, selected, hard, nonzero


def summarize_records(records):
    """Task/status/scene counts plus auditable, deterministic full key records."""
    records = sorted(records, key=lambda r: (r['task'], r['key']))
    by_task = {}
    for task in goal.TASKS:
        rows = [r for r in records if r['task'] == task]
        by_task[task] = dict(count=len(rows), keys=[r['key'] for r in rows],
            by_status=dict(sorted(Counter(r['candidate_status'] for r in rows).items())),
            by_scene=dict(sorted(Counter(r['scene'] for r in rows).items())),
            by_admission=dict(sorted(Counter(r['demonstration_class'] for r in rows).items())))
    return dict(count=len(records), by_task=by_task, records=records)


def classify(candidate_rows, baseline_rows, teachers, plan, goal_report):
    """Pure grouping called only after full paired goal validation in diagnose()."""
    new = unique(candidate_rows, lambda r: (r['task'], r['key']))
    old = unique(baseline_rows, lambda r: (r['task'], r['key']))
    teacher_map = unique(teachers, lambda r: (r['pair']['task'], r['pair']['key']))
    if new.keys() != old.keys():
        raise ValueError('candidate/baseline keys differ')
    metadata, choices, hard, nonzero = validate_plan(plan, old, teacher_map)
    records = {}
    for identity in sorted(new):
        task, key = identity
        after, before = new[identity], old[identity]
        entry = metadata.get(identity)
        choice = choices[identity]
        successful = [name for name in ('lightnav', 'oracle') if choice['results'][name]['success']]
        if entry is not None:
            category = 'admitted_valid_windows' if entry['valid_windows'] else 'admitted_zero_windows'
        elif successful:
            category = 'successful_teacher_not_admitted_selected_fallback'
        else:
            category = 'neither_teacher_success'
        records[identity] = dict(task=task, key=key, scene=key.split('/', 1)[0],
            baseline_success=bool(before['success']), candidate_success=bool(after['success']),
            baseline_status=before['status'], candidate_status=after['status'],
            baseline_steps=before['total_step'], candidate_steps=after['total_step'],
            baseline_collision=bool(before['collision']), candidate_collision=bool(after['collision']),
            candidate_init_valid=bool(after['policy_init_valid']),
            teacher_success=successful, selected_teacher=choice['selected_teacher'],
            demonstration_class=category, valid_windows=entry['valid_windows'] if entry else 0,
            early_windows=entry['early_windows'] if entry else 0,
            original_hard=identity in hard, original_hard_nonzero=identity in nonzero)
    predicates = dict(
        original91_recovered=lambda r: r['original_hard_nonzero'] and r['candidate_success'],
        original91_still_failed=lambda r: r['original_hard_nonzero'] and not r['candidate_success'],
        original2_zero_window_recovered=lambda r: r['original_hard'] and not r['original_hard_nonzero'] and r['candidate_success'],
        original2_zero_window_still_failed=lambda r: r['original_hard'] and not r['original_hard_nonzero'] and not r['candidate_success'],
        gains=lambda r: not r['baseline_success'] and r['candidate_success'],
        regressions=lambda r: r['baseline_success'] and not r['candidate_success'],
        teacher_solvable_still_failed=lambda r: bool(r['teacher_success']) and not r['candidate_success'],
        neither_teacher_solved_still_failed=lambda r: not r['teacher_success'] and not r['candidate_success'])
    groups = {name: summarize_records([r for r in records.values() if predicate(r)])
              for name, predicate in predicates.items()}
    for task in goal.TASKS:
        gains = groups['gains']['by_task'][task]
        losses = groups['regressions']['by_task'][task]
        expected = goal_report['tasks'][task]
        delta = sum(r['candidate_success'] - r['baseline_success'] for r in records.values() if r['task'] == task)
        if (gains['count'] - losses['count'] != delta or delta != expected['net_success'] or
                gains['keys'] != expected['gain_keys'] or losses['keys'] != expected['regression_keys']):
            raise ValueError('group gain/regression reconciliation failed')
    return groups


def diagnose(candidate, baseline, manifest_path, teacher_path, checkpoint_sha, checkpoint_step, plan_path):
    """Read-only entry point; never trusts an externally supplied goal report."""
    contract = dict(checkpoint_sha=checkpoint_sha, step=checkpoint_step)
    if goal.file_sha(manifest_path) != goal.MANIFEST_SHA:
        raise ValueError('manifest changed')
    manifest = json.loads(Path(manifest_path).read_text())
    plan = load_plan(plan_path)
    teachers = load_teachers(teacher_path)
    teacher_metrics = summarize_teachers(teachers, manifest)['metrics_percent']
    base = goal.load_bundle(baseline, manifest,
        dict(checkpoint_sha=goal.BASELINE_SHA, step=goal.BASELINE_STEP), pins=goal.BASELINE_PINS)
    new = base if Path(candidate).resolve() == Path(baseline).resolve() else goal.load_bundle(candidate, manifest, contract)
    checked = goal.compare_goal(new, base, manifest, contract, teachers, teacher_metrics)
    groups = classify(new['rows'], base['rows'], teachers, plan, checked)
    hashes = goal.merge_hashes(new['hashes'], base['hashes'], {
        str(Path(manifest_path).resolve()): goal.MANIFEST_SHA,
        str(Path(teacher_path).resolve()): goal.TEACHER_SHA,
        str(Path(plan_path).resolve()): PLAN_EPISODES_SHA,
        str(Path(__file__).resolve()): goal.file_sha(__file__),
        str(Path(goal.__file__).resolve()): goal.file_sha(goal.__file__),
        str(Path(selection.__file__).resolve()): goal.file_sha(selection.__file__)})
    for path, expected in hashes.items():
        if goal.file_sha(path) != expected:
            raise ValueError('source changed during outcomes audit: ' + path)
    return dict(schema=SCHEMA, audit_status='PASS', goal_status=checked['goal_status'],
        candidate_root=new['root'], baseline_root=base['root'],
        candidate_checkpoint_sha256=checkpoint_sha, candidate_checkpoint_step=checkpoint_step,
        baseline_checkpoint_sha256=goal.BASELINE_SHA, baseline_checkpoint_step=goal.BASELINE_STEP,
        baseline_self_check=checked['baseline_self_check'], episodes=4215,
        original_hard_episodes=93, original_hard_nonzero_episodes=91,
        original_hard_zero_window_episodes=2, groups=groups, tasks=checked['tasks'],
        source_hashes=hashes, frozen_plan_episodes_sha256=PLAN_EPISODES_SHA,
        limits=checked['limits'] + [
            'Grouping uses the frozen 61377 sampling identities, never the candidate failures to redefine original hard cases.',
            'Original91 means admitted nonzero teacher windows in the frozen plan; actual training consumption is certified by the separate training exposure audit.',
            'Teacher-solvable is the observed teacher success union, not a guaranteed student outcome.',
            'Teacher states differ from failed student states; this audit does not establish failure causality.',
            'No media decoded, model loaded, rollout performed or data labels changed.'])


def write_report(report, output, protected):
    raw_output = Path(output)
    if raw_output.is_symlink() or raw_output.exists():
        raise ValueError('refusing to overwrite outcomes output or follow output symlink')
    output = raw_output.resolve()
    if output.exists():
        raise ValueError('refusing to overwrite outcomes output')
    for root in protected:
        root = Path(root).resolve()
        if output == root or root in output.parents:
            raise ValueError('output must be outside immutable input roots')
    output.mkdir(parents=True, exist_ok=False)
    path = output/'hard_stt_outcomes.json'
    with path.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return dict(path=str(path), sha256=goal.file_sha(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'manifest', 'teacher-selections', 'checkpoint-sha', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--checkpoint-step', type=int, required=True)
    parser.add_argument('--baseline', default=str(goal.DEFAULT_BASELINE))
    parser.add_argument('--hard-episodes', default=str(DEFAULT_PLAN))
    args = parser.parse_args(argv)
    report = diagnose(args.candidate, args.baseline, args.manifest, args.teacher_selections,
                      args.checkpoint_sha, args.checkpoint_step, args.hard_episodes)
    artifact = write_report(report, args.output,
        (args.candidate, args.baseline, Path(args.hard_episodes).parent))
    print(json.dumps(dict(audit_status=report['audit_status'], goal_status=report['goal_status'],
        original91_recovered=report['groups']['original91_recovered']['count'],
        original91_still_failed=report['groups']['original91_still_failed']['count'], **artifact)))


if __name__ == '__main__':
    main()
