"""Compare a complete new student evaluation with the pinned 61609 result.

This is a CPU-only paired comparison, not a replacement for the fixed 61377
goal audit. No checkpoint, media, model or simulator is loaded.
"""
import argparse
import json
from pathlib import Path

from wa.tools import audit_student_goal as goal
from wa.wm.dual_teacher_metrics import summarize_teachers
from wa.wm.recovery_replay import check_dynamic_state
from wa.wm.student_eval_finalize import load_teachers

SCHEMA = 'wa.student61609-comparison.v1'
BASELINE_SHA = 'c510c04d987d141423401a25323ea353314107a16f4dac5d2901370ab199fa52'
BASELINE_STEP = 59716
BASELINE_COUNTS = dict(stt=1279, dt=1178, at=1207)
BASELINE_PINS = {
    'summary.json': '223b4b8140e79842408bc7104d16f75b7b4ff1d409354ee547b00e4b49f937f7',
    'combined_episodes.jsonl': '0ab45e1b35bb0b8809fcc77fcaabca59b35bab0839507d716c371ce2d5a6f358',
    'student_teacher_pair_audit.json': 'dc98149bdcfb6baf05d1fae167de572b476679f48384dd91ea92e57851645844',
}
DEFAULT_BASELINE = goal.PROJECT/'artifacts/student61609_full_audit_20261007_v1'


def compare(candidate, baseline, manifest, contract, teachers, teacher_metrics):
    """Validate both complete bundles before computing any paired outcome."""
    if contract['checkpoint_sha'] == goal.OLD_BASELINE_SHA:
        raise ValueError('60502 is not the new student candidate')
    base_contract = dict(checkpoint_sha=BASELINE_SHA, step=BASELINE_STEP)
    new, new_metrics, _ = goal.validate_bundle_data(
        candidate, manifest, contract, teachers, teacher_metrics)
    old, old_metrics, _ = goal.validate_bundle_data(
        baseline, manifest, base_contract, teachers, teacher_metrics)
    if candidate['pair'] != baseline['pair']:
        raise ValueError('paired teacher evidence differs from pinned 61609')
    if new.keys() != old.keys():
        raise ValueError('candidate and 61609 keys differ')
    same_checkpoint = contract == base_contract
    if contract['checkpoint_sha'] == BASELINE_SHA and not same_checkpoint:
        raise ValueError('61609 SHA with foreign step')
    if same_checkpoint and candidate['rows'] != baseline['rows']:
        raise ValueError('61609 self-check requires identical frozen rows')
    for key in new:
        expected, actual = (r[key]['initial_pair_evidence'] for r in (old, new))
        if expected['rgb'] != actual['rgb']:
            raise ValueError('candidate raw start RGB differs from 61609')
        try:
            check_dynamic_state(expected['state'], actual['state'])
        except (KeyError, TypeError, RuntimeError, ValueError) as error:
            raise ValueError('candidate initial dynamic state differs from 61609') from error
    tasks = {}
    for task in goal.TASKS:
        keys = sorted(k for k in new if k[0] == task)
        if len(keys) != 1405:
            raise ValueError('full 1405 per task required')
        previous = sum(int(old[k]['success']) for k in keys)
        successes = sum(int(new[k]['success']) for k in keys)
        if previous != BASELINE_COUNTS[task]:
            raise ValueError('pinned 61609 success counts changed')
        gains = [k[1] for k in keys if new[k]['success'] and not old[k]['success']]
        regressions = [k[1] for k in keys if old[k]['success'] and not new[k]['success']]
        common_success = sum(bool(old[k]['success']) and bool(new[k]['success']) for k in keys)
        common_failed = [k[1] for k in keys if not old[k]['success'] and not new[k]['success']]
        if (len(gains)-len(regressions) != successes-previous or
                common_success+len(gains) != successes or
                common_success+len(regressions) != previous or
                common_success+len(common_failed)+len(gains)+len(regressions) != 1405):
            raise ValueError('paired outcome reconciliation failed')
        values = dict(episodes=1405, candidate_success=successes, baseline_success=previous,
            net_success=successes-previous, gains=len(gains), regressions=len(regressions),
            gain_keys=gains, regression_keys=regressions, common_success=common_success,
            common_failure=len(common_failed), common_failure_keys=common_failed)
        for label, records, count in (('candidate', new, successes), ('baseline', old, previous)):
            collisions = sum(int(records[k]['collision']) for k in keys)
            invalid = sum(not records[k]['policy_init_valid'] for k in keys)
            values.update({label+'_SR': 100*count/1405,
                label+'_collision_count': collisions, label+'_CR': 100*collisions/1405,
                label+'_invalid_init_count': invalid, label+'_invalid_init_percent': 100*invalid/1405})
        tasks[task] = values
    return dict(schema=SCHEMA, audit_status='PASS', comparison_baseline=61609,
        episodes=4215, tasks=tasks, baseline_self_check=same_checkpoint,
        candidate_checkpoint_sha256=contract['checkpoint_sha'],
        candidate_checkpoint_step=contract['step'],
        baseline_checkpoint_sha256=BASELINE_SHA, baseline_checkpoint_step=BASELINE_STEP,
        candidate_metrics_percent=new_metrics['metrics_percent'],
        baseline_metrics_percent=old_metrics['metrics_percent'],
        manifest_sha256=goal.MANIFEST_SHA, teacher_reference_sha256=goal.TEACHER_SHA,
        pair_validation=dict(episodes=4215, reference='pinned 61609 complete paired-start audit',
            comparison='raw sensor SHA exact; dynamic state atol1e-6; teacher evidence hashes exact',
            scope='transitive verification; no media decoding or teacher artifact rehash'),
        goal_audit_required='Run audit_student_goal with its original 61377 baseline and fixed targets.',
        limits=['Evaluation-set adaptation, not untouched-test generalization.',
            'All initialization failures remain in 1405/task denominators.',
            'TR in metrics_percent is reference-step normalized; macro_TR remains separate.',
            'CR is target-person distance ever<0.5m, not general obstacle contact.',
            'Different GPU hardware does not establish bitwise equivalence or sampling-only causality.',
            'No goal_status is decided by this comparison. No model, media or checkpoint was loaded.'])


def diagnose(candidate, baseline, manifest_path, teacher_path, checkpoint_sha, checkpoint_step):
    contract = dict(checkpoint_sha=checkpoint_sha, step=checkpoint_step)
    goal.validate_checkpoint_identity(**contract)
    if goal.file_sha(manifest_path) != goal.MANIFEST_SHA:
        raise ValueError('manifest changed')
    manifest = json.loads(Path(manifest_path).read_text())
    teachers = load_teachers(teacher_path)
    teacher_metrics = summarize_teachers(teachers, manifest)['metrics_percent']
    base = goal.load_bundle(baseline, manifest,
        dict(checkpoint_sha=BASELINE_SHA, step=BASELINE_STEP), pins=BASELINE_PINS)
    same_root = Path(candidate).resolve() == Path(baseline).resolve()
    new = base if same_root else goal.load_bundle(candidate, manifest, contract)
    report = compare(new, base, manifest, contract, teachers, teacher_metrics)
    hashes = goal.merge_hashes(base['hashes'], new['hashes'], {
        str(Path(manifest_path).resolve()): goal.MANIFEST_SHA,
        str(Path(teacher_path).resolve()): goal.TEACHER_SHA,
        str(Path(__file__).resolve()): goal.file_sha(__file__),
        str(Path(goal.__file__).resolve()): goal.file_sha(goal.__file__)})
    for path, expected in hashes.items():
        if goal.file_sha(path) != expected:
            raise ValueError('source changed during comparison: '+path)
    report.update(candidate_root=new['root'], baseline_root=base['root'],
        candidate_partition_roots=new['summary']['partition_roots'],
        baseline_partition_roots=base['summary']['partition_roots'],
        source_hashes=hashes, pinned_baseline_artifacts=dict(BASELINE_PINS))
    return report


def output_path(output, protected):
    raw = Path(output)
    if raw.exists() or raw.is_symlink():
        raise ValueError('refusing to overwrite comparison output or follow symlink')
    resolved = raw.resolve()
    for path in protected:
        source = Path(path).resolve()
        if resolved == source or source in resolved.parents:
            raise ValueError('output must be outside immutable input roots')
    return resolved


def write_report(report, output, protected):
    output = output_path(output, protected)
    payload = json.dumps(report, indent=2, allow_nan=False)
    output.mkdir(parents=True, exist_ok=False)
    path = output/'comparison.json'
    with path.open('x') as stream:
        stream.write(payload)
    return dict(path=str(path), sha256=goal.file_sha(path))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'manifest', 'teacher-selections', 'checkpoint-sha', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--checkpoint-step', type=int, required=True)
    parser.add_argument('--baseline', default=str(DEFAULT_BASELINE))
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    protected = [args.candidate, args.baseline, Path(args.manifest).parent,
                 Path(args.teacher_selections).parent]
    output_path(args.output, protected)  # Fail before any expensive input read.
    report = diagnose(args.candidate, args.baseline, args.manifest,
                      args.teacher_selections, args.checkpoint_sha, args.checkpoint_step)
    protected += list(report['candidate_partition_roots'].values())
    protected += list(report['baseline_partition_roots'].values())
    artifact = write_report(report, args.output, protected)
    print(json.dumps(dict(audit_status=report['audit_status'], comparison_baseline=61609,
        successes={t: r['candidate_success'] for t, r in report['tasks'].items()},
        gains={t: r['gains'] for t, r in report['tasks'].items()},
        regressions={t: r['regressions'] for t, r in report['tasks'].items()}, **artifact)))


if __name__ == '__main__':
    main()
