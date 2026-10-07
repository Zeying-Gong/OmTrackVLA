"""Audit the fixed STT goal offline, without changing rollout or old reports.

Reuses the pinned 61377 full paired-start audit. Candidate starts are compared
with that baseline's stored raw-sensor hash and state; no media is re-decoded.
A valid NOT_MET report is an audit result, not a tool failure.
"""
import argparse
import hashlib
import json
from pathlib import Path

from wa.tools.merge_student_partitions import read_partitions
from wa.wm.dual_teacher_metrics import summarize_teachers
from wa.wm.full_mixed_contract import (
    CHECKPOINT_SHA as OLD_BASELINE_SHA, MANIFEST_SHA, TASKS,
    summarize, validate_checkpoint_identity,
)
from wa.wm.recovery_replay import check_dynamic_state
from wa.wm.student_eval_contract import validate_student_rows
from wa.wm.student_eval_finalize import TEACHER_SHA, load_teachers, superiority
from wa.wm.student_pair_audit import finite_tree

SCHEMA = 'wa.student-goal-audit.v1'
TARGETS = dict(stt=1289, dt=1173, at=1203)
BASELINE_COUNTS = dict(stt=1276, dt=1173, at=1203)
BASELINE_SHA = 'b5236a21f2d2695780029503c97e339c8350dc4f7337d05ec7e8692b9b9c327f'
BASELINE_STEP = 59065
BASELINE_PINS = {
    'summary.json': '30dfdbd9263ca0ba3615cb1845938de235c7e1b15b2141edb5ac584622f6bbf3',
    'combined_episodes.jsonl': '62ac4c5a2bb28e8db89dbf1f67c968725977ca2bb321abcc78536b16e233d14b',
    'student_teacher_pair_audit.json': 'dc98149bdcfb6baf05d1fae167de572b476679f48384dd91ea92e57851645844',
}
PROJECT = Path('/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928')
DEFAULT_BASELINE = PROJECT/'artifacts/student61377_full_audit_20261007_v1'
INSET_LIMIT = 'Student trained on this evaluation set; not unseen generalization'


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def merge_hashes(*inventories):
    result = {}
    for inventory in inventories:
        for path, value in inventory.items():
            if path in result and result[path] != value:
                raise ValueError('conflicting source evidence: ' + path)
            result[path] = value
    return result


def load_bundle(root, manifest, contract, *, pins=None):
    """Recheck the completed merge and all 24 source shards, but no media."""
    root = Path(root).resolve()
    hashes = {str(root/name): file_sha(root/name) for name in BASELINE_PINS}
    if pins is not None and any(hashes[str(root/name)] != value for name, value in pins.items()):
        raise ValueError('pinned 61377 baseline artifact changed')
    summary = json.loads((root/'summary.json').read_text())
    rows = [json.loads(s) for s in (root/'combined_episodes.jsonl').read_text().splitlines() if s]
    pair = json.loads((root/'student_teacher_pair_audit.json').read_text())
    original, source_hashes = read_partitions(summary['partition_roots'], manifest, contract)
    if original != rows or len(source_hashes) != 78 or source_hashes != summary.get('source_hashes'):
        raise ValueError('merge differs from original shards or saved source inventory')
    hashes = merge_hashes(hashes, source_hashes)
    return dict(rows=rows, summary=summary, pair=pair, root=str(root), hashes=hashes)


def validate_bundle_data(bundle, manifest, contract, teachers, teacher_metrics):
    """Pure report/row protocol checks, shared by the real audit and fixtures."""
    validate_checkpoint_identity(**contract)
    rows, saved, pair = bundle['rows'], bundle['summary'], bundle['pair']
    validate_student_rows(rows, contract)
    computed = summarize(rows, manifest, **contract)
    for key, value in computed.items():
        if key != 'limits' and saved.get(key) != value:
            raise ValueError('saved summary differs from rows: ' + key)
    if saved.get('limits') != computed['limits'] + [INSET_LIMIT]:
        raise ValueError('missing or changed evaluation interpretation')
    if (saved.get('experiment') != 'evaluation_set_adaptation_v1' or
            type(saved.get('new_episodes')) is not int or saved['new_episodes'] != 4215 or
            type(saved.get('reused_baseline_episodes')) is not int or saved['reused_baseline_episodes'] != 0):
        raise ValueError('complete new 4215-episode evaluation required')
    if saved.get('teacher_reference_sha256') != TEACHER_SHA:
        raise ValueError('wrong teacher reference')
    comparison = superiority(rows, teachers)
    if saved.get('superiority') != comparison or saved.get('teacher_metrics_percent') != teacher_metrics:
        raise ValueError('saved teacher comparison differs')
    expected = {task + ':' + e['key'] for task in TASKS for e in manifest['tasks'][task]['episodes']}
    if (pair.get('status') != 'PASS' or type(pair.get('episodes')) is not int or
            pair['episodes'] != 4215 or set(pair.get('teacher_artifact_hashes', {})) != expected):
        raise ValueError('incomplete or unverified paired-start audit')
    for row in rows:
        for field in ('success', 'collision'):
            # The frozen runner writes success=False for policy/init failure.
            # Keep JSON boolean outcomes in the full denominator like 0/1.
            if type(row[field]) not in (bool, int, float) or row[field] not in (0, 1):
                raise ValueError('binary JSON outcome required')
        evidence = row.get('initial_pair_evidence')
        if not isinstance(evidence, dict) or not finite_tree(evidence.get('state')):
            raise ValueError('missing/nonfinite persisted start evidence')
        validate_checkpoint_identity(evidence.get('rgb'), 1)
    return {(r['task'], r['key']): r for r in rows}, computed, comparison


def compare_goal(candidate, baseline, manifest, contract, teachers, teacher_metrics):
    """Full paired counts; actual checkpoint identity is supplied, never inferred."""
    if contract['checkpoint_sha'] == OLD_BASELINE_SHA:
        raise ValueError('60502 is not this candidate evaluation')
    base_contract = dict(checkpoint_sha=BASELINE_SHA, step=BASELINE_STEP)
    new, new_metrics, comparison = validate_bundle_data(candidate, manifest, contract, teachers, teacher_metrics)
    old, old_metrics, _ = validate_bundle_data(baseline, manifest, base_contract, teachers, teacher_metrics)
    if candidate['pair'] != baseline['pair']:
        raise ValueError('paired teacher evidence differs from pinned full baseline audit')
    if new.keys() != old.keys():
        raise ValueError('candidate/baseline pairing differs')
    same_checkpoint = contract == base_contract
    if contract['checkpoint_sha'] == BASELINE_SHA and not same_checkpoint:
        raise ValueError('baseline SHA with foreign step')
    if same_checkpoint and candidate['rows'] != baseline['rows']:
        raise ValueError('baseline self-check must use identical frozen rows')
    for key in new:
        expected = old[key]['initial_pair_evidence']
        actual = new[key]['initial_pair_evidence']
        if actual['rgb'] != expected['rgb']:
            raise ValueError('candidate raw start RGB differs from pinned baseline')
        try:
            check_dynamic_state(expected['state'], actual['state'])
        except (KeyError, TypeError, RuntimeError, ValueError) as error:
            raise ValueError('candidate initial dynamic state differs') from error
    tasks = {}
    for task in TASKS:
        keys = sorted(k for k in new if k[0] == task)
        if len(keys) != 1405:
            raise ValueError('full per-task denominator required')
        successes = sum(int(new[k]['success']) for k in keys)
        previous = sum(int(old[k]['success']) for k in keys)
        if previous != BASELINE_COUNTS[task]:
            raise ValueError('baseline success counts changed')
        gained = [k[1] for k in keys if new[k]['success'] and not old[k]['success']]
        lost = [k[1] for k in keys if old[k]['success'] and not new[k]['success']]
        if len(gained) - len(lost) != successes - previous:
            raise ValueError('paired gains/regressions do not explain success delta')
        collisions = sum(int(new[k]['collision']) for k in keys)
        previous_collisions = sum(int(old[k]['collision']) for k in keys)
        invalid = sum(not new[k]['policy_init_valid'] for k in keys)
        previous_invalid = sum(not old[k]['policy_init_valid'] for k in keys)
        tasks[task] = dict(
            episodes=1405, required_success=TARGETS[task], candidate_success=successes,
            baseline_success=previous, met=successes >= TARGETS[task],
            shortfall=max(0, TARGETS[task]-successes), net_success=successes-previous,
            gains=len(gained), regressions=len(lost), gain_keys=gained, regression_keys=lost,
            candidate_SR=100*successes/1405, baseline_SR=100*previous/1405,
            candidate_collision_count=collisions, baseline_collision_count=previous_collisions,
            candidate_CR=100*collisions/1405, baseline_CR=100*previous_collisions/1405,
            candidate_invalid_init_count=invalid, baseline_invalid_init_count=previous_invalid,
            candidate_invalid_init_percent=100*invalid/1405,
            baseline_invalid_init_percent=100*previous_invalid/1405)
    met = all(row['met'] for row in tasks.values())
    return dict(schema=SCHEMA, audit_status='PASS', goal_status='MET' if met else 'NOT_MET',
        episodes=4215, targets=dict(TARGETS), tasks=tasks,
        candidate_checkpoint_sha256=contract['checkpoint_sha'], candidate_checkpoint_step=contract['step'],
        baseline_checkpoint_sha256=BASELINE_SHA, baseline_checkpoint_step=BASELINE_STEP,
        baseline_self_check=same_checkpoint, manifest_sha256=MANIFEST_SHA,
        teacher_reference_sha256=TEACHER_SHA, lightnav_comparison=comparison,
        candidate_metrics_percent=new_metrics['metrics_percent'],
        baseline_metrics_percent=old_metrics['metrics_percent'],
        pair_validation=dict(episodes=4215, source='pinned 61377 complete paired-start audit',
            comparison='stored raw sensor SHA exact; dynamic state atol1e-6; teacher evidence hashes exact',
            scope='Transitive check against immutable audited baseline; no media/RGB re-decoding or teacher artifact rehash'),
        limits=['Evaluation-set adaptation, not untouched-test generalization.',
            'CR is target-person distance ever<0.5m, not general obstacle contact.',
            'Collision and initialization failures reported over full 1405/task; only fixed success counts gate this goal.',
            'LightNav comparison retains its original meaning and different-input-system caveat.',
            'Goal MET is not product safety, real-UWB, latency, or unseen-generalization certification.'])


def audit(candidate, baseline, manifest_path, teacher_path, checkpoint_sha, checkpoint_step, output):
    output = Path(output)
    if output.exists():
        raise ValueError('refusing to overwrite goal audit')
    contract = dict(checkpoint_sha=checkpoint_sha, step=checkpoint_step)
    validate_checkpoint_identity(**contract)
    if file_sha(manifest_path) != MANIFEST_SHA:
        raise ValueError('manifest changed')
    manifest = json.loads(Path(manifest_path).read_text())
    teachers = load_teachers(teacher_path)
    teacher_metrics = summarize_teachers(teachers, manifest)['metrics_percent']
    base = load_bundle(baseline, manifest, dict(checkpoint_sha=BASELINE_SHA, step=BASELINE_STEP), pins=BASELINE_PINS)
    if Path(candidate).resolve() == Path(baseline).resolve():
        new = base
    else:
        new = load_bundle(candidate, manifest, contract)
    report = compare_goal(new, base, manifest, contract, teachers, teacher_metrics)
    hashes = merge_hashes(base['hashes'], new['hashes'], {str(Path(manifest_path).resolve()): MANIFEST_SHA,
                   str(Path(teacher_path).resolve()): TEACHER_SHA,
                   str(Path(__file__).resolve()): file_sha(__file__)})
    for path, expected in hashes.items():
        if file_sha(path) != expected:
            raise ValueError('source changed during goal audit: ' + path)
    report.update(candidate_root=new['root'], baseline_root=base['root'],
                  source_hashes=hashes, pinned_baseline_artifacts=dict(BASELINE_PINS))
    output.mkdir(parents=True, exist_ok=False)
    with (output/'goal_report.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'manifest', 'teacher-selections', 'checkpoint-sha', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--checkpoint-step', required=True, type=int)
    parser.add_argument('--baseline', default=str(DEFAULT_BASELINE))
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    report = audit(args.candidate, args.baseline, args.manifest, args.teacher_selections,
                   args.checkpoint_sha, args.checkpoint_step, args.output)
    print(json.dumps(dict(audit_status=report['audit_status'], goal_status=report['goal_status'],
        candidate_checkpoint_sha256=report['candidate_checkpoint_sha256'],
        candidate_checkpoint_step=report['candidate_checkpoint_step'],
        successes={t: r['candidate_success'] for t, r in report['tasks'].items()},
        baseline_self_check=report['baseline_self_check'], output=str(args.output))))


if __name__ == '__main__':
    main()
