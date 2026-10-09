"""CPU-only same-checkpoint mixed/image comparison, not ablation release.

Re-read both full student bundles and their 48 shards. Teacher selections are
pinned, but stored paired-start evidence is checked transitively: no teacher
media or checkpoint is loaded. This tool never decides the mixed SR goal.
"""
import argparse
import inspect
import json
from pathlib import Path

from wa.tools.audit_student_goal import file_sha, merge_hashes, INSET_LIMIT
from wa.tools.merge_student_partitions import read_partitions
from wa.wm.dual_teacher_metrics import summarize_teachers
from wa.wm.full_mixed_contract import MANIFEST_SHA, TASKS, summarize, validate_checkpoint_identity
from wa.wm.recovery_replay import check_dynamic_state
from wa.wm.student_eval_contract import validate_student_rows
from wa.wm.student_eval_finalize import TEACHER_SHA, load_teachers, superiority
from wa.wm.student_pair_audit import finite_tree

SCHEMA = 'wa.same-checkpoint-uwb-comparison.v1'
CELLS = ('both_success', 'mixed_only_success', 'image_only_success', 'both_failure')
BUNDLE_FILES = ('summary.json', 'combined_episodes.jsonl', 'student_teacher_pair_audit.json')


def output_path(output, protected):
    raw = Path(output).absolute()
    if raw.exists() or any(p.is_symlink() for p in (raw, *raw.parents)):
        raise ValueError('refusing existing output or symlink output/ancestor')
    resolved = raw.resolve()
    for path in protected:
        source = Path(path).resolve()
        if resolved == source or source in resolved.parents or resolved in source.parents:
            raise ValueError('output must be outside and not contain immutable inputs')
    return resolved


def load_bundle(root, manifest, contract, mode):
    root = Path(root).resolve()
    hashes = {str(root/name): file_sha(root/name) for name in BUNDLE_FILES}
    summary = json.loads((root/'summary.json').read_text())
    rows = [json.loads(s) for s in (root/'combined_episodes.jsonl').read_text().splitlines() if s]
    pair = json.loads((root/'student_teacher_pair_audit.json').read_text())
    original, source_hashes = read_partitions(summary['partition_roots'], manifest, contract, mode=mode)
    if original != rows or len(source_hashes) != 78 or source_hashes != summary.get('source_hashes'):
        raise ValueError('merged rows or source inventory differ from original shards')
    for partition in summary['partition_roots'].values():
        for index in range(8):
            ready = json.loads((Path(partition)/f'shard_{index:02d}'/'server_ready.json').read_text())
            if ready.get('seed') != '7+step':
                raise ValueError('missing or foreign persisted ready seed')
    return dict(root=str(root), rows=rows, summary=summary, pair=pair,
                hashes=merge_hashes(hashes, source_hashes))


def validate_bundle(bundle, manifest, contract, mode, teachers, teacher_metrics):
    rows, saved, pair = bundle['rows'], bundle['summary'], bundle['pair']
    validate_student_rows(rows, contract, mode=mode)
    computed = summarize(rows, manifest, mode=mode, **contract)
    for key, value in computed.items():
        expected = value + [INSET_LIMIT] if key == 'limits' else value
        if saved.get(key) != expected:
            raise ValueError('saved summary differs from rows: '+key)
    if (saved.get('experiment') != 'evaluation_set_adaptation_v1' or
            type(saved.get('new_episodes')) is not int or saved['new_episodes'] != 4215 or
            type(saved.get('reused_baseline_episodes')) is not int or saved['reused_baseline_episodes'] != 0):
        raise ValueError('complete new 4215 rows with no reuse required')
    if (saved.get('teacher_reference_sha256') != TEACHER_SHA or
            saved.get('teacher_metrics_percent') != teacher_metrics or
            saved.get('superiority') != superiority(rows, teachers)):
        raise ValueError('saved teacher comparison differs')
    indexed = {(r['task'], r['key']): r for r in rows}
    expected = {t+':'+e['key'] for t in TASKS for e in manifest['tasks'][t]['episodes']}
    inventory = pair.get('teacher_artifact_hashes', {})
    if (pair.get('status') != 'PASS' or type(pair.get('episodes')) is not int or
            pair['episodes'] != 4215 or not isinstance(inventory, dict) or set(inventory) != expected):
        raise ValueError('incomplete saved paired-start evidence')
    for teacher_row in teachers:
        identity = teacher_row['pair']
        key = (identity['task'], identity['key'])
        row = indexed[key]
        evidence = row.get('initial_pair_evidence')
        if not isinstance(evidence, dict) or not finite_tree(evidence.get('state')):
            raise ValueError('missing or nonfinite initial state')
        validate_checkpoint_identity(evidence.get('rgb'), 1)
        if (identity.get('seed') != 7 or identity.get('takeover_step') != 0 or
                evidence['rgb'] != identity.get('initial_rgb_sha256')):
            raise ValueError('student raw start RGB or teacher canonical start differs')
        recorded = inventory[':'.join(key)]
        if not isinstance(recorded, dict) or set(recorded) != {'lightnav', 'oracle'}:
            raise ValueError('missing saved teacher branch evidence')
        for name in ('lightnav', 'oracle'):
            root = Path(teacher_row['branches'][name]['artifact_root'])
            if not root.is_absolute() or '..' in root.parts:
                raise ValueError('absolute safe teacher artifact_root required')
            files = recorded[name]
            required = {str(root/'pair_start.json'), str(root/'observations.json')}
            if not isinstance(files, dict) or len(files) != 3 or not required <= set(files):
                raise ValueError('saved teacher evidence paths differ')
            for path, digest in files.items():
                validate_checkpoint_identity(digest, 1)
                candidate = Path(path)
                if not candidate.is_absolute() or '..' in candidate.parts or root not in candidate.parents:
                    raise ValueError('saved teacher evidence outside branch')
        for field in ('success', 'collision'):
            if type(row[field]) not in (bool, int, float) or row[field] not in (0, 1):
                raise ValueError('binary JSON outcome required')
    return indexed, computed['metrics_percent']


def video_reference(row):
    key = row['key']
    if (not isinstance(key, str) or len(key.split('/')) != 2 or
            any(p in ('', '.', '..') for p in key.split('/')) or key.startswith('/')):
        raise ValueError('unsafe task/key media reference')
    root = Path(row['artifact_root'])
    if not root.is_absolute():
        raise ValueError('absolute artifact_root required')
    return dict(artifact_root=str(root), video_path=str(root/row['task']/'_review'/key/'review.mp4'),
                media_verified_by_this_tool=False)


def compare(mixed, image, manifest, contract, teachers, teacher_metrics):
    validate_checkpoint_identity(**contract)
    if Path(mixed['root']).resolve() == Path(image['root']).resolve():
        raise ValueError('distinct mixed and image bundle roots required')
    left, left_metrics = validate_bundle(mixed, manifest, contract, 'mixed', teachers, teacher_metrics)
    right, right_metrics = validate_bundle(image, manifest, contract, 'image', teachers, teacher_metrics)
    if left.keys() != right.keys() or mixed['pair'] != image['pair']:
        raise ValueError('cross-mode keys or saved teacher pair evidence differ')
    tasks, paired = {}, []
    for task in TASKS:
        keys = sorted(k for k in left if k[0] == task)
        if len(keys) != 1405:
            raise ValueError('full 1405 per task required')
        cells = {name: [] for name in CELLS}
        for key in keys:
            a, b = left[key], right[key]
            ea, eb = a['initial_pair_evidence'], b['initial_pair_evidence']
            if ea['rgb'] != eb['rgb']:
                raise ValueError('cross-mode raw start RGB differs')
            try:
                check_dynamic_state(ea['state'], eb['state'])
            except (KeyError, TypeError, RuntimeError, ValueError) as error:
                raise ValueError('cross-mode initial dynamic state differs') from error
            cell = ('both_success' if a['success'] and b['success'] else
                    'mixed_only_success' if a['success'] else
                    'image_only_success' if b['success'] else 'both_failure')
            cells[cell].append(key[1])
            paired.append(dict(task=task, key=key[1], outcome=cell,
                               mixed=video_reference(a), image=video_reference(b)))
        values = {}
        for mode, records, metrics in (('mixed', left, left_metrics), ('image', right, right_metrics)):
            part = [records[key] for key in keys]
            values[mode] = dict(metrics[task],
                success_count=sum(int(r['success']) for r in part),
                collision_count=sum(int(r['collision']) for r in part),
                invalid_init_percent=100*sum(not r['policy_init_valid'] for r in part)/1405,
                TR_numerator=sum(r['following_step'] for r in part),
                TR_denominator=sum(max(r['total_step'], manifest['reference_steps'].get(r['key'], 0))
                                   for r in part))
        delta = {name: (None if values['mixed'][name] is None or value is None
                       else value-values['mixed'][name]) for name, value in values['image'].items()}
        if (sum(map(len, cells.values())) != 1405 or
                len(cells['image_only_success'])-len(cells['mixed_only_success']) != delta['success_count']):
            raise ValueError('paired counts do not reconcile')
        tasks[task] = dict(**values, delta_image_minus_mixed=delta,
                           paired_counts={k: len(v) for k, v in cells.items()}, paired_keys=cells)
    return dict(schema=SCHEMA, audit_status='PASS_STORED_EVIDENCE_COMPARISON_ONLY',
        ablation_release=False, episodes=4215, checkpoint_sha256=contract['checkpoint_sha'],
        checkpoint_step=contract['step'], tasks=tasks, paired=paired,
        manifest_sha256=MANIFEST_SHA, teacher_reference_sha256=TEACHER_SHA,
        pair_validation=dict(episodes=4215, raw_sensor_sha_exact=True, dynamic_state_atol=1e-6,
            saved_teacher_pair_evidence_equal=True,
            scope='Transitive saved-evidence comparison; no teacher artifacts rehashed or media decoded'),
        limits=[
            'Evaluation-set adaptation; not unseen-test generalization or a goal MET decision.',
            'Same-weight inference removal of ideal UWB, not no-UWB retraining or real-UWB robustness.',
            'Stored initial sensor SHA/state do not prove seed, first-box policy, physics, source/config, '
            'actual RPC/observer isolation; those require separate frozen-runtime/interface evidence.',
            'Persisted server_ready seed is checked as 7+step; execution is not independently replayed.',
            'Video paths are references only; use each mode full review/media audit separately.',
            'TR=sum(following_step)/sum(max(total_step,reference_step)); macro_TR is separate.',
            'Invalid initializations remain in 1405/task; CR is target distance ever<0.5m, not obstacles.',
            'No GPU, checkpoint, simulator or future-trajectory equivalence check is performed.'
        ])


def diagnose(mixed, image, manifest_path, teacher_path, checkpoint_sha, checkpoint_step):
    contract = dict(checkpoint_sha=checkpoint_sha, step=checkpoint_step)
    validate_checkpoint_identity(**contract)
    if Path(mixed).resolve() == Path(image).resolve():
        raise ValueError('distinct mixed and image bundle roots required')
    if file_sha(manifest_path) != MANIFEST_SHA:
        raise ValueError('manifest changed')
    code = {str(Path(inspect.getfile(fn)).resolve()) for fn in
            (diagnose, read_partitions, summarize, validate_student_rows, check_dynamic_state,
             summarize_teachers, load_teachers, finite_tree, file_sha)}
    code_hashes = {p: file_sha(p) for p in code}
    manifest = json.loads(Path(manifest_path).read_text())
    teachers = load_teachers(teacher_path)
    teacher_metrics = summarize_teachers(teachers, manifest)['metrics_percent']
    bundles = {mode: load_bundle(root, manifest, contract, mode)
               for mode, root in (('mixed', mixed), ('image', image))}
    partitions = [Path(p).resolve() for b in bundles.values()
                  for p in b['summary']['partition_roots'].values()]
    if len(set(partitions)) != 6:
        raise ValueError('six distinct source partition roots required')
    report = compare(bundles['mixed'], bundles['image'], manifest, contract, teachers, teacher_metrics)
    hashes = merge_hashes(*(b['hashes'] for b in bundles.values()),
        {str(Path(manifest_path).resolve()): MANIFEST_SHA,
         str(Path(teacher_path).resolve()): TEACHER_SHA},
        code_hashes)
    for path, expected in hashes.items():
        if file_sha(path) != expected:
            raise ValueError('source changed during cross-mode comparison: '+path)
    report.update(mixed_root=bundles['mixed']['root'], image_root=bundles['image']['root'],
                  partition_roots={mode: b['summary']['partition_roots'] for mode, b in bundles.items()},
                  source_hashes=hashes,
                  protected_teacher_artifact_roots=sorted({
                      teacher['branches'][name]['artifact_root']
                      for teacher in teachers for name in ('lightnav', 'oracle')}))
    return report


def write_report(report, output, protected):
    protected = list(protected) + report['protected_teacher_artifact_roots']
    destination = output_path(output, protected)
    payload = json.dumps(report, indent=2, allow_nan=False)
    for path, expected in report['source_hashes'].items():
        if file_sha(path) != expected:
            raise ValueError('source changed before writing comparison: '+path)
    destination = output_path(destination, protected)
    destination.mkdir(parents=True, exist_ok=False)
    path = destination/'comparison.json'
    with path.open('x') as stream:
        stream.write(payload)
    return dict(path=str(path), sha256=file_sha(path))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('mixed', 'image', 'manifest', 'teacher-selections', 'checkpoint-sha', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--checkpoint-step', required=True, type=int)
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    protected = [args.mixed, args.image, Path(args.manifest).parent, Path(args.teacher_selections).parent]
    output_path(args.output, protected)
    report = diagnose(args.mixed, args.image, args.manifest, args.teacher_selections,
                      args.checkpoint_sha, args.checkpoint_step)
    protected += [p for roots in report['partition_roots'].values() for p in roots.values()]
    artifact = write_report(report, args.output, protected)
    print(json.dumps(dict(audit_status=report['audit_status'], episodes=4215,
        ablation_release=False, delta_success={t: r['delta_image_minus_mixed']['success_count']
                                              for t, r in report['tasks'].items()}, **artifact)))


if __name__ == '__main__':
    main()
