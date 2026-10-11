"""Audit full image-only-retrained WA results against direct input removal.

Both sides run the same frozen image/no-UWB evaluator on the same 4,215 keys,
but use different weights. This is NOT the same-checkpoint UWB ablation.
The legacy merge summary's image-mode limit says "not no-UWB training";
that describes the old 62256 checkpoint only. This tool preserves that raw
summary as evidence and emits a corrected derived summary for Job 63395.
No GPU, simulator, video decoding, or new evaluation is performed here.
"""
import argparse
import copy
import inspect
import json
from pathlib import Path

from wa.tools.audit_student_goal import file_sha, merge_hashes
from wa.tools.compare_uwb_modes import (
    load_bundle, output_path, validate_bundle, video_reference, write_report,
)
from wa.tools.merge_student_partitions import read_partitions
from wa.wm.dual_teacher_metrics import summarize_teachers
from wa.wm.full_mixed_contract import MANIFEST_SHA, TASKS, summarize
from wa.wm.recovery_replay import check_dynamic_state
from wa.wm.student_eval_contract import validate_student_rows
from wa.wm.student_eval_finalize import TEACHER_SHA, load_teachers
from wa.wm.student_pair_audit import finite_tree

SCHEMA = 'wa.image-retrain-stored-result-audit.v1'
OLD_SHA = '40915b366ee5a2ef5967e2ce49a94d2b0f45f149955dd5cccf85ac3d6ab178fc'
NEW_SHA = 'ac8ca38158d89ab6dcd11a9f604b09d37a3db7db63075091cfd66458a2171efa'
PARENT_SHA = 'ab39144b0490937ce43c4ab28e2d90cdb0c560f700d3364a29ce1dc9a08b999d'
OLD_AUDIT_SHA = '1698dc68cdcc93c8aaca67c3de93bfd33557ae8f820db90455aa6b09825e3735'
NEW_AUDIT_SHA = '619e35c26916c490c164ce956929e10bc82ce72f473643b76a97348c37411203'
OLD_SOURCE = '199385cd9c826c8f21308ad99b6a8375396d9c90'
NEW_SOURCE = '76635eb257e43ab2c8194e5b358cac023a3351a0'
OLD_TRAIN_CONFIG = '08553eb73e8d2db4ae2fd37ab1c322bc7fbf77ac4a3ba3a7c59ae2508a2f2e69'
NEW_TRAIN_CONFIG = 'f5413cf82f95d6d0d032675445ecb58b7d307d59a907268d02400d34b6a4c0ea'
STEP = 61252
PARENT_STEP = 22707
UPDATES = 38545
EXPOSURES = dict(actual_base=726631, actual_teacher=457641,
                 actual_recovery=49152, actual_total=1233424)
MODE_EXPOSURE = dict(image=1233424, point=0, mixed=0, total=1233424)
CELLS = ('both_success', 'direct_only_success', 'retrained_only_success', 'both_failure')
PROJECT = Path('/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928')
JOBS = Path('/data/nas_ray/project/md-ak/users/zeying.gong')
OLD_CHECKPOINT = JOBS/'job_62256/task_73511/wa_failure_state_train_a800_v1/checkpoint.pt'
NEW_CHECKPOINT = JOBS/'job_63395/task_74738/wa_failure_state_image_only_a800_v1/checkpoint.pt'
PARENT_CHECKPOINT = JOBS/'job_59866/task_70728/wa_history_repeat_v1/checkpoint.pt'
OLD_EVAL_CONFIGS = {
    'stt': '1c22c052f9802d03bac6eb22e9ed9b0f948df7a6b64b257b5855f54997e50758',
    'dt': '04f82200fa66522f852cb7f012752e7ddaedbe2e9b385de9fd1653528b33b263',
    'at': '79c584cc1899585cedca760e64ae354be7e85fa26253711ca8fdc87db17f47c2',
}
NEW_EVAL_CONFIGS = {
    'stt': '9abd61cea349df6bc6f03cf2d4f89d9a5571c854d5a79df5e7e81ed5366e5026',
    'dt': '576130e01691563406ca13c5f48f075a5853dea0ba87b403f22189c16a1db6f4',
    'at': 'f670dc38ddea8fb848a8d31b9e3189ddb292d2a38ddb382614d46627d816effd',
}
OLD_MERGED = PROJECT/'artifacts/student62256_image_full_audit_20261010_v1'
OLD_MERGED_PINS = {
    'summary.json': '2d3c7338318a29102077c38fcbd10482510b993a0604ec6699ccf772e6f6fd15',
    'combined_episodes.jsonl': 'd01d36417c29801a211b213d3d232727b0c08da5d9bca1a05653c1f2fefea0ef',
    'student_teacher_pair_audit.json': 'dc98149bdcfb6baf05d1fae167de572b476679f48384dd91ea92e57851645844',
}
OLD_PARTITIONS = {
    'stt': JOBS/'job_63245/task_74578/wa_student62256_stt_image_4090_v1',
    'dt': JOBS/'job_63246/task_74579/wa_student62256_dt_image_4090_v1',
    'at': JOBS/'job_63247/task_74580/wa_student62256_at_image_4090_v1',
}
NEW_PARTITIONS = {
    'stt': JOBS/'job_63933/task_75322/wa_student63395_stt_image_4090_v1',
    'dt': JOBS/'job_63934/task_75323/wa_student63395_dt_image_4090_v1',
    'at': JOBS/'job_63935/task_75324/wa_student63395_at_image_4090_v1',
}


def exact(document, fields, label):
    for name, expected in fields.items():
        if document.get(name) != expected:
            raise ValueError(label + ' differs: ' + name)


def validate_training_audits(old, new):
    """Separate weight provenance: shared parent/exposure, distinct input training."""
    common = dict(parent_checkpoint=str(PARENT_CHECKPOINT), parent_sha256=PARENT_SHA,
                  final_step=STEP, completed_epochs=2, closed_loop=False)
    for label, audit, schema, status, checkpoint, sha, source, config in (
        ('direct', old, 'failure_state_training_audit_v1',
         'TRAINING_ARTIFACT_AUDIT_PASS_OFFLINE_ONLY', OLD_CHECKPOINT,
         OLD_SHA, OLD_SOURCE, OLD_TRAIN_CONFIG),
        ('retrained', new, 'failure_state_image_only_training_audit_v1',
         'TRAINING_ARTIFACT_AUDIT_PASS_OFFLINE_ONLY_IMAGE_INPUT', NEW_CHECKPOINT,
         NEW_SHA, NEW_SOURCE, NEW_TRAIN_CONFIG),
    ):
        exact(audit, dict(common, schema=schema, status=status,
                          checkpoint=str(checkpoint), checkpoint_sha256=sha,
                          source_commit=source, config_sha256=config), label+' training audit')
        exact(audit.get('optimizer_continuation', {}),
              dict(parent_step=PARENT_STEP, final_step=STEP, updates=UPDATES),
              label+' optimizer continuation')
        exact(audit.get('exposure', {}).get('calculated', {}), EXPOSURES,
              label+' executed exposure')
        if audit.get('source_hashes', {}).get(str(checkpoint)) != sha:
            raise ValueError(label+' audit lacks checkpoint byte hash')
        if len(set(audit.get('hardware', {}).get('unique_gpu_uuids', []))) != 8:
            raise ValueError(label+' training audit lacks eight distinct A800 UUIDs')
    exact(new, dict(train_input_mode='image', mode_exposure=MODE_EXPOSURE),
          'image-only retraining')
    exact(new.get('worker_postcheck', {}),
          dict(train_input_mode='image', mode_exposure=MODE_EXPOSURE,
               status='TRAINING_WORKER_POSTCHECK_PASS_OFFLINE_ONLY'),
          'image-only worker postcheck')
    return dict(old=dict(checkpoint=str(OLD_CHECKPOINT), checkpoint_sha256=OLD_SHA,
                         step=STEP, training_audit_status=old['status'],
                         train_source_commit=OLD_SOURCE, train_config_sha256=OLD_TRAIN_CONFIG,
                         input_interpretation='direct image/no-UWB inference from non-image-only-trained 62256'),
                new=dict(checkpoint=str(NEW_CHECKPOINT), checkpoint_sha256=NEW_SHA,
                         step=STEP, training_audit_status=new['status'],
                         train_source_commit=NEW_SOURCE, train_config_sha256=NEW_TRAIN_CONFIG,
                         train_input_mode='image', mode_exposure=MODE_EXPOSURE),
                same_parent_sha256=PARENT_SHA, same_parent_step=PARENT_STEP,
                independent_updates_each=UPDATES)


def corrected_new_summary(saved):
    """Preserve legacy merge metrics; correct only the derived training description."""
    if saved.get('mode') != 'image':
        raise ValueError('new merge is not image mode')
    result = copy.deepcopy(saved)
    limits = result.get('limits')
    legacy = ('RGB+first GT bbox; no UWB at inference; same model interface, '
              'not no-UWB training; no text')
    if not isinstance(limits, list) or limits.count(legacy) != 1:
        raise ValueError('unrecognized legacy image summary interpretation')
    result['limits'] = [
        'RGB+first GT bbox; no UWB at inference; checkpoint retrained image-only '
        'for one independent epoch from the shared 59866 parent; no text'
        if item == legacy else item for item in limits
    ]
    result['training_provenance_correction'] = dict(
        old_merge_limit=legacy, applies_to='Job 63395 image-only retrained checkpoint',
        original_summary_unchanged=True,
        source='pinned independent terminal training audit; not inferred from closed-loop SR')
    return result


def pair_results(old, new, old_metrics, new_metrics, manifest):
    """All failures, collisions and invalid initializations stay in 1,405 denominators."""
    if old.keys() != new.keys():
        raise ValueError('different direct/retrained image task-key sets')
    tasks, paired = {}, []
    for task in TASKS:
        keys = sorted(key for key in old if key[0] == task)
        if len(keys) != 1405:
            raise ValueError('full 1405 per task required: '+task)
        cells = {name: [] for name in CELLS}
        for identity in keys:
            direct, retrained = old[identity], new[identity]
            start_a, start_b = direct['initial_pair_evidence'], retrained['initial_pair_evidence']
            if (start_a['rgb'] != start_b['rgb'] or
                    direct.get('initial_rgb_sha256') != retrained.get('initial_rgb_sha256')):
                raise ValueError('different initial RGB: '+str(identity))
            try:
                check_dynamic_state(start_a['state'], start_b['state'])
            except (KeyError, TypeError, RuntimeError, ValueError) as error:
                raise ValueError('different initial dynamic state: '+str(identity)) from error
            if (direct.get('semantic_protocol') != retrained.get('semantic_protocol') or
                    direct.get('semantic_ply_repaired') != retrained.get('semantic_ply_repaired') or
                    direct.get('initialization_repair') != retrained.get('initialization_repair')):
                raise ValueError('different semantic or first-box repair: '+str(identity))
            cell = ('both_success' if direct['success'] and retrained['success'] else
                    'direct_only_success' if direct['success'] else
                    'retrained_only_success' if retrained['success'] else 'both_failure')
            cells[cell].append(identity[1])
            paired.append(dict(task=task, key=identity[1], outcome=cell,
                               direct=video_reference(direct),
                               retrained=video_reference(retrained)))
        values = {}
        for name, rows, metrics in (('direct', old, old_metrics),
                                    ('retrained', new, new_metrics)):
            part = [rows[key] for key in keys]
            values[name] = dict(metrics[task],
                success_count=sum(int(row['success']) for row in part),
                collision_count=sum(int(row['collision']) for row in part),
                invalid_init_count=sum(not row['policy_init_valid'] for row in part),
                TR_numerator=sum(row['following_step'] for row in part),
                TR_denominator=sum(max(row['total_step'],
                                       manifest['reference_steps'].get(row['key'], 0))
                                   for row in part))
        delta = {field: (None if value is None or values['direct'][field] is None
                         else value-values['direct'][field])
                 for field, value in values['retrained'].items()}
        if (sum(len(group) for group in cells.values()) != 1405 or
                len(cells['retrained_only_success'])-len(cells['direct_only_success'])
                != delta['success_count']):
            raise ValueError('paired cell accounting differs: '+task)
        tasks[task] = dict(**values, delta_retrained_minus_direct=delta,
                           paired_counts={name: len(group) for name, group in cells.items()},
                           paired_keys=cells)
        if task == 'stt':
            tasks[task]['stt_image_strictly_above_80_goal'] = (
                values['retrained']['success_count'] >= 1125)
    return tasks, paired


def compare(old_bundle, new_bundle, manifest, teachers, teacher_metrics, provenance):
    old_contract = dict(checkpoint_sha=OLD_SHA, step=STEP)
    new_contract = dict(checkpoint_sha=NEW_SHA, step=STEP)
    if OLD_SHA == NEW_SHA or Path(old_bundle['root']).resolve() == Path(new_bundle['root']).resolve():
        raise ValueError('distinct checkpoint SHA and bundle roots required')
    direct, old_metrics = validate_bundle(old_bundle, manifest, old_contract, 'image',
                                          teachers, teacher_metrics)
    retrained, new_metrics = validate_bundle(new_bundle, manifest, new_contract, 'image',
                                             teachers, teacher_metrics)
    if old_bundle['pair'] != new_bundle['pair']:
        raise ValueError('teacher paired-start evidence differs')
    tasks, paired = pair_results(direct, retrained, old_metrics, new_metrics, manifest)
    return dict(schema=SCHEMA, audit_status='PASS_STORED_EVIDENCE_DIFFERENT_WEIGHTS_ONLY',
        episodes=4215, input_mode='image', old_checkpoint_sha256=OLD_SHA,
        new_checkpoint_sha256=NEW_SHA, checkpoint_step=STEP,
        old_root=old_bundle['root'], new_root=new_bundle['root'],
        training_provenance=provenance,
        derived_new_summary=corrected_new_summary(new_bundle['summary']),
        tasks=tasks, paired=paired,
        manifest_sha256=MANIFEST_SHA, teacher_reference_sha256=TEACHER_SHA,
        pair_validation=dict(episodes=4215, raw_sensor_sha_exact=True,
            dynamic_state_atol=1e-6, saved_teacher_pair_evidence_equal=True,
            scope='Stored initial evidence and original 24 shards per model; no simulator replay'),
        limits=[
            'Evaluation-set adaptation upper bound; not unseen-test generalization.',
            'This compares different weights under identical image/no-UWB inference. '
            'It is not a same-weight UWB-input ablation or causal isolation of training.',
            'The original new merge summary is immutable and inherits a false training-mode '
            'sentence; derived_new_summary corrects its wording without changing any metric.',
            'No media decoded or raw teacher artifacts rehashed by this tool; run the separate '
            'first-JPEG/video review and inspect worker logs and scheduler provenance.',
            'TR=sum(following_step)/sum(max(total_step, reference_step)); macro_TR separately.',
            'Invalid and failed episodes stay in the 1405/task denominator. CR means target-person '
            'distance ever<0.5m, not general obstacle collision.',
            'No actual GPU placement, RPC isolation, source checkout or simulator physics '
            'equivalence is proved here; verify separate frozen-runtime evidence.'
        ])


def bound_summary(root, partitions, role):
    root = Path(root)
    if not root.is_absolute() or root.resolve() != root:
        raise ValueError(role+' merged root must be canonical and non-symlink')
    expected = {task: str(path) for task, path in partitions.items()}
    summary = json.loads((root/'summary.json').read_text())
    if summary.get('partition_roots') != expected:
        raise ValueError(role+' merged partition roots differ from pinned Job/Task IDs')
    for path in partitions.values():
        if path.resolve() != path or not path.is_dir():
            raise ValueError(role+' task root is missing or points through a symlink')
    return summary


def validate_bound_roots(old_merged, new_merged):
    """Bind self-declared merge metadata to known submissions before shard reads."""
    old_root, new_root = Path(old_merged), Path(new_merged)
    if old_root != OLD_MERGED:
        raise ValueError('old merged root is not the audited 62256 image bundle')
    if (new_root == old_root or old_root in new_root.parents or
            new_root in old_root.parents or new_root.resolve() == old_root.resolve()):
        raise ValueError('new merged root must be distinct from old audited bundle')
    bound_summary(old_root, OLD_PARTITIONS, 'old')
    bound_summary(new_root, NEW_PARTITIONS, 'new')
    hashes = {str(old_root/name): sha for name, sha in OLD_MERGED_PINS.items()}
    for path, expected in hashes.items():
        if file_sha(path) != expected:
            raise ValueError('audited old merged file changed: '+path)
    return hashes


def diagnose(old_merged, new_merged, manifest_path, teacher_path, old_training_audit,
             new_training_audit):
    bound_files = validate_bound_roots(old_merged, new_merged)
    if file_sha(manifest_path) != MANIFEST_SHA:
        raise ValueError('frozen 4215 manifest changed')
    if file_sha(teacher_path) != TEACHER_SHA:
        raise ValueError('frozen teacher selections changed')
    pinned = {str(Path(old_training_audit).resolve()): OLD_AUDIT_SHA,
              str(Path(new_training_audit).resolve()): NEW_AUDIT_SHA,
              str(OLD_CHECKPOINT): OLD_SHA, str(NEW_CHECKPOINT): NEW_SHA}
    for path, sha in pinned.items():
        if file_sha(path) != sha:
            raise ValueError('training audit or checkpoint bytes changed: '+path)
    provenance = validate_training_audits(
        json.loads(Path(old_training_audit).read_text()),
        json.loads(Path(new_training_audit).read_text()))
    configs = {}
    for role, pins in (('student62256', OLD_EVAL_CONFIGS),
                       ('student63395', NEW_EVAL_CONFIGS)):
        for task, sha in pins.items():
            path = PROJECT/'checkout/wa/jobs'/f'{role}_{task}_image_4090_v1.yaml'
            if file_sha(path) != sha:
                raise ValueError('submitted evaluation YAML changed: '+str(path))
            configs[str(path)] = sha
    manifest = json.loads(Path(manifest_path).read_text())
    teachers = load_teachers(teacher_path)
    teacher_metrics = summarize_teachers(teachers, manifest)['metrics_percent']
    old_bundle = load_bundle(old_merged, manifest, dict(checkpoint_sha=OLD_SHA, step=STEP), 'image')
    new_bundle = load_bundle(new_merged, manifest, dict(checkpoint_sha=NEW_SHA, step=STEP), 'image')
    for role, bundle, expected_roots in (
            ('old', old_bundle, OLD_PARTITIONS), ('new', new_bundle, NEW_PARTITIONS)):
        expected = {task: str(root) for task, root in expected_roots.items()}
        if bundle['summary'].get('partition_roots') != expected:
            raise ValueError(role+' loaded bundle escaped pinned Job/Task roots')
    partitions = [Path(root).resolve() for bundle in (old_bundle, new_bundle)
                  for root in bundle['summary']['partition_roots'].values()]
    if len(partitions) != 6 or len(set(partitions)) != 6:
        raise ValueError('six distinct full task partitions required')
    report = compare(old_bundle, new_bundle, manifest, teachers, teacher_metrics, provenance)
    code = {str(Path(inspect.getfile(fn)).resolve()) for fn in (
        diagnose, load_bundle, validate_bundle, read_partitions, summarize,
        check_dynamic_state, validate_student_rows, load_teachers, finite_tree, file_sha)}
    hashes = merge_hashes(old_bundle['hashes'], new_bundle['hashes'], bound_files, pinned, configs,
        {str(Path(manifest_path).resolve()): MANIFEST_SHA,
         str(Path(teacher_path).resolve()): TEACHER_SHA},
        {path: file_sha(path) for path in code})
    for path, sha in hashes.items():
        if file_sha(path) != sha:
            raise ValueError('source changed during result audit: '+path)
    report.update(source_hashes=hashes, pinned_old_merged_files=bound_files,
        bound_job_roots=dict(direct={task: str(root) for task, root in OLD_PARTITIONS.items()},
                             retrained={task: str(root) for task, root in NEW_PARTITIONS.items()}),
        partition_roots=dict(
        direct=old_bundle['summary']['partition_roots'],
        retrained=new_bundle['summary']['partition_roots']),
        protected_teacher_artifact_roots=sorted({
            row['branches'][name]['artifact_root']
            for row in teachers for name in ('lightnav', 'oracle')}))
    return report


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('old-merged', 'new-merged', 'manifest', 'teacher-selections',
                 'old-training-audit', 'new-training-audit', 'output'):
        parser.add_argument('--'+name, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    protected = [args.old_merged, args.new_merged, args.manifest,
                 args.teacher_selections, args.old_training_audit,
                 args.new_training_audit, OLD_CHECKPOINT, NEW_CHECKPOINT]
    output_path(args.output, protected)
    report = diagnose(args.old_merged, args.new_merged, args.manifest,
                      args.teacher_selections, args.old_training_audit,
                      args.new_training_audit)
    protected += [root for roots in report['partition_roots'].values() for root in roots.values()]
    artifact = write_report(report, args.output, protected)
    print(json.dumps(dict(audit_status=report['audit_status'], episodes=4215,
        successes={task: values['retrained']['success_count']
                   for task, values in report['tasks'].items()},
        delta_success={task: values['delta_retrained_minus_direct']['success_count']
                       for task, values in report['tasks'].items()}, **artifact)))


if __name__ == '__main__':
    main()
