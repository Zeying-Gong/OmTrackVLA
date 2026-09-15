"""Independent CPU admission audit of raw perception sidecars; never trains.

The externally supplied plan file hash and frozen admission policy are trust
anchors. Pinned pure helpers may calculate labels/checks; every saved array and
original prefix is independently read again. No collector/GPU module is loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

POLICY_SHA256 = '4da01fd3ef238e46db9620948e8f1ca767576c37db648e0f83f5a3d12cce60b7'
GRANT = 'verified_development_perception_labels_pending_loader_integration'
REJECT = 'rejected_no_partial_admission'
RUNS = ('at_0401_128steps', 'dt_0200_128steps', 'stt_0000_128steps', 'stt_3100_128steps')
COUNTS = (86, 105, 85, 94)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def parse(text):
    def unique(items):
        value = {}
        for key, item in items:
            require(key not in value, 'duplicate JSON key: ' + key)
            value[key] = item
        return value
    def reject(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    value = json.loads(text, object_pairs_hook=unique, parse_constant=reject)
    require(isinstance(value, dict), 'JSON object required')
    return value


def load(path):
    return parse(Path(path).read_text(encoding='utf-8'))


def pinned(path, expected):
    require(Path(path).is_file() and sha(path) == expected, 'file hash mismatch: ' + str(path))


def strict_equal(left, right):
    # JSON equality must distinguish booleans from numeric 0/1.
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def load_helpers(plan):
    """Import only frozen pure modules after checking their artifact identities."""
    bundle = Path(plan['bundle']).resolve(strict=True)
    registry = {item['path']: item for item in plan['artifacts']}
    modules = {}
    for filename in ('semantic_labels.py', 'collection_contract.py'):
        path = bundle / filename
        require(str(path) in registry, 'helper missing from frozen plan')
        pinned(path, registry[str(path)]['sha256'])
        name = path.stem
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return SimpleNamespace(labels=modules['semantic_labels'], contract=modules['collection_contract'])


def validate_denominators(plan):
    entries = plan['entries']
    require([entry['run_id'] for entry in entries] == list(RUNS), 'four exact sources in frozen order required')
    require(tuple(entry['observation_count'] for entry in entries) == COUNTS, '370 observations with exact source counts required')
    paths = [str(Path(item['sample']['path']).resolve()) for entry in entries for item in entry['prefix_samples']]
    require(len(paths) == len(set(paths)) == 19, '19 distinct original prefix samples required')
    require(all(entry['permanent_partition_role'] == 'train' for entry in entries), 'validation source forbidden')
    return paths


def saved_file(directory, reference, expected_name):
    require(reference['path'] == expected_name, 'unexpected frame filename or observation index')
    path = directory / expected_name
    require(path.resolve().parent == (directory / 'observations').resolve(), 'frame path escapes observations')
    require(not path.is_symlink(), 'symlink frame forbidden')
    pinned(path, reference['sha256'])
    return path


def verify_frame(row, step, entry, plan, source, directory, previous_rgb, previous_time, helpers):
    h, semantic = helpers.contract, helpers.labels
    rgb_path = saved_file(directory, row['rgb_file'], f'observations/rgb_{step:04d}.png')
    raw_path = saved_file(directory, row['raw_panoptic_file'], f'observations/panoptic_{step:04d}.npy')
    rgb = h.read_rgb(rgb_path)
    raw = np.load(raw_path, allow_pickle=False)
    rawref = row['raw_panoptic_file']
    require(rawref['raw_array_sha256'] == semantic.array_sha256(raw)
            and rawref['shape'] == list(raw.shape) and rawref['dtype'] == str(raw.dtype), 'raw panoptic array identity mismatch')
    audit = row['source_audit']
    capture = audit['capture_evidence']
    assigned = h.assignments(capture['assigned_humanoid_semantic_ids'])
    require(assigned == h.assignments(entry['assigned_humanoid_semantic_ids']), 'semantic assignment changed')
    before, after = capture['world_time_before_render_s'], capture['world_time_after_render_s']
    require(h.finite_number(before) and before >= 0 and h.finite_number(after) and after >= 0,
            'measured simulator time required')
    require(before == after and (previous_time is None or after > previous_time), 'render or monotonic time mismatch')
    terminal = capture['terminal_observation']
    require(type(terminal) is bool and terminal == (step == entry['action_count']), 'natural terminal at wrong observation')
    expected = semantic.label_observation(rgb=rgb, panoptic=raw, assigned_humanoid_semantic_ids=assigned,
        environment_step=step, world_time_s=after, terminal_observation=terminal)
    require(all(key in row and strict_equal(row[key], value) for key, value in expected.items()),
            'saved semantic label differs from raw-derived label or causal index')
    require(set(row) == set(expected) | {'rgb_file', 'raw_panoptic_file', 'source_audit',
                                        'formal_training_eligible', 'optimizer_input_allowed'}, 'unexpected label fields')
    require(row['formal_training_eligible'] is False and row['optimizer_input_allowed'] is False,
            'sidecar claims training rights')
    camera_before = h.matrix(capture['camera_before_render'])
    camera_after = h.matrix(capture['camera_after_render'])
    require(strict_equal(audit['camera_after_render'], capture['camera_after_render']), 'camera evidence differs')
    delta = float(np.abs(camera_before - camera_after).max())
    require(delta <= plan['replay_tolerance'], 'camera moved during shared RGB/panoptic render')
    actions = [[float(r['policy']['action'][k]) for k in ('forward', 'lateral', 'yaw')] for r in source['steps']]
    require(audit['source_result_sha256'] == entry['source']['sha256'], 'source hash binding changed')
    require(strict_equal(audit['action_just_replayed'], None if step == 0 else actions[step-1]), 'replayed action differs from source')
    require(audit['rgb_equal_previous'] is (None if previous_rgb is None else bool(np.array_equal(rgb, previous_rgb))),
            'previous RGB equality claim differs')
    checks = h.compare_observation(source, step, rgb, camera_after, audit['gt_distance_m'],
        expected['target']['visible'], after, entry['prefix_samples'], plan['replay_tolerance'], plan['world_time_tolerance_s'])
    rgb_checks, statistics = h.compare_policy_rgb(source, step, rgb, previous_rgb, plan['rgb_statistics_tolerance_0_255'])
    checks.extend(rgb_checks)
    require(strict_equal(statistics, audit['source_policy_input_rgb_statistics']), 'policy RGB statistics differ from saved pixels')
    checks.extend([
        {'kind': 'single_render_worldtime_stable', 'passed': before == after, 'before_s': before, 'after_s': after},
        {'kind': 'single_render_camera_stable', 'passed': delta <= plan['replay_tolerance'], 'maximum_absolute_error': delta},
        {'kind': 'worldtime_strictly_increasing', 'passed': previous_time is None or after > previous_time},
        {'kind': 'exact_natural_terminal_step', 'passed': terminal == (step == entry['action_count']),
         'expected': step == entry['action_count'], 'actual': terminal},
    ])
    if step == 0:
        require(capture['initialization'] == entry['initialization'], 'reset initialization changed')
        require(expected['target']['bbox_xyxy_inclusive'] == entry['initialization']['bbox_xyxy']
                and expected['target']['bbox_xyxy_norm'] == entry['initialization']['bbox_xyxy_norm'],
                'raw reset target bbox differs from original initialization')
        checks.append({'kind': 'same_original_initialization', 'passed': capture['initialization'] == entry['initialization'],
                       'actual': capture['initialization'], 'expected': entry['initialization']})
    else:
        require(capture['initialization'] is None, 'later initialization evidence forbidden')
    require(strict_equal(checks, audit['checks']), 'worker checks differ from independent recomputation')
    require(all(check['passed'] is True for check in checks), 'independently recomputed alignment failed')
    require(audit['passed'] is True, 'recorded alignment failure must be retained')
    # Byte hashes above bind files; equality below independently verifies every
    # original sample's pixels, without substituting a hash-only worker claim.
    prefix_count = 0
    for sample in entry['prefix_samples']:
        if step < len(sample['prefix_rgb']):
            ref = sample['prefix_rgb'][step]
            pinned(ref['path'], ref['sha256'])
            original_rgb = h.read_rgb(ref['path'])
            require(np.array_equal(rgb, original_rgb), 'original prefix pixels differ')
            require(ref['rgb_array_sha256'] == semantic.array_sha256(original_rgb), 'original prefix array hash differs')
            require(abs(after - ref['world_time_s']) <= plan['world_time_tolerance_s'], 'original prefix measured time differs')
            prefix_count += 1
        if step == 0:
            ref = sample['initial_rgb']
            pinned(ref['path'], ref['sha256'])
            original_rgb = h.read_rgb(ref['path'])
            require(np.array_equal(rgb, original_rgb)
                    and ref['rgb_array_sha256'] == semantic.array_sha256(original_rgb), 'original reset pixels differ')
    return rgb, after, len(checks), prefix_count


def verify_source(entry, plan, status, helpers):
    directory = Path(entry['output_dir'])
    require(status['run_id'] == entry['run_id'] and status['output_dir'] == entry['output_dir'], 'batch source identity changed')
    require(type(status['exit_code']) is int and status['exit_code'] == 0
            and status['status'] == 'sidecar_collected_pending_independent_admission', 'failed worker cannot be admitted')
    launch = load(directory / 'launch_contract.json')
    require(launch['entry'] == entry and launch['plan_sha256'] == plan['plan_sha256'], 'launch/plan binding changed')
    require(launch['formal_training_eligible'] is False and launch['optimizer_input_allowed'] is False, 'launch grants training rights')
    worker = load(directory / 'worker_result.json')
    pinned(directory / 'worker_result.json', status['worker_result_sha256'])
    require(worker == status['worker_result'] == load(directory / 'worker_progress.json'), 'worker final status copies differ')
    require(worker['status'] == 'sidecar_collected_pending_independent_admission'
            and worker['run_id'] == entry['run_id'] and worker['observations_saved'] == entry['observation_count']
            and worker['expected_observations'] == entry['observation_count']
            and worker['completed_source_actions'] == entry['action_count']
            and worker['failed_check_count'] == 0 and worker['natural_terminal_reproduced'] is True, 'incomplete or failed worker')
    for key in ('formal_training_eligible', 'optimizer_input_allowed', 'test_locked_used', 'model_loaded'):
        require(worker[key] is False, 'worker violates label-only scope: ' + key)
    require(worker['gt_used_only_on_label_or_audit_side'] is True, 'worker label boundary missing')
    labels_path = directory / 'labels.jsonl'
    pinned(labels_path, worker['labels_sha256'])
    rows = [parse(line) for line in labels_path.read_text(encoding='utf-8').splitlines()]
    require(len(rows) == entry['observation_count'], 'reset or terminal observation missing')
    expected_files = {f'{kind}_{step:04d}.{suffix}' for step in range(len(rows)) for kind, suffix in (('rgb', 'png'), ('panoptic', 'npy'))}
    require({path.name for path in (directory / 'observations').iterdir()} == expected_files, 'missing or extra observation files')
    pinned(entry['source']['path'], entry['source']['sha256'])
    source = load(entry['source']['path'])
    previous_rgb, previous_time, total_checks, prefix_count = None, None, 0, 0
    for step, row in enumerate(rows):
        try:
            previous_rgb, previous_time, checks, matched = verify_frame(row, step, entry, plan, source, directory,
                previous_rgb, previous_time, helpers)
        except Exception as error:
            raise ValueError(f'observation {step}: {type(error).__name__}: {error}') from error
        total_checks += checks
        prefix_count += matched
    require(total_checks == worker['total_checks'] and prefix_count == worker['checked_prefix_observations'], 'worker totals differ from recomputation')
    require(prefix_count == sum(len(s['prefix_rgb']) for s in entry['prefix_samples']), 'not every original prefix matched')
    unique_prefix_steps = {frame['environment_step'] for sample in entry['prefix_samples'] for frame in sample['prefix_rgb']}
    return {'run_id': entry['run_id'], 'status': 'independently_verified', 'observations_verified': len(rows),
            'saved_frame_files_verified': 2 * len(rows), 'checks_recomputed': total_checks,
            'original_prefix_observations_verified': prefix_count,
            'original_prefix_unique_observations_verified': len(unique_prefix_steps),
            'nonterminal_observations_without_original_prefix_pixels': entry['action_count'] - len(unique_prefix_steps),
            'terminal_observations_without_source_policy_rgb_statistics': 1,
            'distinct_original_samples': len(entry['prefix_samples']), 'labels_sha256': worker['labels_sha256']}


def audit_outputs(plan, batch, helpers):
    paths = validate_denominators(plan)
    require(batch['plan_sha256'] == plan['plan_sha256'] and batch['stage'] == plan['stage'], 'batch/plan binding changed')
    require([row['run_id'] for row in batch['entries']] == list(RUNS), 'batch must retain all four sources')
    reports = []
    for entry, status in zip(plan['entries'], batch['entries']):
        try:
            reports.append(verify_source(entry, plan, status, helpers))
        except Exception as error:
            reports.append({'run_id': entry['run_id'], 'status': 'failed_retained', 'error': f'{type(error).__name__}: {error}'})
    passed = all(row['status'] == 'independently_verified' for row in reports)
    batch_ok = (batch['status'] == 'sidecars_pending_independent_admission' and batch['expected_source_count'] == 4
        and batch['expected_observation_count'] == 370 and batch['successful_source_count'] == 4
        and batch['observed_frames'] == 370 and batch['all_four_sources_retained'] is True
        and all(batch[key] is False for key in ('formal_training_eligible', 'optimizer_input_allowed', 'test_locked_used')))
    success = passed and batch_ok
    coverage = {key: sum(row.get(key, 0) for row in reports) for key in (
        'observations_verified', 'saved_frame_files_verified', 'original_prefix_observations_verified',
        'original_prefix_unique_observations_verified', 'nonterminal_observations_without_original_prefix_pixels',
        'terminal_observations_without_source_policy_rgb_statistics')}
    return {'status': GRANT if success else REJECT, 'partial_admission_allowed': False,
            'admitted_observation_count': 370 if success else 0, 'independently_verified_coverage': coverage,
            'optimizer_input_allowed': False, 'formal_training_eligible': False, 'product_acceptance_evidence': False,
            'source_count': 4, 'expected_observation_count': 370, 'required_distinct_original_samples': 19,
            'distinct_original_sample_paths': paths, 'batch_summary_valid': batch_ok, 'sources': reports}


def verify(plan_path, plan_sha, batch_path, policy_path):
    pinned(plan_path, plan_sha)
    pinned(policy_path, POLICY_SHA256)
    policy, plan = load(policy_path), load(plan_path)
    require(policy['grantable_scope'] == GRANT and policy['partial_admission_allowed'] is False, 'unexpected admission policy')
    helpers = load_helpers(plan)
    helpers.contract.verify_plan(plan)  # Pure CPU pin/membership checks; no collector import.
    validate_denominators(plan)
    batch_path = Path(batch_path).resolve(strict=True)
    require(batch_path == Path(plan['output_root']).resolve() / 'batch_status.json', 'batch path is outside frozen plan')
    require(load(Path(plan['output_root']) / 'frozen_plan.json') == plan, 'collected frozen plan differs')
    batch_sha = sha(batch_path)
    report = audit_outputs(plan, load(batch_path), helpers)
    # Detect source/sample/plan mutations during this verification as well.
    references = [plan[k] for k in ('current_manifest', 'collection_protocol', 'permanent_manifest')]
    for entry in plan['entries']:
        references.extend(entry[k] for k in ('source', 'source_status', 'source_launch'))
        for sample in entry['prefix_samples']:
            references.extend([sample['sample'], sample['initial_rgb'], *sample['prefix_rgb']])
    for ref in references:
        pinned(ref['path'], ref['sha256'])
    pinned(plan_path, plan_sha)
    pinned(policy_path, POLICY_SHA256)
    pinned(batch_path, batch_sha)
    report.update(plan_file_sha256=plan_sha, plan_sha256=plan['plan_sha256'], admission_policy_sha256=POLICY_SHA256,
        batch_status_sha256=batch_sha, original_inputs_unchanged=True, source_files_modified=False,
        replay_started=False, model_loaded=False, gt_training_inputs_generated=False,
        method='Re-read saved RGB/raw npy; verify file/array hashes; rederive all semantic fields; independently recompute checks and compare every original prefix pixel/time.',
        limitations=['Stored simulator measurements are checked for consistency with frozen source and prefix evidence; raw pixels alone cannot independently prove physical time or geometry.',
                     'GT anypixel is not an occlusion category, visible fraction, or calibrated same-identity prediction.',
                     'No stop/motion-permission/binding/ego/identity-prediction supervision or optimizer admission is granted.'])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-file-sha256', required=True)
    parser.add_argument('--batch-status', type=Path, required=True)
    parser.add_argument('--admission-policy', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    # Exclusively reserve a new audit directory, never an existing source path.
    output.mkdir(parents=False, exist_ok=False)
    try:
        report = verify(args.plan, args.plan_file_sha256, args.batch_status, args.admission_policy)
    except Exception as error:
        report = {'status': REJECT, 'partial_admission_allowed': False, 'optimizer_input_allowed': False,
                  'formal_training_eligible': False, 'product_acceptance_evidence': False,
                  'error': f'{type(error).__name__}: {error}', 'original_failure_evidence_retained': True}
    report['verifier_sha256'] = sha(__file__)
    with (output / 'verification.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'report': str(output / 'verification.json')}))
    return 0 if report['status'] == GRANT else 2


if __name__ == '__main__':
    raise SystemExit(main())
