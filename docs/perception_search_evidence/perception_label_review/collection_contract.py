"""CPU-only contracts for label-side replay. No policy or Habitat imports."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image

from semantic_labels import array_sha256

EXPECTED = {
    'at_0401_128steps': ('at', 401, '9hjwm8k7gka', '9', 85),
    'dt_0200_128steps': ('dt', 200, 'xxbs57z6pdu', '7', 104),
    'stt_0000_128steps': ('stt', 0, '16tymptm7us', '4', 84),
    'stt_3100_128steps': ('stt', 3100, 'qxwfvs8mq67', '2', 93),
}
STAGE = 'train4_perception_sidecar_collection_v1'


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    def reject(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    value = json.loads(Path(path).read_text(encoding='utf-8'),
                       parse_constant=reject, object_pairs_hook=unique)
    require(isinstance(value, dict), 'JSON object required')
    return value


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def atomic_status(path, value):
    """Only for status files inside this invocation's exclusively reserved output."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def canonical_scene(value):
    name = str(value).replace('\\', '/').rsplit('/', 1)[-1]
    name = name.split('.')[0]
    return re.sub(r'^\d+-', '', name).lower()


def artifact(path, expected=None):
    path = Path(path).resolve(strict=True)
    require(path.is_file(), 'artifact is not a file: ' + str(path))
    actual = digest(path)
    require(expected is None or actual == expected, 'artifact changed: ' + str(path))
    return {'path': str(path), 'sha256': actual, 'bytes': path.stat().st_size}


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def matrix(value):
    require(isinstance(value, list) and len(value) == 4
            and all(isinstance(row, list) and len(row) == 4 for row in value),
            'camera must be a 4x4 matrix')
    require(all(finite_number(v) for row in value for v in row), 'nonfinite camera')
    return np.asarray(value, dtype=np.float64)


def assignments(value):
    require(isinstance(value, dict), 'semantic assignment missing')
    require(all(isinstance(k, str) and re.fullmatch(r'0|[1-9][0-9]*', k) for k in value),
            'noncanonical semantic assignment index')
    result = {int(k): v for k, v in value.items()}
    require(0 in result and 1 not in result, 'target 0 required; robot 1 forbidden')
    require(all(type(v) is int and v > 0 for v in result.values())
            and len(set(result.values())) == len(result), 'ambiguous semantic assignment')
    return result


def validate_source(result, status, launch, run_id, expected_sha, roles):
    """Require the exact four complete original source trajectories, never replacements."""
    require(run_id in EXPECTED, 'unfrozen source')
    task, index, scene, episode, count = EXPECTED[run_id]
    require(result.get('split') == 'train' and result.get('task') == task
            and type(result.get('dataset_index')) is int and result['dataset_index'] == index
            and str(result.get('episode_id')) == episode
            and canonical_scene(result.get('scene_id')) == scene, 'source identity changed')
    require(roles.get(scene) == 'train', 'permanent scene role is not train')
    init = result.get('initialization', {})
    require(type(init.get('environment_step')) is int and init['environment_step'] == 0
            and init.get('used_frames') == [0]
            and init.get('source') == 'first_frame_panoptic_mask', 'step-0 mask initialization required')
    box = init.get('bbox_xyxy')
    require(isinstance(box, list) and len(box) == 4
            and all(type(v) is int and v >= 0 for v in box)
            and box[2] > box[0] and box[3] > box[1], 'invalid source initialization box')
    require(init == launch['run']['original_initialization'], 'source initialization differs from launch')
    require(launch['run']['permanent_partition_role'] == 'train'
            and launch['run']['dataset_index'] == index
            and launch['run']['expected_episode_id'] == episode
            and launch['run']['expected_scene_id'] == result['scene_id'], 'source launch identity changed')
    require(status.get('status') == 'fixed_train_rollout_complete'
            and type(status.get('exit_code')) is int and status['exit_code'] == 0
            and status.get('result_sha256') == expected_sha, 'source process not completed cleanly')
    summary = result.get('summary', {})
    require(summary.get('episode_finished') is True
            and summary.get('termination_reason') == 'episode_over'
            and summary.get('steps') == count and status.get('steps') == count,
            'source must have the exact complete natural termination')
    require(summary.get('requested_max_steps', 0) > count, 'source may be truncated at step limit')
    ids = assignments(result.get('assigned_humanoid_semantic_ids'))
    records = result.get('steps')
    require(isinstance(records, list) and len(records) == count, 'incomplete source action stream')
    actions = []
    for step, row in enumerate(records, 1):
        require(type(row.get('step')) is int and row['step'] == step, 'noncontiguous source steps')
        action = row['policy']['action']
        vector = [action.get(k) for k in ('forward', 'lateral', 'yaw')]
        require(all(finite_number(v) and abs(v) <= 1 for v in vector), 'invalid saved source action')
        actions.append([float(v) for v in vector])
        policy = row['policy']
        require(all(finite_number(policy.get(k)) for k in ('rgb_mean_0_255', 'rgb_std_0_255')),
                'source lacks finite input RGB statistics')
        change = policy.get('rgb_temporal_absdiff_0_255')
        require(change is None if step == 1 else finite_number(change), 'source temporal RGB statistics missing')
        audit = row['evaluation_only_after_action']
        require(finite_number(audit.get('gt_distance_m')) and audit['gt_distance_m'] >= 0,
                'invalid saved GT distance')
        require(type(audit.get('gt_visible')) is bool, 'invalid saved visibility')
        for key in ('camera_transform_before_action', 'camera_transform_after_forced_render'):
            matrix(audit['render_audit'][key])
    return {'task': task, 'dataset_index': index, 'canonical_scene_id': scene,
            'scene_id': result['scene_id'], 'episode_id': episode, 'action_count': count,
            'observation_count': count + 1, 'initialization': init,
            'assigned_humanoid_semantic_ids': {str(k): v for k, v in ids.items()},
            'saved_actions_sha256': canonical_sha(actions)}


def read_rgb(path):
    with Image.open(path) as image:
        require(image.mode == 'RGB', 'frozen prefix must be RGB without implicit conversion')
        return np.asarray(image).copy()


def prefix_reference(row, source_path, source, count):
    require(row['partition_role'] == 'train', 'held-out sample cannot receive labels')
    sample_ref = artifact(row['artifacts']['sample']['path'], row['artifacts']['sample']['sha256'])
    sample = load(sample_ref['path'])
    src = sample['source']
    require(src['split'] == 'train' and src['task'] == source['task']
            and src['dataset_index'] == source['dataset_index']
            and str(src['episode_id']) == str(source['episode_id'])
            and src['scene_id'] == source['scene_id'], 'sample/source identity mismatch')
    require(Path(src['rollout_result']).resolve() == Path(source_path).resolve()
            and src['rollout_result_sha256'] == digest(source_path), 'sample references a different rollout')
    anchor = src['anchor_environment_step']
    require(type(anchor) is int and 0 < anchor < count, 'invalid source anchor')
    inputs = sample['model_inputs']
    states = sample['audit_raw']['prefix_states']
    rows = inputs['prefix_rgb']
    require([v['environment_step'] for v in rows] == list(range(anchor + 1))
            and [v['environment_step'] for v in states] == list(range(anchor + 1)), 'incomplete frozen prefix')
    require(inputs['initial_bbox_xyxy_norm'] == source['initialization']['bbox_xyxy_norm'], 'sample initial target changed')
    calls = inputs['policy_calls']
    require(len(calls) == len(rows), 'policy-call prefix incomplete')
    original_actions = [[float(source['steps'][s]['policy']['action'][k])
                         for k in ('forward', 'lateral', 'yaw')] for s in range(anchor)]
    require(sample['audit_raw']['saved_policy_replay_actions'] == original_actions, 'prefix actions changed')
    directory = Path(sample_ref['path']).parent
    media = {str(Path(v['path']).resolve()): v['sha256'] for v in row['artifacts']['media']}
    frames = []
    previous = -1.0
    for step, value in enumerate(rows):
        path = (directory / value['rgb_path']).resolve(strict=True)
        require(path.parent == directory, 'prefix path escapes sample directory')
        ref = artifact(path, value['sha256'])
        require(media.get(str(path)) == ref['sha256'], 'prefix missing from frozen manifest media')
        t = value['world_time_s']
        require(finite_number(t) and t >= 0 and t > previous
                and states[step]['world_time_s'] == t and calls[step]['world_time_s'] == t
                and calls[step]['policy_call_index'] == step
                and calls[step]['observation_environment_step'] == step, 'prefix clocks misaligned')
        previous = t
        frames.append(dict(ref, environment_step=step, world_time_s=t,
                           rgb_array_sha256=array_sha256(read_rgb(path))))
    initial = inputs['initial_rgb']
    ipath = (directory / initial['rgb_path']).resolve(strict=True)
    require(ipath.parent == directory, 'initial RGB path escapes sample')
    initial_ref = artifact(ipath, initial['sha256'])
    require(media.get(str(ipath)) == initial_ref['sha256'], 'initial RGB not frozen in manifest')
    initial_ref['rgb_array_sha256'] = array_sha256(read_rgb(ipath))
    return {'sample': sample_ref, 'anchor_environment_step': anchor,
            'initial_rgb': initial_ref, 'prefix_rgb': frames}


def compare_observation(source, step, rgb, camera, distance, visible, world_time,
                        prefix_samples, replay_tolerance=1e-4, clock_tolerance=1e-8):
    """Compare only label/audit outputs; the result never influences replay actions."""
    checks = []
    camera = matrix(np.asarray(camera, dtype=np.float64).tolist())
    require(finite_number(distance) and distance >= 0 and finite_number(world_time) and world_time >= 0,
            'invalid replay metric')
    require(type(visible) is bool, 'replay visibility must be boolean')
    if step == 0:
        saved = source['steps'][0]['evaluation_only_after_action']['render_audit']['camera_transform_before_action']
    else:
        saved = source['steps'][step - 1]['evaluation_only_after_action']['render_audit']['camera_transform_after_forced_render']
        reference = source['steps'][step - 1]['evaluation_only_after_action']
        checks.extend([
            {'kind': 'source_distance', 'passed': abs(distance - reference['gt_distance_m']) <= replay_tolerance,
             'absolute_error': abs(distance - reference['gt_distance_m'])},
            {'kind': 'source_visible', 'passed': visible == reference['gt_visible'],
             'expected': reference['gt_visible'], 'actual': visible},
        ])
    error = float(np.abs(camera - matrix(saved)).max())
    checks.append({'kind': 'source_camera', 'passed': error <= replay_tolerance, 'maximum_absolute_error': error})
    actual_rgb = array_sha256(rgb)
    for reference in prefix_samples:
        if step < len(reference['prefix_rgb']):
            frame = reference['prefix_rgb'][step]
            checks.append({'kind': 'stored_prefix_rgb_and_worldtime',
                           'sample_path': reference['sample']['path'], 'prefix_path': frame['path'],
                           'rgb_equal': actual_rgb == frame['rgb_array_sha256'],
                           'world_time_error_s': abs(world_time - frame['world_time_s']),
                           'passed': actual_rgb == frame['rgb_array_sha256']
                               and abs(world_time - frame['world_time_s']) <= clock_tolerance})
        if step == 0:
            checks.append({'kind': 'stored_initial_rgb', 'sample_path': reference['sample']['path'],
                           'passed': actual_rgb == reference['initial_rgb']['rgb_array_sha256']})
    return checks


def compare_policy_rgb(source, step, rgb, previous_rgb, tolerance=1e-5):
    """Observation k conditions action k+1. Terminal k=N has no next policy call."""
    if step == len(source['steps']):
        return [], {'available': False, 'reason': 'terminal_observation_has_no_next_source_policy_call'}
    reference = source['steps'][step]['policy']
    actual = {'rgb_mean_0_255': float(np.asarray(rgb).mean()),
              'rgb_std_0_255': float(np.asarray(rgb).std()),
              'rgb_temporal_absdiff_0_255': None if previous_rgb is None else float(
                  np.abs(np.asarray(rgb, dtype=np.float32) - np.asarray(previous_rgb, dtype=np.float32)).mean())}
    checks = []
    for name, value in actual.items():
        expected = reference.get(name)
        valid = (value is None and expected is None) if name == 'rgb_temporal_absdiff_0_255' and step == 0 else (
            finite_number(value) and finite_number(expected) and abs(value - expected) <= tolerance)
        checks.append({'kind': 'source_policy_input_' + name, 'source_action_step': step + 1,
                       'expected': expected, 'actual': value, 'passed': bool(valid)})
    return checks, {'available': True, 'source_action_step': step + 1, 'actual': actual,
                    'statistics_match_is_not_pixel_identity': True}


def verify_plan(plan, *, verify_files=True):
    value = dict(plan)
    seal = value.pop('plan_sha256')
    require(canonical_sha(value) == seal, 'plan seal mismatch')
    require(plan.get('stage') == STAGE and plan.get('physical_gpu') == 3
            and plan.get('maximum_parallel_collectors') == 1, 'unexpected collection contract')
    require(plan.get('formal_training_eligible') is False and plan.get('optimizer_input_allowed') is False
            and plan.get('test_locked_used') is False, 'sidecar cannot be admitted to training')
    entries = plan['entries']
    require([e['run_id'] for e in entries] == list(EXPECTED), 'all four frozen sources required in order')
    require(sum(e['observation_count'] for e in entries) == 370, 'incorrect observation denominator')
    require(sum(len(e['prefix_samples']) for e in entries) == 19, 'all 19 long-source samples required')
    prefix_paths = [s['sample']['path'] for e in entries for s in e['prefix_samples']]
    require(len(set(prefix_paths)) == 19, 'duplicate prefix sample references')
    require(len(plan['excluded_legacy_train_samples']) == 4 and plan['held_out_sample_count'] == 4,
            'legacy/held-out accounting changed')
    root = Path(plan['output_root']).resolve(strict=False)
    repo = Path(plan['repository']).resolve(strict=True)
    require(root == repo / 'outputs/takeover/perception_label_sidecar_train4_v1', 'unexpected fixed output directory')
    require(len({e['source']['path'] for e in entries}) == 4, 'duplicate source trajectory')
    for entry in entries:
        expected = EXPECTED[entry['run_id']]
        require((entry['task'], entry['dataset_index'], entry['canonical_scene_id'], entry['episode_id'], entry['action_count']) == expected,
                'entry selection mutated')
        require(entry['permanent_partition_role'] == 'train'
                and plan['permanent_scene_roles'][entry['canonical_scene_id']] == 'train', 'val role leakage')
        require(Path(entry['output_dir']).resolve() == root / entry['run_id'], 'output identity mismatch')
        require(entry['observation_count'] == entry['action_count'] + 1, 'terminal/reset frame omission')
    if verify_files:
        require(len({a['path'] for a in plan['artifacts']}) == len(plan['artifacts']), 'duplicate artifact entries')
        registry = {a['path']: a for a in plan['artifacts']}
        def pinned(ref):
            require(ref['path'] in registry and registry[ref['path']]['sha256'] == ref['sha256'],
                    'consumed input is absent from artifact pins: ' + str(ref['path']))
        fixed = {
            'current_manifest': 'f822609f23da1fd48662c509ad1dfd6fed9ab582f7c8794eabd17faf173a6972',
            'collection_protocol': 'f207c85249c774a17cc67970ab38c57b95d4a94db1c46af9497b21d0ec3c79c3',
            'permanent_manifest': '9268ac7bf53a8392ab58a05112e8bd4db662889fa560708973e8afb807ca1e4e',
            'source_config_inventory': '3a5f559e4c604f749a673af490845c3980463954e8110a4a83c88f1857ad2a8e',
            'admission_policy': '4da01fd3ef238e46db9620948e8f1ca767576c37db648e0f83f5a3d12cce60b7',
        }
        for key, sha in fixed.items():
            pinned(plan[key])
            require(plan[key]['sha256'] == sha, 'fixed protocol pin changed: ' + key)
        pinned(plan['python'])
        bundle = Path(plan['bundle']).resolve(strict=True)
        require(bundle == Path(__file__).resolve().parent, 'preflight code is outside frozen bundle')
        for filename in ('semantic_labels.py', 'collection_contract.py', 'freeze_perception_plan.py',
                         'collect_perception_sidecar.py'):
            path = str((bundle/filename).resolve(strict=True))
            require(path in registry, 'runtime code absent from plan: ' + filename)
        for ref in plan['artifacts']:
            artifact(ref['path'], ref['sha256'])
        manifest = load(plan['current_manifest']['path'])
        collection = load(plan['collection_protocol']['path'])
        permanent = load(plan['permanent_manifest']['path'])
        roles = {s['identity']['scene_id']: s['partition_role'] for s in permanent['samples']}
        require(roles == plan['permanent_scene_roles'] == collection['permanent_scene_roles'], 'permanent scene role changed')
        configurations = load(plan['source_config_inventory']['path'])['configs']
        for entry in entries:
            for key in ('source', 'source_status', 'source_launch', 'entry_config', 'dataset'):
                pinned(entry[key])
            require(collection['source_results'][entry['run_id']] == entry['source']['path'], 'source replaced')
            source = load(entry['source']['path'])
            identity = validate_source(source, load(entry['source_status']['path']),
                load(entry['source_launch']['path']), entry['run_id'], entry['source']['sha256'], plan['permanent_scene_roles'])
            require(all(entry[k] == v for k, v in identity.items()), 'source provenance differs from plan')
            cfg = configurations[entry['task']]
            require(entry['resolved_config'] == cfg['resolved_config']
                    and entry['resolved_config_sha256'] == cfg['resolved_config_sha256']
                    and canonical_sha(entry['resolved_config']) == entry['resolved_config_sha256'], 'resolved config changed')
            require(entry['dataset']['sha256'] == cfg['dataset_sha256']
                    and entry['dataset']['path'] == cfg['dataset_path']
                    and entry['entry_config']['sha256'] == cfg['entry_config_sha256']
                    and entry['entry_config']['path'] == cfg['entry_config_path'], 'config/dataset ref changed')
            matched = [s for s in manifest['samples'] if s['artifacts']['rollout']['path'] == entry['source']['path']]
            reconstructed = [prefix_reference(row, entry['source']['path'], source, entry['action_count']) for row in matched]
            require(reconstructed == entry['prefix_samples'], 'prefix reference pixels, time or membership changed')
            for ref in reconstructed:
                pinned(ref['sample'])
                for media in [ref['initial_rgb'], *ref['prefix_rgb']]:
                    pinned(media)
    return {'status': 'read_only_preflight_passed', 'source_count': 4, 'observation_count': 370,
            'prefix_sample_count': 19, 'environment_created': False, 'training_admitted': False,
            'plan_sha256': seal}
