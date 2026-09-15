"""CPU-reviewed permanent-val adapter; intentionally no GPU launch entry point.

A future separate runner must complete/pin remote asset preflight and supply a
backend explicitly. This module reuses the immutable train bundle's neutral
capture core, never its train admission, constants, supervisor or verifier.
Raw outputs are never rewritten to change scope: an independent val launch and
manifest bind the original label bytes to their fixed permanent-val case.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from generate_val4_protocol import CASES, PARENT_SHA, PERMANENT_SHA, REPO, canonical_sha, file_sha, read, require


def validate_protocol(plan):
    value = dict(plan)
    seal = value.pop('protocol_sha256')
    require(canonical_sha(value) == seal, 'val protocol canonical seal mismatch')
    require(plan['stage'] == 'permanent_val4_perception_protocol_v1'
            and plan['status'] == 'frozen_design_pending_val_collector_and_remote_asset_preflight',
            'unexpected protocol stage or premature readiness claim')
    require(plan['case_denominator'] == 4 and plan['label_observation_denominator'] == 229
            and plan['prediction_denominator'] == 225, 'fixed val denominators changed')
    require(plan['optimizer_input_allowed'] is False and plan['formal_training_eligible'] is False
            and plan['training_boundary']['test_locked_read'] is False
            and plan['training_boundary']['calibration_temperature_or_threshold_fit_on_val'] is False,
            'val evaluation scope changed')
    require(plan['baseline_checkpoint']['sha256'] == PARENT_SHA
            and plan['permanent_manifest']['sha256'] == PERMANENT_SHA, 'baseline/permanent manifest changed')
    require(plan['label_collection']['replay_geometry_tolerance'] == 1e-4
            and plan['label_collection']['world_time_tolerance_s'] == 1e-8
            and plan['label_collection']['rgb_statistics_tolerance_0_255'] == 1e-5,
            'replay tolerances changed')
    require(len(plan['entries']) == 4, 'all four val cases required')
    for entry, expected in zip(plan['entries'], CASES):
        actual = (entry['case_id'], entry['task'], entry['dataset_index'], entry['episode_id'],
                  entry['canonical_scene_id'], entry['action_count'])
        require(actual == expected and entry['observation_count'] == expected[-1]+1
                and entry['prediction_count'] == expected[-1], 'fixed case identity/count changed')
        require(entry['source_arm'] == 'phase2_baseline' and entry['permanent_partition_role'] == 'val'
                and entry['source_dataset_split'] == 'train' and entry['optimizer_input_allowed'] is False,
                'wrong source arm or permanent role')
        require(len(entry['saved_actions']) == expected[-1]
                and canonical_sha(entry['saved_actions']) == entry['saved_actions_sha256'], 'source action seal mismatch')
        require(entry['initialization']['environment_step'] == 0
                and entry['initialization']['used_frames'] == [0], 'initialization must remain reset-only')
    return plan


def runtime_projection(plan, case_id, *, source_path, configuration_snapshot):
    """Return data-only core inputs; source pixels from old overlays are excluded."""
    validate_protocol(plan)
    entries = [entry for entry in plan['entries'] if entry['case_id'] == case_id]
    require(len(entries) == 1, 'unknown fixed val case')
    entry = entries[0]
    source_path, configuration_snapshot = Path(source_path), Path(configuration_snapshot)
    require('test_locked' not in str(source_path).lower()
            and 'test_locked' not in str(configuration_snapshot).lower(), 'test_locked input forbidden')
    require(file_sha(source_path) == entry['source_artifacts']['result.json']['sha256'], 'baseline source bytes changed')
    require(file_sha(configuration_snapshot) == plan['configuration_snapshot']['sha256'], 'config snapshot bytes changed')
    source = read(source_path)
    actions = [[float(row['policy']['action'][key]) for key in ('forward', 'lateral', 'yaw')]
               for row in source['steps']]
    require(canonical_sha(actions) == entry['saved_actions_sha256'], 'source actions differ from frozen protocol')
    inventory = read(configuration_snapshot)['configs'][entry['task']]
    require(canonical_sha(inventory['resolved_config']) == inventory['resolved_config_sha256']
            == plan['task_resolved_config_sha256'][entry['task']], 'resolved configuration mismatch')
    projected = dict(entry, run_id=case_id,
        source={'path':str(source_path.resolve()), 'sha256':entry['source_artifacts']['result.json']['sha256']},
        prefix_samples=[], resolved_config=inventory['resolved_config'],
        resolved_config_sha256=inventory['resolved_config_sha256'],
        entry_config={'path':inventory['entry_config_path'], 'sha256':inventory['entry_config_sha256']},
        dataset={'path':inventory['dataset_path'], 'sha256':inventory['dataset_sha256']})
    runtime_plan = dict(repository=REPO, replay_tolerance=plan['label_collection']['replay_geometry_tolerance'],
        world_time_tolerance_s=plan['label_collection']['world_time_tolerance_s'],
        rgb_statistics_tolerance_0_255=plan['label_collection']['rgb_statistics_tolerance_0_255'], permanent_partition_role='val',
        optimizer_input_allowed=False, formal_training_eligible=False,
        source_worldtime_reference_available=False, source_raw_RGB_reference_available=False)
    return projected, runtime_plan


def pinned_core(plan, template_directory):
    """Hash-check the neutral immutable imports without modifying their files."""
    directory = Path(template_directory).resolve()
    require('test_locked' not in str(directory).lower(), 'test_locked template forbidden')
    references = plan['immutable_train_template_refs']
    for name in ('semantic_labels.py', 'collection_contract.py', 'collect_perception_sidecar.py'):
        require(file_sha(directory/name) == references[name]['sha256'], 'immutable capture core changed: '+name)
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        modules = {}
        for name in ('semantic_labels', 'collection_contract', 'collect_perception_sidecar'):
            path = directory/(name+'.py')
            existing = sys.modules.get(name)
            if existing is not None:
                require(Path(existing.__file__).resolve() == path, 'conflicting imported capture module: '+name)
                modules[name] = existing
            else:
                spec = importlib.util.spec_from_file_location(name, path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                try:
                    spec.loader.exec_module(module)
                except BaseException:
                    sys.modules.pop(name, None)
                    raise
                modules[name] = module
        return modules['collect_perception_sidecar']
    finally:
        sys.dont_write_bytecode = previous_bytecode


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def collect_cpu_mock_case(plan, case_id, *, output, source_path, configuration_snapshot,
                          template_directory, backend_factory):
    """Exercise full neutral replay with an explicit CPU-only test backend.

    This is a test harness, not independent data admission or a launchable val
    collector. A real adapter must separately implement remote preflight and a
    subprocess supervisor before ever providing HabitatReplay as a backend.
    """
    require(getattr(backend_factory, 'cpu_mock_only', False) is True, 'only explicit CPU mock backends are accepted')
    projected, runtime_plan = runtime_projection(plan, case_id, source_path=source_path,
                                                 configuration_snapshot=configuration_snapshot)
    core = pinned_core(plan, template_directory)
    output = Path(output)
    require(not output.exists(), 'refusing to overwrite val output')
    output.mkdir(parents=False, exist_ok=False)
    launch = dict(stage='permanent_val4_perception_cpu_adapter_test_v1',
        protocol_sha256=plan['protocol_sha256'], case_id=case_id, entry=projected,
        permanent_partition_role='val', source_dataset_split='train', cpu_mock_only=True,
        optimizer_input_allowed=False, formal_training_eligible=False,
        independent_admission=False, remote_asset_preflight_completed=False,
        real_val_collection=False, source_worldtime_reference_available=False,
        source_unannotated_pixels_reference_available=False,
        adapter_code_sha256=file_sha(Path(__file__)))
    write_new(output/'val_launch_contract.json', launch)
    result = dict(case_id=case_id, status='failed_collection', expected_observations=projected['observation_count'],
        permanent_partition_role='val', optimizer_input_allowed=False, formal_training_eligible=False,
        cpu_mock_only=True, independent_admission=False, real_val_collection=False)
    try:
        neutral = core.collect_episode(projected, runtime_plan, output, backend_factory=backend_factory)
        result.update(status='cpu_mock_completed_pending_independent_review'
                      if neutral['failed_check_count'] == 0 else 'cpu_mock_failed_alignment',
                      neutral_result=neutral, observations_saved=neutral['observations_saved'])
    except Exception as exc:
        # No retries, replacements or selective removal of failed observations.
        result.update(error_type=type(exc).__name__, error=str(exc))
    artifacts = {}
    for path in sorted(output.rglob('*')):
        if path.is_file():
            artifacts[path.relative_to(output).as_posix()] = {'sha256':file_sha(path), 'bytes':path.stat().st_size}
    write_new(output/'val_artifact_manifest.json', artifacts)
    result['artifact_manifest_sha256'] = file_sha(output/'val_artifact_manifest.json')
    write_new(output/'val_case_result.json', result)
    return result
