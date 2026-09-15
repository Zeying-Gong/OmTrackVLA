"""Generate a concrete permanent-val4 perception protocol from existing evidence.

CPU/read-only by default. --output writes a new protocol and never overwrites.
This design generator has no simulator/GPU launch or training entry point.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from perception_metrics import METRIC_SPEC

ROOT = Path(__file__).resolve().parents[1]
REPO = '/data/nfs/share/wam_tracking/OmTrackVLA'
OLD_PROTOCOL_FILE_SHA = '61a1cfdf949964541f96c7489183c01d88e88f199efeda0e195af3c3b176b8d6'
OLD_PROTOCOL_SEAL = '9bf19ab23a9aa11925782eb336a77e095bf3b0497c487491cc1503f1323a0b28'
PARENT_SHA = '32c8f2f73277cfab8c3c7c22a53994061f7178498cfa003e1635b9576641bc9c'
PERMANENT_SHA = '9268ac7bf53a8392ab58a05112e8bd4db662889fa560708973e8afb807ca1e4e'
BASELINE_SUITE_SHA = '3b5293b7b0fa3274744f93753b308c2b0795a01a721e5a895a848a21f4cf6009'
BASELINE_INVENTORY_SHA = '1dd749aa5a28102ff0649d39178ae507271505c64987845d720a48f0a664936b'
TRAIN_PLAN_SHA = 'dd9fcd6b00249a802410f1638216f56a64af51ef998fd1fc9d6566317e080780'
CONFIG_SNAPSHOT_SHA = '3a5f559e4c604f749a673af490845c3980463954e8110a4a83c88f1857ad2a8e'
CASES = [('dt_1300_episode3','dt',1300,'3','gjhyih4upq9',90),
         ('at_1700_episode0','at',1700,'0','ayhkzj2fehp',46),
         ('dt_1700_episode0','dt',1700,'0','ayhkzj2fehp',45),
         ('stt_1700_episode0','stt',1700,'0','ayhkzj2fehp',44)]


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def reject(value):
        raise ValueError('nonfinite source JSON: '+value)
    def unique(pairs):
        value = {}
        for key,item in pairs:
            require(key not in value, 'duplicate source JSON key: '+key)
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=reject,object_pairs_hook=unique)


def ref(path, remote_path=None, expected=None):
    value = file_sha(path)
    require(expected is None or value == expected, 'source artifact SHA mismatch: '+str(path))
    result = {'sha256':value,'bytes':Path(path).stat().st_size}
    if remote_path is not None: result['path'] = remote_path
    result['local_review_path'] = str(Path(path).resolve())
    return result


def finite(value):
    return type(value) in (float,int) and math.isfinite(value)


def validate_baseline_source(source, status, launch, case, expected, old_protocol):
    case_id,task,index,episode,scene,count = expected
    require(case['case_id'] == case_id and case['task'] == task and case['dataset_index'] == index
            and case['episode_id'] == episode and case['canonical_scene_id'] == scene, 'fixed case identity changed')
    require(case['partition_role'] == 'val' and case['optimizer_input_allowed'] is False
            and case['source_dataset_split'] == 'train', 'permanent val role changed')
    require(source['task'] == task and source['dataset_index'] == index
            and source['episode_id'] == episode and source['scene_id'] == case['scene_id']
            and source['split'] == 'train', 'baseline source identity changed')
    require(source['initialization'] == case['initialization'] and source['initialization']['environment_step'] == 0
            and source['initialization']['used_frames'] == [0], 'source initialization changed')
    require(source['loading']['checkpoint_sha256'] == PARENT_SHA
            and source['loading']['checkpoint_step'] == 4096
            and source['loading']['test_locked_used'] is False, 'not the frozen Phase2 baseline')
    require(launch['case'] == case and launch['checkpoint']['sha256'] == PARENT_SHA
            and launch['optimizer_input_allowed'] is False and launch['partition_role'] == 'val', 'launch provenance changed')
    require(status['status'] == 'complete' and status['exit_code'] == 0 and status['actual_steps'] == count
            and status['case_id'] == case_id and status['partition_role'] == 'val'
            and status['optimizer_input_allowed'] is False, 'source case is not complete')
    summary = source['summary']
    require(summary['episode_finished'] is True and summary['termination_reason'] == 'episode_over'
            and summary['steps'] == count and summary['requested_max_steps'] == 300, 'source lacks exact natural completion')
    require(status['summary'] == summary and len(source['steps']) == count, 'incomplete or inconsistent result')
    require(source['model_config'] == old_protocol['model_config']['path'], 'source model config changed')
    assignments = source['assigned_humanoid_semantic_ids']
    require('0' in assignments and '1' not in assignments
            and all(type(v) is int and v > 0 for v in assignments.values())
            and len(set(assignments.values())) == len(assignments), 'ambiguous source target assignment')
    actions = []
    for index,row in enumerate(source['steps'],1):
        require(type(row['step']) is int and row['step'] == index, 'noncontiguous original actions')
        action = [row['policy']['action'][key] for key in ('forward','lateral','yaw')]
        require(all(finite(value) and abs(value) <= 1 for value in action), 'invalid original action')
        actions.append(action)
        observation = row['evaluation_only_after_action']
        require(type(observation['gt_visible']) is bool and finite(observation['gt_distance_m']), 'source observation missing')
        for key in ('camera_transform_before_action','camera_transform_after_forced_render'):
            matrix = observation['render_audit'][key]
            require(len(matrix) == 4 and all(len(r) == 4 and all(finite(v) for v in r) for r in matrix), 'invalid source camera')
    return actions


def generate(audit_root=ROOT):
    root = Path(audit_root)
    old_path = root/'closed_loop_val4_review/phase2_val4_baseline_protocol_v3.json'
    old = read(old_path)
    require(file_sha(old_path) == OLD_PROTOCOL_FILE_SHA
            and old['protocol_sha256'] == OLD_PROTOCOL_SEAL
            and canonical_sha({k:v for k,v in old.items() if k!='protocol_sha256'}) == OLD_PROTOCOL_SEAL,
            'original val4 protocol changed')
    require(old['permanent_manifest']['sha256'] == PERMANENT_SHA, 'permanent manifest identity changed')
    suite_relative = 'outputs/takeover/permanent_val4_closed_loop_v1/phase2_baseline'
    directory = root/'closed_loop_val4_review/evidence'/suite_relative
    require(file_sha(directory/'SUITE_RESULT.json') == BASELINE_SUITE_SHA
            and file_sha(directory/'SUITE_ARTIFACTS.json') == BASELINE_INVENTORY_SHA,
            'independently audited baseline suite/inventory changed')
    suite, suite_inventory = read(directory/'SUITE_RESULT.json'), read(directory/'SUITE_ARTIFACTS.json')
    require(suite['arm'] == 'phase2_baseline' and suite['partition_role'] == 'val'
            and suite['optimizer_input_allowed'] is False and suite['case_denominator'] == 4
            and suite['protocol_sha256'] == OLD_PROTOCOL_SEAL, 'not the audited permanent baseline suite')
    require([r['case_id'] for r in suite['cases']] == [e[0] for e in CASES], 'suite cases replaced or reordered')
    entries = []
    for case, expected, suite_status in zip(old['cases'], CASES, suite['cases']):
        case_id = expected[0]
        case_root = directory/case_id
        source,status,launch = (read(case_root/name) for name in ('result.json','execution_status.json','launch_contract.json'))
        require(status == suite_status, 'case status differs from suite')
        actions = validate_baseline_source(source,status,launch,case,expected,old)
        artifacts = {}
        for filename in ('result.json','execution_status.json','launch_contract.json','artifact_manifest.json','timing.json'):
            key = case_id+'/'+filename
            artifacts[filename] = ref(case_root/filename,REPO+'/'+suite_relative+'/'+key, suite_inventory[key]['sha256'])
        require(canonical_sha(read(case_root/'artifact_manifest.json')['result.json'])
                == canonical_sha(suite_inventory[case_id+'/result.json']), 'source result inventory mismatch')
        entries.append({'case_id':case_id,'task':case['task'],'dataset_index':case['dataset_index'],
            'episode_id':case['episode_id'],'scene_id':case['scene_id'],'canonical_scene_id':case['canonical_scene_id'],
            'permanent_partition_role':'val','source_dataset_split':'train','optimizer_input_allowed':False,
            'source_artifacts':artifacts,'source_arm':'phase2_baseline','action_count':expected[-1],
            'observation_count':expected[-1]+1,'prediction_count':expected[-1],
            'initialization':source['initialization'],'assigned_humanoid_semantic_ids':source['assigned_humanoid_semantic_ids'],
            'saved_actions':actions,'saved_actions_sha256':canonical_sha(actions),'sim_artifacts':case['sim_artifacts'],
            'old_short_prefix_sample':{'reference':case['artifacts']['sample'],
                'use_for_new_label_pixel_matching':False,'reason':'different_original_rollout_not_assumed_identical'},
            'annotated_video_frames_are_policy_inputs':False})
    bundle = root/'perception_label_review'
    require(file_sha(bundle/'frozen_plan.json') == TRAIN_PLAN_SHA, 'immutable train template plan changed')
    training_plan = read(bundle/'frozen_plan.json')
    immutable_template_names = ('semantic_labels.py','collection_contract.py','collect_perception_sidecar.py','freeze_perception_plan.py')
    template_refs = {}
    for name in immutable_template_names:
        path = bundle/name
        original = next(item for item in training_plan['artifacts'] if item['path'].endswith('/perception_label_collection_v1/'+name))
        template_refs[name] = ref(path,original['path'],original['sha256'])
    require(file_sha(bundle/'source_config_inventory.json') == CONFIG_SNAPSHOT_SHA, 'source configuration snapshot changed')
    configs = read(bundle/'source_config_inventory.json')
    plan = {
        'stage':'permanent_val4_perception_protocol_v1','schema_version':1,
        'status':'frozen_design_pending_val_collector_and_remote_asset_preflight',
        'declaration_timing':'original_val4_closed_loop_results_already_known; declared_before_new_val_raw_labels_or_perception_candidate_outputs',
        'case_denominator':4,'label_observation_denominator':229,'prediction_denominator':225,
        'entries':entries,'baseline_checkpoint':old['baseline_checkpoint'],'model_config':old['model_config'],
        'permanent_manifest':old['permanent_manifest'],'source_protocol':ref(old_path,REPO+'/.codex_upload/closed_loop_val4_review_v3/phase2_val4_baseline_protocol_v3.json',OLD_PROTOCOL_FILE_SHA),
        'source_suite':ref(directory/'SUITE_RESULT.json',REPO+'/'+suite_relative+'/SUITE_RESULT.json'),
        'source_suite_inventory':ref(directory/'SUITE_ARTIFACTS.json',REPO+'/'+suite_relative+'/SUITE_ARTIFACTS.json'),
        'immutable_train_template_refs':template_refs,
        'task_resolved_config_sha256':{key:value['resolved_config_sha256'] for key,value in configs['configs'].items()},
        'configuration_snapshot':ref(bundle/'source_config_inventory.json',REPO+'/.codex_upload/perception_label_collection_v1/source_config_inventory.json'),
        'environment':dict(old['environment']),'maximum_parallel_collectors':1,'physical_gpu':3,
        'label_output_root':REPO+'/outputs/takeover/perception_label_sidecar_permanent_val4_v1',
        'label_independent_admission_scope':'verified_permanent_val_perception_labels_evaluation_only',
        'label_collection':{'same_render_rgb_panoptic':True,'raw_arrays_preserved':True,
            'replay_geometry_tolerance':1e-4,'world_time_tolerance_s':1e-8,
            'rgb_statistics_tolerance_0_255':1e-5,
            'observations':'reset_0_through_exact_natural_terminal_N', 'actions':'all_original_Phase2_actions_unchanged',
            'source_checks':['identity','initialization','assigned_IDs','camera','GT_distance','GT_anypixel_visibility',
                             'input_RGB_mean_std_temporal_for_k_less_than_N'],
            'source_worldtime_available':False,'original_unannotated_pixel_reference_available':False,
            'new_capture_worldtime_strictly_monotonic':True,'no_substitution_retry_or_overwrite':True,
            'all_four_and_229_observations_required':True,'partial_admission_allowed':False},
        'prediction_contract':{'common_new_raw_RGB_dataset_for_baseline_and_candidates':True,
            'replay_every_causal_policy_call':True,'case_reset_hidden_and_target_memory':True,
            'initial_RGB_bbox_once_only':True,'missing_UWB_condition_kept':True,
            'later_GT_bbox_panoptic_semantic_IDs_not_model_inputs':True,
            'prediction_fields':['observation_environment_step','source_rgb_array_sha256','source_world_time_s',
                                 'visibility_probability','predicted_bbox_xyxy_norm'],
            'candidate_actions_not_executed':True,'gate_unchanged':True,
            'terminal_has_no_prediction':True,'candidate_checkpoint_hash_fixed_before_evaluation':True,
            'old_logged_Phase2_predictions_are_diagnostic_only_until_same_input_verified':True,
            'deterministic_eval_seed_precision_resize_and_code_pins_required':True},
        'metric_spec':METRIC_SPEC,
        'training_boundary':{'train4_only_for_training_or_training_development':True,
            'permanent_val4_never_optimizer_input':True,'source_split_train_does_not_override_permanent_val_role':True,
            'test_locked_read':False,'calibration_temperature_or_threshold_fit_on_val':False,
            'automatic_model_or_gate_promotion':False},
        'gpu_launch_available_in_this_generator':False,'gpu_collection_started':False,
        'optimizer_input_allowed':False,'formal_training_eligible':False,'product_acceptance_evidence':False,
        'generator_code_sha256':file_sha(Path(__file__)),
        'metrics_code_sha256':file_sha(Path(__file__).with_name('perception_metrics.py')),
        'cpu_adapter':{'path':str(Path(__file__).with_name('val_collection_adapter.py')),
            'sha256':file_sha(Path(__file__).with_name('val_collection_adapter.py')),
            'cpu_mock_backend_only':True,'real_val_collection_available':False,
            'new_val_manifest_without_rewriting_label_bytes':True},
    }
    require(sum(e['action_count'] for e in entries)==225 and sum(e['observation_count'] for e in entries)==229, 'wrong denominators')
    plan['protocol_sha256'] = canonical_sha(plan)
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args(argv)
    if args.output is not None:
        require(not args.output.exists(),'refusing to overwrite existing protocol')
    plan = generate(args.audit_root)
    if args.output is not None:
        with args.output.open('x',encoding='utf-8') as stream:
            json.dump(plan,stream,sort_keys=True,indent=2,allow_nan=False)
            stream.write('\n')
    print(json.dumps({key:plan[key] for key in ('status','protocol_sha256','case_denominator','label_observation_denominator',
        'prediction_denominator','gpu_collection_started','optimizer_input_allowed')},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
