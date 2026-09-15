"""Freeze label-only train4 replay inputs; never imports Habitat or starts a GPU.

Without --write-plan this only validates inputs and prints a summary. An explicit
--write-plan creates a new immutable plan, never overwriting an existing file.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from collection_contract import (EXPECTED, STAGE, artifact, canonical_scene,
    canonical_sha, load, prefix_reference, require, validate_source, verify_plan, write_new)

REPO = Path('/data/nfs/share/wam_tracking/OmTrackVLA')
COLLECTION_SHA = 'f207c85249c774a17cc67970ab38c57b95d4a94db1c46af9497b21d0ec3c79c3'
MANIFEST_SHA = 'f822609f23da1fd48662c509ad1dfd6fed9ab582f7c8794eabd17faf173a6972'
CONFIG_INVENTORY_SHA = '3a5f559e4c604f749a673af490845c3980463954e8110a4a83c88f1857ad2a8e'
PERMANENT_SHA = '9268ac7bf53a8392ab58a05112e8bd4db662889fa560708973e8afb807ca1e4e'


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def build_plan(repository=REPO, bundle=None):
    repo = Path(repository).resolve(strict=True)
    require(repo == REPO, 'v1 freezes the authorized remote repository only')
    bundle = Path(bundle or __file__).resolve()
    if bundle.is_file():
        bundle = bundle.parent
    refs = {}
    def add(path, expected=None):
        ref = artifact(path, expected)
        require(str(path).lower().find('test_locked') < 0, 'test_locked input forbidden')
        previous = refs.get(ref['path'])
        require(previous is None or previous == ref, 'conflicting artifact pins')
        refs[ref['path']] = ref
        return ref
    collection_ref = add(repo/'outputs/takeover/long_prefix_collection_v1/frozen_collection_plan.json', COLLECTION_SHA)
    manifest_ref = add(repo/'outputs/takeover/long_prefix_admission_v2/manifest.json', MANIFEST_SHA)
    config_ref = add(bundle/'source_config_inventory.json', CONFIG_INVENTORY_SHA)
    admission_ref = add(bundle/'perception_sidecar_admission_policy_v1.json',
        '4da01fd3ef238e46db9620948e8f1ca767576c37db648e0f83f5a3d12cce60b7')
    collection, manifest, configs = load(collection_ref['path']), load(manifest_ref['path']), load(config_ref['path'])
    require(canonical_sha({k:v for k,v in collection.items() if k != 'plan_sha256'}) == collection['plan_sha256'], 'original collection seal changed')
    permanent_ref = add(collection['permanent_manifest']['path'], PERMANENT_SHA)
    permanent = load(permanent_ref['path'])
    roles = {}
    for sample in permanent['samples']:
        scene, role = sample['identity']['scene_id'], sample['partition_role']
        require(scene not in roles or roles[scene] == role, 'conflicting permanent scene roles')
        roles[scene] = role
    require(roles == collection['permanent_scene_roles'], 'permanent role table changed')
    require(manifest['coverage']['train']['sample_count'] == 23
            and manifest['coverage']['val']['sample_count'] == 4 and len(manifest['samples']) == 27,
            'current manifest denominator changed')
    require(set(collection['source_results']) == set(EXPECTED), 'four exact long sources required')
    expected_source_shas = {}
    for row in collection['entries']:
        source = row['source_rollout']
        require(source['path'] not in expected_source_shas or expected_source_shas[source['path']] == source['sha256'], 'conflicting source hashes')
        expected_source_shas[source['path']] = source['sha256']
    source_paths = set(collection['source_results'].values())
    held_out = [s for s in manifest['samples'] if s['partition_role'] == 'val']
    legacy = [s for s in manifest['samples'] if s['partition_role'] == 'train' and s['artifacts']['rollout']['path'] not in source_paths]
    require(len(held_out) == 4 and len(legacy) == 4, 'legacy/held-out accounting changed')
    entries = []
    output = repo/'outputs/takeover/perception_label_sidecar_train4_v1'
    for run_id in EXPECTED:
        source_path = Path(collection['source_results'][run_id])
        source_ref = add(source_path, expected_source_shas[str(source_path)])
        launch_ref = add(source_path.parent/'launch_contract.json')
        status_ref = add(source_path.parent/'execution_status.json')
        source, launch, status = load(source_path), load(launch_ref['path']), load(status_ref['path'])
        identity = validate_source(source, status, launch, run_id, source_ref['sha256'], roles)
        task = identity['task']
        inventory = configs['configs'][task]
        resolved = inventory['resolved_config']
        original = launch['run']['original_sim_configuration']
        require(canonical_sha(resolved) == inventory['resolved_config_sha256'] == original['resolved_config_sha256'],
                'source resolved config mismatch')
        add(inventory['entry_config_path'], inventory['entry_config_sha256'])
        data_ref = add(inventory['dataset_path'], inventory['dataset_sha256'])
        require(original['dataset_sha256'] == data_ref['sha256'], 'source dataset no longer matches')
        with gzip.open(data_ref['path'], 'rt', encoding='utf-8') as stream:
            dataset = json.load(stream)
        episode = dataset['episodes'][identity['dataset_index']]
        require(str(episode['episode_id']) == identity['episode_id']
                and canonical_scene(episode['scene_id']) == identity['canonical_scene_id']
                and int(episode['info']['main_human_semantic_id']) == int(identity['assigned_humanoid_semantic_ids']['0']),
                'raw dataset identity or target assignment changed')
        matched = [s for s in manifest['samples'] if s['artifacts']['rollout']['path'] == str(source_path)]
        prefix_samples = []
        for row in matched:
            require(row['partition_role'] == 'train'
                    and row['identity']['scene_id'] == identity['canonical_scene_id'], 'held-out or different-scene sample')
            ref = prefix_reference(row, source_path, source, identity['action_count'])
            prefix_samples.append(ref)
            add(ref['sample']['path'], ref['sample']['sha256'])
            for media in [ref['initial_rgb'], *ref['prefix_rgb']]:
                add(media['path'], media['sha256'])
            add(row['artifacts']['report']['path'], row['artifacts']['report']['sha256'])
        # Freeze original runtime code, including the measured source runner.
        for relative, sha in launch['code_sha256'].items():
            add(repo/relative, sha)
        # Selected scene closure: the four scene directories (glb, navmesh,
        # semantic materials/metadata), without enumerating other scenes.
        scene_path = (repo/source['scene_id']).resolve(strict=True)
        for path in sorted(scene_path.parent.rglob('*')):
            if path.is_file():
                add(path)
        # Concrete files referenced by resolved config/selected raw episode.
        # URDF/motion parents include their local meshes and textures.
        dependency_paths = set()
        for value in strings({'config': resolved, 'episode': episode}):
            if value.startswith(('data/', './data/')) and '{' not in value:
                path = (repo/value).resolve(strict=False)
                if path.is_file():
                    dependency_paths.add(path)
                    if path.suffix.lower() == '.urdf' or ('humanoid' in str(path) and path.suffix == '.pkl'):
                        dependency_paths.update(p for p in path.parent.rglob('*') if p.is_file())
        for path in sorted(dependency_paths):
            add(path)
        entries.append(dict(identity, run_id=run_id, source=source_ref, source_launch=launch_ref,
            source_status=status_ref, permanent_partition_role='train', prefix_samples=prefix_samples,
            resolved_config=resolved, resolved_config_sha256=canonical_sha(resolved),
            entry_config=add(inventory['entry_config_path'], inventory['entry_config_sha256']),
            dataset=data_ref, selected_raw_episode_sha256=canonical_sha(episode),
            output_dir=str(output/run_id)))
    # Pin importable replay/labels code and configuration inheritance. No policy
    # weights are loaded. Original checkpoint metadata stays in the source pin.
    for directory in (repo/'habitat-lab/habitat', repo/'evt_bench'):
        require(directory.is_dir(), 'missing runtime code directory: ' + str(directory))
        for path in sorted(directory.rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.yaml', '.yml'):
                add(path)
    for relative in ('scripts/probe_next007_recovery_sequence_v2.py',
                     'scripts/recovery_sequence_export_v2.py', 'scripts/run_egl.sh',
                     'omtrackvla/evaluation/end_to_end_closed_loop.py'):
        add(repo/relative)
    for name in ('semantic_labels.py', 'collection_contract.py', 'freeze_perception_plan.py',
                 'collect_perception_sidecar.py'):
        add(bundle/name)
    python = add(collection['python'], collection['python_sha256'])
    plan = dict(stage=STAGE, schema_version=1, repository=str(repo), bundle=str(bundle),
        status='frozen_before_any_new_sidecar_render', physical_gpu=3, maximum_parallel_collectors=1,
        output_root=str(output), python=python, entries=entries, permanent_scene_roles=roles,
        collection_protocol=collection_ref, current_manifest=manifest_ref, permanent_manifest=permanent_ref,
        admission_policy=admission_ref,
        source_config_inventory=config_ref, expected_source_count=4, expected_observation_count=370,
        expected_prefix_sample_count=19, held_out_sample_count=4,
        excluded_legacy_train_samples=[dict(sample=s['artifacts']['sample'], source=s['artifacts']['rollout'],
            reason='different_old_short_source_not_forced_to_match') for s in legacy],
        formal_training_eligible=False, optimizer_input_allowed=False, test_locked_used=False,
        product_acceptance_evidence=False, replay_tolerance=1e-4, world_time_tolerance_s=1e-8,
        rgb_statistics_tolerance_0_255=1e-5,
        maximum_worker_wall_seconds=1800,
        source_time_reference='source_result_does_not_record_world_time; verify_every_existing_sample_prefix_time',
        gt_boundary='raw_panoptic_and_geometry_only_for_labels_and_audit; replay_actions_unchanged; no_policy_loaded',
        data_inventory_scope='full_compressed_task_episode_files; selected_scene_directories; concrete_files_from_resolved_config_and_selected_episode; local_URDF_motion_asset_directories',
        failure_policy='all_four_entries_retained; no_source_substitution; no_retry; output_never_overwritten',
        environment={'MAGNUM_CUDA_DEVICE': '0', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                     'OPENBLAS_NUM_THREADS': '1', 'OPENCV_FOR_THREADS_NUM': '1'},
        artifacts=sorted(refs.values(), key=lambda ref: ref['path']))
    plan['plan_sha256'] = canonical_sha(plan)
    verify_plan(plan, verify_files=False)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=REPO)
    parser.add_argument('--write-plan', type=Path)
    args = parser.parse_args()
    if args.write_plan is not None:
        require(not args.write_plan.exists(), 'refusing to overwrite plan')
    plan = build_plan(args.repository)
    if args.write_plan is not None:
        write_new(args.write_plan, plan)
    print(json.dumps(dict(verify_plan(plan, verify_files=False), artifact_count=len(plan['artifacts']),
        plan_written=None if args.write_plan is None else str(args.write_plan)), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
