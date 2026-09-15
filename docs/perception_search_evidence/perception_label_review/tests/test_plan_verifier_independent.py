"""Verifier graph checks with mocked file I/O; real-file hashing is not mocked in production."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from collection_contract import EXPECTED, STAGE, canonical_sha, verify_plan


class PlanInputGraphTests(unittest.TestCase):
    def setUp(self):
        self.documents, registry = {}, {}
        def ref(name, sha=None):
            path = str(ROOT / name)
            value = {'path': path, 'sha256': sha or canonical_sha(name), 'bytes': 1}
            registry[path] = value
            return copy.deepcopy(value)
        roles = {row[2]: 'train' for row in EXPECTED.values()}
        output = ROOT/'outputs/takeover/perception_label_sidecar_train4_v1'
        plan = dict(stage=STAGE, physical_gpu=3, maximum_parallel_collectors=1,
            formal_training_eligible=False, optimizer_input_allowed=False, test_locked_used=False,
            repository=str(ROOT), bundle=str(ROOT), output_root=str(output), permanent_scene_roles=roles,
            excluded_legacy_train_samples=[{}]*4, held_out_sample_count=4, python=ref('python'), entries=[])
        fixed = {
            'current_manifest': 'f822609f23da1fd48662c509ad1dfd6fed9ab582f7c8794eabd17faf173a6972',
            'collection_protocol': 'f207c85249c774a17cc67970ab38c57b95d4a94db1c46af9497b21d0ec3c79c3',
            'permanent_manifest': '9268ac7bf53a8392ab58a05112e8bd4db662889fa560708973e8afb807ca1e4e',
            'source_config_inventory': '3a5f559e4c604f749a673af490845c3980463954e8110a4a83c88f1857ad2a8e',
            'admission_policy': '4da01fd3ef238e46db9620948e8f1ca767576c37db648e0f83f5a3d12cce60b7',
        }
        for name, sha in fixed.items():
            plan[name] = ref(name, sha)
        for name in ('semantic_labels.py', 'collection_contract.py', 'freeze_perception_plan.py',
                     'collect_perception_sidecar.py'):
            ref(name)
        manifest = {'samples': []}
        collection = {'permanent_scene_roles': roles, 'source_results': {}}
        configs = {}
        for index, (run_id, (task, dataset_index, scene, episode, count)) in enumerate(EXPECTED.items()):
            entry = dict(run_id=run_id, task=task, dataset_index=dataset_index, canonical_scene_id=scene,
                episode_id=episode, action_count=count, observation_count=count+1,
                permanent_partition_role='train', output_dir=str(output/run_id), prefix_samples=[],
                resolved_config={'task': task}, resolved_config_sha256=canonical_sha({'task': task}))
            for key in ('source', 'source_status', 'source_launch'):
                entry[key] = ref(run_id+'/'+key)
                self.documents[entry[key]['path']] = {}
            entry['entry_config'], entry['dataset'] = ref(task+'/config'), ref(task+'/dataset')
            configs[task] = dict(resolved_config=entry['resolved_config'],
                resolved_config_sha256=entry['resolved_config_sha256'],
                entry_config_path=entry['entry_config']['path'], entry_config_sha256=entry['entry_config']['sha256'],
                dataset_path=entry['dataset']['path'], dataset_sha256=entry['dataset']['sha256'])
            collection['source_results'][run_id] = entry['source']['path']
            for sample_index in range((5, 5, 5, 4)[index]):
                prefix = dict(sample=ref(f'{run_id}/sample{sample_index}'),
                    initial_rgb=ref(f'{run_id}/initial{sample_index}'),
                    prefix_rgb=[dict(ref(f'{run_id}/prefix{sample_index}'),
                                     world_time_s=.1, rgb_array_sha256='a'*64)])
                entry['prefix_samples'].append(copy.deepcopy(prefix))
                manifest['samples'].append(dict(artifacts={'rollout': entry['source']}, frozen_reference=prefix))
            plan['entries'].append(entry)
        self.documents.update({plan['current_manifest']['path']: manifest,
            plan['collection_protocol']['path']: collection,
            plan['source_config_inventory']['path']: {'configs': configs},
            plan['permanent_manifest']['path']: {'samples': [dict(identity={'scene_id': scene}, partition_role=role)
                                                            for scene, role in roles.items()]}})
        plan['artifacts'] = list(registry.values())
        self.plan = plan

    def verify(self, plan=None):
        value = copy.deepcopy(self.plan if plan is None else plan)
        value['plan_sha256'] = canonical_sha(value)
        with patch('collection_contract.artifact'), \
             patch('collection_contract.load', side_effect=lambda path: copy.deepcopy(self.documents[path])), \
             patch('collection_contract.validate_source', return_value={}), \
             patch('collection_contract.prefix_reference', side_effect=lambda row, *args: copy.deepcopy(row['frozen_reference'])):
            return verify_plan(value, verify_files=True)

    def test_complete_reference_graph_is_accepted(self):
        self.assertEqual(self.verify()['prefix_sample_count'], 19)

    def test_duplicate_sample_is_rejected_with_unchanged_denominator(self):
        self.plan['entries'][0]['prefix_samples'][1] = copy.deepcopy(self.plan['entries'][0]['prefix_samples'][0])
        with self.assertRaisesRegex(ValueError, 'duplicate prefix sample references'):
            self.verify()

    def test_every_consumed_source_reference_must_be_in_artifact_registry(self):
        for key in ('source', 'source_status', 'source_launch', 'entry_config', 'dataset'):
            with self.subTest(key=key):
                plan = copy.deepcopy(self.plan)
                missing = plan['entries'][0][key]['path']
                plan['artifacts'] = [ref for ref in plan['artifacts'] if ref['path'] != missing]
                with self.assertRaisesRegex(ValueError, 'consumed input is absent from artifact pins'):
                    self.verify(plan)

    def test_missing_sample_or_media_pin_is_rejected(self):
        original = self.plan['entries'][0]['prefix_samples'][0]
        for ref in (original['sample'], original['initial_rgb'], original['prefix_rgb'][0]):
            with self.subTest(path=ref['path']):
                plan = copy.deepcopy(self.plan)
                plan['artifacts'] = [item for item in plan['artifacts'] if item['path'] != ref['path']]
                with self.assertRaisesRegex(ValueError, 'consumed input is absent from artifact pins'):
                    self.verify(plan)

    def test_resealed_prefix_hash_or_time_cannot_replace_authoritative_reference(self):
        for field, value in (('rgb_array_sha256', 'b'*64), ('world_time_s', 123.)):
            with self.subTest(field=field):
                plan = copy.deepcopy(self.plan)
                plan['entries'][0]['prefix_samples'][0]['prefix_rgb'][0][field] = value
                with self.assertRaisesRegex(ValueError, 'prefix reference pixels, time or membership changed'):
                    self.verify(plan)

    def test_resealed_scene_role_cannot_replace_permanent_manifest(self):
        self.plan['permanent_scene_roles']['invented_scene'] = 'val'
        with self.assertRaisesRegex(ValueError, 'permanent scene role changed'):
            self.verify()


if __name__ == '__main__':
    unittest.main()
