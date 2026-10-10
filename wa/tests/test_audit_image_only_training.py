"""CPU contract tests for a future image-only terminal audit; no GPU or real run."""
import copy
from pathlib import Path
import shlex
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from wa.tools import audit_failure_state_training as prior
from wa.tools import audit_image_only_training as audit


class ImageOnlyTrainingAuditTests(unittest.TestCase):
    def setUp(self):
        self.profile = audit.ImageAuditProfile(
            source=prior.ROOT / 'source_test_image_v1',
            source_commit='a' * 40,
            config=prior.ROOT / 'checkout/wa/jobs/test_image_a800_v1.yaml',
            config_sha256='b' * 64,
            run_name='wa_test_image_a800_v1',
            task_name='wa_test_image_epoch2_v1')
        self.run = prior.JOBS / 'job_1/task_2/wa_test_image_a800_v1'

    def config_fixture(self):
        config = prior.expected_recipe(self.run, prior.PLAN_ROOT)
        config['train_input_mode'] = 'image'
        config['base_lrs'] = [2.5e-7, 5e-7, 5e-6, 1e-6, 1e-5]
        env = dict(WA_SOURCE_COMMIT=self.profile.source_commit,
                   WA_ROOT=str(prior.ROOT),
                   WA_WLA_SOURCE=config['wla_source'],
                   WA_WLA_CHECKPOINT=config['wla_checkpoint'],
                   WA_ENCODER_WEIGHTS=config['encoder_weight'],
                   WA_PARENT_CHECKPOINT=prior.PARENT_PATH,
                   WA_CACHE=str(prior.BASE),
                   WA_INDEX_ROOT=str(prior.INDEX),
                   WA_PYTHON=str(prior.ROOT / 'probe_env/bin/python'),
                   WA_WORLD_KIND='jepa', WA_GPUS='8',
                   WA_OUTPUT=str(self.run),
                   OMP_NUM_THREADS='2', NCCL_DEBUG='WARN',
                   PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
        lines = [f'cd {self.profile.source}',
                 f'test "$(git rev-parse HEAD)" = {self.profile.source_commit}',
                 'test -z "$(git status --porcelain)"']
        lines += [f'export {key}={shlex.quote(value)}' for key, value in env.items()]
        flags = []
        for flag in sorted(prior.FLAGS | {'train-input-mode'}):
            flags.append('--' + flag)
            if flag != 'evaluation-set-adaptation':
                flags.append(shlex.quote(str(config[flag.replace('-', '_')])))
        lines.append('bash wa/scripts/train_world.sh ' + ' '.join(flags))
        yaml_config = dict(cluster='baidu_a800',
                           image='x5-builder:cuda12.8-isaac5.0.0-v2.test1',
                           num_gpus=8, log_folder=self.profile.run_name,
                           tasks=[dict(name=self.profile.task_name, type='shell',
                                       workload_backend='k8s', num_gpus=8,
                                       timeout=86400, cmd='\n'.join(lines))])
        return config, yaml_config

    def test_profile_and_fixed_image_command_accept_exact_fixture(self):
        config, yaml_config = self.config_fixture()
        result = audit.validate_image_config(config, yaml_config,
                                             self.run, self.profile)
        self.assertEqual(result['train_input_mode'], 'image')
        self.assertEqual(result['arguments']['--train-input-mode'], 'image')
        self.assertEqual(result['hardware_profile'], 'a800')

    def test_unready_yaml_guard_cannot_pass_formal_config_audit(self):
        config, yaml_config = self.config_fixture()
        yaml_config['tasks'][0]['cmd'] = (
            "echo 'NOT_SUBMIT_READY' >&2\nexit 2\n"
            + yaml_config['tasks'][0]['cmd'])
        with self.assertRaisesRegex(ValueError, 'NOT_SUBMIT_READY'):
            audit.validate_image_config(config, yaml_config,
                                        self.run, self.profile)

    def test_profile_does_not_accept_missing_or_arbitrary_pins(self):
        self.assertEqual(audit.frozen_profile().source_commit,
                         audit.FROZEN_SOURCE_COMMIT)
        for name in ('FROZEN_SOURCE_COMMIT', 'FROZEN_CONFIG_SHA256',
                     'FROZEN_TASK_NAME'):
            with self.subTest(unset=name), patch.object(audit, name, None):
                with self.assertRaisesRegex(ValueError, 'NOT_RUN_ON_REAL_OUTPUT'):
                    audit.frozen_profile()
        for change in ('source_commit', 'config_sha256', 'source', 'config',
                       'run_name', 'task_name'):
            profile = copy.copy(self.profile)
            values = vars(profile).copy()
            values[change] = {
                'source_commit': 'not-a-sha',
                'config_sha256': 'not-a-sha',
                'source': Path('/tmp/foreign_source'),
                'config': Path('/tmp/foreign_config.yaml'),
                'run_name': 'freeform',
                'task_name': '',
            }[change]
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.ImageAuditProfile(**values).checked()

    def test_wrong_formal_profile_is_rejected_before_reading_artifacts(self):
        with patch.object(prior, 'terminal_metrics', side_effect=AssertionError('run read')):
            with self.assertRaisesRegex(
                    ValueError, 'only the frozen image-only training profile'):
                audit.audit_training(self.run, self.profile)

    def test_config_and_cli_mode_cannot_disagree_or_be_omitted(self):
        for change in ('runtime_missing', 'runtime_sampled',
                       'cli_missing', 'cli_sampled', 'cli_point',
                       'cli_duplicate', 'wrong_cluster', 'wrong_run',
                       'wrong_source_commit'):
            config, yaml_config = self.config_fixture()
            run = self.run
            if change == 'runtime_missing':
                del config['train_input_mode']
            elif change == 'runtime_sampled':
                config['train_input_mode'] = 'sampled'
            elif change == 'cli_missing':
                yaml_config['tasks'][0]['cmd'] = yaml_config['tasks'][0]['cmd'].replace(
                    '--train-input-mode image', '')
            elif change == 'cli_sampled':
                yaml_config['tasks'][0]['cmd'] = yaml_config['tasks'][0]['cmd'].replace(
                    '--train-input-mode image', '--train-input-mode sampled')
            elif change == 'cli_point':
                yaml_config['tasks'][0]['cmd'] = yaml_config['tasks'][0]['cmd'].replace(
                    '--train-input-mode image', '--train-input-mode point')
            elif change == 'cli_duplicate':
                yaml_config['tasks'][0]['cmd'] += ' --train-input-mode image'
            elif change == 'wrong_cluster':
                yaml_config['cluster'] = 'baidu_4090'
            elif change == 'wrong_run':
                run = prior.JOBS / 'job_1/task_2/wa_other_image_a800_v1'
            else:
                yaml_config['tasks'][0]['cmd'] = yaml_config['tasks'][0]['cmd'].replace(
                    self.profile.source_commit, 'c' * 40)
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.validate_image_config(config, yaml_config, run, self.profile)

    def test_actual_mode_exposure_and_checkpoint_tag_are_exact(self):
        config, _ = self.config_fixture()
        actual = dict(train_input_mode='image', actual_total=1233424,
                      mode_exposure=dict(audit.MODE_EXPOSURE))
        checkpoint = dict(train_input_mode='image')
        self.assertEqual(audit.validate_image_exposure(config, actual, checkpoint),
                         audit.MODE_EXPOSURE)
        changes = (
            ('missing_actual_tag', lambda c, a, p: a.pop('train_input_mode')),
            ('wrong_actual_tag', lambda c, a, p: a.update(train_input_mode='sampled')),
            ('missing_mode_count', lambda c, a, p: a.pop('mode_exposure')),
            ('one_point_row', lambda c, a, p: a['mode_exposure'].update(
                image=1233423, point=1)),
            ('one_mixed_row', lambda c, a, p: a['mode_exposure'].update(
                image=1233423, mixed=1)),
            ('wrong_total', lambda c, a, p: a.update(actual_total=1233423)),
            ('bool_count', lambda c, a, p: a['mode_exposure'].update(point=False)),
            ('missing_checkpoint_tag', lambda c, a, p: p.pop('train_input_mode')),
            ('mixed_checkpoint', lambda c, a, p: p.update(train_input_mode='sampled')),
        )
        for name, mutate in changes:
            new_config, new_actual, new_checkpoint = copy.deepcopy(
                (config, actual, checkpoint))
            mutate(new_config, new_actual, new_checkpoint)
            with self.subTest(change=name), self.assertRaises(ValueError):
                audit.validate_image_exposure(new_config, new_actual, new_checkpoint)

    def test_worker_postcheck_mode_and_artifact_bindings(self):
        checkpoint_sha = 'c' * 64
        files = {str(self.run / name): 'd' * 64
                 for name in audit.POSTCHECK_FILES}
        files[str(self.run / 'checkpoint.pt')] = checkpoint_sha
        files[str(self.run) + '.launch.json'] = 'e' * 64
        pins = SimpleNamespace(files=files)
        postcheck = dict(status='TRAINING_WORKER_POSTCHECK_PASS_OFFLINE_ONLY',
                         source_commit=self.profile.source_commit,
                         final_step=61252, new_optimizer_updates=38545,
                         completed_epochs=2, plan_sha256=prior.PLAN_SHA,
                         launch_sha256='e' * 64, train_input_mode='image',
                         mode_exposure=dict(audit.MODE_EXPOSURE),
                         closed_loop=False, SR=None,
                         artifacts_sha256={
                             name: files[str(self.run / name)]
                             for name in audit.POSTCHECK_FILES})
        result = audit.validate_image_postcheck(postcheck, self.profile,
                                                pins, self.run, checkpoint_sha)
        self.assertEqual(result['train_input_mode'], 'image')
        for change in ('mode', 'counts', 'source', 'launch',
                       'missing_artifact', 'checkpoint', 'claim_sr'):
            bad = copy.deepcopy(postcheck)
            if change == 'mode':
                bad['train_input_mode'] = 'sampled'
            elif change == 'counts':
                bad['mode_exposure']['point'] = 1
            elif change == 'source':
                bad['source_commit'] = 'f' * 64
            elif change == 'launch':
                bad['launch_sha256'] = 'f' * 64
            elif change == 'missing_artifact':
                del bad['artifacts_sha256']['metrics.json']
            elif change == 'checkpoint':
                bad['artifacts_sha256']['checkpoint.pt'] = 'f' * 64
            else:
                bad['SR'] = .9
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.validate_image_postcheck(bad, self.profile,
                                               pins, self.run, checkpoint_sha)

    def image_developer_bundle(self):
        root = audit.IMAGE_DEVELOPER_RUN
        config = prior.read_json(root / 'config.json')
        environment = prior.read_json(root / 'environment.json')
        actual = prior.read_json(root / 'actual_exposure_epoch1.json')
        arrays = prior.load_npz(root / 'actual_exposure_epoch1.npz')
        metrics = prior.read_json(root / 'metrics.json')
        rows = [prior.strict_json(line)
                for line in (root / 'train.jsonl').read_text().splitlines()]
        log_text = Path(str(root) + '.log').read_text()
        baseline_wla = prior.read_json(
            prior.A800_DEVELOPER_RUN / 'environment.json')['wla']
        return [config, environment, actual, arrays, metrics, rows,
                log_text, copy.deepcopy(environment['source_sha256']), baseline_wla]

    def test_pinned_one_a800_image_developer_is_separate_and_revalidated(self):
        self.assertEqual(len(audit.IMAGE_DEVELOPER_PINS), 9)
        self.assertNotEqual(audit.IMAGE_DEVELOPER_PINS,
                            prior.A800_DEVELOPER_PINS)
        source_hashes = prior.inventory(audit.FROZEN_SOURCE)
        launch = dict(
            image_only_developer_sha256=dict(audit.IMAGE_DEVELOPER_PINS),
            image_only_developer_evidence=audit.expected_image_developer_evidence())
        pins = prior.Pins()
        result = audit.bind_image_developer_training(
            pins, launch, source_hashes, audit.IMAGE_DEVELOPER_COMMIT)
        pins.finish()
        self.assertEqual(result['independently_validated']['mode_exposure'],
                         dict(image=16, point=0, mixed=0, total=16))
        self.assertEqual(result['independently_validated']['finite_gradients'], 4)
        self.assertFalse(result['independently_validated']['checkpoint_produced'])
        self.assertIsNone(result['independently_validated']['SR'])

    def test_image_developer_rejects_bad_launch_pins_or_source_commit(self):
        source_hashes = prior.inventory(audit.FROZEN_SOURCE)
        good = dict(image_only_developer_sha256=dict(audit.IMAGE_DEVELOPER_PINS),
                    image_only_developer_evidence=audit.expected_image_developer_evidence())
        for change in ('sha', 'evidence', 'source'):
            launch = copy.deepcopy(good)
            commit = audit.IMAGE_DEVELOPER_COMMIT
            if change == 'sha':
                first = next(iter(launch['image_only_developer_sha256']))
                launch['image_only_developer_sha256'][first] = 'f' * 64
            elif change == 'evidence':
                launch['image_only_developer_evidence']['mode_exposure']['point'] = 1
            else:
                commit = 'f' * 40
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.bind_image_developer_training(
                    prior.Pins(), launch, source_hashes, commit)

    def test_image_developer_cpu_negative_records(self):
        for change in ('sampled_config', 'one_point', 'wrong_total',
                       'dirty_source', 'wrong_gpu', 'source_inventory',
                       'nonfinite_gradient', 'stale_step', 'fake_sr',
                       'missing_complete', 'array_missing'):
            bundle = copy.deepcopy(self.image_developer_bundle())
            if change == 'sampled_config':
                bundle[0]['train_input_mode'] = 'sampled'
            elif change == 'one_point':
                bundle[2]['mode_exposure']['point'] = 1
            elif change == 'wrong_total':
                bundle[2]['actual_total'] = 15
            elif change == 'dirty_source':
                bundle[1]['dirty'] = ' M wa/wm/train.py'
            elif change == 'wrong_gpu':
                bundle[1]['gpu_names'] = ['NVIDIA GeForce RTX 4090']
            elif change == 'source_inventory':
                bundle[7]['wa/wm/train.py'] = 'f' * 64
            elif change == 'nonfinite_gradient':
                bundle[5][0]['grad_norm'] = float('nan')
            elif change == 'stale_step':
                bundle[5][-1]['step'] = 22710
            elif change == 'fake_sr':
                bundle[4]['metrics']['image']['SR'] = .8
            elif change == 'missing_complete':
                bundle[6] = bundle[6].replace('COMPLETE ', 'FINISHED ')
            else:
                del bundle[3]['position_counts']
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.validate_image_developer_records(*bundle)

if __name__ == '__main__':
    unittest.main()
