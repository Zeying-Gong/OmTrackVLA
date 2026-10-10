"""Independent, opt-in terminal auditor for the fixed image-only failure-state run.

The formal source, YAML, and task profile is frozen below, but this module has
not audited a real image-only run. It does not certify closed-loop success.
"""
import argparse
from dataclasses import dataclass
import importlib
import json
import math
from pathlib import Path
import re
import shlex

import torch
import numpy as np
import yaml

from wa.tools import audit_failure_state_training as prior

# Frozen to the clean, committed image-only source and final full-task YAML.
# Unsetting any pin makes the formal CLI fail before reading any run artifact.
FROZEN_SOURCE_COMMIT = '76635eb257e43ab2c8194e5b358cac023a3351a0'
FROZEN_CONFIG_SHA256 = 'f5413cf82f95d6d0d032675445ecb58b7d307d59a907268d02400d34b6a4c0ea'
FROZEN_TASK_NAME = 'wa_failure_state_image_only_epoch2_v1'
FROZEN_SOURCE = prior.ROOT / 'source_failure_state_image_only_v1'
FROZEN_CONFIG = prior.ROOT / 'checkout/wa/jobs/failure_state_image_only_a800_v1.yaml'
FROZEN_RUN_NAME = 'wa_failure_state_image_only_a800_v1'
MODE_EXPOSURE = dict(image=1233424, point=0, mixed=0, total=1233424)
POSTCHECK_FILES = ('train.jsonl', 'metrics.json', 'actual_exposure_epoch1.json',
                   'actual_exposure_epoch1.npz', 'config.json', 'environment.json',
                   'failure_state_exposure.json', 'checkpoint.pt')
IMAGE_DEVELOPER_RUN = prior.ROOT / 'artifacts/failure_state_image_only_developer_20261010_v1'
IMAGE_DEVELOPER_COMMIT = '76635eb257e43ab2c8194e5b358cac023a3351a0'
IMAGE_DEVELOPER_EXPOSURE = dict(image=16, point=0, mixed=0, total=16)
IMAGE_DEVELOPER_FILE_PINS = {
    'actual_exposure_epoch1.json': ('f08520404a53087ae66e3105e58733f9b0647526f66d60df733e5f0efde28adf', 3961),
    'actual_exposure_epoch1.npz': ('236e39ebd3626bdb0cb625671b15e614688361fb15aae5a0720379f772793d13', 19766),
    'config.json': ('3053df655ebec91cb4cf25322732ededa7aebeca4a766a735db4f9598c28f561', 3837),
    'dual_teacher_exposure.json': ('51e11ca68db685843335fa6a6e5bef1ed2eb7565e62700951549b34e9d012309', 659),
    'environment.json': ('e0e1f499e6015427186413fa1e4bc7844032bfbaedf8d6d633ba52b971295b83', 61931),
    'failure_state_exposure.json': ('5965f07c0aa617cd19b3f12aa73e9e9370934de3ed25aa85df3d0d4b3ab9451d', 38264),
    'metrics.json': ('3ef5c1c83cb852a35719874af3cc55be77099e3073ce967ec4523b6773c7846d', 825),
    'train.jsonl': ('12e1697d9d532fef0bbb91d2bf536318990871a0d11e9ef2a7bc46d16436824a', 2099),
}
IMAGE_DEVELOPER_LOG_PIN = ('54347ae0401cd546c950735c04f50e11af8f56211bc032c18d6afd8f42c911d7', 3414)
IMAGE_DEVELOPER_PINS = {str(IMAGE_DEVELOPER_RUN / name): sha
                        for name, (sha, _) in IMAGE_DEVELOPER_FILE_PINS.items()}
IMAGE_DEVELOPER_PINS[str(IMAGE_DEVELOPER_RUN) + '.log'] = IMAGE_DEVELOPER_LOG_PIN[0]



@dataclass(frozen=True)
class ImageAuditProfile:
    source: Path
    source_commit: str
    config: Path
    config_sha256: str
    run_name: str
    task_name: str

    def checked(self):
        prior.require(type(self.source_commit) is str
                      and re.fullmatch(r'[0-9a-f]{40}', self.source_commit),
                      'explicit lowercase 40-hex source commit required')
        prior.sha_string(self.config_sha256)
        prior.require(isinstance(self.source, Path) and isinstance(self.config, Path)
                      and self.source.is_absolute() and self.config.is_absolute()
                      and self.source.resolve() == self.source
                      and self.config.resolve() == self.config
                      and self.source.parent == prior.ROOT
                      and self.config.parent == prior.ROOT / 'checkout/wa/jobs',
                      'image-only source/config must use fixed NAS roots')
        prior.require(type(self.run_name) is str
                      and re.fullmatch(r'wa_[a-z0-9_]+_a800_v[0-9]+', self.run_name)
                      and type(self.task_name) is str
                      and re.fullmatch(r'wa_[a-z0-9_]+', self.task_name),
                      'explicit image-only run/task names required')
        return self


def frozen_profile():
    prior.require(FROZEN_SOURCE_COMMIT is not None
                  and FROZEN_CONFIG_SHA256 is not None
                  and FROZEN_TASK_NAME is not None,
                  'NOT_RUN_ON_REAL_OUTPUT: image-only source/YAML/task pins are not frozen')
    return ImageAuditProfile(FROZEN_SOURCE, FROZEN_SOURCE_COMMIT,
                             FROZEN_CONFIG, FROZEN_CONFIG_SHA256,
                             FROZEN_RUN_NAME, FROZEN_TASK_NAME).checked()


def parse_image_command(cmd, expected_env):
    prior.require(type(cmd) is str and type(expected_env) is dict, 'image command/env types')
    folded = re.sub(r'\\\r?\n[ \t]*', ' ', cmd)
    lines = [line.strip() for line in folded.splitlines()
             if re.match(r'^\s*bash\s+wa/scripts/train_world\.sh(?:\s|$)', line)]
    prior.require(len(lines) == 1, 'exactly one formal image train command required')
    tokens = shlex.split(lines[0], comments=True)
    prior.require(tokens[:2] == ['bash', 'wa/scripts/train_world.sh'],
                  'wrong image training entrypoint')
    allowed = prior.FLAGS | {'train-input-mode'}
    arguments = {}
    i = 2
    while i < len(tokens):
        key = tokens[i]
        prior.require(key.startswith('--') and key[2:] in allowed and key not in arguments,
                      'duplicate/unknown image training option: ' + key)
        if key == '--evaluation-set-adaptation':
            arguments[key] = True
            i += 1
        else:
            prior.require(i + 1 < len(tokens) and not tokens[i + 1].startswith('--'),
                          'missing image training argument')
            arguments[key] = prior.expand(tokens[i + 1], expected_env)
            i += 2
    return arguments


def validate_image_config(config, yaml_config, run, profile):
    profile.checked()
    recipe = prior.expected_recipe(Path(run), prior.PLAN_ROOT)
    recipe['train_input_mode'] = 'image'
    for key, value in recipe.items():
        prior.exact(config.get(key), value, 'image runtime config: ' + key)
    for key in ('recovery_cache', 'recovery_index', 'data_root', 'source_prefix'):
        prior.require(config.get(key) is None, 'foreign image data route: ' + key)
    rates = config.get('base_lrs')
    prior.require(type(rates) is list and len(rates) == 5
                  and all(prior.finite(x) and x > 0 for x in rates),
                  'five resumed image LR groups required')
    for key, value in dict(cluster='baidu_a800', num_gpus=8,
                           image='x5-builder:cuda12.8-isaac5.0.0-v2.test1',
                           log_folder=profile.run_name).items():
        prior.exact(yaml_config.get(key), value, 'image YAML: ' + key)
    tasks = yaml_config.get('tasks')
    prior.require(type(tasks) is list and len(tasks) == 1, 'one full image training task')
    task = tasks[0]
    for key, value in dict(name=profile.task_name, num_gpus=8, type='shell',
                           workload_backend='k8s', timeout=86400).items():
        prior.exact(task.get(key), value, 'image YAML task: ' + key)
    cmd = task.get('cmd')
    prior.require(type(cmd) is str and 'NOT_SUBMIT_READY' not in cmd,
                  'image YAML still marked NOT_SUBMIT_READY')
    prior.require(type(cmd) is str and 'cd ' + str(profile.source) in cmd
                  and profile.source_commit in cmd and 'git status --porcelain' in cmd,
                  'frozen image source/clean guard absent')
    match = re.fullmatch(re.escape(str(prior.JOBS)) +
                         r'/job_([0-9]+)/task_([0-9]+)/' +
                         re.escape(profile.run_name), str(run))
    prior.require(match is not None, 'exact image Job/Task output route required')
    env = dict(MD_AK_JOB_ID=match.group(1), MD_AK_TASK_ID=match.group(2))
    for line in cmd.splitlines():
        if not line.strip().startswith('export '):
            continue
        for token in shlex.split(line.strip())[1:]:
            prior.require('=' in token, 'plain image export assignment required')
            name, value = token.split('=', 1)
            prior.require(name not in env, 'duplicate image export: ' + name)
            env[name] = prior.expand(value, env)
    wanted = dict(WA_SOURCE_COMMIT=profile.source_commit,
                  WA_ROOT=str(prior.ROOT), WA_WLA_SOURCE=recipe['wla_source'],
                  WA_WLA_CHECKPOINT=recipe['wla_checkpoint'],
                  WA_ENCODER_WEIGHTS=recipe['encoder_weight'],
                  WA_PARENT_CHECKPOINT=prior.PARENT_PATH, WA_CACHE=str(prior.BASE),
                  WA_INDEX_ROOT=str(prior.INDEX),
                  WA_PYTHON=str(prior.ROOT / 'probe_env/bin/python'),
                  WA_WORLD_KIND='jepa', WA_GPUS='8', WA_OUTPUT=str(run),
                  OMP_NUM_THREADS='2', NCCL_DEBUG='WARN', PYTHONNOUSERSITE='1',
                  PYTHONDONTWRITEBYTECODE='1')
    prior.same({k: v for k, v in env.items() if not k.startswith('MD_AK_')},
               wanted, 'image worker environment differs')
    arguments = parse_image_command(cmd, env)
    expected = {'--' + flag:
                (True if flag == 'evaluation-set-adaptation'
                 else str(recipe[flag.replace('-', '_')]))
                for flag in prior.FLAGS | {'train-input-mode'}}
    prior.same(arguments, expected, 'image YAML/runtime training flags differ')
    return dict(arguments=arguments, environment=env, hardware_profile='a800',
                train_input_mode='image')


def validate_image_exposure(config, actual, checkpoint=None):
    prior.exact(config.get('train_input_mode'), 'image', 'image config mode')
    prior.exact(actual.get('train_input_mode'), 'image', 'image actual mode')
    prior.exact(actual.get('actual_total'), MODE_EXPOSURE['total'],
                'image actual consumed positions')
    prior.same(actual.get('mode_exposure'), MODE_EXPOSURE,
               'all actual consumed positions must be image mode')
    if checkpoint is not None:
        prior.exact(checkpoint.get('train_input_mode'), 'image',
                    'image checkpoint mode')
    return dict(MODE_EXPOSURE)


def validate_image_postcheck(postcheck, profile, pins, run, checkpoint_sha):
    profile.checked()
    for key, value in dict(status='TRAINING_WORKER_POSTCHECK_PASS_OFFLINE_ONLY',
                           source_commit=profile.source_commit,
                           final_step=prior.FINAL, new_optimizer_updates=prior.UPDATES,
                           completed_epochs=2, plan_sha256=prior.PLAN_SHA,
                           launch_sha256=pins.files[str(run) + '.launch.json'],
                           train_input_mode='image', mode_exposure=MODE_EXPOSURE,
                           closed_loop=False, SR=None).items():
        prior.same(postcheck.get(key), value, 'image worker postcheck: ' + key)
    expected = {name: pins.files[str(run / name)] for name in POSTCHECK_FILES}
    prior.exact(expected['checkpoint.pt'], checkpoint_sha,
                'postcheck image checkpoint binding')
    prior.same(postcheck.get('artifacts_sha256'), expected,
               'image worker postcheck artifact inventory')
    return dict(status=postcheck['status'], train_input_mode='image',
                mode_exposure=dict(MODE_EXPOSURE))


def expected_image_developer_evidence():
    return dict(
        schema='failure_state_image_only_developer_training_evidence_v1',
        status='ONE_A800_FOUR_UPDATE_IMAGE_ONLY_DIAGNOSTIC_PASS_NOT_EIGHT_RANK_PROOF',
        run=str(IMAGE_DEVELOPER_RUN), source_commit=IMAGE_DEVELOPER_COMMIT,
        input_sha256=dict(IMAGE_DEVELOPER_PINS), world_size=1,
        gpu_name='NVIDIA A800-SXM4-80GB', first_step=prior.START + 1,
        final_step=prior.START + 4, optimizer_updates=4,
        consumed_positions=16, mode_exposure=dict(IMAGE_DEVELOPER_EXPOSURE),
        source_counts=dict(base=7, teacher=6, recovery=3),
        heldout_windows_per_mode=2, checkpoint_produced=False,
        eight_rank_gpu_execution_verified=False, closed_loop=False, SR=None)


def validate_image_developer_records(config, environment, actual, arrays,
                                     metrics, rows, log_text, source_hashes,
                                     baseline_wla):
    """CPU checks on the pinned one-A800 diagnostic, not full training proof."""
    recipe = prior.expected_recipe(IMAGE_DEVELOPER_RUN, prior.PLAN_ROOT)
    recipe.update(diagnostic=True, effective_batch=4, train_rows=16,
                  heldout_rows=2, world_size=1, train_input_mode='image')
    for key, value in recipe.items():
        prior.exact(config.get(key), value, 'image developer config: ' + key)
    for key in ('recovery_cache', 'recovery_index', 'data_root', 'source_prefix'):
        prior.exact(config.get(key), None, 'image developer foreign route: ' + key)
    rates = config.get('base_lrs')
    prior.require(type(rates) is list and len(rates) == 5
                  and all(prior.finite(x) and x > 0 for x in rates),
                  'image developer resumed LR groups')
    for key, value in dict(commit=IMAGE_DEVELOPER_COMMIT, dirty='',
                           torch='2.8.0+cu128',
                           gpu_names=['NVIDIA A800-SXM4-80GB']).items():
        prior.exact(environment.get(key), value, 'image developer environment: ' + key)
    prior.require(type(source_hashes) is dict and len(source_hashes) == 264,
                  'image developer frozen source inventory count')
    prior.same(environment.get('source_sha256'), source_hashes,
               'image developer source inventory differs from frozen source')
    prior.same(environment.get('wla'), baseline_wla,
               'image developer WLA initialization differs from admitted parent')
    for key, value in dict(schema='failure_state_sampling_runtime_candidate_v1',
                           status='DIAGNOSTIC_PARTIAL_EXPOSURE_ONLY',
                           plan_sha256=prior.PLAN_SHA, pre_ddp_total=1233424,
                           actual_total=16, actual_base=7, actual_teacher=6,
                           actual_recovery=3, unconsumed_positions=1233408,
                           repeated_positions=0, optimizer_steps_completed=4,
                           diagnostic=True, metadata_entered_policy=False,
                           train_input_mode='image').items():
        prior.exact(actual.get(key), value, 'image developer exposure: ' + key)
    prior.same(actual.get('mode_exposure'), IMAGE_DEVELOPER_EXPOSURE,
               'image developer consumed-mode counts')
    dimensions = dict(position_counts=(1233424, 16, 1),
                      base_counts=(726631, 7, 1),
                      teacher_counts=(436816, 6, 2),
                      recovery_counts=(6864, 3, 1))
    prior.require(type(arrays) is dict and set(arrays) == set(dimensions),
                  'image developer actual array inventory')
    for name, (length, total, maximum) in dimensions.items():
        value = arrays[name]
        prior.require(isinstance(value, np.ndarray) and value.dtype == np.int64
                      and value.shape == (length,) and bool((value >= 0).all())
                      and int(value.sum()) == total and int(value.max()) == maximum,
                      'image developer actual exposure array: ' + name)
    for key, value in dict(status='DIAGNOSTIC_PASS', kind='jepa',
                           steps=prior.START + 4, closed_loop=False,
                           edge_latency_verified=False,
                           point_source='simulated_uwb').items():
        prior.exact(metrics.get(key), value, 'image developer metrics: ' + key)
    modes = metrics.get('metrics')
    prior.require(type(modes) is dict and set(modes) == {'image', 'point', 'mixed'},
                  'three image developer offline validation modes')
    for name, row in modes.items():
        prior.exact(row.get('windows'), 2, 'image developer heldout windows: ' + name)
        prior.require(all(prior.finite(row.get(key)) for key in
                          ('ADE_m', 'FDE_m', 'yaw_MAE_rad')),
                      'image developer finite offline metrics: ' + name)
        prior.require('SR' in row and 'collision_rate' in row
                      and row['SR'] is None and row['collision_rate'] is None,
                      'image developer metrics cannot claim SR/CR')
    prior.require(type(rows) is list and len(rows) == 4,
                  'four image developer optimizer log records required')
    for phase_step, row in enumerate(rows, 1):
        step = prior.START + phase_step
        for key, value in dict(step=step, total_steps=prior.START + 4,
                               phase_step=phase_step, epoch=1,
                               loss_aggregation='DDP_mean_over_accumulation_group').items():
            prior.exact(row.get(key), value, 'image developer train log: ' + key)
        prior.require(all(prior.finite(row.get(key)) for key in
                          ('loss', 'flow', 'geometry', 'world', 'grad_norm',
                           'peak_allocated_gib', 'elapsed_s',
                           'loss_rank0_last_microbatch'))
                      and row['grad_norm'] > 0
                      and math.isclose(row['loss'],
                                       row['flow'] + .5 * row['geometry']
                                       + .1 * row['world'],
                                       rel_tol=2e-5, abs_tol=2e-6),
                      'image developer nonfinite or changed loss/gradient')
        multiplier = .1 + .9 * .5 * (1 + math.cos(math.pi * (phase_step - 1) / 4))
        logged = row.get('lr')
        prior.require(type(logged) is list and len(logged) == 5
                      and all(prior.finite(value) and math.isclose(
                          value, base * multiplier, rel_tol=1e-12, abs_tol=0)
                          for value, base in zip(logged, rates)),
                      'image developer resumed four-step LR schedule')
    prior.require(type(log_text) is str and log_text.count('COMPLETE ') == 1
                  and not any(token in log_text for token in
                              ('Traceback (most recent call last)',
                               'CUDA out of memory', 'FloatingPointError',
                               'Segmentation fault', 'Fatal Python error')),
                  'image developer fatal or missing completion log')
    return dict(mode_exposure=dict(IMAGE_DEVELOPER_EXPOSURE),
                optimizer_updates=4, finite_gradients=4,
                heldout_windows_per_mode=2, gpu_names=environment['gpu_names'],
                checkpoint_produced=False, closed_loop=False, SR=None)


def bind_image_developer_training(pins, launch, source_hashes, source_commit):
    """Independently rehash exact image diagnostic bytes and bind launch claim."""
    prior.exact(source_commit, IMAGE_DEVELOPER_COMMIT,
                'image developer/frozen source commit mismatch')
    evidence = expected_image_developer_evidence()
    prior.same(launch.get('image_only_developer_sha256'), IMAGE_DEVELOPER_PINS,
               'image developer independent launch SHA inventory')
    prior.same(launch.get('image_only_developer_evidence'), evidence,
               'image developer independent launch evidence')
    prior.require(IMAGE_DEVELOPER_RUN.is_dir()
                  and not IMAGE_DEVELOPER_RUN.is_symlink()
                  and {p.name for p in IMAGE_DEVELOPER_RUN.iterdir()}
                  == set(IMAGE_DEVELOPER_FILE_PINS),
                  'image developer file inventory changed or checkpoint exists')
    for name, (sha, size) in IMAGE_DEVELOPER_FILE_PINS.items():
        path = IMAGE_DEVELOPER_RUN / name
        prior.exact(path.stat().st_size, size, 'image developer file size: ' + name)
        pins.add(path, sha)
    log_path = Path(str(IMAGE_DEVELOPER_RUN) + '.log')
    prior.exact(log_path.stat().st_size, IMAGE_DEVELOPER_LOG_PIN[1],
                'image developer log size')
    pins.add(log_path, IMAGE_DEVELOPER_LOG_PIN[0])
    config = prior.read_json(IMAGE_DEVELOPER_RUN / 'config.json')
    environment = prior.read_json(IMAGE_DEVELOPER_RUN / 'environment.json')
    actual = prior.read_json(IMAGE_DEVELOPER_RUN / 'actual_exposure_epoch1.json')
    arrays = prior.load_npz(IMAGE_DEVELOPER_RUN / 'actual_exposure_epoch1.npz')
    metrics = prior.read_json(IMAGE_DEVELOPER_RUN / 'metrics.json')
    rows = [prior.strict_json(line) for line in
            (IMAGE_DEVELOPER_RUN / 'train.jsonl').read_text().splitlines()]
    baseline = prior.read_json(prior.A800_DEVELOPER_RUN / 'environment.json')
    validation = validate_image_developer_records(
        config, environment, actual, arrays, metrics, rows,
        log_path.read_text(), source_hashes, baseline['wla'])
    return dict(evidence=evidence, independently_validated=validation)
def audit_training(run, profile):
    """Check a completed formal run; never infer success from a checkpoint alone."""
    profile.checked()
    prior.require(profile == frozen_profile(),
                  'only the frozen image-only training profile may be audited')
    run = Path(run)
    metrics = prior.terminal_metrics(run)  # Before any checkpoint hash/load.
    post_path = run / 'worker_postcheck.json'
    prior.require(post_path.is_file() and not post_path.is_symlink(),
                  'image terminal worker postcheck missing/nonregular')
    prior.require(run.is_absolute() and run.resolve() == run,
                  'absolute nonsymlink image run required')
    prior.require(prior.git(profile.source, 'rev-parse', 'HEAD') == profile.source_commit
                  and not prior.git(profile.source, 'status', '--porcelain'),
                  'frozen image training source changed/dirty')
    pins = prior.Pins()
    for name in prior.TERMINAL:
        if name != 'checkpoint.pt':
            pins.add(run / name)
    pins.add(post_path)
    pinned_metrics = prior.validate_metrics(prior.read_json(run / 'metrics.json'))
    prior.same(metrics, pinned_metrics,
               'image final metrics changed between completion gate and pinning')
    metrics = pinned_metrics
    pins.add(profile.config, profile.config_sha256)
    pins.add(Path(__file__).resolve())
    pins.add(Path(prior.__file__).resolve())
    launch_path = Path(str(run) + '.launch.json')
    pins.add(launch_path)
    launch = prior.read_json(launch_path)
    prior.exact(launch.get('output'), str(run), 'image launch output identity')
    config = prior.read_json(run / 'config.json')
    environment = prior.read_json(run / 'environment.json')
    yaml_config = yaml.safe_load(profile.config.read_text())
    recipe = validate_image_config(config, yaml_config, run, profile)
    source_hashes = prior.inventory(profile.source)
    prior.require('wa/wm/train_input_mode.py' in source_hashes,
                  'frozen image mode selector missing')
    for relative, sha in source_hashes.items():
        pins.add(profile.source / relative, sha)
    for name in ('wa.wm.failure_state_sampling',
                 'wa.wm.failure_state_sampling_runtime',
                 'wa.wm.teacher_window_plan', 'wa.wm.loaders'):
        module_path = Path(importlib.import_module(name).__file__).resolve()
        pins.add(module_path, source_hashes[name.replace('.', '/') + '.py'])
    hardware = prior.validate_environment(
        environment, source_hashes, launch, profile.source_commit,
        hardware_profile='a800')
    plan, admission, old_arrays, simulation = prior.load_plan(prior.PLAN_ROOT, pins)
    for name, sha in admission['provenance']['code_sha256'].items():
        prior.exact(source_hashes.get(name.replace('.', '/') + '.py'), sha,
                    'admitted image runtime/frozen code binding: ' + name)
    inputs = prior.bind_inputs(pins, config, launch, admission)
    fit = prior.bind_fit(pins, launch)
    developer = prior.bind_developer_training(pins, launch, 'a800')
    image_developer = bind_image_developer_training(
        pins, launch, source_hashes, profile.source_commit)
    dependencies = prior.bind_dependencies(pins, config, environment, launch,
                                           hardware_profile='a800')
    for path, sha in pins.files.items():
        if path in launch['input_sha256']:
            prior.exact(launch['input_sha256'][path], sha,
                        'image launch/input byte mismatch: ' + path)
    failure = prior.read_json(run / 'failure_state_exposure.json')
    for key, value in dict(experiment='evaluation_adaptation_failure_state_mix_v1',
                           plan_sha256=prior.PLAN_SHA,
                           loader_audit_sha256=prior.LOADER_SHA,
                           total_planned_positions=1233424,
                           loader_positions=1233424, diagnostic=False,
                           initialization='59866 model AND optimizer; one new epoch, cumulative two').items():
        prior.exact(failure.get(key), value,
                    'image runtime three-source provenance: ' + key)
    prior.same(failure.get('provenance'), admission['provenance'],
               'image runtime/admitted provenance differs')
    prior.same(failure.get('simulation'), simulation,
               'image runtime/admitted simulation differs')
    actual = prior.read_json(run / 'actual_exposure_epoch1.json')
    arrays = prior.load_npz(run / 'actual_exposure_epoch1.npz')
    exposure = prior.validate_exposure(plan, actual, arrays, old_arrays, simulation)
    for name, count in dict(base=726631, teacher=457641,
                            recovery=49152, total=1233424).items():
        prior.exact(actual.get('actual_' + name), count,
                    'formal image actual source exposure: ' + name)
    mode_exposure = validate_image_exposure(config, actual)
    dual = prior.read_json(run / 'dual_teacher_exposure.json')
    old_plan = prior.read_json(prior.OLD_PLAN / 'plan.json')
    from wa.wm.teacher_window_plan import canonical_sha
    prior.exact(dual.get('hard_stt_plan_sha256'), canonical_sha(old_plan),
                'image old pre-DDP canonical plan')
    prior.exact(dual.get('experiment'), 'evaluation_set_adaptation_v1',
                'image old teacher experiment')
    prior.exact(dual.get('cache_sha256'),
                pins.files[str(prior.TEACHER / 'complete.json')],
                'image old teacher cache pin')
    for key, value in dict(base_windows=726631, teacher_unique_windows=436816,
                           teacher_repeats=1, teacher_exposures=457642,
                           total_windows=1184273).items():
        prior.exact(dual.get(key), value, 'image old pre-DDP declaration: ' + key)
    prior.require(math.ceil(exposure['ranks'][0]['batches'] / 2) == prior.UPDATES
                  and all(x['loader_positions'] == 154178 and x['batches'] == 77089
                          for x in exposure['ranks']),
                  'formal image eight-rank/update budget')
    rows = [prior.strict_json(line)
            for line in (run / 'train.jsonl').read_text().splitlines()]
    logs = prior.validate_logs(rows, config['base_lrs'])
    # Only now may the auditor hash/load large weights.
    pins.add(prior.PARENT_PATH, prior.PARENT_SHA)
    prior.exact(launch['input_sha256'].get(prior.PARENT_PATH), prior.PARENT_SHA,
                'image worker parent model AND optimizer SHA')
    checkpoint_sha = pins.add(run / 'checkpoint.pt')
    parent = torch.load(prior.PARENT_PATH, map_location='cpu',
                        weights_only=True, mmap=True)
    checkpoint = torch.load(run / 'checkpoint.pt', map_location='cpu',
                            weights_only=True, mmap=True)
    mode_exposure = validate_image_exposure(config, actual, checkpoint)
    continuation = prior.validate_parent_continuation(parent, checkpoint, config)
    del parent, checkpoint
    postcheck = validate_image_postcheck(
        prior.read_json(post_path), profile, pins, run, checkpoint_sha)
    pins.finish()
    prior.require(prior.git(profile.source, 'rev-parse', 'HEAD') == profile.source_commit
                  and not prior.git(profile.source, 'status', '--porcelain')
                  and prior.inventory(profile.source) == source_hashes,
                  'image source changed during terminal audit')
    return dict(schema='failure_state_image_only_training_audit_v1',
                status='TRAINING_ARTIFACT_AUDIT_PASS_OFFLINE_ONLY_IMAGE_INPUT',
                run=str(run), checkpoint=str(run / 'checkpoint.pt'),
                checkpoint_sha256=checkpoint_sha, final_step=prior.FINAL,
                completed_epochs=2, parent_checkpoint=prior.PARENT_PATH,
                parent_sha256=prior.PARENT_SHA, train_input_mode='image',
                mode_exposure=mode_exposure, metrics=metrics['metrics'],
                source_commit=profile.source_commit, source_hashes=pins.files,
                config_path=str(profile.config),
                config_sha256=profile.config_sha256, recipe=recipe,
                hardware=hardware, plan_admission_sha256=prior.PLAN_ADMISSION_SHA,
                plan_sha256=prior.PLAN_SHA, exposure=exposure,
                input_admission=inputs, fixed_group_fit=fit,
                dependencies=dependencies, developer_training_evidence=developer,
                image_only_developer_evidence=image_developer,
                optimizer_continuation=continuation, logs=logs,
                worker_postcheck=postcheck, closed_loop=False,
                limits=['No closed-loop SR or untouched-test generalization claim.',
                        'Prior A800 four-update developer evidence used sampled modes;',
                        'New image-only four-update evidence is one A800, not eight-rank proof.',
                        'image-only branch requires separate terminal and rollout checks.',
                        'Rank assignments reconstructed; merged actual counters and '
                        'mode exposure are worker recorded, not independent rank telemetry.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile-status', action='store_true')
    parser.add_argument('--run')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.profile_status:
        print(json.dumps(dict(status='NOT_RUN_ON_REAL_OUTPUT',
                              frozen=all(x is not None for x in
                                         (FROZEN_SOURCE_COMMIT, FROZEN_CONFIG_SHA256,
                                          FROZEN_TASK_NAME))),
                         sort_keys=True))
        return
    profile = frozen_profile()
    prior.require(type(args.run) is str and type(args.output) is str,
                  'formal image audit requires run and output')
    output = Path(args.output)
    run = Path(args.run)
    protected = (run.resolve(), profile.source.resolve(), prior.PLAN_ROOT.resolve())
    prior.require(output.is_absolute() and output.resolve() == output
                  and not output.exists() and not output.is_symlink()
                  and prior.ROOT / 'artifacts' in output.parents
                  and all(output != p and p not in output.parents for p in protected)
                  and output.parent.is_dir(),
                  'fresh independent NAS image audit output required')
    report = audit_training(run, profile)
    with output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True,
                                allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], output=str(output),
                          sha256=prior.digest(output),
                          final_step=report['final_step'], closed_loop=False),
                     sort_keys=True))


if __name__ == '__main__':
    main()
