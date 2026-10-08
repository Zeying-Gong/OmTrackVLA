"""Independent terminal audit for the fixed three-source failure-state recipe.

Does not train, run a policy, modify source/cache, or certify closed-loop SR.
Metrics are checked BEFORE any checkpoint hash/load. Per-rank assignment is a
deterministic reconstruction, not separately measured rank-local telemetry.
"""
import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
from collections import Counter
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from wa.wm.failure_state_sampling import build_plan, simulate_exposure, _array_sha
from wa.wm.failure_state_sampling_runtime import ThreeSourceMix, RuntimeExposure
from wa.wm.teacher_window_plan import PlannedTeacherMix

ROOT = Path('/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928')
JOBS = Path('/data/nas_ray/project/md-ak/users/zeying.gong')
SOURCE_COMMIT = '199385cd9c826c8f21308ad99b6a8375396d9c90'
START, FINAL, UPDATES, HELDOUT = 22707, 61252, 38545, 73368
CONTRACT = 'evt_normalized_command3_actual_dt_v1'
PARENT_PATH = str(JOBS/'job_59866/task_70728/wa_history_repeat_v1/checkpoint.pt')
PARENT_SHA = 'ab39144b0490937ce43c4ab28e2d90cdb0c560f700d3364a29ce1dc9a08b999d'
PLAN_ROOT = ROOT/'artifacts/failure_state_sampling_candidate_20261008_v1'
PLAN_ADMISSION_SHA = '230079f1e99836dc7b3bf20942859e127e0b42dfa0b64c76b2c6ab67dd373902'
PLAN_SHA = '11cb7150e33c8b66cbd3353a7f95b903c8b8b282ebe84faee4aa91e4b0beeaee'
OLD_PLAN = ROOT/'artifacts/hard_stt_candidate_61377_20261007_v1'
OLD_RUN = JOBS/'job_61609/task_72803/wa_hard_stt_train_a800_v1'
OLD_AUDIT = ROOT/'artifacts/hard_stt_training_audit_61609_v1.json'
OLD_AUDIT_SHA = '59d5b468eec4283db1a1d87ad7b9a9d1bc2123739cffc38cebfc9a1e14dcc727'
OLD_REPORT_SHA = '663c66b16f1784e1a5a9962501fc4b66c32c32792dc0483be95fad2c16680c66'
RECOVERY = ROOT/'artifacts/failure_state_se2_cache_20261008_v2'
TEACHER = ROOT/'artifacts/dual_teacher_se2_cache_20261006_v1'
BASE = Path('/data/nas_ray/home/zeying.gong/datasets/wla_evt_se2_cache_20260925_v2')
INDEX = ROOT/'artifacts/robot_transition_audit_v2'
DEDUP = ROOT/'checkout/wa/results/FAILURE_STATE_DEDUP_20261008.json'
LOADER_AUDIT = ROOT/'checkout/wa/results/FAILURE_STATE_CACHE_LOADER_20261008.json'
LOADER_SHA = 'a330f7d39ecd89f7f1f67e33f0a2d6cdb67bf62f7dd57758ea270ec6a0644755'
SOURCE_PINS = dict(old_plan='c5533396f8a454bf8dd7281ff73d2b2d2737598b7a2f7e08b97260897e296d0c',
    old_actual_exposure='65bde6ac0f43835e09df12df413e935ba7f9538612c5c7738eb51631e7f8ba62',
    recovery_admission='c690957761133f98e8925bf0f491373c9b0ca30d74f427834d9d4a155093bc69',
    dedup_report='599d60ea2d40ff4f9ae69f87df03e3a5a192b749ce030504bf19177999262859')
SELECTION = ROOT/'artifacts/failure_state_group_fit_selection_20261008_v1.json'
SELECTION_SHA = '7e7db844c09f67d7b0e3b5de7376c0573c18fccc485dfa235b879ed4dab111a9'
FIT = ROOT/'artifacts/failure_state_group_fit_pretrain_20261008_v1.json'
FIT_SHA = '6ecff4bda1c19d153e7e4be0742914d2b761698759a9157c17ec2920a6725cd9'
BEST_SHA = 'c510c04d987d141423401a25323ea353314107a16f4dac5d2901370ab199fa52'
TERMINAL = ('metrics.json', 'checkpoint.pt', 'config.json', 'environment.json',
    'actual_exposure_epoch1.json', 'actual_exposure_epoch1.npz', 'train.jsonl',
    'failure_state_exposure.json', 'dual_teacher_exposure.json')


class IncompleteTraining(ValueError):
    """A final checkpoint alone is not completion of heldout validation."""


def require(ok, message):
    if not ok:
        raise ValueError(message)


def exact(actual, expected, message):
    require(type(actual) is type(expected) and actual == expected, message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def same(actual, expected, message):
    require(canonical(actual) == canonical(expected), message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def sha_string(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'explicit lowercase SHA256 required')
    return value


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def strict_json(blob):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(blob, object_pairs_hook=pairs, parse_constant=nonfinite)


def read_json(path):
    return strict_json(Path(path).read_text())


def validate_metrics(metrics):
    for key, value in dict(status='OFFLINE_ONLY', kind='jepa', steps=FINAL,
            closed_loop=False, edge_latency_verified=False, point_source='simulated_uwb').items():
        exact(metrics.get(key), value, 'final metrics: ' + key)
    modes = metrics.get('metrics')
    require(type(modes) is dict and set(modes) == {'image', 'point', 'mixed'}, 'three complete validation modes required')
    for mode, row in modes.items():
        exact(row.get('windows'), HELDOUT, mode + ' full heldout denominator')
        require(all(finite(row.get(k)) for k in ('ADE_m', 'FDE_m', 'yaw_MAE_rad')), 'nonfinite offline metric')
        require('SR' in row and 'collision_rate' in row and row['SR'] is None
            and row['collision_rate'] is None, 'offline metrics must contain null SR/CR')
    require(finite(metrics.get('elapsed_s')), 'invalid validation elapsed time')
    return metrics


def terminal_metrics(run):
    run = Path(run)
    try:
        metrics = read_json(run/'metrics.json')
    except (OSError, ValueError) as error:
        raise IncompleteTraining('final metrics not completely readable; checkpoint not inspected') from error
    validate_metrics(metrics)
    missing = [name for name in TERMINAL if not (run/name).is_file() or (run/name).is_symlink()]
    if missing:
        raise IncompleteTraining('terminal files missing/nonregular: ' + ', '.join(missing))
    return metrics


def lr_multiplier(update):
    return .1 + .9*.5*(1 + math.cos(math.pi*(update-1)/UPDATES))


def validate_logs(rows, base_lrs):
    require(type(base_lrs) is list and len(base_lrs) == 5 and all(finite(x) and x > 0 for x in base_lrs), 'five resumed LR groups required')
    expected = [step for step in range(START+1, FINAL+1) if step == START+1 or step % 25 == 0]
    require(type(rows) is list and [x.get('step') for x in rows] == expected, 'exact scheduled training log sequence required')
    previous_time = -1.
    for row, step in zip(rows, expected):
        for key, value in dict(step=step, total_steps=FINAL, phase_step=step-START,
                epoch=1, loss_aggregation='DDP_mean_over_accumulation_group').items():
            exact(row.get(key), value, 'train log: ' + key)
        require(all(finite(row.get(k)) for k in ('loss', 'flow', 'geometry', 'world',
            'grad_norm', 'elapsed_s', 'peak_allocated_gib', 'loss_rank0_last_microbatch')), 'nonfinite train log')
        require(row['elapsed_s'] >= previous_time, 'training elapsed time regressed')
        previous_time = row['elapsed_s']
        require(math.isclose(row['loss'], row['flow']+.5*row['geometry']+.1*row['world'],
                            rel_tol=2e-5, abs_tol=2e-6), 'original loss composition changed')
        rates = row.get('lr')
        require(type(rates) is list and len(rates) == 5 and all(finite(x) and x > 0 for x in rates), 'invalid logged LR groups')
        require(all(math.isclose(got, base*lr_multiplier(step-START), rel_tol=1e-12, abs_tol=0.)
                    for got, base in zip(rates, base_lrs)), 'resumed cosine LR differs')
    return dict(log_records=len(rows), first_step=START+1, last_logged_step=expected[-1],
        final_step_logged=False, final_step_authority='final checkpoint AND complete three-mode metrics')


FLAGS = frozenset(('epochs', 'batch-size', 'accumulation', 'workers', 'seed', 'world-weight',
    'history-repeat-probability', 'completed-epochs', 'resume', 'resume-sha256', 'dual-teacher-cache',
    'dual-teacher-repeats', 'evaluation-set-adaptation', 'teacher-window-plan', 'teacher-plan-report-sha256',
    'failure-state-cache', 'failure-state-admission-sha256', 'failure-state-plan',
    'failure-state-plan-admission-sha256', 'failure-state-dedup-report', 'failure-state-dedup-sha256',
    'failure-state-old-run', 'failure-state-old-plan-sha256', 'failure-state-old-exposure-sha256',
    'failure-state-loader-audit', 'failure-state-loader-audit-sha256'))
VARIABLE = re.compile(r'\$(?:\{([A-Za-z_][A-Za-z_0-9]*)\}|([A-Za-z_][A-Za-z_0-9]*))')


def expand(value, expected_env):
    require(not any(x in value for x in ('`', '$(', ';', '|', '&', '<', '>')), 'shell expression forbidden in training argument')
    def replace(match):
        name = match.group(1) or match.group(2)
        require(name in expected_env and type(expected_env[name]) is str, 'unknown shell variable: ' + name)
        return expected_env[name]
    result = VARIABLE.sub(replace, value)
    require('$' not in result and '\n' not in result, 'unresolved shell expansion')
    return result


def parse_training_command(cmd, expected_env):
    require(type(cmd) is str and type(expected_env) is dict, 'command/environment types')
    folded = re.sub(r'\\\r?\n[ \t]*', ' ', cmd)
    lines = [line.strip() for line in folded.splitlines() if re.match(r'^\s*bash\s+wa/scripts/train_world\.sh(?:\s|$)', line)]
    require(len(lines) == 1, 'exactly one formal train command required')
    tokens = shlex.split(lines[0], comments=True)
    require(tokens[:2] == ['bash', 'wa/scripts/train_world.sh'], 'wrong training entrypoint')
    arguments, i = {}, 2
    while i < len(tokens):
        key = tokens[i]
        require(key.startswith('--') and key[2:] in FLAGS and key not in arguments, 'duplicate/unknown training option: ' + key)
        if key == '--evaluation-set-adaptation':
            arguments[key] = True
            i += 1
        else:
            require(i+1 < len(tokens) and not tokens[i+1].startswith('--'), 'missing training argument')
            arguments[key] = expand(tokens[i+1], expected_env)
            i += 2
    return arguments


def expected_recipe(run, plan_root):
    return dict(kind='jepa', batch_size=2, accumulation=2, workers=2, epochs=1, seed=42,
        world_weight=.1, diagnostic=False, lane='managed', completed_epochs=1,
        history_repeat_probability=.25, resume=PARENT_PATH, resume_sha256=PARENT_SHA,
        dual_teacher_repeats=1, evaluation_set_adaptation=True, contract=CONTRACT,
        effective_batch=32, train_rows=1233424, heldout_rows=HELDOUT, world_size=8,
        teacher_plan_report_sha256=OLD_REPORT_SHA, root=str(ROOT), output=str(run),
        cache=str(BASE), index_root=str(INDEX), dual_teacher_cache=str(TEACHER),
        teacher_window_plan=str(OLD_PLAN), failure_state_cache=str(RECOVERY),
        failure_state_admission_sha256=SOURCE_PINS['recovery_admission'],
        failure_state_plan=str(plan_root), failure_state_plan_admission_sha256=PLAN_ADMISSION_SHA,
        failure_state_dedup_report=str(DEDUP), failure_state_dedup_sha256=SOURCE_PINS['dedup_report'],
        failure_state_old_run=str(OLD_RUN), failure_state_old_plan_sha256=SOURCE_PINS['old_plan'],
        failure_state_old_exposure_sha256=SOURCE_PINS['old_actual_exposure'],
        failure_state_loader_audit=str(LOADER_AUDIT), failure_state_loader_audit_sha256=LOADER_SHA,
        encoder_weight='/data/nas_ray/home/zeying.gong/datasets_processed/threepanel_debug_v2/traversability_stepp_v1/_weights/dinov2_vits14_pretrain.pth',
        wla_source=str(ROOT/'dependencies/wla_v1'),
        wla_checkpoint=str(JOBS/'job_58346/task_69086/wla_teacher_train/training/checkpoints/step-0043203.pt'),
        precision='fp32 training / bf16 frozen encoder')


def validate_config(config, yaml_config, source, run, plan_root, source_commit):
    exact(source_commit, SOURCE_COMMIT, 'unsupported frozen training source')
    recipe = expected_recipe(Path(run), Path(plan_root))
    for key, value in recipe.items():
        exact(config.get(key), value, 'runtime config: ' + key)
    for key in ('recovery_cache', 'recovery_index', 'data_root', 'source_prefix'):
        require(config.get(key) is None, 'foreign data route: ' + key)
    require(type(config.get('base_lrs')) is list and len(config['base_lrs']) == 5
        and all(finite(x) and x > 0 for x in config['base_lrs']), 'five resumed LR groups required')
    for key, value in dict(cluster='baidu_4090', num_gpus=8,
            image='x5-builder:cuda12.8-isaac5.0.0-v2.test1').items():
        exact(yaml_config.get(key), value, 'YAML: ' + key)
    tasks = yaml_config.get('tasks')
    require(type(tasks) is list and len(tasks) == 1, 'single full training task required')
    task = tasks[0]
    for key, value in dict(num_gpus=8, type='shell', workload_backend='k8s', timeout=86400).items():
        exact(task.get(key), value, 'YAML task: ' + key)
    cmd = task['cmd']
    require('cd '+str(source) in cmd and source_commit in cmd and 'git status --porcelain' in cmd,
            'frozen source/clean guard absent')
    match = re.fullmatch(re.escape(str(JOBS))+r'/job_([0-9]+)/task_([0-9]+)/wa_failure_state_train_4090_v1', str(run))
    require(match is not None, 'exact scheduler Job/Task output route required')
    env = dict(MD_AK_JOB_ID=match.group(1), MD_AK_TASK_ID=match.group(2))
    for line in cmd.splitlines():
        if not line.strip().startswith('export '):
            continue
        for token in shlex.split(line.strip())[1:]:
            require('=' in token, 'plain export assignment required')
            name, value = token.split('=', 1)
            require(name not in env, 'duplicate export: ' + name)
            env[name] = expand(value, env)
    wanted = dict(WA_ROOT=str(ROOT), WA_WLA_SOURCE=recipe['wla_source'], WA_WLA_CHECKPOINT=recipe['wla_checkpoint'],
        WA_ENCODER_WEIGHTS=recipe['encoder_weight'], WA_PARENT_CHECKPOINT=PARENT_PATH, WA_CACHE=str(BASE),
        WA_INDEX_ROOT=str(INDEX), WA_PYTHON=str(ROOT/'probe_env/bin/python'), WA_WORLD_KIND='jepa', WA_GPUS='8',
        WA_OUTPUT=str(run), OMP_NUM_THREADS='2', NCCL_DEBUG='WARN', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    same({k:v for k,v in env.items() if not k.startswith('MD_AK_')}, wanted, 'worker environment differs')
    args = parse_training_command(cmd, env)
    expected = {'--'+flag: (True if flag == 'evaluation-set-adaptation' else str(recipe[flag.replace('-', '_')])) for flag in FLAGS}
    same(args, expected, 'YAML/runtime training flags differ')
    return dict(arguments=args, environment=env)


def validate_environment(env, source_hashes, launch, source_commit):
    for key, value in dict(commit=source_commit, dirty='', torch='2.8.0+cu128').items():
        exact(env.get(key), value, 'worker environment: ' + key)
    names = env.get('gpu_names')
    require(type(names) is list and len(names) == 8 and all(type(x) is str and 'RTX 4090' in x for x in names), 'eight actual RTX4090 GPUs required')
    require(bool(source_hashes) and env.get('source_sha256') == source_hashes, 'complete frozen Python inventory mismatch')
    wla = env.get('wla', {})
    require(wla.get('strict_loaded') == ['action_expert','metaquery','target_head']
        and wla.get('omitted') == ['Qwen backbone','language LoRA']
        and wla.get('action_contract') == '7x4 XY/sin(yaw)/cos(yaw)'
        and type(wla.get('source_files')) is dict and bool(wla['source_files']), 'WLA initialization/source provenance')
    for key, value in dict(status='WORKER_PREFLIGHT_PASSED_NOT_TRAINING_COMPLETE',
            source_commit=source_commit, torch='2.8.0+cu128').items():
        exact(launch.get(key), value, 'launch: ' + key)
    cuda, physical = launch.get('cuda'), launch.get('uuid_model_memory_driver')
    require(type(cuda) is list and len(cuda) == 8 and type(physical) is list and len(physical) == 8, 'eight launch CUDA/physical GPU records required')
    uuids = []
    for i, (logical, row) in enumerate(zip(cuda, physical)):
        exact(logical.get('index'), i, 'CUDA lane index')
        same(logical.get('capability'), [8, 9], 'actual sm89 required')
        exact(logical.get('name'), names[i], 'environment/launch GPU name')
        require(type(row) is list and len(row) == 4 and all(type(x) is str for x in row), 'nvidia-smi GPU row')
        require(re.fullmatch(r'GPU-[0-9a-fA-F-]+', row[0]) and 'RTX 4090' in row[1]
                and row[2].isdigit() and int(row[2]) >= 24000 and bool(row[3]), 'GPU UUID/model/memory/driver evidence')
        uuids.append(row[0])
    require(len(set(uuids)) == 8, 'eight distinct physical GPU UUIDs required')
    return dict(gpu_names=names, unique_gpu_uuids=uuids, torch=env['torch'])


def optimizer_steps(optimizer):
    groups, state = optimizer.get('param_groups'), optimizer.get('state')
    require(type(groups) is list and len(groups) == 5 and type(state) is dict and bool(state), 'five groups and nonempty optimizer state required')
    params = [p for group in groups for p in group.get('params', [])]
    require(params and all(type(x) is int for x in params) and len(set(params)) == len(params), 'optimizer parameter mapping')
    require(set(state) <= set(params), 'foreign optimizer state')
    out = {}
    for key, item in state.items():
        require(type(item) is dict and {'step', 'exp_avg', 'exp_avg_sq'} <= set(item), 'incomplete AdamW state')
        value = item['step']
        if isinstance(value, torch.Tensor):
            require(value.device.type == 'cpu' and value.numel() == 1, 'optimizer step must be CPU scalar')
            value = float(value)
        require(finite(value) and float(value).is_integer(), 'nonfinite/noninteger optimizer step')
        for name in ('exp_avg', 'exp_avg_sq'):
            tensor = item[name]
            require(isinstance(tensor, torch.Tensor) and tensor.device.type == 'cpu'
                and bool(torch.isfinite(tensor).all()), 'nonfinite/missing AdamW moment')
        require(item['exp_avg'].shape == item['exp_avg_sq'].shape, 'AdamW moment shape mismatch')
        require(item['exp_avg'].dtype == item['exp_avg_sq'].dtype
            and bool((item['exp_avg_sq'] >= 0).all()), 'AdamW moment dtype or negative squared moment')
        out[key] = int(value)
    return out


def validate_parent_continuation(parent, checkpoint, config):
    for key, value in dict(step=START, kind='jepa', contract=CONTRACT).items():
        exact(parent.get(key), value, 'parent: ' + key)
    for key, value in dict(step=FINAL, kind='jepa', contract=CONTRACT, completed_epochs=2,
            parent_checkpoint=PARENT_PATH, parent_sha256=PARENT_SHA).items():
        exact(checkpoint.get(key), value, 'final checkpoint: ' + key)
    before, after = parent.get('model'), checkpoint.get('model')
    require(type(before) is dict and bool(before) and type(after) is dict and before.keys() == after.keys(), 'parent/final model identities changed')
    require(not any(k.startswith('encoder.') for k in after), 'unexpected mutable frozen encoder')
    changed_model_tensors = 0
    for key in before:
        a, b = before[key], after[key]
        require(isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.shape == b.shape
            and a.dtype == b.dtype and a.device.type == b.device.type == 'cpu'
            and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), 'model shape/dtype/finiteness changed')
        changed_model_tensors += int(not torch.equal(a, b))
    require(changed_model_tensors > 0, 'all nonencoder model tensors unchanged from parent')
    old, new = parent.get('optimizer', {}), checkpoint.get('optimizer', {})
    old_steps, new_steps = optimizer_steps(old), optimizer_steps(new)
    require(old_steps.keys() <= new_steps.keys(), 'parent optimizer state lost')
    for key in old_steps:
        for name in ('exp_avg', 'exp_avg_sq'):
            a, b = old['state'][key][name], new['state'][key][name]
            require(a.shape == b.shape and a.dtype == b.dtype,
                    'parent/final optimizer moment shape/dtype changed')
    require(max(old_steps.values()) == START and max(new_steps.values()) == FINAL, 'full optimizer continuation missing')
    require(all(0 <= value-old_steps.get(key, 0) <= UPDATES for key, value in new_steps.items()), 'optimizer reset or extra updates')
    # Frozen forward always evaluates flow, geometry and world losses, and each
    # synchronization performs one shared optimizer.step(). A state that reached
    # every parent update cannot silently stop/skip here while another state
    # supplies the global maximum. Do not extrapolate this rule to historically
    # conditional states whose parent counter is already below START.
    full_states = [key for key, value in old_steps.items() if value == START]
    require(all(new_steps[key] == FINAL for key in full_states),
            'previously fully active optimizer state lacks every continuation update')
    rates = config.get('base_lrs')
    require(type(rates) is list and len(rates) == 5 and all(finite(x) and x > 0 for x in rates), 'base LR groups')
    require([g['lr'] for g in old['param_groups']] == rates, 'base LR not resumed from parent optimizer')
    for i, (a, b) in enumerate(zip(old['param_groups'], new['param_groups'])):
        same({k:v for k,v in a.items() if k != 'lr'}, {k:v for k,v in b.items() if k != 'lr'}, 'optimizer group/hyperparameters changed')
        require(finite(b.get('lr')) and math.isclose(b['lr'], rates[i]*lr_multiplier(UPDATES), rel_tol=1e-12, abs_tol=0.), 'final LR schedule changed')
    return dict(parent_step=START, final_step=FINAL, updates=UPDATES, resumed_base_lrs=rates,
        changed_model_tensors=changed_model_tensors, total_model_tensors=len(after),
        planned_full_continuation_state_count=len(full_states),
        conditional_parent_state_count=len(old_steps)-len(full_states),
        new_optimizer_state_count=len(new_steps.keys()-old_steps.keys()),
        conditional_state_scope='Parent counters below START and newly created states have only bounded nonnegative deltas; no claim that each was active on every update')


def validate_exposure(plan, actual, arrays, old_arrays, simulation=None):
    require(type(arrays) is dict and set(arrays) == {'position_counts','base_counts','teacher_counts','recovery_counts'}, 'exact four actual exposure arrays required')
    require(type(old_arrays) is dict and set(old_arrays) == {'position_counts','base_counts','teacher_counts'}, 'old actual array inventory')
    metadata, pa = plan['metadata'], plan['arrays']
    windows = metadata['recovery_windows']
    recovery_len = max(row['dataset_index'] for row in windows)+1
    dimensions = dict(position_counts=metadata['total_positions'], base_counts=len(pa['old_base_counts']),
        teacher_counts=len(pa['old_teacher_counts']), recovery_counts=recovery_len)
    for name, length in dimensions.items():
        value = arrays[name]
        require(isinstance(value, np.ndarray) and value.dtype == np.int64 and value.shape == (length,)
            and bool((value >= 0).all()), 'invalid actual exposure dtype/shape: ' + name)
    for name, length in dict(position_counts=metadata['old_pre_ddp_positions'],
            base_counts=dimensions['base_counts'], teacher_counts=dimensions['teacher_counts']).items():
        value = old_arrays[name]
        require(isinstance(value, np.ndarray) and value.dtype == np.int64 and value.shape == (length,)
            and bool((value >= 0).all()), 'invalid old actual exposure: ' + name)
    expected_old = np.ones(metadata['old_pre_ddp_positions'], dtype=np.int64)
    expected_old[metadata['old_dropped_positions']] = 0
    require(np.array_equal(old_arrays['position_counts'], expected_old), 'old omitted positions changed')
    for name in ('base', 'teacher'):
        require(np.array_equal(old_arrays[name+'_counts'], pa['old_'+name+'_counts']), 'old per-window exposure changed: ' + name)
    # Reuse the frozen counter on synthetic index-only datasets, never load images.
    old_mix = SimpleNamespace(base=range(dimensions['base_counts']), teacher=range(dimensions['teacher_counts']))
    mix = ThreeSourceMix(plan, old_mix, range(recovery_len))
    counter = RuntimeExposure(mix)
    state = dict(position_counts=torch.from_numpy(arrays['position_counts'].copy()))
    calculated = counter.finalize(state=state, world=8, batch=2, seed=42, epoch=1)
    for name, counts in zip(('base_counts','teacher_counts','recovery_counts'), counter.source_counts(state)):
        require(np.array_equal(arrays[name], counts.numpy()), 'actual NPZ per-window counts differ: ' + name)
    for key, value in calculated.items():
        same(actual.get(key), value, 'actual exposure JSON differs: ' + key)
    exact(actual.get('optimizer_steps_completed'), UPDATES, 'optimizer update count')
    exact(actual.get('diagnostic'), False, 'diagnostic exposure cannot pass formal audit')
    rebuilt = simulate_exposure(plan, seed=42, epoch=1, world_size=8, batch_size=2)
    if simulation is not None:
        same(simulation, rebuilt, 'saved eight-rank simulation differs')
    require(rebuilt['actual_total'] == int(arrays['position_counts'].sum()), 'reconstructed ranks total differs')
    return dict(calculated=calculated, ranks=rebuilt['ranks'], simulation=rebuilt,
        rank_evidence='Deterministic eight-rank sampler/DataLoader reconstruction; no independently recorded per-rank actual counters')


class Pins:
    """Hash once, then verify file identity and digest again before releasing PASS."""
    def __init__(self):
        self.files, self.stats = {}, {}

    def add(self, value, expected=None, *, allow_symlink=False):
        path = Path(value)
        require(path.is_absolute() and path.is_file() and (allow_symlink or not path.is_symlink()), 'absolute existing regular input required: ' + str(path))
        require(path.parent.resolve() == path.parent and (allow_symlink or path.resolve() == path),
                'intermediate/undeclared symlink input: ' + str(path))
        before = path.stat()
        value = digest(path)
        after = path.stat()
        stamp = lambda x: (x.st_ino, x.st_size, x.st_mtime_ns)
        require(stamp(before) == stamp(after), 'file changed while hashing: ' + str(path))
        if expected is not None:
            require(value == sha_string(expected), 'input SHA mismatch: ' + str(path))
        require(str(path) not in self.files or self.files[str(path)] == value, 'input changed between reads')
        self.files[str(path)], self.stats[str(path)] = value, stamp(after)
        return value

    def finish(self):
        for name, value in self.files.items():
            stat = Path(name).stat()
            require((stat.st_ino, stat.st_size, stat.st_mtime_ns) == self.stats[name]
                and digest(name) == value, 'input changed during audit: ' + name)


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key:archive[key] for key in archive.files}


def inventory(root):
    return {str(path.relative_to(root)):digest(path) for path in sorted((root/'wa').rglob('*.py'))}


def git(root, *arguments):
    return subprocess.check_output(['git', '-C', str(root), *arguments], text=True).strip()


def load_plan(plan_root, pins):
    pins.add(plan_root/'admission.json', PLAN_ADMISSION_SHA)
    admission = read_json(plan_root/'admission.json')
    exact(admission.get('schema'), 'failure_state_sampling_runtime_candidate_v1', 'plan admission schema')
    exact(admission.get('status'), 'CANDIDATE_ONLY_NOT_TRAINING_RELEASE', 'plan admission status')
    exact(admission.get('training_released'), False, 'immutable candidate flag changed')
    same(admission.get('source_pins'), SOURCE_PINS, 'four admitted input pins')
    exact(admission.get('plan_sha256'), PLAN_SHA, 'canonical plan identity')
    require(set(admission['files']) == {'plan.json','positions.npz','exposure.json'}, 'plan file inventory')
    for name, sha in admission['files'].items():
        pins.add(plan_root/name, sha)
    plan = read_json(plan_root/'plan.json')
    plan['arrays'] = load_npz(plan_root/'positions.npz')
    pins.add(OLD_PLAN/'plan.json', SOURCE_PINS['old_plan'])
    pins.add(OLD_PLAN/'report.json', OLD_REPORT_SHA)
    pins.add(OLD_RUN/'actual_exposure_epoch1.npz', SOURCE_PINS['old_actual_exposure'])
    pins.add(OLD_AUDIT, OLD_AUDIT_SHA)
    old_plan = read_json(OLD_PLAN/'plan.json')
    old_arrays = load_npz(OLD_RUN/'actual_exposure_epoch1.npz')
    old_mix = PlannedTeacherMix(range(726631), range(436816), old_plan,
        source_hashes=old_plan['source_hashes'], eligible_teacher_indices=old_plan['extra_teacher_indices'])
    old_positions = [old_mix.locate(i) for i in range(len(old_mix))]
    rebuilt = build_plan(old_positions, old_arrays['position_counts'], plan['metadata']['recovery_windows'],
        source_pins=SOURCE_PINS, recovery_budget=49152, expected_dropped_positions=(1151263,))
    same({k:v for k,v in plan.items() if k != 'arrays'}, {k:v for k,v in rebuilt.items() if k != 'arrays'}, 'independently rebuilt plan metadata/hash differs')
    require(set(plan['arrays']) == set(rebuilt['arrays']), 'plan array inventory')
    for key, value in rebuilt['arrays'].items():
        require(plan['arrays'][key].dtype == value.dtype and np.array_equal(plan['arrays'][key], value), 'reconstructed sampling map differs: ' + key)
    exact(rebuilt['metadata']['total_positions'], 1233424, 'formal three-source size')
    exact(len(rebuilt['metadata']['recovery_windows']), 6864, 'formal admitted recovery rows')
    return plan, admission, old_arrays, read_json(plan_root/'exposure.json')


def bind_inputs(pins, config, launch, admission):
    # These are admission/array contracts, not a repeat of full collection/media audits.
    fixed = {BASE/'complete.json':'eb5a52d3478bab1391d12a990dcdc57ce2c77cef787109d29f7f286be797f901',
        INDEX/'audit.json':'3138364dea80543fc83be476e1af8191a4eca3711a9b353591311fc8daf5c2d3',
        TEACHER/'complete.json':'79e5a7dc43b61a23bb64d5676c4c1cac17b459a5e390b308377012e582ba1a8d',
        RECOVERY/'admission.json':SOURCE_PINS['recovery_admission'], DEDUP:SOURCE_PINS['dedup_report'],
        LOADER_AUDIT:LOADER_SHA, SELECTION:SELECTION_SHA, FIT:FIT_SHA}
    for path, sha in fixed.items():
        pins.add(path, sha)
    exact(config.get('cache_sha256'), fixed[BASE/'complete.json'], 'runtime base cache pin')
    exact(config.get('index_audit_sha256'), fixed[INDEX/'audit.json'], 'runtime base index pin')
    trusted_old = read_json(OLD_AUDIT)['source_hashes']
    required_arrays = {split+'_'+suffix for split in ('train','heldout')
        for suffix in ('pose.npy','history.npy','episode.npy','episodes.json')}
    for cache in (BASE, TEACHER):
        manifest = read_json(cache/'complete.json')
        require(required_arrays <= set(manifest['files']), 'consumed cache array manifest incomplete')
        for name in sorted(required_arrays):
            path = cache/name
            if path.is_symlink():
                require(name.startswith('heldout_') and cache == TEACHER
                    and path.resolve() == (BASE/name).resolve(), 'teacher heldout link changed')
            pins.add(path, manifest['files'][name], allow_symlink=True)
            if cache == TEACHER and name.startswith('train_'):
                exact(manifest['files'][name], trusted_old.get(str(path)), 'teacher array old audit binding')
    for name in ('audit.json','train_valid.npy','train_episodes_audit.json'):
        path = TEACHER/'index'/name
        require(str(path) in trusted_old, 'teacher index absent from trusted old audit')
        pins.add(path, trusted_old[str(path)])
    teacher_index = read_json(TEACHER/'index/audit.json')
    exact(teacher_index.get('contract'), CONTRACT, 'teacher index contract')
    exact(teacher_index.get('cache_complete_sha256'), fixed[TEACHER/'complete.json'], 'teacher index/cache binding')
    exact(teacher_index['splits']['train']['index_sha256'],
        pins.files[str(TEACHER/'index/train_valid.npy')], 'teacher valid index pin')
    teacher_valid = np.load(TEACHER/'index/train_valid.npy', allow_pickle=False)
    require(teacher_valid.shape == (436816,) and teacher_valid.dtype.kind in 'iu', 'teacher index dimensions')
    base_index = read_json(INDEX/'audit.json')
    exact(base_index.get('contract'), CONTRACT, 'base index contract')
    exact(base_index.get('cache_complete_sha256'), fixed[BASE/'complete.json'], 'base index/cache binding')
    for split, count in (('train',726631), ('heldout',HELDOUT)):
        pins.add(INDEX/(split+'_valid.npy'), base_index['splits'][split]['index_sha256'])
        values = np.load(INDEX/(split+'_valid.npy'), allow_pickle=False)
        require(values.ndim == 1 and len(values) == count and values.dtype.kind in 'iu', 'base split index count/type')
    recovery = read_json(RECOVERY/'admission.json')
    require(type(recovery.get('files')) is dict and len(recovery['files']) == 16, 'recovery admitted file graph')
    for name, sha in recovery['files'].items():
        path = RECOVERY/name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'unsafe admitted cache path')
        if path.is_symlink():
            expected = (INDEX/name.split('/')[-1]) if name.startswith('index/') else BASE/name
            require(name.startswith('heldout_') or name.startswith('index/heldout_'), 'unapproved recovery link')
            require(path.resolve() == expected.resolve(), 'original heldout link changed')
        pins.add(path, sha, allow_symlink=True)
    for suffix in ('pose.npy','history.npy','episode.npy','episodes.json'):
        value = pins.add(BASE/('heldout_'+suffix))
        pins.add(TEACHER/('heldout_'+suffix), value, allow_symlink=True)
        pins.add(RECOVERY/('heldout_'+suffix), value, allow_symlink=True)
    valid = np.load(RECOVERY/'index/train_valid.npy', allow_pickle=False)
    windows = read_json(PLAN_ROOT/'plan.json')['metadata']['recovery_windows']
    require(valid.shape == (6864,) and valid.dtype.kind in 'iu' and len(set(valid.tolist())) == len(valid), 'recovery valid index identity')
    for i, row in enumerate(windows):
        exact(row['dataset_index'], i, 'recovery loader index order')
        exact(row['raw_row'], int(valid[i]), 'recovery loader raw-row identity')
    same(admission['provenance']['source_pins'], SOURCE_PINS, 'provenance source pins')
    exact(admission['provenance'].get('valid_rows'), 6864, 'provenance valid rows')
    recorded = launch.get('input_sha256')
    require(type(recorded) is dict and all(recorded.get(str(path)) == sha for path,sha in fixed.items()), 'worker launch/admitted input bindings')
    return dict(recovery_rows=len(valid), original_heldout_rows=HELDOUT,
        scope='Rehashed consumed cache arrays and admission graph; prior immutable collection/PNG and unused historical weight audits are referenced, not rerun')


def bind_fit(pins, launch):
    selection, fit = read_json(SELECTION), read_json(FIT)
    for doc in (selection, fit):
        exact(doc.get('closed_loop'), False, 'fit must not claim closed loop')
        exact(doc.get('training_released'), False, 'fit must not release training')
    exact(selection.get('schema'), 'failure_state_group_fit_selection_v1', 'selection schema')
    exact(selection.get('status'), 'FIXED152_INPUTS_ADMITTED_NO_MODEL_RESULT', 'selection status')
    exact(fit.get('schema'), 'failure_state_group_fit_v1', 'fit schema')
    exact(fit.get('status'), 'FIXED152_TEACHER_STATE_LABEL_FIT_ONLY_NOT_SR', 'fit status')
    require(len(selection['selected']) == 152 and fit['selected'] == selection['selected'], 'fixed152 selection changed')
    exact(fit.get('selection'), str(SELECTION), 'fixed fit selection path')
    exact(fit.get('selection_sha256'), SELECTION_SHA, 'fixed fit selection SHA')
    exact(fit.get('source_hashes', {}).get(str(SELECTION)), SELECTION_SHA, 'fit source selection binding')
    same(dict(Counter(row['group'] for row in selection['selected'])),
        dict(hard_stt_collision=24, hard_stt_other=16, successful_stt_control=16,
             dt_teacher_control=16, at_teacher_control=16, recovery_early=32, recovery_late=32),
        'fixed fit group selection changed')
    evidence = launch.get('fixed_group_fit_evidence', {})
    for key, value in dict(selection_path=str(SELECTION), selection_sha256=SELECTION_SHA,
            fit_path=str(FIT), fit_sha256=FIT_SHA, schema='failure_state_group_fit_v1',
            status='FIXED152_TEACHER_STATE_LABEL_FIT_ONLY_NOT_SR', windows=152, window_predictions=608,
            policy_forward_only=True, closed_loop=False, training_released=False).items():
        exact(evidence.get(key), value, 'fixed fit launch binding: ' + key)
    expected = dict(parent59866=dict(sha256=PARENT_SHA,step=START), best61609=dict(sha256=BEST_SHA,step=59716))
    same(evidence.get('models'), expected, 'fixed fit launch model identity')
    require(set(fit['models']) == set(expected), 'fixed fit model set')
    for name, identity in expected.items():
        model = fit['models'][name]
        for key, value in identity.items():
            exact(model.get(key), value, 'fixed fit model ' + key)
        require(len(model['records']) == 304, 'fixed fit two history modes incomplete')
        require(all(type(row.get('selection_index')) is int and type(row.get('repeat_history')) is bool
                    for row in model['records']), 'typed fit record identity')
        require({(row['selection_index'], row['repeat_history']) for row in model['records']}
            == {(i, repeat) for i in range(152) for repeat in (False, True)},
            'fixed fit record identity set')
    return dict(selection_sha256=SELECTION_SHA, fit_sha256=FIT_SHA, windows=152,
        predictions=608, closed_loop=False, scope='Pinned pretraining label-fit evidence, not new model fit or SR')


def bind_dependencies(pins, config, environment, launch):
    from wa.wm import loaders
    for name in ('dinov2','jepa-wms'):
        loaders.verify_source(ROOT/'upstream_audit'/name, name)
    fixed = {Path(config['encoder_weight']):loaders.HASHES['encoder'],
        ROOT/'models/jepa_wms/mz_jepa-wm.pth.tar':loaders.HASHES['jepa'],
        Path(config['wla_checkpoint']):'0b8f036fd282474d8c9efe9e34e1f4dc56b9282c44295bcda1f16edcf48460e1'}
    for path, sha in fixed.items():
        pins.add(path, sha)
        exact(launch['input_sha256'].get(str(path)), sha, 'worker/model input binding')
    wla = environment['wla']
    baseline_path = ROOT/'artifacts/failure_state_train_developer_20261008_v1/environment.json'
    baseline_sha = '74f43531afb2a690e1162c361b54c4653472fceb894e6bc47044cedb7487ea81'
    pins.add(baseline_path, baseline_sha)
    baseline = read_json(baseline_path)
    same(wla, baseline['wla'], 'worker WLA differs from admitted real developer initialization')
    expected = {str(path):pins.files[str(path)] for path in (
        baseline_path, Path(config['encoder_weight']), ROOT/'models/jepa_wms/mz_jepa-wm.pth.tar',
        Path(config['wla_checkpoint']), BASE/'complete.json', INDEX/'audit.json', TEACHER/'complete.json',
        OLD_PLAN/'report.json', OLD_PLAN/'plan.json', OLD_RUN/'actual_exposure_epoch1.npz',
        RECOVERY/'admission.json', RECOVERY/'complete.json', PLAN_ROOT/'admission.json', DEDUP, LOADER_AUDIT,
        SELECTION, FIT)}
    expected[PARENT_PATH] = PARENT_SHA  # Parent is hashed after all non-weight terminal gates.
    same(launch.get('input_sha256'), expected, 'exact complete worker launch input inventory')
    exact(wla.get('checkpoint_sha256'), fixed[Path(config['wla_checkpoint'])], 'WLA checkpoint identity')
    exact(wla.get('checkpoint_step'), 43203, 'WLA checkpoint step')
    root = Path(config['wla_source'])
    files = {str(p.relative_to(root)):p for p in (root/'src/md_wla').rglob('*.py')}
    require(set(files) == set(wla['source_files']), 'WLA source inventory changed')
    for name, path in files.items():
        pins.add(path, wla['source_files'][name])
    exact(launch.get('wla_source_files'), len(files), 'launch WLA source count')
    return dict(upstream_source_commits=loaders.PINS, wla_source_files=len(files),
        encoder_sha256=loaders.HASHES['encoder'], jepa_sha256=loaders.HASHES['jepa'])


def audit_training(run, source, config_path, plan_root, *, source_commit,
                   config_sha256, plan_admission_sha256, checkpoint_loader=None):
    run, source, config_path, plan_root = map(Path, (run,source,config_path,plan_root))
    metrics = terminal_metrics(run)  # MUST precede checkpoint hash/load, even for diagnostic callers.
    exact(source_commit, SOURCE_COMMIT, 'only the frozen admitted training source is supported')
    exact(plan_admission_sha256, PLAN_ADMISSION_SHA, 'fixed sampling admission pin required')
    require(plan_root == PLAN_ROOT and source == ROOT/'source_failure_state_train_v1', 'fixed source/plan paths required')
    require(all(p.is_absolute() and p.resolve() == p for p in (run,source,config_path,plan_root)), 'absolute nonsymlink input routes')
    require(git(source,'rev-parse','HEAD') == source_commit and not git(source,'status','--porcelain'), 'frozen source changed/dirty')
    pins = Pins()
    for name in TERMINAL:
        if name != 'checkpoint.pt':
            pins.add(run/name)
    # Keep the earliest no-weight completion gate, but do not report its first
    # document against a different version hashed later in the read sequence.
    pinned_metrics = validate_metrics(read_json(run/'metrics.json'))
    same(metrics, pinned_metrics, 'final metrics changed between completion gate and file pinning')
    metrics = pinned_metrics
    # Pins.finish also rehashes this version before releasing any PASS report.
    pins.add(config_path, config_sha256)
    pins.add(Path(__file__).resolve())
    launch_path = Path(str(run)+'.launch.json')
    pins.add(launch_path)
    launch = read_json(launch_path)
    exact(launch.get('output'), str(run), 'launch output identity')
    config, environment = read_json(run/'config.json'), read_json(run/'environment.json')
    yaml_config = yaml.safe_load(config_path.read_text())
    recipe = validate_config(config,yaml_config,source,run,plan_root,source_commit)
    source_hashes = inventory(source)
    for relative, sha in source_hashes.items():
        pins.add(source/relative, sha)
    # Only reused executable helpers must match the freeze; extra audit tools are permitted.
    for name in ('wa.wm.failure_state_sampling','wa.wm.failure_state_sampling_runtime',
                 'wa.wm.teacher_window_plan','wa.wm.loaders'):
        module_path = Path(importlib.import_module(name).__file__).resolve()
        relative = name.replace('.', '/')+'.py'
        pins.add(module_path, source_hashes[relative])
    hardware = validate_environment(environment,source_hashes,launch,source_commit)
    plan, admission, old_arrays, simulation = load_plan(plan_root,pins)
    for name, sha in admission['provenance']['code_sha256'].items():
        relative = name.replace('.', '/')+'.py'
        exact(source_hashes.get(relative), sha, 'admitted runtime/frozen code binding: '+name)
    inputs = bind_inputs(pins,config,launch,admission)
    fit = bind_fit(pins,launch)
    dependencies = bind_dependencies(pins,config,environment,launch)
    for path, sha in pins.files.items():
        if path in launch['input_sha256']:
            exact(launch['input_sha256'][path], sha, 'launch/input byte mismatch: '+path)
    failure = read_json(run/'failure_state_exposure.json')
    for key, value in dict(experiment='evaluation_adaptation_failure_state_mix_v1',
            plan_sha256=PLAN_SHA, loader_audit_sha256=LOADER_SHA, total_planned_positions=1233424,
            loader_positions=1233424, diagnostic=False,
            initialization='59866 model AND optimizer; one new epoch, cumulative two').items():
        exact(failure.get(key),value,'runtime three-source provenance: '+key)
    same(failure.get('provenance'),admission['provenance'],'runtime/admitted provenance differs')
    same(failure.get('simulation'),simulation,'runtime/admitted simulation differs')
    actual = read_json(run/'actual_exposure_epoch1.json')
    arrays = load_npz(run/'actual_exposure_epoch1.npz')
    exposure = validate_exposure(plan,actual,arrays,old_arrays,simulation)
    for name, count in dict(base=726631,teacher=457641,recovery=49152,total=1233424).items():
        exact(actual['actual_'+name],count,'formal actual exposure total: '+name)
    dual = read_json(run/'dual_teacher_exposure.json')
    old_plan = read_json(OLD_PLAN/'plan.json')
    from wa.wm.teacher_window_plan import canonical_sha
    exact(dual.get('hard_stt_plan_sha256'), canonical_sha(old_plan), 'old pre-DDP canonical plan')
    exact(dual.get('experiment'), 'evaluation_set_adaptation_v1', 'old teacher experiment')
    exact(dual.get('cache_sha256'), pins.files[str(TEACHER/'complete.json')], 'old teacher cache pin')
    for key,value in dict(base_windows=726631,teacher_unique_windows=436816,teacher_repeats=1,
            teacher_exposures=457642,total_windows=1184273).items():
        exact(dual.get(key),value,'old pre-DDP declaration: '+key)
    require(math.ceil(exposure['ranks'][0]['batches']/2) == UPDATES
        and all(x['loader_positions'] == 154178 and x['batches'] == 77089 for x in exposure['ranks']), 'formal eight-rank/update budget')
    rows = [strict_json(line) for line in (run/'train.jsonl').read_text().splitlines()]
    logs = validate_logs(rows,config['base_lrs'])
    # Large weights are reached only after all terminal metrics and CPU contracts pass.
    pins.add(PARENT_PATH,PARENT_SHA)
    exact(launch['input_sha256'].get(PARENT_PATH), PARENT_SHA, 'worker parent model/optimizer SHA')
    checkpoint_sha = pins.add(run/'checkpoint.pt')
    loader = checkpoint_loader or (lambda path: torch.load(path,map_location='cpu',weights_only=True,mmap=True))
    parent, checkpoint = loader(PARENT_PATH), loader(run/'checkpoint.pt')
    continuation = validate_parent_continuation(parent,checkpoint,config)
    del parent, checkpoint
    auxiliary = None
    if (run/'worker_postcheck.json').is_file():
        auxiliary = dict(path=str(run/'worker_postcheck.json'),sha256=pins.add(run/'worker_postcheck.json'),
            authority='Auxiliary only; did not substitute any independent check')
    pins.finish()
    require(git(source,'rev-parse','HEAD') == source_commit and not git(source,'status','--porcelain')
        and inventory(source) == source_hashes,'source changed during terminal audit')
    return dict(schema='failure_state_training_audit_v1', status='TRAINING_ARTIFACT_AUDIT_PASS_OFFLINE_ONLY',
        run=str(run), checkpoint=str(run/'checkpoint.pt'),checkpoint_sha256=checkpoint_sha,
        final_step=FINAL,completed_epochs=2,parent_checkpoint=PARENT_PATH,parent_sha256=PARENT_SHA,
        metrics=metrics['metrics'],source_commit=source_commit,source_hashes=pins.files,
        config_path=str(config_path),config_sha256=config_sha256,recipe=recipe,hardware=hardware,
        plan_admission_sha256=plan_admission_sha256,plan_sha256=PLAN_SHA,exposure=exposure,
        input_admission=inputs,fixed_group_fit=fit,dependencies=dependencies,optimizer_continuation=continuation,logs=logs,
        worker_postcheck=auxiliary,closed_loop=False,
        limits=['No closed-loop SR, untouched-test generalization, or product-success claim.',
            'Scheduler terminal success and full worker logs require separate operational verification.',
            'Rank assignments reconstructed; only merged actual counters were recorded by this trainer.',
            'Prior collection, raw media and full loader admission are hash-bound, not rerun here.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run','source','config','plan-root','source-commit','config-sha256',
                 'plan-admission-sha256','output'):
        parser.add_argument('--'+name,required=True)
    args = parser.parse_args()
    output = Path(args.output)
    protected = tuple(Path(x).resolve() for x in (args.run,args.source,args.plan_root))
    require(output.is_absolute() and output.resolve() == output and not output.exists()
        and not output.is_symlink() and ROOT/'artifacts' in output.parents
        and all(output != p and p not in output.parents for p in protected)
        and output.parent.is_dir(), 'fresh independent NAS audit output required')
    report = audit_training(args.run,args.source,args.config,args.plan_root,
        source_commit=args.source_commit,config_sha256=args.config_sha256,
        plan_admission_sha256=args.plan_admission_sha256)
    with output.open('x') as stream:
        stream.write(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],output=str(output),sha256=digest(output),
        final_step=report['final_step'],closed_loop=False),sort_keys=True))


if __name__ == '__main__':
    main()
