"""Read-only final artifact acceptance for hard-STT job 61609/72803.

Never load an active checkpoint: complete final metrics and all terminal artifacts
are required first. No scheduler changes, training, model inference, or CUDA use.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess

import numpy as np
import torch

from wa.wm.runtime_exposure import RuntimeExposure
from wa.wm.teacher_window_plan import PlannedTeacherMix, canonical_sha

SOURCE_COMMIT = '8d8efe3aa8a7ce9714913b6f65e5e3c8196eae06'
CANDIDATE_SHA = '663c66b16f1784e1a5a9962501fc4b66c32c32792dc0483be95fad2c16680c66'
CONFIG_SHA = '99f8fc7491b4f7c4c347c73b0ba4f1bf363ff292c76de0d4b29fd9850df19184'
PARENT_SHA = 'ab39144b0490937ce43c4ab28e2d90cdb0c560f700d3364a29ce1dc9a08b999d'
PARENT_PATH = '/data/nas_ray/project/md-ak/users/zeying.gong/job_59866/task_70728/wa_history_repeat_v1/checkpoint.pt'
CONTRACT = 'evt_normalized_command3_actual_dt_v1'
START, FINAL, UPDATES, HELDOUT = 22707, 59716, 37009, 73368
TERMINAL = ('metrics.json', 'checkpoint.pt', 'actual_exposure_epoch1.json',
    'actual_exposure_epoch1.npz', 'config.json', 'environment.json',
    'dual_teacher_exposure.json', 'train.jsonl')

class IncompleteTraining(ValueError): pass

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def require(ok, message):
    if not ok: raise ValueError(message)

def verified_sha(path, expected=None):
    value = digest(path)
    require(expected is None or expected == value, 'changed file: ' + str(path))
    return value

def exact(actual, expected, label):
    require(type(actual) is type(expected) and actual == expected, label)

def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0

def read_json(path):
    return json.loads(Path(path).read_text())

def validate_metrics(metrics):
    for k, v in dict(status='OFFLINE_ONLY', kind='jepa', steps=FINAL,
        closed_loop=False, edge_latency_verified=False, point_source='simulated_uwb').items():
        exact(metrics.get(k), v, 'final metrics contract: ' + k)
    require(set(metrics['metrics']) == {'image', 'point', 'mixed'}, 'three complete modes required')
    for mode, row in metrics['metrics'].items():
        exact(row.get('windows'), HELDOUT, 'incomplete heldout: ' + mode)
        require(row.get('SR') is None and row.get('collision_rate') is None, 'offline score must not claim SR')
        require(all(finite(row.get(k)) for k in ('ADE_m', 'FDE_m', 'yaw_MAE_rad')), 'invalid offline metric')
    require(finite(metrics.get('elapsed_s')), 'invalid final elapsed time')

def terminal_metrics(run):
    missing = [name for name in TERMINAL if not (run/name).is_file()]
    if missing: raise IncompleteTraining('terminal files missing: ' + ', '.join(missing))
    try: metrics = read_json(run/'metrics.json')
    except (ValueError, OSError) as e: raise IncompleteTraining('final metrics not completely readable') from e
    validate_metrics(metrics)
    return metrics

def validate_config(config, yaml_config, source, candidate, run):
    expected = dict(kind='jepa', batch_size=2, accumulation=2, workers=2, epochs=1,
        seed=42, world_weight=.1, diagnostic=False, lane='managed', completed_epochs=1,
        history_repeat_probability=.25, resume=PARENT_PATH, resume_sha256=PARENT_SHA,
        dual_teacher_repeats=1, evaluation_set_adaptation=True, contract=CONTRACT,
        effective_batch=32, train_rows=1184273, heldout_rows=HELDOUT, world_size=8,
        teacher_plan_report_sha256=CANDIDATE_SHA)
    for key, value in expected.items(): exact(config.get(key), value, 'runtime config: ' + key)
    for key in ('recovery_cache', 'recovery_index', 'data_root', 'source_prefix'):
        require(config.get(key) is None, 'foreign data recipe: ' + key)
    require(Path(config['teacher_window_plan']).resolve() == candidate.resolve(), 'candidate path mismatch')
    require(Path(config['output']).resolve() == run.resolve(), 'run output mismatch')
    require(len(config['base_lrs']) == 5 and all(finite(x) and x > 0 for x in config['base_lrs']), 'invalid resumed LR groups')
    exact(yaml_config.get('cluster'), 'baidu_a800', 'wrong cluster')
    exact(yaml_config.get('num_gpus'), 8, 'wrong YAML GPU count')
    tasks = yaml_config.get('tasks')
    require(isinstance(tasks, list) and len(tasks) == 1, 'one full training task required')
    task = tasks[0]
    for key, value in dict(num_gpus=8, type='shell', workload_backend='k8s').items():
        exact(task.get(key), value, 'task contract: ' + key)
    cmd = task['cmd']
    require('cd ' + str(source) in cmd and SOURCE_COMMIT in cmd and 'git status --porcelain' in cmd,
        'frozen source/clean guard missing from YAML')
    lines = [line.strip() for line in cmd.splitlines() if line.strip().startswith('bash wa/scripts/train_world.sh ')]
    require(len(lines) == 1, 'exactly one formal train command required')
    tokens = shlex.split(lines[0])[2:]
    arguments = {}
    i = 0
    while i < len(tokens):
        key = tokens[i]
        require(key.startswith('--') and key not in arguments, 'duplicate/invalid training option')
        if key == '--evaluation-set-adaptation': arguments[key] = True; i += 1
        else:
            require(i+1 < len(tokens) and not tokens[i+1].startswith('--'), 'missing training option value')
            arguments[key] = tokens[i+1]; i += 2
    flags = dict(epochs='1', batch_size='2', accumulation='2', workers='2', seed='42',
        world_weight='0.1', history_repeat_probability='0.25', completed_epochs='1',
        resume=PARENT_PATH, resume_sha256=PARENT_SHA, dual_teacher_cache=config['dual_teacher_cache'],
        dual_teacher_repeats='1', evaluation_set_adaptation=True,
        teacher_window_plan=config['teacher_window_plan'], teacher_plan_report_sha256=CANDIDATE_SHA)
    flags = {'--'+k.replace('_', '-'): v for k, v in flags.items()}
    require(arguments == flags, 'YAML/runtime recipe options differ')

def validate_environment(env, source_hashes):
    exact(env.get('commit'), SOURCE_COMMIT, 'worker source commit mismatch')
    exact(env.get('dirty'), '', 'worker source was dirty')
    require(isinstance(env.get('gpu_names'), list) and len(env['gpu_names']) == 8 and
        all(isinstance(n, str) and 'A800' in n for n in env['gpu_names']), 'eight actual A800 GPUs required')
    require(isinstance(env.get('source_sha256'), dict) and bool(source_hashes) and
        env['source_sha256'] == source_hashes, 'worker Python source inventory/hash mismatch')

def validate_checkpoint(checkpoint, config):
    for key, value in dict(step=FINAL, kind='jepa', contract=CONTRACT, completed_epochs=2,
        parent_checkpoint=PARENT_PATH, parent_sha256=PARENT_SHA).items():
        exact(checkpoint.get(key), value, 'checkpoint provenance: ' + key)
    require(isinstance(checkpoint.get('model'), dict) and bool(checkpoint['model']), 'missing model state')
    require(not any(k.startswith('encoder.') for k in checkpoint['model']), 'unexpected mutable frozen encoder')
    optimizer = checkpoint.get('optimizer')
    require(isinstance(optimizer, dict) and bool(optimizer.get('state')), 'missing optimizer state')
    groups = optimizer.get('param_groups')
    require(isinstance(groups, list) and len(groups) == len(config['base_lrs']), 'optimizer group provenance mismatch')
    params = [p for group in groups for p in group.get('params', [])]
    require(params and len(params) == len(set(params)), 'optimizer parameter identity invalid')
    require(set(optimizer['state']).issubset(params), 'foreign optimizer parameter state')
    state_steps = []
    for state in optimizer['state'].values():
        require(isinstance(state, dict) and all(k in state for k in ('step', 'exp_avg', 'exp_avg_sq')), 'incomplete AdamW state')
        step = state['step']
        if isinstance(step, torch.Tensor):
            require(step.numel() == 1, 'invalid optimizer step tensor'); step = float(step)
        require(finite(step) and 0 < step <= FINAL and float(step).is_integer(), 'invalid optimizer state step')
        state_steps.append(step)
    # Some conditional parameters can legitimately receive fewer updates.
    require(max(state_steps) >= UPDATES, 'optimizer state lacks full continuation evidence')

def validate_logs(rows):
    require(rows and type(rows[0].get('step')) is int and rows[0]['step'] == START+1, 'missing first update evidence')
    previous = START
    for row in rows:
        step = row.get('step')
        require(type(step) is int and previous < step <= FINAL, 'invalid/nonmonotonic train log steps')
        exact(row.get('total_steps'), FINAL, 'wrong planned total steps')
        exact(row.get('phase_step'), step-START, 'wrong phase step')
        exact(row.get('epoch'), 1, 'wrong continuation epoch')
        require(all(finite(row.get(k)) for k in ('loss', 'flow', 'geometry', 'world', 'grad_norm', 'elapsed_s')), 'nonfinite train log')
        previous = step
    # Logger emits first step and multiples of 25, not necessarily the final step.
    require(previous in (FINAL//25*25, FINAL), 'last scheduled training log missing')
    return dict(first_step=rows[0]['step'], last_logged_step=previous,
        final_step_logged=previous == FINAL, log_records=len(rows), final_step_authority='checkpoint + final metrics')

def validate_exposure(mix, records, windows, actual, arrays, simulated):
    require(set(arrays) == {'position_counts', 'base_counts', 'teacher_counts'}, 'unexpected exposure array inventory')
    for key, value in arrays.items():
        require(value.ndim == 1 and value.dtype == np.int64, 'invalid actual exposure dtype/shape: ' + key)
    counter = RuntimeExposure(mix, episode_index=windows['episode_index'], episodes=records,
        hard=windows['hard'], early=windows['early'])
    state = counter.state(); state['position_counts'] = torch.from_numpy(arrays['position_counts'])
    calculated = counter.finalize(state=state, world=8, batch=2, seed=42, epoch=1)
    base, teacher = counter.source_counts(state)
    require(np.array_equal(base.numpy(), arrays['base_counts']), 'base per-window exposure mismatch')
    require(np.array_equal(teacher.numpy(), arrays['teacher_counts']), 'teacher per-window exposure mismatch')
    # JSON turns integer histogram keys into strings; normalize structurally.
    normalize = lambda x: json.loads(json.dumps(x, sort_keys=True, allow_nan=False))
    for key, value in calculated.items():
        require(normalize(actual.get(key)) == normalize(value), 'runtime exposure report mismatch: ' + key)
    exact(actual.get('optimizer_steps_completed'), UPDATES, 'wrong optimizer update count')
    exact(actual.get('diagnostic'), False, 'diagnostic exposure is not formal')
    for key in ('plan_sha256', 'seed', 'epoch', 'world_size', 'batch_size', 'pre_ddp_total',
                'actual_total', 'actual_teacher', 'actual_base', 'dropped', 'per_window_exposure_histogram'):
        require(normalize(calculated[key]) == normalize(simulated[key]), 'candidate exposure mismatch: ' + key)
    require(calculated['simulation_ranks'] == simulated['ranks'], 'candidate rank simulation changed')
    expected_groups = dict(calculated['groups'], hard_stt=calculated['hard_teacher_exposures'],
        hard_stt_early=calculated['hard_early_teacher_exposures'],
        nonhard_teacher=calculated['actual_teacher']-calculated['hard_teacher_exposures'],
        nonhard_teacher_early=calculated['early_teacher_exposures']-calculated['hard_early_teacher_exposures'])
    require(expected_groups == simulated['groups'], 'candidate grouped exposures changed')
    return calculated

def audit_training(run, source, config_path, candidate, *, checkpoint_loader=None):
    run, source, config_path, candidate = map(lambda p: Path(p).resolve(), (run, source, config_path, candidate))
    metrics = terminal_metrics(run)  # Must precede checkpoint hash/load.
    hashes = {}
    def check(path, expected=None):
        path = Path(path).resolve(); value = verified_sha(path, expected)
        hashes[str(path)] = value
        return value
    for name in TERMINAL: check(run/name)
    check(config_path, CONFIG_SHA); check(candidate/'report.json', CANDIDATE_SHA)
    config, env = read_json(run/'config.json'), read_json(run/'environment.json')
    import yaml
    validate_config(config, yaml.safe_load(config_path.read_text()), source, candidate, run)
    git = lambda *args: subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()
    require(git('rev-parse', 'HEAD') == SOURCE_COMMIT and not git('status', '--porcelain'), 'frozen source not clean/pinned')
    inventory = {str(p.relative_to(source)): p for p in (source/'wa').rglob('*.py')}
    validate_environment(env, {name: check(path) for name, path in inventory.items()})
    check(source/'wa/scripts/train_world.sh')
    check(__file__)
    for name in ('teacher_window_plan.py', 'runtime_exposure.py'):
        check(Path(__file__).resolve().parents[1]/'wm'/name, env['source_sha256']['wa/wm/'+name])
    check(PARENT_PATH, PARENT_SHA)
    check(Path(config['cache'])/'complete.json', config['cache_sha256'])
    check(Path(config['index_root'])/'audit.json', config['index_audit_sha256'])
    base_index = read_json(Path(config['index_root'])/'audit.json')
    require(base_index['contract'] == CONTRACT and base_index['cache_complete_sha256'] == config['cache_sha256'], 'base cache/index contract mismatch')
    for split in ('train', 'heldout'):
        check(Path(config['index_root'])/(split+'_valid.npy'), base_index['splits'][split]['index_sha256'])
    report = read_json(candidate/'report.json')
    for name, expected in report['artifacts'].items():
        require(Path(name).name == name, 'unsafe candidate artifact path'); check(candidate/name, expected)
    require(set(report['artifacts']) == {'plan.json', 'episodes.json', 'teacher_windows.npz', 'exposure.json'}, 'candidate inventory changed')
    for path, expected in report['source_hashes'].items(): check(path, expected)
    plan, records, simulated = [read_json(candidate/name) for name in ('plan.json', 'episodes.json', 'exposure.json')]
    require(canonical_sha(plan) == report['plan_canonical_sha256'], 'candidate plan changed')
    with np.load(candidate/'teacher_windows.npz', allow_pickle=False) as z: windows = {k: z[k] for k in z.files}
    eligible = np.flatnonzero(windows['hard']).tolist()
    require(plan['extra_teacher_indices'] == eligible, 'candidate whitelist mismatch')
    mix = PlannedTeacherMix(range(plan['base_count']), range(plan['teacher_count']), plan,
        source_hashes=plan['source_hashes'], eligible_teacher_indices=eligible)
    require(len(mix) == config['train_rows'], 'runtime dataset size mismatch')
    with np.load(run/'actual_exposure_epoch1.npz', allow_pickle=False) as z: arrays = {k: z[k] for k in z.files}
    calculated = validate_exposure(mix, records, windows, read_json(run/'actual_exposure_epoch1.json'), arrays, simulated)
    pre = read_json(run/'dual_teacher_exposure.json')
    for key, value in dict(base_windows=len(mix.base), teacher_unique_windows=len(mix.teacher),
        teacher_repeats=1, teacher_exposures=len(mix)-len(mix.base), total_windows=len(mix),
        hard_stt_plan_sha256=mix.plan_sha256, experiment='evaluation_set_adaptation_v1').items():
        exact(pre.get(key), value, 'pre-DDP recipe mismatch: ' + key)
    logs = [json.loads(x) for x in (run/'train.jsonl').read_text().splitlines() if x]
    log_report = validate_logs(logs)
    # Final metrics were emitted after final checkpoint save and heldout evaluation.
    # Hashes are rechecked after lazy CPU loading to detect concurrent changes.
    loader = checkpoint_loader or (lambda p: torch.load(p, map_location='cpu', weights_only=True, mmap=True))
    checkpoint = loader(run/'checkpoint.pt')
    validate_checkpoint(checkpoint, config)
    del checkpoint
    for path, expected in hashes.items(): require(digest(path) == expected, 'file changed during audit: ' + path)
    return dict(status='TRAINING_ARTIFACT_AUDIT_PASS_OFFLINE_ONLY', source_commit=SOURCE_COMMIT,
        run=str(run), checkpoint=str(run/'checkpoint.pt'), checkpoint_sha256=hashes[str(run/'checkpoint.pt')],
        parent_checkpoint=PARENT_PATH, parent_sha256=PARENT_SHA, completed_epochs=2,
        final_step=FINAL, optimizer_updates=UPDATES, metrics=metrics['metrics'],
        actual_exposure={k: v for k, v in calculated.items() if k != 'per_episode'},
        train_log=log_report, source_hashes=hashes,
        checkpoint_provenance='Pinned clean source loads parent model AND optimizer; final metadata and optimizer state validated',
        limits=['Scheduler terminal status must be checked separately.',
                'Offline heldout trajectory error is not closed-loop SR.',
                'Evaluation-set adaptation; no untouched-test generalization claim.',
                'Exposure proves batch consumption, not performance improvement.'])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('run', 'source', 'config', 'candidate', 'output'): parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    protected = [Path(x).resolve() for x in (args.run, args.source, args.candidate)]
    require(output.is_relative_to('/data/nas_ray') and not any(output.is_relative_to(p) for p in protected),
        'audit output must be separate persistent NAS artifact, never inside run/source/candidate')
    require(not output.exists() and output.parent.is_dir(), 'new audit path with existing parent required')
    try: result = audit_training(args.run, args.source, args.config, args.candidate)
    except IncompleteTraining as e:
        result = dict(status='INCOMPLETE', reason=str(e), checkpoint_loaded=False)
    with output.open('x') as f: json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(dict(status=result['status'], output=str(output), output_sha256=digest(output))))
    if result['status'] == 'INCOMPLETE': raise SystemExit(2)

if __name__ == '__main__': main()
