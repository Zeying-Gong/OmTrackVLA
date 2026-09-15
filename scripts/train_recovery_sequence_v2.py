#!/usr/bin/env python3
"""Controlled, single-GPU 64-step development training of real recovery sequences.

No candidate preflight override is used for optimization. Scene-manifest train
and val roles are independently verified before constructing admitted loaders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any, Mapping

PARENT_SHA = '32c8f2f73277cfab8c3c7c22a53994061f7178498cfa003e1635b9576641bc9c'
MODES = ('visual_uwb', 'visual_only', 'uwb_only', 'safe_stop')
LABEL_KEYS = ('supervision_mask', 'target_waypoints', 'waypoint_mask',
              'stop_label_valid', 'binding_label_valid', 'identity_label_valid', 'ego_label_valid')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def parameter_group(name):
    parts = name.split('.')
    if 'adapter' in parts:
        return 'adapter'
    if 'gru' in parts:
        return 'gru'
    return 'policy'


def sample_clean_indices(length, rng=random):
    require(length >= len(MODES), 'clean dataset has fewer than four modes')
    return [mode + len(MODES) * rng.randrange((length - 1 - mode) // len(MODES) + 1)
            for mode in range(len(MODES))]


def fixed_validation_indices(length, *, per_mode=4, seed=48031):
    rng = random.Random(seed)
    indices = []
    for mode in range(len(MODES)):
        pool = list(range(mode, length, len(MODES)))
        require(len(pool) >= per_mode, 'validation split is too small for a balanced fixed set')
        indices.extend(rng.sample(pool, per_mode))
    return indices


def learning_rate_multiplier(step, maximum_steps=64, warmup_steps=8):
    if step < warmup_steps:
        return max(1e-3, (step + 1) / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, maximum_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(1., progress)))


def flatten_metrics(value, prefix=''):
    result = {}
    for name, item in value.items():
        key = f'{prefix}.{name}' if prefix else name
        if isinstance(item, Mapping):
            result.update(flatten_metrics(item, key))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            result[key] = float(item)
    return result


def clean_retention_gate(current, baseline, *, relative=0.05, absolute=1e-4):
    now, old = flatten_metrics(current), flatten_metrics(baseline)
    require(now.keys() == old.keys() and bool(old), 'clean validation metric keys changed')
    failures = {}
    for key, previous in old.items():
        value = now[key]
        require(math.isfinite(previous) and math.isfinite(value), 'nonfinite clean validation metric')
        limit = previous * (1 + relative) + absolute
        if value > limit:
            failures[key] = {'baseline': previous, 'current': value, 'limit': limit}
    return {'passed': not failures, 'relative_tolerance': relative,
            'absolute_tolerance': absolute, 'failed_metrics': failures}


def should_select_best(current_ade, baseline_ade, best_ade, clean_gate):
    require(all(math.isfinite(x) for x in (current_ade, baseline_ade, best_ade)), 'nonfinite recovery ADE')
    return bool(clean_gate['passed'] and current_ade < baseline_ade and current_ade < best_ade)


def stop_at_boundary(step, stop_after_step, interrupted, *, maximum_steps=64):
    # The final optimizer boundary always proceeds to artifact finalization.
    return step < maximum_steps and (interrupted or (stop_after_step is not None and step >= stop_after_step))


def validate_restored_step(step, scheduler_step, *, maximum_steps=64):
    require(isinstance(step, int) and 0 <= step <= maximum_steps, 'restored optimizer step is outside budget')
    require(scheduler_step == step, 'scheduler step differs from optimizer boundary')


def sequential_backward(clean_forward, recovery_forward, *, clean_weight=1., recovery_weight=.25):
    """Release the clean graph before materializing the recovery learning graph."""
    clean_loss, clean_stats = clean_forward()
    (clean_loss * clean_weight).backward()
    del clean_loss
    recovery_loss, recovery_stats = recovery_forward()
    (recovery_loss * recovery_weight).backward()
    del recovery_loss
    return clean_stats, recovery_stats


def effective_configuration(config, base, *, output_dir):
    require(config.get('schema_version') == 1 and config.get('formal_training') is False,
            'this script only permits development pilot configuration')
    require(config.get('test_locked_used') is False and config.get('parent_checkpoint_sha256') == PARENT_SHA,
            'locked split or parent checkpoint contract changed')
    train = config['training']
    for key, expected in {'maximum_steps': 64, 'burn_in_steps': 6, 'learning_policy_calls': 4,
                          'clean_sequence_steps': 2, 'clean_samples_per_mode': 1,
                          'recovery_batch_size': 1, 'num_workers': 0,
                          'validation_every_steps': 16, 'checkpoint_every_steps': 16}.items():
        require(train.get(key) == expected, f'controlled pilot setting changed: {key}')
    require(config['validation']['automatic_promotion'] is False, 'automatic promotion is forbidden')
    require(config['validation']['sage_samples_per_mode'] == 4, 'validation requires 16 balanced SAGE samples')
    for key, expected in {'clean_weight': 1., 'recovery_weight': .25, 'adapter_learning_rate': 1e-6,
                          'policy_learning_rate': 1e-5, 'gru_learning_rate': 1e-5}.items():
        require(train.get(key) == expected, f'controlled pilot setting changed: {key}')
    require(0 <= config['validation']['clean_relative_tolerance'] <= .05
            and 0 <= config['validation']['clean_absolute_tolerance'] <= 1e-4,
            'clean retention thresholds may not be weakened')
    require(base.get('method') == 'architecture_v1_end_to_end' and base.get('test_locked_used') is False,
            'invalid base model configuration')
    return {'pilot': config, 'base_model': base, 'output_dir': str(output_dir),
            'resolved_maximum_steps': 64, 'world_size': 1,
            'sampler': 'python_random_per_mode_clean_and_uniform_recovery_without_workers'}


def resolve(repository, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else repository / path).resolve(strict=True)


def source_identities(repository, base, hash_file):
    """Hash source bytes, including the external DA3/runtime trees actually used."""
    roots = [repository / 'omtrackvla', Path(__file__),
             repository / 'scripts/build_recovery_scene_manifest.py',
             resolve(repository, base['da3_source']), resolve(repository, base['da3_runtime'])]
    result = {}
    for root in roots:
        paths = [root] if root.is_file() else sorted(root.rglob('*.py'))
        require(bool(paths), f'no Python source found for training dependency: {root}')
        for path in paths:
            result[str(path.resolve(strict=True))] = hash_file(path)
    return result


def move_batch(batch, torch, device):
    return {key: value.to(device, non_blocking=False) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def recovery_labels(batch, burn_in):
    require(batch['supervision_mask'].shape[1] == 10, 'recovery requires ten actual policy calls')
    require(batch['supervision_mask'].sum().item() == batch['supervision_mask'].shape[0]
            and batch['supervision_mask'][:, 9].all().item(), 'recovery labels must select anchor 9 only')
    return {key: batch[key][:, burn_in:] for key in LABEL_KEYS}


def average_metrics(values):
    require(bool(values), 'empty metric set')
    keys = set(values[0])
    require(all(set(value) == keys for value in values), 'metric keys differ')
    return {key: sum(value[key] for value in values) / len(values) for key in sorted(keys)}


def prediction_metrics(outputs, batch, weights, torch, loss_fn):
    with torch.no_grad():
        loss, losses = loss_fn(outputs, batch, weights)
        valid = batch['waypoint_mask'].bool().clone()
        valid[..., 0] = False
        if 'supervision_mask' in batch:
            valid &= batch['supervision_mask'].bool()[..., None]
        require(valid.any().item(), 'no supervised future waypoints')
        distance = torch.linalg.vector_norm(outputs['waypoints'].float() - batch['target_waypoints'].float(), dim=-1)
        result = {'ade_m': float(distance[valid].mean().cpu()), 'loss_total': float(loss.float().cpu())}
        result.update({f'loss_{key}': float(value.float().cpu()) for key, value in losses.items()})
        require(all(math.isfinite(value) for value in result.values()), 'nonfinite validation metric')
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--stop-after-step', type=int)
    args = parser.parse_args(argv)
    repository = Path(__file__).resolve().parents[1]
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    if str(repository / 'scripts') not in sys.path:
        sys.path.insert(0, str(repository / 'scripts'))
    config_path = args.config.expanduser().resolve(strict=True)
    config = load_json(config_path)
    output_dir = args.output_dir.expanduser().absolute()
    require(int(os.environ.get('WORLD_SIZE', '1')) == 1, 'this trainer only supports one GPU/rank')
    physical = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    require(physical in tuple(str(n) for n in range(1, 8)), 'select exactly one physical GPU 1..7; GPU 0 is forbidden')
    if args.stop_after_step is not None:
        require(0 < args.stop_after_step < 64, 'stop-after-step must be in 1..63 and never changes the 64-step budget')
    if args.resume_from is not None:
        require(args.resume_from.resolve(strict=True) == (output_dir / 'checkpoints/last.ckpt').resolve(strict=True),
                'resume is restricted to this run directory last.ckpt')
    elif output_dir.exists():
        raise FileExistsError(f'refusing existing output directory: {output_dir}')

    import yaml
    import numpy as np
    import torch
    from torch.utils.data._utils.collate import default_collate
    from omtrackvla.training.recovery_state import (
        RunDirectory, atomic_json, atomic_write, canonical_hash, file_hash,
        capture_rng, restore_rng, run_contract, resume_payload, validate_resume,
    )
    from omtrackvla.data.recovery_sequence import RecoverySequenceDataset, recovery_sequence_collate, validate_recovery_sample
    from build_recovery_scene_manifest import verify_manifest
    base_path = resolve(repository, config['base_model_config'])
    base = yaml.safe_load(base_path.read_text(encoding='utf-8'))
    effective = effective_configuration(config, base, output_dir=output_dir)
    training, validation = config['training'], config['validation']
    manifest_path = resolve(repository, config['recovery_manifest'])
    validator = lambda path: validate_recovery_sample(path, artifact_root=repository)
    manifest, train_paths = verify_manifest(manifest_path, validator=validator, role='train', for_training=True)
    _, val_paths = verify_manifest(manifest_path, validator=validator, role='val', for_training=False)
    require(bool(train_paths) and bool(val_paths) and not set(train_paths) & set(val_paths), 'invalid recovery train/val partition')
    parent_path = resolve(repository, config['parent_checkpoint'])
    require(file_hash(parent_path) == PARENT_SHA, 'Phase 2 parent checkpoint hash mismatch')
    require(manifest['parent_checkpoint']['sha256'] == PARENT_SHA, 'manifest parent checkpoint differs')
    for field in ('da3_source', 'da3_runtime'):
        sys.path.insert(0, str(resolve(repository, base[field])))
    from omtrackvla.models.end_to_end import ArchitectureV1Config, ArchitectureV1Ablation, EndToEndFollowPolicy, load_official_da3_small_l11
    from omtrackvla.data.end_to_end_training import Sage3DEndToEndSequenceDataset, CONDITION_MODES
    from omtrackvla.training.sequence_training import SequenceTrainingPolicy, sequence_model_inputs, phase3_sequence_loss
    require(tuple(CONDITION_MODES) == MODES, 'condition modes changed')
    architecture = ArchitectureV1Config(**base.get('architecture', {}))
    ablation = ArchitectureV1Ablation(**base.get('ablation', {}))
    architecture.validate()
    ablation.validate()
    sage_index_path = resolve(repository, base['data']['sequence_index'])
    clean_train = Sage3DEndToEndSequenceDataset(sage_index_path, split='train', config=architecture, modes=MODES)
    clean_val = Sage3DEndToEndSequenceDataset(sage_index_path, split='val', config=architecture, modes=MODES)
    require(clean_train.sequence_steps == clean_val.sequence_steps == 2, 'clean replay must retain the original two-step sequences')
    fixed_val_indices = fixed_validation_indices(len(clean_val), per_mode=validation['sage_samples_per_mode'], seed=validation['sage_index_seed'])
    loader_options = dict(artifact_root=repository, development_manifest=manifest_path,
                          required_anchor=9, image_height=architecture.image_height, image_width=architecture.image_width)
    recovery_train = RecoverySequenceDataset(train_paths, partition_role='train', **loader_options)
    recovery_val = RecoverySequenceDataset(val_paths, partition_role='val', **loader_options)
    data_hashes = {'recovery_manifest': file_hash(manifest_path), 'parent_checkpoint': PARENT_SHA,
                   'sage_index': file_hash(sage_index_path)}
    sage_index = load_json(sage_index_path)
    for key, path in {
        'sage_source_index': Path(sage_index['source_root']) / 'index.json',
        'sage_sidecar_manifest': Path(sage_index['sidecar_root']) / 'manifest.json',
        'sage_split_manifest': Path(sage_index['split_manifest']),
        'sage_policy_admission': Path(sage_index['policy_admission']),
    }.items():
        data_hashes[key] = file_hash(path)
    source_hashes = source_identities(repository, base, file_hash)
    # Set deterministic-related flags explicitly; their effective values are
    # bound into resume metadata. ROIAlign CUDA may require a nondeterministic
    # implementation, so the default does not claim bitwise kernel replay.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(bool(training['deterministic_algorithms']))
    torch.set_float32_matmul_precision('highest')
    require(torch.cuda.is_available(), 'CUDA is required for this development pilot')
    torch.cuda.set_device(0)
    device = torch.device('cuda', 0)
    use_bf16 = bool(training['bfloat16'])
    require(not use_bf16 or torch.cuda.is_bf16_supported(), 'configured bfloat16 is unsupported')
    runtime = {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
               'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
               'gpu_name': torch.cuda.get_device_name(0), 'physical_gpu': physical,
               'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
               'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG'],
               'tf32': False, 'bfloat16': use_bf16, 'autocast_cache_enabled': True,
               'num_threads': torch.get_num_threads()}
    contract = run_contract(effective_config=effective, code_sha256=canonical_hash(source_hashes),
                            data_hashes=data_hashes, world_size=1, runtime=runtime)
    seed = int(config['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    clean_weights = {key: float(value) for key, value in base['training']['loss_weights'].items() if float(value) != 0.}
    require(clean_weights.get('world_action', 0.) == clean_weights.get('inverse', 0.) == 0., 'training-only dynamics are outside this pilot')
    recovery_weights = {'waypoint': 1.}
    interrupted = False

    def request_stop(*_):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    with RunDirectory(output_dir, contract, resume=args.resume_from is not None) as run:
        checkpoint_dir = output_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        da3, loading = load_official_da3_small_l11(resolve(repository, base['da3_model']), architecture, ablation,
                          resolve(repository, base['dinov2_model']) if base.get('dinov2_model') else None)
        policy = EndToEndFollowPolicy(da3, architecture, ablation).to(device)
        parent = torch.load(parent_path, map_location='cpu', weights_only=False)
        require(parent.get('phase') == 2 and parent.get('test_locked_used') is False, 'parent is not admitted Phase 2')
        policy.load_state_dict(parent['model'], strict=True)
        del parent
        wrapped = SequenceTrainingPolicy(policy)
        groups = {name: [] for name in ('adapter', 'policy', 'gru')}
        group_names = {name: [] for name in groups}
        for name, parameter in policy.named_parameters():
            if parameter.requires_grad:
                groups[parameter_group(name)].append(parameter)
                group_names[parameter_group(name)].append(name)
        require(all(groups.values()), 'adapter/policy/GRU optimizer groups must all be present')
        optimizer = torch.optim.AdamW([
            {'params': values, 'name': name, 'lr': training[f'{name}_learning_rate']}
            for name, values in groups.items()
        ], weight_decay=training['weight_decay'])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
            lambda step: learning_rate_multiplier(step, 64, training['warmup_steps']))
        start_step = 0
        baseline = None
        best = None
        resume = None
        if args.resume_from is not None:
            resume = torch.load(args.resume_from, map_location='cpu', weights_only=False)
            validate_resume(resume['exact_resume'], contract)
            require(resume['run_directory'] == str(output_dir), 'checkpoint belongs to another run')
            require(resume['global_step'] == resume['exact_resume']['global_step'], 'checkpoint step mismatch')
            require(resume['best_selection'] == resume['exact_resume']['best'], 'best selection metadata mismatch')
            policy.load_state_dict(resume['model'], strict=True)
            optimizer.load_state_dict(resume['optimizer'])
            scheduler.load_state_dict(resume['scheduler'])
            start_step = int(resume['global_step'])
            baseline, best = resume['baseline'], resume['best_selection']
            validate_restored_step(start_step, scheduler.last_epoch)
            require(args.stop_after_step is None or args.stop_after_step > start_step, 'stop boundary is not after restored step')

        def autocast():
            # sequence_training owns the burn-in/learning cache boundary fix;
            # all callers use that shared implementation and its regression test.
            return torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_bf16)

        def evaluate():
            rng = capture_rng(rank=0, numpy=np, torch=torch)
            previous_training = policy.training
            policy.eval()
            by_mode = {mode: [] for mode in MODES}
            recovery_metrics = []
            try:
                with torch.inference_mode(), autocast():
                    for index in fixed_val_indices:
                        sample = clean_val[index]
                        batch = move_batch(default_collate([sample]), torch, device)
                        outputs = wrapped(**sequence_model_inputs(batch), burn_in_steps=0)
                        by_mode[sample['condition_mode']].append(prediction_metrics(outputs, batch, clean_weights, torch, phase3_sequence_loss))
                        del outputs, batch, sample
                    for index in range(len(recovery_val)):
                        batch = move_batch(recovery_sequence_collate([recovery_val[index]]), torch, device)
                        outputs = wrapped(**sequence_model_inputs(batch), burn_in_steps=6)
                        labels = recovery_labels(batch, 6)
                        recovery_metrics.append(prediction_metrics(outputs, labels, recovery_weights, torch, phase3_sequence_loss))
                        del outputs, labels, batch
            finally:
                policy.train(previous_training)
                restore_rng(rng, rank=0, numpy=np, torch=torch)
            return {'clean': {'aggregate': average_metrics([m for values in by_mode.values() for m in values]),
                              'per_mode': {mode: average_metrics(values) for mode, values in by_mode.items()}},
                    'recovery': average_metrics(recovery_metrics),
                    'recovery_val_samples': len(recovery_metrics), 'sage_val_indices': fixed_val_indices}

        def checkpoint_payload(step):
            rng = capture_rng(rank=0, numpy=np, torch=torch)
            return {
                'schema_version': 2, 'phase': 3, 'method': 'architecture_v1_end_to_end',
                'stage': config['stage'], 'global_step': step, 'model': policy.state_dict(),
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'architecture': architecture.to_dict(), 'ablation': ablation.to_dict(),
                'run_directory': str(output_dir), 'source_parent_checkpoint_sha256': PARENT_SHA,
                'baseline': baseline, 'best_selection': best,
                'exact_resume': resume_payload(contract=contract, global_step=step,
                    rank_states=[{'rng': rng, 'data_state': {'kind': 'python_random_sampling_no_workers', 'next_step': step}}], best=best),
                'formal_training': False, 'test_locked_used': False, 'automatic_promotion': False,
                'training_only_dynamics': False, 'training_only_heads': {},
            }

        def save_checkpoint(path, step):
            atomic_write(path, lambda handle: torch.save(checkpoint_payload(step), handle))

        def publish_best_alias():
            artifact = output_dir / best['artifact_path']
            require(file_hash(artifact) == best['artifact_sha256'], 'bound best artifact changed')
            def copy(handle):
                with artifact.open('rb') as source:
                    shutil.copyfileobj(source, handle, length=1024 * 1024)
            atomic_write(checkpoint_dir / 'best.ckpt', copy)

        def save_best(step):
            # A unique immutable artifact keeps best/last crash-consistent: last
            # stores this file's hash; resume republishes exactly that artifact.
            best['artifact_path'] = f'checkpoints/best_step_{step:04d}_{run.token}.ckpt'
            save_checkpoint(output_dir / best['artifact_path'], step)
            best['artifact_sha256'] = file_hash(output_dir / best['artifact_path'])
            publish_best_alias()

        if resume is None:
            atomic_json(output_dir / 'run_manifest.json', {
                'effective_config': effective, 'contract': contract, 'source_hashes': source_hashes,
                'data_hashes': data_hashes, 'runtime': runtime, 'recovery_manifest': str(manifest_path),
                'recovery_train_paths': [str(path) for path in train_paths], 'recovery_val_paths': [str(path) for path in val_paths],
                'fixed_sage_val_indices': fixed_val_indices, 'optimizer_parameter_names': group_names,
                'loading': loading, 'formal_training': False, 'test_locked_used': False,
            })
            baseline = evaluate()
            best = {'step': 0, 'recovery_val_ade_m': baseline['recovery']['ade_m'],
                    'source': 'unchanged_phase2_parent', 'clean_gate_passed': True}
            atomic_json(output_dir / 'baseline_validation.json', baseline)
            save_best(0)
            save_checkpoint(checkpoint_dir / 'last.ckpt', 0)
        else:
            require(load_json(output_dir / 'baseline_validation.json') == baseline, 'baseline artifact changed')
            require(baseline['sage_val_indices'] == fixed_val_indices, 'fixed validation indices changed')
            publish_best_alias()
            restore_rng(resume['exact_resume']['rank_states'][0]['rng'], rank=0, numpy=np, torch=torch)
            require(resume['exact_resume']['rank_states'][0]['data_state']['next_step'] == start_step, 'sampler boundary mismatch')
            del resume
        policy.train()
        started = time.time()
        for step in range(start_step, 64):
            clean_indices = sample_clean_indices(len(clean_train))
            recovery_index = random.randrange(len(recovery_train))
            optimizer.zero_grad(set_to_none=True)

            def clean_forward():
                samples = [clean_train[index] for index in clean_indices]
                require([sample['condition_mode'] for sample in samples] == list(MODES), 'clean batch condition modes drifted')
                batch = move_batch(default_collate(samples), torch, device)
                with autocast():
                    outputs = wrapped(**sequence_model_inputs(batch), burn_in_steps=0)
                    loss, _ = phase3_sequence_loss(outputs, batch, clean_weights)
                require(torch.isfinite(loss).item(), 'nonfinite clean loss')
                stats = {}
                with torch.no_grad():
                    for mode_index, mode in enumerate(MODES):
                        out = {key: value[mode_index:mode_index + 1] for key, value in outputs.items()}
                        item = {key: value[mode_index:mode_index + 1] if torch.is_tensor(value) else value for key, value in batch.items()}
                        stats[mode] = prediction_metrics(out, item, clean_weights, torch, phase3_sequence_loss)
                return loss, stats

            def recovery_forward():
                batch = move_batch(recovery_sequence_collate([recovery_train[recovery_index]]), torch, device)
                with autocast():
                    outputs = wrapped(**sequence_model_inputs(batch), burn_in_steps=6)
                    labels = recovery_labels(batch, 6)
                    loss, _ = phase3_sequence_loss(outputs, labels, recovery_weights)
                require(torch.isfinite(loss).item(), 'nonfinite recovery loss')
                return loss, prediction_metrics(outputs, labels, recovery_weights, torch, phase3_sequence_loss)

            clean_stats, recovery_stats = sequential_backward(clean_forward, recovery_forward,
                clean_weight=training['clean_weight'], recovery_weight=training['recovery_weight'])
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for values in groups.values() for parameter in values], training['gradient_clip_norm'])
            require(torch.isfinite(gradient_norm).item(), 'nonfinite gradient norm')
            gradient_groups = {
                name: sum(float(parameter.grad.detach().float().square().sum().cpu())
                          for parameter in values if parameter.grad is not None) ** .5
                for name, values in groups.items()
            }
            require(all(value > 0 and math.isfinite(value) for value in gradient_groups.values()), 'required optimizer group has no finite nonzero gradient')
            optimizer.step()
            scheduler.step()
            global_step = step + 1
            record = {'global_step': global_step, 'clean_indices': clean_indices, 'recovery_index': recovery_index,
                      'clean_per_mode': clean_stats, 'recovery_train': recovery_stats,
                      'gradient_norm_before_clip': float(gradient_norm.cpu()), 'gradient_norms_after_clip': gradient_groups,
                      'learning_rates': {group['name']: group['lr'] for group in optimizer.param_groups},
                      'elapsed_wall_s': time.time() - started}
            atomic_json(output_dir / f'train_step_{global_step:04d}.json', record)
            print(json.dumps(record, sort_keys=True), flush=True)
            if global_step % training['validation_every_steps'] == 0:
                evaluation = evaluate()
                gate = clean_retention_gate(evaluation['clean'], baseline['clean'],
                    relative=validation['clean_relative_tolerance'], absolute=validation['clean_absolute_tolerance'])
                selected = should_select_best(evaluation['recovery']['ade_m'], baseline['recovery']['ade_m'], best['recovery_val_ade_m'], gate)
                if selected:
                    previous_best = best
                    best = {'step': global_step, 'recovery_val_ade_m': evaluation['recovery']['ade_m'],
                            'source': 'recovery_val_improvement_with_clean_retention', 'clean_gate_passed': True}
                    try:
                        save_best(global_step)
                    except BaseException:
                        best = previous_best
                        raise
                atomic_json(output_dir / f'validation_step_{global_step:04d}.json', {
                    'global_step': global_step, 'metrics': evaluation, 'clean_gate': gate,
                    'selected_best': selected, 'best_selection': best, 'automatic_promotion': False})
            stopping = stop_at_boundary(global_step, args.stop_after_step, interrupted)
            if global_step % training['checkpoint_every_steps'] == 0 or stopping or global_step == 64:
                save_checkpoint(checkpoint_dir / f'step_{global_step:04d}.ckpt', global_step)
                save_checkpoint(checkpoint_dir / 'last.ckpt', global_step)
            if stopping:
                atomic_json(output_dir / 'intentional_interruption.json', {
                    'global_step': global_step, 'maximum_steps_unchanged': 64,
                    'reason': 'signal' if interrupted else 'stop_after_step', 'resume_from': str(checkpoint_dir / 'last.ckpt')})
                return 0  # RunDirectory marks this failed: it is intentionally incomplete.
        artifacts = {'checkpoints/last.ckpt': file_hash(checkpoint_dir / 'last.ckpt'),
                     'checkpoints/best.ckpt': file_hash(checkpoint_dir / 'best.ckpt'),
                     'baseline_validation.json': file_hash(output_dir / 'baseline_validation.json'),
                     'validation_step_0064.json': file_hash(output_dir / 'validation_step_0064.json')}
        atomic_json(output_dir / 'TRAINING_COMPLETE.json', {
            'status': 'development_pilot_complete', 'global_step': 64, 'best_selection': best,
            'formal_training': False, 'automatic_promotion': False, 'test_locked_used': False,
            'artifacts': artifacts, 'limitations': config['limitations']})
        artifacts['TRAINING_COMPLETE.json'] = file_hash(output_dir / 'TRAINING_COMPLETE.json')
        run.complete(global_step=64, artifacts=artifacts)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
