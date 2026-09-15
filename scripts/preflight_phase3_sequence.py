"""Real-model backward preflight for stateful Phase 3; no optimizer updates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    import yaml
    from torch.utils.data._utils.collate import default_collate
    torch.set_num_threads(1)
    torch.manual_seed(20260914)
    repository = Path.cwd().resolve()
    config_path = args.config.resolve(strict=True)
    config = yaml.safe_load(config_path.read_text())
    if config.get('method') != 'architecture_v1_end_to_end' or config.get('test_locked_used') is not False:
        raise ValueError('requires admitted Architecture-v1 config')
    if config['data']['split'] != 'train':
        raise ValueError('preflight may only use train data')
    for field in ('da3_source', 'da3_runtime'):
        sys.path.insert(0, str(Path(config[field]).resolve(strict=True)))
    from omtrackvla.models.end_to_end import (
        ArchitectureV1Ablation, ArchitectureV1Config, EndToEndFollowPolicy,
        load_official_da3_small_l11,
    )
    from omtrackvla.data.end_to_end_training import Sage3DEndToEndSequenceDataset
    from omtrackvla.training.sequence_training import (
        SequenceTrainingPolicy, sequence_model_inputs, phase3_sequence_loss,
    )
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    architecture = ArchitectureV1Config(**config.get('architecture', {}))
    ablation = ArchitectureV1Ablation(**config.get('ablation', {}))
    backbone, loading = load_official_da3_small_l11(
        Path(config['da3_model']), architecture, ablation,
    )
    policy = EndToEndFollowPolicy(backbone, architecture, ablation).to(device)
    checkpoint_path = args.checkpoint.resolve(strict=True)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint.get('test_locked_used') is not False or checkpoint.get('phase') != 2:
        raise ValueError('preflight starts from the pre-recovery Phase 2 baseline')
    policy.load_state_dict(checkpoint['model'], strict=True)
    index = Path(config['data']['sequence_index'])
    if not index.is_absolute():
        index = repository / index
    dataset = Sage3DEndToEndSequenceDataset(index, split='train', config=architecture)
    batch = default_collate([dataset[index] for index in range(4)])
    inputs = sequence_model_inputs(batch, device)
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    wrapped = SequenceTrainingPolicy(policy)
    wrapped.train()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        outputs = wrapped(**inputs)
        main_loss, _ = phase3_sequence_loss(outputs, batch, {'waypoint': 1.0})
    main_loss.backward(retain_graph=True)
    recurrent_norm = float(policy.gru.weight_hh.grad.float().norm().item())
    main_gradients = {
        name: float(parameter.grad.float().norm().item())
        for name, parameter in policy.named_parameters()
        if parameter.grad is not None and ('gru.' in name or 'fusion.' in name or '.adapter.' in name)
    }
    policy.zero_grad(set_to_none=True)
    weights = {'waypoint': 1.0, 'stop': 0.5, 'bbox': 0.5,
               'visibility': 0.5, 'identity_or_binding': 0.1, 'ego': 0.1}
    total_loss, components = phase3_sequence_loss(outputs, batch, weights)
    total_loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0).item())
    checks = {
        'multi_decision_sequence': inputs['ego_rgb'].shape[1] >= 2,
        'waypoint_loss_finite': bool(torch.isfinite(main_loss)),
        'total_loss_finite': bool(torch.isfinite(total_loss)),
        'recurrent_weight_receives_waypoint_gradient': recurrent_norm > 0,
        'gradient_norm_finite': bool(torch.isfinite(torch.tensor(gradient_norm))),
        'visibility_head_gradient': any(p.grad is not None and p.grad.abs().max() > 0 for p in policy.visibility_head.parameters()),
        'stop_head_gradient': any(p.grad is not None and p.grad.abs().max() > 0 for p in policy.stop_head.parameters()),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    report = {
        'schema_version': 1, 'stage': 'phase3_stateful_sequence_backward_preflight',
        'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
        'optimizer_steps': 0, 'formal_recovery_training_started': False,
        'data_split': 'train', 'test_locked_used': False,
        'checkpoint_sha256': sha256(checkpoint_path), 'config_sha256': sha256(config_path),
        'data_index_sha256': sha256(index), 'da3_loading': loading,
        'sequence_rgb_shape': list(inputs['ego_rgb'].shape),
        'waypoint_shape': list(outputs['waypoints'].shape),
        'waypoint_loss': float(main_loss.detach()), 'total_loss': float(total_loss.detach()),
        'loss_components': {key: float(value.detach()) for key, value in components.items()},
        'gru_weight_hh_waypoint_gradient_norm': recurrent_norm,
        'waypoint_only_gradients': main_gradients,
        'gradient_norm_before_clip': gradient_norm,
        'limitation': 'This verifies real-model sequence gradients; it does not establish recovery or product performance.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'checks': checks, 'output': str(args.output)}), flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
