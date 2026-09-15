"""Check admitted sidecar wiring, optionally backpropagate with zero optimizer steps."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VERIFICATION_SHA = 'aec0690b2d2a3727d7e3ba80afce07670876568e49ed2028e48d6545d70cfc5d'
PLAN_SHA = 'dd9fcd6b00249a802410f1638216f56a64af51ef998fd1fc9d6566317e080780'
PARENT_SHA = '32c8f2f73277cfab8c3c7c22a53994061f7178498cfa003e1635b9576641bc9c'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()
    require(not args.output.exists(), 'preflight output already exists')
    import torch
    from omtrackvla.data.recovery_sequence_groups import VariableRecoverySequences, MODEL_KEYS
    from omtrackvla.data.perception_sidecar import PerceptionSidecarStore, overlay_recovery_sequence
    from omtrackvla.training.sequence_training import sequence_model_inputs, phase3_sequence_loss, SequenceTrainingPolicy

    torch.set_num_threads(1)
    torch.manual_seed(20260914)
    verification_path = ROOT / 'outputs/takeover/perception_sidecar_independent_admission_v1/verification.json'
    plan_path = ROOT / '.codex_upload/perception_label_collection_v1/frozen_plan.json'
    manifest_path = ROOT / 'outputs/takeover/long_prefix_admission_v2/manifest.json'
    require(sha(verification_path) == VERIFICATION_SHA and sha(plan_path) == PLAN_SHA, 'admission pin changed')
    verification = json.loads(verification_path.read_text())
    store = PerceptionSidecarStore(
        verification_path=verification_path, verification_sha256=VERIFICATION_SHA,
        plan_path=plan_path, plan_file_sha256=PLAN_SHA, artifact_root=ROOT)
    dataset = VariableRecoverySequences(manifest_path, partition_role='train', for_training=False,
                                        artifact_root=ROOT)
    admitted = set(verification['distinct_original_sample_paths'])
    identities = [item for item in dataset.identities if str(item.sample_path) in admitted]
    require(len(identities) == len(admitted) == 19, 'must exercise all 19 original prefixes')
    records = []
    backward_batch = None
    for identity in identities:
        original = dataset[identity.manifest_index].as_batch().batch
        before = {key: value.clone() for key, value in original.items()}
        supervision = store.supervision_for(identity.sample_path, learning_steps=4, for_optimizer=False)
        combined = overlay_recovery_sequence(original, supervision, for_optimizer=False)
        require(all(torch.equal(original[k], before[k]) for k in original), 'overlay mutated the original batch')
        require(all(combined[k] is original[k] for k in MODEL_KEYS), 'overlay replaced original model inputs')
        require(all(torch.equal(combined[k], before[k]) for k in MODEL_KEYS), 'overlay changed original model input values')
        del before
        inputs = sequence_model_inputs(combined)
        require(set(inputs) == set(MODEL_KEYS), 'GT supervision entered model inputs')
        require(torch.equal(combined['waypoint_mask'], original['waypoint_mask']), 'waypoint mask changed')
        require(torch.equal(combined['target_waypoints'], original['target_waypoints']), 'waypoint targets changed')
        require(int(combined['waypoint_mask'].any(-1).sum()) == 1, 'waypoints leaked into prefix')
        for key in ('stop_label_valid', 'binding_label_valid', 'identity_label_valid', 'ego_label_valid'):
            require(not bool(combined[key].any()), 'unavailable auxiliary supervision was invented')
        burn = max(0, identity.sequence_length - 4)
        require(not bool(combined['visibility_label_valid'][:, :burn].any()), 'burn-in supervised')
        require(int(combined['visibility_label_valid'].sum()) == min(4, identity.sequence_length), 'missing perception suffix')
        visible = combined['target_visible'].bool() & combined['visibility_label_valid']
        require(not bool((combined['bbox_label_valid'] & ~visible).any()), 'invisible bbox marked valid')
        record = {'sample_path': str(identity.sample_path), 'sample_sha256': sha(identity.sample_path),
                  'sequence_length': identity.sequence_length, 'burn_in_steps': burn,
                  'visibility_labels': int(combined['visibility_label_valid'].sum()),
                  'visible_labels': int(visible.sum()), 'bbox_labels': int(combined['bbox_label_valid'].sum()),
                  'waypoint_anchors': int(combined['waypoint_mask'].any(-1).sum()),
                  'model_inputs_unchanged': True}
        records.append(record)
        if 'stt_0000_128steps__anchor027/' in str(identity.sample_path):
            require(identity.sequence_length == 28 and burn == 24, 'fixed transition prefix length changed')
            require(combined['target_visible'][0, burn:].tolist() == [1., 1., 1., 0.], 'fixed visibility transition shifted')
            require(combined['bbox_label_valid'][0, burn:].tolist() == [True, True, True, False], 'fixed bbox transition shifted')
            backward_batch = (combined, burn, record)
        print(json.dumps({'verified_sample': len(records), 'total': 19, **record}), flush=True)
    denied = False
    try:
        store.supervision_for(identities[0].sample_path, for_optimizer=True)
    except ValueError:
        denied = True
    require(denied, 'read-only admission unexpectedly allowed optimizer use')
    report = {'status': 'passed', 'stage': 'perception_supervision_wiring_preflight',
              'optimizer_steps': 0, 'optimizer_created': False, 'optimizer_input_allowed': False,
              'formal_training_eligible': False, 'product_acceptance_evidence': False,
              'test_locked_used': False, 'verification_sha256': VERIFICATION_SHA,
              'plan_file_sha256': PLAN_SHA, 'manifest_sha256': sha(manifest_path),
              'sample_count': len(records), 'samples': records, 'optimizer_access_rejected': denied,
              'model_loaded': False,
              'source_hashes': {name: sha(ROOT / name) for name in (
                  'scripts/preflight_perception_supervision.py', 'omtrackvla/data/perception_sidecar.py',
                  'omtrackvla/training/sequence_training.py')}}
    if args.backward:
        require(backward_batch is not None, 'fixed transition sample missing')
        import yaml
        config_path = ROOT / 'configs/phases/phase2_end_to_end_v1_phase1_init_formal.yaml'
        config = yaml.safe_load(config_path.read_text())
        require(config['test_locked_used'] is False and config['method'] == 'architecture_v1_end_to_end', 'unexpected model config')
        for key in ('da3_source', 'da3_runtime'):
            sys.path.insert(0, str(Path(config[key]).resolve(strict=True)))
        from omtrackvla.models.end_to_end import (
            ArchitectureV1Ablation, ArchitectureV1Config, EndToEndFollowPolicy, load_official_da3_small_l11)
        architecture = ArchitectureV1Config(**config.get('architecture', {}))
        ablation = ArchitectureV1Ablation(**config.get('ablation', {}))
        parent_path = ROOT / 'outputs/ablations/next026_abl08_converged_v1/teacher_on/gru_curriculum_4k/checkpoints/best.ckpt'
        require(sha(parent_path) == PARENT_SHA, 'Phase2 checkpoint changed')
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        backbone, loading = load_official_da3_small_l11(Path(config['da3_model']), architecture, ablation)
        policy = EndToEndFollowPolicy(backbone, architecture, ablation).to(device)
        checkpoint = torch.load(parent_path, map_location='cpu', weights_only=False)
        require(checkpoint['phase'] == 2 and checkpoint['test_locked_used'] is False, 'wrong checkpoint')
        policy.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        combined, burn, record = backward_batch
        labels = {key: value[:, burn:].to(device) for key, value in combined.items() if key not in MODEL_KEYS}
        model = SequenceTrainingPolicy(policy).train()
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=True):
            outputs = model(**sequence_model_inputs(combined, device), burn_in_steps=burn)
            loss, components = phase3_sequence_loss(outputs, labels, {'visibility': 1., 'bbox': 1.})
        require(bool(torch.isfinite(loss)), 'nonfinite perception loss')
        loss.backward()
        gradients = {name: float(p.grad.float().norm()) for name, p in policy.named_parameters()
                     if p.grad is not None and any(token in name for token in ('bbox', 'visibility', 'gru'))}
        require(all(bool(torch.isfinite(p.grad).all()) for p in policy.parameters() if p.grad is not None), 'nonfinite gradient')
        for token in ('bbox', 'visibility'):
            require(any(value > 0 for name, value in gradients.items() if token in name), token + ' head has no gradient')
        require(sha(parent_path) == PARENT_SHA, 'parent file changed during preflight')
        report['model_loaded'] = True
        report['backward'] = {'sample': record, 'loss': float(loss.detach()),
                              'components': {k: float(v.detach()) for k, v in components.items()},
                              'gradients': gradients, 'parent_checkpoint_sha256': PARENT_SHA,
                              'config_sha256': sha(config_path), 'precision': 'bf16',
                              'da3_loading': loading, 'checkpoint_saved': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'sample_count': len(records),
                      'backward': args.backward, 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
