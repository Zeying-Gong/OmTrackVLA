"""Bounded old-constructor/direct-WA loading equivalence; never a benchmark SR."""
import argparse
import base64
import dataclasses
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from wa.wm.eval_server import Session
from wa.wm.loaders import sha
from wa.wm.robot_data import CONTRACT
from wa.wm.standalone_inference import load_standalone
from wa.wm.training import JointRobotModel

CORE = ('training.py', 'loaders.py', 'adapters.py', 'robot_data.py',
        'eval_server.py', 'sim_uwb.py')
REFERENCE_COMMIT = '192b57f5e270acfffd8c7c1a4590cb1b257d92a3'


def tensor_signature(tensor):
    value = tensor.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(raw).hexdigest())


def plain(value):
    """Fail closed for unknown non-state construction properties."""
    if value is None or type(value) in (bool, int, str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {'float': str(value)}
    if isinstance(value, torch.Tensor):
        return {'tensor': tensor_signature(value)}
    if isinstance(value, (torch.dtype, torch.device, Path)):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((plain(v) for v in value), key=repr)
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if dataclasses.is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if callable(value) and hasattr(value, '__qualname__'):
        return {'callable': value.__module__ + '.' + value.__qualname__}
    raise TypeError('unhandled non-state property type: ' + str(type(value)))


def nonstate(model):
    result = {}
    for name, module in model.named_modules():
        attrs = {}
        for key, value in vars(module).items():
            if key in ('_parameters', '_buffers', '_modules'):
                continue
            if name == '' and key == 'provenance':
                continue
            attrs[key] = plain(value)
        # Including non-persistent registered buffers, omitted by state_dict.
        attrs['ALL_REGISTERED_BUFFERS'] = plain(module._buffers)
        result[name] = attrs
    return result


def compare_models(old, direct):
    left, right = old.state_dict(), direct.state_dict()
    if left.keys() != right.keys():
        raise ValueError('full state keys differ')
    signatures = {}
    for key in left:
        a, b = left[key], right[key]
        if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
            raise ValueError('full state tensor differs: ' + key)
        signatures[key] = tensor_signature(a)
    lm, rm = dict(old.named_modules()), dict(direct.named_modules())
    if lm.keys() != rm.keys():
        raise ValueError('module tree differs')
    classes = {}
    for name in lm:
        a, b = type(lm[name]), type(rm[name])
        if a is not b and not (name in ('', 'policy.world') and issubclass(b, a)):
            raise ValueError('unexpected module class difference: ' + name)
        classes[name] = [a.__module__ + '.' + a.__name__, b.__module__ + '.' + b.__name__]
    lp, rp = nonstate(old), nonstate(direct)
    for name in lp:
        if lp[name] != rp[name]:
            keys = sorted(k for k in lp[name].keys() | rp[name].keys()
                          if lp[name].get(k) != rp[name].get(k))
            raise ValueError('non-state properties differ: ' + name + ':' + repr(keys))
    if any(m.training for m in old.modules()) or any(m.training for m in direct.modules()):
        raise ValueError('model not eval')
    if any(p.requires_grad for p in old.parameters()) or any(p.requires_grad for p in direct.parameters()):
        raise ValueError('model not frozen')
    return dict(state_signatures=signatures, nonstate_properties=lp, module_classes=classes,
                state_count=len(left), encoder_count=sum(k.startswith('encoder.') for k in left),
                direct_provenance=direct.provenance, legacy_provenance=old.provenance)


def fixture():
    y, x = np.indices((224, 224))
    rgb = np.stack([x % 256, y % 256, (x * 3 + y * 5) % 256], -1).astype(np.uint8)
    stream = io.BytesIO()
    Image.fromarray(rgb).save(stream, format='PNG')
    raw = stream.getvalue()
    return base64.b64encode(raw).decode(), hashlib.sha256(raw).hexdigest()


def seed():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(7)


def run_sessions(model):
    seed()
    png, png_sha = fixture()
    outputs, counts = {}, {'world': 0, 'robot_action': 0, 'robot_state': 0}
    hooks = []
    for key, module in [('world', model.policy.world), ('robot_action', model.robot_action),
                        ('robot_state', model.robot_state)]:
        def count(_module, _inputs, name=key):
            counts[name] += 1
        hooks.append(module.register_forward_pre_hook(count))
    try:
        for name, mode, first_box in [('mixed_bbox', 'mixed', [20, 20, 120, 200]),
                                      ('image_bbox', 'image', [20, 20, 120, 200]),
                                      ('mixed_no_bbox', 'mixed', None)]:
            session = Session(model, mode, 'zero')
            rows = []
            for step, timestamp in enumerate([0., .5]):
                req = dict(rgb_png=png, timestamp_s=timestamp,
                           initial_bbox=first_box if step == 0 else None)
                if mode == 'mixed':
                    req['uwb'] = dict(polar=[2., .1], timestamp_s=timestamp,
                                      valid=True, source='ideal_simulated_uwb',
                                      range_noise_m=0., bearing_noise_rad=0., delay_s=0.)
                value = session.predict(req)
                value.pop('inference_s')
                rows.append(value)
            outputs[name] = rows
            for bad in [dict(req), dict(req, timestamp_s=1., initial_bbox=[20, 20, 120, 200])]:
                try:
                    session.predict(bad)
                except ValueError:
                    pass
                else:
                    raise ValueError('invalid later request accepted')
        try:
            Session(model, 'image', 'zero').predict(dict(rgb_png=png, timestamp_s=0., initial_bbox=None))
        except ValueError:
            pass
        else:
            raise ValueError('image without initial bbox accepted')
        if any(counts.values()):
            raise ValueError('training-only module invoked in Session')
        return dict(outputs=outputs, training_only_calls=counts, synthetic_png_sha256=png_sha,
                    fixture_scope='synthetic deterministic RGB/current UWB; not a trajectory')
    finally:
        for hook in hooks:
            hook.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'reference-source', 'encoder-weight', 'wla-source', 'wla-checkpoint',
                'checkpoint', 'checkpoint-sha256', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--checkpoint-step', required=True, type=int)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--allocator-cap-bytes', type=int, default=5368709120)
    args = parser.parse_args()
    if os.environ.get('MD_AK_JOB_ID'):
        raise ValueError('developer audit only; no cluster smoke')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    started = time.time()
    report = dict(status='FAILED', benchmark_sr=False, h100_verified=False,
                  backward_verified=False, nccl_verified=False, gpu_requested=args.gpu,
                  checkpoint_sha256=args.checkpoint_sha256, checkpoint_step=args.checkpoint_step)
    try:
        torch.set_num_threads(2)
        reference = Path(args.reference_source)
        if subprocess.check_output(['git', '-C', str(reference), 'rev-parse', 'HEAD'], text=True).strip() != REFERENCE_COMMIT:
            raise ValueError('wrong original reference source')
        if subprocess.check_output(['git', '-C', str(reference), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
            raise ValueError('reference source modified')
        checkout = Path(__file__).resolve().parents[2]
        report['source_sha256'] = {}
        for name in CORE:
            current, original = checkout / 'wa/wm' / name, reference / 'wa/wm' / name
            if sha(current) != sha(original):
                raise ValueError('original core changed: ' + name)
            report['source_sha256'][name] = sha(current)
        report['source_sha256']['standalone_inference.py'] = sha(checkout / 'wa/wm/standalone_inference.py')
        report['source_sha256']['auditor'] = sha(__file__)
        if sha(args.checkpoint) != args.checkpoint_sha256:
            raise ValueError('wrong WA checkpoint')
        seed()
        old = JointRobotModel(args.root, args.encoder_weight, args.wla_source, args.wla_checkpoint, 'jepa')
        doc = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
        if doc['step'] != args.checkpoint_step or doc['kind'] != 'jepa' or doc['contract'] != CONTRACT:
            raise ValueError('wrong original checkpoint contract')
        if set(doc['model']) != {k for k in old.state_dict() if not k.startswith('encoder.')}:
            raise ValueError('original checkpoint coverage')
        missing = old.load_state_dict(doc['model'], strict=False)
        if missing.unexpected_keys or any(not k.startswith('encoder.') for k in missing.missing_keys):
            raise ValueError('original partial load')
        del doc
        old.eval().requires_grad_(False)
        seed()
        reads, original_load = [], torch.load
        allowed = {Path(args.checkpoint).resolve(), Path(args.encoder_weight).resolve()}
        def guarded_load(path, *a, **kw):
            resolved = Path(path).resolve()
            if resolved not in allowed:
                raise ValueError('standalone opened unapproved weight: ' + str(resolved))
            reads.append(str(resolved))
            return original_load(path, *a, **kw)
        with patch('torch.load', guarded_load):
            direct = load_standalone(args.root, args.encoder_weight, args.wla_source,
                                     args.checkpoint, args.checkpoint_sha256, args.checkpoint_step)
        report['direct_torch_load_files'] = reads
        report['cpu_comparison'] = compare_models(old, direct)
        print('CPU_FULL_STATE_AND_NONSTATE_EQUAL', flush=True)
        if args.gpu:
            torch.cuda.set_device(0)
            prop = torch.cuda.get_device_properties(0)
            torch.cuda.set_per_process_memory_fraction(args.allocator_cap_bytes / prop.total_memory, 0)
            report['gpu'] = dict(name=prop.name, uuid=str(prop.uuid), capability=[prop.major, prop.minor],
                                 torch=torch.__version__, cuda=torch.version.cuda, allocator_cap_bytes=args.allocator_cap_bytes)
            results = {}
            for label, model in [('legacy', old), ('direct', direct)]:
                torch.cuda.reset_peak_memory_stats()
                model.cuda()
                results[label] = run_sessions(model)
                results[label]['peak_allocated'] = torch.cuda.max_memory_allocated()
                results[label]['peak_reserved'] = torch.cuda.max_memory_reserved()
                model.cpu()
                gc.collect()
                torch.cuda.empty_cache()
            report['gpu_results'] = results
            if results['legacy']['outputs'] != results['direct']['outputs']:
                raise ValueError('Session predictions differ; no relaxed tolerance')
        report['status'] = 'FULL_STATE_NONSTATE_AND_SESSION_EQUAL' if args.gpu else 'CPU_FULL_STATE_NONSTATE_EQUAL_GPU_UNTESTED'
    except Exception as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        report['elapsed_s'] = time.time() - started
        with output.open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
    print(report['status'], flush=True)


if __name__ == '__main__':
    main()
