"""Bounded developer diagnostic of unmodified upstream DINO-WM, not a benchmark."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument('--project', required=True)
p.add_argument('--output', required=True)
args = p.parse_args()
project = Path(args.project)
result = {'kind': 'DEVELOPER_DIAGNOSTIC_NOT_EFFECTIVENESS', 'upstream': {}}
for name in ('dino_wm', 'jepa-wms', 'vjepa2'):
    root = project / 'upstream_audit' / name
    result['upstream'][name] = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()

sys.path.insert(0, str(project / 'upstream_audit/dino_wm'))
from models.vit import ViTPredictor
torch.manual_seed(42)
torch.set_num_threads(2)
model = ViTPredictor(num_patches=196, num_frames=3, dim=404,
                     depth=6, heads=16, mlp_dim=2048, pool='mean', dropout=.1).cuda().eval()
x = torch.randn(1, 588, 404, device='cuda')
with torch.no_grad():
    y = model(x)
    later = x.clone()
    later[:, 392:] += 1
    y_later = model(later)
    action_changed = x.clone()
    action_changed[:, :, -10:] += 1
    y_action = model(action_changed)
causal_error = (y[:, :392] - y_later[:, :392]).abs().max().item()
action_effect = (y[:, :, :384] - y_action[:, :, :384]).abs().mean().item()
assert causal_error < 1e-6 and action_effect > 0
model.train()
loss = model(x).square().mean()
loss.backward()
assert all(torch.isfinite(t.grad).all() for t in model.parameters() if t.grad is not None)
result['dino_predictor'] = {
    'parameters': sum(t.numel() for t in model.parameters()),
    'config': {'dim':404,'depth':6,'heads':16,'history':3,'patches':196},
    'gpu': torch.cuda.get_device_name(), 'causal_error': causal_error,
    'action_feature_effect_random_weights': action_effect, 'backward_finite': True,
    'weights': 'random; no learned forecasting claim',
    'latency': 'not measured; shared development GPU is not deployment target',
}

cache = Path('/data/nas_ray/home/zeying.gong/datasets/wla_evt_se2_cache_20260925_v2')
episodes = json.loads((cache / 'train_episodes.json').read_text())
groups = {}
for e in episodes:
    # Roots encode teacher source; select bounded deterministic samples.
    key = next((s for s in ('lightnav', 'official', 'oracle') if s in e['root']), 'other')
    groups.setdefault(key, [])
    if len(groups[key]) < 4:
        groups[key].append(e)
result['data_samples'] = []
for entries in groups.values():
    for e in entries:
        root = Path(e['root'])
        obs = json.loads((root / 'observations.json').read_text())
        dt = np.diff([o['timestamp_s'] for o in obs])
        positions = np.asarray([o['robot_position_world'] for o in obs])
        speed = np.linalg.norm(np.diff(positions, axis=0)[:, [0,2]], axis=1) / dt
        actions_path = root / 'actions.json'
        actions = json.loads(actions_path.read_text()) if actions_path.exists() else None
        first = actions[0] if isinstance(actions, list) and actions else actions
        result['data_samples'].append({
            'root': str(root), 'frames':len(obs), 'dt_min':float(dt.min()),
            'dt_max':float(dt.max()), 'apparent_speed_mps_quantiles':np.quantile(speed,[.5,.95,1]).tolist(),
            'first_action_record':first,
        })
out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
