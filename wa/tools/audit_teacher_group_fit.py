"""Bounded paired label fitting: developer-only, no training or rollout."""
import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from wa.wm.dual_teacher_data import DualTeacherData
from wa.wm.training import JointRobotModel
from wa.wm.loaders import sha
from wa.tools.audit_stt_start_coverage import window_ages

CACHE_SHA = '79e5a7dc43b61a23bb64d5676c4c1cac17b459a5e390b308377012e582ba1a8d'
COVERAGE_SHA = '8559e02a411358ba6dd59d7f2cb7da968d81e1542398b08dc09271d6d62ae3ec'
MODELS = {
    'parent59866': ('job_59866/task_70728/wa_history_repeat_v1/checkpoint.pt',
        'ab39144b0490937ce43c4ab28e2d90cdb0c560f700d3364a29ce1dc9a08b999d', 22707),
    'student61377': ('job_61377/task_72474/wa_dual_teacher_train_a800_v1/checkpoint.pt',
        'b5236a21f2d2695780029503c97e339c8350dc4f7337d05ec7e8692b9b9c327f', 59065)}
LIMITS = {'hard_stt_collision': 24, 'hard_stt_other': 16,
          'successful_stt_control': 16, 'dt_teacher_control': 16, 'at_teacher_control': 16}

def select_episodes(entries, hard):
    pools = {k: [] for k in LIMITS}
    for ep, entry in enumerate(entries):
        task, key = entry['episode_uid'].split(':', 1)
        if task != entry['task']: raise ValueError('task identity mismatch')
        if task == 'stt':
            h = hard.get(key)
            if h is not None and not h['early_valid']: continue
            group = ('hard_stt_collision' if h['status'] == 'Collision' else 'hard_stt_other') if h else 'successful_stt_control'
        elif task in ('dt', 'at'):
            group = task + '_teacher_control'
        else: raise ValueError('foreign task')
        pools[group].append((ep, entry))
    result = []
    for group, limit in LIMITS.items():
        ordered = sorted(pools[group], key=lambda item: hashlib.sha256(
            ('wa-fit-v1:' + item[1]['episode_uid']).encode()).hexdigest())
        # Later skip zero-valid/zero-early controls; retain the fixed hash order.
        result.extend((group, ep, entry) for ep, entry in ordered)
    return result

def choose_early_index(ages):
    ages = np.asarray(ages, dtype=float)
    if ages.ndim != 1 or not np.isfinite(ages).all() or (ages < 0).any():
        raise ValueError('invalid age')
    ids = np.flatnonzero(ages <= 2)
    if not len(ids): return None
    return int(ids[np.argmin(abs(ages[ids] - 1.0))])

def fit_metrics(pred, labels):
    if pred.shape != labels.shape or pred.ndim != 3 or pred.shape[1:] != (7, 4):
        raise ValueError('expected paired Bx7x4')
    if not torch.isfinite(pred).all() or not torch.isfinite(labels).all():
        raise ValueError('nonfinite trajectories')
    err = torch.linalg.vector_norm(pred[..., :2] - labels[..., :2], dim=-1)
    yaw = torch.atan2(pred[..., 2], pred[..., 3]) - torch.atan2(labels[..., 2], labels[..., 3])
    return {'ADE_m': err.mean(1), 'FDE_m': err[:, -1],
            'yaw_MAE_rad': torch.atan2(yaw.sin(), yaw.cos()).abs().mean(1),
            'first_xy_error_m': err[:, 0]}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--job-root', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if os.environ.get('MD_AK_JOB_ID'): raise ValueError('developer diagnostic only')
    root, jobroot, out = Path(a.root), Path(a.job_root), Path(a.output)
    if out.exists(): raise ValueError('preserve prior audit')
    start = time.monotonic()
    cache = root/'artifacts/dual_teacher_se2_cache_20261006_v1'
    coverage_path = root/'artifacts/stt_start_coverage_61377_20261007_v2.json'
    hashes = {str(cache/'complete.json'): CACHE_SHA, str(coverage_path): COVERAGE_SHA}
    if any(sha(path) != expected for path, expected in hashes.items()):
        raise ValueError('coverage/cache identity changed')
    coverage = json.loads(coverage_path.read_text())
    if coverage['status'] != 'COVERAGE_AUDIT_PASS': raise ValueError('coverage not audited')
    hashes.update(coverage['source_hashes'])
    if any(sha(path) != expected for path, expected in hashes.items()):
        raise ValueError('coverage sources changed')
    hard = {r['key']: r for r in coverage['hard_details']}
    data = DualTeacherData(cache)
    epids = np.asarray(data.episode)[data.rows]
    selected, samples, count = [], [], Counter()
    for group, ep, entry in select_episodes(data.episodes, hard):
        if count[group] >= LIMITS[group]: continue
        left, right = np.searchsorted(epids, [ep, ep + 1])
        if left == right: continue
        obs_path = Path(entry['root'])/'observations.json'
        expected = data.source_records[ep]['observations.json']
        if sha(obs_path) != expected: raise ValueError('observation changed')
        hashes[str(obs_path)] = expected
        times = [o['timestamp_s'] for o in json.loads(obs_path.read_text())]
        ages = window_ages(times, data.history, data.rows[left:right])
        pick = choose_early_index(ages)
        if pick is None: continue
        index = int(left + pick)
        raw = int(data.rows[index])
        item = data[index]  # Original strict admission verifies artifacts and images.
        samples.append({k: item[k] for k in ('rgb', 'template', 'polar', 'times', 'pose')})
        selected.append(dict(group=group, index=index, raw_row=raw, episode=ep,
            episode_uid=entry['episode_uid'], teacher=entry['teacher'], age_s=float(ages[pick])))
        count[group] += 1
        if len(selected) % 8 == 0:
            print(json.dumps(dict(stage='load_selected', windows=len(selected),
                elapsed_s=time.monotonic()-start)), flush=True)
    if dict(count) != LIMITS: raise ValueError('insufficient bounded group samples: ' + str(count))
    for name, (rel, expected, step) in MODELS.items():
        cp = jobroot/rel
        if sha(cp) != expected: raise ValueError('checkpoint changed: ' + name)
        hashes[str(cp)] = expected
    if torch.cuda.device_count() != 1: raise ValueError('explicitly isolate one idle developer GPU')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.cuda.set_device(0)
    model = JointRobotModel(str(root),
        '/data/nas_ray/home/zeying.gong/datasets_processed/threepanel_debug_v2/traversability_stepp_v1/_weights/dinov2_vits14_pretrain.pth',
        str(root/'dependencies/wla_v1'),
        str(jobroot/'job_58346/task_69086/wla_teacher_train/training/checkpoints/step-0043203.pt'),
        'jepa').cuda().eval()
    report = dict(status='BOUNDED_PAIRED_LABEL_FIT_ONLY', selected=selected, models={},
        selection='Fixed SHA order, one valid teacher window closest to actual t=1s within first2s per episode; 88 windows; no selection by fit',
        scope='Evaluation-set adaptation diagnostic; teacher-state errors, not student-state errors or closed-loop SR; repeated-history is a synthetic startup perturbation',
        input='mixed, current polar UWB, first template; zero flow integration start; float32; no metadata/future RGB supplied to predict',
        gpu=torch.cuda.get_device_name(0), source_hashes=hashes, tool_sha256=sha(__file__))
    for name, (rel, expected, step) in MODELS.items():
        ck = torch.load(jobroot/rel, map_location='cpu', weights_only=True, mmap=True)
        if ck['step'] != step: raise ValueError('checkpoint step mismatch')
        loaded = model.load_state_dict(ck['model'], strict=False)
        if loaded.unexpected_keys or any(not k.startswith('encoder.') for k in loaded.missing_keys):
            raise ValueError('partial model load')
        records = []
        for i in range(0, len(samples), 2):
            b = {k: v.cuda() for k, v in default_collate(samples[i:i+2]).items()}
            for repeat in (False, True):
                bb = dict(b)
                if repeat:
                    bb['rgb'] = b['rgb'][:, -1:].expand_as(b['rgb'])
                    bb['times'] = torch.zeros_like(b['times'])
                with torch.no_grad():
                    pred, _ = model.predict(bb, torch.full((len(bb['pose']),), 2, device='cuda'),
                        torch.zeros_like(bb['pose']))
                measures = fit_metrics(pred, b['pose'])
                for j in range(len(b['pose'])):
                    records.append(dict(selection_index=i+j, group=selected[i+j]['group'],
                        repeat_history=repeat, **{k: float(v[j]) for k, v in measures.items()},
                        label_first=b['pose'][j, 0].tolist(), pred_first=pred[j, 0].tolist()))
            if (i+2) % 16 == 0:
                print(json.dumps(dict(stage='predict', model=name, windows=i+2,
                    elapsed_s=time.monotonic()-start)), flush=True)
        summaries = {}
        for group in LIMITS:
            summaries[group] = {}
            for repeat in (False, True):
                rs = [r for r in records if r['group'] == group and r['repeat_history'] == repeat]
                summaries[group][str(repeat)] = {k: float(np.mean([r[k] for r in rs]))
                    for k in ('ADE_m', 'FDE_m', 'yaw_MAE_rad', 'first_xy_error_m')}
        report['models'][name] = dict(sha256=expected, step=step, records=records, summaries=summaries)
        print(json.dumps(dict(model=name, summaries=summaries)), flush=True)
        del ck
    # Inputs stay immutable throughout diagnosis, including both checkpoints.
    if any(sha(path) != expected for path, expected in hashes.items()):
        raise ValueError('source changed during diagnostic')
    report['elapsed_s'] = time.monotonic()-start
    report['peak_gpu_GiB'] = torch.cuda.max_memory_allocated()/2**30
    with out.open('x') as f: json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(dict(status=report['status'], output=str(out),
        elapsed_s=report['elapsed_s'], sha256=sha(out))), flush=True)

if __name__ == '__main__': main()
