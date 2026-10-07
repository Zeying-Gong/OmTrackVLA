"""Replay the SHA-pinned 88-window diagnostic; no selection, training or rollout.

Only the candidate is loaded. Old student61377 per-window errors are immutable
reference data; its full seven-point predictions were NOT stored in the old file.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import default_collate

from wa.tools import audit_teacher_group_fit as original
from wa.tools.audit_stt_start_coverage import window_ages
from wa.wm.dual_teacher_data import DualTeacherData
from wa.wm.loaders import HASHES, sha, verify_source
from wa.wm.robot_data import CONTRACT
from wa.wm.training import JointRobotModel

REFERENCE_SHA = '1463ea39a6fe1acf90fcbd813d1ae23bee6ee0e475a314e5bde174b3fa4f8158'
ORIGINAL_TOOL_SHA = 'c5619c59c7d0f9c6ceacbb85b233ae29748b65cc6d703dc579e7696e432d67d1'
METRICS = ('ADE_m', 'FDE_m', 'yaw_MAE_rad', 'first_xy_error_m')
INPUT_KEYS = ('rgb', 'template', 'polar', 'times')
BASELINE = 'student61377'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def verify(path, expected, hashes):
    require(isinstance(expected, str) and len(expected) == 64 and
            all(x in '0123456789abcdef' for x in expected), 'invalid SHA256')
    path = Path(path)
    require(path.is_file(), 'missing file: ' + str(path))
    require(sha(path) == expected, 'changed file: ' + str(path))
    hashes[str(path)] = expected


def refuse_output(path):
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), 'preserve existing output, including symlink')
    require(path.parent.is_dir(), 'output parent must already exist')


def write_report(path, report):
    refuse_output(path)
    # O_EXCL also rejects a path created (or a dangling symlink) after the check.
    with Path(path).open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)


def records_by_key(records, selected):
    require(isinstance(records, list) and len(records) == 2 * len(selected), 'missing paired records')
    mapped = {}
    for row in records:
        i, repeat = row.get('selection_index'), row.get('repeat_history')
        require(type(i) is int and 0 <= i < len(selected) and type(repeat) is bool,
                'invalid paired identity')
        key = (i, repeat)
        require(key not in mapped, 'duplicate paired identity')
        require(row.get('group') == selected[i]['group'], 'record group changed')
        require(all(finite_number(row.get(k)) and row[k] >= 0 for k in METRICS), 'invalid fit metric')
        for field in ('label_first', 'pred_first'):
            arr = np.asarray(row.get(field), dtype=float)
            require(arr.shape == (4,) and np.isfinite(arr).all(), 'invalid first point')
        mapped[key] = row
    require(set(mapped) == {(i, r) for i in range(len(selected)) for r in (False, True)},
            'paired coverage changed')
    for i in range(len(selected)):
        require(mapped[i, False]['label_first'] == mapped[i, True]['label_first'], 'history label changed')
    return mapped


def summaries(records):
    return {group: {str(repeat): {
        key: float(np.mean([r[key] for r in records
                            if r['group'] == group and r['repeat_history'] == repeat]))
        for key in METRICS} for repeat in (False, True)} for group in original.LIMITS}


def validate_reference(report):
    require(report.get('status') == 'BOUNDED_PAIRED_LABEL_FIT_ONLY', 'wrong reference status')
    require(report.get('tool_sha256') == ORIGINAL_TOOL_SHA, 'reference tool changed')
    selected = report.get('selected')
    require(isinstance(selected, list) and len(selected) == 88, 'exactly 88 fixed windows required')
    require(Counter(s.get('group') for s in selected) == original.LIMITS, 'fixed group counts changed')
    for field in ('index', 'raw_row', 'episode', 'episode_uid'):
        require(len({s.get(field) for s in selected}) == 88, 'duplicate selected ' + field)
    for s in selected:
        require(all(type(s.get(k)) is int and s[k] >= 0 for k in ('index', 'raw_row', 'episode')),
                'invalid window indices')
        require(isinstance(s.get('episode_uid'), str) and ':' in s['episode_uid'], 'invalid episode UID')
        task = s['episode_uid'].split(':', 1)[0]
        require(task == ('dt' if s['group'] == 'dt_teacher_control' else
                         'at' if s['group'] == 'at_teacher_control' else 'stt'), 'group/task mismatch')
        require(s.get('teacher') in ('lightnav', 'oracle'), 'foreign teacher')
        require(finite_number(s.get('age_s')) and 0 <= s['age_s'] <= 2, 'not a fixed early window')
    require(set(report.get('models', {})) == set(original.MODELS), 'reference models changed')
    paired = {}
    for name, (_, expected, step) in original.MODELS.items():
        model = report['models'][name]
        require(model.get('sha256') == expected and type(model.get('step')) is int
                and model['step'] == step, 'reference checkpoint identity')
        paired[name] = records_by_key(model['records'], selected)
        require(model.get('summaries') == summaries(model['records']), 'reference summaries changed')
    require(all(paired[BASELINE][key]['label_first'] == paired['parent59866'][key]['label_first']
                for key in paired[BASELINE]), 'reference cross-model labels differ')
    require(isinstance(report.get('source_hashes'), dict) and report['source_hashes'],
            'missing reference source manifest')
    return paired[BASELINE]


def read_reference(path, hashes):
    verify(path, REFERENCE_SHA, hashes)
    verify(original.__file__, ORIGINAL_TOOL_SHA, hashes)
    report = json.loads(Path(path).read_text())
    return report, validate_reference(report)


def verify_reference_sources(report, jobroot, hashes):
    skipped = {str(Path(jobroot) / rel): expected
               for rel, expected, _ in original.MODELS.values()}
    require(all(report['source_hashes'].get(p) == h for p, h in skipped.items()),
            'reference old checkpoint paths changed')
    for path, expected in report['source_hashes'].items():
        if path not in skipped:
            verify(path, expected, hashes)
    return skipped


def fixed_sample(data, selected, baseline, position, hashes):
    s = selected[position]
    i, raw, ep = s['index'], s['raw_row'], s['episode']
    require(i < len(data.rows) and int(data.rows[i]) == raw, 'valid index remapped')
    require(raw < len(data.episode) and int(data.episode[raw]) == ep
            and ep < len(data.episodes), 'raw episode remapped')
    entry = data.episodes[ep]
    require(entry['episode_uid'] == s['episode_uid'] and entry['teacher'] == s['teacher']
            and entry['task'] == s['episode_uid'].split(':', 1)[0], 'selected identity changed')
    obs_path = Path(entry['root']) / 'observations.json'
    verify(obs_path, data.source_records[ep]['observations.json'], hashes)
    times = [o['timestamp_s'] for o in json.loads(obs_path.read_text())]
    age = float(window_ages(times, data.history, np.asarray([raw]))[0])
    require(math.isclose(age, s['age_s'], rel_tol=0, abs_tol=1e-12) and 0 <= age <= 2,
            'selected timestamp changed')
    item = data[i]  # Original admission, artifact/image hashes and window filters.
    sample = {key: item[key] for key in INPUT_KEYS + ('pose',)}
    require(all(isinstance(v, torch.Tensor) and v.dtype == torch.float32
                and torch.isfinite(v).all() for v in sample.values()), 'invalid sample tensors')
    label = sample['pose'].detach().cpu().numpy()
    require(label.shape == (7, 4) and np.array_equal(label, np.asarray(data.pose_labels[raw])),
            'cached seven-step label changed')
    for repeat in (False, True):
        require(np.array_equal(label[0], np.asarray(baseline[position, repeat]['label_first'],
                                                  dtype=np.float32)), 'reference first label changed')
    return sample


def tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous().numpy()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(json.dumps(list(value.shape)).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def validate_candidate_gate(audit, checkpoint, expected_sha, step):
    require(audit.get('status') == 'TRAINING_ARTIFACT_AUDIT_PASS_OFFLINE_ONLY', 'training not audited complete')
    require(type(step) is int and step > 0 and type(audit.get('final_step')) is int
            and audit['final_step'] == step, 'candidate step differs from audit')
    require(Path(audit['checkpoint']).resolve() == Path(checkpoint).resolve()
            and audit.get('checkpoint_sha256') == expected_sha, 'candidate differs from audit')
    require(type(audit.get('completed_epochs')) is int and audit['completed_epochs'] == 2,
            'expected independent cumulative-two-epoch branch')
    require(expected_sha not in {m[1] for m in original.MODELS.values()}, 'old model must not be reloaded')
    require(audit['source_hashes'].get(str(Path(checkpoint))) == expected_sha,
            'candidate absent from audited source manifest')


def candidate_inputs(a, hashes):
    verify(a.training_audit, a.training_audit_sha256, hashes)
    audit = json.loads(Path(a.training_audit).read_text())
    checkpoint = Path(a.candidate_checkpoint)
    validate_candidate_gate(audit, checkpoint, a.candidate_sha256, a.candidate_step)
    run = Path(audit['run'])
    require(checkpoint.parent.resolve() == run.resolve(), 'candidate outside audited run')
    for name in ('metrics.json', 'config.json', 'environment.json'):
        path = run / name
        verify(path, audit['source_hashes'][str(path)], hashes)
    metrics = json.loads((run / 'metrics.json').read_text())
    config = json.loads((run / 'config.json').read_text())
    env = json.loads((run / 'environment.json').read_text())
    require(metrics.get('status') == 'OFFLINE_ONLY' and metrics.get('kind') == 'jepa'
            and metrics.get('steps') == a.candidate_step and metrics.get('closed_loop') is False
            and metrics.get('metrics') == audit['metrics'], 'final metrics not the audited result')
    require(config['kind'] == 'jepa' and config['contract'] == CONTRACT
            and config['completed_epochs'] + config['epochs'] == 2 and not config['diagnostic'],
            'candidate training contract changed')
    require(env['commit'] == audit['source_commit'] and env['dirty'] == '', 'training source contract changed')
    # This validates actual candidate data before the only candidate torch.load.
    verify(checkpoint, a.candidate_sha256, hashes)
    return audit, config, env


def verify_dependencies(root, config, env, hashes):
    verify(config['encoder_weight'], HASHES['encoder'], hashes)
    verify(root / 'models/jepa_wms/mz_jepa-wm.pth.tar', HASHES['jepa'], hashes)
    verify_source(root / 'upstream_audit/dinov2', 'dinov2')
    verify_source(root / 'upstream_audit/jepa-wms', 'jepa-wms')
    verify(config['wla_checkpoint'], env['wla']['checkpoint_sha256'], hashes)
    wla = Path(config['wla_source'])
    files = {str(p.relative_to(wla)) for p in (wla / 'src/md_wla').rglob('*.py')}
    require(files == set(env['wla']['source_files']), 'WLA source inventory changed')
    for name, expected in env['wla']['source_files'].items():
        verify(wla / name, expected, hashes)
    checkout = Path(__file__).resolve().parents[2]
    # Audit imported inference/data code against the actual training environment.
    for name in ('wa/data.py', 'wa/wm/robot_data.py', 'wa/wm/dual_teacher_data.py',
                 'wa/wm/dual_teacher_selection.py', 'wa/wm/training.py', 'wa/wm/adapters.py',
                 'wa/wm/loaders.py'):
        verify(checkout / name, env['source_sha256'][name], hashes)
    return dict(encoder=config['encoder_weight'], wla_source=config['wla_source'],
                wla_checkpoint=config['wla_checkpoint'])


def load_candidate(model, checkpoint, expected_step, audit, loader=torch.load):
    ck = loader(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    require(type(ck.get('step')) is int and ck['step'] == expected_step, 'loaded candidate step mismatch')
    for key, expected in dict(kind='jepa', contract=CONTRACT, completed_epochs=2,
                              parent_checkpoint=audit['parent_checkpoint'],
                              parent_sha256=audit['parent_sha256']).items():
        require(ck.get(key) == expected, 'loaded candidate provenance: ' + key)
    require(isinstance(ck.get('model'), dict) and bool(ck['model']), 'empty candidate weights')
    require(not any(k.startswith('encoder.') for k in ck['model']), 'frozen encoder unexpectedly saved')
    loaded = model.load_state_dict(ck['model'], strict=False)
    require(not loaded.unexpected_keys and all(k.startswith('encoder.') for k in loaded.missing_keys),
            'partial candidate state load')
    del ck


def predict_records(model, samples, selected, device='cuda'):
    records = []
    for i in range(0, len(samples), 2):
        batch = {k: v.to(device) for k, v in default_collate(samples[i:i+2]).items()}
        label = batch['pose']
        for repeat in (False, True):
            policy_batch = {k: batch[k] for k in INPUT_KEYS}
            if repeat:
                policy_batch['rgb'] = batch['rgb'][:, -1:].expand_as(batch['rgb'])
                policy_batch['times'] = torch.zeros_like(batch['times'])
            with torch.no_grad():
                pred, _ = model.predict(policy_batch, torch.full((len(label),), 2, device=device),
                                        torch.zeros_like(label))
            values = original.fit_metrics(pred, label)
            for j in range(len(label)):
                records.append(dict(selection_index=i+j, group=selected[i+j]['group'],
                    repeat_history=repeat, **{k: float(v[j]) for k, v in values.items()},
                    label_first=label[j, 0].tolist(), pred_first=pred[j, 0].tolist(),
                    label_7x4=label[j].tolist(), prediction_7x4=pred[j].tolist(),
                    label_tensor_sha256=tensor_digest(label[j]),
                    input_tensor_sha256={k: tensor_digest(v[j]) for k, v in policy_batch.items()}))
        if (i+2) % 16 == 0:
            print(json.dumps(dict(stage='candidate_predict', windows=i+2)), flush=True)
    return records


def paired_deltas(records, selected, baseline):
    candidate = records_by_key(records, selected)
    rows = []
    for key in sorted(candidate):
        new, old = candidate[key], baseline[key]
        require(new['label_first'] == old['label_first'], 'paired labels differ')
        rows.append(dict(selection_index=key[0], repeat_history=key[1], group=new['group'],
                         **{k: new[k] - old[k] for k in METRICS}))
    groups = {}
    for group in original.LIMITS:
        groups[group] = {}
        for repeat in (False, True):
            rs = [r for r in rows if r['group'] == group and r['repeat_history'] == repeat]
            groups[group][str(repeat)] = {k: dict(
                mean_delta=float(np.mean([r[k] for r in rs])),
                median_delta=float(np.median([r[k] for r in rs])),
                improved=sum(r[k] < 0 for r in rs), tied=sum(r[k] == 0 for r in rs),
                worsened=sum(r[k] > 0 for r in rs), count=len(rs)) for k in METRICS}
    return rows, groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'job-root', 'reference-fit', 'candidate-checkpoint',
                 'candidate-sha256', 'training-audit', 'training-audit-sha256', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--candidate-step', type=int, required=True)
    a = p.parse_args()
    require(not os.environ.get('MD_AK_JOB_ID'), 'developer diagnostic only, not a formal job')
    root, output = Path(a.root), Path(a.output)
    require(all(Path(v).is_absolute() and str(Path(v).resolve()).startswith('/data/nas_ray/')
                for v in (a.root, a.job_root, a.reference_fit, a.candidate_checkpoint,
                          a.training_audit, a.output)), 'all artifacts must remain on verified NAS')
    refuse_output(output)
    start, hashes = time.monotonic(), {str(Path(__file__)): sha(__file__)}
    reference, baseline = read_reference(a.reference_fit, hashes)
    skipped = verify_reference_sources(reference, a.job_root, hashes)
    cache = root / 'artifacts/dual_teacher_se2_cache_20261006_v1'
    verify(cache / 'complete.json', original.CACHE_SHA, hashes)
    audit, config, env = candidate_inputs(a, hashes)
    require(Path(config['dual_teacher_cache']).resolve() == cache.resolve(), 'candidate uses another teacher cache')
    deps = verify_dependencies(root, config, env, hashes)
    data = DualTeacherData(cache)
    complete = json.loads((cache / 'complete.json').read_text())
    for name, expected in complete['files'].items():
        verify(cache / name, expected, hashes)
    verify(cache / 'source_image_hashes.json', complete['source_image_inventory_sha256'], hashes)
    verify(complete['source_release'], complete['source_sha256'], hashes)
    samples = [fixed_sample(data, reference['selected'], baseline, i, hashes) for i in range(88)]
    require(torch.cuda.device_count() == 1, 'explicitly isolate one idle developer GPU')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.cuda.set_device(0)
    model = JointRobotModel(str(root), deps['encoder'], deps['wla_source'],
                            deps['wla_checkpoint'], 'jepa').cuda().eval()
    require(model.provenance == env['wla'], 'constructed WLA provenance differs from training')
    load_candidate(model, Path(a.candidate_checkpoint), a.candidate_step, audit)
    records = predict_records(model, samples, reference['selected'])
    deltas, groups = paired_deltas(records, reference['selected'], baseline)
    for path, expected in hashes.copy().items():
        verify(path, expected, hashes)
    verify_source(root / 'upstream_audit/dinov2', 'dinov2')
    verify_source(root / 'upstream_audit/jepa-wms', 'jepa-wms')
    report = dict(status='FIXED88_CANDIDATE_LABEL_FIT_ONLY',
        reference_fit=str(Path(a.reference_fit)), reference_sha256=REFERENCE_SHA,
        selected=reference['selected'], baseline=reference['models'][BASELINE],
        candidate=dict(path=a.candidate_checkpoint, sha256=a.candidate_sha256,
                       step=a.candidate_step, records=records, summaries=summaries(records)),
        paired_metric_deltas=deltas, paired_summary=groups,
        old_checkpoints_not_rehashed_or_loaded=skipped, source_hashes=hashes,
        scope='Same fixed teacher-state windows; no reselection or relabeling; not closed-loop SR. '
              'Old full trajectories unavailable: only stored per-window errors and first points paired.',
        input='RGB/template/current ideal polar UWB/times only; metadata and pose labels excluded from predict; '
              'zero flow start, mode2, seed42, FP32 action inference and original BF16 frozen encoder',
        reference_gpu=reference['gpu'], gpu=torch.cuda.get_device_name(0),
        gpu_bitwise_equivalence_claim=False, torch_version=torch.__version__,
        tool_sha256=sha(__file__), elapsed_s=time.monotonic()-start,
        peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30)
    write_report(output, report)
    print(json.dumps(dict(status=report['status'], output=str(output), sha256=sha(output))), flush=True)


if __name__ == '__main__':
    main()
