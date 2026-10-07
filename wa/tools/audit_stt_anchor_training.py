"""Independent final acceptance for the fixed STT-anchor v2 continuation.

Read-only CPU audit. No checkpoint is hashed or loaded until final metrics,
source/configuration, immutable input evidence and actual exposure gates pass.
The scheduler terminal state remains an independent operator check.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import shlex
import subprocess

import numpy as np
import torch
from torch.utils.data import DistributedSampler

from wa.tools import audit_hard_stt_training as previous
from wa.tools import build_stt_anchor_candidate as recipe
from wa.wm import loaders
from wa.wm.runtime_exposure import RuntimeExposure
from wa.wm.teacher_plan_runtime_v2 import derive_schedule, BUDGET
from wa.wm.teacher_window_plan import canonical_sha
from wa.wm.teacher_window_schedule import ScheduledTeacherMix, SCHEMA

REPORT_SHA = 'bbe21c78d1b38731f812866158ea1492c0950d6f05843bfbc8d95fd4da3ff8bd'
PLAN_SHA = '42d33d92ba33cebaf2d40b17ea1f282169cf22599a7060f954b3fd5740da07a3'
GROUP_TOTALS = dict(hard_stt_early=9810, hard_stt_late=14286,
                    new_regression_stt=1772, successful_stt_anchor=12514)
GROUP_EXTRAS = dict(hard_stt_early=6540, hard_stt_late=7143,
                    new_regression_stt=886, successful_stt_anchor=6257)
GROUP_UNIQUE = dict(hard_stt_early=3270, hard_stt_late=7143,
                    new_regression_stt=886, successful_stt_anchor=6257)
PARENT_PATH, PARENT_SHA = previous.PARENT_PATH, previous.PARENT_SHA
START, FINAL, UPDATES, HELDOUT = previous.START, previous.FINAL, previous.UPDATES, previous.HELDOUT
CONTRACT = previous.CONTRACT
IncompleteTraining = previous.IncompleteTraining
require, exact, digest, read_json = previous.require, previous.exact, previous.digest, previous.read_json


def normalize(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))

def pin(value, length, label):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{%d}' % length, value), label)

def validate_config(config, yaml_config, source, candidate, run, source_commit):
    pin(source_commit, 40, 'explicit full release commit required')
    expected = dict(kind='jepa', batch_size=2, accumulation=2, workers=2, epochs=1,
        seed=42, world_weight=.1, diagnostic=False, lane='managed', completed_epochs=1,
        history_repeat_probability=.25, resume=PARENT_PATH, resume_sha256=PARENT_SHA,
        dual_teacher_repeats=1, evaluation_set_adaptation=True, contract=CONTRACT,
        effective_batch=32, train_rows=1184273, heldout_rows=HELDOUT, world_size=8,
        teacher_plan_report_sha256=REPORT_SHA, precision='fp32 training / bf16 frozen encoder')
    for key, value in expected.items(): exact(config.get(key), value, 'runtime config: ' + key)
    for key in ('recovery_cache', 'recovery_index', 'data_root', 'source_prefix'):
        require(config.get(key) is None, 'foreign source recipe: ' + key)
    require(Path(config['teacher_window_plan']).resolve() == candidate.resolve(), 'candidate path mismatch')
    require(Path(config['output']).resolve() == run.resolve(), 'run path mismatch')
    require(len(config['base_lrs']) == 5 and all(previous.finite(x) and x > 0
            for x in config['base_lrs']), 'five positive resumed LRs required')
    exact(yaml_config.get('cluster'), 'baidu_a800', 'A800 cluster required')
    exact(yaml_config.get('num_gpus'), 8, 'eight allocated GPUs required')
    tasks = yaml_config.get('tasks')
    require(isinstance(tasks, list) and len(tasks) == 1, 'one complete training task required')
    task = tasks[0]
    for key, value in dict(num_gpus=8, type='shell', workload_backend='k8s').items():
        exact(task.get(key), value, 'task contract: ' + key)
    cmd = task['cmd']
    lines = [line.strip() for line in cmd.splitlines() if line.strip()]
    require(any(shlex.split(line) == ['cd', str(source)] for line in lines if line.startswith('cd '))
            and source_commit in cmd and 'git status --porcelain' in cmd,
            'pinned clean source guards required')
    launch = [line for line in lines if line.startswith('bash wa/scripts/train_world.sh ')]
    require(len(launch) == 1, 'exactly one training launch required')
    tokens, arguments, i = shlex.split(launch[0])[2:], {}, 0
    while i < len(tokens):
        key = tokens[i]
        require(key.startswith('--') and key not in arguments, 'duplicate/invalid train option')
        if key == '--evaluation-set-adaptation': arguments[key] = True; i += 1
        else:
            require(i+1 < len(tokens) and not tokens[i+1].startswith('--'), 'missing train option value')
            arguments[key] = tokens[i+1]; i += 2
    flags = dict(epochs='1', batch_size='2', accumulation='2', workers='2', seed='42',
        world_weight='0.1', history_repeat_probability='0.25', completed_epochs='1',
        resume=PARENT_PATH, resume_sha256=PARENT_SHA, dual_teacher_cache=config['dual_teacher_cache'],
        dual_teacher_repeats='1', evaluation_set_adaptation=True,
        teacher_window_plan=config['teacher_window_plan'], teacher_plan_report_sha256=REPORT_SHA)
    require(arguments == {'--'+k.replace('_','-'):v for k,v in flags.items()},
            'frozen YAML/runtime train options differ')


def validate_environment(env, source_hashes, source_commit):
    exact(env.get('commit'), source_commit, 'worker release commit mismatch')
    exact(env.get('dirty'), '', 'worker source was not clean')
    exact(env.get('torch'), '2.8.0+cu128', 'training framework changed')
    require(isinstance(env.get('gpu_names'), list) and len(env['gpu_names']) == 8
            and all(isinstance(x,str) and 'A800' in x for x in env['gpu_names']), 'actual eight A800s required')
    require(bool(source_hashes) and env.get('source_sha256') == source_hashes,
            'complete worker/frozen Python inventory mismatch')
    wla = env.get('wla', {})
    require(wla.get('strict_loaded') == ['action_expert','metaquery','target_head']
            and wla.get('omitted') == ['Qwen backbone','language LoRA']
            and wla.get('action_contract') == '7x4 XY/sin(yaw)/cos(yaw)'
            and isinstance(wla.get('source_files'), dict) and bool(wla['source_files']),
            'WLA model/source provenance changed')


def recorded_exposure(mix, records, windows, actual, arrays, simulated):
    """Rebuild eight ranks from recorded positions; no checkpoint or model IO."""
    require(set(arrays) == {'position_counts','base_counts','teacher_counts'}, 'actual array inventory')
    for name, length in (('position_counts',len(mix)), ('base_counts',len(mix.base)),
                         ('teacher_counts',len(mix.teacher))):
        value = arrays[name]
        require(value.dtype == np.int64 and value.shape == (length,), 'actual shape/dtype: '+name)
    counter = RuntimeExposure(mix, episodes=records, episode_index=windows['episode_index'],
                              hard=windows['hard'], early=windows['early'])
    state = counter.state(); state['position_counts'] = torch.from_numpy(arrays['position_counts'])
    calculated = counter.finalize(state=state, world=8, batch=2, seed=42, epoch=1)
    base, teacher = counter.source_counts(state)
    require(np.array_equal(base.numpy(), arrays['base_counts']) and
            np.array_equal(teacher.numpy(), arrays['teacher_counts']), 'actual source-window counts differ')
    for key, value in calculated.items():
        require(normalize(actual.get(key)) == normalize(value), 'actual JSON differs: '+key)
    exact(actual.get('optimizer_steps_completed'), UPDATES, 'wrong optimizer updates')
    exact(actual.get('diagnostic'), False, 'diagnostic cannot pass formal audit')
    for key in ('plan_sha256','seed','epoch','world_size','batch_size','pre_ddp_total',
                'actual_total','actual_teacher','actual_base','dropped','per_window_exposure_histogram'):
        require(normalize(calculated[key]) == normalize(simulated[key]), 'candidate simulation differs: '+key)
    require(calculated['simulation_ranks'] == simulated['ranks'], 'candidate rank totals differ')
    # Independent rank mapping, not merely comparison of aggregate histograms.
    rank_teacher = np.zeros((8,len(mix.teacher)), dtype=np.int64)
    rank_base = np.zeros((8,len(mix.base)), dtype=np.int64)
    for rank in range(8):
        sampler = DistributedSampler(mix, num_replicas=8, rank=rank, shuffle=True, seed=42, drop_last=True)
        sampler.set_epoch(1); ids = list(sampler); ids = ids[:len(ids)//2*2]
        require((arrays['position_counts'][ids] == 1).all(), 'missing actual rank positions')
        for pos in ids:
            kind, index = mix.locate(pos)
            (rank_base if kind == 'base' else rank_teacher)[rank,index] += 1
    require(np.array_equal(rank_base.sum(0), arrays['base_counts']) and
            np.array_equal(rank_teacher.sum(0), arrays['teacher_counts']), 'actual rank-source remapping differs')
    return calculated, rank_base, rank_teacher


def recipe_exposure(calculated, arrays, rank_base, rank_teacher, simulated_arrays,
                    records, windows, allocation, simulated):
    expected = dict(plan_sha256=PLAN_SHA, actual_total=1184272, actual_base=726631,
                    actual_teacher=457641, pre_ddp_total=1184273, dropped=1,
                    hard_teacher_exposures=24096, hard_early_teacher_exposures=9810,
                    early_teacher_exposures=154136, schedule_schema=SCHEMA)
    for key,value in expected.items(): exact(calculated.get(key), value, 'fixed recipe exposure: '+key)
    require(calculated['sampling_group_exposures'] == GROUP_TOTALS, 'four group total exposure changed')
    require(calculated['sampling_group_extra_exposures'] == GROUP_EXTRAS, 'four group extra exposure changed')
    require(calculated['sampling_group_unique_windows'] == GROUP_UNIQUE, 'four unique-window groups changed')
    require(set(simulated_arrays) == {'planned_teacher_counts','simulated_teacher_counts',
            'extra_counts','simulated_base_counts','rank_teacher_counts'}, 'simulation array inventory')
    for key,shape in dict(planned_teacher_counts=(436816,),simulated_teacher_counts=(436816,),
            extra_counts=(436816,),simulated_base_counts=(726631,),rank_teacher_counts=(8,436816)).items():
        value = simulated_arrays[key]
        require(value.dtype == np.int64 and value.shape == shape and (value >= 0).all(), 'simulation array invalid:'+key)
    for actual,expected in ((rank_teacher,simulated_arrays['rank_teacher_counts']),
            (rank_base.sum(0),simulated_arrays['simulated_base_counts']),
            (arrays['teacher_counts'],simulated_arrays['simulated_teacher_counts'])):
        require(np.array_equal(actual,expected), 'exact candidate window/rank simulation mismatch')
    masks = recipe.masks_for(records,windows,allocation)
    grouped = recipe.grouped_counts(records,windows,arrays['teacher_counts'],masks)
    require(grouped == simulated['grouped_simulated'], 'complete cohort/teacher/scene exposure mismatch')
    exact(grouped['cohorts']['balanced_success_anchor_pool']['total'],122679,'full anchor-episode pool differs')
    require(grouped['cohorts']['old15_gains']['total'] == 3032 and
            grouped['cohorts']['old76_hard_still_failed']['total'] == 21064, 'original gains/hard groups differ')
    return grouped


def lr_multiplier(update):
    require(type(update) is int and 1 <= update <= UPDATES, 'update outside the sole continuation epoch')
    return .1 + .9*.5*(1+math.cos(math.pi*(update-1)/UPDATES))

def validate_lr_loss_logs(rows, base_lrs):
    result = previous.validate_logs(rows)
    for row in rows:
        expected = [x*lr_multiplier(row['phase_step']) for x in base_lrs]
        require(isinstance(row.get('lr'),list) and len(row['lr']) == 5 and all(
            type(x) in (int,float) and math.isfinite(x) and math.isclose(x,y,rel_tol=1e-12,abs_tol=0.)
            for x,y in zip(row['lr'],expected)), 'resumed LR schedule changed')
        require(math.isclose(row['loss'],row['flow']+.5*row['geometry']+.1*row['world'],
                             rel_tol=2e-5,abs_tol=2e-6), 'original loss decomposition changed')
    return result

def optimizer_steps(state):
    result = {}
    for key, value in state.items():
        step = value['step']
        if isinstance(step,torch.Tensor):
            require(step.device.type == 'cpu' and step.numel() == 1, 'optimizer step must be CPU scalar')
            step = float(step)
        require(previous.finite(step) and float(step).is_integer(), 'invalid optimizer step')
        result[key] = int(step)
    return result

def validate_parent_continuation(parent, checkpoint, config):
    previous.validate_checkpoint(checkpoint,config)
    for key,value in dict(step=START,kind='jepa',contract=CONTRACT).items():
        exact(parent.get(key),value,'parent provenance: '+key)
    require(isinstance(parent.get('model'),dict) and parent['model'].keys() == checkpoint['model'].keys(),
            'parent/final model state identities changed')
    before,after = parent['optimizer'],checkpoint['optimizer']
    require(len(before['param_groups']) == len(after['param_groups']) == 5, 'optimizer group count changed')
    require([x['lr'] for x in before['param_groups']] == config['base_lrs'], 'base LR not resumed from parent optimizer')
    for i,(a,b) in enumerate(zip(before['param_groups'],after['param_groups'])):
        require(a['params'] == b['params'], 'optimizer param mapping changed')
        require({k:v for k,v in a.items() if k!='lr'} == {k:v for k,v in b.items() if k!='lr'},
                'optimizer hyperparameters changed')
        require(math.isclose(b['lr'],config['base_lrs'][i]*lr_multiplier(UPDATES),
                             rel_tol=1e-12,abs_tol=0.), 'final LR schedule changed')
    old_steps,new_steps = optimizer_steps(before['state']),optimizer_steps(after['state'])
    require(old_steps and old_steps.keys() <= new_steps.keys(), 'parent optimizer states lost')
    require(max(old_steps.values()) == START and max(new_steps.values()) == FINAL,
            'full 22707 to59716 continuation missing')
    require(all(0 <= value-old_steps.get(key,0) <= UPDATES for key,value in new_steps.items()),
            'optimizer reset or extra updates')
    for key in parent['model']:
        a,b = parent['model'][key],checkpoint['model'][key]
        require(isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor) and
                a.shape == b.shape and a.dtype == b.dtype and a.device.type == b.device.type == 'cpu',
                'model parameter shape/type changed')
    return dict(parent_step=START,final_step=FINAL,updates=UPDATES,resumed_base_lrs=config['base_lrs'])


def load_npz(path):
    with np.load(path,allow_pickle=False) as z: return {k:z[k] for k in z.files}

def inventory(root):
    return {str(p.relative_to(root)):p for p in (root/'wa').rglob('*.py')}

def audit_training(run, source, config_path, candidate, *, source_commit, config_sha256,
                   checkpoint_loader=None):
    run,source,config_path,candidate = [Path(p).resolve() for p in (run,source,config_path,candidate)]
    metrics = previous.terminal_metrics(run)  # No checkpoint hashing/loading before this gate.
    pin(source_commit,40,'explicit release commit required'); pin(config_sha256,64,'explicit config SHA required')
    hashes = {}
    def check(path,expected=None):
        path=Path(path).resolve(); value=previous.verified_sha(path,expected); hashes[str(path)]=value; return value
    for name in previous.TERMINAL:
        if name != 'checkpoint.pt': check(run/name)
    check(config_path,config_sha256); check(candidate/'report.json',REPORT_SHA)
    config,env,report = read_json(run/'config.json'),read_json(run/'environment.json'),read_json(candidate/'report.json')
    import yaml
    validate_config(config,yaml.safe_load(config_path.read_text()),source,candidate,run,source_commit)
    git=lambda *args:subprocess.check_output(['git','-C',str(source),*args],text=True).strip()
    require(git('rev-parse','HEAD') == source_commit and not git('status','--porcelain'), 'unclean or unpinned release')
    files=inventory(source)
    validate_environment(env,{name:check(path) for name,path in files.items()},source_commit)
    # Audit helpers may not silently come from a later modified checkout.
    current=Path(__file__).resolve().parents[2]
    require({name:check(path) for name,path in inventory(current).items()} == env['source_sha256'],
            'audit-loaded Python implementation differs from worker release')
    check(source/'wa/scripts/train_world.sh')
    root=Path(config['root']).resolve()
    verified,records,windows,new_rows,outcome,pins,base_count,old_dir=recipe.load_context(root)
    for path,expected in verified.hashes.items(): check(path,expected)
    require(set(report['artifacts']) == {'plan.json','episodes.json','teacher_windows.npz','exposure.json',
            'simulated_counts.npz'}, 'candidate file inventory changed')
    for name,expected in report['artifacts'].items(): check(candidate/name,expected)
    for path,expected in report['source_hashes'].items(): check(path,expected)
    require(read_json(candidate/'episodes.json') == records, 'candidate episode identities changed')
    saved_windows=load_npz(candidate/'teacher_windows.npz')
    require(set(saved_windows) == set(windows) and all(np.array_equal(saved_windows[k],windows[k]) for k in windows),
            'candidate original timestamps/row metadata changed')
    extras,groups,allocation=derive_schedule(records,windows,new_rows)
    plan,simulated=read_json(candidate/'plan.json'),read_json(candidate/'exposure.json')
    require(canonical_sha(plan) == PLAN_SHA == report['plan_canonical_sha256'], 'wrong anchor schedule')
    mix=ScheduledTeacherMix(range(base_count),range(len(windows['dataset_index'])),plan,
        source_hashes=pins,selection_source_hashes=recipe.SOURCE_SHA,
        expected_extra_teacher_indices=extras,sampling_groups=groups)
    arrays=load_npz(run/'actual_exposure_epoch1.npz')
    calculated,rank_base,rank_teacher=recorded_exposure(mix,records,windows,
        read_json(run/'actual_exposure_epoch1.json'),arrays,simulated)
    grouped=recipe_exposure(calculated,arrays,rank_base,rank_teacher,load_npz(candidate/'simulated_counts.npz'),
                           records,windows,allocation,simulated)
    pre=read_json(run/'dual_teacher_exposure.json')
    for key,value in dict(base_windows=726631,teacher_unique_windows=436816,teacher_repeats=1,
        teacher_exposures=457642,total_windows=1184273,hard_stt_plan_sha256=PLAN_SHA,
        experiment='evaluation_set_adaptation_v1',teacher_plan_schema=SCHEMA,
        teacher_candidate_kind='stt_anchor_v1',selection_source_hashes=recipe.SOURCE_SHA).items():
        exact(pre.get(key),value,'pre-DDP v2 recipe: '+key)
    # Bind unchanged base and heldout contents, not only their row totals.
    check(Path(config['cache'])/'complete.json',config['cache_sha256'])
    check(Path(config['index_root'])/'audit.json',recipe.old.PINNED['base_index_audit'])
    index=read_json(Path(config['index_root'])/'audit.json')
    exact(config['index_audit_sha256'],recipe.old.PINNED['base_index_audit'],'base audit hash changed')
    require(index['contract'] == CONTRACT and index['cache_complete_sha256'] == config['cache_sha256'],
            'base cache/contract changed')
    for split in ('train','heldout'):
        check(Path(config['index_root'])/(split+'_valid.npy'),index['splits'][split]['index_sha256'])
    for suffix in ('pose.npy','history.npy','episode.npy','episodes.json'):
        expected=check(Path(config['cache'])/('heldout_'+suffix))
        check(Path(config['dual_teacher_cache'])/('heldout_'+suffix),expected)
    check(Path(config['dual_teacher_cache'])/'complete.json',recipe.old.PINNED['cache_complete'])
    for name in ('dinov2','jepa-wms'): loaders.verify_source(root/'upstream_audit'/name,name)
    check(config['encoder_weight'],loaders.HASHES['encoder'])
    check(root/'models/jepa_wms/mz_jepa-wm.pth.tar',loaders.HASHES['jepa'])
    wla=env['wla']; check(config['wla_checkpoint'],wla['checkpoint_sha256'])
    wla_root=Path(config['wla_source']).resolve()
    current_wla={str(p.relative_to(wla_root)):p for p in (wla_root/'src/md_wla').rglob('*.py')}
    require(set(current_wla) == set(wla['source_files']), 'WLA source inventory changed')
    for name,path in current_wla.items(): check(path,wla['source_files'][name])
    rows=[json.loads(x) for x in (run/'train.jsonl').read_text().splitlines() if x]
    log_report=validate_lr_loss_logs(rows,config['base_lrs'])
    # All non-checkpoint gates passed; final metrics follow checkpoint save and validation.
    check(PARENT_PATH,PARENT_SHA); checkpoint_sha=check(run/'checkpoint.pt')
    loader=checkpoint_loader or (lambda p:torch.load(p,map_location='cpu',weights_only=True,mmap=True))
    parent=loader(Path(PARENT_PATH)); checkpoint=loader(run/'checkpoint.pt')
    continuation=validate_parent_continuation(parent,checkpoint,config)
    del parent,checkpoint
    for path,expected in hashes.items(): require(digest(path) == expected,'file changed during audit: '+path)
    require(git('rev-parse','HEAD') == source_commit and not git('status','--porcelain'),'release changed during audit')
    return dict(status='STT_ANCHOR_TRAINING_ARTIFACT_AUDIT_PASS_OFFLINE_ONLY',run=str(run),
        source_commit=source_commit,config_sha256=config_sha256,candidate_report_sha256=REPORT_SHA,
        plan_canonical_sha256=PLAN_SHA,checkpoint=str(run/'checkpoint.pt'),checkpoint_sha256=checkpoint_sha,
        parent_checkpoint=PARENT_PATH,parent_sha256=PARENT_SHA,completed_epochs=2,**continuation,
        actual_exposure={k:v for k,v in calculated.items() if k!='per_episode'},actual_cohorts=grouped,
        anchor_scope='selected6257window exposures12514; entire1248episode pool122679; distinct denominators',
        metrics=metrics['metrics'],train_log=log_report,source_hashes=hashes,
        limits=['Scheduler terminal status must be checked independently.',
            'Exact merged positions validate deterministic eight-rank assignment, not separate per-rank process telemetry.',
            'Checkpoint61609 selects groups only; model AND optimizer initialize from59866.',
            'Offline label error and exposure are not closed-loop SR or performance improvement.',
            'Evaluation-set adaptation; no untouched-test generalization claim.'])


def output_path(path,protected):
    path=Path(path)
    require(not path.exists() and not path.is_symlink(),'refuse existing output including dangling symlink')
    resolved=path.resolve()
    require(resolved.is_relative_to('/data/nas_ray') and path.parent.is_dir() and
            not any(resolved.is_relative_to(Path(p).resolve()) for p in protected),'separate persistent NAS audit output required')
    return path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','source','config','candidate','source-commit','config-sha256','output'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args(); output=output_path(a.output,(a.run,a.source,a.candidate))
    try:
        result=audit_training(a.run,a.source,a.config,a.candidate,
                              source_commit=a.source_commit,config_sha256=a.config_sha256)
    except IncompleteTraining as e:
        result=dict(status='INCOMPLETE',reason=str(e),checkpoint_loaded=False)
    with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(status=result['status'],output=str(output),sha256=digest(output))))
    if result['status']=='INCOMPLETE':raise SystemExit(2)

if __name__=='__main__':main()
