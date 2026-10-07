import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
import torch
from torch.utils.data import DistributedSampler

from wa.tools.audit_hard_stt_training import (CONTRACT, FINAL, UPDATES, HELDOUT, PARENT_PATH,
    PARENT_SHA, CANDIDATE_SHA, SOURCE_COMMIT, IncompleteTraining, audit_training,
    validate_metrics, validate_checkpoint, validate_logs, validate_config, validate_exposure, validate_environment)
from wa.tools.audit_hard_stt_training import CONFIG_SHA, verified_sha, digest
from wa.wm.teacher_window_plan import SCHEMA, SOURCE_KEYS, PlannedTeacherMix, simulate_exposure
from wa.wm.runtime_exposure import RuntimeExposure

class HardTrainingAuditTest(unittest.TestCase):
    def metrics(self):
        return dict(status='OFFLINE_ONLY', kind='jepa', steps=FINAL, closed_loop=False,
            edge_latency_verified=False, point_source='simulated_uwb', elapsed_s=10.,
            metrics={m:dict(ADE_m=.2,FDE_m=.4,yaw_MAE_rad=.1,windows=HELDOUT,SR=None,collision_rate=None)
                     for m in ('image','point','mixed')})

    def log(self, step):
        return dict(step=step, total_steps=FINAL, phase_step=step-22707, epoch=1,
            loss=1.,flow=.9,geometry=.1,world=.1,grad_norm=1.,elapsed_s=step-22707.)

    def test_missing_terminal_files_never_load_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'checkpoint.pt').write_bytes(b'active partial checkpoint')
            loader = Mock(side_effect=AssertionError('must not load active checkpoint'))
            with self.assertRaises(IncompleteTraining):
                audit_training(root, root, root/'x.yaml', root, checkpoint_loader=loader)
            loader.assert_not_called()

    def test_metrics_full_offline_three_modes_and_strict_types(self):
        validate_metrics(self.metrics())
        for key,value in [('steps',59065),('closed_loop',True),('kind','dino'),('status','DIAGNOSTIC_PASS')]:
            bad=self.metrics();bad[key]=value
            with self.assertRaises(ValueError): validate_metrics(bad)
        for key,value in [('windows',16),('ADE_m',float('nan')),('SR',.9),('FDE_m',True)]:
            bad=self.metrics();bad['metrics']['mixed'][key]=value
            with self.assertRaises(ValueError): validate_metrics(bad)

    def test_actual_gpu_and_worker_source_inventory_required(self):
        hashes={'wa/wm/train.py':'a'*64}
        env=dict(commit=SOURCE_COMMIT,dirty='',gpu_names=['NVIDIA A800-SXM4-80GB']*8,source_sha256=hashes)
        validate_environment(env,hashes)
        for key,value in [('commit','a'*40),('dirty',' M wa/wm/train.py'),
            ('gpu_names',['RTX4090']*8),('gpu_names',['A800']*7),('source_sha256',{})]:
            bad=copy.deepcopy(env);bad[key]=value
            with self.assertRaises(ValueError):validate_environment(bad,hashes)
        with self.assertRaises(ValueError):validate_environment(env,{'wa/wm/train.py':'b'*64})

    def test_sparse_last_log_59700_is_valid_not_final_step_failure(self):
        rows=[self.log(22708),self.log(22725),self.log(59700)]
        report=validate_logs(rows)
        self.assertFalse(report['final_step_logged']);self.assertEqual(report['last_logged_step'],59700)
        self.assertTrue(validate_logs(rows+[self.log(59716)])['final_step_logged'])
        for bad in (rows[1:],rows[::-1],rows+[self.log(59700)],rows[:-1]):
            with self.assertRaises(ValueError):validate_logs(bad)
        bad=copy.deepcopy(rows);bad[1]['phase_step']=0
        with self.assertRaises(ValueError):validate_logs(bad)

    def test_checkpoint_parent_optimizer_and_epoch_provenance(self):
        config=dict(base_lrs=[1e-7]*5)
        cp=dict(step=FINAL,kind='jepa',contract=CONTRACT,completed_epochs=2,
            parent_checkpoint=PARENT_PATH,parent_sha256=PARENT_SHA,model={'head':torch.zeros(1)},
            optimizer=dict(param_groups=[{'params':[i]} for i in range(5)],
                state={i:dict(step=torch.tensor(float(FINAL)),exp_avg=torch.zeros(1),exp_avg_sq=torch.zeros(1)) for i in range(5)}))
        validate_checkpoint(cp,config)
        for key,value in [('step',59065),('completed_epochs',3),('parent_sha256','a'*64),('optimizer',{})]:
            bad=copy.deepcopy(cp);bad[key]=value
            with self.assertRaises(ValueError):validate_checkpoint(bad,config)
        bad=copy.deepcopy(cp)
        for state in bad['optimizer']['state'].values(): state['step']=1
        with self.assertRaises(ValueError):validate_checkpoint(bad,config)

    def config_fixture(self):
        config=dict(kind='jepa',batch_size=2,accumulation=2,workers=2,epochs=1,seed=42,
            world_weight=.1,diagnostic=False,lane='managed',completed_epochs=1,
            history_repeat_probability=.25,resume=PARENT_PATH,resume_sha256=PARENT_SHA,
            dual_teacher_repeats=1,evaluation_set_adaptation=True,contract=CONTRACT,
            effective_batch=32,train_rows=1184273,heldout_rows=HELDOUT,world_size=8,
            teacher_plan_report_sha256=CANDIDATE_SHA,teacher_window_plan='/candidate',output='/run',
            base_lrs=[1e-7]*5,dual_teacher_cache='/cache')
        flags='--epochs 1 --batch-size 2 --accumulation 2 --workers 2 --seed 42 --world-weight 0.1 --history-repeat-probability 0.25 --completed-epochs 1 --resume '+PARENT_PATH+' --resume-sha256 '+PARENT_SHA+' --dual-teacher-cache /cache --dual-teacher-repeats 1 --evaluation-set-adaptation --teacher-window-plan /candidate --teacher-plan-report-sha256 '+CANDIDATE_SHA
        yaml=dict(cluster='baidu_a800',num_gpus=8,tasks=[dict(num_gpus=8,type='shell',workload_backend='k8s',
            cmd='cd /frozen\ntest commit = '+SOURCE_COMMIT+'\ntest -z "$(git status --porcelain)"\nbash wa/scripts/train_world.sh '+flags)])
        return config,yaml

    def test_config_and_yaml_must_agree_no_duplicate_override(self):
        config,yaml=self.config_fixture()
        validate_config(config,yaml,Path('/frozen'),Path('/candidate'),Path('/run'))
        for key,value in [('world_size',1),('world_weight',.2),('accumulation',1),('diagnostic',True)]:
            bad=copy.deepcopy(config);bad[key]=value
            with self.assertRaises(ValueError):validate_config(bad,yaml,Path('/frozen'),Path('/candidate'),Path('/run'))
        bad=copy.deepcopy(yaml);bad['tasks'][0]['cmd']+=' --epochs 3'
        with self.assertRaises(ValueError):validate_config(config,bad,Path('/frozen'),Path('/candidate'),Path('/run'))

    def exposure_fixture(self):
        return self._exposure_fixture()

    def test_pinned_yaml_hash_rejects_extra_shell_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fixture.yaml'
            path.write_text('cmd: formal fixture command\n')
            expected=digest(path)
            self.assertEqual(verified_sha(path,expected),expected)
            with self.assertRaises(ValueError):verified_sha(path,CONFIG_SHA)
            path.write_text('cmd: formal fixture command\nextra: unsafe command\n')
            with self.assertRaises(ValueError):verified_sha(path,expected)

    def _exposure_fixture(self):
        plan=dict(schema=SCHEMA,base_count=101,teacher_count=17,extra_repeats=2,
            extra_teacher_indices=[0,3],source_hashes={k:'a'*64 for k in SOURCE_KEYS})
        mix=PlannedTeacherMix(range(101),range(17),plan,source_hashes=plan['source_hashes'],eligible_teacher_indices=[0,3])
        records=[dict(episode_uid='stt:a/1',task='stt',teacher='lightnav')]
        windows=dict(episode_index=np.zeros(17,dtype=np.int64),hard=np.arange(17)%3==0,early=np.arange(17)<8)
        counter=RuntimeExposure(mix,episodes=records,**windows)
        for rank in range(8):
            sampler=DistributedSampler(mix,num_replicas=8,rank=rank,seed=42,drop_last=True);sampler.set_epoch(1)
            ids=list(sampler);counter.consume(torch.tensor(ids[:len(ids)//2*2]))
        actual=counter.finalize();actual.update(optimizer_steps_completed=UPDATES,diagnostic=False)
        base,teacher=counter.source_counts();arrays=dict(position_counts=counter.state()['position_counts'].numpy(),base_counts=base.numpy(),teacher_counts=teacher.numpy())
        simulated,_=simulate_exposure(mix)
        simulated['groups']=dict(actual['groups'],hard_stt=actual['hard_teacher_exposures'],hard_stt_early=actual['hard_early_teacher_exposures'],nonhard_teacher=actual['actual_teacher']-actual['hard_teacher_exposures'],nonhard_teacher_early=actual['early_teacher_exposures']-actual['hard_early_teacher_exposures'])
        return mix,records,windows,actual,arrays,simulated

    def test_exact_positions_windows_and_json_are_all_checked(self):
        fixture=self.exposure_fixture();validate_exposure(*fixture)
        for key in ('position_counts','base_counts','teacher_counts'):
            bad=copy.deepcopy(fixture);bad[4][key][0]+=1
            with self.assertRaises(ValueError):validate_exposure(*bad)
        bad=copy.deepcopy(fixture);bad[3]['hard_teacher_exposures']+=1
        with self.assertRaises(ValueError):validate_exposure(*bad)
        bad=copy.deepcopy(fixture);bad[5]['groups']['hard_stt']+=1
        with self.assertRaises(ValueError):validate_exposure(*bad)

if __name__ == '__main__':unittest.main()
