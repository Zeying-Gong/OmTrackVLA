import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock,patch

import numpy as np
import torch
from torch.utils.data import DistributedSampler

from wa.tools import audit_stt_anchor_training as audit
from wa.tests.test_audit_hard_stt_training import HardTrainingAuditTest
from wa.wm.runtime_exposure import RuntimeExposure
from wa.wm.teacher_window_schedule import ScheduledTeacherMix,SCHEMA
from wa.wm.teacher_window_plan import SOURCE_KEYS,simulate_exposure
from wa.tools import build_stt_anchor_candidate as recipe

class AnchorTrainingAuditTest(unittest.TestCase):
    def fixture(self):
        groups=dict(hard_stt_early=[0],hard_stt_late=[1],new_regression_stt=[2],successful_stt_anchor=[3,4])
        extra=[0,0,1,2,3,4]
        plan=dict(schema=SCHEMA,base_count=101,teacher_count=17,extra_teacher_indices=extra,
            source_hashes={k:'a'*64 for k in SOURCE_KEYS},selection_source_hashes={k:'b'*64 for k in recipe.SOURCE_SHA})
        mix=ScheduledTeacherMix(range(101),range(17),plan,source_hashes=plan['source_hashes'],
            selection_source_hashes=plan['selection_source_hashes'],expected_extra_teacher_indices=extra,sampling_groups=groups)
        records=[dict(episode_uid='stt:x/1',task='stt',teacher='oracle')]
        windows=dict(episode_index=np.zeros(17,dtype=np.int64),hard=np.arange(17)<2,early=np.arange(17)%2==0)
        counter=RuntimeExposure(mix,episodes=records,**windows)
        for rank in range(8):
            sampler=DistributedSampler(mix,num_replicas=8,rank=rank,seed=42,drop_last=True);sampler.set_epoch(1)
            ids=list(sampler);counter.consume(torch.tensor(ids[:len(ids)//2*2]))
        actual=counter.finalize();actual.update(optimizer_steps_completed=audit.UPDATES,diagnostic=False)
        base,teacher=counter.source_counts()
        arrays=dict(position_counts=counter.state()['position_counts'].numpy(),base_counts=base.numpy(),teacher_counts=teacher.numpy())
        simulated,_=simulate_exposure(mix)
        return mix,records,windows,actual,arrays,simulated

    def test_actual_exact_positions_and_four_groups(self):
        f=self.fixture();result,base,teacher=audit.recorded_exposure(*f)
        np.testing.assert_array_equal(base.sum(0),f[4]['base_counts'])
        np.testing.assert_array_equal(teacher.sum(0),f[4]['teacher_counts'])
        self.assertEqual(result['sampling_group_unique_windows'],dict(hard_stt_early=1,
            hard_stt_late=1,new_regression_stt=1,successful_stt_anchor=2))
        for key in ('position_counts','base_counts','teacher_counts'):
            bad=copy.deepcopy(f);bad[4][key][0]+=1
            with self.assertRaises(ValueError):audit.recorded_exposure(*bad)
        for key in ('diagnostic','optimizer_steps_completed','plan_sha256','sampling_group_exposures'):
            bad=copy.deepcopy(f);bad[3][key]={'diagnostic':True,'optimizer_steps_completed':1,
                'plan_sha256':'a'*64,'sampling_group_exposures':{}}[key]
            with self.assertRaises(ValueError):audit.recorded_exposure(*bad)

    def test_histogram_not_enough_and_wrong_seed_rejected(self):
        f=self.fixture();bad=copy.deepcopy(f)
        a=bad[4]['position_counts'];one=np.flatnonzero(a==1)[0];zero=np.flatnonzero(a==0)[0]
        a[one],a[zero]=a[zero],a[one]
        with self.assertRaises(ValueError):audit.recorded_exposure(*bad)
        for value in (7,0,True):
            bad=copy.deepcopy(f);bad[5]['seed']=value
            with self.assertRaises(ValueError):audit.recorded_exposure(*bad)

    def test_actual_arrays_strict_dtype_shape(self):
        for key in ('position_counts','base_counts','teacher_counts'):
            f=self.fixture();f[4][key]=f[4][key].astype(float)
            with self.assertRaises(ValueError):audit.recorded_exposure(*f)
            f=self.fixture();f[4][key]=f[4][key][:-1]
            with self.assertRaises(ValueError):audit.recorded_exposure(*f)

    def test_no_checkpoint_hash_or_load_until_terminal_metrics(self):
        with TemporaryDirectory() as d:
            root=Path(d);(root/'checkpoint.pt').write_bytes(b'active')
            loader=Mock(side_effect=AssertionError('checkpoint must not load'))
            with patch.object(audit.previous,'verified_sha',side_effect=AssertionError('must not hash')):
                with self.assertRaises(audit.IncompleteTraining):
                    audit.audit_training(root,root,root/'x.yaml',root,source_commit='a'*40,
                                         config_sha256='b'*64,checkpoint_loader=loader)
            loader.assert_not_called()

    def test_full_final_metrics_before_checkpoint(self):
        with TemporaryDirectory() as d:
            root=Path(d)
            for name in audit.previous.TERMINAL:(root/name).write_text('{}')
            metrics=HardTrainingAuditTest().metrics();metrics['metrics']['mixed']['windows']=16
            (root/'metrics.json').write_text(json.dumps(metrics))
            loader=Mock()
            with patch.object(audit.previous,'verified_sha',side_effect=AssertionError('must not hash')):
                with self.assertRaises(ValueError):
                    audit.audit_training(root,root,root/'x.yaml',root,source_commit='a'*40,
                                         config_sha256='b'*64,checkpoint_loader=loader)
            loader.assert_not_called()

    def config_fixture(self):
        config,yaml=HardTrainingAuditTest().config_fixture()
        old=audit.previous.CANDIDATE_SHA
        config.update(teacher_plan_report_sha256=audit.REPORT_SHA,precision='fp32 training / bf16 frozen encoder')
        yaml['tasks'][0]['cmd']=yaml['tasks'][0]['cmd'].replace(old,audit.REPORT_SHA)
        return config,yaml

    def test_explicit_release_config_parent_and_schedule(self):
        c,y=self.config_fixture()
        args=(Path('/frozen'),Path('/candidate'),Path('/run'),audit.previous.SOURCE_COMMIT)
        audit.validate_config(c,y,*args)
        for key,value in [('resume','selection61609.pt'),('resume_sha256',recipe.CHECKPOINT_SHA),
            ('completed_epochs',2),('epochs',2),('teacher_plan_report_sha256',audit.previous.CANDIDATE_SHA),
            ('world_size',1),('world_weight',.2),('precision','bf16')]:
            bad=copy.deepcopy(c);bad[key]=value
            with self.assertRaises(ValueError):audit.validate_config(bad,y,*args)
        bad=copy.deepcopy(y);bad['tasks'][0]['cmd']+=' --epochs 3'
        with self.assertRaises(ValueError):audit.validate_config(c,bad,*args)
        with self.assertRaises(ValueError):audit.validate_config(c,y,*args[:-1],'short')

    def test_worker_all_python_hashes_and_real_gpu_environment(self):
        hashes={'wa/wm/train.py':'a'*64,'wa/tools/audit_stt_anchor_training.py':'b'*64}
        env=dict(commit='c'*40,dirty='',torch='2.8.0+cu128',gpu_names=['NVIDIA A800-SXM4-80GB']*8,
            source_sha256=hashes,wla=dict(strict_loaded=['action_expert','metaquery','target_head'],
            omitted=['Qwen backbone','language LoRA'],action_contract='7x4 XY/sin(yaw)/cos(yaw)',source_files={'x':'d'*64}))
        audit.validate_environment(env,hashes,'c'*40)
        for key,value in [('commit','d'*40),('dirty',' M train.py'),('torch','2.7'),
            ('gpu_names',['RTX4090']*8),('gpu_names',['A800']*7),('source_sha256',{})]:
            bad=copy.deepcopy(env);bad[key]=value
            with self.assertRaises(ValueError):audit.validate_environment(bad,hashes,'c'*40)
        bad=copy.deepcopy(env);bad['wla']['strict_loaded']=['action_expert']
        with self.assertRaises(ValueError):audit.validate_environment(bad,hashes,'c'*40)

    def checkpoints(self):
        lrs=[1e-7]*5
        parent=dict(step=audit.START,kind='jepa',contract=audit.CONTRACT,model={'head':torch.zeros(1)},
            optimizer=dict(param_groups=[dict(params=[i],lr=lrs[i],weight_decay=.01) for i in range(5)],
                state={i:dict(step=torch.tensor(float(audit.START)),exp_avg=torch.zeros(1),
                              exp_avg_sq=torch.zeros(1)) for i in range(5)}))
        cp=copy.deepcopy(parent);cp.update(step=audit.FINAL,completed_epochs=2,
            parent_checkpoint=audit.PARENT_PATH,parent_sha256=audit.PARENT_SHA)
        for group in cp['optimizer']['param_groups']:group['lr']*=audit.lr_multiplier(audit.UPDATES)
        for state in cp['optimizer']['state'].values():state['step']=torch.tensor(float(audit.FINAL))
        return parent,cp,dict(base_lrs=lrs)

    def test_parent_optimizer_true_continuation_and_unchanged_groups(self):
        fixture=self.checkpoints();audit.validate_parent_continuation(*fixture)
        for which,key,value in [(0,'step',59065),(1,'completed_epochs',3),(1,'parent_sha256',recipe.CHECKPOINT_SHA)]:
            bad=copy.deepcopy(fixture);bad[which][key]=value
            with self.assertRaises(ValueError):audit.validate_parent_continuation(*bad)
        for field,value in [('lr',1e-4),('weight_decay',.1),('params',[100])]:
            bad=copy.deepcopy(fixture);bad[1]['optimizer']['param_groups'][0][field]=value
            with self.assertRaises(ValueError):audit.validate_parent_continuation(*bad)
        bad=copy.deepcopy(fixture);bad[2]['base_lrs'][0]*=10
        with self.assertRaises(ValueError):audit.validate_parent_continuation(*bad)
        bad=copy.deepcopy(fixture);bad[1]['optimizer']['state'][0]['step']=torch.tensor(1.)
        with self.assertRaises(ValueError):audit.validate_parent_continuation(*bad)
        bad=copy.deepcopy(fixture);bad[1]['model']['head']=torch.zeros(2)
        with self.assertRaises(ValueError):audit.validate_parent_continuation(*bad)

    def test_loss_lr_sparse_logging_and_no_epoch3(self):
        lrs=[1e-7]*5
        rows=[HardTrainingAuditTest().log(step) for step in (22708,22725,59700)]
        for row in rows:
            row['loss']=row['flow']+.5*row['geometry']+.1*row['world']
            row['lr']=[x*audit.lr_multiplier(row['phase_step']) for x in lrs]
        audit.validate_lr_loss_logs(rows,lrs)
        for bad in (copy.deepcopy(rows),copy.deepcopy(rows)):
            bad[0]['lr'][0]*=10
            with self.assertRaises(ValueError):audit.validate_lr_loss_logs(bad,lrs)
        bad=copy.deepcopy(rows);bad[1]['loss']+=.1
        with self.assertRaises(ValueError):audit.validate_lr_loss_logs(bad,lrs)
        for update in (0,audit.UPDATES+1,True):
            with self.assertRaises(ValueError):audit.lr_multiplier(update)

    def test_pins_and_output_symlink_rejection(self):
        for value in ('a'*39,'g'*40,True,None):
            with self.assertRaises(ValueError):audit.pin(value,40,'pin')
        with TemporaryDirectory() as d:
            p=Path(d)/'dangling';p.symlink_to(Path(d)/'missing')
            with self.assertRaisesRegex(ValueError,'existing output'):audit.output_path(p,[])

if __name__=='__main__':unittest.main()
