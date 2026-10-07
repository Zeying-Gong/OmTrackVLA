import copy
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from torch.utils.data import DataLoader, DistributedSampler

from wa.tools import build_stt_anchor_candidate as anchor
from wa.wm.teacher_window_plan import PlannedTeacherMix, SCHEMA as OLD_SCHEMA, SOURCE_KEYS, simulate_exposure


def fixture():
    records, candidate, ids, ages = [], {}, [], []
    definitions = [('hard1','stt',False,False,7), ('gain','stt',False,True,6),
                   ('reg','stt',True,False,6), ('anchorA','stt',True,True,7),
                   ('anchorB','stt',True,True,6), ('dt','dt',True,True,6),
                   ('zero','stt',False,False,0)]
    for ep, (key, task, old_success, success, n) in enumerate(definitions):
        time = np.arange(n, dtype=float)*.5
        records.append(dict(episode_uid=task+':scene/'+key, task=task, key='scene/'+key,
            teacher='lightnav', student_success=old_success, hard=task=='stt' and not old_success,
            valid_windows=n, early_windows=int((time<=2).sum()),
            selected_teacher_success=True, selected_teacher_fallback=False))
        candidate[task, 'scene/'+key] = dict(success=success)
        ids.extend([ep]*n)
        ages.extend(time.tolist())
    ep = np.array(ids, dtype=np.int32)
    ages = np.array(ages)
    n = len(ep)
    windows = dict(dataset_index=np.arange(n), raw_row=np.arange(n),
        episode_index=ep, current_observation_index=np.arange(n), age_s=ages,
        hard=np.array([records[i]['hard'] for i in ep]), early=ages<=2)
    return records, windows, candidate


def tiny_plan():
    return dict(schema=anchor.SCHEMA, base_count=101, teacher_count=17,
        extra_teacher_indices=[0, 0, 3, 5, 8], source_hashes={k:'a'*64 for k in SOURCE_KEYS},
        selection_source_hashes={k:'b'*64 for k in anchor.SOURCE_SHA})


class AnchorCandidateTest(unittest.TestCase):
    def test_budget_arithmetic(self):
        self.assertEqual(10413+3270, 13683)
        self.assertEqual(1231*5+17*6, 6257)
        self.assertEqual(13683+886+6257, 20826)
        self.assertEqual(726631+436816+20826, 1184273)

    def test_quantiles_use_time_order_and_unique(self):
        ids = np.array([10,20,30,40,50,60])
        ages = np.array([5.,0.,4.,1.,3.,2.])
        self.assertEqual(anchor.time_quantiles(ids, ages, 6), [20,40,60,50,30,10])
        picked = anchor.time_quantiles(ids, ages, 5)
        self.assertEqual(len(set(picked)), 5)
        self.assertEqual(picked[0],20)
        self.assertEqual(picked[-1],10)
        for bad_ids, bad_ages, count in ((ids[:4],ages[:4],5),(ids,np.ones(6),5),
                                        (ids,ages*np.nan,5),(np.array([0,0]),np.array([0.,1.]),2)):
            with self.assertRaises(ValueError):
                anchor.time_quantiles(bad_ids,bad_ages,count)

    def test_cohorts_and_multiplicity_preserve_base_and_tasks(self):
        records, windows, candidate = fixture()
        extras, detail = anchor.allocate(records, windows, candidate, sixth_count=1)
        count = np.bincount(extras,minlength=len(windows['dataset_index']))
        early_hard = windows['hard'] & windows['early']
        self.assertTrue((count[early_hard]==2).all())
        self.assertTrue((count[windows['hard'] & ~windows['early']]==1).all())
        self.assertTrue((count[windows['episode_index']==2]==1).all())
        self.assertTrue((count[windows['episode_index']==5]==0).all())
        self.assertEqual(len(detail['anchor_details']),2)
        self.assertEqual(sorted(d['requested'] for d in detail['anchor_details']),[5,6])
        self.assertEqual(detail['original_gain_episode_indices'],[1])
        self.assertEqual(detail['original_remaining_episode_indices'],[0,6])
        self.assertEqual(count.max(),2)
        self.assertEqual(extras,anchor.allocate(records,windows,candidate,sixth_count=1)[0])

    def test_hash_sixth_assignment_is_stable(self):
        records, windows, candidate=fixture()
        _, detail=anchor.allocate(records,windows,candidate,sixth_count=1)
        order=sorted((3,4),key=lambda i:anchor.hashlib.sha256(
            (anchor.ANCHOR_SALT+records[i]['episode_uid']).encode()).hexdigest())
        self.assertEqual(detail['anchor_episode_indices'],order)
        self.assertEqual(detail['anchor_details'][0]['requested'],6)

    def test_no_silent_short_episode_replacement(self):
        records,windows,candidate=fixture()
        remove=np.flatnonzero(windows['episode_index']==3)[4:]
        keep=np.ones(len(windows['episode_index']),dtype=bool);keep[remove]=False
        windows={k:v[keep] for k,v in windows.items()}
        windows['dataset_index']=np.arange(len(windows['dataset_index']))
        records[3]['valid_windows']=4;records[3]['early_windows']=4
        with self.assertRaisesRegex(ValueError,'too few valid windows'):
            anchor.allocate(records,windows,candidate,sixth_count=0)

    def test_foreign_metadata_failed_teacher_and_invalid_flags_rejected(self):
        for mutation in ('hard','early','valid','fallback','success','outcome','uid'):
            records,windows,candidate=fixture()
            if mutation=='hard':
                windows['hard'][0]=False
            elif mutation=='early':
                windows['early'][0]=False
            elif mutation=='valid':
                records[0]['valid_windows']+=1
            elif mutation=='fallback':
                records[0]['selected_teacher_fallback']=True
            elif mutation=='success':
                records[0]['selected_teacher_success']=False
            elif mutation=='uid':
                records[0]['episode_uid']='stt:other'
            else:
                candidate['stt','scene/hard1']['success']=.3
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                anchor.allocate(records,windows,candidate,sixth_count=1)

    def test_v2_view_exact_positions(self):
        p=tiny_plan();mix=anchor.IndexSimulationView(p)
        self.assertEqual(len(mix),123)
        counts=Counter(mix.locate(i) for i in range(len(mix)))
        self.assertEqual(counts['teacher',0],3)
        self.assertEqual(counts['teacher',3],2)
        self.assertEqual(counts['teacher',1],1)
        self.assertTrue(all(counts['base',i]==1 for i in range(101)))
        for i in (-1,True,len(mix)):
            with self.assertRaises(IndexError):mix.locate(i)

    def test_v1_cannot_accept_or_disguise_new_schedule(self):
        p=tiny_plan()
        with self.assertRaises(ValueError):
            PlannedTeacherMix(range(101),range(17),p,source_hashes=p['source_hashes'],
                              eligible_teacher_indices=range(17))
        old=dict(schema=OLD_SCHEMA,base_count=101,teacher_count=17,extra_repeats=1,
                 extra_teacher_indices=p['extra_teacher_indices'],source_hashes=p['source_hashes'])
        with self.assertRaises(ValueError):
            PlannedTeacherMix(range(101),range(17),old,source_hashes=p['source_hashes'],
                              eligible_teacher_indices=range(17))

    def test_v2_invalid_multiplicity_and_types(self):
        changes=[('extra_teacher_indices',[0,0,0]),('extra_teacher_indices',[3,0]),
                 ('extra_teacher_indices',[True]),('extra_teacher_indices',[-1]),
                 ('extra_teacher_indices',[17]),('base_count',True),
                 ('source_hashes',{}),('selection_source_hashes',{}),('schema',OLD_SCHEMA)]
        for key,value in changes:
            p=tiny_plan();p[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                anchor.IndexSimulationView(p)

    def test_real_cpu_dataloader_matches_existing_index_simulation(self):
        p=tiny_plan()
        class Readable(anchor.IndexSimulationView):
            def __getitem__(self,i):
                src,index=self.locate(i)
                return index if src=='base' else len(self.base)+index
        mix=Readable(p)
        report,teacher=simulate_exposure(mix,world=8,batch=2,seed=42,epoch=1)
        counted=Counter()
        for rank in range(8):
            sampler=DistributedSampler(mix,num_replicas=8,rank=rank,shuffle=True,seed=42,drop_last=True)
            sampler.set_epoch(1)
            for batch in DataLoader(mix,sampler=sampler,batch_size=2,drop_last=True):
                counted.update(batch.tolist())
        self.assertEqual(sum(counted.values()),report['actual_total'])
        self.assertEqual([counted[101+i] for i in range(17)],teacher.tolist())

    def test_output_refuses_existing_and_dangling(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'new';anchor.refuse_output(path)
            path.mkdir()
            with self.assertRaises(ValueError):anchor.refuse_output(path)
            link=Path(tmp)/'link';link.symlink_to(Path(tmp)/'missing',target_is_directory=True)
            with self.assertRaises(ValueError):anchor.refuse_output(link)
            with self.assertRaises(ValueError):anchor.refuse_output(Path(tmp)/'inside',protected=(tmp,))

    def test_rank_group_counts_reconcile(self):
        records,windows,candidate=fixture()
        extras,allocation=anchor.allocate(records,windows,candidate,sixth_count=1)
        p=tiny_plan();p.update(teacher_count=len(windows['dataset_index']),extra_teacher_indices=extras)
        mix=anchor.IndexSimulationView(p)
        report,counts=simulate_exposure(mix,world=8,batch=2,seed=42,epoch=1)
        masks=anchor.masks_for(records,windows,allocation)
        base,rank_teacher,ranks,dropped=anchor.rank_counts(mix,records,windows,masks,report)
        np.testing.assert_array_equal(rank_teacher.sum(0),counts)
        self.assertEqual(sum(r['base'] for r in ranks),base.sum())
        self.assertEqual(len(dropped),report['dropped'])
        self.assertEqual(sum(r['cohorts']['original_hard_early']['total'] for r in ranks),
                         int(counts[windows['hard'] & windows['early']].sum()))


class BinarySuccessTypeTest(unittest.TestCase):
    def test_contract_binary_float_matches_integer(self):
        records, windows, candidate = fixture()
        expected = anchor.allocate(records, windows, candidate, sixth_count=1)
        for row in candidate.values():
            row['success'] = float(row['success'])
        self.assertEqual(anchor.allocate(records, windows, candidate, sixth_count=1), expected)

    def test_nonbinary_or_nonfinite_success_rejected(self):
        for invalid in (float('nan'), float('inf'), -.1, .5, 2.0, '1'):
            records, windows, candidate = fixture()
            next(iter(candidate.values()))['success'] = invalid
            with self.assertRaisesRegex(ValueError, 'invalid candidate outcome'):
                anchor.allocate(records, windows, candidate, sixth_count=1)


if __name__ == '__main__':
    unittest.main()
