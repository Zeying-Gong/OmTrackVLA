import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from wa.tools import failure_state_group_fit as m
from wa.tests.test_replay_teacher_group_fit import reference_fixture


def recovery_rows(episodes=40):
    rows=[]
    for ep in range(episodes):
        for age in (0.,.4,.6,1.,1.2,1.9,2.1,3.):
            i=len(rows)
            rows.append(dict(position=i,new_cache_row=i,new_episode=ep,new_current_index=i+4,
                new_actual_timestamp_s=20.+age,elapsed_after_takeover_s=age,task="stt",
                key=f"scene/{ep}",source_branch=f"/raw/{ep}/oracle_0020",
                source_teacher="oracle",takeover_step=20))
    return rows


def full_selection():
    return [dict(s,source="old88") for s in reference_fixture()["selected"]] + m.recovery_selection(recovery_rows())


def item(small=True):
    size=2 if small else 224
    pose=torch.zeros(7,4);pose[:,3]=1
    return dict(rgb=torch.arange(12*size*size).float().reshape(4,3,size,size),
        template=torch.ones(3,size,size),polar=torch.tensor([2.,.5]),
        times=torch.tensor([-1.5,-1.,-.5,0.]),pose=pose)


def paired_fixture():
    selected=full_selection();samples=[item() for _ in selected]
    class Predictor:
        def predict(self,batch,mode,noise):
            if set(batch)!=set(m.INPUTS):raise AssertionError("label/metadata leaked")
            if not (mode==2).all() or not (noise==0).all():raise AssertionError("inference changed")
            pred=noise.clone();pred[...,3]=1
            return pred,None
    records=m.replay.predict_records(Predictor(),samples,selected,device="cpu")
    return selected,samples,records


def dedup_fixture():
    current=[]
    for i in range(6864):
        ep=6+i%90
        current.append(dict(position=i,new_cache_row=i,new_episode=ep,new_current_index=i+4,
            new_actual_timestamp_s=float(i),elapsed_after_takeover_s=float(i),task="stt",
            key=f"scene/{ep}",source_branch=f"/raw/{ep}",source_teacher="oracle",
            takeover_step=0,nonimage_components={"polar":"a"*64},nonimage_sha256="b"*64))
    windows=[dict(r,status="EXACT_DUPLICATE" if i<526 else "NO_NONIMAGE_MATCH",
                  old_matching_rows=[i] if i<526 else []) for i,r in enumerate(current)]
    release=dict(path="/release.json",sha256="c"*64)
    report=dict(schema="failure_state_exact_dedup_v1",status="PASS_SCOPED_EXACT_EQUIVALENCE",
        release=release,old_cache=dict(path="/old",complete_sha256=m.original.CACHE_SHA),
        training_released=False,cache_modified=False,sampling_changed=False,
        summary=dict(original_episodes=96,candidate_windows=7396,valid_windows=6864,
            exact_duplicate_count=526,deduplicated_new_count_within_same_key_scope=6338),
        windows=windows,duplicate_new_cache_rows=list(range(526)),
        nonduplicate_new_cache_rows_within_scope=list(range(526,6864)))
    return report,current,release


class SelectionTests(unittest.TestCase):
    def test_fixed88_verbatim_before64(self):
        ref=reference_fixture()
        m.replay.validate_reference(ref)
        selected=full_selection()
        self.assertEqual([{k:v for k,v in s.items() if k!="source"} for s in selected[:88]],ref["selected"])
        self.assertEqual(len(selected),152)
        self.assertEqual(dict(m.Counter(s["group"] for s in selected)),m.GROUPS)

    def test_hash_order_not_input_order_or_error_score(self):
        rows=recovery_rows()
        expected=m.recovery_selection(rows)
        for row in rows:row["fit_error"]=999999-row["new_cache_row"]
        self.assertEqual(expected,m.recovery_selection(list(reversed(rows))))
        early=[s for s in expected if s["group"]=="recovery_early"]
        late=[s for s in expected if s["group"]=="recovery_late"]
        self.assertEqual(len({s["episode_uid"] for s in early}),32)
        self.assertEqual({s["episode_uid"] for s in early},{s["episode_uid"] for s in late})
        self.assertEqual(len({s["raw_row"] for s in expected}),64)

    def test_strata_and_nearest_times(self):
        selected=m.recovery_selection(recovery_rows())
        self.assertTrue(all(s["elapsed_after_takeover_s"]==.4 for s in selected[:32]))
        self.assertTrue(all(s["elapsed_after_takeover_s"]==1.9 for s in selected[32:]))
        rows=recovery_rows()
        rows=[r for r in rows if r["elapsed_after_takeover_s"] in (1.,1.2)]
        selected=m.recovery_selection(rows)
        self.assertTrue(all(s["elapsed_after_takeover_s"]==1. for s in selected[:32]))
        self.assertTrue(all(s["elapsed_after_takeover_s"]==1.2 for s in selected[32:]))

    def test_nearest_tie_earlier_index(self):
        rows=recovery_rows()
        for r in rows:
            if r["elapsed_after_takeover_s"] in (.4,.6):
                r["elapsed_after_takeover_s"]=.25 if r["elapsed_after_takeover_s"]==.4 else .75
        self.assertTrue(all(r["elapsed_after_takeover_s"]==.25 for r in m.recovery_selection(rows)[:32]))

    def test_short_stratum_refused_no_relaxation(self):
        with self.assertRaisesRegex(ValueError,"insufficient"):m.recovery_selection(recovery_rows(31))
        with self.assertRaisesRegex(ValueError,"insufficient"):
            m.recovery_selection([r for r in recovery_rows() if r["elapsed_after_takeover_s"]<=1])

    def test_duplicate_rawrow_invalidtime_and_bool_refused(self):
        for field,value in (("elapsed_after_takeover_s",float("nan")),
                            ("elapsed_after_takeover_s",True),("elapsed_after_takeover_s",-1.),
                            ("new_cache_row",True),("new_episode",True)):
            rows=recovery_rows();rows[0][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):m.recovery_selection(rows)
        rows=recovery_rows();rows[1]["new_cache_row"]=rows[0]["new_cache_row"]
        with self.assertRaises(ValueError):m.recovery_selection(rows)

    def test_all6864_bound_exact526_excluded(self):
        report,current,release=dedup_fixture()
        got=m.validate_dedup(report,current,release,Path("/old"))
        self.assertEqual(len(got),6338)
        self.assertFalse({r["new_cache_row"] for r in got}&set(range(526)))

    def test_dedup_rejects_every_identity_or_scalar_mismatch(self):
        for field,value in (("new_cache_row",99999),("new_episode",False),
                           ("new_current_index",999),("takeover_step",True),
                           ("new_actual_timestamp_s",999.),("task","at"),("key","other/1"),
                           ("source_branch","/repeat"),("source_teacher","lightnav"),
                           ("nonimage_components",{}),("nonimage_sha256","e"*64)):
            report,current,release=dedup_fixture();report["windows"][1000][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                m.validate_dedup(report,current,release,Path("/old"))

    def test_dedup_rejects_short_duplicate_unknown_or_resigned_sets(self):
        for mutate in (
            lambda d:d["windows"].pop(),
            lambda d:d["windows"].__setitem__(1,d["windows"][0]),
            lambda d:d["windows"][1000].update(status="SKIPPED"),
            lambda d:d["windows"][1000].update(old_matching_rows=[4]),
            lambda d:d["duplicate_new_cache_rows"].pop(),
            lambda d:d.update(training_released=True),
            lambda d:d["summary"].update(exact_duplicate_count=True),
            lambda d:d["release"].update(sha256="e"*64),
            lambda d:d["old_cache"].update(complete_sha256="f"*64)):
            report,current,release=dedup_fixture();release=copy.deepcopy(release);mutate(report)
            with self.assertRaises(ValueError):m.validate_dedup(report,current,release,Path("/old"))

    def test_loader_gate_not_converter_status_or_different_admission(self):
        audit=dict(schema="failure_state_real_loader_audit_v1",status=m.LOADER_STATUS,
            cache="/cache",admission_sha256="a"*64,
            loader_admission=dict(status="LOADER_ADMISSION_PASS_NOT_NEW_MODEL_RESULT",
                summary=m.EXPECTED,admission_sha256="a"*64,actual_consumed_sources_hashed=True,
                training_released_on_disk=False),
            selection=dict(all_valid_rows_inspected=6864,nonzero_episodes=90))
        m.validate_loader_gate(audit,"/cache","a"*64)
        for mutate in (lambda x:x.update(status="CACHE_CONVERTED_NOT_TRAINING_RELEASED"),
                       lambda x:x.update(cache="/other"),
                       lambda x:x["loader_admission"].update(actual_consumed_sources_hashed=False),
                       lambda x:x["loader_admission"].update(training_released_on_disk=True),
                       lambda x:x["selection"].update(all_valid_rows_inspected=100)):
            d=copy.deepcopy(audit);mutate(d)
            with self.assertRaises(ValueError):m.validate_loader_gate(d,"/cache","a"*64)

    def test_sample_only_five_allowed_fields_and_actual_shapes(self):
        source=item(False);source.update(future=torch.ones(3,224,224),metadata=torch.tensor(42))
        sample=m.sample(source,source["pose"].numpy())
        self.assertEqual(set(sample),set(m.INPUTS)|{"pose"})
        for field in m.SHAPES:
            bad=dict(source);bad[field]=source[field].double()
            with self.subTest(field=field),self.assertRaises(ValueError):m.sample(bad,source["pose"].numpy())
        with self.assertRaises(ValueError):m.sample(source,np.ones((7,4),np.float32))

    def test_physical_rows_uses_actual_time_and_current_polar(self):
        obs=[dict(timestamp_s=i*.1,robot_position_world=[i*.01,0,0],
                  robot_rotation_world_from_body=np.eye(3).tolist(),
                  target_position_world_label_only=[2,0,1]) for i in range(20)]
        class Data:
            episodes=[dict(task="stt",episode_uid="stt:s/0",teacher="oracle",root="/r",takeover_step=4)]
            rows=np.array([0,1]);episode=np.array([0,0]);history=np.array([[0,0,0,6],[0,0,2,8]])
            pose_labels=np.zeros((2,7,4),np.float32)
            def episode_info(self,ep):return self.episodes[ep],Path("/r"),obs,np.arange(20)*.1,None
            def transitions(self,ep):return np.zeros((19,4),np.float32),np.zeros(19,bool)
        rows=m.physical_rows(Data())
        self.assertAlmostEqual(rows[0]["elapsed_after_takeover_s"],.2)
        self.assertEqual(rows[0]["new_current_index"],6)
        before=rows[0]["nonimage_components"]["polar"]
        obs[19]["target_position_world_label_only"]=[9,0,9]
        self.assertEqual(m.physical_rows(Data())[0]["nonimage_components"]["polar"],before)
        obs[6]["target_position_world_label_only"]=[9,0,9]
        self.assertNotEqual(m.physical_rows(Data())[0]["nonimage_components"]["polar"],before)


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selected,cls.samples,cls.records=paired_fixture()
        cls.fingerprints=[m.fingerprint(s) for s in cls.samples]

    def test_all304_full_predictions_recomputed_metrics(self):
        bykey=m.validate_records(self.records,self.selected,self.fingerprints)
        self.assertEqual(len(bykey),304)
        self.assertTrue(all(v["ADE_m"]==0 for v in bykey.values()))
        for i in range(152):
            self.assertEqual(bykey[i,False]["label_7x4"],bykey[i,True]["label_7x4"])
            self.assertEqual(bykey[i,False]["input_tensor_sha256"]["polar"],
                             bykey[i,True]["input_tensor_sha256"]["polar"])
            self.assertNotEqual(bykey[i,False]["input_tensor_sha256"]["rgb"],
                                bykey[i,True]["input_tensor_sha256"]["rgb"])

    def test_invalid_prediction_or_invented_metric_rejected(self):
        for mutation in (
            lambda r:r[0].update(ADE_m=42.),
            lambda r:r[0].update(prediction_7x4=[[0]*4]*6),
            lambda r:r[0].update(label_tensor_sha256="0"*64),
            lambda r:r[0]["input_tensor_sha256"].update(polar="0"*64),
            lambda r:next(row for row in r if row["selection_index"]==0 and row["repeat_history"]).update(repeat_history=False),
            lambda r:r[0].update(selection_index=True),
            lambda r:r.pop()):
            rows=copy.deepcopy(self.records);mutation(rows)
            with self.assertRaises(ValueError):m.validate_records(rows,self.selected,self.fingerprints)

    def test_crossmodel_compare_requires_identical_inputs(self):
        result=m.compare(self.records,self.records,self.selected,self.fingerprints)
        self.assertEqual(len(result["records"]),304)
        self.assertTrue(all(r["ADE_m"]==0 for r in result["records"]))
        bad=copy.deepcopy(self.records);next(row for row in bad if row["selection_index"]==0 and row["repeat_history"])["input_tensor_sha256"]["rgb"]="0"*64
        with self.assertRaisesRegex(ValueError,"cross-model"):
            m.compare(bad,self.records,self.selected,self.fingerprints)

    def test_parent_exact_loaded_contract_and_strict_state(self):
        model=SimpleNamespace(load_state_dict=lambda *a,**k:
                              SimpleNamespace(missing_keys=["encoder.x"],unexpected_keys=[]))
        ck=dict(step=22707,kind="jepa",contract=m.CONTRACT,completed_epochs=1,
                model={"policy.x":torch.zeros(1)})
        reader=Mock(return_value=ck);m.load_parent(model,"/p",loader=reader)
        reader.assert_called_once_with("/p",map_location="cpu",weights_only=True,mmap=True)
        for key,value in (("step",59716),("completed_epochs",True),("kind","behavior"),
                          ("model",{}),("model",{"encoder.x":torch.zeros(1)})):
            with self.subTest(key=key),self.assertRaises(ValueError):
                m.load_parent(model,"/p",loader=lambda *a,**kw:dict(ck,**{key:value}))

    def test_candidate_requires_explicit_sha_step(self):
        for digest,step in (("",59716),("f"*64,True),("f"*64,-1),("X"*64,10)):
            with self.assertRaises(ValueError):m.candidate_args("/p",digest,step,"/a","c"*64)

    def test_candidate_audit_fails_before_gpu_or_model(self):
        args=SimpleNamespace(selection="/s",selection_sha256="a"*64,model="candidate",
            candidate_checkpoint="/checkpoint",candidate_sha256="b"*64,candidate_step=59716,
            training_audit="/a",training_audit_sha256="c"*64,baseline_fit="/baseline",
            baseline_fit_sha256="d"*64)
        selected=dict(context=dict(root="/root",job_root="/jobs"))
        with patch.object(m,"read_selection",return_value=(selected,[],{})), \
             patch.object(m.replay,"candidate_inputs",side_effect=ValueError("audit incomplete")), \
             patch.object(m,"JointRobotModel",side_effect=AssertionError("model must not load")), \
             patch.object(torch.cuda,"device_count",side_effect=AssertionError("no GPU")):
            with self.assertRaisesRegex(ValueError,"audit incomplete"):m.run(args)

    def test_output_does_not_overwrite_or_follow_symlink(self):
        with TemporaryDirectory() as t:
            p=Path(t)/"o.json";m.replay.write_report(p,{"a":1})
            with self.assertRaises(ValueError):m.replay.write_report(p,{"b":2})
            q=Path(t)/"link";q.symlink_to(Path(t)/"missing")
            with self.assertRaises(ValueError):m.replay.write_report(q,{})

    def test_fixed_dedup_and_reference_sha_not_caller_chosen(self):
        self.assertEqual(m.DEDUP_SHA,"599d60ea2d40ff4f9ae69f87df03e3a5a192b749ce030504bf19177999262859")
        self.assertEqual(m.replay.REFERENCE_SHA,"1463ea39a6fe1acf90fcbd813d1ae23bee6ee0e475a314e5bde174b3fa4f8158")
        self.assertEqual(m.BEST_SHA,"c510c04d987d141423401a25323ea353314107a16f4dac5d2901370ab199fa52")


if __name__=="__main__":unittest.main()
