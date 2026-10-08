"""Synthetic file-graph tests for the isolated adapter, not real cache admission."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from wa.wm import failure_state_sampling_runtime as mod
from wa.wm.teacher_window_plan import PlannedTeacherMix, SOURCE_KEYS


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,allow_nan=False)+"\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ToyData:
    def __init__(self,cache,rows=None):
        self.cache=cache
        self.pose=np.load(cache/"train_pose.npy")
        self.history=np.load(cache/"train_history.npy")
        self.episode=np.load(cache/"train_episode.npy")
        self.episodes=json.loads((cache/"train_episodes.json").read_text())
        self.rows=np.arange(len(self.pose),dtype=np.int64) if rows is None else rows

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        return {name:torch.tensor([float(index)],dtype=torch.float32) for name in mod.TENSORS}


class ToyBase(ToyData):pass
class ToyTeacher(ToyData):pass
class ToyRecovery(ToyData):pass


def cache(root,n,episodes=None,history=None,episode=None,rows=None):
    root.mkdir(parents=True)
    np.save(root/"train_pose.npy",np.arange(n*28,dtype=np.float32).reshape(n,7,4)/100.)
    np.save(root/"train_history.npy",np.asarray(history if history is not None else [[0,1,2,4]]*n,dtype=np.int32))
    np.save(root/"train_episode.npy",np.asarray(episode if episode is not None else [0]*n,dtype=np.int32))
    save(root/"train_episodes.json",episodes if episodes is not None else [dict(root=str(root/"raw"),frames=[])])
    files={p.name:sha(p) for p in root.glob("train_*")}
    save(root/"complete.json",dict(files=files))
    (root/"index").mkdir()
    np.save(root/"index/train_valid.npy",np.arange(n,dtype=np.int64) if rows is None else rows)
    save(root/"index/audit.json",dict(splits=dict(train=dict(index_sha256=sha(root/"index/train_valid.npy")))))
    save(root/"index/train_episodes_audit.json",dict(stats=[]))
    return root


def fixture(root):
    artifacts=root/"artifacts";artifacts.mkdir()
    base_root=cache(artifacts/"base",8);teacher_root=cache(artifacts/"old",5)
    base,teacher=ToyBase(base_root),ToyTeacher(teacher_root)
    plan_root=artifacts/"oldplan";plan_root.mkdir()
    hashes={k:str(i+1)*64 for i,k in enumerate(sorted(SOURCE_KEYS))}
    oldplan=dict(schema="wa.teacher-window-plan.v1",base_count=8,teacher_count=5,
                 extra_repeats=1,extra_teacher_indices=[0,1,2,3],source_hashes=hashes)
    mix=PlannedTeacherMix(base,teacher,oldplan,source_hashes=hashes,eligible_teacher_indices=[0,1,2,3])
    save(plan_root/"plan.json",oldplan)
    save(plan_root/"episodes.json",[])
    np.savez(plan_root/"teacher_windows.npz",index=np.arange(5))
    save(plan_root/"exposure.json",dict(fixture=True))
    artifacts_hash={name:sha(plan_root/name) for name in ("plan.json","episodes.json","teacher_windows.npz","exposure.json")}
    save(plan_root/"report.json",dict(status="CANDIDATE_ONLY_NOT_TRAINING_RELEASE",
        plan_canonical_sha256=mix.plan_sha256,artifacts=artifacts_hash))
    run=artifacts/"oldrun";run.mkdir()
    position=np.ones(17,dtype=np.int64);position[12]=0
    np.savez(run/"actual_exposure_epoch1.npz",position_counts=position,
             base_counts=np.ones(8,dtype=np.int64),teacher_counts=np.array([2,2,2,2,0],dtype=np.int64))
    save(run/"actual_exposure_epoch1.json",dict(diagnostic=False,world_size=8,batch_size=2,seed=42,epoch=1,
        plan_sha256=mix.plan_sha256,pre_ddp_total=17,actual_total=16,actual_base=8,actual_teacher=8))
    save(run/"config.json",dict(cache=str(base_root),dual_teacher_cache=str(teacher_root),
        index_root=str(base_root/"index"),index_audit_sha256=sha(base_root/"index/audit.json")))
    trusted={str(p):sha(p) for folder in (plan_root,run) for p in folder.iterdir() if p.is_file()}
    for c in (base_root,teacher_root):
        for name in ("complete.json","index/audit.json"):
            trusted[str(c/name)]=sha(c/name)
    oldaudit=artifacts/"trusted.json"
    save(oldaudit,dict(status="TRAINING_ARTIFACT_AUDIT_PASS_OFFLINE_ONLY",source_hashes=trusted))
    demos,entries,obsdocs=[],[],[]
    currents=[[4,5,6,7],[13,14],[4]]
    validsets=[[4,6,7],[14],[]]
    for eid in range(3):
        branch=artifacts/f"raw{eid}";branch.mkdir()
        obs=[dict(timestamp_s=i*.1,frame=f"rgb_{i:04d}.png") for i in range(16)]
        save(branch/"observations.json",obs);obsdocs.append(obs)
        source=dict(source_experiment="fixture",source_protocol_sha256="e"*64,source_job_id=1,source_task_id=2)
        demo=dict(branch=str(branch),repeat_branch=str(branch)+"_repeat",task="stt",key=f"scene/{eid}",
            teacher="oracle",takeover_step=4 if eid!=1 else 2,window_indices=currents[eid],
            valid_window_indices=validsets[eid],candidate_windows=len(currents[eid]),
            expected_valid_count=len(validsets[eid]),hashes={"observations.json":sha(branch/"observations.json")},**source)
        demos.append(demo)
        entries.append(dict(root=str(branch),episode_uid="stt:"+demo["key"],teacher="oracle",
            takeover_step=demo["takeover_step"],template_index=0,repeat_frames_referenced=False,
            frames=[o["frame"] for o in obs],expected_valid_count=len(validsets[eid]),**source))
    rows=np.array([0,2,3,5],dtype=np.int64)
    recroot=cache(artifacts/"recovery",7,entries,history=[[0,1,2,x] for group in currents for x in group],
                  episode=[0,0,0,0,1,1,2],rows=rows)
    release=artifacts/"release.json";save(release,dict(teacher_demonstrations=demos))
    ref=dict(path=str(release),sha256=sha(release))
    filenames=("complete.json","index/audit.json","index/train_valid.npy","index/train_episodes_audit.json")
    admission=dict(training_released=False,collection_release=ref,files={n:sha(recroot/n) for n in filenames})
    save(recroot/"admission.json",admission)
    recovery=ToyRecovery(recroot,rows)
    recovery.admission_sha256=sha(recroot/"admission.json")
    recovery.admission_report=dict(status="LOADER_ADMISSION_PASS_NOT_NEW_MODEL_RESULT",release_sha256=ref["sha256"])
    windows=[];offset=0
    for eid,d in enumerate(demos):
        for j,current in enumerate(d["window_indices"]):
            raw=offset+j
            if raw not in rows:continue
            windows.append(dict(new_cache_row=raw,new_episode=eid,task="stt",key=d["key"],
                source_branch=d["branch"],source_teacher=d["teacher"],takeover_step=d["takeover_step"],
                new_current_index=current,new_actual_timestamp_s=obsdocs[eid][current]["timestamp_s"],
                status="EXACT_DUPLICATE" if raw==2 else "NO_NONIMAGE_MATCH",
                nonimage_components=dict(pose=mod.tensor_digest(recovery.pose[raw]))))
        offset+=len(d["window_indices"])
    dedup=artifacts/"dedup.json"
    save(dedup,dict(schema="failure_state_exact_dedup_v1",status="PASS_SCOPED_EXACT_EQUIVALENCE",
        training_released=False,sampling_changed=False,cache_modified=False,release=ref,windows=windows,
        duplicate_new_cache_rows=[2],nonduplicate_new_cache_rows_within_scope=[0,3,5],
        source_files={str(Path(d["branch"])/"observations.json"):d["hashes"]["observations.json"] for d in demos},
        old_cache=dict(path=str(teacher_root),complete_sha256=sha(teacher_root/"complete.json")),
        old_trust_anchor=dict(path=str(oldaudit),sha256=sha(oldaudit)),scope="same task/key only"))
    pins=dict(old_plan=sha(plan_root/"plan.json"),old_actual_exposure=sha(run/"actual_exposure_epoch1.npz"),
              recovery_admission=recovery.admission_sha256,dedup_report=sha(dedup))
    fixed=dict(base=8,teacher=5,old_positions=17,old_executed=16,old_teacher_exposures=8,
               candidate=7,valid=4,episodes=3,duplicate=1,new=3,recovery_budget=16,omitted_position=12)
    return dict(mix=mix,recovery=recovery,old_plan_root=plan_root,old_run_root=run,dedup_path=dedup,
                source_pins=pins,anchor=oldaudit,release=release,root=root,fixed=fixed)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.f=fixture(Path(self.tmp.name))
        self.addCleanup(patch.stopall)
        for key,value in dict(PROJECT=self.f["root"],OLD_AUDIT=self.f["anchor"],OLD_AUDIT_SHA=sha(self.f["anchor"]),
            RELEASE_SHA=sha(self.f["release"]),FIXED=self.f["fixed"],
            RobotWorldData=ToyBase,DualTeacherData=ToyTeacher,FailureStateData=ToyRecovery).items():
            patch.object(mod,key,value).start()

    def kwargs(self):
        return {k:self.f[k] for k in ("old_plan_root","old_run_root","dedup_path","source_pins")}

    def prepare(self):
        return mod.prepare_candidate(self.f["mix"],self.f["recovery"],**self.kwargs())

    def saved(self):
        plan,proof=self.prepare();out=self.f["root"]/"artifacts/candidate"
        result=mod.save_candidate(out,plan,proof)
        return out,result

    def load(self,out,result):
        return mod.load_candidate(out,result["admission_sha256"],self.f["mix"],self.f["recovery"],**self.kwargs())

    def test_true_raw_to_local_dataset_mapping_and_age(self):
        plan,proof=self.prepare();r=plan["metadata"]["recovery_windows"]
        self.assertEqual([x["raw_row"] for x in r],[0,2,3,5])
        self.assertEqual([x["dataset_index"] for x in r],[0,1,2,3])
        self.assertEqual([x["exact_duplicate"] for x in r],[False,True,False,False])
        self.assertEqual(r[0]["current_age_s"],0.)
        self.assertAlmostEqual(r[3]["current_age_s"],1.2)
        self.assertEqual(plan["arrays"]["recovery_counts"][1],0)
        self.assertEqual(proof["recovery"]["zero_valid_episodes"],["stt:scene/2"])
        self.assertTrue(proof["all_valid_pose_hashes_checked"])
        self.assertFalse(proof["training_released"])

    def test_roundtrip_external_pins_and_no_metadata_policy(self):
        out,result=self.saved();mix,proof,sim=self.load(out,result)
        self.assertEqual(len(mix),32)
        self.assertEqual(sim["dropped"],0)
        for i in (0,8,16,31):
            item=mix[i]
            self.assertEqual(set(item),mod.TENSORS)
            code,idx=mix.locate(i)
            expected=mix._datasets[code][idx]
            for key in mod.TENSORS:self.assertTrue(torch.equal(item[key],expected[key]))
        self.assertFalse(proof["training_released"])

    def test_save_is_exclusive(self):
        out,result=self.saved()
        plan,proof=self.prepare()
        with self.assertRaises(ValueError):mod.save_candidate(out,plan,proof)

    def test_npz_or_manifest_or_extra_file_tamper_rejected(self):
        out,result=self.saved()
        with (out/"positions.npz").open("ab") as f:f.write(b"x")
        with self.assertRaisesRegex(ValueError,"source SHA mismatch"):self.load(out,result)

    def test_manifest_unpinned_and_extra_file_rejected(self):
        out,result=self.saved();(out/"unknown.txt").write_text("unknown")
        with self.assertRaises(ValueError):self.load(out,result)

    def test_wrong_external_pin_rejected(self):
        self.f["source_pins"]["old_actual_exposure"]="a"*64
        with self.assertRaisesRegex(ValueError,"not anchored"):self.prepare()

    def test_old_audit_and_actual_file_mutations_rejected(self):
        p=self.f["old_run_root"]/"actual_exposure_epoch1.npz"
        with p.open("ab") as f:f.write(b"tamper")
        with self.assertRaisesRegex(ValueError,"source SHA mismatch"):self.prepare()

    def test_mutated_old_extra_sequence_is_not_same_plan(self):
        self.f["mix"].extra=tuple(reversed(self.f["mix"].extra))
        with self.assertRaisesRegex(ValueError,"persisted plan differ"):self.prepare()

    def test_old_runtime_rows_changed_rejected(self):
        self.f["mix"].teacher.rows=self.f["mix"].teacher.rows[::-1]
        with self.assertRaisesRegex(ValueError,"row mapping changed"):self.prepare()

    def test_new_loader_rows_not_candidate_rows(self):
        self.f["recovery"].rows=np.array([0,1,2,3],dtype=np.int64)
        with self.assertRaisesRegex(ValueError,"loader rows changed"):self.prepare()

    def test_new_dataset_pose_must_equal_physical_admitted_array(self):
        self.f["recovery"].pose[0,0,0]+=1
        with self.assertRaisesRegex(ValueError,"dataset arrays differ"):self.prepare()

    def test_each_dedup_pose_hash_checked_even_zero_duplicate_exposure(self):
        p=self.f["dedup_path"];d=json.loads(p.read_text())
        next(w for w in d["windows"] if w["status"]=="EXACT_DUPLICATE")["nonimage_components"]["pose"]="0"*64
        save(p,d);self.f["source_pins"]["dedup_report"]=sha(p)
        with self.assertRaisesRegex(ValueError,"all valid cached SE2"):self.prepare()

    def test_dedup_row_coverage_exclusion_rejected(self):
        p=self.f["dedup_path"];d=json.loads(p.read_text());d["windows"][0]["new_cache_row"]=1
        save(p,d);self.f["source_pins"]["dedup_report"]=sha(p)
        with self.assertRaisesRegex(ValueError,"row IDs"):self.prepare()

    def test_dedup_other_old_cache_rejected(self):
        p=self.f["dedup_path"];d=json.loads(p.read_text());d["old_cache"]["path"]="/other/cache"
        save(p,d);self.f["source_pins"]["dedup_report"]=sha(p)
        with self.assertRaisesRegex(ValueError,"comparison source differs"):self.prepare()

    def test_takeover_age_is_rederived_not_dedup_supplied(self):
        p=self.f["dedup_path"];d=json.loads(p.read_text());d["windows"][0]["current_age_s"]=42.
        save(p,d);self.f["source_pins"]["dedup_report"]=sha(p)
        plan,_=self.prepare()
        self.assertEqual(plan["metadata"]["recovery_windows"][0]["current_age_s"],0.)

    def test_repeat_source_identity_rejected(self):
        p=self.f["dedup_path"];d=json.loads(p.read_text());d["windows"][0]["source_branch"]+="_repeat"
        save(p,d);self.f["source_pins"]["dedup_report"]=sha(p)
        with self.assertRaisesRegex(ValueError,"source identity"):self.prepare()

    def test_serialized_source_change_after_load_fails_at_getitem(self):
        out,result=self.saved();mix,_,_=self.load(out,result)
        (out/"plan.json").write_text("{}")
        with self.assertRaisesRegex(ValueError,"source changed"):mix[0]

    def test_runtime_exposure_api_global_complete_and_three_sources(self):
        out,result=self.saved();mix,_,_=self.load(out,result)
        tracker=mod.RuntimeExposure(mix)
        tracker.consume(torch.arange(len(mix),dtype=torch.int64))
        state=tracker.state();a,b,c=tracker.source_counts(state)
        self.assertEqual(a.tolist(),[1]*8)
        self.assertEqual(b.tolist(),[2,2,2,2,0])
        self.assertEqual(int(c.sum()),16);self.assertEqual(int(c[1]),0)
        report=tracker.finalize(state=state,world=8,batch=2,seed=42,epoch=1)
        self.assertEqual(report["actual_total"],32)
        self.assertTrue(report["old_per_window_counts_exact"])
        tracker.validate_contexts([tracker.context_sha256]*8)
        with self.assertRaises(ValueError):tracker.validate_contexts([tracker.context_sha256,"bad"])
        with self.assertRaises(ValueError):tracker.consume(torch.tensor([0],dtype=torch.int64))
        with self.assertRaises(ValueError):tracker.finalize(state=state,world=1,batch=2,seed=42,epoch=1)

    def test_partial_and_duplicate_actual_count_cannot_finalize(self):
        out,result=self.saved();mix,_,_=self.load(out,result);tracker=mod.RuntimeExposure(mix)
        tracker.consume(torch.tensor([0,1],dtype=torch.int64))
        self.assertEqual(tracker.summarize(tracker.state())["actual_total"],2)
        with self.assertRaises(ValueError):tracker.finalize(state=tracker.state(),world=8,batch=2,seed=42,epoch=1)
        with self.assertRaises(ValueError):tracker.consume(torch.tensor([2,2],dtype=torch.int64))

    def test_tag_stripped_before_policy_and_unknown_policy_key_rejected(self):
        out,result=self.saved();mix,_,_=self.load(out,result)
        tagged=mod.ExposureTaggedData(mix)[0]
        batch={k:v.unsqueeze(0) for k,v in tagged.items()}
        stripped,ids=mod.strip_exposure_tag(batch)
        self.assertEqual(set(stripped),mod.TENSORS);self.assertEqual(ids.tolist(),[0])
        with patch.object(ToyBase,"__getitem__",return_value={"metadata":torch.tensor([0.])}):
            with self.assertRaisesRegex(ValueError,"ten float32"):mix[0]

    def test_no_symlink_read_and_output_path_escape(self):
        out,result=self.saved()
        original=out/"plan.json";target=out.parent/"copy.json";target.write_bytes(original.read_bytes())
        original.unlink();original.symlink_to(target)
        with self.assertRaises(ValueError):self.load(out,result)
        plan,proof=self.prepare()
        with self.assertRaises(ValueError):mod.save_candidate(self.f["root"]/"outside",plan,proof)

    def test_diagnostic_positions_cover_recovery_and_exclude_dropped_old(self):
        out,result=self.saved();mix,_,_=self.load(out,result)
        with patch("wa.wm.teacher_plan_runtime.diagnostic_positions",return_value=[0,7,8,9,10,11,12,16]):
            positions=mod.diagnostic_positions(mix,[],{},16)
        self.assertEqual(len(positions),16);self.assertEqual(len(set(positions)),16)
        self.assertTrue(any(mix.locate(i)[0]==2 for i in positions))
        self.assertNotIn(12,mix.plan["arrays"]["old_plan_positions"][positions])


if __name__=="__main__":unittest.main()
