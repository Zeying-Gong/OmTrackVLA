"""Production-sized synthetic CPU admission tests; no safety gate is patched."""
import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from wa.wm import failure_state_data as mod
from wa.wm.failure_state_numeric import audit_numeric_windows
from wa.wm.failure_state_labels import derive_labels
from wa.wm.robot_data import transition_records
from wa.wm.training import JointRobotModel


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,sort_keys=True,allow_nan=False))


def full_fixture(root):
    """96 original winners, exact7396/6864/532, including6 zero-valid episodes."""
    out=root/"cache";out.mkdir();(out/"index").mkdir()
    base=root/"base";base.mkdir();index=root/"base-index";index.mkdir()
    np.save(base/"heldout_pose.npy",np.zeros((1,7,4),np.float32))
    np.save(base/"heldout_history.npy",np.array([[0,0,0,4]],np.int32))
    np.save(base/"heldout_episode.npy",np.array([0],np.int32))
    save(base/"heldout_episodes.json",[dict(scene="held.glb")])
    save(base/"complete.json",dict(files={f"heldout_{s}":sha(base/f"heldout_{s}") for s in mod.SUFFIXES}))
    np.save(index/"heldout_valid.npy",np.array([0],np.int64))
    save(index/"heldout_episodes_audit.json",dict(stats=[],errors=[]))
    held=dict(original=1,retained=1,excluded=0,index_sha256=sha(index/"heldout_valid.npy"),episode_errors=[])
    save(index/"audit.json",dict(contract=mod.CONTRACT,cache_complete_sha256=sha(base/"complete.json"),splits=dict(heldout=held)))
    demos=[];entries=[];stats=[];images={};sources={}
    poses=[];histories=[];episodeids=[];valid=[];offset=0;positions={}
    for ep in range(96):
        count=78 if ep<4 else 77
        k=0 if ep<17 or ep>=90 else 2 if ep==17 else 6
        dt=.2 if ep>=90 else .05
        indices=list(range(k,k+count));n=indices[-1]+20
        branch=root/f"source/stt/scene/{ep}/oracle_{k:04d}"
        repeat=branch.with_name(branch.name+"_repeat")
        branch.mkdir(parents=True);repeat.mkdir()
        pair=(61833,73055) if ep==0 else (61836,73058) if ep<24 else (61844,73066)
        experiment,protocol=mod.SOURCE_JOBS[pair]
        obs=[dict(sim_step=i,timestamp_s=i*dt,frame=f"rgb_{i:04d}.png",
                  robot_position_world=[i*.01,0.,0.],robot_rotation_world_from_body=np.eye(3).tolist(),
                  target_position_world_label_only=[2.+i*.001,0.,-.1]) for i in range(n)]
        acts=[dict(sim_step=i,normalized_action=[.1+i*.001,0.,0.],
                   owner="teacher" if i>=k else "student",teacher="oracle" if i>=k else None) for i in range(n)]
        derived=derive_labels(obs,acts,indices,k)
        wins=[dict(current_index=i,trajectory_xy_m=derived["trajectory_xy_m"][j].tolist(),
            future_bracket_indices=derived["future_bracket_indices"][j].tolist(),
            future_times_s=(np.arange(1,8)/10.).tolist(),
            teacher_owned_action_indices=[i,int(derived["end_indices"][j])-1],
            label_endpoint_observation_index=int(derived["end_indices"][j]),training_eligible=False)
              for j,i in enumerate(indices)]
        numeric=audit_numeric_windows(obs,acts,wins,k,per_window=False)
        ident=dict(experiment=experiment,protocol_sha256=protocol,task="stt",key=f"scene/{ep}",
                   teacher="oracle",takeover_step=k,verification_only=False)
        meta=dict(**ident,partition="evaluation_adaptation",training_eligible=False,
                  camera_alignment_verified=True,initial_bbox_status="VERIFIED_FROZEN_FIRST_RGB_REPAIR" if ep==24 else "VERIFIED_CONFIG_AND_SEMANTIC",
                  rgb_shape=[8,8,3],initial_bbox_rgb_xyxy=[1,1,7,7],
                  timebase_version="v3_actual_world_time_interpolated",scene_id="scene.glb")
        if pair==(61844,73066):meta["teacher_boundary_policy"]=mod.CONTINUATION_POLICY
        result=dict(success=1.,collision=0.,policy_init_valid=True)
        branchdoc=dict(**ident,artifact_root=str(branch),complete=True,replay_verified=True,transport_fallback=False,result=result)
        adm=dict(**ident,complete=True,replay_verified=True,transport_fallback=False,result=result,
                 training_eligible=False,training_released=False)
        complete=dict(complete=True,status="FAILURE_STATE_RAW_COLLECTION_COMPLETE",frames=n,
                      candidate_windows=count,training_eligible=False,training_released=False)
        docs={"metadata.json":meta,"observations.json":obs,"actions.json":acts,"windows.json":wins,
              "result.json":result,"branch.json":branchdoc,"admission.json":adm,"complete.json":complete,
              "replay.json":[],"first_start_pair.json":{},"takeover_pair.json":{}}
        for name,value in docs.items():save(branch/name,value)
        for i in range(n):
            Image.new("RGB",(8,8),((3*i)%255,ep,i%255)).save(branch/f"rgb_{i:04d}.png")
        hashes={p.name:sha(p) for p in branch.iterdir()}
        sources.update({str(branch/name):h for name,h in hashes.items()})
        demo=dict(task="stt",key=f"scene/{ep}",teacher="oracle",takeover_step=k,branch=str(branch),
            repeat_branch=str(repeat),source_experiment=experiment,source_protocol_sha256=protocol,
            source_job_id=pair[0],source_task_id=pair[1],window_indices=indices,
            candidate_windows=count,valid_window_indices=numeric["valid_window_indices"],
            numeric_exclusions=numeric["excluded"],expected_valid_count=numeric["summary"]["valid_windows"],
            expected_excluded_count=numeric["summary"]["excluded_windows"],hashes=hashes)
        if pair==(61844,73066):demo["source_boundary_policy"]=mod.CONTINUATION_POLICY
        demos.append(demo)
        frames=[o["frame"] for o in obs];images[str(branch)]={f:hashes[f] for f in frames}
        rows=(offset+np.flatnonzero(numeric["effective_mask"])).tolist()
        if rows:positions[ep]=len(valid)
        valid.extend(rows);poses.append(derived["pose"]);histories.append(derived["history"])
        episodeids.append(np.full(count,ep,np.int32))
        commands,bad,trans=transition_records(branch,obs)
        stats.append({**trans,**numeric["summary"],"episode":ep,"rows":count,"retained":len(rows),
            "candidate_row_start":offset,"candidate_current_indices":indices,
            "valid_current_indices":numeric["valid_window_indices"],"valid_cache_rows":rows,
            "numeric_exclusions":numeric["excluded"],"source_sha256":{name:hashes[name] for name in ("metadata.json","observations.json","actions.json")}})
        entry=dict(root=str(branch),frames=frames,task="stt",episode_uid=f"stt:scene/{ep}",scene="scene.glb",
            takeover_step=k,category="failure_state_evaluation_adaptation",teacher="oracle",template_index=0,
            source_experiment=experiment,source_protocol_sha256=protocol,source_job_id=pair[0],source_task_id=pair[1],
            source_repeat_branch=str(repeat),repeat_frames_referenced=False,expected_valid_count=len(rows))
        if pair==(61844,73066):entry["source_boundary_policy"]=mod.CONTINUATION_POLICY
        entries.append(entry);offset+=count
    assert offset==7396 and len(valid)==6864 and len(positions)==90
    release=dict(schema="failure_state_collection_release_v1",collection_validated=True,
        expected=126,completed_searches=126,evaluation_adaptation=True,untouched_test=False,no_success_rate=True,
        score_backfill_allowed=False,training_released=False,cache_conversion_required=True,
        teacher_demonstrations=demos,summary={k:v for k,v in mod.EXPECTED.items() if k!="completed_searches"},
        source_files=dict(sources))
    rp=root/"release.json";save(rp,release);sources[str(rp)]=sha(rp)
    # Real local code bytes are pinned; there is no patched admission/count gate.
    code={str(Path(mod.__file__).resolve()):sha(mod.__file__)}
    sources.update(code)
    sources.update({str(p):sha(p) for folder in (base,index) for p in folder.iterdir()})
    for key,arrays in (("pose",poses),("history",histories),("episode",episodeids)):
        np.save(out/f"train_{key}.npy",np.concatenate(arrays))
    np.save(out/"index/train_valid.npy",np.array(valid,np.int64))
    save(out/"train_episodes.json",entries)
    save(out/"index/train_episodes_audit.json",dict(stats=stats,errors=[]))
    save(out/"source_image_hashes.json",images);save(out/"source_files.json",sources)
    for suffix in mod.SUFFIXES:(out/f"heldout_{suffix}").symlink_to(base/f"heldout_{suffix}")
    for name in ("heldout_valid.npy","heldout_episodes_audit.json"):(out/"index"/name).symlink_to(index/name)
    complete=dict(schema=mod.SCHEMA,experiment=mod.EXPERIMENT,contract=mod.CONTRACT,
        source_release=str(rp),source_sha256=sha(rp),source_image_inventory_sha256=sha(out/"source_image_hashes.json"),
        source_file_inventory_sha256=sha(out/"source_files.json"),
        files={f:sha(out/f) for f in mod.CACHE_FILES},summary=dict(mod.EXPECTED),code_sha256=code,
        template_index=0,original_frame_sequences_preserved=True,future_labels_teacher_owned=True,
        independent_cache_admission_required=True,student_prefix_actions_used_as_future_labels=0,
        repeat_windows_counted=0,known_heldout_scene_overlap=[],training_released=False)
    save(out/"complete.json",complete)
    audit=dict(schema=mod.SCHEMA+"_index",contract=mod.CONTRACT,experiment=mod.EXPERIMENT,
        cache_complete_sha256=sha(out/"complete.json"),training_released=False,
        train_episode_audit_sha256=sha(out/"index/train_episodes_audit.json"),
        heldout_episode_audit_sha256=sha(out/"index/heldout_episodes_audit.json"),
        splits=dict(train=dict(original=7396,retained=6864,excluded=532,index_sha256=sha(out/"index/train_valid.npy"),episode_errors=[]),heldout=held))
    save(out/"index/audit.json",audit)
    admission=dict(schema=mod.SCHEMA+"_admission_candidate",status="CACHE_CONVERTED_NOT_TRAINING_RELEASED",
        collection_release=dict(path=str(rp),sha256=sha(rp)),files={f:sha(out/f) for f in mod.FILES},
        source_files_sha256=sha(out/"source_files.json"),summary=dict(mod.EXPECTED),code_sha256=code,
        original_heldout=dict(cache=str(base),cache_complete_sha256=sha(base/"complete.json"),
                              index=str(index),index_audit_sha256=sha(index/"audit.json")),
        expected_admission_sha_required_by_loader=True,independent_audit_required=True,training_released=False)
    save(out/"admission.json",admission)
    return dict(root=root,out=out,release=rp,entries=entries,positions=positions)


class FailureDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp=tempfile.TemporaryDirectory()
        cls.f=full_fixture(Path(cls.tmp.name).resolve())
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def load(self):
        return mod.FailureStateData(self.f["out"],expected_admission_sha256=sha(self.f["out"]/"admission.json"))

    def edit(self,path,value):
        """Re-sign a deliberately changed synthetic graph; safety checks remain real."""
        path=Path(path);before=path.read_bytes()
        self.addCleanup(path.write_bytes,before)
        if isinstance(value,bytes):path.write_bytes(value)
        else:save(path,value)

    def rebind(self,raw=None):
        out=self.f["out"];rp=self.f["release"]
        release=json.loads(rp.read_bytes())
        sources=json.loads((out/"source_files.json").read_bytes())
        images=json.loads((out/"source_image_hashes.json").read_bytes())
        if raw is not None:
            raw=Path(raw);d=next(d for d in release["teacher_demonstrations"] if str(raw.parent)==d["branch"])
            d["hashes"][raw.name]=sha(raw);release["source_files"][str(raw)]=sha(raw)
            sources[str(raw)]=sha(raw)
            if raw.name.endswith(".png"):images[str(raw.parent)][raw.name]=sha(raw)
            self.edit(rp,release)
            if raw.name in ("metadata.json","observations.json","actions.json"):
                statpath=out/"index/train_episodes_audit.json"
                stats=json.loads(statpath.read_bytes())
                ep=next(i for i,x in enumerate(release["teacher_demonstrations"]) if x["branch"]==str(raw.parent))
                stats["stats"][ep]["source_sha256"][raw.name]=sha(raw)
                self.edit(statpath,stats)
        sources[str(rp)]=sha(rp)
        self.edit(out/"source_files.json",sources);self.edit(out/"source_image_hashes.json",images)
        c=json.loads((out/"complete.json").read_bytes())
        c.update(source_sha256=sha(rp),source_image_inventory_sha256=sha(out/"source_image_hashes.json"),
                 source_file_inventory_sha256=sha(out/"source_files.json"),files={f:sha(out/f) for f in mod.CACHE_FILES})
        self.edit(out/"complete.json",c)
        audit=json.loads((out/"index/audit.json").read_bytes())
        audit.update(cache_complete_sha256=sha(out/"complete.json"),
                     train_episode_audit_sha256=sha(out/"index/train_episodes_audit.json"),
                     heldout_episode_audit_sha256=sha(out/"index/heldout_episodes_audit.json"))
        audit["splits"]["train"]["index_sha256"]=sha(out/"index/train_valid.npy")
        self.edit(out/"index/audit.json",audit)
        a=json.loads((out/"admission.json").read_bytes())
        a.update(collection_release=dict(path=str(rp),sha256=sha(rp)),files={f:sha(out/f) for f in mod.FILES},
                 source_files_sha256=sha(out/"source_files.json"))
        self.edit(out/"admission.json",a)

    def test_full_production_counts_real_getitem_and_zero_valid(self):
        data=self.load();self.assertEqual(len(data),6864)
        self.assertEqual(data.admission_report["summary"],mod.EXPECTED)
        self.assertFalse(data.admission_report["unconsumed_historical_weights_rehashed"])
        item=data[self.f["positions"][18]]
        self.assertEqual(tuple(item["pose"].shape),(7,4))
        self.assertEqual(tuple(item["rgb"].shape),(4,3,224,224))
        self.assertEqual(tuple(item["wm_rgb"].shape),(4,3,224,224))
        self.assertEqual(tuple(item["commands"].shape),(4,4))
        np.testing.assert_allclose(item["times"].numpy(),[-.3,-.3,-.3,0.],atol=1e-7)
        np.testing.assert_allclose(item["commands"][:,0],[.103,.104,.105,.106],atol=1e-7)
        np.testing.assert_allclose(item["proprio"][:,0],[.102,.103,.104,.105],atol=1e-7)
        np.testing.assert_allclose(item["commands"][:,3],[.5]*4,atol=1e-7)
        self.assertTrue(all(e["expected_valid_count"]==0 for e in data.episodes[-6:]))
        self.assertFalse(np.isin(np.asarray(data.episode)[data.rows],range(90,96)).any())

    def test_explicit_sha_required_and_wrong_pin_rejected(self):
        with self.assertRaises(TypeError):mod.FailureStateData(self.f["out"])
        for bad in ("0"*64,True,None,"bad"):
            with self.assertRaises(ValueError):mod.FailureStateData(self.f["out"],expected_admission_sha256=bad)

    def test_partial_graph_or_production_counts_rejected(self):
        p=self.f["out"]/"admission.json";a=json.loads(p.read_bytes())
        del a["files"]["index/train_valid.npy"];self.edit(p,a)
        with self.assertRaises(ValueError):self.load()

    def test_bad_production_summary_rejected(self):
        p=self.f["out"]/"admission.json";a=json.loads(p.read_bytes())
        a["summary"]["valid_windows"]=3;self.edit(p,a)
        with self.assertRaises(ValueError):self.load()

    def test_cache_entry_first_frame_cannot_be_takeover_frame(self):
        p=self.f["out"]/"train_episodes.json";a=json.loads(p.read_bytes())
        a[0]["frames"][0]=a[0]["frames"][6];self.edit(p,a);self.rebind()
        with self.assertRaises(ValueError):self.load()

    def test_reordered_cached_labels_rejected_even_with_new_hashes(self):
        p=self.f["out"]/"train_pose.npy";a=np.load(p).copy();a[0,0,0]+=.01
        buff=io.BytesIO();np.save(buff,a);self.edit(p,buff.getvalue());self.rebind()
        with self.assertRaises(ValueError):self.load()

    def test_duplicate_valid_rows_rejected_even_with_new_hashes(self):
        p=self.f["out"]/"index/train_valid.npy";a=np.load(p).copy();a[1]=a[0]
        buff=io.BytesIO();np.save(buff,a);self.edit(p,buff.getvalue());self.rebind()
        with self.assertRaises(ValueError):self.load()

    def test_float_history_rejected(self):
        p=self.f["out"]/"train_history.npy";a=np.load(p).astype(np.float32)
        buff=io.BytesIO();np.save(buff,a);self.edit(p,buff.getvalue());self.rebind()
        with self.assertRaises(ValueError):self.load()

    def test_lazy_source_mutation_rejected(self):
        data=self.load();ep=18;p=Path(self.f["entries"][ep]["root"])/"rgb_0007.png"
        self.edit(p,b"not an image")
        with self.assertRaises(ValueError):data[self.f["positions"][ep]]

    def test_lazy_cache_mutation_rejected(self):
        data=self.load();p=self.f["out"]/"train_episodes.json"
        self.edit(p,p.read_bytes()+b" ")
        with self.assertRaises(ValueError):data[0]

    def test_future_target_does_not_change_policy_or_current_polar(self):
        ep=18;pos=self.f["positions"][ep];before=self.load()[pos]
        p=Path(self.f["entries"][ep]["root"])/"observations.json";obs=json.loads(p.read_bytes())
        for item in obs[7:]:item["target_position_world_label_only"]=[900.,30.,-700.]
        self.edit(p,obs);self.rebind(p);after=self.load()[pos]
        for key in before:self.assertTrue(torch.equal(before[key],after[key]),key)

    def test_future_rgb_changes_only_jepa_target_and_not_policy_conditions(self):
        ep=18;pos=self.f["positions"][ep];before=self.load()[pos]
        p=Path(self.f["entries"][ep]["root"])/"rgb_0007.png"
        buff=io.BytesIO();Image.new("RGB",(8,8),(255,255,255)).save(buff,format="PNG")
        self.edit(p,buff.getvalue());self.rebind(p);after=self.load()[pos]
        for key in before:
            self.assertEqual(torch.equal(before[key],after[key]),key!="future",key)
        class CPUEncoder:
            def encode(self,images):return images.mean(dim=(-3,-2,-1))
        def conditions(item):
            batch={k:v.unsqueeze(0) for k,v in item.items()}
            return JointRobotModel.make_conditions(CPUEncoder(),batch,torch.tensor([2]))
        a,b=conditions(before),conditions(after)
        for key in a:self.assertTrue(torch.equal(a[key],b[key]),key)

    def test_current_polar_actually_uses_current_measurement(self):
        ep=18;pos=self.f["positions"][ep];before=self.load()[pos]
        p=Path(self.f["entries"][ep]["root"])/"observations.json";obs=json.loads(p.read_bytes())
        obs[6]["target_position_world_label_only"]=[4.,0.,-2.]
        self.edit(p,obs);self.rebind(p);after=self.load()[pos]
        self.assertFalse(torch.equal(before["polar"],after["polar"]))
        self.assertFalse(torch.equal(before["geometry"],after["geometry"]))
        for key in ("rgb","template","times","wm_rgb","commands","proprio","future","pose"):
            self.assertTrue(torch.equal(before[key],after[key]),key)

    def test_student_future_action_cannot_be_relabelled_teacher(self):
        p=Path(self.f["entries"][18]["root"])/"actions.json";a=json.loads(p.read_bytes())
        a[6]["owner"]="student";a[6]["teacher"]=None
        self.edit(p,a);self.rebind(p)
        with self.assertRaises(ValueError):self.load()

    def test_unknown_boundary_metadata_rejected(self):
        p=Path(self.f["entries"][24]["root"])/"metadata.json";a=json.loads(p.read_bytes())
        a["teacher_boundary_policy"]="unknown";self.edit(p,a);self.rebind(p)
        with self.assertRaises(ValueError):self.load()


    def test_raw_admission_cannot_prematurely_release(self):
        p=Path(self.f["entries"][24]["root"])/"admission.json";a=json.loads(p.read_bytes())
        a["training_eligible"]=True;self.edit(p,a);self.rebind(p)
        with self.assertRaises(ValueError):self.load()

    def test_lazy_heldout_mutation_rejected(self):
        data=self.load();p=self.f["root"]/"base"/"heldout_episodes.json"
        self.edit(p,p.read_bytes()+b" ")
        with self.assertRaises(ValueError):data[0]

    def test_missing_image_inventory_entry_rejected(self):
        p=self.f["out"]/"source_image_hashes.json";a=json.loads(p.read_bytes())
        del a[self.f["entries"][0]["root"]]["rgb_0000.png"]
        self.edit(p,a);self.rebind()
        with self.assertRaises(ValueError):self.load()

    def test_original_branch_cannot_be_a_repeat(self):
        p=Path(self.f["entries"][24]["root"])/"metadata.json";a=json.loads(p.read_bytes())
        a["verification_only"]=True;self.edit(p,a);self.rebind(p)
        with self.assertRaises(ValueError):self.load()

    def test_bool_dataset_position_rejected(self):
        data=self.load()
        for index in (True,np.bool_(True),.5):
            with self.assertRaises(ValueError):data[index]


if __name__=="__main__":unittest.main()
