"""CPU-only synthetic cache conversion; never reads or converts real data."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from wa.tools import build_failure_state_cache as mod
from wa.wm.failure_state_labels import derive_labels
from wa.wm.failure_state_numeric import audit_numeric_windows
from wa.wm.robot_data import RobotWorldData


def save(path, data):
    path.write_text(json.dumps(data, sort_keys=True, allow_nan=False))


def fixture(root):
    entries, inventory = [], {}
    for eid, k, indices in ((0, 6, [6, 8, 20]), (1, 0, [0, 1, 2, 3])):
        branch = root/f"source/stt/scene/{eid}/oracle_{k:04d}"
        repeat = branch.parent/f"oracle_{k:04d}_repeat"
        branch.mkdir(parents=True); repeat.mkdir()
        n = 40
        obs = [dict(sim_step=i,timestamp_s=i*.05,frame=f"rgb_{i:04d}.png",
                    robot_position_world=[i*.01,0.,0.],
                    robot_rotation_world_from_body=np.eye(3).tolist(),
                    target_position_world_label_only=[2.,0.,0.]) for i in range(n)]
        acts = [dict(sim_step=i,normalized_action=[.1,0.,0.],
                     owner="teacher" if i>=k else "student",
                     teacher="oracle" if i>=k else None) for i in range(n)]
        d = derive_labels(obs,acts,indices,k)
        windows = [dict(current_index=i,trajectory_xy_m=d["trajectory_xy_m"][j].tolist(),
            future_bracket_indices=d["future_bracket_indices"][j].tolist(),
            future_times_s=(np.arange(1,8)/10.).tolist(),
            teacher_owned_action_indices=[d["ownership_audit"]["future_transition_indices"][j][0],
                                          d["ownership_audit"]["future_transition_indices"][j][-1]],
            label_endpoint_observation_index=int(d["end_indices"][j]),training_eligible=False)
                   for j,i in enumerate(indices)]
        numeric = audit_numeric_windows(obs,acts,windows,k,per_window=False)
        ident = dict(experiment=mod.V2,protocol_sha256=mod.P2,task="stt",key=f"scene/{eid}",
                     teacher="oracle",takeover_step=k,verification_only=False)
        meta = dict(**ident,partition="evaluation_adaptation",training_eligible=False,
                    camera_alignment_verified=True,initial_bbox_status="VERIFIED_CONFIG_AND_SEMANTIC",
                    rgb_shape=[8,8,3],initial_bbox_rgb_xyxy=[1,1,7,7],
                    timebase_version="v3_actual_world_time_interpolated",scene_id="scene.glb")
        result = dict(success=1.,collision=0.,policy_init_valid=True)
        branchdoc = dict(**ident,artifact_root=str(branch),complete=True,replay_verified=True,
                         transport_fallback=False,result=result)
        admission = dict(**ident,complete=True,replay_verified=True,transport_fallback=False,
                         result=result,training_eligible=False,training_released=False)
        complete = dict(complete=True,status="FAILURE_STATE_RAW_COLLECTION_COMPLETE",
                        frames=n,training_eligible=False,training_released=False)
        docs={"metadata.json":meta,"observations.json":obs,"actions.json":acts,"windows.json":windows,
              "result.json":result,"branch.json":branchdoc,"admission.json":admission,
              "complete.json":complete,"replay.json":[],"first_start_pair.json":{},
              "takeover_pair.json":{}}
        for name, value in docs.items():save(branch/name,value)
        for i in range(n):
            Image.new("RGB",(8,8),(i,0,0)).save(branch/f"rgb_{i:04d}.png")
        hashes={p.name:mod.file_sha(p) for p in branch.iterdir()}
        inventory.update({str(branch/n):h for n,h in hashes.items()})
        entries.append(dict(task="stt",key=f"scene/{eid}",teacher="oracle",takeover_step=k,
            branch=str(branch),repeat_branch=str(repeat),source_experiment=mod.V2,
            source_protocol_sha256=mod.P2,source_job_id=61844,source_task_id=73066,
            source_boundary_policy="missing_rvq_or_final_l2_v1",window_indices=indices,
            valid_window_indices=numeric["valid_window_indices"],
            numeric_exclusions=numeric["excluded"],candidate_windows=len(indices),
            expected_valid_count=numeric["summary"]["valid_windows"],
            expected_excluded_count=numeric["summary"]["excluded_windows"],hashes=hashes))
    expected=dict(completed_searches=126,accepted_original_episodes=2,candidate_windows=7,
                  valid_windows=3,excluded_windows=4,episodes_with_valid_windows=1)
    release=dict(schema=mod.RELEASE_SCHEMA,collection_validated=True,expected=126,completed_searches=126,
                 evaluation_adaptation=True,untouched_test=False,no_success_rate=True,
                 score_backfill_allowed=False,training_released=False,cache_conversion_required=True,
                 teacher_demonstrations=entries,summary={k:v for k,v in expected.items() if k!="completed_searches"},
                 source_files=inventory)
    rp=root/"release.json";save(rp,release)
    base=root/"base";idx=root/"base-index";base.mkdir();idx.mkdir()
    np.save(base/"heldout_pose.npy",np.zeros((1,7,4),dtype=np.float32))
    np.save(base/"heldout_history.npy",np.array([[0,0,0,4]],dtype=np.int32))
    np.save(base/"heldout_episode.npy",np.array([0],dtype=np.int32))
    save(base/"heldout_episodes.json",[dict(scene="held.glb")])
    files={f"heldout_{s}":mod.file_sha(base/f"heldout_{s}") for s in mod.SUFFIXES}
    save(base/"complete.json",dict(files=files))
    np.save(idx/"heldout_valid.npy",np.array([0],dtype=np.int64))
    save(idx/"heldout_episodes_audit.json",dict(stats=[],errors=[]))
    split=dict(original=1,retained=1,excluded=0,index_sha256=mod.file_sha(idx/"heldout_valid.npy"),
               episode_errors=[])
    save(idx/"audit.json",dict(contract=mod.CONTRACT,
        cache_complete_sha256=mod.file_sha(base/"complete.json"),splits=dict(heldout=split)))
    return dict(root=root,release=rp,base=base,index=idx,out=root/"out",expected=expected)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.f=fixture(Path(self.tmp.name))
        self.addCleanup(patch.stopall)
        patch.object(mod,"NAS",self.f["root"]).start()
        patch.object(mod,"EXPECTED",self.f["expected"]).start()

    def convert(self):
        f=self.f
        return mod.convert(f["release"],mod.file_sha(f["release"]),f["base"],
            mod.file_sha(f["base"]/"complete.json"),f["index"],
            mod.file_sha(f["index"]/"audit.json"),f["out"])

    def edit_release(self,mutate):
        p=self.f["release"];d=json.loads(p.read_text());mutate(d);save(p,d)

    def change_branch(self,name,mutate,eid=0):
        r=json.loads(self.f["release"].read_text());e=r["teacher_demonstrations"][eid]
        p=Path(e["branch"])/name;d=json.loads(p.read_text());mutate(d);save(p,d)
        h=mod.file_sha(p);e["hashes"][name]=h;r["source_files"][str(p)]=h
        save(self.f["release"],r)

    def test_full_frames_original_masks_and_actual_loader(self):
        got=self.convert();self.assertFalse(got["training_released"])
        out=self.f["out"];eps=json.loads((out/"train_episodes.json").read_text())
        self.assertEqual(len(eps),2);self.assertEqual(len(eps[0]["frames"]),40)
        self.assertEqual(eps[0]["frames"][0],"rgb_0000.png")
        self.assertEqual(eps[0]["takeover_step"],6);self.assertEqual(eps[1]["expected_valid_count"],0)
        np.testing.assert_array_equal(np.load(out/"index/train_valid.npy"),[0,1,2])
        self.assertEqual(np.load(out/"train_pose.npy").shape,(7,7,4))
        np.testing.assert_array_equal(np.load(out/"train_history.npy")[0],[0,0,0,6])
        data=RobotWorldData(out,"train",out/"index")
        item=data[0]
        self.assertEqual(len(data),3);self.assertEqual(tuple(item["wm_rgb"].shape),(4,3,224,224))
        self.assertEqual(tuple(item["pose"].shape),(7,4))
        self.assertAlmostEqual(float(item["commands"][0,3]),.5)
        self.assertFalse((out/"rgb_0000.png").exists())

    def test_hash_graph_and_heldout_unchanged(self):
        before={p:mod.file_sha(p) for d in (self.f["base"],self.f["index"]) for p in d.iterdir()}
        got=self.convert();out=self.f["out"]
        a=json.loads((out/"admission.json").read_text())
        self.assertEqual(len(a["files"]),16)
        self.assertFalse(a["training_released"]);self.assertTrue(a["independent_audit_required"])
        for name,h in a["files"].items():self.assertEqual(mod.file_sha(out/name),h)
        self.assertEqual(mod.file_sha(out/"admission.json"),got["admission_sha256"])
        for p,h in before.items():self.assertEqual(mod.file_sha(p),h)
        for s in mod.SUFFIXES:self.assertEqual((out/f"heldout_{s}").resolve(),self.f["base"]/f"heldout_{s}")
        self.assertNotIn("admission.json",a["files"])
        complete=json.loads((out/"complete.json").read_text())
        self.assertEqual(len(complete["files"]),8)
        self.assertEqual(complete["student_prefix_actions_used_as_future_labels"],0)
        audit=json.loads((out/"index/audit.json").read_text())
        self.assertEqual(audit["cache_complete_sha256"],mod.file_sha(out/"complete.json"))
        self.assertEqual(audit["experiment"],complete["experiment"])

    def test_requires_production_counts_without_test_patch(self):
        with patch.object(mod,"EXPECTED",dict(completed_searches=126,accepted_original_episodes=96,
            candidate_windows=7396,valid_windows=6864,excluded_windows=532,episodes_with_valid_windows=90)):
            with self.assertRaises(ValueError):self.convert()
        self.assertFalse(self.f["out"].exists())

    def test_wrong_release_sha(self):
        f=self.f
        with self.assertRaises(ValueError):
            mod.convert(f["release"],"0"*64,f["base"],mod.file_sha(f["base"]/"complete.json"),
                        f["index"],mod.file_sha(f["index"]/"audit.json"),f["out"])

    def test_incomplete_or_premature_release(self):
        for field,value in (("collection_validated",False),("training_released",True),
                            ("untouched_test",True),("expected",125)):
            original=self.f["release"].read_bytes()
            self.edit_release(lambda r:r.update({field:value}))
            with self.assertRaises(ValueError):self.convert()
            self.f["release"].write_bytes(original)

    def test_source_tamper(self):
        r=json.loads(self.f["release"].read_text())
        (Path(r["teacher_demonstrations"][0]["branch"])/"rgb_0000.png").write_bytes(b"tamper")
        with self.assertRaises(ValueError):self.convert()

    def test_symbolic_source_rejected(self):
        r=json.loads(self.f["release"].read_text());p=Path(r["teacher_demonstrations"][0]["branch"])/"replay.json"
        backup=p.with_name("outside.json");backup.write_bytes(p.read_bytes());p.unlink();p.symlink_to(backup)
        with self.assertRaises(ValueError):self.convert()

    def test_duplicate_original_key(self):
        self.edit_release(lambda r:r["teacher_demonstrations"][1].update(key="scene/0"))
        with self.assertRaises(ValueError):self.convert()

    def test_repeat_cannot_be_original(self):
        self.edit_release(lambda r:r["teacher_demonstrations"][0].update(
            repeat_branch=r["teacher_demonstrations"][0]["branch"]))
        with self.assertRaises(ValueError):self.convert()

    def test_relabel_source_protocol(self):
        self.edit_release(lambda r:r["teacher_demonstrations"][0].update(source_experiment=mod.V1))
        with self.assertRaises(ValueError):self.convert()

    def test_repeat_metadata_rejected(self):
        self.change_branch("metadata.json",lambda d:d.update(verification_only=True))
        with self.assertRaises(ValueError):self.convert()
        self.assertTrue((self.f["out"]/"failed_conversion.json").exists())
        self.assertFalse((self.f["out"]/"admission.json").exists())

    def test_fallback_rejected(self):
        self.change_branch("branch.json",lambda d:d.update(transport_fallback=True))
        with self.assertRaises(ValueError):self.convert()

    def test_missing_initialization_rejected(self):
        self.change_branch("result.json",lambda d:d.pop("policy_init_valid"))
        with self.assertRaises(ValueError):self.convert()

    def test_teacher_action_cannot_be_student(self):
        self.change_branch("actions.json",lambda d:d[6].update(owner="student",teacher=None))
        with self.assertRaises(ValueError):self.convert()

    def test_exact_valid_set_not_only_count(self):
        self.edit_release(lambda r:r["teacher_demonstrations"][0].update(valid_window_indices=[6,8,21]))
        with self.assertRaises(ValueError):self.convert()

    def test_exact_exclusions(self):
        self.edit_release(lambda r:r["teacher_demonstrations"][1]["numeric_exclusions"][0].update(reasons=[]))
        with self.assertRaises(ValueError):self.convert()

    def test_no_suffix_frame_reindexing(self):
        self.change_branch("observations.json",lambda d:d.pop(0))
        with self.assertRaises(ValueError):self.convert()

    def test_no_future_frame_escape(self):
        self.change_branch("observations.json",lambda d:d[0].update(frame="../escape.png"))
        with self.assertRaises(ValueError):self.convert()

    def test_unmanifested_artifact(self):
        r=json.loads(self.f["release"].read_text())
        (Path(r["teacher_demonstrations"][0]["branch"])/"unknown.json").write_text("{}")
        with self.assertRaises(ValueError):self.convert()

    def test_base_manifest_pin_and_index_contract(self):
        save(self.f["index"]/"audit.json",dict(contract="wrong"))
        with self.assertRaises(ValueError):self.convert()

    def test_preserve_existing_output(self):
        self.f["out"].mkdir();sentinel=self.f["out"]/"untouched";sentinel.write_text("user")
        with self.assertRaises(ValueError):self.convert()
        self.assertEqual(sentinel.read_text(),"user")


if __name__=="__main__":
    unittest.main()
