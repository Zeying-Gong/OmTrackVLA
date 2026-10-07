"""CPU fixtures for exact one-key continuation; no models or simulator."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from wa.wm import failure_state_continuation as mod


class ContinuationTests(unittest.TestCase):
    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")

    def lines(self, path, values):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(v, allow_nan=False)+"\n" for v in values))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/"job_61833/task_73055/wa_failure_state_collect_a800_v1"
        self.root.mkdir(parents=True)
        self.collection = self.root/"lane0/collection"
        self.base = self.collection/"stt"/mod.KEY
        self.plan_path = Path(self.tmp.name)/"base.json"
        result = self.result(False)
        entries = [dict(task="stt", key=mod.KEY if i==0 else f"scene/{i}",
                        manifest_index=i, baseline_result=result,
                        baseline_row_sha256="a"*64, artifact_root=f"/old/{i}")
                   for i in range(126)]
        self.plan = dict(entries=entries, lanes=[entries[i::8] for i in range(8)],
                         checkpoint=dict(path="/pinned/checkpoint.pt", sha256="c"*64, step=59716))
        self.write(self.plan_path, self.plan)
        self.base_sha = mod._sha(self.plan_path)
        for field, value in (("SOURCE_ROOT", self.root), ("BASE_SHA", self.base_sha)):
            self.patch(field, value)
        self.loader = self.patch("load_plan", mock.Mock(return_value=self.plan))
        self._terminal()
        self._records()
        files = [p for name in mod.BRANCHES for p in (self.base/name).iterdir()]
        files += [self.base/"search.json"]+[self.collection/name for name in (
            "records.jsonl", "plan_identity.json", "model.json", "simulator_config.yaml")]
        self.assertEqual(len(files), 180)
        inventory = {str(p.relative_to(self.collection)): mod._sha(p) for p in sorted(files)}
        self.patch("INVENTORY_SHA", mod.canonical_sha(inventory))
        self.patch("INVENTORY_BYTES", sum(p.stat().st_size for p in files))

    def patch(self, name, value):
        patcher = mock.patch.object(mod, name, value)
        self.addCleanup(patcher.stop)
        return patcher.start()

    @staticmethod
    def result(success):
        return dict(success=float(success), collision=float(not success), policy_init_valid=True,
                    following_rate=38/47 if success else 8/11, total_step=47 if success else 11,
                    following_step=38 if success else 8, status="Normal" if success else "Collision",
                    finish=success)

    def _terminal(self):
        common = dict(training_released=False, no_success_rate=True)
        launch = dict(experiment=mod.EXPERIMENT, source="/frozen/source",
                      source_commit=mod.SOURCE_COMMIT, source_git_status="",
                      plan=str(self.plan_path), plan_sha256=self.base_sha,
                      protocol_sha256=mod.PROTOCOL_SHA, formal=True,
                      expected_entries=126, expected_lanes=8,
                      devices=[str(i) for i in range(8)], **common)
        self.write(self.root/"launch.json", launch)
        self.write(self.root/"ERROR.json", dict(error_type="RuntimeError",
                   message="collector failed; preserve original logs and partial outputs", **common))
        for lane in range(8):
            path = self.root/f"lane{lane}"
            self.write(path/"ERROR.json", dict(error_type="RuntimeError", message="lane aborted", **common))
            self.write(path/"launch.json", dict(experiment=mod.EXPERIMENT, plan_sha256=self.base_sha,
                       protocol_sha256=mod.PROTOCOL_SHA, lane=lane,
                       keys=[e["key"] for e in self.plan["lanes"][lane]]))
            events = []
            for i, role in enumerate(("wa","lightnav","worker")):
                pid = 100+lane*3+i
                events.append(dict(utc="2026-10-07T00:00:00+00:00", role=role, event="spawn",
                                   pid=pid, process_group=pid, command={"role":role}))
            if lane==1:
                events.append(dict(utc="2026-10-07T00:00:01+00:00", role="worker",
                                   event="worker_return", pid=105, returncode=1))
            for i, role in enumerate(("wa","lightnav","worker")):
                events.append(dict(utc="2026-10-07T00:00:02+00:00", role=role,
                                   event="exit", pid=100+lane*3+i,
                                   returncode=1 if lane==1 and role=="worker" else -15))
            self.lines(path/"processes.jsonl", events)

    def _branch(self, name, prefix=None):
        path = self.base/name
        path.mkdir(parents=True)
        teacher = "student" if name=="student" else ("lightnav" if name.startswith("lightnav") else "oracle")
        repeat = name.endswith("_repeat")
        n = 47 if teacher=="oracle" else 11
        times = np.array([i*.048+(i//12)*.008 for i in range(n)])
        obs, actions, replay = [], [], []
        for i in range(n):
            owner = "student" if teacher=="student" or i<6 else "teacher"
            action = [-.2,0.,0.] if owner=="student" else [.2,0.,0.]
            state = dict(timestamp=float(times[i]), agents=[dict(transform=np.eye(4).tolist(),joints=[0.])])
            rgbsha = hashlib.sha256(str(i).encode()).hexdigest()
            frame = f"rgb_{i:04d}.png"
            (path/frame).write_bytes(b"FIXTURE_RGB_"+str(i).encode())
            obs.append(dict(frame=frame, sim_step=i, timestamp_s=float(times[i]),
                            simulator_world_time_s=float(times[i]), robot_position_world=[i*.01,0.,0.],
                            robot_rotation_world_from_body=np.eye(3).tolist()))
            actions.append(dict(sim_step=i, normalized_action=action, owner=owner))
            replay.append(dict(step=i, observation_index=i, timestamp_s=float(times[i]),
                               simulator_world_time_s=float(times[i]), rgb_sha256=rgbsha,
                               dynamic_state=state, dynamic_state_sha256=mod.canonical_sha(state),
                               action=action, normalized_action=action, owner=owner,
                               post_state_recorded=False))
        first = {**copy.deepcopy(replay[0]), "initial_bbox_sensor_xyxy_original":[1,1,3,3]}
        takeover = copy.deepcopy(replay[6]) if teacher!="student" else None
        raw = []
        positions = np.asarray([o["robot_position_world"] for o in obs])
        for i in range(n):
            offsets = np.arange(1,8)/10
            query = times[i]+offsets
            if query[-1]>times[-1]+1e-8:
                break
            right = np.clip(np.searchsorted(times,query,side="left"),1,n-1)
            future = np.stack([np.interp(query,times,positions[:,a]) for a in range(3)],axis=-1)
            delta = future-positions[i]
            raw.append(dict(current_index=i, future_times_s=offsets.tolist(),
                            future_bracket_indices=np.stack((right-1,right),axis=-1).tolist(),
                            trajectory_xy_m=np.stack((delta[:,0],-delta[:,2]),axis=-1).tolist()))
        windows = []
        if name=="oracle_0006":
            for w in raw[6:]:
                v=copy.deepcopy(w)
                end=max(b[1] for b in v["future_bracket_indices"])
                v.update(teacher_owned_action_indices=[v["current_index"],end-1],
                         label_endpoint_observation_index=end, training_eligible=False)
                windows.append(v)
        result = self.result(teacher=="oracle")
        shared = dict(experiment=mod.EXPERIMENT, partition="evaluation_adaptation", task="stt",
                      key=mod.KEY, teacher=teacher, verification_only=repeat, training_eligible=False,
                      takeover_step=None if teacher=="student" else 6)
        metadata = {**shared, "observed_frame_dt_s":np.diff(times).tolist(),
                    "initial_bbox_sensor_xyxy_original":[1,1,3,3]}
        admission = {**shared, "complete":True, "transport_fallback":False, "training_released":False,
                     "issues":[], "result":result, "candidate_windows":len(windows), "raw_windows":len(raw)}
        if teacher!="student":
            admission.update(
                replay_verified=True, seed=7, protocol_sha256=mod.PROTOCOL_SHA,
                prefix_sha256=mod.canonical_sha(prefix), initial_rgb_sha256=prefix[0]["rgb_sha256"],
                takeover_state_sha256=mod.canonical_sha(prefix[6]["dynamic_state"]),
                actual_takeover_state_sha256=mod.canonical_sha(takeover["dynamic_state"]),
                owned_suffix={"action_count":n-6},
                agent_validation=dict(verified_prefix_frames=7, required_prefix_frames=7,
                    agent_step_before_reset=n, replay_wrapper=True, prefix_hash_matches=True,
                    environment_bound=True, takeover_matches=True, hidden_rng_contact_state_proven=False))
        complete = dict(complete=True, frames=n, windows=len(windows), candidate_windows=len(windows),
                        raw_windows=len(raw), training_eligible=False, training_released=False)
        documents = dict(metadata=metadata, observations=obs, actions=actions, replay=replay,
                         first_start=first, pair_start=first, takeover=takeover, fallback_events=[],
                         admission=admission, complete=complete, result=result, windows=windows, raw_windows=raw)
        for filename,value in documents.items():
            self.write(path/(filename+".json"), value)
        (path/"initial_panoptic.npy").write_bytes(b"FIXTURE_PANOPTIC")
        if teacher=="student":
            return dict(result=result, first_start=first, artifact_root=str(path), initial_pair_verified=True),replay
        fields=mod.PAIR_FIELDS+("complete","replay_verified","transport_fallback",
                              "actual_takeover_state_sha256","candidate_windows","verification_only")
        branch={f:admission[f] for f in fields}
        branch.update(experiment=mod.EXPERIMENT, teacher=teacher, result=result, artifact_root=str(path))
        self.write(path/"branch.json", branch)
        return branch,replay

    def _records(self):
        student,prefix=self._branch("student")
        ln,_=self._branch("lightnav_0006",prefix)
        oracle,_=self._branch("oracle_0006",prefix)
        repeat,_=self._branch("oracle_0006_repeat",prefix)
        selection=mod.select_recovery_teacher(ln,oracle)
        accepted=dict(teacher="oracle",takeover_step=6,artifact_root=oracle["artifact_root"],
                      repeat_artifact_root=repeat["artifact_root"],selection=selection,
                      prefix_sha256=oracle["prefix_sha256"],candidate_only=True,training_released=False)
        self.row=dict(experiment=mod.EXPERIMENT,task="stt",key=mod.KEY,
            plan_sha256=self.base_sha,protocol_sha256=mod.PROTOCOL_SHA,
            outcome="repeated_teacher_recovery_candidate",student=student,accepted=accepted,
            training_released=False,score_backfill_allowed=False,
            baseline_result_unchanged=self.plan["entries"][0]["baseline_result"],
            attempts=[dict(takeover_step=6,branches=dict(lightnav=ln,oracle=oracle),
                           selection=selection,repeat=repeat,repeat_valid=True,selected_candidate_windows=26)])
        self.save_row()
        self.write(self.collection/"plan_identity.json",dict(path=str(self.plan_path),sha256=self.base_sha,
                   protocol_sha256=mod.PROTOCOL_SHA))
        self.write(self.collection/"model.json",{**self.plan["checkpoint"],
                   "checkpoint":self.plan["checkpoint"]["path"],
                   "checkpoint_sha256":self.plan["checkpoint"]["sha256"],
                   "text_used":False,"world_predictor_inference":False})
        (self.collection/"simulator_config.yaml").write_text("fixture: true\n")

    def save_row(self):
        self.lines(self.collection/"records.jsonl",[self.row])
        self.write(self.base/"search.json",self.row)

    def build(self, **kwargs):
        return mod.build_continuation(self.plan_path,self.base_sha,self.root,
                                      expected_job_status=kwargs.get("status","FAILED"))

    def save_overlay(self,value):
        path=Path(self.tmp.name)/"overlay.json"
        self.write(path,value)
        return path,mod._sha(path)

    def test_build_preserves_base_and_original_lanes(self):
        before=copy.deepcopy(self.plan)
        blob=self.plan_path.read_bytes()
        result=self.build()
        self.assertEqual(result["expected_count"],126)
        self.assertEqual(result["new_count"],125)
        self.assertEqual(result["reused_count"],1)
        self.assertEqual(result["reused_keys"],[mod.KEY])
        self.assertEqual([len(x) for x in result["remaining_lanes"]],mod.REMAINING_LANE_COUNTS)
        for actual,original in zip(result["remaining_lanes"],before["lanes"]):
            self.assertEqual(actual,[e for e in original if e["key"]!=mod.KEY])
        self.assertEqual(self.plan,before)
        self.assertEqual(self.plan_path.read_bytes(),blob)
        self.assertFalse(result["training_released"])
        self.assertTrue(result["source"]["scheduler_recheck_required"])
        self.assertEqual(result["artifact_inventory"]["file_count"],180)
        self.assertEqual(len(result["source"]["source_pins"]),26)
        self.assertEqual(sum(len(x["roles"]) for x in result["source"]["owned_process_exits"]),24)

    def test_roundtrip_load_rebuilds_everything(self):
        original=self.build()
        path,sha=self.save_overlay(original)
        self.assertEqual(mod.load_continuation(path,sha),original)
        self.assertGreaterEqual(self.loader.call_count,2)

    def test_scheduler_status_must_be_explicit_failed(self):
        for status in (None,"RUNNING","SUCCEEDED","failed",True):
            with self.subTest(status=status),self.assertRaisesRegex(ValueError,"FAILED"):
                self.build(status=status)

    def test_old_root_and_base_sha_fixed(self):
        with self.assertRaisesRegex(ValueError,"base plan SHA"):
            mod.build_continuation(self.plan_path,"0"*64,self.root,expected_job_status="FAILED")
        with self.assertRaisesRegex(ValueError,"61833/73055"):
            mod.build_continuation(self.plan_path,self.base_sha,self.root.parent,expected_job_status="FAILED")

    def test_missing_exit_and_cleanup_failure_rejected(self):
        path=self.root/"lane2/processes.jsonl"
        rows=mod._lines(path)
        self.lines(path,rows[:-1])
        with self.assertRaisesRegex(ValueError,"three exited roles"):
            self.build()
        self.lines(path,rows)
        self.write(self.root/"lane2/cleanup_errors.json",[{"error":"failed"}])
        with self.assertRaisesRegex(ValueError,"cleanup failed"):
            self.build()

    def test_foreign_pid_or_duplicate_exit_rejected(self):
        path=self.root/"lane3/processes.jsonl"
        rows=mod._lines(path)
        rows[-1]["pid"]+=1
        self.lines(path,rows)
        with self.assertRaisesRegex(ValueError,"unmatched"):
            self.build()

    def test_root_error_is_required_not_inferred_from_no_complete(self):
        (self.root/"ERROR.json").unlink()
        with self.assertRaisesRegex(ValueError,"missing"):
            self.build()

    def test_changed_image_or_inventory_member_rejected(self):
        path=self.base/"oracle_0006/rgb_0046.png"
        path.write_bytes(path.read_bytes()+b"changed")
        with self.assertRaises(ValueError):
            self.build()

    def test_missing_or_duplicate_row_rejected(self):
        self.lines(self.collection/"records.jsonl",[self.row,self.row])
        with self.assertRaisesRegex(ValueError,"exactly one"):
            self.build()

    def test_row_search_mismatch_rejected(self):
        changed=copy.deepcopy(self.row);changed["key"]="foreign/1"
        self.write(self.base/"search.json",changed)
        with self.assertRaisesRegex(ValueError,"records/search"):
            self.build()

    def test_wrong_winner_or_repeat_flag_rejected(self):
        self.row["accepted"]["teacher"]="lightnav"
        self.save_row()
        with self.assertRaisesRegex(ValueError,"accepted candidate"):
            self.build()
        self.row["accepted"]["teacher"]="oracle"
        self.row["attempts"][0]["repeat_valid"]=False
        self.save_row()
        with self.assertRaisesRegex(ValueError,"repeat_valid"):
            self.build()

    def test_no_repeat_windows_or_prefix_labels(self):
        path=self.base/"oracle_0006_repeat/windows.json"
        self.write(path,[{"current_index":6}])
        with self.assertRaisesRegex(ValueError,"candidate window set"):
            self.build()

    def test_bad_seven_step_bracket_rejected(self):
        path=self.base/"oracle_0006/raw_windows.json"
        windows=mod._read(path)
        windows[6]["future_bracket_indices"][0][1]+=1
        self.write(path,windows)
        with self.assertRaisesRegex(ValueError,"interpolation bracket"):
            self.build()

    def test_duplicate_or_missing_original_plan_key_rejected(self):
        self.plan["entries"][1]["key"]=mod.KEY
        with self.assertRaisesRegex(ValueError,"duplicate/missing"):
            self.build()

    def test_wrong_original_lane_partition_rejected(self):
        self.plan["lanes"][0],self.plan["lanes"][1]=self.plan["lanes"][1],self.plan["lanes"][0]
        with self.assertRaisesRegex(ValueError,"original lane"):
            self.build()

    def test_unknown_skip_or_overlap_or_bad_pin_in_overlay_rejected(self):
        original=self.build()
        for name in ("unknown_skip","overlap","omission","inventory","source_pin","extra"):
            altered=copy.deepcopy(original)
            if name=="unknown_skip":altered["reused_keys"]=["foreign/1"]
            elif name=="overlap":altered["remaining_lanes"][0].append(self.plan["entries"][0])
            elif name=="omission":altered["remaining_entries"].pop()
            elif name=="inventory":
                first=next(iter(altered["artifact_inventory"]["files"]))
                altered["artifact_inventory"]["files"][first]="0"*64
            elif name=="source_pin":altered["source"]["source_pins"]["ERROR.json"]="0"*64
            else:altered["skip_keys"]=["foreign/1"]
            path,sha=self.save_overlay(altered)
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,"reconstructed"):
                mod.load_continuation(path,sha)

    def test_same_numeric_value_float_to_int_overlay_is_rejected(self):
        original=self.build()
        changed=copy.deepcopy(original)
        entries=changed["remaining_entries"]+[
            entry for lane in changed["remaining_lanes"] for entry in lane]
        converted=0
        for entry in entries:
            result=entry["baseline_result"]
            for field in ("success","collision"):
                value=result[field]
                self.assertIs(type(value),float)
                integer=int(value)
                self.assertEqual(value,integer)  # Python numeric equality is insufficient.
                result[field]=integer
                self.assertIs(type(result[field]),int)
                converted+=1
        self.assertEqual(converted,500)
        self.assertEqual(changed,original)
        self.assertNotEqual(mod.canonical_sha(changed),mod.canonical_sha(original))
        path,sha=self.save_overlay(changed)
        with self.assertRaisesRegex(ValueError,"reconstructed frozen evidence"):
            mod.load_continuation(path,sha)

    def test_bad_overlay_file_sha_rejected_before_rebuild(self):
        path,sha=self.save_overlay(self.build())
        with self.assertRaisesRegex(ValueError,"file SHA"):
            mod.load_continuation(path,"0"*64)

    def test_actual_source_pin_mutation_rejected_on_load(self):
        path,sha=self.save_overlay(self.build())
        error=self.root/"lane4/ERROR.json"
        value=mod._read(error);value["message"]="changed-but-still-error"
        self.write(error,value)
        with self.assertRaisesRegex(ValueError,"reconstructed"):
            mod.load_continuation(path,sha)


if __name__=="__main__":
    unittest.main()
