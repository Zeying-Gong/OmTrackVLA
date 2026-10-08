"""Isolated CPU report fixtures; no cache graph, media, model or GPU execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wa.wm import failure_state_train_gate as mod


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


class FailureStateTrainGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.cache = self.root/"cache"
        self.cache.mkdir()
        self.report_path = self.root/"audit.json"
        self.sources = {}
        self.source_root = self.root/"calling-checkout"
        for relative in mod.TOOL_SOURCES:
            path = self.source_root/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            blob = ("fixture source: " + relative).encode()
            path.write_bytes(blob)
            self.sources["/previous-checkout/"+relative] = sha(blob)
        self.admission = dict(schema="failure_state_cache_v1_admission_candidate",
            status="CACHE_CONVERTED_NOT_TRAINING_RELEASED", training_released=False,
            expected_admission_sha_required_by_loader=True, independent_audit_required=True,
            summary=mod.EXPECTED.copy(),
            collection_release=dict(path="/not-opened/release.json",sha256="f"*64))
        self.admission_blob = json.dumps(self.admission).encode()
        (self.cache/"admission.json").write_bytes(self.admission_blob)
        self.admission_sha = sha(self.admission_blob)
        self.doc = self.fixture()
        self.root_patch = patch.object(mod,"_source_root",return_value=self.source_root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.addCleanup(self.tmp.cleanup)

    def fixture(self):
        episodes,windows,conditions = [],[],[]
        offset = 0
        for ep in range(96):
            count = (76 if ep < 89 else 100) if ep < 90 else 0
            entry = dict(episode=ep,episode_uid=f"stt:scene/{ep}",
                         valid_windows=count,selected_positions=[])
            if count:
                positions = [offset,offset+1,offset+count-1]
                entry.update(root=f"/not-opened/scene/{ep}/oracle_0004",
                    teacher="oracle",takeover_step=4,first_position=positions[0],
                    early_position=positions[1],tail_position=positions[2],
                    early_definition="second chronological valid window",
                    early_elapsed_after_takeover_s=.35,representative_unique_count=3,
                    selected_positions=positions)
                for position,now,reason in zip(positions,(10,11,20),("first","early_second","tail")):
                    windows.append(dict(position=position,cache_row=position,episode=ep,now=now,
                        history=[now-3,now-2,now-1,now],jepa_indices=list(range(now-3,now+1)),
                        proprio_indices=list(range(now-4,now)),reasons=[reason],prefix_crossing=[],
                        max_absolute_errors={key:0. for key in mod.SHAPES},
                        pose_tensor_sha256="a"*64,template_tensor_sha256="b"*64,
                        current_polar=[2.,.2],getitem_seconds=.01,
                        independent_verification_seconds=.005))
                conditions.append(dict(episode=ep,
                    status="ACTUAL_MAKE_CONDITIONS_ROUTING_PASS_CPU_ENCODER_STANDIN",
                    loaded_encoder_weights=False,loaded_policy_weights=False,mode="mixed",
                    perturbed_nonpolicy_fields=list(mod.NONPOLICY_FIELDS)))
                offset += count
            episodes.append(entry)
        return dict(schema=mod.SCHEMA,status=mod.STATUS,started_utc="2026-10-08T08:00:00+00:00",
            finished_utc="2026-10-08T08:01:00+00:00",cache=str(self.cache),
            admission_sha256=self.admission_sha,
            loader_admission=dict(status="LOADER_ADMISSION_PASS_NOT_NEW_MODEL_RESULT",
                summary=mod.EXPECTED.copy(),admission_sha256=self.admission_sha,release_sha256="f"*64,
                actual_consumed_sources_hashed=True,consumed_source_files=15034,
                unconsumed_historical_weights_rehashed=False,
                input_scope="RGB/episode-zero BBox/current ideal zero-delay simulated polar UWB; no text",
                training_released_on_disk=False),
            tool_sources_sha256=self.sources.copy(),
            runtime=dict(python="3.11",numpy="1.26",torch="2.6",torch_cpu_threads=1,cuda_used=False),
            selection=dict(rule=mod.SELECTION_RULE,all_valid_rows_inspected=6864,
                nonzero_episodes=90,zero_valid_episodes=6,selected_unique_windows=len(windows),
                all_prefix_crossing_windows=0,prefix_crossing_by_input={}),
            shapes=deepcopy(mod.SHAPES),dtype="torch.float32",pose_cache_comparison="bit-exact",
            raw_pose_max_absolute_tolerance=2e-6,episodes=episodes,windows=windows,
            condition_routing=conditions,timing={},stat_gate_probe={},
            per_getitem_stat_scope={},boundaries=["fixture only; not training release"],
            training_released=False,no_new_model_success_rate=True)

    def call(self,doc=None,*,pin=None):
        blob = json.dumps(self.doc if doc is None else doc,allow_nan=False).encode()
        self.report_path.write_bytes(blob)
        return mod.verify_loader_audit(self.report_path,pin or sha(blob),
                                      self.cache,self.admission_sha)

    def test_valid_report_nonrelease_and_no_graph_scan(self):
        result = self.call()
        self.assertEqual(result["report"],self.doc)
        self.assertFalse(result["training_released"])
        self.assertEqual(set(result["tool_sources_by_relative_path"]),set(mod.TOOL_SOURCES))
        self.assertEqual(result["report_sha256"],sha(self.report_path.read_bytes()))
        # No cache npy, raw branch or media fixture exists: this gate does not scan them.
        self.assertEqual(list(self.cache.iterdir()),[self.cache/"admission.json"])

    def test_external_report_pin_required(self):
        for bad in ("0"*64,"A"*64,"x",True,None):
            blob = json.dumps(self.doc).encode()
            self.report_path.write_bytes(blob)
            with self.subTest(pin=bad),self.assertRaises(ValueError):
                mod.verify_loader_audit(self.report_path,bad,self.cache,self.admission_sha)

    def test_changed_actual_admission_rejected(self):
        (self.cache/"admission.json").write_text("{}")
        with self.assertRaisesRegex(ValueError,"SHA mismatch"):
            self.call()

    def test_report_admission_or_cache_mismatch(self):
        for key,value in (("admission_sha256","0"*64),("cache",str(self.root))):
            doc=deepcopy(self.doc);doc[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): self.call(doc)

    def test_false_training_flags_strict(self):
        mutations = [(None,"training_released",True),(None,"training_released",0),
            (None,"no_new_model_success_rate",False),
            ("loader_admission","training_released_on_disk",True),
            ("loader_admission","actual_consumed_sources_hashed",False),
            ("runtime","cuda_used",True)]
        for section,key,value in mutations:
            doc=deepcopy(self.doc);target=doc if section is None else doc[section];target[key]=value
            with self.subTest(section=section,key=key),self.assertRaises(ValueError): self.call(doc)

    def test_schema_status_and_required_fields(self):
        for key,value in (("schema","unknown"),("status","FAILED_NOT_RELEASED"),
                          ("status","PASS"),("dtype","torch.float64")):
            doc=deepcopy(self.doc);doc[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError): self.call(doc)
        for mutation in ("missing","extra"):
            doc=deepcopy(self.doc)
            if mutation=="missing": del doc["windows"]
            else: doc["future_policy_inputs"]=True
            with self.assertRaises(ValueError): self.call(doc)

    def test_all_valid_counts_and_summary_bound(self):
        for section,key,value in (("selection","all_valid_rows_inspected",6863),
            ("selection","nonzero_episodes",89),("selection","zero_valid_episodes",7),
            ("selection","selected_unique_windows",271)):
            doc=deepcopy(self.doc);doc[section][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): self.call(doc)
        doc=deepcopy(self.doc);doc["loader_admission"]["summary"]["valid_windows"]=6863
        with self.assertRaises(ValueError): self.call(doc)

    def test_episode_counts_identity_and_zero_valid_coverage(self):
        for mutate in (
            lambda d:d["episodes"].pop(),
            lambda d:d["episodes"][1].update(episode_uid=d["episodes"][0]["episode_uid"]),
            lambda d:d["episodes"][0].update(valid_windows=75),
            lambda d:d["episodes"][95].update(selected_positions=[1]),
            lambda d:d["episodes"][0].update(episode=True)):
            doc=deepcopy(self.doc);mutate(doc)
            with self.assertRaises(ValueError): self.call(doc)

    def test_missing_representative_and_bidirectional_coverage(self):
        doc=deepcopy(self.doc)
        doc["episodes"][0]["selected_positions"].remove(0)
        doc["windows"].pop(0);doc["selection"]["selected_unique_windows"]-=1
        with self.assertRaises(ValueError): self.call(doc)
        doc=deepcopy(self.doc);doc["episodes"][0]["selected_positions"]=[0,1,74]
        with self.assertRaises(ValueError): self.call(doc)

    def test_duplicate_or_cross_episode_window_rejected(self):
        for update in (dict(position=0),dict(cache_row=0),dict(episode=0)):
            doc=deepcopy(self.doc);doc["windows"][3].update(update)
            with self.subTest(update=update),self.assertRaises(ValueError): self.call(doc)

    def test_causal_and_teacher_owned_indices(self):
        for update in (dict(now=3),dict(now=True),dict(history=[7,8,9,11]),
            dict(jepa_indices=[8,9,10,11]),dict(proprio_indices=[7,8,9,10])):
            doc=deepcopy(self.doc);doc["windows"][0].update(update)
            with self.subTest(update=update),self.assertRaises(ValueError): self.call(doc)
        doc=deepcopy(self.doc);doc["episodes"][0]["takeover_step"]=11
        with self.assertRaises(ValueError): self.call(doc)

    def test_prefix_crossing_recomputed_not_status_only(self):
        doc=deepcopy(self.doc);row=doc["windows"][0]
        row.update(now=4,history=[0,0,0,4],jepa_indices=[1,2,3,4],proprio_indices=[0,1,2,3],
                   reasons=["first","student_prefix"],prefix_crossing=["policy","jepa_command","proprio"])
        doc["selection"].update(all_prefix_crossing_windows=1,
            prefix_crossing_by_input=dict(policy=1,jepa_command=1,proprio=1))
        self.call(doc)
        row["prefix_crossing"]=[]
        with self.assertRaises(ValueError): self.call(doc)

    def test_shapes_exact_ten_with_strict_dimension_types(self):
        for mutate in (
            lambda d:d["shapes"].update(hidden_gt=[3]),
            lambda d:d["shapes"]["commands"].__setitem__(0,5),
            lambda d:d["shapes"]["polar"].__setitem__(0,True)):
            doc=deepcopy(self.doc);mutate(doc)
            with self.assertRaises(ValueError): self.call(doc)

    def test_tensor_comparison_and_tolerance(self):
        for field,error in (("pose",2.1e-6),("rgb",1e-8),("future",-1),("polar",True)):
            doc=deepcopy(self.doc);doc["windows"][0]["max_absolute_errors"][field]=error
            with self.subTest(field=field),self.assertRaises(ValueError): self.call(doc)
        doc=deepcopy(self.doc);doc["raw_pose_max_absolute_tolerance"]=1e-3
        with self.assertRaises(ValueError): self.call(doc)

    def test_routing_every_nonzero_episode_and_no_future_conditions(self):
        mutations=(
            lambda d:d["condition_routing"].pop(),
            lambda d:d["condition_routing"][1].update(episode=0),
            lambda d:d["condition_routing"][0].update(loaded_policy_weights=True),
            lambda d:d["condition_routing"][0].update(mode="image"),
            lambda d:d["condition_routing"][0]["perturbed_nonpolicy_fields"].remove("future"),
            lambda d:d["condition_routing"][0].update(future_policy_input=True))
        for mutate in mutations:
            doc=deepcopy(self.doc);mutate(doc)
            with self.assertRaises(ValueError): self.call(doc)

    def test_release_binding_not_status_only(self):
        doc=deepcopy(self.doc);doc["loader_admission"]["release_sha256"]="a"*64
        with self.assertRaises(ValueError): self.call(doc)

    def test_recorded_root_can_move_but_source_content_must_match(self):
        doc=deepcopy(self.doc)
        doc["tool_sources_sha256"]={name.replace("/previous-checkout/","/frozen-source/"):pin
                                    for name,pin in self.sources.items()}
        self.call(doc)
        (self.source_root/mod.TOOL_SOURCES[0]).write_text("modified current source")
        with self.assertRaisesRegex(ValueError,"SHA mismatch"): self.call(doc)

    def test_sources_exact_unique_relative_five_and_one_root(self):
        for mode in ("missing","duplicate-relative","unrecognized","mixed-root","relative","badpin"):
            doc=deepcopy(self.doc);pins=doc["tool_sources_sha256"]
            key=next(iter(pins));pin=pins.pop(key)
            if mode=="duplicate-relative":
                pins[key]=pin;pins["/elsewhere/"+mod.TOOL_SOURCES[0]]=pin
                pins.pop("/previous-checkout/"+mod.TOOL_SOURCES[1])
            elif mode=="unrecognized": pins["/previous-checkout/wa/new.py"]=pin
            elif mode=="mixed-root": pins["/elsewhere/"+mod.TOOL_SOURCES[0]]=pin
            elif mode=="relative": pins[mod.TOOL_SOURCES[0]]=pin
            elif mode=="badpin": pins[key]="bad"
            with self.subTest(mode=mode),self.assertRaises(ValueError): self.call(doc)

    def test_duplicate_nonfinite_and_overflow_json_rejected(self):
        for blob in (b'{"schema":1,"schema":2}',b'{"x":NaN}',b'{"x":1e999}'):
            self.report_path.write_bytes(blob)
            with self.subTest(blob=blob),self.assertRaises(ValueError):
                mod.verify_loader_audit(self.report_path,sha(blob),self.cache,self.admission_sha)

    def test_symlink_report_cache_and_source_rejected(self):
        self.call()
        alias=self.root/"alias.json";alias.symlink_to(self.report_path)
        with self.assertRaises(ValueError):
            mod.verify_loader_audit(alias,sha(self.report_path.read_bytes()),self.cache,self.admission_sha)
        target=self.source_root/mod.TOOL_SOURCES[0]
        other=self.root/"source-copy";other.write_bytes(target.read_bytes())
        target.unlink();target.symlink_to(other)
        with self.assertRaises(ValueError): self.call()

    def test_source_change_during_verification_rejected(self):
        original=mod._verify_sources
        def changed(pins,states):
            result=original(pins,states)
            (self.source_root/mod.TOOL_SOURCES[0]).write_text("changed after hashing")
            return result
        with patch.object(mod,"_verify_sources",side_effect=changed):
            with self.assertRaisesRegex(ValueError,"changed before"): self.call()

    def test_timestamps_need_completed_aware_order(self):
        for update in (dict(finished_utc="2026-10-08T07:59:00+00:00"),
                       dict(finished_utc="2026-10-08T08:01:00")):
            doc=deepcopy(self.doc);doc.update(update)
            with self.assertRaises(ValueError): self.call(doc)


if __name__ == "__main__":
    unittest.main()
