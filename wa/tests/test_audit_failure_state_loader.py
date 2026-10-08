"""Bounded CPU tests for the independent audit; no model or GPU weights."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from wa.tests.test_failure_state_data import full_fixture,sha
from wa.wm.failure_state_data import FailureStateData
from wa.tools import audit_failure_state_loader as mod


class LoaderAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp=tempfile.TemporaryDirectory()
        cls.fixture=full_fixture(Path(cls.tmp.name).resolve())
        cls.cache=cls.fixture["out"];cls.pin=sha(cls.cache/"admission.json")
        cls.data=FailureStateData(cls.cache,expected_admission_sha256=cls.pin)
        cls.selected,cls.episodes=mod.select_windows(cls.data)
        cls.ep=18;cls.position=cls.fixture["positions"][cls.ep]
        cls.row=int(cls.data.rows[cls.position])
        cls.now=int(cls.data.history[cls.row,-1])
    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def manual(self):
        return mod.ManualEpisode(self.data.episodes[self.ep])

    def test_selection_all_90_and_zero6(self):
        self.assertEqual(sum(x["valid_windows"]>0 for x in self.episodes),90)
        self.assertEqual(sum(x["valid_windows"]==0 for x in self.episodes),6)
        for ep in self.episodes:
            if ep["valid_windows"]:
                self.assertTrue({ep["first_position"],ep["early_position"],ep["tail_position"]}
                                <=set(ep["selected_positions"]))
                self.assertEqual(ep["representative_unique_count"],3)
        positions=[r["position"] for r in self.selected]
        self.assertEqual(len(positions),len(set(positions)))

    def test_selection_no_prefix_omission(self):
        chosen={r["position"]:r for r in self.selected}
        for pos,row in enumerate(self.data.rows):
            ep=int(self.data.episode[int(row)]);k=self.data.episodes[ep]["takeover_step"]
            now=int(self.data.history[int(row),-1])
            must=int(self.data.history[int(row),0])<k or now-4<k
            if must:
                self.assertIn(pos,chosen)
                self.assertIn("student_prefix",chosen[pos]["reasons"])

    def test_manual_all_ten_tensors_and_real_getitem(self):
        actual=self.data[self.position]
        expected=self.manual().expected(self.now)
        error=mod.compare_item(actual,expected,self.data.pose_labels[self.row])
        self.assertEqual(set(error),set(mod.SHAPES))
        self.assertLessEqual(max(error.values()),2e-6)
        self.assertEqual(self.now,6)
        self.assertEqual(self.selected[next(i for i,r in enumerate(self.selected)
                                           if r["position"]==self.position)]["proprio_indices"],[2,3,4,5])

    def test_current_polar_manual_formula(self):
        item=self.manual().expected(6)
        dx=(2.+6*.001)-.06
        expected=torch.tensor([np.hypot(dx,-.1),np.arctan2(.1,dx)],dtype=torch.float32)
        self.assertTrue(torch.equal(item["polar"],expected))

    def test_future_targets_never_change_inputs(self):
        manual=self.manual();before=manual.expected(6)
        for obs in manual.obs[7:]:
            obs["target_position_world_label_only"]=[1e5,-5e4,2e5]
        after=manual.expected(6)
        self.assertTrue(all(torch.equal(before[k],after[k]) for k in mod.SHAPES))

    def test_current_target_changes_only_polar_geometry(self):
        manual=self.manual();before=manual.expected(6)
        manual.obs[6]["target_position_world_label_only"]=[5.,0.,-.4]
        after=manual.expected(6)
        self.assertEqual({k for k in mod.SHAPES if not torch.equal(before[k],after[k])},
                         {"polar","geometry"})

    def test_teacher_future_ownership_rejected(self):
        manual=self.manual()
        manual.by_step[7]["owner"]="student"
        with self.assertRaisesRegex(ValueError,"owned"):
            manual.expected(6)

    def test_raw_command_not_reclipped_to_hide_error(self):
        manual=self.manual()
        manual.by_step[3]["normalized_action"]=[1.1,0.,0.]
        with self.assertRaisesRegex(ValueError,"normalized"):
            manual.expected(6)

    def test_template_must_be_episode_zero(self):
        manual=self.manual();actual=self.data[self.position]
        wrong=dict(manual.expected(6))
        wrong["template"]=mod.image_tensor(manual.root/manual.entry["frames"][6],
                                           manual.meta["initial_bbox_rgb_xyxy"])
        with self.assertRaisesRegex(ValueError,"template"):
            mod.compare_item(actual,wrong,self.data.pose_labels[self.row])

    def test_actual_plus_one_future_not_label_endpoint(self):
        manual=self.manual();actual=self.data[self.position]
        wrong=dict(manual.expected(6));wrong["future"]=manual.picture(20)
        with self.assertRaisesRegex(ValueError,"future"):
            mod.compare_item(actual,wrong,self.data.pose_labels[self.row])

    def test_shape_dtype_finite_and_exact_keys_rejected(self):
        expected=self.manual().expected(self.now)
        for field,value in (("polar",torch.ones(3)),("times",torch.ones(4,dtype=torch.float64)),
                            ("commands",torch.full((4,4),float("nan")))):
            actual=dict(expected);actual[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                mod.compare_item(actual,expected,self.data.pose_labels[self.row])
        with self.assertRaisesRegex(ValueError,"ten"):
            mod.compare_item({**expected,"hidden_gt":torch.ones(1)},expected,self.data.pose_labels[self.row])

    def test_pose_cache_comparison_is_bit_exact(self):
        expected=self.manual().expected(self.now)
        actual=dict(self.data[self.position]);actual["pose"]=actual["pose"].clone()
        actual["pose"][0,0]+=1e-7
        with self.assertRaisesRegex(ValueError,"bit-exact"):
            mod.compare_item(actual,expected,self.data.pose_labels[self.row])

    def test_actual_make_conditions_uses_no_supervised_future(self):
        result=mod.check_conditions(self.data[self.position])
        self.assertFalse(result["loaded_encoder_weights"])
        self.assertFalse(result["loaded_policy_weights"])
        self.assertEqual(result["mode"],"mixed")

    def test_sparse_continuous_and_predecessor_indices_differ(self):
        history,wm,previous=mod.indices(np.arange(60)*.05,30)
        np.testing.assert_array_equal(history,[0,10,20,30])
        np.testing.assert_array_equal(wm,[27,28,29,30])
        np.testing.assert_array_equal(previous,[26,27,28,29])

    def test_existing_output_rejected_before_admission(self):
        path=Path(self.tmp.name)/"existing.json";path.write_text("preserve")
        with patch.object(mod,"audit",side_effect=AssertionError("must not run")):
            with self.assertRaises(FileExistsError):
                mod.run_to_file(self.cache,self.pin,path)
        self.assertEqual(path.read_text(),"preserve")

    def test_bad_pin_saves_failure_never_pass(self):
        path=Path(self.tmp.name)/"failure.json"
        with self.assertRaises(ValueError):
            mod.run_to_file(self.cache,"0"*64,path)
        result=json.loads(path.read_text())
        self.assertEqual(result["status"],"FAILED_NOT_RELEASED")
        self.assertFalse(result["training_released"])

    def test_output_symlink_and_parent_alias_rejected(self):
        path=Path(self.tmp.name)/"dangling.json";path.symlink_to(Path(self.tmp.name)/"missing")
        with self.assertRaises(FileExistsError):
            mod.run_to_file(self.cache,self.pin,path)
        alias=Path(self.tmp.name)/"alias";alias.symlink_to(self.cache)
        with self.assertRaisesRegex(ValueError,"parent"):
            mod.run_to_file(self.cache,self.pin,alias/"audit.json")

    def test_existing_gate_only_probe_bounded_and_real(self):
        probe=mod.stat_gate_probe(self.data)
        self.assertEqual(probe["total_measurements"],9)
        self.assertTrue(probe["actual_getitem_not_replaced"])
        self.assertEqual(len(probe["samples"]),3)
        for row in probe["samples"]:
            self.assertEqual(len(row["seconds"]),3)
            self.assertGreater(row["median_seconds"],0)
            self.assertGreater(row["episode_paths"],row["png_paths"])

    def test_full_synthetic_audit_real_loader_no_gate_mock(self):
        events=[]
        result=mod.audit(self.cache,self.pin,progress=events.append)
        self.assertEqual(result["status"],mod.STATUS)
        self.assertEqual(result["selection"]["all_valid_rows_inspected"],6864)
        self.assertEqual(result["selection"]["nonzero_episodes"],90)
        self.assertEqual(len(result["condition_routing"]),90)
        self.assertEqual(len(result["windows"]),len(self.selected))
        self.assertGreater(result["timing"]["initialization_seconds"],0)
        self.assertGreater(result["timing"]["measured_getitems_per_second"],0)
        self.assertTrue(result["per_getitem_stat_scope"]["every_episode_png_statted"])
        self.assertFalse(result["training_released"])
        self.assertFalse(json.loads((self.cache/"admission.json").read_text())["training_released"])
        self.assertTrue(all(e["stage"]=="CHECKING_NOT_PASS" for e in events))


if __name__=="__main__":
    unittest.main()
