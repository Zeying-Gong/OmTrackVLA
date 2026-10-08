"""CPU fixtures for unchanged numeric contract and fail-closed endpoint edges."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from wa.wm.failure_state_labels import derive_labels
from wa.wm.failure_state_numeric import audit_numeric_windows
from wa.tools.audit_failure_state_numeric import Inputs, successful


def fixture(k=6,n=60,indices=None):
    obs=[dict(sim_step=i,timestamp_s=i*.05,frame=f"rgb_{i:04d}.png",
              robot_position_world=[i*.01,0.,0.],robot_rotation_world_from_body=np.eye(3).tolist())
         for i in range(n)]
    acts=[dict(sim_step=i,normalized_action=[.1,0.,0.],
               owner="teacher" if i>=k else "student",teacher="oracle" if i>=k else None)
          for i in range(n)]
    return obs,acts,list(range(k,n-15)) if indices is None else indices


def windows(obs,acts,indices,k):
    d=derive_labels(obs,acts,indices,k)
    return [dict(current_index=i,trajectory_xy_m=d["trajectory_xy_m"][r].tolist(),
                 future_bracket_indices=d["future_bracket_indices"][r].tolist(),
                 future_times_s=(np.arange(1,8)/10.).tolist(),
                 teacher_owned_action_indices=[d["ownership_audit"]["future_transition_indices"][r][0],
                                               d["ownership_audit"]["future_transition_indices"][r][-1]],
                 label_endpoint_observation_index=int(d["end_indices"][r]),training_eligible=False)
            for r,i in enumerate(indices)]


class NumericTests(unittest.TestCase):
    def check(self,obs,acts,indices,k):
        return audit_numeric_windows(obs,acts,windows(obs,acts,indices,k),k)
    def test_exact_labels_history_jepa(self):
        obs,acts,ix=fixture()
        got=self.check(obs,acts,ix,6)
        self.assertEqual(got["summary"]["valid_windows"],len(ix))
        self.assertEqual(got["summary"]["max_deltas"]["old_dual_pose_max_abs"],0.)
        np.testing.assert_array_equal(got["derived"]["jepa_indices"][0],[3,4,5,6])
        np.testing.assert_array_equal(got["derived"]["history"][0],[0,0,0,6])
        self.assertEqual(got["summary"]["valid_jepa_history_uses_student_prefix"],3)
        self.assertEqual(got["summary"]["valid_proprio_uses_student_prefix"],4)
        self.assertFalse(got["training_released"])
    def test_start_excludes_first_four(self):
        obs,acts,ix=fixture(k=0)
        got=self.check(obs,acts,ix,0)
        self.assertEqual(got["summary"]["excluded_windows"],4)
        self.assertEqual(got["summary"]["rejection_reason_counts"],{"CURRENT_INDEX_LT_4":4})
    def test_endpoint_within_tolerance_excluded_by_old_runtime(self):
        obs,acts,ix=fixture(k=4,n=23,indices=[8])
        obs[-1]["timestamp_s"]=1.1-5e-9
        got=self.check(obs,acts,ix,4)
        self.assertTrue(got["derived"]["valid_mask"][0])
        self.assertFalse(got["effective_mask"][0])
        self.assertEqual(got["summary"]["old_mask_disagreement_windows"],1)
        self.assertEqual(got["summary"]["clipped_unclipped_endpoint_disagreement_windows"],1)
        self.assertIn("OLD_RUNTIME_ENDPOINT_UNOBSERVED",got["excluded"][0]["reasons"])
    def test_endpoint_exact_real_state_retained(self):
        obs,acts,ix=fixture(k=4,n=23,indices=[8])
        obs[-1]["timestamp_s"]=obs[8]["timestamp_s"]+.7
        got=self.check(obs,acts,ix,4)
        self.assertTrue(got["effective_mask"][0])
    def test_no_extrapolation(self):
        obs,acts,ix=fixture(k=4,n=23,indices=[8])
        obs[-1]["timestamp_s"]=1.1-2e-8
        with self.assertRaises(ValueError):self.check(obs,acts,ix,4)
    def test_oversized_prefix_excludes_history_spanning_window(self):
        obs,acts,ix=fixture(indices=[6,40])
        obs[2]["robot_position_world"]=[1.,0.,0.]
        got=self.check(obs,acts,ix,6)
        self.assertEqual(got["effective_mask"].tolist(),[False,True])
    def test_dt_filter(self):
        obs,acts,ix=fixture(indices=[6])
        obs[7]["timestamp_s"]=obs[6]["timestamp_s"]+.001
        got=self.check(obs,acts,ix,6)
        self.assertFalse(got["effective_mask"][0])
    def test_teacher_ownership_mutation(self):
        obs,acts,ix=fixture(indices=[6])
        w=windows(obs,acts,ix,6);acts[7]["owner"]="student"
        with self.assertRaises(ValueError):audit_numeric_windows(obs,acts,w,6)
    def test_recorded_bracket_mutation(self):
        obs,acts,ix=fixture(indices=[6]);w=windows(obs,acts,ix,6)
        w[0]["future_bracket_indices"][0][0]+=1
        with self.assertRaises(ValueError):audit_numeric_windows(obs,acts,w,6)
    def test_recorded_owner_range_mutation(self):
        obs,acts,ix=fixture(indices=[6]);w=windows(obs,acts,ix,6)
        w[0]["teacher_owned_action_indices"][0]=5
        with self.assertRaises(ValueError):audit_numeric_windows(obs,acts,w,6)
    def test_recorded_xy_mutation(self):
        obs,acts,ix=fixture(indices=[6]);w=windows(obs,acts,ix,6)
        w[0]["trajectory_xy_m"][0][0]+=.01
        with self.assertRaises(ValueError):audit_numeric_windows(obs,acts,w,6)
    def test_missing_poststate_action_never_labelled(self):
        obs,acts,ix=fixture(indices=[44])
        got=self.check(obs,acts,ix,6)
        self.assertEqual(got["summary"]["selected_actions_without_poststate"],1)
        self.assertLess(max(got["derived"]["ownership_audit"]["future_transition_indices"][0]),len(obs)-1)
    def test_empty_rejected(self):
        obs,acts,ix=fixture()
        with self.assertRaises(ValueError):audit_numeric_windows(obs,acts,[],6)
    def test_repeat_role_and_fallback(self):
        ref=dict(complete=True,replay_verified=True,transport_fallback=False,verification_only=False,
                 result=dict(success=1.,collision=0.,policy_init_valid=True))
        successful(ref,False)
        with self.assertRaises(ValueError):successful(ref,True)
        ref["transport_fallback"]=True
        with self.assertRaises(ValueError):successful(ref,False)
    def test_source_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"x.json";p.write_text('{"x":1}')
            r=Inputs();r.read(p);p.write_text('{"x":2}')
            with self.assertRaises(ValueError):r.verify()
    def test_source_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"x.json";p.write_text('{}')
            q=Path(tmp)/"y.json";q.symlink_to(p)
            with self.assertRaises(ValueError):Inputs().read(q)
    def test_duplicate_json_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"x.json";p.write_text('{"x":1,"x":2}')
            with self.assertRaises(ValueError):Inputs().read(p)


if __name__=="__main__":
    unittest.main()
