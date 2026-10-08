import copy
import unittest
import numpy as np

from wa.wm.failure_state_labels import derive_labels


def fixture(n=35, takeover=4, dt=None, yaw=None):
    times = np.arange(n) * .1 if dt is None else np.r_[0., np.cumsum(dt)]
    yaw = np.zeros(n) if yaw is None else np.asarray(yaw)
    obs = []
    for i, (t, a) in enumerate(zip(times, yaw)):
        c, s = np.cos(a), np.sin(a)
        obs.append(dict(sim_step=i, timestamp_s=float(t),
                        robot_position_world=[float(.2*t), 0., float(-.1*t)],
                        robot_rotation_world_from_body=[[c, 0., s], [0., 1., 0.], [-s, 0., c]]))
    acts = [dict(sim_step=i, normalized_action=[.2, .1, 0.],
                 owner="student" if i < takeover else "teacher",
                 teacher=None if i < takeover else "lightnav") for i in range(n)]
    return obs, acts


class LabelsTests(unittest.TestCase):
    def test_exact_contract_history_template_and_dtype(self):
        obs, acts = fixture()
        out = derive_labels(obs, acts, [4, 20], 4)
        self.assertEqual(out["pose"].shape, (2, 7, 4))
        self.assertEqual(out["pose"].dtype, np.float32)
        np.testing.assert_allclose(out["pose"][0, :, :2], np.arange(1, 8)[:, None] * [.02, .01])
        np.testing.assert_array_equal(out["history"][0], [0, 0, 0, 4])
        np.testing.assert_array_equal(out["history"][1], [5, 10, 15, 20])
        np.testing.assert_array_equal(out["jepa_indices"][0], [1, 2, 3, 4])
        np.testing.assert_array_equal(out["previous_action_indices"][0], [0, 1, 2, 3])
        self.assertTrue(out["valid_mask"].all())
        self.assertEqual(out["template_index"], 0)
        self.assertFalse(out["training_released"])
        self.assertEqual(out["ownership_audit"]["selected_actions_without_poststate"], 1)

    def test_variable_actual_dt(self):
        dt = np.array([.025, .075, .1, .125, .05, .1, .075] * 6)
        obs, acts = fixture(len(dt)+1, dt=dt)
        out = derive_labels(obs, acts, [8], 4)
        np.testing.assert_allclose(out["trajectory_xy_m"][0], np.arange(1, 8)[:, None] * [.02, .01])
        self.assertTrue(out["valid_mask"][0])
        np.testing.assert_allclose(out["commands"][:, 3], dt/.1)

    def test_yaw_wrap_and_local_axes(self):
        yaw = np.deg2rad(175 + np.arange(35)*1.5)
        obs, acts = fixture(yaw=yaw)
        out = derive_labels(obs, acts, [4], 4)
        expected = np.deg2rad(np.arange(1, 8)*1.5)
        np.testing.assert_allclose(out["relative_yaw_rad"][0], expected, atol=1e-12)
        np.testing.assert_allclose(out["pose"][0, :, 2], np.sin(expected), atol=1e-7)
        delta = np.array([.02, 0., -.01])
        local = delta @ np.asarray(obs[4]["robot_rotation_world_from_body"])
        np.testing.assert_allclose(out["trajectory_xy_m"][0, 0], local[[0, 2]] * [1, -1])

    def test_now_less_than_four_keeps_labels_not_valid(self):
        obs, acts = fixture(takeover=0)
        out = derive_labels(obs, acts, [0, 3, 4], 0)
        np.testing.assert_array_equal(out["valid_mask"], [False, False, True])
        self.assertFalse(out["previous_actions_available"][0].all())
        self.assertTrue(out["previous_actions_available"][2].all())
        self.assertEqual(out["rejection_reasons"][0], ["CURRENT_INDEX_LT_4"])

    def test_prefix_history_allowed_not_labels(self):
        obs, acts = fixture(takeover=10)
        out = derive_labels(obs, acts, [10], 10)
        self.assertLess(out["history"][0, 0], 10)
        self.assertEqual(out["ownership_audit"]["future_transition_indices"][0][0], 10)
        with self.assertRaisesRegex(ValueError, "at/after takeover"):
            derive_labels(obs, acts, [9], 10)

    def test_real_endpoint_and_last_action_never_used(self):
        obs, acts = fixture(n=12)
        out = derive_labels(obs, acts, [4], 4)
        self.assertEqual(out["end_indices"][0], 11)
        self.assertEqual(out["ownership_audit"]["future_transition_indices"][0][-1], 10)
        no_unobserved_action = derive_labels(obs, acts[:-1], [4], 4)
        np.testing.assert_array_equal(out["pose"], no_unobserved_action["pose"])
        with self.assertRaisesRegex(ValueError, "no real future"):
            derive_labels(obs[:-1], acts[:-1], [4], 4)

    def test_missing_transition_or_duplicate_action_rejected(self):
        obs, acts = fixture()
        with self.assertRaises(ValueError):
            derive_labels(obs, acts[:-2], [4], 4)
        acts[6]["sim_step"] = 5
        with self.assertRaises(ValueError):
            derive_labels(obs, acts, [4], 4)

    def test_ownership_boundary_rejected(self):
        for index, field, value in [(4, "owner", "student"), (3, "owner", "teacher"),
                                    (4, "teacher", "unknown"), (5, "teacher", "oracle")]:
            obs, acts = fixture()
            acts[index][field] = value
            with self.assertRaises(ValueError):
                derive_labels(obs, acts, [4], 4)

    def test_actions_finite_range_shape(self):
        for value in ([1.001, 0., 0.], [np.nan, 0., 0.], [0., 0.],
                      [True, False, True], ["0", "0", "0"]):
            obs, acts = fixture()
            acts[5]["normalized_action"] = value
            with self.assertRaises(ValueError):
                derive_labels(obs, acts, [4], 4)

    def test_rotation_rejects_reflection_scale_nonfinite(self):
        for rotation in (np.diag([1., 1., -1.]), np.eye(3)*1.01,
                         np.eye(3)*np.nan, np.zeros((2, 2))):
            obs, acts = fixture()
            obs[5]["robot_rotation_world_from_body"] = rotation.tolist()
            with self.assertRaises(ValueError):
                derive_labels(obs, acts, [4], 4)

    def test_timestamp_and_position_rejections(self):
        for field, value in (("timestamp_s", np.nan), ("timestamp_s", .1),
                             ("robot_position_world", [np.inf, 0., 0.])):
            obs, acts = fixture()
            obs[5][field] = value
            with self.assertRaises(ValueError):
                derive_labels(obs, acts, [4], 4)
        obs, acts = fixture()
        obs[0]["timestamp_s"] = -.1
        with self.assertRaises(ValueError):
            derive_labels(obs, acts, [4], 4)

    def test_strict_indices(self):
        obs, acts = fixture()
        for indices in ([4, 4], [5, 4], [True], [4.0], [-1], [35]):
            with self.assertRaises(ValueError):
                derive_labels(obs, acts, indices, 4)
        for takeover in (True, -1, 4.0, 35):
            with self.assertRaises(ValueError):
                derive_labels(obs, acts, [4], takeover)

    def test_existing_transition_filter_not_redefined(self):
        obs, acts = fixture()
        # A correction in true prefix history excludes a row, not just future labels.
        obs[1]["robot_position_world"][0] = 2.
        out = derive_labels(obs, acts, [4], 4)
        self.assertFalse(out["valid_mask"][0])
        self.assertIn("EXISTING_TRANSITION_FILTER", out["rejection_reasons"][0])

    def test_existing_dt_filter(self):
        obs, acts = fixture(n=36, dt=[.16]+[.1]*34)
        out = derive_labels(obs, acts, [4], 4)
        self.assertFalse(out["valid_mask"][0])
        self.assertTrue(out["transition_bad"][0])

    def test_empty_candidate_shape_and_no_input_mutation(self):
        obs, acts = fixture()
        before = copy.deepcopy((obs, acts))
        out = derive_labels(obs, acts, [], 4)
        self.assertEqual(out["pose"].shape, (0, 7, 4))
        self.assertEqual(out["history"].shape, (0, 4))
        self.assertEqual((obs, acts), before)

    def test_future_target_is_not_consumed(self):
        obs, acts = fixture()
        a = derive_labels(obs, acts, [4], 4)
        for row in obs:
            row["target_position_world_label_only"] = [np.nan, np.inf, -np.inf]
        b = derive_labels(obs, acts, [4], 4)
        np.testing.assert_array_equal(a["pose"], b["pose"])

    def test_mixed_sequence_containers_and_observation_steps(self):
        obs, acts = fixture()
        out = derive_labels(tuple(obs), acts, np.array([4]), 4)
        self.assertTrue(out["valid_mask"][0])
        obs[5]["sim_step"] = 4
        with self.assertRaisesRegex(ValueError, "consecutive"):
            derive_labels(obs, tuple(acts), [4], 4)


if __name__ == "__main__":
    unittest.main()
