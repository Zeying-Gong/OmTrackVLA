"""Pure numeric audit of teacher suffixes; NOT data admission or training release."""
from collections import Counter
import numpy as np
from wa.wm.failure_state_labels import derive_labels


def require(ok, message):
    if not ok:
        raise ValueError(message)


def audit_numeric_windows(observations, actions, windows, takeover_step, *, per_window=True):
    """Return effective_mask = original runtime mask intersected with audited labels.

    The returned derived tensors are diagnostic. Source/pair/repeat/media admission
    is external. Any clipped/unclipped endpoint or mask disagreement is excluded.
    No relaxed validity mask and no supervised student-prefix action are produced.
    """
    require(isinstance(windows, list) and bool(windows), "nonempty candidate windows required")
    indices = [w["current_index"] for w in windows]
    d = derive_labels(observations, actions, indices, takeover_step)
    if per_window:
        for row, i in enumerate(indices):
            one = derive_labels(observations, actions, [i], takeover_step)
            for field in ("pose", "history", "current_indices", "end_indices",
                          "future_bracket_indices", "valid_mask", "jepa_indices",
                          "previous_action_indices"):
                require(np.array_equal(one[field][0], d[field][row]),
                        "individual/batch discrepancy: " + field)
            require(one["rejection_reasons"][0] == d["rejection_reasons"][row],
                    "individual/batch rejection discrepancy")
    times = np.array([o["timestamp_s"] for o in observations], dtype=np.float64)
    pos = np.array([o["robot_position_world"] for o in observations])
    rot = np.array([o["robot_rotation_world_from_body"] for o in observations])
    idx = np.array(indices, dtype=np.int64)
    query = times[idx, None] + np.arange(1, 8) / 10.
    future = np.stack([np.interp(query.ravel(), times, pos[:, a]).reshape(-1, 7)
                       for a in range(3)], axis=-1)
    local = np.einsum("nki,nij->nkj", future-pos[idx, None], rot[idx])
    xy = np.stack([local[..., 0], -local[..., 2]], -1)
    yaw = np.unwrap(np.arctan2(rot[:, 0, 2], rot[:, 0, 0]))
    relative_yaw = np.interp(query.ravel(), times, yaw).reshape(-1, 7)-yaw[idx, None]
    old_dual = np.concatenate([xy, np.sin(relative_yaw)[..., None],
                               np.cos(relative_yaw)[..., None]], -1).astype(np.float32)
    recorded_xy = np.array([w["trajectory_xy_m"] for w in windows])
    require(recorded_xy.shape == xy.shape and np.isfinite(recorded_xy).all(),
            "invalid recorded trajectory")
    require(np.allclose(recorded_xy, xy, rtol=0, atol=1e-6), "executed position labels disagree")
    old_recovery = np.concatenate([recorded_xy.astype(np.float32),
                                   np.sin(relative_yaw)[..., None],
                                   np.cos(relative_yaw)[..., None]], -1).astype(np.float32)
    require(np.allclose(old_dual, d["pose"], rtol=0, atol=1e-6), "old dual SE2 mismatch")
    require(np.allclose(old_recovery, d["pose"], rtol=0, atol=1e-6), "old recovery SE2 mismatch")
    history = np.searchsorted(times, times[idx, None]+[-1.5, -1., -.5, 0.],
                              side="right")-1
    history = np.minimum(np.maximum(history, 0), idx[:, None]).astype(np.int32)
    require(np.array_equal(history, d["history"]), "sparse history mismatch")
    require(np.array_equal(idx[:, None]+np.arange(-3, 1), d["jepa_indices"]),
            "continuous JEPA indices mismatch")
    require(np.array_equal(d["jepa_indices"]-1, d["previous_action_indices"]),
            "preceding command indices mismatch")
    # Independent exact old robot_data.transition_records calculations.
    dt = np.diff(times)
    rel = np.einsum("nji,njk->nik", rot[:-1], rot[1:])
    step_yaw = np.arctan2(rel[:, 0, 2], rel[:, 0, 0])
    distance = np.linalg.norm(np.diff(pos[:, [0, 2]], axis=0), axis=1)
    bad = ((distance > np.hypot(15, 10)*.025+.02)
           | (abs(step_yaw) > 6.28*.025+.01) | (dt < .02) | (dt > .15))
    commands = np.concatenate([np.clip(np.array(
        [a["normalized_action"] for a in actions[:len(times)-1]], dtype=np.float32), -1, 1),
        (dt/.1).astype(np.float32)[:, None]], 1)
    require(np.array_equal(commands, d["commands"]), "old command/dt mismatch")
    require(np.array_equal(bad, d["transition_bad"]), "old transition filter mismatch")
    end = np.searchsorted(times, times[idx]+.7)  # Deliberately NOT clipped.
    cumulative = np.r_[0, np.cumsum(bad)]
    old_mask = ((idx >= 4) & (end < len(times))
                & (cumulative[np.minimum(end, len(bad))] == cumulative[history[:, 0]]))
    mask_diff = old_mask != d["valid_mask"]
    endpoint_diff = end != d["end_indices"]
    effective = old_mask & d["valid_mask"] & ~mask_diff & ~endpoint_diff
    future_owned = d["ownership_audit"]["future_transition_indices"]
    teacher = d["ownership_audit"]["teacher"]
    for row, w in enumerate(windows):
        support = future_owned[row]
        require(w.get("training_eligible") is False, "candidate prematurely eligible")
        require(w["future_bracket_indices"] == d["future_bracket_indices"][row].tolist(),
                "recorded interpolation brackets differ")
        require(np.array_equal(w["future_times_s"], np.arange(1, 8)/10.),
                "seven original future offsets required")
        require(w["teacher_owned_action_indices"] == [support[0], support[-1]],
                "recorded ownership range differs")
        require(w["label_endpoint_observation_index"] == int(d["end_indices"][row]),
                "recorded endpoint differs")
        require(support[0] >= takeover_step and support[-1] < len(times)-1,
                "unobserved or prefix transition in future label")
        require(all(actions[j]["owner"] == "teacher" and actions[j]["teacher"] == teacher
                    for j in support), "future action not teacher-owned")
        require(actions[int(idx[row])]["owner"] == "teacher" and idx[row]+1 < len(times),
                "JEPA next-frame transition not teacher-owned/observed")
    require((history <= idx[:, None]).all() and (history >= 0).all(), "noncausal history")
    require(d["template_index"] == 0 and observations[0]["sim_step"] == 0,
            "episode-zero template changed")
    require(d["command_available"][effective].all()
            and d["previous_actions_available"][effective].all(), "unavailable JEPA conditions")
    age = times[idx]-times[takeover_step]
    early = age <= 1.+1e-8
    reasons = []
    for row in range(len(idx)):
        rr = list(d["rejection_reasons"][row])
        if end[row] >= len(times):
            rr.append("OLD_RUNTIME_ENDPOINT_UNOBSERVED")
        if mask_diff[row]:
            rr.append("OLD_RUNTIME_MASK_DISAGREEMENT")
        if endpoint_diff[row]:
            rr.append("CLIPPED_UNCLIPPED_ENDPOINT_DISAGREEMENT")
        reasons.append(rr)
        require(bool(rr) == (not bool(effective[row])), "missing/extra exclusion reason")
    reason_counts = Counter(r for rr in reasons for r in rr)
    history_prefix = (history < takeover_step).any(axis=1)
    jepa_prefix = (d["jepa_indices"] < takeover_step).any(axis=1)
    proprio_prefix = (d["previous_action_indices"] < takeover_step).any(axis=1)
    deltas = dict(old_dual_pose_max_abs=float(abs(old_dual-d["pose"]).max()),
                  old_recovery_pose_max_abs=float(abs(old_recovery-d["pose"]).max()),
                  recorded_xy_max_abs=float(abs(recorded_xy-xy).max()),
                  old_relative_yaw_max_abs=float(abs(relative_yaw-d["relative_yaw_rad"]).max()))
    summary = dict(candidate_windows=len(idx), derived_valid_windows=int(d["valid_mask"].sum()),
        valid_windows=int(effective.sum()), excluded_windows=int((~effective).sum()),
        early_candidates=int(early.sum()), late_candidates=int((~early).sum()),
        early_valid=int((effective & early).sum()), late_valid=int((effective & ~early).sum()),
        valid_history_uses_student_prefix=int((effective & history_prefix).sum()),
        valid_jepa_history_uses_student_prefix=int((effective & jepa_prefix).sum()),
        valid_proprio_uses_student_prefix=int((effective & proprio_prefix).sum()),
        old_mask_disagreement_windows=int(mask_diff.sum()),
        clipped_unclipped_endpoint_disagreement_windows=int(endpoint_diff.sum()),
        old_endpoint_out_of_range_windows=int((end >= len(times)).sum()),
        future_label_student_action_count=0, jepa_next_target_student_action_count=0,
        selected_actions_without_poststate=max(0, len(actions)-len(times)+1),
        per_window_calls=len(idx) if per_window else 0,
        rejection_reason_counts=dict(reason_counts),
        rejection_reason_overlap_windows=sum(len(rr)>1 for rr in reasons), max_deltas=deltas)
    return dict(derived=d, effective_mask=effective, old_runtime_mask=old_mask,
        unclipped_end=end, summary=summary,
        valid_window_indices=idx[effective].tolist(),
        excluded=[dict(current_index=int(idx[j]),reasons=reasons[j])
                  for j in np.flatnonzero(~effective)],
        old_mask_disagreement_indices=idx[mask_diff].tolist(),
        clipped_unclipped_endpoint_disagreement_indices=idx[endpoint_diff].tolist(),
        training_eligible=False, training_released=False)
