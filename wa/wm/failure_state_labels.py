"""Numeric-only failure-state SE(2) labels; never an admission or release.

Future robot poses are supervised targets only. The episode-zero template and
causal history may include the real student prefix; teacher ownership is required
for every transition contributing to a future label. No files or model inputs
are read here. A caller must independently audit branch/pair/repeat provenance.
"""
import numpy as np

OFFSETS = np.arange(1, 8, dtype=np.float64) / 10.
HISTORY_OFFSETS = np.array([-1.5, -1., -.5, 0.], dtype=np.float64)
ROTATION_ATOL = 1e-5
ENDPOINT_ATOL = 1e-8  # Existing recorder's floating point endpoint tolerance.


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ": integer required")
    return int(value)


def _numeric(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iuf":
        raise ValueError(name + ": numeric shape " + str(shape) + " required")
    array = array.astype(np.float64)
    if not np.isfinite(array).all():
        raise ValueError(name + ": nonfinite value")
    return array


def derive_labels(observations, actions, window_indices, takeover_step):
    """Return NumPy labels/indices and the unchanged robot validity filter.

    Schema errors and unsupported future/ownership evidence raise ValueError.
    Valid numeric candidates with now < 4 or old transition outliers retain
    their labels but have valid_mask=False. Negative JEPA/proprio indices for
    those rows are diagnostic only; callers must apply availability/valid masks.
    Complete recorder branches have N observations and N selected actions: only
    N-1 transitions have a real poststate. N-1 actions are also sufficient for
    numeric evidence, but this function never claims a branch is complete.
    """
    if not isinstance(observations, (list, tuple)) or len(observations) < 2:
        raise ValueError("observations: at least two real states required")
    if not isinstance(actions, (list, tuple)):
        raise ValueError("actions: recorded sequence required")
    n = len(observations)
    takeover = _integer(takeover_step, "takeover_step")
    if not 0 <= takeover < n:
        raise ValueError("takeover_step: outside observations")
    if len(actions) not in (n - 1, n):
        raise ValueError("actions: each observed transition needs one action")
    if any(not isinstance(row, dict) for row in tuple(observations) + tuple(actions)):
        raise ValueError("observations/actions: dictionaries required")
    if any(_integer(row.get("sim_step"), "observation sim_step") != j
           for j, row in enumerate(observations)):
        raise ValueError("observations: consecutive episode-zero sim_step required")
    times = _numeric([row.get("timestamp_s") for row in observations], (n,), "timestamps")
    if abs(times[0]) > ENDPOINT_ATOL or (np.diff(times) <= 0).any():
        raise ValueError("timestamps: episode-zero and strict monotonicity required")
    pos = _numeric([row.get("robot_position_world") for row in observations], (n, 3), "positions")
    rot = _numeric([row.get("robot_rotation_world_from_body") for row in observations],
                   (n, 3, 3), "rotations")
    gram = np.einsum("nji,njk->nik", rot, rot)
    if (not np.allclose(gram, np.eye(3)[None], atol=ROTATION_ATOL, rtol=0)
            or not np.allclose(np.linalg.det(rot), 1., atol=ROTATION_ATOL, rtol=0)):
        raise ValueError("rotations: proper orthonormal world-from-body matrices required")
    raw_commands = []
    teachers = set()
    for j, row in enumerate(actions):
        if _integer(row.get("sim_step"), "action sim_step") != j:
            raise ValueError("actions: consecutive unique sim_step required")
        command = _numeric(row.get("normalized_action"), (3,), "action")
        if (abs(command) > 1).any():
            raise ValueError("actions: normalized command outside [-1,1]")
        expected_owner = "teacher" if j >= takeover else "student"
        if row.get("owner") != expected_owner:
            raise ValueError("actions: takeover ownership mismatch at " + str(j))
        if expected_owner == "teacher":
            if row.get("teacher") not in ("lightnav", "oracle"):
                raise ValueError("actions: unknown teacher")
            teachers.add(row["teacher"])
        elif row.get("teacher") is not None:
            raise ValueError("actions: student prefix cannot claim a teacher")
        raw_commands.append(command)
    if len(teachers) > 1:
        raise ValueError("actions: teacher identity changes within branch")
    if not isinstance(window_indices, (list, tuple, np.ndarray)):
        raise ValueError("window_indices: index sequence required")
    indices = np.array([_integer(i, "window index") for i in window_indices], dtype=np.int64)
    if indices.ndim != 1 or (len(indices) and
            ((indices < takeover).any() or (indices >= n).any() or (np.diff(indices) <= 0).any())):
        raise ValueError("window_indices: unique increasing indices at/after takeover required")
    query = times[indices, None] + OFFSETS
    if len(indices) and (query[:, -1] > times[-1] + ENDPOINT_ATOL).any():
        raise ValueError("label endpoint: no real future observation; extrapolation forbidden")
    # np.interp agrees with the recorder at floating-point endpoint equality.
    right = np.clip(np.searchsorted(times, query, side="left"), 1, n - 1)
    brackets = np.stack([right - 1, right], axis=-1)
    end = right.max(axis=1) if len(indices) else np.empty(0, dtype=np.int64)
    owned = []
    for now, last in zip(indices, end):
        if last <= now:
            raise ValueError("label interval: empty future interval")
        support = list(range(int(now), int(last)))
        if any(actions[j]["owner"] != "teacher" for j in support):
            raise ValueError("label interval: non-teacher transition")
        owned.append(support)
    futurepos = np.stack([np.interp(query.ravel(), times, pos[:, axis]).reshape(-1, 7)
                          for axis in range(3)], axis=-1)
    local = np.einsum("nki,nij->nkj", futurepos - pos[indices, None], rot[indices])
    xy = np.stack([local[..., 0], -local[..., 2]], axis=-1)
    yaw = np.unwrap(np.arctan2(rot[:, 0, 2], rot[:, 0, 0]))
    relative_yaw = np.interp(query.ravel(), times, yaw).reshape(-1, 7) - yaw[indices, None]
    pose = np.concatenate([xy, np.sin(relative_yaw)[..., None],
                           np.cos(relative_yaw)[..., None]], axis=-1).astype(np.float32)
    history = np.searchsorted(times, times[indices, None] + HISTORY_OFFSETS, side="right") - 1
    history = np.minimum(np.maximum(history, 0), indices[:, None]).astype(np.int32)
    jepa = indices[:, None] + np.arange(-3, 1)
    previous = jepa - 1
    transition_count = n - 1
    command_available = (jepa >= 0) & (jepa < transition_count)
    previous_available = (previous >= 0) & (previous < transition_count)
    dt = np.diff(times)
    relative_rot = np.einsum("nji,njk->nik", rot[:-1], rot[1:])
    step_yaw = np.arctan2(relative_rot[:, 0, 2], relative_rot[:, 0, 0])
    distance = np.linalg.norm(np.diff(pos[:, [0, 2]], axis=0), axis=1)
    # Exact existing robot_data.transition_records exclusion rules.
    bad = ((distance > np.hypot(15, 10) * .025 + .02)
           | (abs(step_yaw) > 6.28 * .025 + .01) | (dt < .02) | (dt > .15))
    cumulative = np.r_[0, np.cumsum(bad)]
    valid = (indices >= 4) & (end < n)
    valid &= cumulative[end] == cumulative[history[:, 0]]
    rejections = []
    for row, (now, last) in enumerate(zip(indices, end)):
        reasons = []
        if now < 4:
            reasons.append("CURRENT_INDEX_LT_4")
        if bad[int(history[row, 0]):int(last)].any():
            reasons.append("EXISTING_TRANSITION_FILTER")
        rejections.append(reasons)
    commands = np.concatenate([np.asarray(raw_commands[:transition_count], dtype=np.float32),
                               (dt / .1).astype(np.float32)[:, None]], axis=1)
    return dict(
        pose=pose, trajectory_xy_m=xy, relative_yaw_rad=relative_yaw,
        history=history, current_indices=indices, end_indices=end,
        future_bracket_indices=brackets, future_timestamps_s=query,
        template_index=0, jepa_indices=jepa, command_indices=jepa.copy(),
        previous_action_indices=previous, command_available=command_available,
        previous_actions_available=previous_available, commands=commands,
        valid_mask=valid, valid_window_indices=indices[valid],
        rejection_reasons=rejections, transition_bad=bad,
        ownership_audit=dict(takeover_step=takeover, teacher=next(iter(teachers), None),
            future_transition_indices=owned, prefix_allowed_for_history=True,
            real_transition_count=transition_count,
            selected_actions_without_poststate=max(0, len(actions) - transition_count),
            future_labels_teacher_owned=True),
        training_eligible=False, training_released=False,
        scope="numeric labels only; branch/pair/repeat admission is external")
