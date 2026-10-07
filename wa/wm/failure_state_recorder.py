"""Evidence-only recording for explicitly authorized evaluation-set adaptation.

This schema does not relax the train-only RecoveryRecorder or the old 4215
start-state release. Raw future target positions are offline observer labels,
never additional policy inputs. Candidate windows are NOT training releases.
"""
import copy
import hashlib
import json
from collections import Counter

import numpy as np

from scripts.da3.teacher_recorder import TeacherRecorder
from wa.wm.initial_bbox_repair import repaired_detector
from wa.wm.recovery_replay import ReplayThenTeacher, dynamic_state, rgb_hash

EXPERIMENT = "evaluation_adaptation_failure_state_v1"
PARTITION = "evaluation_adaptation"
RGB_KEY = "agent_1_articulated_agent_jaw_rgb"


def canonical_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def _flag(value, name):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if np.isfinite(value) and value in (0, 1):
            return bool(value)
    raise ValueError("expected explicit boolean or 0/1: " + name)


class FailureStateRecorder(TeacherRecorder):
    """Observe causal rollout evidence without deciding pair/repeat admission.

    `takeover_step=None` is student-only diagnosis; integer k is a teacher
    branch (including k=0). Step means the evaluator's preaction step, whereas
    current_index/future_bracket_indices refer to observation-array indices.
    """

    def __init__(self, root, metadata, environment_provider, takeover_step=None,
                 repair=None, expected_prefix=None, agent=None):
        metadata = copy.deepcopy(metadata)
        if (metadata.get("experiment") != EXPERIMENT or
                metadata.get("partition") != PARTITION):
            raise ValueError("wrong independent failure-state experiment/partition")
        if not callable(environment_provider):
            raise TypeError("environment_provider must be callable")
        if takeover_step is not None:
            if (isinstance(takeover_step, bool) or
                    not isinstance(takeover_step, (int, np.integer)) or takeover_step < 0):
                raise ValueError("invalid takeover_step")
            takeover_step = int(takeover_step)
            if not expected_prefix or takeover_step >= len(expected_prefix):
                raise ValueError("teacher requires an observed takeover frame: k < N")
            if metadata.get("teacher") not in ("lightnav", "oracle"):
                raise ValueError("teacher branch must name lightnav/oracle")
            prefix_sha = canonical_sha256(expected_prefix)
            if metadata.get("prefix_sha256") != prefix_sha:
                raise ValueError("prefix hash mismatch")
        else:
            if metadata.get("teacher") != "student" or expected_prefix is not None:
                raise ValueError("student branch must not claim a teacher prefix")
            prefix_sha = None
        if ("takeover_step" in metadata and
                metadata["takeover_step"] != takeover_step):
            raise ValueError("metadata takeover differs")
        verification = metadata.get("verification_only", False)
        if not isinstance(verification, bool):
            raise ValueError("verification_only must be boolean")
        metadata.update(takeover_step=takeover_step, verification_only=verification,
                        training_eligible=False, schema=EXPERIMENT,
                        target_positions_use="offline_label_only_never_policy_input",
                        action_recording="selected command before env.step; poststate not fabricated")
        super().__init__(root, metadata)
        self.environment_provider = environment_provider
        self.takeover_step, self.repair, self.agent = takeover_step, repair, agent
        self.expected_prefix = copy.deepcopy(expected_prefix)
        self.prefix_sha256 = prefix_sha
        self.replay, self.fallback_events, self.issues = [], [], []
        self.first_start = self.first = self.takeover = None
        self._fallback_seen = set()
        self._finalized = False
        self._observed_environment_binding = []

    def _issue(self, reason):
        if reason not in self.issues:
            self.issues.append(reason)

    def _write(self, name, value):
        (self.root / name).write_text(json.dumps(
            _json_value(value), indent=2, allow_nan=False) + "\n")

    def _inspect_agent(self, stage, step):
        """Retain transient errors, including teacher warm-up during WA prefix."""
        node, seen = self.agent, set()
        depth = 0
        while node is not None and id(node) not in seen:
            seen.add(id(node))
            indicators = {}
            for key in ("episode_fallback_count", "fallback_count"):
                value = getattr(node, key, 0)
                if value not in (None, 0):
                    indicators[key] = _json_value(value)
            events = getattr(node, "fallback_events", None)
            if events:
                indicators["fallback_events"] = _json_value(events)
            error = getattr(node, "reply_error", None)
            if error is not None:
                indicators["reply_error"] = str(error)
            if indicators:
                # Stringification of unexpected diagnostic types never releases a branch.
                try:
                    signature = canonical_sha256(indicators)
                except (TypeError, ValueError):
                    indicators = {"invalid_fallback_diagnostic": str(indicators)}
                    signature = canonical_sha256(indicators)
                key = (depth, signature)
                if key not in self._fallback_seen:
                    self.fallback_events.append(dict(
                        step=int(step), stage=stage, agent_depth=depth,
                        agent_type=type(node).__name__, indicators=indicators))
                    self._fallback_seen.add(key)
            node = getattr(node, "__dict__", {}).get("teacher")
            depth += 1

    def observe(self, obs, detector, robot, human, episode, step, frequency,
                world_time=None):
        if self._finalized:
            raise RuntimeError("recorder already finalized")
        if isinstance(step, bool) or int(step) != step or step != len(self.frames):
            self._issue("NONCONSECUTIVE_PREACTION_STEP")
            raise ValueError("preaction steps must start at zero and be consecutive")
        step = int(step)
        if self.replay and "action" not in self.replay[-1]:
            self._issue("PREVIOUS_ACTION_MISSING")
            raise ValueError("cannot advance after an unrecorded action")
        self._inspect_agent("preaction", step)
        environment = self.environment_provider()
        if type(self.agent) is ReplayThenTeacher:
            self._observed_environment_binding.append(self.agent.environment is environment)
        state = dynamic_state(environment)
        state_sha = canonical_sha256(state)  # reject NaN observer evidence
        raw_rgb_sha = rgb_hash(obs[RGB_KEY])
        if step == 0:
            original_box = np.asarray(
                detector["agent_1_main_humanoid_detector_sensor"]["box"]).tolist()
            self.metadata["initial_bbox_sensor_xyxy_original"] = original_box
            if self.repair is not None:
                detector = repaired_detector(detector, obs[RGB_KEY], self.repair)
        super().observe(obs, detector, robot, human, episode, step, frequency, world_time)
        if step == 0 and self.repair is not None:
            self.metadata.update(
                initial_bbox_rgb_xyxy=list(self.repair["bbox"]),
                initial_bbox_status="VERIFIED_FROZEN_FIRST_RGB_REPAIR",
                initial_bbox_repair=copy.deepcopy(self.repair))
            # Never replace the original first-frame GT sensor box by a takeover box.
            self.metadata["initial_bbox_sensor_xyxy"] = copy.deepcopy(
                self.metadata["initial_bbox_sensor_xyxy_original"])
        frame = self.frames[-1]
        evidence = dict(
            step=step, observation_index=len(self.frames) - 1,
            timestamp_s=frame["timestamp_s"],
            simulator_world_time_s=frame["simulator_world_time_s"],
            rgb_sha256=raw_rgb_sha, dynamic_state=state,
            dynamic_state_sha256=state_sha)
        self.replay.append(copy.deepcopy(evidence))
        if step == 0:
            evidence.update(
                initial_rgb_sha256=raw_rgb_sha,
                initial_bbox_sensor_xyxy_original=copy.deepcopy(
                    self.metadata["initial_bbox_sensor_xyxy_original"]))
            self.first_start = self.first = copy.deepcopy(evidence)
        if self.takeover_step is not None and step == self.takeover_step:
            evidence.update(
                initial_rgb_sha256=self.first_start["rgb_sha256"],
                initial_bbox_sensor_xyxy_original=copy.deepcopy(
                    self.metadata["initial_bbox_sensor_xyxy_original"]),
                expected_rgb_sha256=self.expected_prefix[step]["rgb_sha256"],
                expected_dynamic_state_sha256=canonical_sha256(
                    self.expected_prefix[step]["dynamic_state"]))
            self.takeover = copy.deepcopy(evidence)

    def record_action(self, step, action, predicted):
        if self._finalized:
            raise RuntimeError("recorder already finalized")
        self._inspect_agent("selected_action", step)
        if (not self.replay or self.replay[-1]["step"] != step or
                "action" in self.replay[-1]):
            self._issue("ACTION_OBSERVATION_MISMATCH")
            raise ValueError("action must match one fresh preaction observation")
        array = np.asarray(action, dtype=float)
        if array.shape != (3,) or not np.isfinite(array).all() or (abs(array) > 1).any():
            self._issue("INVALID_NORMALIZED_ACTION")
            raise ValueError("normalized action must be three finite values in [-1,1]")
        predicted = None if predicted is None else _json_value(predicted)
        try:
            json.dumps(predicted, allow_nan=False)
        except (TypeError, ValueError):
            self._issue("INVALID_PREDICTED_TRAJECTORY")
            raise ValueError("invalid predicted trajectory") from None
        owner = ("teacher" if self.takeover_step is not None and
                 step >= self.takeover_step else "student")
        super().record_action(int(step), array.tolist(), predicted)
        self.actions[-1].update(owner=owner, teacher=self.metadata["teacher"]
                               if owner == "teacher" else None)
        self.replay[-1].update(
            action=array.tolist(), normalized_action=array.tolist(), owner=owner,
            action_timing="before_env_step", post_state_recorded=False)

    def _agent_validation(self):
        # Must run before evaluator's final agent.reset(). No extra policy calls.
        step = getattr(self.agent, "step", None)
        if not isinstance(step, (int, np.integer)) or isinstance(step, bool):
            step = None
        else:
            step = int(step)
        wrapped = type(self.agent) is ReplayThenTeacher
        prefix_matches = False
        if wrapped and self.expected_prefix is not None:
            try:
                prefix_matches = canonical_sha256(self.agent.prefix) == self.prefix_sha256
            except (TypeError, ValueError):
                pass
        environment_bound = bool(wrapped and self.agent.environment is not None
                                 and self._observed_environment_binding
                                 and all(self._observed_environment_binding))
        same_takeover = bool(wrapped and self.agent.takeover == self.takeover_step)
        checked = (min(max(step or 0, 0), self.takeover_step + 1)
                   if wrapped and prefix_matches and environment_bound and same_takeover else 0)
        required = self.takeover_step + 1 if self.takeover_step is not None else 0
        return dict(
            agent_step_before_reset=step,
            replay_wrapper=wrapped, prefix_hash_matches=prefix_matches,
            environment_bound=environment_bound, takeover_matches=same_takeover,
            verified_prefix_frames=checked, required_prefix_frames=required,
            replay_verified=bool(required and checked == required),
            verification_source="successful ReplayThenTeacher.act calls before final reset",
            proof_scope="t=0..k raw RGB plus world time and articulated transforms/joints, atol=1e-6",
            hidden_rng_contact_state_proven=False)

    def _owned_suffix(self):
        pairs = [(i, row["step"]) for i, row in enumerate(self.replay)
                 if row.get("owner") == "teacher" and "action" in row]
        return dict(
            action_count=len(pairs),
            start_observation_index=pairs[0][0] if pairs else None,
            end_observation_index_inclusive=pairs[-1][0] if pairs else None,
            start_step=pairs[0][1] if pairs else None,
            end_step_inclusive=pairs[-1][1] if pairs else None,
            contiguous=bool(pairs) and pairs == list(zip(
                range(pairs[0][0], pairs[-1][0] + 1),
                range(pairs[0][1], pairs[-1][1] + 1))),
            poststate_claim="none; only next preaction frames are observed")

    def _filter_windows(self, windows):
        """Check every transition through all interpolation bracket endpoints."""
        admitted, reasons = [], Counter()
        actions = {a["sim_step"]: a for a in self.actions}
        times = np.asarray([f["timestamp_s"] for f in self.frames], dtype=float)
        for window in windows:
            try:
                i = window["current_index"]
                brackets = np.asarray(window["future_bracket_indices"])
                offsets = np.asarray(window["future_times_s"], dtype=float)
                if (not isinstance(i, int) or self.takeover is None or
                        i < self.takeover["observation_index"] or i >= len(self.frames)):
                    raise ValueError("before_takeover")
                if self.frames[i]["sim_step"] < self.takeover_step:
                    raise ValueError("before_takeover")
                if (brackets.shape != (7, 2) or not np.issubdtype(brackets.dtype, np.integer)
                        or offsets.shape != (7,) or not np.allclose(
                            offsets, np.arange(1, 8) / 10, atol=1e-12, rtol=0)):
                    raise ValueError("invalid_future_brackets")
                if (not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
                        or np.any(brackets < 0) or np.any(brackets >= len(self.frames))):
                    raise ValueError("invalid_timebase")
                query = times[i] + offsets
                right = np.clip(np.searchsorted(times, query, side="left"),
                                1, len(times) - 1)
                if (query[-1] > times[-1] + 1e-8 or
                        not np.array_equal(brackets, np.stack((right - 1, right), axis=-1))):
                    raise ValueError("noncausal_or_incorrect_brackets")
                end = int(right.max())
                if end <= i:
                    raise ValueError("empty_label_interval")
                for j in range(i, end):
                    action = actions.get(self.frames[j]["sim_step"])
                    replay = self.replay[j]
                    if (action is None or action.get("owner") != "teacher" or
                            replay.get("owner") != "teacher" or
                            replay.get("action") != action.get("normalized_action")):
                        raise ValueError("label_interval_not_teacher_owned")
                    values = np.asarray(action["normalized_action"], dtype=float)
                    if values.shape != (3,) or not np.isfinite(values).all() or (abs(values) > 1).any():
                        raise ValueError("invalid_label_interval_action")
                candidate = copy.deepcopy(window)
                candidate.update(
                    teacher_owned_action_indices=[i, end - 1],
                    label_endpoint_observation_index=end,
                    training_eligible=False)
                admitted.append(candidate)
            except (KeyError, TypeError, IndexError, ValueError) as exc:
                reasons[str(exc)] += 1
        return admitted, dict(reasons)

    def _summary(self, result=None):
        self._inspect_agent("finish" if result is not None else "development_finish",
                            len(self.replay))
        validation = self._agent_validation()
        complete = bool(self.frames) and len(self.frames) == len(self.actions)
        complete = complete and all("action" in row for row in self.replay) and not self.issues
        fallback = bool(self.fallback_events)
        summary = dict(
            experiment=EXPERIMENT, partition=PARTITION, schema=EXPERIMENT,
            task=self.metadata.get("task"), key=self.metadata.get("key"),
            teacher=self.metadata["teacher"], takeover_step=self.takeover_step,
            seed=self.metadata.get("seed"), protocol_sha256=self.metadata.get("protocol_sha256"),
            prefix_sha256=self.prefix_sha256,
            initial_rgb_sha256=self.first_start["rgb_sha256"] if self.first_start else None,
            takeover_state_sha256=self.takeover["expected_dynamic_state_sha256"]
            if self.takeover else None,
            actual_takeover_state_sha256=self.takeover["dynamic_state_sha256"]
            if self.takeover else None,
            takeover_observation_index=self.takeover["observation_index"] if self.takeover else None,
            complete=bool(complete and result is not None),
            replay_verified=validation["replay_verified"],
            transport_fallback=fallback, agent_validation=validation,
            owned_suffix=self._owned_suffix(), issues=list(self.issues),
            verification_only=self.metadata["verification_only"],
            training_eligible=False, training_released=False)
        if result is not None:
            summary["result"] = _json_value(result)
        return summary

    def finish(self, result):
        if self._finalized:
            raise RuntimeError("recorder already finalized")
        result = copy.deepcopy(_json_value(result))
        success, collision = _flag(result["success"], "success"), _flag(result["collision"], "collision")
        if "policy_init_valid" not in result:
            raise ValueError("missing required policy_init_valid")
        init_valid = _flag(result["policy_init_valid"], "policy_init_valid")
        summary = self._summary(result)
        # Keep the real executed-pose interpolation and raw result unchanged.
        super().finish(result)
        raw = json.loads((self.root / "windows.json").read_text())
        self._write("raw_windows.json", raw)
        windows, reasons = self._filter_windows(raw)
        student = self.takeover_step is None
        eligible = bool(not student and success and not collision and init_valid and summary["complete"]
                        and summary["replay_verified"] and not summary["transport_fallback"]
                        and self.takeover is not None and not self.metadata["verification_only"])
        if not eligible:
            windows = []
        summary.update(
            candidate_windows=len(windows), raw_windows=len(raw),
            window_rejections=reasons,
            status=("rerun_student_success" if success else "student_failure_recorded")
            if student else ("verification_only" if self.metadata["verification_only"]
                             else "teacher_suffix_candidate" if eligible else "teacher_branch_rejected"),
            no_recovery_needed=bool(student and success),
            pending="same-k paired selection, one additional winner repeat and independent SE2/input audit")
        self._write("windows.json", windows)
        self._write("replay.json", self.replay)
        self._write("first_start.json", self.first_start)
        self._write("pair_start.json", self.first_start)
        self._write("takeover.json", self.takeover)
        self._write("fallback_events.json", self.fallback_events)
        self._write("admission.json", summary)
        complete = json.loads((self.root / "complete.json").read_text())
        complete.update(
            schema=EXPERIMENT, status="FAILURE_STATE_RAW_COLLECTION_COMPLETE",
            complete=summary["complete"], windows=len(windows), raw_windows=len(raw),
            candidate_windows=len(windows), training_eligible=False,
            training_released=False, blocker=summary["pending"])
        self._write("complete.json", complete)
        self._finalized = True
        return summary

    def finish_development(self, reason):
        """Finalize genuine partial evidence, with NO terminal result or SR."""
        if self._finalized:
            raise RuntimeError("recorder already finalized")
        summary = self._summary()
        summary.update(
            status="DEVELOPMENT_PARTIAL_EVIDENCE_ONLY", reason=str(reason),
            no_success_rate=True, no_terminal_result=True,
            training_eligible=False, training_released=False,
            metadata=self.metadata, observations=self.frames, actions=self.actions,
            replay=self.replay, first_start=self.first_start, takeover=self.takeover,
            fallback_events=self.fallback_events)
        self._write("development.json", summary)
        self._finalized = True
        return summary
