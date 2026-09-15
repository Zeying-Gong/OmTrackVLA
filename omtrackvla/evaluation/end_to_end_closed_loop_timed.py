"""NEXT-027 Habitat closed-loop runner for the Architecture-v1 policy.

Torch and the DA3 implementation are imported only after Habitat has rendered
the first frame.  Habitat-Sim owns the EGL context, and creating a CUDA context
before that first render can produce black camera observations on this host.

The policy decision boundary is deliberately narrow: one initialization RGB
frame and bbox are retained at reset, while every later decision receives only
the current RGB frame and an optional UWB sensor sample.  Simulator poses,
panoptic labels, depth, and point goals are used only after an action for
evaluation metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from omtrackvla.evaluation.closed_loop_timing import (
    StageTimer, elapsed_ms, summarize_steps, runtime_identity, timing_contract,
    append_step_timing, assert_new_rollout_artifacts,
)

RGB_KEY = "agent_1_articulated_agent_jaw_rgb"
PANOPTIC_KEY = "agent_1_articulated_agent_jaw_panoptic"
INITIALIZATION_SENSOR_KEY = "agent_1_main_humanoid_detector_sensor"
ACTION_NAMES = (
    "agent_0_humanoid_navigate_action",
    "agent_1_base_velocity",
    "agent_2_oracle_nav_randcoord_action_obstacle",
    "agent_3_oracle_nav_randcoord_action_obstacle",
    "agent_4_oracle_nav_randcoord_action_obstacle",
    "agent_5_oracle_nav_randcoord_action_obstacle",
    "agent_6_oracle_nav_randcoord_action_obstacle",
    "agent_7_oracle_nav_randcoord_action_obstacle",
    "agent_8_oracle_nav_randcoord_action_obstacle",
)
DEFAULT_SCENE_DATASET = (
    "data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
)


@dataclass(frozen=True)
class ContinuousAction:
    forward: float
    lateral: float
    yaw: float

    def as_habitat(self) -> list[float]:
        return [self.forward, self.lateral, self.yaw]


def configure(config: Any, scene_dataset: str) -> Any:
    from habitat.config import read_write

    with read_write(config):
        config.habitat.simulator.scene_dataset = scene_dataset
        measurements = config.habitat.task.measurements
        if "top_down_map_following" in measurements:
            del measurements["top_down_map_following"]
    return config


def target_mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    binary = np.asarray(mask, dtype=bool).squeeze()
    if binary.ndim != 2 or not binary.any():
        return None
    ys, xs = np.nonzero(binary)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def initialization_bbox(
    task_observations: Mapping[str, Any], raw_observations: Mapping[str, Any],
    target_semantic_id: int,
) -> tuple[tuple[int, int, int, int] | None, str | None]:
    """Read the benchmark's one-time initialization bbox, with mask fallback."""

    detector = task_observations.get(INITIALIZATION_SENSOR_KEY)
    if isinstance(detector, Mapping) and "box" in detector:
        box = np.asarray(detector["box"], dtype=np.float32).reshape(-1)
        if box.size == 4 and box[2] > box[0] and box[3] > box[1]:
            return tuple(int(round(float(value))) for value in box), "task_initialization_bbox"
    mask = (
        np.asarray(raw_observations[PANOPTIC_KEY]).squeeze()
        == int(target_semantic_id)
    )
    return target_mask_to_bbox(mask), "first_frame_panoptic_mask"


DEPLOYMENT_MODEL_INPUT_KEYS = (
    "initial_rgb",
    "initial_bbox",
    "ego_rgb",
    "visual_initialization_valid",
    "rgb_valid",
    "binding_valid",
    "uwb_xy",
    "uwb_covariance_xy",
    "uwb_quality",
    "uwb_age_s",
    "uwb_valid",
    "camera_intrinsics",
    "camera_from_base",
    "hidden_state",
    "_target_memory_override",
)
FORBIDDEN_DECISION_NAMES = {
    "depth",
    "gt_bbox",
    "gt_depth",
    "gt_motion",
    "observations",
    "panoptic",
    "perception_cache",
    "pointgoal",
    "robot",
    "sim",
    "target",
    "target_agent",
    "target_point",
    "target_pose",
    "target_position",
}


@dataclass(frozen=True)
class UWBSensorSample:
    """A real sensor-shaped UWB sample, never an oracle target point."""

    xy_m: tuple[float, float]
    covariance_m2: tuple[tuple[float, float], tuple[float, float]]
    quality: float
    age_s: float
    valid: bool


@dataclass(frozen=True)
class ClosedLoopDecision:
    action: ContinuousAction
    mode: str
    waypoints: tuple[tuple[float, float], ...]
    stop_probability: float
    visibility_probability: float
    predicted_bbox_xyxy_norm: tuple[float, float, float, float]
    selected_waypoint_index: int
    inference_ms: float


class WaypointActionAdapter:
    """Turn a receding-horizon local path into one bounded Habitat command."""

    def __init__(
        self,
        *,
        lookahead_m: float = 0.15,
        translation_gain: float = 2.0,
        heading_gain: float = 0.75,
        stop_threshold: float = 0.80,
        max_forward: float = 1.0,
        max_reverse: float = 0.30,
        max_lateral: float = 0.75,
        max_yaw: float = 0.75,
        translation_slew: float = 0.30,
        yaw_slew: float = 0.30,
    ) -> None:
        if lookahead_m <= 0.0 or translation_gain <= 0.0:
            raise ValueError("lookahead and translation gain must be positive")
        if not 0.0 < stop_threshold < 1.0:
            raise ValueError("stop threshold must lie in (0,1)")
        self.lookahead_m = float(lookahead_m)
        self.translation_gain = float(translation_gain)
        self.heading_gain = float(heading_gain)
        self.stop_threshold = float(stop_threshold)
        self.max_forward = float(max_forward)
        self.max_reverse = float(max_reverse)
        self.max_lateral = float(max_lateral)
        self.max_yaw = float(max_yaw)
        self.translation_slew = float(translation_slew)
        self.yaw_slew = float(yaw_slew)
        self.reset()

    def reset(self) -> None:
        self._previous = np.zeros(3, dtype=np.float32)

    def __call__(
        self, waypoints: np.ndarray, stop_probability: float
    ) -> tuple[ContinuousAction, str, int]:
        points = np.asarray(waypoints, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
            raise ValueError("waypoints must be [H,2] with H >= 2")
        if not np.isfinite(points).all() or not math.isfinite(stop_probability):
            self._previous[:] = 0.0
            return ContinuousAction(0.0, 0.0, 0.0), "invalid_prediction_stop", 0
        if stop_probability >= self.stop_threshold:
            self._previous[:] = 0.0
            return ContinuousAction(0.0, 0.0, 0.0), "policy_stop", 0

        radius = np.linalg.norm(points, axis=1)
        candidates = np.flatnonzero(radius >= self.lookahead_m)
        selected = int(candidates[0]) if candidates.size else int(np.argmax(radius))
        selected = max(1, selected)
        forward_m, left_m = (float(value) for value in points[selected])
        heading = math.atan2(left_m, max(forward_m, 1.0e-4))
        requested = np.asarray(
            [
                np.clip(
                    self.translation_gain * forward_m,
                    -self.max_reverse,
                    self.max_forward,
                ),
                np.clip(
                    self.translation_gain * left_m,
                    -self.max_lateral,
                    self.max_lateral,
                ),
                np.clip(
                    self.heading_gain * heading,
                    -self.max_yaw,
                    self.max_yaw,
                ),
            ],
            dtype=np.float32,
        )
        limits = np.asarray(
            [self.translation_slew, self.translation_slew, self.yaw_slew],
            dtype=np.float32,
        )
        command = self._previous + np.clip(
            requested - self._previous, -limits, limits
        )
        self._previous = command
        return (
            ContinuousAction(*(float(value) for value in command)),
            "waypoint_follow",
            selected,
        )


def deployment_action(
    action_adapter: WaypointActionAdapter,
    waypoints: np.ndarray,
    stop_probability: float,
    visibility_probability: float,
    predicted_bbox_xyxy_norm: Sequence[float] | None = None,
    *,
    uwb_valid: bool,
    visual_stop_threshold: float = 0.90,
    proximity_bbox_height_threshold: float = 0.85,
) -> tuple[ContinuousAction, str, int]:
    """Apply a sensor-only fail-safe before converting waypoints to motion."""

    if not 0.0 < visual_stop_threshold < 1.0:
        raise ValueError("visual stop threshold must lie in (0,1)")
    if not 0.0 < proximity_bbox_height_threshold <= 1.0:
        raise ValueError("proximity bbox threshold must lie in (0,1]")
    if not math.isfinite(visibility_probability):
        action_adapter.reset()
        return ContinuousAction(0.0, 0.0, 0.0), "invalid_visibility_stop", 0
    # With no independent UWB confirmation, visual uncertainty always wins.
    # This ordering keeps the three-axis fail-safe intact even when a stale
    # predicted box is spuriously large.
    if not uwb_valid and visibility_probability < visual_stop_threshold:
        action_adapter.reset()
        return ContinuousAction(0.0, 0.0, 0.0), "visual_uncertain_stop", 0
    if predicted_bbox_xyxy_norm is not None:
        bbox = np.asarray(predicted_bbox_xyxy_norm, dtype=np.float32)
        if bbox.shape != (4,) or not np.isfinite(bbox).all():
            action_adapter.reset()
            return ContinuousAction(0.0, 0.0, 0.0), "invalid_bbox_stop", 0
        x1, y1, x2, y2 = (float(value) for value in bbox)
        if x2 <= x1 or y2 <= y1:
            action_adapter.reset()
            return ContinuousAction(0.0, 0.0, 0.0), "invalid_bbox_stop", 0
        if y2 - y1 >= proximity_bbox_height_threshold:
            action, mode, selected = action_adapter(waypoints, stop_probability)
            if mode != "waypoint_follow":
                return action, mode, selected
            # Do not drive closer, but retain the learned lateral/yaw command so
            # a nearby target walking across the image remains in view.
            held = ContinuousAction(
                min(0.0, action.forward), action.lateral, action.yaw
            )
            action_adapter._previous = np.asarray(
                held.as_habitat(), dtype=np.float32
            )
            return held, "visual_proximity_hold", selected
    return action_adapter(waypoints, stop_probability)


def normalized_bbox(
    bbox_xyxy: Sequence[float], image_shape: Sequence[int]
) -> tuple[float, float, float, float]:
    if len(bbox_xyxy) != 4 or len(image_shape) < 2:
        raise ValueError("bbox/image shape mismatch")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image must be non-empty")
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    result = (x1 / width, y1 / height, x2 / width, y2 / height)
    if not (
        0.0 <= result[0] < result[2] <= 1.0
        and 0.0 <= result[1] < result[3] <= 1.0
    ):
        raise ValueError("initial bbox must be a valid in-frame xyxy box")
    return result


def habitat_camera_calibration(
    image_shape: Sequence[int], output_height: int, output_width: int, hfov_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return resized pinhole intrinsics and canonical-base to OpenCV camera."""

    source_height, source_width = int(image_shape[0]), int(image_shape[1])
    if source_height <= 0 or source_width <= 0 or not 0.0 < hfov_deg < 180.0:
        raise ValueError("invalid Habitat camera geometry")
    focal_source = 0.5 * source_width / math.tan(math.radians(hfov_deg) * 0.5)
    intrinsics = np.asarray(
        [
            [focal_source * output_width / source_width, 0.0, 0.5 * output_width],
            [0.0, focal_source * output_height / source_height, 0.5 * output_height],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    # canonical base x-forward/y-left/z-up -> OpenCV x-right/y-down/z-forward.
    camera_from_base = np.eye(4, dtype=np.float32)
    camera_from_base[:3, :3] = np.asarray(
        [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    return intrinsics, camera_from_base


def decision_api_audit(controller: Any) -> dict[str, Any]:
    parameters = [
        name
        for name in inspect.signature(controller.decide).parameters
        if name != "self"
    ]
    forbidden = sorted(set(parameters) & FORBIDDEN_DECISION_NAMES)
    if forbidden:
        raise ValueError(f"closed-loop decision API exposes privileged inputs: {forbidden}")
    if parameters != ["rgb", "uwb"]:
        raise ValueError(f"closed-loop decision API changed unexpectedly: {parameters}")
    return {
        "passed": True,
        "decision_parameters": parameters,
        "model_input_keys": list(DEPLOYMENT_MODEL_INPUT_KEYS),
        "forbidden_names": sorted(FORBIDDEN_DECISION_NAMES),
        "gt_target_point_used": False,
        "later_bbox_used": False,
        "depth_used": False,
        "perception_cache_used": False,
    }


class ArchitectureV1HabitatController:
    """Stateful Architecture-v1 controller with an audited sensor-only API."""

    def __init__(
        self,
        policy: Any,
        architecture: Any,
        device: Any,
        initial_rgb: np.ndarray,
        initial_bbox_xyxy_norm: Sequence[float],
        camera_intrinsics: np.ndarray,
        camera_from_base: np.ndarray,
        action_adapter: WaypointActionAdapter | None = None,
        visual_stop_threshold: float = 0.90,
    ) -> None:
        import torch

        self.torch = torch
        self.policy = policy
        self.architecture = architecture
        self.device = device
        self.action_adapter = action_adapter or WaypointActionAdapter()
        if not 0.0 < visual_stop_threshold < 1.0:
            raise ValueError("visual stop threshold must lie in (0,1)")
        self.visual_stop_threshold = float(visual_stop_threshold)
        self.initial_rgb = self._rgb_tensor(initial_rgb)
        self.initial_bbox = torch.tensor(
            initial_bbox_xyxy_norm, dtype=torch.float32, device=device
        )[None]
        self.camera_intrinsics = torch.tensor(
            camera_intrinsics, dtype=torch.float32, device=device
        )[None]
        self.camera_from_base = torch.tensor(
            camera_from_base, dtype=torch.float32, device=device
        )[None]
        self.history: deque[Any] = deque(maxlen=int(architecture.history_size))
        first = self._rgb_tensor(initial_rgb)
        for _ in range(int(architecture.history_size)):
            self.history.append(first)
        self.hidden_state = None
        self.target_memory = None
        self.binding_valid = torch.ones(1, dtype=torch.float32, device=device)

    def _rgb_tensor(self, rgb: np.ndarray) -> Any:
        frame = np.asarray(rgb)[..., :3].astype(np.uint8)
        resized = cv2.resize(
            frame,
            (int(self.architecture.image_width), int(self.architecture.image_height)),
            interpolation=cv2.INTER_LINEAR,
        )
        tensor = self.torch.from_numpy(resized.copy()).to(
            device=self.device, dtype=self.torch.float32
        )
        return tensor.permute(2, 0, 1).contiguous().div_(255.0)

    def decide(
        self, rgb: np.ndarray, uwb: UWBSensorSample | None = None
    ) -> ClosedLoopDecision:
        decision_timer = StageTimer()
        torch = self.torch
        self.history.append(self._rgb_tensor(rgb))
        decision_timer.mark("rgb_resize_transfer_history")
        if uwb is None:
            uwb = UWBSensorSample(
                xy_m=(0.0, 0.0),
                covariance_m2=((1.0, 0.0), (0.0, 1.0)),
                quality=0.0,
                age_s=0.0,
                valid=False,
            )
        model_inputs = {
            "initial_rgb": self.initial_rgb[None],
            "initial_bbox": self.initial_bbox,
            "ego_rgb": torch.stack(tuple(self.history), dim=0)[None],
            "visual_initialization_valid": torch.ones(1, device=self.device),
            "rgb_valid": torch.ones(1, device=self.device),
            "binding_valid": self.binding_valid,
            "uwb_xy": torch.tensor(uwb.xy_m, dtype=torch.float32, device=self.device)[None],
            "uwb_covariance_xy": torch.tensor(
                uwb.covariance_m2, dtype=torch.float32, device=self.device
            )[None],
            "uwb_quality": torch.tensor([uwb.quality], dtype=torch.float32, device=self.device),
            "uwb_age_s": torch.tensor([uwb.age_s], dtype=torch.float32, device=self.device),
            "uwb_valid": torch.tensor([uwb.valid], dtype=torch.float32, device=self.device),
            "camera_intrinsics": self.camera_intrinsics,
            "camera_from_base": self.camera_from_base,
            "hidden_state": self.hidden_state,
            "_target_memory_override": self.target_memory,
        }
        if tuple(model_inputs) != DEPLOYMENT_MODEL_INPUT_KEYS:
            raise RuntimeError("Architecture-v1 deployment input contract drifted")
        decision_timer.mark("model_inputs_build")
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda" and torch.cuda.is_bf16_supported(),
        ):
            outputs = self.policy(**model_inputs)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_ms = (time.perf_counter() - start) * 1000.0
        decision_timer.mark("model_forward_and_existing_cuda_sync")
        self.hidden_state = outputs["hidden_state"]
        self.target_memory = outputs["target_memory_next"]
        waypoints = outputs["waypoints"][0].detach().float().cpu().numpy()
        stop_probability = float(
            torch.sigmoid(outputs["stop_logit"][0, 0]).detach().float().cpu()
        )
        visibility_probability = float(
            torch.sigmoid(outputs["visibility_logit"][0, 0]).detach().float().cpu()
        )
        bbox = outputs["bbox_pred"][0].detach().float().cpu().numpy()
        decision_timer.mark("prediction_transfer_and_postprocess")
        action, mode, selected = deployment_action(
            self.action_adapter,
            waypoints,
            stop_probability,
            visibility_probability,
            bbox,
            uwb_valid=bool(uwb.valid),
            visual_stop_threshold=self.visual_stop_threshold,
        )
        decision_timer.mark("action_controller")
        decision = ClosedLoopDecision(
            action=action,
            mode=mode,
            waypoints=tuple(tuple(float(v) for v in point) for point in waypoints),
            stop_probability=stop_probability,
            visibility_probability=visibility_probability,
            predicted_bbox_xyxy_norm=tuple(float(v) for v in bbox),
            selected_waypoint_index=selected,
            inference_ms=inference_ms,
        )
        decision_timer.mark("decision_pack")
        self.last_timings = {
            "decision_ms": decision_timer.durations,
            "decision_total_ms": decision_timer.total_ms,
        }
        return decision


def _architecture_v1_policy_worker(connection: Any, kwargs: Mapping[str, Any]) -> None:
    """Own Torch/DA3 CUDA state outside the Habitat EGL process."""

    try:
        worker_startup = StageTimer()
        policy, architecture, device, loading = _load_policy_after_first_render(
            Path(kwargs["model_config"]),
            Path(kwargs["checkpoint"]),
            str(kwargs["device_name"]),
        )
        worker_startup.mark("torch_da3_checkpoint_load")
        controller = ArchitectureV1HabitatController(
            policy=policy,
            architecture=architecture,
            device=device,
            initial_rgb=kwargs["initial_rgb"],
            initial_bbox_xyxy_norm=kwargs["initial_bbox_xyxy_norm"],
            camera_intrinsics=kwargs["camera_intrinsics"],
            camera_from_base=kwargs["camera_from_base"],
        )
        worker_startup.mark("controller_initialization_and_initial_rgb")
        worker_runtime = runtime_identity()
        worker_runtime.update({
            "torch_version": str(controller.torch.__version__),
            "torch_num_threads": controller.torch.get_num_threads(),
            "torch_num_interop_threads": controller.torch.get_num_interop_threads(),
            "device": str(device),
            "gpu_name": controller.torch.cuda.get_device_name(device),
        })
        connection.send({
            "event": "ready",
            "timing_startup_ms": worker_startup.durations,
            "timing_runtime": worker_runtime,
            "device": str(device),
            "loading": loading,
            "audit": decision_api_audit(controller),
        })
        while True:
            receive_start = time.perf_counter_ns()
            request = connection.recv()
            receive_wait_ms = elapsed_ms(receive_start)
            command = request[0]
            if command == "close":
                break
            if command != "infer":
                raise ValueError(f"unknown Architecture-v1 worker command: {command}")
            decision = controller.decide(request[1], request[2])
            connection.send({
                "event": "result", "decision": decision,
                "timing": {**controller.last_timings,
                    "request_receive_wait_ms_overlaps_parent": receive_wait_ms},
            })
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            connection.send({
                "event": "error",
                "error": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            })
        except (EOFError, BrokenPipeError):
            pass
    finally:
        connection.close()


class ArchitectureV1PolicyWorker:
    """Run DA3 policy CUDA inference in a child process so EGL stays live."""

    def __init__(self, **kwargs: Any) -> None:
        parent_startup = StageTimer()
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        self._connection = parent
        self._process = context.Process(
            target=_architecture_v1_policy_worker,
            args=(child, kwargs),
            daemon=True,
            name="architecture-v1-policy",
        )
        self._process.start()
        child.close()
        parent_startup.mark("spawn_start_and_pipe_setup")
        response = self._connection.recv()
        self._check_response(response, "ready")
        self.device = response["device"]
        self.loading = response["loading"]
        self.worker_audit = response["audit"]
        parent_startup.mark("wait_for_worker_ready")
        self.timing_startup = {
            "parent_ms": parent_startup.durations,
            "worker_ms_nested": response["timing_startup_ms"],
            "worker_runtime": response["timing_runtime"],
        }
        self.last_timings = {}

    @staticmethod
    def _check_response(response: Mapping[str, Any], expected_event: str) -> None:
        if response.get("event") == "error":
            raise RuntimeError(
                "Architecture-v1 policy worker failed: "
                f"{response['error']}: {response['message']}\n"
                f"{response['traceback']}"
            )
        if response.get("event") != expected_event:
            raise RuntimeError(
                f"expected policy worker event {expected_event}, got {response}"
            )

    def decide(
        self, rgb: np.ndarray, uwb: UWBSensorSample | None = None
    ) -> ClosedLoopDecision:
        ipc_timer = StageTimer()
        self._connection.send(("infer", np.asarray(rgb), uwb))
        ipc_timer.mark("parent_prepare_serialize_send")
        response = self._connection.recv()
        ipc_timer.mark("parent_wait_receive_deserialize_including_worker")
        self._check_response(response, "result")
        ipc_timer.mark("response_check")
        self.last_timings = {
            **response["timing"], "parent_ipc_ms_nested": ipc_timer.durations,
            "parent_exchange_total_ms": ipc_timer.total_ms,
            "exchange_minus_worker_decide_ms_residual": (
                ipc_timer.total_ms - response["timing"]["decision_total_ms"]
            ),
        }
        return response["decision"]

    def close(self) -> None:
        if getattr(self, "_connection", None) is None:
            return
        try:
            self._connection.send(("close",))
        except (EOFError, BrokenPipeError):
            pass
        self._connection.close()
        self._connection = None
        self._process.join(timeout=5.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)

    def __del__(self) -> None:
        self.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_policy_after_first_render(
    config_path: Path, checkpoint_path: Path, device_name: str
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Load Torch/DA3 only after Habitat has created and rendered its EGL scene."""

    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("method") != "architecture_v1_end_to_end":
        raise ValueError("NEXT-027 requires an Architecture-v1 config")
    if config.get("test_locked_used") is not False:
        raise ValueError("NEXT-027 refuses configs that used test_locked")
    repository = config_path.parents[2]

    def resolve(value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else repository / path

    for field in ("da3_source", "da3_runtime"):
        path = resolve(config[field]).resolve(strict=True)
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import torch

    from omtrackvla.models.end_to_end import (
        ArchitectureV1Ablation,
        ArchitectureV1Config,
        EndToEndFollowPolicy,
        load_official_da3_small_l11,
    )

    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real DA3 closed-loop smoke requires a CUDA device")
    torch.cuda.set_device(device)
    architecture = ArchitectureV1Config(**config.get("architecture", {}))
    ablation = ArchitectureV1Ablation(**config.get("ablation", {}))
    da3, loading_report = load_official_da3_small_l11(
        resolve(config["da3_model"]), architecture, ablation,
        resolve(config["dinov2_model"]) if config.get("dinov2_model") else None,
    )
    policy = EndToEndFollowPolicy(da3, architecture, ablation).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("phase") not in {2, 3}
        or checkpoint.get("method") != "architecture_v1_end_to_end"
        or checkpoint.get("test_locked_used") is not False
    ):
        raise ValueError("checkpoint is not an admitted Architecture-v1 Phase 2/3 model")
    policy.load_state_dict(checkpoint["model"], strict=True)
    policy.eval()
    return policy, architecture, device, {
        "da3_loading": loading_report,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_step": checkpoint.get("global_step", checkpoint.get("step")),
        "checkpoint_phase": checkpoint.get("phase"),
        "checkpoint_method": checkpoint.get("method"),
        "test_locked_used": checkpoint.get("test_locked_used"),
    }


def _assign_unique_humanoid_semantic_ids(env: Any) -> dict[int, int]:
    target_id = int(env.current_episode.info["main_human_semantic_id"])
    assigned = {0: target_id}
    for agent_index in range(len(env.sim.agents_mgr)):
        if agent_index == 1:
            continue
        semantic_id = target_id if agent_index == 0 else 2000 + agent_index
        articulated_agent = env.sim.agents_mgr[agent_index].articulated_agent
        for node in articulated_agent.sim_obj.visual_scene_nodes:
            node.semantic_id = semantic_id
        assigned[agent_index] = semantic_id
    return assigned


def _json_scalar(value: Any) -> float | None:
    if value is None:
        return None
    array = np.asarray(value)
    return float(array.reshape(-1)[0]) if array.size else None


def _mean_absdiff(first: np.ndarray, second: np.ndarray) -> float:
    """Return a scale-preserving pixel difference for two sensor frames."""

    first_array = np.asarray(first, dtype=np.float32)
    second_array = np.asarray(second, dtype=np.float32)
    if first_array.shape != second_array.shape:
        raise ValueError(
            f"cannot compare sensor frames {first_array.shape} and "
            f"{second_array.shape}"
        )
    return float(np.abs(first_array - second_array).mean())


def _matrix4_list(value: Any) -> list[list[float]]:
    """Convert a Magnum Matrix4 to JSON without relying on its repr."""

    return [
        [float(value[row, column]) for column in range(4)]
        for row in range(4)
    ]


def _camera_transform(sim: Any) -> np.ndarray:
    sensor = sim._sensors[RGB_KEY]._sensor_object
    return np.asarray(_matrix4_list(sensor.node.absolute_transformation()))


def _draw_frame(
    rgb: np.ndarray,
    decision: ClosedLoopDecision,
    step: int,
    initial_bbox: Sequence[float] | None,
) -> np.ndarray:
    frame = np.asarray(rgb)[..., :3].astype(np.uint8).copy()
    height, width = frame.shape[:2]
    if initial_bbox is not None:
        x1, y1, x2, y2 = (int(round(v)) for v in initial_bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 60, 60), 3)
        cv2.putText(
            frame, "initial target bbox (used once)", (8, height - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )
    px1, py1, px2, py2 = decision.predicted_bbox_xyxy_norm
    predicted = (
        int(round(px1 * width)), int(round(py1 * height)),
        int(round(px2 * width)), int(round(py2 * height)),
    )
    cv2.rectangle(frame, predicted[:2], predicted[2:], (60, 220, 60), 2)

    panel = np.full((height, height, 3), 245, dtype=np.uint8)
    origin = (height // 2, height - 30)
    scale = max(45.0, height / 4.0)
    cv2.line(panel, (origin[0], 10), origin, (180, 180, 180), 1)
    cv2.line(panel, (10, origin[1]), (height - 10, origin[1]), (180, 180, 180), 1)
    pixels = []
    for forward, left in decision.waypoints:
        pixels.append(
            (int(round(origin[0] - left * scale)), int(round(origin[1] - forward * scale)))
        )
    if len(pixels) >= 2:
        cv2.polylines(
            panel, [np.asarray(pixels, dtype=np.int32).reshape(-1, 1, 2)],
            False, (30, 180, 30), 3, cv2.LINE_AA,
        )
    for index, point in enumerate(pixels):
        color = (20, 70, 230) if index == decision.selected_waypoint_index else (30, 180, 30)
        cv2.circle(panel, point, 4, color, -1, cv2.LINE_AA)
    cv2.putText(panel, "predicted local waypoints", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(panel, "forward", (origin[0] + 5, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1, cv2.LINE_AA)
    lines = (
        f"step={step} mode={decision.mode}",
        f"stop={decision.stop_probability:.3f} visible={decision.visibility_probability:.3f}",
        f"cmd=({decision.action.forward:+.2f},{decision.action.lateral:+.2f},{decision.action.yaw:+.2f})",
        f"DA3+policy={decision.inference_ms:.0f} ms | UWB=missing",
    )
    for index, line in enumerate(lines):
        y = 18 + index * 19
        cv2.rectangle(frame, (4, y - 14), (min(width - 4, 365), y + 3), (0, 0, 0), -1)
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate((frame, panel), axis=1)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def partial_rollout_result(
    base: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Build the crash-recoverable result written after every successful step."""

    completed_steps = len(records)
    if any(
        int(record.get("step", -1)) != expected_step
        for expected_step, record in enumerate(records, 1)
    ):
        raise ValueError("partial rollout steps must be contiguous and one-indexed")
    return {
        **dict(base),
        "status": "partial",
        "completed_steps": completed_steps,
        "steps": list(records),
    }


def encode_rollout_frames(
    frame_directory: Path, video_path: Path, fps: int
) -> dict[str, Any]:
    """Encode spooled PNG frames in a child process after the live rollout.

    Keeping the video encoder out of the Habitat render loop prevents native
    FFmpeg/OpenCV teardown from taking the only copy of long-run evidence with
    it.  The numbered PNGs remain independently inspectable if encoding fails.
    """

    if fps <= 0:
        raise ValueError("video fps must be positive")
    frames = sorted(frame_directory.glob("frame_*.png"))
    if not frames:
        raise RuntimeError(f"no rollout frames found in {frame_directory}")
    expected = [f"frame_{index:06d}.png" for index in range(len(frames))]
    if [path.name for path in frames] != expected:
        raise RuntimeError("rollout frame sequence is not contiguous from zero")

    from imageio_ffmpeg import get_ffmpeg_exe

    temporary = video_path.with_name(
        f"{video_path.stem}.{os.getpid()}.tmp{video_path.suffix}"
    )
    command = [
        get_ffmpeg_exe(),
        "-y",
        "-loglevel", "error",
        "-framerate", str(fps),
        "-start_number", "0",
        "-i", str(frame_directory / "frame_%06d.png"),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(temporary),
    ]
    completed = subprocess.run(
        command, check=False, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"isolated FFmpeg failed with {completed.returncode}: "
            f"{completed.stderr[-2000:]}"
        )
    temporary.replace(video_path)
    return {
        "method": "spooled_png_isolated_ffmpeg",
        "frame_count": len(frames),
        "fps": fps,
        "ffmpeg": get_ffmpeg_exe(),
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("stt", "dt", "at"), default="stt")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--initialization-wait-steps", type=int, default=0)
    parser.add_argument("--initialization-scan-episodes", type=int, default=1)
    parser.add_argument("--expected-scene-id", required=True)
    parser.add_argument("--expected-episode-id", required=True)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--scene-dataset", default=DEFAULT_SCENE_DATASET)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--video-fps", type=int, default=8)
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def main() -> int:
    main_started_ns = time.perf_counter_ns()
    startup_timer = StageTimer()
    lifecycle_ms = {}
    initialization_timings = []
    args = arguments()
    if args.initialization_scan_episodes != 1 or args.initialization_wait_steps != 0:
        raise ValueError("timed fixed-episode diagnostics require scan=1 and initialization wait=0")
    if (
        args.dataset_index < 0
        or args.max_steps <= 0
        or args.initialization_wait_steps < 0
        or args.initialization_scan_episodes <= 0
    ):
        raise ValueError(
            "dataset index/wait must be non-negative and max_steps positive"
        )
    model_config = args.model_config.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)
    timing_path = output_root / "timing.json"
    timing_steps_path = output_root / "timing.steps.jsonl"
    assert_new_rollout_artifacts(output_root)
    source_identity = {
        "timed_runner": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__))},
        "timing_helper": {"path": inspect.getfile(StageTimer), "sha256": _sha256(Path(inspect.getfile(StageTimer)))},
        "original_runner_actual": {
            "path": str(Path(__file__).with_name("end_to_end_closed_loop.py")),
            "sha256": _sha256(Path(__file__).with_name("end_to_end_closed_loop.py")),
        },
        "reviewed_original_base_sha256": "dc99cf00e719f4613ce0792c7247d6d0bcc1677a7fb6b3cb7296705df9f8c8d5",
    }
    startup_timer.mark("arguments_paths_output_setup")

    # Habitat/EGL must be live before _load_policy_after_first_render imports Torch.
    import habitat
    from habitat.config import read_write
    from habitat.datasets import make_dataset
    import evt_bench  # noqa: F401

    config_kind = "train" if args.split == "train" else "infer"
    habitat_config_path = (
        "habitat-lab/habitat/config/benchmark/nav/track/"
        f"track_{config_kind}_{args.task}.yaml"
    )
    config = configure(
        habitat.get_config(habitat_config_path), args.scene_dataset,
    )
    with read_write(config):
        for key in (RGB_KEY, PANOPTIC_KEY):
            if key not in config.habitat.gym.obs_keys:
                config.habitat.gym.obs_keys.append(key)
    dataset = make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
    if args.dataset_index >= len(dataset.episodes):
        raise ValueError(
            f"dataset-index={args.dataset_index} outside {len(dataset.episodes)} episodes"
        )
    expected_episode = dataset.episodes[args.dataset_index]
    if (str(expected_episode.scene_id) != args.expected_scene_id
            or str(expected_episode.episode_id) != args.expected_episode_id):
        raise ValueError("requested dataset index no longer matches the frozen scene/episode identity")
    indexed_candidates = list(enumerate(dataset.episodes))[args.dataset_index :]
    indexed_candidates = indexed_candidates[: args.initialization_scan_episodes]
    for index, episode in indexed_candidates:
        episode.info["_next027_dataset_index"] = index
    dataset.episodes = [episode for _, episode in indexed_candidates]
    startup_timer.mark("habitat_import_config_dataset_and_identity_check")

    result_path = output_root / "result.json"
    partial_result_path = output_root / "result.partial.json"
    video_path = output_root / "rollout.mp4"
    frame_directory = output_root / "rollout_frames"
    if not args.no_video:
        frame_directory.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    startup_timer.mark("output_artifacts_prepare")
    with habitat.TrackEnv(config=config, dataset=dataset) as env:
        startup_timer.mark("environment_construct_and_enter")
        initial_bbox = None
        initialization_source = None
        selected_dataset_index = None
        for _ in indexed_candidates:
            initialization_timer = StageTimer()
            task_observations = env.reset()
            initialization_timer.mark("env_reset_inclusive")
            if (str(env.current_episode.scene_id) != args.expected_scene_id
                    or str(env.current_episode.episode_id) != args.expected_episode_id):
                raise RuntimeError("reset selected a different frozen scene/episode")
            assigned_ids = _assign_unique_humanoid_semantic_ids(env)
            initialization_timer.mark("semantic_ids_and_identity_check")
            observations = env.sim.get_sensor_observations()
            initialization_timer.mark("explicit_initial_sensor_render")
            rgb = np.asarray(observations[RGB_KEY])[..., :3]
            target_semantic_id = int(
                env.current_episode.info["main_human_semantic_id"]
            )
            initial_bbox, initialization_source = initialization_bbox(
                task_observations, observations, target_semantic_id
            )
            selected_dataset_index = int(
                env.current_episode.info["_next027_dataset_index"]
            )
            print(json.dumps({
                "event": "next027_initialization_probe",
                "dataset_index": selected_dataset_index,
                "episode_id": str(env.current_episode.episode_id),
                "bbox": initial_bbox,
                "source": initialization_source,
            }), flush=True)
            initialization_timer.mark("initialization_bbox_and_probe_log")
            initialization_timings.append(initialization_timer.durations)
            if initial_bbox is not None:
                break
        initialization_wait_steps = 0
        while (
            initial_bbox is None
            and not env.episode_over
            and initialization_wait_steps < args.initialization_wait_steps
        ):
            # Before the one-time target initialization exists, the robot is
            # held still. Only the benchmark humanoid/distractor actions run.
            # This is not a learned policy action and no target coordinate is
            # consulted. Panoptic is used solely to emulate the user-provided
            # initialization bbox on the first frame where that is possible.
            observations = env.step({
                "action": ACTION_NAMES,
                "action_args": {"agent_1_base_vel": [0.0, 0.0, 0.0]},
            })
            initialization_wait_steps += 1
            rgb = np.asarray(observations[RGB_KEY])[..., :3]
            initial_bbox, initialization_source = initialization_bbox(
                observations, observations, target_semantic_id
            )
        if initial_bbox is None:
            raise RuntimeError(
                "none of the scanned episodes exposed a one-time initialization "
                f"bbox (scanned={len(indexed_candidates)}, held-still "
                f"steps={args.initialization_wait_steps})"
            )
        initial_bbox_norm = normalized_bbox(initial_bbox, rgb.shape)
        intrinsics, camera_from_base = habitat_camera_calibration(
            rgb.shape, output_height=280, output_width=504, hfov_deg=90.0
        )

        startup_timer.mark("fixed_episode_initialization_and_calibration")
        controller = ArchitectureV1PolicyWorker(
            model_config=str(model_config),
            checkpoint=str(checkpoint),
            device_name=args.device,
            initial_rgb=rgb,
            initial_bbox_xyxy_norm=initial_bbox_norm,
            camera_intrinsics=intrinsics,
            camera_from_base=camera_from_base,
        )
        startup_timer.mark("policy_worker_spawn_load_initialize")
        audit = decision_api_audit(controller)
        if controller.worker_audit != audit:
            raise RuntimeError("parent/worker decision input audits disagree")
        loading = controller.loading
        print(json.dumps({"event": "next027_input_audit", **audit}), flush=True)

        partial_result_base = {
            "schema_version": 1,
            "stage": "next027_architecture_v1_habitat_closed_loop_smoke",
            "task": args.task,
            "split": args.split,
            "dataset_index": selected_dataset_index,
            "episode_id": str(env.current_episode.episode_id),
            "scene_id": env.current_episode.scene_id,
            "checkpoint": str(checkpoint),
            "model_config": str(model_config),
            "initialization": {
                "source": initialization_source,
                "bbox_xyxy": list(initial_bbox),
                "bbox_xyxy_norm": list(initial_bbox_norm),
                "environment_step": initialization_wait_steps,
                "used_frames": [initialization_wait_steps],
                "robot_action_before_initialization": "fixed_zero_hold",
            },
            "uwb_mode": "missing",
            "policy_rgb_source": "explicit_post_action_sensor_render",
            "assigned_humanoid_semantic_ids": assigned_ids,
            "input_audit": audit,
            "loading": loading,
        }
        atomic_json(
            partial_result_path,
            partial_rollout_result(partial_result_base, records),
        )

        startup_timer.mark("input_audit_and_initial_partial_report")
        robot = env.sim.agents_mgr[1].articulated_agent
        target_agent = env.sim.agents_mgr[0].articulated_agent
        previous_robot_position = np.asarray(robot.base_pos, dtype=np.float64).copy()
        previous_target_position = np.asarray(target_agent.base_pos, dtype=np.float64).copy()
        previous_camera_transform = _camera_transform(env.sim)
        robot_path_m = 0.0
        target_path_m = 0.0
        visible_steps = 0
        losses = 0
        reacquisitions = 0
        previous_visible = True
        collisions = 0.0
        following_sum = 0.0
        previous_policy_rgb = None
        startup_timer.mark("pre_loop_state_setup")
        loop_started_ns = time.perf_counter_ns()
        start = time.monotonic()
        try:
            step = 0
            while not env.episode_over and step < args.max_steps:
                step_timer = StageTimer()
                # The decision call receives RGB and no UWB sample.  Nothing
                # else from Habitat crosses this boundary.
                policy_rgb = np.asarray(observations[RGB_KEY])[..., :3].copy()
                rgb_mean = float(policy_rgb.mean())
                rgb_std = float(policy_rgb.std())
                rgb_change = (
                    None
                    if previous_policy_rgb is None
                    else float(np.abs(
                        policy_rgb.astype(np.float32)
                        - previous_policy_rgb.astype(np.float32)
                    ).mean())
                )
                previous_policy_rgb = policy_rgb.copy()
                step_timer.mark("rgb_copy_history_and_statistics")
                decision = controller.decide(policy_rgb, uwb=None)
                step_timer.mark("policy_ipc_exchange_including_worker")
                if not args.no_video:
                    rendered = _draw_frame(
                        observations[RGB_KEY], decision, step,
                        initial_bbox if step == 0 else None,
                    )
                    step_timer.mark("video_frame_draw")
                    frame_path = frame_directory / f"frame_{step:06d}.png"
                    if not cv2.imwrite(
                        str(frame_path), cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
                    ):
                        raise RuntimeError(f"could not write rollout frame: {frame_path}")
                    step_timer.mark("video_png_encode_and_file_write")
                else:
                    step_timer.mark("video_disabled_branch")
                step_observations = env.step({
                    "action": ACTION_NAMES,
                    "action_args": {
                        "agent_1_base_vel": decision.action.as_habitat()
                    },
                })
                step_timer.mark("env_step_inclusive_internal_sensors_and_task")
                camera_transform_after_step = _camera_transform(env.sim)
                step_timer.mark("camera_state_after_step")
                forced_observations = env.sim.get_sensor_observations()
                step_timer.mark("explicit_post_action_sensor_render")
                camera_transform_after_forced_render = _camera_transform(env.sim)
                step_rgb = np.asarray(step_observations[RGB_KEY])[..., :3].copy()
                forced_rgb = np.asarray(forced_observations[RGB_KEY])[..., :3].copy()
                render_audit = {
                    "env_step_vs_forced_rgb_absdiff_0_255": _mean_absdiff(
                        step_rgb, forced_rgb
                    ),
                    "policy_vs_next_env_step_rgb_absdiff_0_255": _mean_absdiff(
                        policy_rgb, step_rgb
                    ),
                    "policy_vs_next_forced_rgb_absdiff_0_255": _mean_absdiff(
                        policy_rgb, forced_rgb
                    ),
                    "camera_transform_before_action": previous_camera_transform.tolist(),
                    "camera_transform_after_step": camera_transform_after_step.tolist(),
                    "camera_transform_after_forced_render": (
                        camera_transform_after_forced_render.tolist()
                    ),
                    "camera_translation_delta_after_step_m": float(np.linalg.norm(
                        camera_transform_after_step[3, :3]
                        - previous_camera_transform[3, :3]
                    )),
                    "camera_transform_delta_after_step": float(np.linalg.norm(
                        camera_transform_after_step - previous_camera_transform
                    )),
                }
                # Prefer the explicit post-action render at the policy boundary.
                # The audit above remains in every result so this choice is
                # measurable rather than silently masking a stale env.step frame.
                observations = forced_observations
                step += 1
                step_timer.mark("post_render_frame_copy_and_freshness_audit")

                # Everything below is evaluation-only and executes after the
                # action has already been selected and submitted.
                metrics = env.get_metrics()
                robot_position = np.asarray(robot.base_pos, dtype=np.float64).copy()
                target_position = np.asarray(target_agent.base_pos, dtype=np.float64).copy()
                robot_path_m += float(np.linalg.norm(robot_position - previous_robot_position))
                target_path_m += float(np.linalg.norm(target_position - previous_target_position))
                previous_robot_position = robot_position
                previous_target_position = target_position
                previous_camera_transform = camera_transform_after_forced_render
                gt_distance = float(np.linalg.norm(robot_position - target_position))
                current_mask = (
                    np.asarray(observations[PANOPTIC_KEY]).squeeze()
                    == target_semantic_id
                )
                current_visible = bool(current_mask.any())
                visible_steps += int(current_visible)
                losses += int(previous_visible and not current_visible)
                reacquisitions += int(not previous_visible and current_visible)
                previous_visible = current_visible
                following = _json_scalar(metrics.get("human_following")) or 0.0
                collision = _json_scalar(metrics.get("human_collision")) or 0.0
                following_sum += following
                collisions = max(collisions, collision)
                record = {
                    "step": step,
                    "policy": {
                        "mode": decision.mode,
                        "action": asdict(decision.action),
                        "waypoints": [list(point) for point in decision.waypoints],
                        "selected_waypoint_index": decision.selected_waypoint_index,
                        "stop_probability": decision.stop_probability,
                        "visibility_probability": decision.visibility_probability,
                        "predicted_bbox_xyxy_norm": list(
                            decision.predicted_bbox_xyxy_norm
                        ),
                        "inference_ms": decision.inference_ms,
                        "uwb_valid": False,
                        "rgb_mean_0_255": rgb_mean,
                        "rgb_std_0_255": rgb_std,
                        "rgb_temporal_absdiff_0_255": rgb_change,
                    },
                    "evaluation_only_after_action": {
                        "gt_distance_m": gt_distance,
                        "gt_visible": current_visible,
                        "human_following": following,
                        "human_collision": collision,
                        "render_audit": render_audit,
                    },
                }
                step_timer.mark("evaluation_metrics_and_record_build")
                record["timing"] = {
                    "parent_ms": step_timer.durations,
                    "worker": controller.last_timings,
                    "complete": False,
                }
                records.append(record)
                # Habitat can terminate the whole process in native code, so
                # publish a complete replay prefix before attempting another
                # simulator step.  The atomic rename prevents torn JSON.
                atomic_json(
                    partial_result_path,
                    partial_rollout_result(partial_result_base, records),
                )
                step_timer.mark("partial_report_build_serialize_write_replace")
                print(json.dumps({"event": "next027_step", **record}), flush=True)
                step_timer.mark("step_stdout_serialize_and_flush")
                # Preserve the current step's report/log timings without a
                # second rewrite of result.partial.json. This added I/O is timed.
                append_step_timing(timing_steps_path, record)
                step_timer.mark("timing_jsonl_write")
                record["timing"]["parent_total_ms"] = step_timer.total_ms
                record["timing"]["complete"] = True
        finally:
            lifecycle_ms["policy_loop_wall"] = elapsed_ms(loop_started_ns)
            close_started_ns = time.perf_counter_ns()
            print(json.dumps({"event": "next027_policy_worker_close_start"}), flush=True)
            controller.close()
            print(json.dumps({"event": "next027_policy_worker_close_complete"}), flush=True)
            lifecycle_ms["worker_close_and_logs"] = elapsed_ms(close_started_ns)

        video_encoding = None
        video_started_ns = time.perf_counter_ns()
        if not args.no_video:
            print(json.dumps({"event": "next027_video_encode_start"}), flush=True)
            video_encoding = encode_rollout_frames(
                frame_directory, video_path, args.video_fps
            )
            print(json.dumps({
                "event": "next027_video_publish_complete", **video_encoding,
            }), flush=True)

        lifecycle_ms["video_encode_and_publish_including_logs"] = elapsed_ms(video_started_ns)
        elapsed = time.monotonic() - start
        final_build_started_ns = time.perf_counter_ns()
        final_metrics = env.get_metrics()
        official_success = _json_scalar(
            final_metrics.get("human_following_success")
        )
        inference_times = [
            record["policy"]["inference_ms"] for record in records
        ]
        forced_rgb_differences = [
            record["evaluation_only_after_action"]["render_audit"][
                "policy_vs_next_forced_rgb_absdiff_0_255"
            ]
            for record in records
        ]
        stale_env_step_frames = sum(
            record["evaluation_only_after_action"]["render_audit"][
                "policy_vs_next_env_step_rgb_absdiff_0_255"
            ]
            == 0.0
            and record["evaluation_only_after_action"]["render_audit"][
                "policy_vs_next_forced_rgb_absdiff_0_255"
            ]
            > 0.0
            for record in records
        )
        summary = {
            "steps": len(records),
            "requested_max_steps": args.max_steps,
            "episode_finished": bool(env.episode_over),
            "termination_reason": (
                "episode_over" if env.episode_over else
                "requested_step_limit" if len(records) >= args.max_steps else
                "loop_terminated"
            ),
            "following_rate": following_sum / max(1, len(records)),
            "success": official_success if env.episode_over else None,
            "official_success_so_far": official_success,
            "collision": collisions,
            "visible_rate": visible_steps / max(1, len(records)),
            "target_losses": losses,
            "target_reacquisitions": reacquisitions,
            "robot_path_m": robot_path_m,
            "target_path_m": target_path_m,
            "robot_to_target_path_ratio": robot_path_m / max(target_path_m, 1.0e-6),
            "mean_inference_ms": float(np.mean(inference_times))
            if inference_times else None,
            "mean_inference_ms_after_warmup": float(np.mean(inference_times[1:]))
            if len(inference_times) > 1 else None,
            "p95_inference_ms_after_warmup": float(np.percentile(
                inference_times[1:], 95
            )) if len(inference_times) > 1 else None,
            "visual_uncertain_stop_steps": sum(
                record["policy"]["mode"] == "visual_uncertain_stop"
                for record in records
            ),
            "visual_proximity_hold_steps": sum(
                record["policy"]["mode"] == "visual_proximity_hold"
                for record in records
            ),
            "fresh_rgb_nonzero_steps": sum(
                difference > 0.0 for difference in forced_rgb_differences
            ),
            "fresh_rgb_absdiff_min_0_255": min(forced_rgb_differences)
            if forced_rgb_differences else None,
            "fresh_rgb_absdiff_mean_0_255": float(np.mean(forced_rgb_differences))
            if forced_rgb_differences else None,
            "stale_env_step_frames": stale_env_step_frames,
            "video_frame_count": None if video_encoding is None else (
                video_encoding["frame_count"]
            ),
            "wall_time_seconds": elapsed,
        }
        result = {
            "schema_version": 1,
            "stage": "next027_architecture_v1_habitat_closed_loop_smoke",
            "task": args.task,
            "split": args.split,
            "dataset_index": selected_dataset_index,
            "episode_id": str(env.current_episode.episode_id),
            "scene_id": env.current_episode.scene_id,
            "checkpoint": str(checkpoint),
            "model_config": str(model_config),
            "initialization": {
                "source": initialization_source,
                "bbox_xyxy": list(initial_bbox),
                "bbox_xyxy_norm": list(initial_bbox_norm),
                "environment_step": initialization_wait_steps,
                "used_frames": [initialization_wait_steps],
                "robot_action_before_initialization": "fixed_zero_hold",
            },
            "uwb_mode": "missing",
            "policy_rgb_source": "explicit_post_action_sensor_render",
            "assigned_humanoid_semantic_ids": assigned_ids,
            "input_audit": audit,
            "loading": loading,
            "summary": summary,
            "steps": records,
            "video": None if args.no_video else str(video_path),
            "video_encoding": video_encoding,
        }
        lifecycle_ms["final_metrics_summary_and_result_build"] = elapsed_ms(final_build_started_ns)
        result["performance_diagnostics"] = {
            "timing_report": str(timing_path), "timing_steps_jsonl": str(timing_steps_path),
            "scope": "host_closed_loop_bottleneck_diagnostic",
            "final_timings_available_after_environment_close": True,
        }
        result_write_started_ns = time.perf_counter_ns()
        atomic_json(result_path, result)
        lifecycle_ms["final_result_serialize_write_replace"] = elapsed_ms(result_write_started_ns)
        final_log_started_ns = time.perf_counter_ns()
        print(json.dumps({
            "event": "next027_closed_loop_complete",
            "result": str(result_path),
            "video": None if args.no_video else str(video_path),
            "summary": summary,
        }, indent=2), flush=True)
        lifecycle_ms["completion_stdout"] = elapsed_ms(final_log_started_ns)
        environment_close_started_ns = time.perf_counter_ns()
    lifecycle_ms["environment_context_exit"] = elapsed_ms(environment_close_started_ns)
    timing_result = {
        **timing_contract(), "status": "complete",
        "parent_runtime": runtime_identity(),
        "source_identity": source_identity,
        "requested_identity": {"task": args.task, "split": args.split,
            "dataset_index": args.dataset_index, "scene_id": args.expected_scene_id,
            "episode_id": args.expected_episode_id, "max_steps": args.max_steps},
        "actual_identity": {key: result[key] for key in ("task", "split", "dataset_index", "scene_id", "episode_id")},
        "initialization": result["initialization"],
        "checkpoint": str(checkpoint), "checkpoint_sha256": loading["checkpoint_sha256"],
        "startup_parent_ms": startup_timer.durations,
        "initialization_attempts_nested_ms": initialization_timings,
        "policy_worker_startup_nested": controller.timing_startup,
        "lifecycle_ms": lifecycle_ms,
        "main_wall_ms_before_timing_publication": elapsed_ms(main_started_ns),
        "legacy_summary_wall_time_seconds_unchanged_scope": elapsed,
        "summary": summarize_steps(records),
        "steps": [{"step": record["step"], **record["timing"]} for record in records],
        "video_enabled": not args.no_video,
        "result": str(result_path), "termination": result["summary"]["termination_reason"],
    }
    atomic_json(timing_path, timing_result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
