"""Trace EVT physics callers and test metadata reads for simulation-time effects.

Train episodes only. No policy, labels, dataset mutation, or timing admission.
Run from the remote OmTrackVLA repository with its Habitat Python environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import traceback


def finite_config(value):
    """Keep strict JSON valid while preserving explicit nonfinite config values."""
    if isinstance(value, dict):
        return {str(key): finite_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_config(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


class ClockTracer:
    """Shadow instance methods, preserving call nesting and exact restoration."""

    def __init__(self, simulator):
        self.simulator = simulator
        self.events = []
        self.parents = []
        self.originals = {}
        self.hook_errors = {}
        self.phase = "initialization"

    def now(self):
        return float(self.simulator.get_world_time())

    def install(self):
        for name in ("step_physics", "step_world", "internal_step"):
            original = getattr(self.simulator, name, None)
            if not callable(original):
                self.hook_errors[name] = "method unavailable"
                continue
            namespace = vars(self.simulator)
            prior = (name in namespace, namespace.get(name), original)

            def measured(*args, _name=name, _original=original, **kwargs):
                event_id = len(self.events)
                requested = args[0] if args else kwargs.get("dt")
                event = {
                    "id": event_id,
                    "parent_id": self.parents[-1] if self.parents else None,
                    "method": _name,
                    "phase": self.phase,
                    "requested_dt_s": float(requested) if isinstance(requested, (int, float)) else None,
                    "time_before_s": self.now(),
                    "caller_stack": [
                        {"file": frame.filename, "line": frame.lineno, "function": frame.name}
                        for frame in traceback.extract_stack(limit=12)[:-1]
                    ],
                }
                self.events.append(event)
                self.parents.append(event_id)
                try:
                    return _original(*args, **kwargs)
                except BaseException as exc:
                    event["exception_type"] = type(exc).__name__
                    raise
                finally:
                    event["time_after_s"] = self.now()
                    event["elapsed_sim_s"] = event["time_after_s"] - event["time_before_s"]
                    self.parents.pop()

            try:
                setattr(self.simulator, name, measured)
                self.originals[name] = prior
            except (AttributeError, TypeError) as exc:
                self.hook_errors[name] = f"{type(exc).__name__}: {exc}"

    def restore(self):
        for name, (existed, previous, _) in self.originals.items():
            if existed:
                setattr(self.simulator, name, previous)
            else:
                delattr(self.simulator, name)

    def run(self, phase, callback):
        previous_phase = self.phase
        self.phase = phase
        start = len(self.events)
        before = self.now()
        try:
            result = callback()
        finally:
            after = self.now()
            self.phase = previous_phase
        events = self.events[start:]
        root_elapsed = sum(event["elapsed_sim_s"] for event in events if event["parent_id"] is None)
        return result, {
            "phase": phase,
            "time_before_s": before,
            "time_after_s": after,
            "elapsed_sim_s": after - before,
            "root_hook_elapsed_sim_s": root_elapsed,
            "unaccounted_sim_s": after - before - root_elapsed,
            "event_ids": [event["id"] for event in events],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("stt", "dt", "at"), default="stt")
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--steps-per-phase", type=int, default=3)
    parser.add_argument("--metadata-repetitions", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 1 <= args.steps_per_phase <= 10 or not 1 <= args.metadata_repetitions <= 5:
        raise ValueError("keep this diagnostic bounded: 1..10 steps/phase, 1..5 metadata repetitions")

    import numpy as np
    from omegaconf import OmegaConf
    import habitat
    from habitat.datasets import make_dataset
    import evt_bench  # noqa: F401
    from omtrackvla.evaluation.end_to_end_closed_loop import (
        ACTION_NAMES, DEFAULT_SCENE_DATASET, RGB_KEY, configure,
    )

    config = configure(habitat.get_config(
        f"habitat-lab/habitat/config/benchmark/nav/track/track_train_{args.task}.yaml"
    ), DEFAULT_SCENE_DATASET)
    dataset = make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
    if not 0 <= args.dataset_index < len(dataset.episodes):
        raise ValueError("dataset index outside train split")
    dataset.episodes = [dataset.episodes[args.dataset_index]]
    config_data = finite_config(OmegaConf.to_container(config, resolve=True))
    config_bytes = json.dumps(config_data, sort_keys=True, default=str).encode("utf-8")
    report = {
        "schema_version": 2, "stage": "evt_control_clock_call_trace",
        "task": args.task, "split": "train", "dataset_index": args.dataset_index,
        "test_locked_used": False, "policy_loaded": False, "timing_admitted": False,
        "resolved_config": config_data,
        "resolved_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "source_sha256": {}, "metadata_only_checks": [], "records": [],
    }
    for source in (
        Path(__file__), Path("habitat-lab/habitat/core/embodied_task.py"),
        Path("habitat-lab/habitat/tasks/rearrange/rearrange_sim.py"),
        Path("habitat-lab/habitat/tasks/rearrange/actions/actions.py"),
        Path("evt_bench/additional_action.py"),
    ):
        if source.is_file():
            report["source_sha256"][str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()

    started = time.time()
    with habitat.TrackEnv(config=config, dataset=dataset) as env:
        env.reset()
        sim, task = env.sim, env.task
        if not callable(getattr(sim, "get_world_time", None)):
            raise RuntimeError("this probe requires simulator world time")
        runtime_config = finite_config(OmegaConf.to_container(env._config, resolve=True))
        report["resolved_runtime_env_config"] = runtime_config
        report["resolved_runtime_env_config_sha256"] = hashlib.sha256(
            json.dumps(runtime_config, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        pending = [runtime_config]
        while pending:
            item = pending.pop()
            if isinstance(item, dict):
                for key, value in item.items():
                    if key == "physics_config_file" and isinstance(value, str):
                        source = Path(value)
                        if source.is_file():
                            report["source_sha256"][str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
                    elif isinstance(value, (dict, list)):
                        pending.append(value)
            elif isinstance(item, list):
                pending.extend(item)
        report.update({
            "scene_id": env.current_episode.scene_id,
            "episode_id": str(env.current_episode.episode_id),
            "simulator_class": f"{type(sim).__module__}.{type(sim).__qualname__}",
            "task_class": f"{type(task).__module__}.{type(task).__qualname__}",
            "ctrl_freq": float(sim.ctrl_freq), "ac_freq_ratio": int(sim.ac_freq_ratio),
            "task_physics_target_sps": (
                float(task._physics_target_sps) if hasattr(task, "_physics_target_sps") else None
            ),
        })
        tracer = ClockTracer(sim)
        tracer.install()

        def state():
            before = tracer.now()
            result = {
                "robot_xyz_m": np.asarray(sim.agents_mgr[1].articulated_agent.base_pos).tolist(),
                "robot_yaw_rad": float(sim.agents_mgr[1].articulated_agent.base_rot),
                "target_xyz_m": np.asarray(sim.agents_mgr[0].articulated_agent.base_pos).tolist(),
            }
            result.update({"time_before_reads_s": before, "world_time_s": tracer.now()})
            return result

        def controllers():
            result = {}
            for name, action in task.actions.items():
                fields = {"class": f"{type(action).__module__}.{type(action).__qualname__}"}
                action_config = getattr(action, "_config", None)
                for key in ("lin_speed", "ang_speed", "longitudinal_lin_speed", "lateral_lin_speed"):
                    value = action_config.get(key) if action_config is not None else None
                    if value is not None:
                        fields[key] = float(value)
                controller = getattr(action, "humanoid_controller", None)
                if controller is not None:
                    for key in ("motion_fps", "walk_mocap_frame", "dist_per_step_size", "turning_step_amount"):
                        value = getattr(controller, key, None)
                        if value is not None:
                            number = float(value)
                            fields[key] = number if math.isfinite(number) else str(number)
                    motion = getattr(controller, "walk_motion", None)
                    if motion is not None:
                        fields["walk_motion_fps"] = float(motion.fps)
                result[name] = fields
            return result

        def rgb_capture():
            rgb = np.asarray(sim.get_sensor_observations()[RGB_KEY])[..., :3]
            return {"rgb_mean": float(rgb.mean()), "capture_world_time_s": tracer.now()}

        try:
            # These calls deliberately contain no env.step. Their elapsed
            # simulator time must be checked before attributing clock changes.
            for repetition in range(args.metadata_repetitions):
                for name, callback in (
                    ("world_time_reads", lambda: [tracer.now() for _ in range(5)]),
                    ("pose_reads", state), ("controller_reads", controllers),
                    ("rgb_capture", rgb_capture), ("get_metrics", env.get_metrics),
                ):
                    _, timing = tracer.run(f"metadata_only_{repetition}.{name}", callback)
                    report["metadata_only_checks"].append(timing)
            actions = ([[0.0, 0.0, 0.0]] * args.steps_per_phase
                       + [[0.1, 0.0, 0.0]] * args.steps_per_phase
                       + [[0.0, 0.0, 0.1]] * args.steps_per_phase)
            for step, action in enumerate(actions, 1):
                if env.episode_over:
                    break
                before, before_timing = tracer.run(f"step_{step}.before_pose", state)
                controller_before, controller_before_timing = tracer.run(f"step_{step}.before_controllers", controllers)
                _, step_timing = tracer.run(f"step_{step}.env_step", lambda: env.step({
                    "action": ACTION_NAMES, "action_args": {"agent_1_base_vel": action},
                }))
                after, after_timing = tracer.run(f"step_{step}.after_pose", state)
                controller_after, controller_after_timing = tracer.run(f"step_{step}.after_controllers", controllers)
                rgb, rgb_timing = tracer.run(f"step_{step}.explicit_rgb", rgb_capture)
                metrics, metrics_timing = tracer.run(f"step_{step}.get_metrics", env.get_metrics)
                report["records"].append({
                    "step": step, "action": action, "before": before, "after": after,
                    "controller_before": controller_before, "controller_after": controller_after,
                    "rgb": rgb,
                    "human_collision": float(np.asarray(metrics.get("human_collision", 0)).reshape(-1)[0]),
                    "phases": [before_timing, controller_before_timing, step_timing,
                               after_timing, controller_after_timing, rgb_timing, metrics_timing],
                })
        finally:
            tracer.restore()
            report["hooked_methods"] = list(tracer.originals)
            report["hook_errors"] = tracer.hook_errors
            report["physics_calls"] = tracer.events
    report["elapsed_wall_s"] = time.time() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False, default=str)
        stream.write("\n")
    print(json.dumps({
        "output": str(args.output), "records": len(report["records"]),
        "physics_calls": len(report["physics_calls"]),
        "hook_errors": report["hook_errors"], "timing_admitted": False,
    }), flush=True)


if __name__ == "__main__":
    main()
