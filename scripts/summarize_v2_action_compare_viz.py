from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODES = ("direct", "waypoint_dt", "lookahead")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def first_true(values: list[bool]) -> int | None:
    return next((index + 1 for index, value in enumerate(values) if value), None)


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def summarize(result: dict) -> dict:
    records = result["steps"]
    distances = [float(r["evaluation_only_after_action"]["gt_distance_m"]) for r in records]
    visible = [bool(r["evaluation_only_after_action"]["gt_visible"]) for r in records]
    collisions = [float(r["evaluation_only_after_action"]["human_collision"]) > 0 for r in records]
    actions = [r["policy"]["action"] for r in records]
    horizons = [
        float(np.linalg.norm(np.asarray(r["policy"]["waypoints"][-1], dtype=np.float64)))
        for r in records
    ]
    stops = [float(r["policy"]["stop_probability"]) for r in records]
    candidates = [r["policy"]["action_candidates_normalized"] for r in records]
    yaw_saturation = {
        name: mean([float(abs(c[name][2]) >= 0.999) for c in candidates])
        for name in MODES
    }
    return {
        "steps": len(records),
        "success": result["summary"].get("success"),
        "following_rate": result["summary"]["following_rate"],
        "collision": result["summary"]["collision"],
        "first_collision_step": first_true(collisions),
        "visible_rate": result["summary"]["visible_rate"],
        "first_target_loss_step": first_true([not value for value in visible]),
        "final_distance_m": distances[-1] if distances else None,
        "mean_distance_m": mean(distances),
        "min_distance_m": min(distances) if distances else None,
        "robot_path_m": result["summary"]["robot_path_m"],
        "target_path_m": result["summary"]["target_path_m"],
        "mean_abs_action": {
            axis: mean([abs(float(action[axis])) for action in actions])
            for axis in ("forward", "lateral", "yaw")
        },
        "mean_trajectory_horizon_m": mean(horizons),
        "max_trajectory_horizon_m": max(horizons) if horizons else None,
        "mean_stop_probability": mean(stops),
        "max_stop_probability": max(stops) if stops else None,
        "candidate_yaw_saturation_rate": yaw_saturation,
    }


def copy_keyframes(root: Path, mode: str, result: dict, summary: dict) -> list[str]:
    records = result["steps"]
    requested = {1, len(records)}
    for key in ("first_target_loss_step", "first_collision_step"):
        center = summary[key]
        if center is not None:
            requested.update(range(max(1, center - 2), min(len(records), center + 2) + 1))
    output = root / "keyframes" / mode
    output.mkdir(parents=True, exist_ok=True)
    copied = []
    # A decision made at logical step N is drawn into zero-based frame N-1.
    for step in sorted(requested):
        source = root / mode / "rollout_frames" / f"frame_{step - 1:06d}.png"
        if source.is_file():
            destination = output / f"step_{step:04d}.png"
            shutil.copy2(source, destination)
            copied.append(str(destination))
    return copied


def plot(root: Path, results: dict[str, dict]) -> Path:
    colors = {"direct": "#4C78A8", "waypoint_dt": "#F58518", "lookahead": "#54A24B"}
    figure, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=False)
    for mode, result in results.items():
        records = result["steps"]
        steps = np.arange(1, len(records) + 1)
        color = colors[mode]
        distance = [r["evaluation_only_after_action"]["gt_distance_m"] for r in records]
        forward = [r["policy"]["action"]["forward"] for r in records]
        yaw = [r["policy"]["action"]["yaw"] for r in records]
        horizon = [
            math.hypot(*r["policy"]["waypoints"][-1]) for r in records
        ]
        stop = [r["policy"]["stop_probability"] for r in records]
        axes[0].plot(steps, distance, label=mode, color=color)
        axes[1].plot(steps, forward, label=f"{mode}:forward", color=color)
        axes[1].plot(steps, yaw, label=f"{mode}:yaw", color=color, linestyle="--", alpha=0.8)
        axes[2].plot(steps, horizon, label=mode, color=color)
        axes[3].plot(steps, stop, label=mode, color=color)
        invisible = [not r["evaluation_only_after_action"]["gt_visible"] for r in records]
        collision = [r["evaluation_only_after_action"]["human_collision"] > 0 for r in records]
        if any(invisible):
            axes[0].axvline(first_true(invisible), color=color, linestyle=":", alpha=0.55)
        if any(collision):
            axes[0].scatter(
                [first_true(collision)], [distance[first_true(collision) - 1]],
                marker="x", s=80, color=color,
            )
    axes[0].set_ylabel("GT distance (m)")
    axes[0].set_title("Dotted line: first target loss; x: first collision")
    axes[1].set_ylabel("normalized action")
    axes[2].set_ylabel("last waypoint (m)")
    axes[3].set_ylabel("stop probability")
    axes[3].set_xlabel("environment step")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    destination = root / "diagnostic_curves.png"
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def main() -> int:
    args = arguments()
    root = args.root.resolve(strict=True)
    results = {}
    summaries = {}
    for mode in MODES:
        result_path = root / mode / "result.json"
        results[mode] = json.loads(result_path.read_text())
        summaries[mode] = summarize(results[mode])
    keyframes = {
        mode: copy_keyframes(root, mode, results[mode], summaries[mode])
        for mode in MODES
    }
    curve_path = plot(root, results)
    payload = {
        "schema_version": 1,
        "comparison_contract": {
            "same_checkpoint": len({r["checkpoint"] for r in results.values()}) == 1,
            "same_episode": len({r["episode_id"] for r in results.values()}) == 1,
            "same_dataset_index": len({r["dataset_index"] for r in results.values()}) == 1,
            "task": "stt",
            "split": "train_non_locked",
            "uwb_mode": "missing",
        },
        "modes": summaries,
        "diagnostic_curves": str(curve_path),
        "keyframes": keyframes,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

