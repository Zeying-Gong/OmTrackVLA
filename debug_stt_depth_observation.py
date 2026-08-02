#!/usr/bin/env python3
"""Inspect STT jaw RGB/depth/panoptic observations after one environment reset."""

from __future__ import annotations

import argparse
import json

import habitat
import numpy as np

import evt_bench  # noqa: F401 - registers tracking sensors, actions, and TrackEnv

from oracle_modular_follow import DEFAULT_SCENE_DATASET, configure


SENSOR_KEYS = (
    "agent_1_articulated_agent_jaw_rgb",
    "agent_1_articulated_agent_jaw_depth",
    "agent_1_articulated_agent_jaw_panoptic",
)


def array_stats(value) -> dict:
    array = np.asarray(value)
    finite = np.isfinite(array)
    finite_values = array[finite]
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite_rate": float(finite.mean()),
        "nonzero_rate": float(np.count_nonzero(array) / array.size),
        "min": float(finite_values.min()),
        "max": float(finite_values.max()),
        "mean": float(finite_values.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=(
            "habitat-lab/habitat/config/benchmark/nav/track/"
            "track_infer_stt_depth_debug.yaml"
        ),
    )
    parser.add_argument("--scene-dataset", default=DEFAULT_SCENE_DATASET)
    args = parser.parse_args()

    config = configure(habitat.get_config(args.config), args.scene_dataset)
    dataset = habitat.make_dataset(
        config.habitat.dataset.type, config=config.habitat.dataset
    )
    dataset.episodes = dataset.episodes[:1]
    with habitat.TrackEnv(config=config, dataset=dataset) as env:
        observations = env.reset()
        missing = [key for key in SENSOR_KEYS if key not in observations]
        if missing:
            raise KeyError(f"Missing observations: {missing}")
        print(
            json.dumps(
                {key: array_stats(observations[key]) for key in SENSOR_KEYS},
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
