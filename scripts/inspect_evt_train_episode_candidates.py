#!/usr/bin/env python3
"""Inspect non-locked EVT train episode metadata before Phase-3 collection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from omtrackvla.evaluation.end_to_end_closed_loop import (
    DEFAULT_SCENE_DATASET,
    configure,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", action="append", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    output = args.output.expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite episode inspection: {output}")
    if any(index < 0 for index in args.index):
        raise ValueError("episode indices must be non-negative")

    import habitat
    from habitat.datasets import make_dataset
    import evt_bench  # noqa: F401

    tasks = {}
    for task in ("stt", "dt", "at"):
        config = configure(
            habitat.get_config(
                "habitat-lab/habitat/config/benchmark/nav/track/"
                f"track_train_{task}.yaml"
            ),
            DEFAULT_SCENE_DATASET,
        )
        dataset = make_dataset(
            config.habitat.dataset.type, config=config.habitat.dataset
        )
        candidates = []
        for index in args.index:
            if index >= len(dataset.episodes):
                continue
            episode = dataset.episodes[index]
            candidates.append(
                {
                    "dataset_index": index,
                    "episode_id": str(episode.episode_id),
                    "scene_id": str(episode.scene_id),
                    "main_human_semantic_id": episode.info.get(
                        "main_human_semantic_id"
                    ),
                    "main_human": episode.info.get("main_human"),
                }
            )
        tasks[task] = {
            "episode_count": len(dataset.episodes),
            "candidates": candidates,
        }
    result = {
        "schema_version": 1,
        "stage": "v2_phase3_evt_train_episode_candidate_inspection",
        "split": "train",
        "requested_indices": args.index,
        "tasks": tasks,
        "test_locked_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
