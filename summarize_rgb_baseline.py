#!/usr/bin/env python3
"""Write per-episode and per-task CSV summaries for RGB baseline runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


SUMMARY_FIELDS = (
    "success",
    "following_rate",
    "collision",
    "finish",
    "perception_detection_rate",
    "perception_target_precision",
    "perception_target_recall",
    "perception_mean_target_iou",
    "coordinate_takeover_rate",
    "total_step",
    "distance_end_m",
)


def load_episodes(root: Path):
    episodes = []
    for path in root.glob("*/val/episodes/*.json"):
        value = json.loads(path.read_text())
        if "summary" in value and "error" not in value:
            episodes.append(value)
    return sorted(episodes, key=lambda value: (value["task"], value["dataset_index"]))


def write_csv(path: Path, fieldnames, rows) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    episodes = load_episodes(args.root)
    episode_rows = []
    for episode in episodes:
        summary = episode["summary"]
        episode_rows.append({
            "task": episode["task"],
            "split": episode["split"],
            "dataset_index": episode["dataset_index"],
            "episode_key": episode["episode_key"],
            "perception": episode["perception"],
            "person_score_threshold": episode.get("person_score_threshold"),
            "target_initialization": episode["target_initialization"],
            "lost_target_policy": episode.get("lost_target_policy"),
            **{field: summary.get(field) for field in SUMMARY_FIELDS},
            "status": summary.get("status"),
        })
    episode_fields = tuple(episode_rows[0]) if episode_rows else ()
    write_csv(args.root / "episodes.csv", episode_fields, episode_rows)

    summary_rows = []
    for task in ("stt", "dt", "at"):
        selected = [row for row in episode_rows if row["task"] == task]
        if not selected:
            continue
        summary_rows.append({
            "task": task,
            "split": selected[0]["split"],
            "episodes": len(selected),
            **{
                field: mean(float(row[field] or 0.0) for row in selected)
                for field in SUMMARY_FIELDS
            },
        })
    summary_fields = tuple(summary_rows[0]) if summary_rows else ()
    write_csv(args.root / "summary.csv", summary_fields, summary_rows)

    print(f"wrote {len(episode_rows)} episodes to {args.root}")
    for row in summary_rows:
        print(
            f"{row['task']}: n={row['episodes']} "
            f"SR={100 * row['success']:.2f}% "
            f"TR={100 * row['following_rate']:.2f}% "
            f"CR={100 * row['collision']:.2f}% "
            f"IDP={100 * row['perception_target_precision']:.2f}% "
            f"IDR={100 * row['perception_target_recall']:.2f}%"
        )


if __name__ == "__main__":
    main()
