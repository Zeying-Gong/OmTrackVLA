#!/usr/bin/env python3
"""Report aggregate progress for sharded modular-oracle evaluation."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from pathlib import Path


EXPECTED_CONTROLLER_VERSION = int(os.environ.get("ORACLE_CONTROLLER_VERSION", "5"))


def dataset_count(repo_root: Path, task: str, split: str) -> int:
    path = (
        repo_root / "data" / "datasets" / "track" / task.upper()
        / split / f"{split}.json.gz"
    )
    with gzip.open(path, "rt") as handle:
        return len(json.load(handle)["episodes"])


def eligible_indices(
    total: int,
    num_shards: int,
    excluded: set[int],
    max_episodes_per_shard: int | None,
) -> set[int]:
    eligible = set()
    for shard_id in range(num_shards):
        shard_indices = [
            index for index in range(shard_id, total, num_shards)
            if index not in excluded
        ]
        if max_episodes_per_shard is not None:
            shard_indices = shard_indices[:max_episodes_per_shard]
        eligible.update(shard_indices)
    return eligible


def completed_count(
    output_root: Path,
    task: str,
    split: str,
    eligible: set[int],
    save_video: bool,
    require_success: bool,
) -> int:
    group_root = output_root / task / split
    videos_dir = group_root / "videos"
    completed = set()
    for path in (group_root / "episodes").glob("*.json"):
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        index = value.get("dataset_index")
        if (
            index not in eligible
            or value.get("controller_version") != EXPECTED_CONTROLLER_VERSION
            or "summary" not in value
        ):
            continue
        if require_success and not value["summary"].get("success", 0.0):
            continue
        if save_video:
            video_path = videos_dir / f"{path.stem}.mp4"
            if not video_path.exists() or video_path.stat().st_size <= 0:
                continue
        completed.add(int(index))
    return len(completed)


def format_duration(seconds: float | None) -> str:
    if seconds is None or not float(seconds) >= 0.0:
        return "unknown"
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def progress_line(
    task: str,
    split: str,
    completed: int,
    total: int,
    baseline: int,
    elapsed: float,
) -> str:
    fraction = min(1.0, completed / total) if total else 1.0
    width = 24
    filled = min(width, int(fraction * width))
    bar = "#" * filled + "-" * (width - filled)
    produced = max(0, completed - baseline)
    rate_per_second = produced / elapsed if elapsed > 0.0 else 0.0
    remaining = max(0, total - completed)
    eta = remaining / rate_per_second if rate_per_second > 0.0 else None
    return (
        f"[oracle-progress] {task}/{split} [{bar}] "
        f"{completed}/{total} ({100.0 * fraction:.2f}%) "
        f"session={produced} elapsed={format_duration(elapsed)} "
        f"rate={60.0 * rate_per_second:.2f} eps/min "
        f"eta={format_duration(eta)}"
    )


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--task", choices=("stt", "dt", "at"), required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--exclude-dataset-indices", default="")
    parser.add_argument("--max-episodes-per-shard", type=int)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--require-success", action="store_true")
    parser.add_argument("--baseline", type=int)
    parser.add_argument("--started-at", type=float)
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--watch-pid", type=int)
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    excluded = {
        int(value) for value in args.exclude_dataset_indices.split(",") if value.strip()
    }
    total_dataset = dataset_count(args.repo_root, args.task, args.split)
    eligible = eligible_indices(
        total_dataset,
        args.num_shards,
        excluded,
        args.max_episodes_per_shard,
    )

    def count() -> int:
        return completed_count(
            args.output_root,
            args.task,
            args.split,
            eligible,
            args.save_video,
            args.require_success,
        )

    initial = count()
    if args.count_only:
        print(initial)
        return
    baseline = initial if args.baseline is None else args.baseline
    started_at = time.time() if args.started_at is None else args.started_at
    while True:
        completed = count()
        print(
            progress_line(
                args.task,
                args.split,
                completed,
                len(eligible),
                baseline,
                max(0.0, time.time() - started_at),
            ),
            flush=True,
        )
        if args.once or completed >= len(eligible):
            return
        if args.watch_pid is not None and not process_exists(args.watch_pid):
            return
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    main()
