#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _episode_sort_key(path: Path) -> tuple[str, int | str]:
    stem = path.stem
    episode: int | str
    try:
        episode = int(stem)
    except ValueError:
        episode = stem
    return (path.parent.name, episode)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _write_markdown(rows: list[dict[str, str]], out_path: Path | None) -> None:
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[h] for h in headers) + " |")
    text = "\n".join(lines)
    if out_path is None:
        print(text)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n")


def _write_csv(rows: list[dict[str, str]], out_path: Path | None) -> None:
    headers = list(rows[0].keys())
    if out_path is None:
        import sys

        writer = csv.DictWriter(sys.stdout, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--expected", type=int, default=1405)
    parser.add_argument("--table", action="store_true", help="Print per-episode metrics plus a TOTAL row")
    parser.add_argument("--format", choices=["markdown", "csv"], default="markdown")
    parser.add_argument("--out", type=Path, default=None, help="Optional output path for --table")
    args = parser.parse_args()

    # Results use <result_dir>/<scene>/<episode>.json. Avoid recursive scans:
    # _live can contain hundreds of thousands of frame images on shared storage.
    files = sorted(
        (
            p for p in args.result_dir.glob("*/*.json")
            if not p.name.endswith("_info.json")
        ),
        key=_episode_sort_key,
    )
    mp4_count = sum(1 for _ in args.result_dir.glob("*/*.mp4"))
    video_enabled = mp4_count > 0

    rows: list[dict[str, str]] = []
    totals = {
        "success": 0.0,
        "following_rate": 0.0,
        "collision": 0.0,
        "following_step": 0.0,
        "total_step": 0.0,
        "finish": 0.0,
        "video_exists": 0.0,
    }
    for path in files:
        with path.open("r") as fh:
            val = json.load(fh)

        video_path = path.with_suffix(".mp4")
        success = _as_float(val.get("success", 0.0))
        following_rate = _as_float(val.get("following_rate", 0.0))
        collision = _as_float(val.get("collision", 0.0))
        following_step = _as_float(val.get("following_step", 0.0))
        total_step = _as_float(val.get("total_step", 0.0))
        finish = bool(val.get("finish", False))
        video_exists = video_path.exists()

        totals["success"] += success
        totals["following_rate"] += following_rate
        totals["collision"] += collision
        totals["following_step"] += following_step
        totals["total_step"] += total_step
        totals["finish"] += 1.0 if finish else 0.0
        totals["video_exists"] += 1.0 if video_exists else 0.0

        rows.append(
            {
                "scene": path.parent.name,
                "episode": path.stem,
                "finish": "1" if finish else "0",
                "status": str(val.get("status", "")),
                "success": _fmt(success, 1),
                "following_rate": _fmt(following_rate),
                "following_step": str(int(following_step)),
                "total_step": str(int(total_step)),
                "collision": _fmt(collision, 1),
                "video_exists": "1" if video_exists else "0",
                "video_path": str(video_path) if video_exists else "",
            }
        )

    n = len(rows)
    if n == 0:
        print(f"{args.result_dir}: no result json files found")
        return

    sr = totals["success"] / n * 100.0
    tr = totals["following_rate"] / n * 100.0
    cr = totals["collision"] / n * 100.0
    done = "complete" if n == args.expected else f"partial {n}/{args.expected}"
    print(f"{args.result_dir} [{done}]")
    print(f"SR / TR / CR = {sr:.2f} / {tr:.2f} / {cr:.2f}")
    print(f"Videos = {int(totals['video_exists'])}/{n} mp4 files (SAVE_VIDEO/TRACKVLA_SAVE_VIDEO {'enabled or used previously' if video_enabled else 'not detected'})")

    if not args.table:
        return

    rows.append(
        {
            "scene": "TOTAL",
            "episode": str(n),
            "finish": _fmt(totals["finish"] / n * 100.0, 2) + "%",
            "status": done,
            "success": _fmt(sr, 2) + "%",
            "following_rate": _fmt(tr, 2) + "%",
            "following_step": str(int(totals["following_step"])),
            "total_step": str(int(totals["total_step"])),
            "collision": _fmt(cr, 2) + "%",
            "video_exists": f"{int(totals['video_exists'])}/{n}",
            "video_path": "",
        }
    )

    if args.format == "csv":
        _write_csv(rows, args.out)
    else:
        _write_markdown(rows, args.out)


if __name__ == "__main__":
    main()
