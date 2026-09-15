#!/usr/bin/env python3
"""Strict full-val EVT-Bench person-identification comparison report."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.summarize_evt_reid import aggregate_counts, episode_counts
except ModuleNotFoundError:  # Direct execution: python scripts/<name>.py
    from summarize_evt_reid import aggregate_counts, episode_counts


DEFAULT_TASKS = ("stt", "dt", "at")


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def expected_val_counts(repo_root: Path, tasks: Iterable[str]) -> dict[str, int]:
    counts = {}
    for task in tasks:
        dataset = repo_root / "data" / "datasets" / "track" / task.upper() / "val" / "val.json.gz"
        with gzip.open(dataset, "rt", encoding="utf-8") as handle:
            counts[task] = len(json.load(handle)["episodes"])
    return counts


def _trajectory_signature(data: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for step in data.get("steps", []):
        value = (
            bool(step.get("visible")),
            round(float(step.get("distance_m") or 0.0), 6),
        )
        digest.update(repr(value).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _metric_view(counts: dict[str, Any]) -> dict[str, float]:
    precision = float(counts["perception_target_precision"])
    recall = float(counts["perception_target_recall"])
    return {
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
    }


def _finalize_group(
    count_rows: list[dict[str, int | float]],
    episode_metrics: list[dict[str, float]],
    status_counts: Counter[str],
    planned: int,
    errors: int,
    missing: list[int],
) -> dict[str, Any]:
    aggregate = aggregate_counts(count_rows)
    precision = float(aggregate["perception_target_precision"])
    recall = float(aggregate["perception_target_recall"])
    aggregate["micro_f1"] = _f1(precision, recall)
    aggregate["episode_macro_precision"] = (
        sum(item["precision"] for item in episode_metrics) / len(episode_metrics)
        if episode_metrics else 0.0
    )
    aggregate["episode_macro_recall"] = (
        sum(item["recall"] for item in episode_metrics) / len(episode_metrics)
        if episode_metrics else 0.0
    )
    aggregate["episode_macro_f1"] = (
        sum(item["f1"] for item in episode_metrics) / len(episode_metrics)
        if episode_metrics else 0.0
    )
    aggregate.update(
        completed_episodes=len(count_rows),
        planned_episodes=planned,
        coverage=(len(count_rows) / planned if planned else 0.0),
        error_episodes=errors,
        missing_episodes=len(missing),
        missing_dataset_indices=missing[:100],
        status_counts=dict(sorted(status_counts.items())),
    )
    return aggregate


def _load_method(
    root: Path,
    method: str,
    tasks: tuple[str, ...],
    expected: dict[str, int],
) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    all_counts: list[dict[str, int | float]] = []
    all_episode_metrics: list[dict[str, float]] = []
    all_statuses: Counter[str] = Counter()
    total_errors = 0
    total_missing: list[int] = []
    paired: dict[tuple[str, int], dict[str, Any]] = {}
    by_task: dict[str, Any] = {}
    protocol_values: dict[str, set[Any]] = {
        "controller": set(),
        "controller_input": set(),
        "perception": set(),
    }
    manifest_issues: list[str] = []

    for task in tasks:
        task_counts: list[dict[str, int | float]] = []
        task_episode_metrics: list[dict[str, float]] = []
        task_statuses: Counter[str] = Counter()
        seen: set[int] = set()
        errors = 0
        manifest_paths = sorted((root / method / task / "val").glob("shard_*_manifest.json"))
        manifests = [json.loads(path.read_text(encoding="utf-8")) for path in manifest_paths]
        if not manifests:
            manifest_issues.append(f"{method}/{task}: no shard manifests")
        else:
            shard_counts = {int(item.get("num_shards", -1)) for item in manifests}
            shard_ids = {int(item.get("shard_id", -1)) for item in manifests}
            if len(shard_counts) != 1:
                manifest_issues.append(f"{method}/{task}: inconsistent num_shards {sorted(shard_counts)}")
            else:
                shard_count = next(iter(shard_counts))
                if shard_ids != set(range(shard_count)):
                    manifest_issues.append(
                        f"{method}/{task}: shard ids incomplete ({len(shard_ids)}/{shard_count})"
                    )
            required_manifest = {
                "dataset_episodes": expected[task],
                "max_steps": 300,
                "save_steps": True,
                "controller": "reactive",
                "target_mode": "point",
                "controller_input": "oracle-pointgoal",
                "person_detector_architecture": "fasterrcnn_resnet50_fpn_v2",
                "person_reid_backend": method,
            }
            for manifest_path, manifest in zip(manifest_paths, manifests):
                for key, expected_value in required_manifest.items():
                    if manifest.get(key) != expected_value:
                        manifest_issues.append(
                            f"{manifest_path}: {key}={manifest.get(key)!r}, expected {expected_value!r}"
                        )
        episode_dir = root / method / task / "val" / "episodes"
        for path in sorted(episode_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            index = int(data["dataset_index"])
            if index in seen:
                raise RuntimeError(f"{method}/{task}: duplicate dataset_index {index}")
            seen.add(index)
            if "error" in data:
                errors += 1
                continue
            if "summary" not in data or "steps" not in data:
                raise RuntimeError(f"{path}: full-val result requires summary and saved steps")
            if not 0 <= index < expected[task]:
                raise RuntimeError(f"{method}/{task}: dataset_index {index} is out of range")
            counts = episode_counts(data)
            metrics = _metric_view(aggregate_counts([counts]))
            status = str(data["summary"].get("status", "Unknown"))
            task_counts.append(counts)
            task_episode_metrics.append(metrics)
            task_statuses[status] += 1
            for key in protocol_values:
                protocol_values[key].add(data.get(key))
            paired[(task, index)] = {
                "episode_id": str(data.get("episode_id")),
                "scene_id": data.get("scene_id"),
                "steps": len(data["steps"]),
                "trajectory_sha256": _trajectory_signature(data),
            }

        missing = sorted(set(range(expected[task])) - seen)
        by_task[task] = _finalize_group(
            task_counts,
            task_episode_metrics,
            task_statuses,
            expected[task],
            errors,
            missing,
        )
        all_counts.extend(task_counts)
        all_episode_metrics.extend(task_episode_metrics)
        all_statuses.update(task_statuses)
        total_errors += errors
        total_missing.extend(missing)

    aggregate = _finalize_group(
        all_counts,
        all_episode_metrics,
        all_statuses,
        sum(expected.values()),
        total_errors,
        total_missing,
    )
    complete = (
        aggregate["completed_episodes"] == aggregate["planned_episodes"]
        and not aggregate["error_episodes"]
        and not aggregate["missing_episodes"]
        and not manifest_issues
    )
    return ({
        "complete": complete,
        "manifest_validation": {
            "passed": not manifest_issues,
            "issues": manifest_issues[:100],
        },
        "protocol_values": {key: sorted(str(v) for v in values) for key, values in protocol_values.items()},
        "aggregate": aggregate,
        "by_task": by_task,
    }, paired)


def summarize_full(
    root: Path,
    expected: dict[str, int],
    methods: tuple[str, ...] = ("osnet", "kpr"),
    tasks: tuple[str, ...] = DEFAULT_TASKS,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "EVT-Bench",
        "split": "val",
        "evaluation_type": "full-val person-identification comparison",
        "tasks": list(tasks),
        "expected_episodes": expected,
        "planned_episodes_per_method": sum(expected.values()),
        "protocol": {
            "controller": "reactive",
            "controller_input": "oracle-pointgoal",
            "max_episode_steps": 300,
            "artificial_step_truncation": False,
            "target_initialization": "goal-crop",
            "test_locked_used": False,
            "threshold_selection_split": "train only",
        },
        "methods": {},
    }
    pair_data = {}
    for method in methods:
        method_report, pair_data[method] = _load_method(root, method, tasks, expected)
        report["methods"][method] = method_report

    reference = methods[0]
    mismatch_examples = []
    mismatch_count = 0
    expected_keys = {(task, index) for task in tasks for index in range(expected[task])}
    for method in methods[1:]:
        for key in sorted(expected_keys):
            if pair_data[reference].get(key) != pair_data[method].get(key):
                mismatch_count += 1
                if len(mismatch_examples) < 100:
                    mismatch_examples.append({
                        "task": key[0],
                        "dataset_index": key[1],
                        "reference_method": reference,
                        "method": method,
                    })
    paired_complete = all(report["methods"][method]["complete"] for method in methods)
    report["paired_protocol"] = {
        "consistent": paired_complete and mismatch_count == 0,
        "mismatch_count": mismatch_count,
        "mismatch_examples": mismatch_examples,
    }

    ranking = sorted(
        methods,
        key=lambda method: (
            report["methods"][method]["aggregate"]["micro_f1"],
            report["methods"][method]["aggregate"]["perception_target_precision"],
            report["methods"][method]["aggregate"]["perception_target_recall"],
        ),
        reverse=True,
    )
    report["ranking"] = ranking
    report["winner"] = ranking[0] if paired_complete and mismatch_count == 0 else None
    report["complete"] = paired_complete and mismatch_count == 0
    return report


CSV_FIELDS = [
    "日期", "Benchmark", "Split", "评测类型", "方法", "统计范围",
    "完成Episode", "计划Episode", "覆盖率(%)", "评测步数", "GT可见帧",
    "输出帧", "正确输出帧", "Precision(%)", "Recall(%)", "Micro-F1(%)",
    "Episode-Macro-F1(%)", "平均输出IoU", "Detector目标召回(%)",
    "Memory更新次数", "正确Memory更新次数", "Memory更新Precision(%)",
    "碰撞Episode", "MaxSteps Episode", "正常结束Episode", "错误Episode",
    "配对轨迹一致", "排名", "结论", "备注",
]


def _pct(value: float | None) -> str:
    return "" if value is None else f"{100.0 * float(value):.4f}"


def write_csv(report: dict[str, Any], path: Path) -> None:
    methods = list(report["methods"])
    scopes = [("总体", "aggregate")] + [(task.upper(), task) for task in report["tasks"]]
    ranks: dict[tuple[str, str], int] = {}
    for label, key in scopes:
        metric_rows = {
            method: (
                report["methods"][method]["aggregate"]
                if key == "aggregate"
                else report["methods"][method]["by_task"][key.lower()]
            )
            for method in methods
        }
        ordered = sorted(methods, key=lambda method: metric_rows[method]["micro_f1"], reverse=True)
        for rank, method in enumerate(ordered, 1):
            ranks[(label, method)] = rank

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for label, key in scopes:
            for method in methods:
                values = (
                    report["methods"][method]["aggregate"]
                    if key == "aggregate"
                    else report["methods"][method]["by_task"][key.lower()]
                )
                statuses = values["status_counts"]
                memory_precision = values["perception_memory_update_precision"]
                conclusion = ""
                if label == "总体" and report.get("winner"):
                    conclusion = (
                        "全量val主指标最优"
                        if method == report["winner"]
                        else f"低于{report['winner']}"
                    )
                writer.writerow({
                    "日期": date.today().isoformat(),
                    "Benchmark": report["benchmark"],
                    "Split": report["split"],
                    "评测类型": "全量val人物识别对照",
                    "方法": method,
                    "统计范围": label,
                    "完成Episode": values["completed_episodes"],
                    "计划Episode": values["planned_episodes"],
                    "覆盖率(%)": _pct(values["coverage"]),
                    "评测步数": values["steps"],
                    "GT可见帧": values["gt_visible"],
                    "输出帧": values["detected"],
                    "正确输出帧": values["correct"],
                    "Precision(%)": _pct(values["perception_target_precision"]),
                    "Recall(%)": _pct(values["perception_target_recall"]),
                    "Micro-F1(%)": _pct(values["micro_f1"]),
                    "Episode-Macro-F1(%)": _pct(values["episode_macro_f1"]),
                    "平均输出IoU": f"{float(values['perception_mean_target_iou']):.6f}",
                    "Detector目标召回(%)": _pct(values["detector_target_recall"]),
                    "Memory更新次数": values["memory_updates"],
                    "正确Memory更新次数": values["correct_memory_updates"],
                    "Memory更新Precision(%)": _pct(memory_precision),
                    "碰撞Episode": statuses.get("Collision", 0),
                    "MaxSteps Episode": statuses.get("MaxSteps", 0),
                    "正常结束Episode": statuses.get("Normal", 0),
                    "错误Episode": values["error_episodes"],
                    "配对轨迹一致": report["paired_protocol"]["consistent"],
                    "排名": ranks[(label, method)],
                    "结论": conclusion,
                    "备注": "GT point + reactive controller；官方300步上限且无人工截断；train-only阈值；感知结果不等同于导航成绩",
                })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--methods", nargs="+", default=("osnet", "kpr"))
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    tasks = tuple(args.tasks)
    expected = expected_val_counts(args.repo_root, tasks)
    report = summarize_full(args.root, expected, tuple(args.methods), tasks)
    output_json = args.output_json or args.root / "REPORT.json"
    output_csv = args.output_csv or args.root / "REPORT.csv"
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(report, output_csv)
    print(json.dumps({
        "complete": report["complete"],
        "winner": report["winner"],
        "ranking": report["ranking"],
        "json": str(output_json),
        "csv": str(output_csv),
    }, sort_keys=True))
    if args.require_complete and not report["complete"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
