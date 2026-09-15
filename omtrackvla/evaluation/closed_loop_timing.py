"""Small wall-clock diagnostics; no Torch, simulator or GPU calls."""
from __future__ import annotations

import math
import os
import platform
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


def elapsed_ms(start_ns: int, *, clock: Callable[[], int] = time.perf_counter_ns) -> float:
    delta = clock() - start_ns
    if delta < 0:
        raise ValueError("monotonic clock moved backwards")
    return delta / 1_000_000.0


class StageTimer:
    """Consecutive, nonoverlapping parent stages; marks include intervening work."""

    def __init__(self, *, clock: Callable[[], int] = time.perf_counter_ns):
        self.clock = clock
        self.started_ns = self.previous_ns = clock()
        self.durations: dict[str, float] = {}

    def mark(self, name: str) -> float:
        if not name or name in self.durations:
            raise ValueError("stage names must be nonempty and unique within one timer")
        now = self.clock()
        if now < self.previous_ns:
            raise ValueError("monotonic clock moved backwards")
        value = (now - self.previous_ns) / 1_000_000.0
        self.durations[name] = value
        self.previous_ns = now
        return value

    @property
    def total_ms(self) -> float:
        return (self.previous_ns - self.started_ns) / 1_000_000.0


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not 0 <= fraction <= 1:
        raise ValueError("percentile fraction must be in [0,1]")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0 for value in ordered):
        raise ValueError("timings must be finite and nonnegative")
    if not ordered:
        return None
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def describe(values: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in values]
    p95 = percentile(values, .95)
    return {"count": len(values), "sum_ms": sum(values),
            "mean_ms": sum(values) / len(values) if values else None,
            "p50_ms": percentile(values, .5), "p95_ms": p95,
            "max_ms": max(values) if values else None}


def summarize_steps(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize parent stages separately from nested worker measurements."""
    def summarize(selected):
        parents: dict[str, list[float]] = {}
        workers: dict[str, list[float]] = {}
        totals = []
        complete = 0
        for record in selected:
            timing = record.get("timing", {})
            for name, value in timing.get("parent_ms", {}).items():
                parents.setdefault(name, []).append(value)
            for name, value in timing.get("worker", {}).get("decision_ms", {}).items():
                workers.setdefault(name, []).append(value)
            if timing.get("complete") is True:
                complete += 1
                total = float(timing["parent_total_ms"])
                stages = sum(timing["parent_ms"].values())
                if not math.isclose(total, stages, rel_tol=1e-10, abs_tol=1e-7):
                    raise ValueError("complete parent stage sum differs from total")
                totals.append(total)
        return {"record_count": len(selected), "complete_step_count": complete,
                "parent_step_total": describe(totals),
                "parent_stages": {key: describe(value) for key, value in sorted(parents.items())},
                "worker_nested_stages_do_not_add_to_parent": {
                    key: describe(value) for key, value in sorted(workers.items())}}
    return {"all_steps": summarize(records), "after_first_step": summarize(records[1:]),
            "excluded_warmup_policy_calls": min(1, len(records))}


def runtime_identity() -> dict[str, Any]:
    return {"pid": os.getpid(), "host": platform.node(), "python": platform.python_version(),
            "platform": platform.platform(), "clock": "time.perf_counter_ns",
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def assert_new_rollout_artifacts(output_root: Path) -> None:
    names = ("result.json", "result.partial.json", "rollout.mp4", "rollout_frames",
             "timing.json", "timing.steps.jsonl")
    existing = [name for name in names if (output_root / name).exists() or (output_root / name).is_symlink()]
    if existing:
        raise FileExistsError(f"refusing to overwrite or mix existing rollout artifacts: {existing}")


def timing_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scope": "host_closed_loop_bottleneck_diagnostic",
        "deployment_target_latency_benchmark": False,
        "thor_latency_claim": False,
        "parent_process": "Habitat/EGL, sensor calls, evaluation, PNG/report/video output",
        "worker_process": "spawned Torch/DA3, RGB preprocessing, recurrent policy, controller",
        "additivity": "parent_ms stages are consecutive; worker stages are nested within parent policy_ipc_exchange",
        "cuda_timing": "host wall time with original post-forward synchronize; no added CUDA synchronization",
        "preprocess_caveat": "CUDA transfer/kernel completion may be charged to later original synchronize",
        "env_step_caveat": "env.step includes simulator/task/internal sensors; only explicit extra sensor render is separately timed",
        "ipc_caveat": "parent receive includes worker compute; exchange minus decision is a residual, not a pure IPC benchmark",
        "worker_receive_wait_caveat": "overlaps parent simulation/output work and is not policy latency",
        "partial_report_caveat": "newest step may lack its own report/log/sidecar write duration until next snapshot",
        "timing_sidecar_caveat": "JSONL entry stops before its own write; final timing.json includes that per-step write duration",
        "final_report_caveat": "timing.json publication itself is excluded to avoid self-referential timing",
        "startup_scope": "starts at main entry, excludes interpreter and module imports before main",
    }


def append_step_timing(path: Path, record: Mapping[str, Any]) -> None:
    import json
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": record["step"], "timing": record["timing"],
            "snapshot_complete_through_stdout": True,
            "parent_subtotal_ms_before_timing_jsonl_write": sum(record["timing"]["parent_ms"].values())},
                                sort_keys=True, allow_nan=False) + "\n")
