"""Fail-closed run ownership and exact-resume metadata for training jobs.

The module has no required third-party dependencies. Torch / NumPy are injected
so importing it on a CPU host cannot initialize CUDA. Every distributed rank
must capture its own RNG and sampler state at the SAME optimizer boundary.
Checkpoints containing Python RNG objects require loading only trusted files.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import socket
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

SCHEMA = 1


class ResumeError(ValueError):
    """The checkpoint cannot reproduce the requested run."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_hash(root: str | Path, relative_paths: list[str]) -> str:
    """Hash explicit source files including names; do not hash only git HEAD."""
    root = Path(root).resolve(strict=True)
    result = {}
    for name in sorted(relative_paths):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("source paths must remain under repository root")
        path = (root / relative).resolve(strict=True)
        if root not in path.parents:
            raise ValueError("source path escaped repository root")
        result[relative.as_posix()] = file_hash(path)
    if not result:
        raise ValueError("at least one source file is required")
    return canonical_hash(result)


def run_contract(*, effective_config: Mapping[str, Any], code_sha256: str,
                 data_hashes: Mapping[str, str], world_size: int,
                 runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Include resolved CLI step budget, base config, schedule, precision in config.

    runtime should include Python / Torch / CUDA versions and deterministic flags;
    data_hashes should include train and validation manifests and parent weights.
    """
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    if len(code_sha256) != 64 or not data_hashes or not runtime:
        raise ValueError("code, data and runtime identities are required")
    return {
        "schema_version": SCHEMA,
        "world_size": world_size,
        "config_sha256": canonical_hash(effective_config),
        "code_sha256": code_sha256,
        "data_sha256": canonical_hash(data_hashes),
        "runtime_sha256": canonical_hash(runtime),
    }


def assert_contract(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for key in ("schema_version", "world_size", "config_sha256", "code_sha256",
                "data_sha256", "runtime_sha256"):
        if key not in actual or key not in expected or actual[key] != expected[key]:
            raise ResumeError(f"resume contract mismatch: {key}")


def atomic_write(path: str | Path, writer: Callable[[Any], None]) -> None:
    """writer receives a binary handle; errors preserve the previous artifact."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"refusing to overwrite a symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{path.name}.",
                                         suffix=".tmp", dir=path.parent,
                                         delete=False) as handle:
            temporary = Path(handle.name)
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_json(path: str | Path, value: Any) -> None:
    content = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    atomic_write(path, lambda handle: handle.write(content))


def capture_rng(*, rank: int, numpy: Any = None, torch: Any = None,
                generators: Mapping[str, Any] | None = None) -> dict[str, Any]:
    state = {"rank": rank, "python": random.getstate()}
    if numpy is not None:
        state["numpy"] = numpy.random.get_state()
    if torch is not None:
        state["torch_cpu"] = torch.get_rng_state().clone()
        # is_available() is intentionally avoided: CPU verification must not
        # initialize or touch the shared GPUs. Production initializes CUDA first.
        if torch.cuda.is_initialized():
            device = torch.cuda.current_device()
            state["torch_cuda_current"] = torch.cuda.get_rng_state(device).cpu().clone()
    state["generators"] = {
        name: generator.get_state().clone()
        for name, generator in (generators or {}).items()
    }
    return state


def restore_rng(state: Mapping[str, Any], *, rank: int, numpy: Any = None,
                torch: Any = None,
                generators: Mapping[str, Any] | None = None) -> None:
    if state.get("rank") != rank:
        raise ResumeError("rank RNG state mismatch")
    if ("numpy" in state) != (numpy is not None):
        raise ResumeError("NumPy RNG state missing or unexpected")
    if ("torch_cpu" in state) != (torch is not None):
        raise ResumeError("Torch RNG state missing or unexpected")
    if set(state.get("generators", {})) != set(generators or {}):
        raise ResumeError("explicit RNG generator names differ")
    if torch is not None:
        if ("torch_cuda_current" in state) != torch.cuda.is_initialized():
            raise ResumeError("CUDA initialization differs from checkpoint")
    random.setstate(state["python"])
    if numpy is not None:
        numpy.random.set_state(state["numpy"])
    if torch is not None:
        torch.set_rng_state(state["torch_cpu"].cpu())
        if "torch_cuda_current" in state:
            torch.cuda.set_rng_state(state["torch_cuda_current"].cpu(),
                                     torch.cuda.current_device())
    for name, generator in (generators or {}).items():
        generator.set_state(state["generators"][name].cpu())


def resume_payload(*, contract: Mapping[str, Any], global_step: int,
                   rank_states: list[dict[str, Any]], best: Mapping[str, Any] | None = None
                   ) -> dict[str, Any]:
    """Add this as checkpoint['exact_resume']; model / optimizer remain caller-owned.

    rank_states entries are {'rng': capture_rng(...), 'data_state': ...}.
    For independent per-step random sampling data_state may be {'kind':
    'step_random_sampling'}; for a loader save epoch, batch offset and generator.
    Do not promise exact replay for unsaved worker prefetch / augmentation state.
    """
    value = {"contract": dict(contract), "global_step": global_step,
             "rank_states": rank_states, "best": None if best is None else dict(best)}
    validate_resume(value, contract)
    return value


def validate_resume(payload: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    assert_contract(payload.get("contract", {}), expected)
    step = payload.get("global_step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ResumeError("invalid optimizer step")
    states = payload.get("rank_states", [])
    if len(states) != expected["world_size"]:
        raise ResumeError("one RNG/data state per rank is required")
    for rank, value in enumerate(states):
        if value.get("rng", {}).get("rank") != rank or "data_state" not in value:
            raise ResumeError(f"missing or reordered rank state: {rank}")


@dataclass
class BestMetric:
    """Select a best artifact only using an explicitly named held-out validation."""
    name: str
    split: str = "val"
    mode: str = "min"
    value: float | None = None
    step: int | None = None

    def __post_init__(self) -> None:
        if not self.name or self.split != "val" or self.mode not in {"min", "max"}:
            raise ValueError("best metric requires a named validation metric and min/max mode")
        if (self.value is None) != (self.step is None):
            raise ValueError("restored best metric requires both value and step")
        if self.value is not None and (not math.isfinite(self.value)
                or isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0):
            raise ValueError("invalid restored best metric")

    def improves(self, value: float, *, split: str, step: int) -> bool:
        if split != self.split:
            raise ValueError("training / test data cannot select a validation checkpoint")
        if step < 0 or not math.isfinite(value):
            raise ValueError("validation metric must be finite and step nonnegative")
        return self.value is None or (value < self.value if self.mode == "min" else value > self.value)

    def commit(self, value: float, *, split: str, step: int) -> None:
        """Commit AFTER atomic checkpoint serialization succeeds."""
        if not self.improves(value, split=split, step=step):
            raise ValueError("metric did not improve")
        self.value, self.step = float(value), step


class RunDirectory:
    """Single-writer ownership for rank zero, with explicit complete / failed state.

    Do not enter this on every DDP rank. Acquire once before launching torchrun or
    on rank zero and broadcast acquisition success before any distributed work.
    Stale locks are intentionally never auto-deleted (remote PID checks are unsafe).
    """
    def __init__(self, path: str | Path, contract: Mapping[str, Any], *, resume: bool = False):
        self.path = Path(path).absolute()
        self.contract = dict(contract)
        self.resume = resume
        self.token = uuid.uuid4().hex
        self.owned = False
        self.completed = False

    @property
    def lock_path(self) -> Path:
        return self.path / ".run.lock"

    @property
    def state_path(self) -> Path:
        return self.path / "run_state.json"

    def _state(self, status: str, **extra: Any) -> None:
        atomic_json(self.state_path, {"schema_version": SCHEMA, "status": status,
                    "token": self.token, "updated_utc": utc_now(),
                    "contract": self.contract, **extra})

    def __enter__(self) -> "RunDirectory":
        if self.path.is_symlink():
            raise ValueError("run directory must not be a symlink")
        if self.resume:
            if not self.path.is_dir():
                raise ResumeError("resume directory does not exist")
        else:
            self.path.mkdir(parents=True, exist_ok=False)
        owner = {"token": self.token, "pid": os.getpid(), "host": socket.gethostname(),
                 "created_utc": utc_now()}
        descriptor = os.open(self.lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(owner, handle)
                handle.flush()
                os.fsync(handle.fileno())
            self.owned = True
            if self.resume:
                state = json.loads(self.state_path.read_text(encoding="utf-8"))
                if state.get("status") == "complete":
                    raise ResumeError("completed run is immutable; use a new run directory")
                if state.get("status") not in {"running", "failed"}:
                    raise ResumeError("unknown run state")
                assert_contract(state.get("contract", {}), self.contract)
            self._state("running")
            return self
        except BaseException:
            self._release()
            raise

    def complete(self, *, global_step: int, artifacts: Mapping[str, str]) -> None:
        if not self.owned or self.completed:
            raise RuntimeError("run is not owned or already complete")
        if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 1:
            raise ValueError("completion requires a positive optimizer step")
        if not artifacts:
            raise ValueError("completion requires verified artifacts")
        for relative, expected_hash in artifacts.items():
            item = (self.path / relative).resolve(strict=True)
            if self.path.resolve() not in item.parents or file_hash(item) != expected_hash:
                raise ValueError("completion artifact path/hash mismatch")
        self._state("complete", global_step=global_step, artifacts=dict(artifacts))
        self.completed = True

    def _release(self) -> None:
        if self.owned:
            owner = json.loads(self.lock_path.read_text(encoding="utf-8"))
            if owner.get("token") != self.token:
                raise RuntimeError("run lock ownership changed; refusing to remove it")
            self.lock_path.unlink()
            self.owned = False

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        try:
            if not self.completed:
                reason = "exited without complete()" if exc is None else f"{type(exc).__name__}: {exc}"
                self._state("failed", error=reason[:2000])
        finally:
            self._release()
        return False
