"""Bounded failure-state collection launcher; never a training release or score.

Developer mode runs one real 24-action / k=20 interface check. Formal mode
requires a scheduler allocation of eight GPUs and consumes all 126 planned
failure keys. Only process groups created by this invocation are terminated.
"""
import argparse
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import threading
import time
from urllib.parse import urlparse

from wa.wm.failure_state_protocol import (
    CHECKPOINT_PATH, CHECKPOINT_SHA, EXPERIMENT, PROTOCOL_SHA, REPAIR_SHA, load_plan,
)

R = Path("/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928")
S = Path(__file__).resolve().parents[2]
B = R.parent / "OmTrackVLA-da3-polar-20260924"
W = R.parent / "WLA-EVT-20260925"
E = R.parent.parent / "envs"
L = R.parent / "LightNav-0"
ENCODER = "/data/nas_ray/home/zeying.gong/datasets_processed/threepanel_debug_v2/traversability_stepp_v1/_weights/dinov2_vits14_pretrain.pth"
WLA_CHECKPOINT = "/data/nas_ray/project/md-ak/users/zeying.gong/job_58346/task_69086/wla_teacher_train/training/checkpoints/step-0043203.pt"
OUTPUT_BASES = (R / "artifacts", Path("/data/nas_ray/project/md-ak/users/zeying.gong"))
STARTUP_SECONDS = 600
DEVELOPER_SECONDS = 600
MAX_RUNS_PER_ENTRY = 16  # Student + at most five (two teachers + one verification).
SECONDS_PER_RUN = 600
ENV_RECORD_KEYS = (
    "CUDA_VISIBLE_DEVICES", "PYTHONPATH", "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE",
    "PYTHONSAFEPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE", "LD_LIBRARY_PATH", "WA_DEVELOPMENT", "WA_DIAG_CONTROLLER",
    "WA_SEMANTIC_PLY_FIX", "WA_STUDENT_EVAL", "WA_EVAL_CHECKPOINT_SHA",
    "WA_EVAL_CHECKPOINT_STEP", "WA_INIT_REPAIR_PLAN", "WA_INIT_REPAIR_PLAN_SHA",
    "WA_EVAL_MODE", "WA_CUDA_MOUNT_COMPAT", "OMTRACKVLA_XVFB_DISPLAY_NUM", "OMTRACKVLA_HAB_SIM_GLX_ROOT",
    "OMTRACKVLA_XVFB_LOG", "OMTRACKVLA_GLX_LIB_DIRS", "XKB_CONFIG_ROOT",
)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output")
    p.add_argument("--plan", required=True)
    p.add_argument("--plan-sha", required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--development-key")
    mode.add_argument("--formal", action="store_true")
    p.add_argument("--port-base", type=int, default=19170)
    p.add_argument("--display-base", type=int, default=570)
    return p.parse_args(argv)


def validate_route(args, env, cuda_count=None):
    """Pure routing guard. CUDA is queried separately only for a formal launch."""
    if bool(args.formal) == (args.development_key is not None):
        raise ValueError("choose exactly formal or an explicit developer key")
    if not re.fullmatch(r"[0-9a-f]{64}", args.plan_sha):
        raise ValueError("explicit plan SHA256 required")
    if not Path(args.plan).is_absolute():
        raise ValueError("absolute pinned plan path required")
    if any(env.get(k) for k in ("WA_RESUME_PLAN", "WA_TARGETED_PLAN")):
        raise ValueError("failure-state collection cannot inherit old row reuse")
    raw = env.get("CUDA_VISIBLE_DEVICES")
    if args.formal:
        if not env.get("MD_AK_JOB_ID") or not env.get("MD_AK_TASK_ID"):
            raise ValueError("formal launch requires scheduler Job and Task identity")
        if type(cuda_count) is not int or cuda_count != 8:
            raise ValueError("formal collection requires eight actually visible CUDA GPUs")
        devices = raw.split(",") if raw else [str(i) for i in range(8)]
    else:
        if env.get("MD_AK_JOB_ID") or env.get("MD_AK_TASK_ID"):
            raise ValueError("developer check cannot run as a cluster smoke")
        if not re.fullmatch(r"[A-Za-z0-9]+/[0-9]+", args.development_key):
            raise ValueError("explicit STT scene/episode key required")
        if not raw:
            raise ValueError("developer GPU must be explicitly prechecked and selected")
        devices = raw.split(",")
    expected = 8 if args.formal else 1
    if len(devices) != expected or len(set(devices)) != expected or any(
        not re.fullmatch(r"(?:[0-9]+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9/-]+)", d)
        for d in devices
    ):
        raise ValueError("visible-device list must be explicit, distinct and match scope")
    if not 1024 <= args.port_base <= 65535 - expected + 1:
        raise ValueError("invalid teacher port range")
    if any(p in {18798, 18799, 18800} for p in range(args.port_base, args.port_base + expected)):
        raise ValueError("existing review ports are reserved")
    if not 1 <= args.display_base <= 4096 - expected:
        raise ValueError("invalid X display range")
    if 530 in range(args.display_base, args.display_base + expected):
        raise ValueError("existing X530 is reserved")
    return devices


def output_path(raw, protected=(), *, allowed_roots=OUTPUT_BASES):
    path = Path(raw)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("explicit absolute output directory required")
    if path.exists() or path.is_symlink():
        raise ValueError("output already exists; no overwrite or resume")
    resolved = path.resolve()
    if not any(parent.resolve() in resolved.parents for parent in allowed_roots):
        raise ValueError("output must be an independent persistent project NAS directory")
    for source in protected:
        source = Path(source).resolve()
        if resolved == source or source in resolved.parents:
            raise ValueError("output overlaps protected source or old artifacts")
    return resolved


def check_address(port, display):
    """No listener is killed or reused, including stale X lock/socket paths."""
    for name in (f"/tmp/.X11-unix/X{display}", f"/tmp/.X{display}-lock"):
        if os.path.lexists(name):
            raise ValueError("X display occupied or stale: " + name)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", port))
    # Linux abstract X sockets may exist without a filesystem entry.
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        if probe.connect_ex("\0/tmp/.X11-unix/X" + str(display)) == 0:
            raise ValueError("abstract X display occupied")


def planned_entries(plan, args, lane):
    if args.formal:
        return plan["lanes"][lane]
    entries = [e for e in plan["entries"] if e["key"] == args.development_key]
    if len(entries) != 1:
        raise ValueError("developer key must belong to the frozen failure set")
    return entries


def make_commands(args, plan, lane, device, dest, inherited, ln_libraries):
    """Pure command/environment construction; does not import CUDA or write."""
    dest = Path(dest)
    env = dict(inherited, CUDA_VISIBLE_DEVICES=device, PYTHONNOUSERSITE="1",
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TRACKVLA_SAVE_VIDEO="0",
               SAVE_VIDEO="0", TRACKVLA_VERBOSE_STEPS="0", TRACKVLA_LIVE_FRAME_INTERVAL="0",
               HABITAT_SIM_LOG="quiet", MAGNUM_LOG="quiet")
    env.pop("DA3_EVT_CHECKPOINT", None)
    xb = Path(env.get("XVFB_ROOT", "/tmp/omtrackvla_xvfb_root_0"))
    env.update(PATH=str(xb / "usr/bin") + ":" + env.get("PATH", ""),
               XKB_CONFIG_ROOT=str(xb / "usr/share/X11/xkb"),
               OMTRACKVLA_GLX_LIB_DIRS="/usr/lib/x86_64-linux-gnu:/usr/lib64:/usr/local/nvidia/lib64:"
               + str(xb / "usr/lib/x86_64-linux-gnu") + ":" + str(xb / "lib/x86_64-linux-gnu"))
    ready = dest / "wa_ready.json"
    port = args.port_base + lane
    wa = dict(role="wa", cwd=str(S), log=str(dest / "wa.log"),
              env=dict(env, PYTHONPATH=str(S)),
              argv=[str(R / "probe_env/bin/python"), "-u", "-m", "wa.wm.eval_server",
                    "--root", str(R), "--encoder-weight", ENCODER, "--wla-source",
                    str(R / "dependencies/wla_v1"), "--wla-checkpoint", WLA_CHECKPOINT,
                    "--checkpoint", CHECKPOINT_PATH, "--ready", str(ready),
                    "--mode", "mixed", "--noise-mode", "zero"])
    ln_env = dict(env, PYTHONPATH=str(L / "src") + ":" + str(S), PYTHONSAFEPATH="1",
                  WA_CUDA_MOUNT_COMPAT="1",
                  LD_LIBRARY_PATH=":".join(ln_libraries) + ":" + inherited.get("LD_LIBRARY_PATH", ""))
    teacher = dict(role="lightnav", cwd=str(B), log=str(dest / "teacher.log"), env=ln_env,
                   argv=[str(E / "lightnav/bin/python"), "-P", str(S / "wa/tools/lightnav_server_compat.py"),
                         "--task", "tracking", "--model_path", str(L / "checkpoints/LightNav-0"),
                         "--backend", "vllm_local", "--gpu_memory_utilization", "0.45",
                         "--max_batch_size", "1", "--host", "127.0.0.1", "--port", str(port),
                         "--ready_file", str(dest / "teacher.ready")])
    worker_env = dict(env, WA_DEVELOPMENT="0" if args.formal else "1",
        WA_DIAG_CONTROLLER="learned_yaw_guard_v1", WA_SEMANTIC_PLY_FIX="mp3d_semantic_ply_v1",
        WA_STUDENT_EVAL="evaluation_set_adaptation_v1", WA_EVAL_MODE="mixed", WA_EVAL_CHECKPOINT_SHA=CHECKPOINT_SHA,
        WA_EVAL_CHECKPOINT_STEP="59716", WA_INIT_REPAIR_PLAN=plan["repair_plan"]["path"],
        WA_INIT_REPAIR_PLAN_SHA=REPAIR_SHA,
        PYTHONPATH=":".join(map(str, [B / "artifacts/lightnav_client_runtime",
             B / "artifacts/official_runtime", B / "torch_overlay", B, B / "habitat-lab", S, W])),
        OMTRACKVLA_XVFB_DISPLAY_NUM=str(args.display_base + lane),
        OMTRACKVLA_HAB_SIM_GLX_ROOT=str(B), OMTRACKVLA_XVFB_LOG=str(dest / "xvfb.log"))
    argv = [str(B / "scripts/runtime/run_xvfb.sh"), str(E / "habitat/bin/python"),
            "-u", "-m", "wa.wm.failure_state_collect", "--plan", args.plan,
            "--plan-sha", args.plan_sha, "--ready", str(ready), "--teacher-url",
            f"ws://127.0.0.1:{port}", "--output", str(dest / "collection"), "--shard", str(lane)]
    if not args.formal:
        argv += ["--development-key", args.development_key]
    worker = dict(role="worker", cwd=str(B), log=str(dest / "worker.log"), env=worker_env, argv=argv)
    return [wa, teacher, worker]


def command_record(spec):
    return dict(role=spec["role"], argv=spec["argv"], cwd=spec["cwd"], log=spec["log"],
                env={k: spec["env"][k] for k in ENV_RECORD_KEYS if k in spec["env"]})


def check_paths(plan, inherited):
    xb = Path(inherited.get("XVFB_ROOT", "/tmp/omtrackvla_xvfb_root_0"))
    files = [R / "probe_env/bin/python", E / "lightnav/bin/python", E / "habitat/bin/python",
             B / "scripts/runtime/run_xvfb.sh", xb / "usr/bin/Xvfb", xb / "usr/bin/xkbcomp",
             S / "wa/tools/lightnav_server_compat.py", W / "lightnav_transport_20260927/server_transport.py",
             Path(ENCODER), Path(WLA_CHECKPOINT), Path(CHECKPOINT_PATH),
             Path(plan["repair_plan"]["path"])]
    dirs = [L / "src", L / "checkpoints/LightNav-0", R / "dependencies/wla_v1",
            B / "artifacts/lightnav_client_runtime", B / "artifacts/official_runtime",
            B / "torch_overlay", B / "habitat-lab", xb / "usr/share/X11/xkb"]
    for path in files:
        if not path.is_file():
            raise ValueError("missing required file: " + str(path))
    for path in dirs:
        if not path.is_dir():
            raise ValueError("missing required directory: " + str(path))
    for path in files[:6]:
        if not os.access(path, os.X_OK):
            raise ValueError("nonexecutable runtime: " + str(path))
    libs = list(map(str, (E / "lightnav/lib").glob("python*/site-packages/nvidia/*/lib")))
    if not libs:
        raise ValueError("LightNav NVIDIA runtime libraries missing")
    return libs


def validate_wa_ready(ready):
    expected = dict(checkpoint=CHECKPOINT_PATH, checkpoint_sha256=CHECKPOINT_SHA, step=59716,
                    mode="mixed", noise_mode="zero", sampling_steps=4, seed="7+step",
                    text_used=False, world_predictor_inference=False)
    for key, value in expected.items():
        if type(ready.get(key)) is not type(value) or ready[key] != value:
            raise ValueError("wrong WA server contract: " + key)
    address = urlparse(ready.get("url", ""))
    if (address.scheme != "http" or address.hostname != "127.0.0.1"
            or address.username or address.password or not address.port
            or address.path not in ("", "/") or address.query or address.fragment):
        raise ValueError("WA server must publish its own localhost HTTP URL")


def check_completion(collection, expected_keys, plan_sha, lane, development):
    collection = Path(collection)
    name = "DEVELOPMENT_CHECK.json" if development else "COMPLETE.json"
    marker = json.loads((collection / name).read_text())
    expected = dict(experiment=EXPERIMENT, entries=len(expected_keys), keys=expected_keys,
                    expected=len(expected_keys), shard=lane, development=development,
                    training_released=False, no_success_rate=True,
                    plan_sha256=plan_sha, protocol_sha256=PROTOCOL_SHA)
    if (set(marker) != set(expected) or any(type(marker.get(k)) is not type(v)
            or marker[k] != v for k, v in expected.items())):
        raise ValueError("collector completion differs from exact planned lane")
    raw = (collection / "records.jsonl").read_text()
    if not raw.endswith("\n"):
        raise ValueError("partial records JSONL")
    rows = [json.loads(s) for s in raw.splitlines() if s]
    if [r.get("key") for r in rows] != expected_keys:
        raise ValueError("records differ from planned keys/order")
    for row in rows:
        if development:
            checks = dict(status="DEVELOPER_NONZERO_REPLAY_CHECK_ONLY", prefix_actions=24,
                          takeover_step=20, branch_total_actions=24, teacher_owned_actions=4,
                          no_success_rate=True, training_eligible=False)
            if any(type(row.get(k)) is not type(v) or row[k] != v for k, v in checks.items()):
                raise ValueError("wrong developer scope or nonzero interface proof")
        elif (row.get("task") != "stt" or row.get("plan_sha256") != plan_sha
              or row.get("protocol_sha256") != PROTOCOL_SHA
              or row.get("outcome") not in {"rerun_student_success_no_recovery_needed", "student_invalid_initialization",
                                            "repeated_teacher_recovery_candidate", "no_valid_teacher_recovery"}):
            raise ValueError("foreign or incomplete formal record")
    return marker


class OwnedProcess:
    def __init__(self, spec, journal):
        self.spec, self.journal = spec, journal
        with Path(spec["log"]).open("x") as log:
            self.process = subprocess.Popen(spec["argv"], env=spec["env"], cwd=spec["cwd"],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            self.event("spawn", pid=self.process.pid, process_group=self.process.pid,
                       command=command_record(spec))
        except BaseException:
            # A logging failure must not leave an unregistered GPU process alive.
            self.stop()
            raise

    def event(self, event, **fields):
        with self.journal.open("a") as stream:
            stream.write(json.dumps(dict(utc=now(), role=self.spec["role"], event=event, **fields),
                                    allow_nan=False) + "\n")

    def exit_code(self):
        # WNOWAIT retains the leader PID until its group is cleaned; Popen.poll()
        # would reap it and could leave an orphan Xvfb or permit PID reuse.
        p = self.process
        if p.returncode is not None:
            return p.returncode
        status = os.waitid(os.P_PID, p.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        if status is None:
            return None
        return status.si_status if status.si_code == os.CLD_EXITED else -status.si_status

    def live_group(self):
        """Read PID/state only; never inspect unrelated process commands or env."""
        group = self.process.pid
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                fields = (entry / "stat").read_text().rsplit(") ", 1)[1].split()
            except (FileNotFoundError, ProcessLookupError):
                continue
            if int(fields[2]) == group and int(fields[3]) == group and fields[0] != "Z":
                return True
        return False

    def stop(self):
        p = self.process
        # No call to poll()/wait() may reap the leader before group cleanup.
        if p.returncode is not None:
            raise RuntimeError("owned leader was prematurely reaped")
        if os.getpgid(p.pid) != p.pid:
            raise RuntimeError("refusing to signal an unexpected process group")
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(p.pid, sig)
            except ProcessLookupError:
                break
            deadline = time.monotonic() + 15
            while self.live_group() and time.monotonic() < deadline:
                time.sleep(0.1)
            if not self.live_group():
                break
        if self.live_group():
            raise RuntimeError("owned process group still has live members")
        code = p.wait(timeout=15)
        self.event("exit", pid=p.pid, returncode=code)


def wait_ready(wa, teacher, dest, cancelled):
    deadline = time.monotonic() + STARTUP_SECONDS
    while True:
        if cancelled.is_set():
            raise RuntimeError("another lane failed or launcher was interrupted")
        if wa.exit_code() is not None or teacher.exit_code() is not None:
            raise RuntimeError("model server exited during startup; retain logs")
        path = dest / "wa_ready.json"
        if path.exists() and (dest / "teacher.ready").exists():
            try:
                ready = json.loads(path.read_text())
            except json.JSONDecodeError:
                ready = None  # The existing server writes ready non-atomically.
            if ready is not None:
                validate_wa_ready(ready)
                return ready
        if time.monotonic() >= deadline:
            raise TimeoutError("600s model-server startup deadline")
        time.sleep(1)


def run_lane(args, plan, lane, device, out, inherited, libs, cancelled):
    dest = out / f"lane{lane}"
    dest.mkdir()
    entries = planned_entries(plan, args, lane)
    keys = [e["key"] for e in entries]
    timeout = len(entries) * MAX_RUNS_PER_ENTRY * SECONDS_PER_RUN if args.formal else DEVELOPER_SECONDS
    specs = make_commands(args, plan, lane, device, dest, inherited, libs)
    write_json(dest / "launch.json", dict(experiment=EXPERIMENT, plan_sha256=args.plan_sha,
        protocol_sha256=PROTOCOL_SHA, device=device, lane=lane, keys=keys,
        startup_timeout_seconds=STARTUP_SECONDS, worker_timeout_seconds=timeout,
        commands=[command_record(s) for s in specs], training_released=False, no_success_rate=True))
    children = []
    try:
        for spec in specs[:2]:
            children.append(OwnedProcess(spec, dest / "processes.jsonl"))
        wait_ready(*children, dest, cancelled)
        worker = OwnedProcess(specs[2], dest / "processes.jsonl")
        children.append(worker)
        deadline = time.monotonic() + timeout
        while worker.exit_code() is None:
            if cancelled.is_set():
                raise RuntimeError("another lane failed or launcher was interrupted")
            if any(p.exit_code() is not None for p in children[:2]):
                raise RuntimeError("model server exited during collection")
            if time.monotonic() >= deadline:
                raise TimeoutError("bounded collection worker timeout")
            time.sleep(1)
        worker.event("worker_return", pid=worker.process.pid, returncode=worker.exit_code())
        if worker.exit_code():
            raise RuntimeError("collector failed; preserve original logs and partial outputs")
        marker = check_completion(dest / "collection", keys, args.plan_sha, lane, not args.formal)
        write_json(dest / "lane_outcome.json", dict(status="PROCESS_AND_COMPLETION_VERIFIED",
                   completion=marker, training_released=False, no_success_rate=True))
        return marker
    except BaseException as error:
        cancelled.set()
        write_json(dest / "ERROR.json", dict(error_type=type(error).__name__, message=str(error)[:2000],
                   training_released=False, no_success_rate=True))
        raise
    finally:
        errors = []
        for child in reversed(children):
            try:
                child.stop()
            except BaseException as error:
                errors.append(dict(pid=child.process.pid, error=type(error).__name__ + ": " + str(error)))
        if errors:
            cancelled.set()
            write_json(dest / "cleanup_errors.json", errors)
            raise RuntimeError("owned subprocess cleanup incomplete; see cleanup_errors.json")


def main(argv=None):
    args = arguments(argv)
    inherited = dict(os.environ)
    cuda_count = None
    if args.formal:
        if not inherited.get("MD_AK_JOB_ID") or not inherited.get("MD_AK_TASK_ID"):
            raise ValueError("formal launch requires scheduler identity before CUDA import")
        import torch
        cuda_count = torch.cuda.device_count()
    devices = validate_route(args, inherited, cuda_count)
    plan = load_plan(args.plan, args.plan_sha)
    for i in range(len(devices)):
        planned_entries(plan, args, i)
    protected = [S, R / "probe_env", R / "dependencies", Path(CHECKPOINT_PATH).parent,
                 Path(plan["sources"]["summary"]["path"]).parent]
    protected += [Path(e["artifact_root"]) for e in plan["entries"]]
    out = output_path(args.output, protected)
    libs = check_paths(plan, inherited)
    for i in range(len(devices)):
        check_address(args.port_base + i, args.display_base + i)
    commit = subprocess.run(["git", "-C", str(S), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(S), "status", "--porcelain"], check=True,
                           capture_output=True, text=True).stdout
    if args.formal and dirty:
        raise ValueError("formal source must be independently frozen and clean")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "launch.json", dict(experiment=EXPERIMENT, source=str(S), source_commit=commit,
        source_git_status=dirty, plan=args.plan, plan_sha256=args.plan_sha,
        protocol_sha256=PROTOCOL_SHA, formal=args.formal, devices=devices,
        expected_entries=126 if args.formal else 1, expected_lanes=len(devices),
        port_base=args.port_base, display_base=args.display_base, started_utc=now(),
        no_success_rate=True, training_released=False))
    cancelled = threading.Event()
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: cancelled.set())
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(devices))
    markers = []
    try:
        futures = [pool.submit(run_lane, args, plan, i, d, out, inherited, libs, cancelled)
                   for i, d in enumerate(devices)]
        for future in concurrent.futures.as_completed(futures):
            markers.append(future.result())
        expected = 126 if args.formal else 1
        keys = [k for marker in markers for k in marker["keys"]]
        wanted = [e["key"] for e in plan["entries"]] if args.formal else [args.development_key]
        if len(markers) != len(devices) or len(keys) != expected or len(set(keys)) != expected or set(keys) != set(wanted):
            raise ValueError("incomplete formal126 or developer1 coverage")
        write_json(out / ("COMPLETE.json" if args.formal else "DEVELOPMENT_CHECK.json"),
            dict(status="COLLECTION_PROCESSES_COMPLETE_NOT_TRAINING_RELEASE", experiment=EXPERIMENT,
                 plan_sha256=args.plan_sha, protocol_sha256=PROTOCOL_SHA, entries=len(keys),
                 lanes=len(markers), keys=wanted, formal=args.formal, training_released=False,
                 no_success_rate=True, finished_utc=now()))
        print("FAILURE_STATE_LAUNCH_COMPLETE", len(keys), flush=True)
    except BaseException as error:
        cancelled.set()
        write_json(out / "ERROR.json", dict(error_type=type(error).__name__, message=str(error)[:2000],
                   training_released=False, no_success_rate=True))
        raise
    finally:
        cancelled.set()
        pool.shutdown(wait=True, cancel_futures=True)
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
