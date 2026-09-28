> Historical snapshot before DA3 retirement; instructions and RUNNING states below are superseded.

# OmTrackVLA Project Rules

## Scope

The repository keeps two maintained benchmark baselines:

1. the official OmTrackVLA 0.6B training and EVT-Bench evaluation baseline;
2. the modular person-following baseline, including oracle references, RGB/RGB-D perception, ReID, obstacle mapping, and modular control.

The user has additionally authorized one narrowly scoped experimental route under `da3_policy/`: DA3-SMALL/L11 visual tokens, a persistent initialization-frame RoI identity, optional polar UWB `(range_m, bearing_rad)` prompt, causal 32-frame fusion, and a seven-point robot trajectory decoder. Treat it as `PARTIAL` until identity conditioning is demonstrated on target-switch data; it is not yet a strong baseline.

The archived older end-to-end policy and Hybrid FLUX/online-RL routes remain failed research directions. Do not import, execute, extend, or silently restore code under `archive/2026-09/failed_routes/`; the new DA3 route must not depend on them.

## Research objective

The immediate objective is reproducible STT/DT/AT benchmark evaluation. Real-robot transfer is a secondary objective, so active interfaces should retain deployable RGB/RGB-D, target-image, coordinate, waypoint, and `[forward, lateral, yaw]` abstractions where they do not change the benchmark protocol.

## Benchmark discipline

- Latest user authorization (2026-09-24): expand collection to ALL AT/DT/STT train tasks (7,257 each) with the three teachers;65,313 planned attempts, qualified totals reported separately. Use one shared scene partition across tasks; task-scoped identity keys. Expand SAGE beyond the69-sequence audit and assess TPT RGB+robot-odometry supervision; do not mix unverified scenes/times or annotated quickview frames into policy training.

- User clarification (2026-09-24): the ten-episode limit applies ONLY to quick downstream comparisons, never to training data collection. Full STT train-split collection (7,257 source tasks) for LightNav, official 0.6B and privileged oracle, plus formal DA3 training, is now authorized. Use scene-disjoint internal holdout; never train on benchmark validation. Collection order is LightNav first, then official, then oracle. Immutable data releases may start training during continuing collection only after the declared substantial-data gate (1,000 qualified train episodes, 100,000 windows, 100 scenes), not the tiny ten-episode audit.

- User decision (2026-09-24): start each downstream comparison with exactly the same 10 EVT-Bench episodes for DA3 and official OmTrackVLA 0.6B. Freeze scene/episode IDs, seed, simulator and metric configuration before either run. Review paired metrics and rollouts before any full evaluation; do not automatically launch all 1,405 episodes.
- Downloading the public official 0.6B checkpoint and required encoders to NAS, and running this paired 10-episode comparison, are authorized. This bounded evaluation takes precedence over older no-small-evaluation wording. Keep historical full-set results separate.

- Latest conditional post-training authorization:
- User authorization (2026-09-26): after mixed Job58085 finishes and its final checkpoint is verified, evaluate on EVT-Bench validation. Retain the existing matched-ten DA3 vs official baseline first; freeze IDs/seeds/controller/timebase/metrics, inspect failures before expanding evaluation. If performance is poor, expert-guided retraining on EVT TRAIN ONLY (DAgger-style on-policy state collection and dataset aggregation) is authorized, followed by the same paired validation. Never aggregate validation rollouts/labels into training. Keep internal train/holdout scenes separate, expert identity/provenance and simulator time explicit. Compare SR/TR/CR and rollouts, not training loss alone; report exploratory validation reuse. Do not alter the running frozen training snapshot, restore failed FLUX/RL routes, or submit conditional retraining before evaluation evidence and a documented correction plan.
- Keep official and modular runs separate.
- Record Success Rate, Tracking Rate, and Collision Rate; record completion/finish rate when available.
- Do not compare partial runs with full 1,405-episode validation runs as if they were equivalent.
- Do not change datasets, avatar selection, detector thresholds, metrics, episode filtering, or retry semantics without documenting the comparability impact.
- A smoke test, decreasing loss, or successful job submission is not a benchmark result.
- Add every material run to `EXPERIMENTS.csv`, including failed and partial runs.

## Project state

- `CURRENT_TASK.md` is the only active task and must stay within 80 lines.
- `PROGRESS.md` records current truth and must stay within 150 lines.
- Historical notes belong under `archive/YYYY-MM/`, not in the repository root.
- Generated logs, checkpoints, videos, datasets, and `artifacts/` remain on persistent NAS and are not committed.

## Development environment

The Aliyun repository is `/data/nas_ray/home/zeying.gong/algorithm/repos/OmTrackVLA` on its NAS. The development pod is CPU-only; GPU work must use the Aliyun Ray submission path. Record the GPU model reported by the worker instead of inferring it from the scheduler or namespace name. Do not run long jobs in the development shell.

Before any job, verify the live submission CLI, connectivity, environment, dataset, checkpoint and output paths. Follow the user's latest global rules: do not submit preliminary smoke jobs; submit the complete authorized workload after static/development checks. Baidu K8s shell tasks support at most eight GPUs. The user authorized the 4090 execution snapshot and eight-GPU offline evaluation; this is not authorization to claim or launch a full EVT-Bench run.

## Git and verification

- Preserve unrelated user changes and NAS artifacts.
- Keep active entry points free of imports from `archive/`.
- Run Python compilation, shell syntax checks, and relevant unit tests after structural changes.
- Commit only reproducible source, compact state, and small benchmark summaries.
- Do not commit credentials, local absolute secrets, raw logs, model weights, or large generated outputs.
