# OmTrackVLA Project Rules

## Scope

The active repository has exactly two research routes:

1. the official OmTrackVLA 0.6B training and EVT-Bench evaluation baseline;
2. the modular person-following baseline, including oracle references, RGB/RGB-D perception, ReID, obstacle mapping, and modular control.

The archived end-to-end policy and Hybrid FLUX/online-RL routes are failed research directions. Do not import, execute, extend, or silently restore code under `archive/2026-09/failed_routes/` unless the user explicitly reopens one of those directions.

## Research objective

The immediate objective is reproducible STT/DT/AT benchmark evaluation. Real-robot transfer is a secondary objective, so active interfaces should retain deployable RGB/RGB-D, target-image, coordinate, waypoint, and `[forward, lateral, yaw]` abstractions where they do not change the benchmark protocol.

## Benchmark discipline

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

The H100-side repository is `/data/nas_ray/home/zeying.gong/algorithm/repos/OmTrackVLA` on the Aliyun NAS. The H100 development pod is CPU-only; GPU work must use the Aliyun Ray submission path. Do not run long jobs in the development shell.

Before an H100 job, verify the live submission CLI, Ray connectivity, environment, dataset, checkpoint, output path, and a one-GPU smoke configuration. Formal benchmark or training submission requires explicit user authorization.

## Git and verification

- Preserve unrelated user changes and NAS artifacts.
- Keep active entry points free of imports from `archive/`.
- Run Python compilation, shell syntax checks, and relevant unit tests after structural changes.
- Commit only reproducible source, compact state, and small benchmark summaries.
- Do not commit credentials, local absolute secrets, raw logs, model weights, or large generated outputs.
