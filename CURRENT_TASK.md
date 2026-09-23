# Current Task

Status: READY_TO_COMMIT

## Objective

Reduce the active repository to two maintained routes:

1. official OmTrackVLA 0.6B benchmark baseline;
2. modular person-following benchmark baseline.

## Acceptance criteria

- Root directory contains only project state, primary entry points, and core modules.
- End-to-end and Hybrid FLUX/online-RL implementations are absent from active code paths and preserved only under the September 2026 archive.
- Historical notes and compact log evidence are moved under `archive/`.
- `AGENTS.md`, `CURRENT_TASK.md`, `PROGRESS.md`, and `EXPERIMENTS.csv` exist and reflect current truth.
- Official and modular launchers resolve their new paths.
- Active Python files compile, shell launchers pass `bash -n`, and maintained unit tests pass or have explicit blockers.
- The generated `habitat_lab.egg-info/SOURCES.txt` change is removed.
- `artifacts/` remains on NAS but is ignored by Git.
- Final commit is pushed to `origin/gzy/local-mods`.

## Verification

- Root directory reduced to 25 files.
- Active Python compilation passed.
- Active shell syntax checks passed.
- Maintained unit suite passed: 105/105.
- No active imports from archived failed routes.

## Out of scope

- Formal training or full benchmark submission.
- Revival of archived failed routes.
- Direct merge into GitHub `main`.
