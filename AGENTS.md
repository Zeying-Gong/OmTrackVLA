# Project rules

## Current authority
- User decision 2026-09-28: use WLA as the primary tracking route and stop all DA3 cluster work.
- DA3 is retired, including collection, training, evaluation and conditional DAgger retraining.
- Do not submit, restart or resume DA3 jobs without a new explicit instruction.
- Preserve failed runs, partial data, logs and checkpoints. Retirement does not authorize deletion.
- Earlier DA3 execution authorizations in archive/ are historical and superseded.
- Cancellation of Job58613 is explicitly authorized; verify actual terminal scheduler status.
- WLA Job58638 is separate and must not be stopped as part of DA3 cleanup.

- User clarification2026-09-28: WLA architecture may change; preserve the WLA research paradigm and use LightNav/USS as references.
- User authorized16selected pairs/32complete video rollouts; no validation videos or review annotations enter training.

## Scope and evidence
- Official OmTrackVLA and modular following are maintained reference baselines.
- WLA text+RGB and DA3 bbox+RGB are different interfaces; the comparison is not a backbone ablation.
- Bbox/UWB adaptation for WLA is not yet implemented.
- No validation trajectories or labels may enter training.
- Report SR/TR/CR with coverage, split, scene/episode IDs and protocol limits.
- Training completion and loss reduction are not evidence of following success.
- Historical failed end-to-end and FLUX/RL routes must not be restored silently.

## State and storage
- CURRENT_TASK.md is the sole active task, at most 80 lines.
- PROGRESS.md records current facts, at most 150 lines.
- EXPERIMENTS.csv preserves successful, failed and stopped experiments.
- Archive superseded state under archive/YYYY-MM/; do not use README as a task board.
- Datasets, weights, logs, videos and generated outputs stay on persistent NAS, outside Git.

## Development and Git
- Follow the user's cluster manuals and remote-first rules; check actual CLI help and live status.
- Do not bypass authentication failures, copy credentials or switch identities to stop jobs.
- Preserve existing uncommitted source modifications.
- Documentation cleanup commit/push is authorized; do not include unreviewed source or data.
- Use explicit staged paths. Check staged diff, file sizes, links and sensitive strings before push.
