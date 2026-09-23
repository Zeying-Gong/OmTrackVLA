# Project Progress

Last updated: 2026-09-23
Status: COMPLETE

## Repository

- H100 development entry: `nas-h100`
- Host checked: `devpod-zeying-gong-cpu-785c989ff4-577fb`
- Repository: `/data/nas_ray/home/zeying.gong/algorithm/repos/OmTrackVLA`
- Branch: `gzy/local-mods`
- Pre-cleanup commit: `16733f1e24dcd292f44e607dd8044dd418d96c3c`
- Cleanup commit: `89b89ca` (pushed to `origin/gzy/local-mods`)
- GitHub `origin/gzy/local-mods` matched the pre-cleanup commit before this task.
- GitHub default `main`: `a667bf78b933898d5f1bc2a86d63a43e1edad341`; the working branch was 56 commits ahead.

## Maintained routes

1. Official OmTrackVLA 0.6B training, conversion, and EVT-Bench evaluation.
2. Modular person-following with oracle references, RGB/RGB-D perception, ReID, map/reactive control, batch evaluation, and summaries.

The experimental end-to-end policy and Hybrid FLUX/online-RL routes are classified as failed and moved out of active code. Their source, tests, notes, and compact evidence remain recoverable under `archive/2026-09/failed_routes/` and Git history.

## Established benchmark evidence

Official checkpoint `ckpt_0401_text`, full validation runs after the avatar/protocol fixes:

- STT: SR 77.15, TR 81.57, CR 4.56.
- DT: SR 41.07, TR 61.22, CR 11.46.
- AT: SR 57.37, TR 74.38, CR 7.76.

Oracle Modular V5 full or nearly full evaluation is preserved in `results/oracle_v5_summary.csv`. Validation results:

- STT: 1,405/1,405, SR 88.47, TR 83.19, CR 4.77.
- DT: 1,404/1,405, SR 85.83, TR 77.37, CR 4.77; one episode missing.
- AT: 1,405/1,405, SR 85.27, TR 81.88, CR 4.20.

These numbers are historical evidence, not newly reproduced on H100.

## H100 status

- Aliyun NAS mount and repository path are available.
- Existing artifacts show three fixed validation episodes were rendered for STT, DT, and AT.
- The current development pod is CPU-only.
- Prototype job YAMLs used incompatible or duplicate schemas; they are archived pending creation of one live-validated job configuration.
- Official checkpoint paths, complete scene assets, Python environment, Ray connectivity, and a maintained one-GPU benchmark smoke have not yet been jointly revalidated.
- No formal training or benchmark run is authorized by the current task.

## Current work

- Repository cleanup, local verification, commit, and GitHub push are complete.

## Verification

- Root directory reduced to 25 files.
- Active Python compilation passed.
- Active shell syntax checks passed.
- Maintained unit suite passed: 105/105.
- No active imports from archived failed routes.
- Cleanup commit `89b89ca` is present on GitHub.
