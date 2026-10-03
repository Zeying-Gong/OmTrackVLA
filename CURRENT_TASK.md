# WA current task — full mixed closed-loop validation
Updated: 2026-10-03T18:45+08:00. Status: READY_TO_SUBMIT / FULL_RESULTS_UNVERIFIED.

## User-authorized scope and acceptance
- Full existing validation STT1405 + DT1405 + AT1405 =4215; no subset/smoke, no retraining, no new LightNav full evaluation.
- Fixed manifest WLA-EVT-20260925/evt_full_20260926/manifest.json SHA a1f515534153ccaa720f787f01ea23b9097c174fe25020f756c7d7947eedf98f.
- Complete all4215 unique task/key pairs across8fixed shards, retain invalid initializations, report per-task SR/TR/CR and counts; videos/traces for every episode.
- Seed7, same original simulator physics/success criterion. Validation includes previous development/confirmation; not untouched test.
- WA RGB+initialBBox+ideal simulated polar UWB noise0/delay0; no text. HumanCollision is target-person distance ever<0.5m, not general obstacle contact.
- Do not infer product90% SR, real-UWB robustness, edge latency or general superiority from small-set results.

## Frozen method and submission
- Same60502/71381 checkpoint step45900 cumulative2epochs, SHA20cc84b3f231ad4056e16b91c52c84cb14e18d886a80324b5f27ffd773be5331.
- learned_yaw_guard_v1 retains learned yaw and original translation guard; JEPA/MetaQuery/ActionExpert preserved, world predictor training-only.
- Frozen source_full_mixed_learned_yaw_v2 commit d16c5a9efb73a6712306d44a0f2b0a543dd4b4d7; independent new entry, old image-only full entry unchanged.
- Config wa/jobs/full_mixed_learned_yaw_a800_v2.yaml:2A800, timeout86400;8shards multiplexed over2allocated GPUs.
- A800 aggregate3free checked; prefer2GPU eval to8GPU queue. No crossNAS. Do not interfere with WLA60857/60766.
- Developer14CPU tests and8x3 real dataset audits PASS; dependency manifests and checkpoint SHA PASS.
- First developer dataset check used incorrect cwd and hit trained_agent import shadowing; rerun with exact BENCH worker cwd passed. No cluster failure or smoke submitted.
- Job/Task: retry NOT_SUBMITTED; first60883/71806 FAILED startup0episodes (unset CUDA_VISIBLE_DEVICES incorrectly defaulted8); preserved logs/source. Fixed actual CUDA count,18CPU testsPASS. No current full mixed result.
- Output follows /data/nas_ray/project/md-ak/users/zeying.gong/job_<JOB>/task_<TASK>/wa_full_mixed_learned_yaw_v2.
- Next: submit once, check real worker/model readiness and episode counts, report every20min, final completeness/metric/video audit then pause monitor.

## Prior evidence retained
- Confirmation WA60770/71649 20/24 vsLightNav60771/71650 17/24; collision2vs3; invalid0; initialRGB24matched.
- STT6vs7, DT7vs5, AT7vs5. Development60767 17/24 equalLN. Not all-task superiority.
- Full prior phase state archived archive/2026-10/CURRENT_TASK_before_full_mixed_20261003.md.
- Existing confirmation HTML18793, development18792, TB6006 (not full4215 results).
- Forward: ssh -N -L 18793:127.0.0.1:18793 -L 16006:127.0.0.1:6006 devpod-a800
