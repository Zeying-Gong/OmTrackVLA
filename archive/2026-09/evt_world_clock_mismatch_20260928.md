> Archived DA3 evidence. All earlier continuation/retraining instructions and live status claims are superseded by the 2026-09-28 retirement decision; Job58613 is STOPPED. See docs/tracking_route_decision.md.

# Real EVT action-clock mismatch, 2026-09-28

Status: DIAGNOSED; controller v2 not yet repaired; no valid benchmark metrics.

## Evidence

- Training58085 SUCCEEDED; final679878steps2epochs4ranks; strict CPU and first eval GPU model load passed.
- Full paired Job58558/Task69305 FAILED01:39:56Asia during task initialization: missing cwd humanoid_infos.json. No complete rollout, no official rollout; all logs/configs retained.
- New scene_cwd contains the exact existing metadata SHA2564d230a3f39b80dae0641e7179660f045a49e06de065838dad3f51c50d8c18659 plus data/frozen-habitat symlinks. Original335sourcefiles unchanged.
- Bounded developer check on nas-a800 (CUDA_VISIBLE_DEVICES3) initializes real scene2n8kARJN3HM episode119 and resets successfully. Not a scheduler smoke and not a benchmark result.
- First zero-action world clock:0.0->0.048. get_physics_time_step() reports0.008;ac_freq_ratio4;ctrl_freq40;task.physics_target_sps60.
- v2 strict assertion correctly FAILED: World clock delta0.048 differs from action duration0.032. This is an interface bug, not evidence for model failure or insufficient epochs.
- Diagnostic wrapper around original internal_step preserves call arguments and behavior. Four calls recorded:
  - requested dt=-1;world0.016->0.024;getter0.008
  - requested dt=-1;world0.024->0.032;getter0.008
  - requested dt=-1;world0.032->0.040;getter0.008
  - requested dt=-1;world0.040->0.048;getter0.008
- Root cause in frozen habitat/core/embodied_task.py:347: task.step invokes sim.step_physics(1.0/_physics_target_sps) BEFORE sim.step(None). Then registered evt_bench/rearrange_sim_v2.py:1018-1019 executes ac_freq_ratio internal_step(-1). Earlier v2 accounted only for latter.
- No unexplained extra internal_step calls; task-layer advance contributes observed0.016s.
- The developer run also emitted a humanoid_rearrange_controller.py:114 divide-by-zero warning; no semantic/protocol changes made to suppress it.

## Next work

- Measure several complete action transitions and save timestamp/parameter provenance on NAS; distinguish configured durations from realized physics quantization.
- Correct controller interface using calibrated actual world-action duration, with runtime parameter matching and strict observed delta verification on subsequent actions. Do not hardcode0.048 from nominal40Hz or silence assertions.
- Keep existing7waypoint0.1-0.7s labels, model/loss, simulator behavior, fixed10IDs/seed and official controller unchanged.
- Preserve frozen v1 code and failed v2 protocol. Create new frozen evaluation source version for changed controller; extend CPU tests including task-level physics advance and mismatch rejection, then bounded scene check.
- Prepared scene_v3 YAML currently canonical-only, NOT deployed/submitted and NOT sufficient alone: it fixes cwd but still points at the unrepaired controller snapshot. Do not submit it before controller fix/new snapshot/admission.
- Repeat entire authorized paired20rollouts after successful checks, preserving Job58558. Conditional TRAIN-only DAgger still waits for real paired metrics.
