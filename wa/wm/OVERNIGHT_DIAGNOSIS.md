# WA mixed-mode diagnosis, 2026-09-29 night

## Authorization and stopping rule

User authorizes continued diagnosis, fixes, retraining and evaluation overnight without repeated confirmation. Main task is RGB + UWB polar goals, NO TEXT. Existing architecture (official JEPA training auxiliary, DINOv2, 64 MetaQueries, original ActionExpert) is preserved. World predictor is not online planning; do not mislabel it.

Do not extend beyond two epochs merely because loss decreases. Use fixed development episodes; operational interpretation of "most" is SR >=80% on each STT/DT/AT stratum (8 episodes each), report CR, failed initialization, and counts. This is a debugging gate, NOT product acceptance or a statistically strong generalization result. Confirm promising candidates on reserved scenes before any broader claim. Do not feed diagnostic trajectories into training or pick successful episodes.

## Verified facts

- Training59566/70423 SUCCEEDED, checkpoint22707 (one epoch), 8RTX4090. Offline ADE/FDE image0.2691/0.4697m; point0.2588/0.4515m; mixed0.2576/0.4487m. These are offline only.
- Image full evaluation59678/70540 remains untouched at frozen8ebbb30f; 8A800. Last23:29 snapshot1879/4215; STT1405 complete SR63.4164%, CR8.0427%,57 invalid initializations. DT474 partial SR23.6287%, CR15.1899%. Not mixed-mode results.
- CONFIRMED integration omission: previous eval_server forces mode0, polar=zeros, UWB masked; training randomly samples image/point/mixed. It cannot evaluate the requested mixed-input capability.
- CONFIRMED evaluation protocol difference, NOT yet causal: offline zero initial flow state versus per-step seeded Gaussian state in closed loop. Four-step decoding, checkpoint, controller and RGB preprocessing unchanged.
- Training UWB is ideal simulator instantaneous target-relative position, not real sensor measurements. It uses (target-robot) @ world_from_body_rotation then r=hypot(x,z), theta=atan2(-z,x). New simulation sensor matches exactly. No future positions, planned paths, text, world poses or semantic IDs enter model RPC.

## First complete diagnostic experiment

`wa/wm/mixed_diagnostic_plan_v1.json`, SHA256 e7d9b8597435f9e7b3208a7f74abe092e66c83fb94b3c5ad1a82e0e750cf61dd.

Outcome-blind SHA-based selection: 8 distinct development scenes, 8 different confirmation scenes; one episode/scene/task. Development24, confirmation24. Confirmation not run during initial diagnosis. Existing full image run has seen the validation split: do not claim untouched final test.

Four paired variants: image_random, mixed_random, image_zero, mixed_zero. Each runs all24 episodes. Complete diagnostic96 episodes, split into8 lanes on the available assigned GPUs. Identical episode seed7, weights, four Euler steps, FP32 action decoder and learned_target_guard_v3. Initial RGB hash must match across all conditions. Invalid first boxes retained as failures in ALL conditions to isolate UWB/noise effects; fallback-to-point can be tested as a separately labeled future fix.

Save trace per policy step (trajectory, learned target, polar packet, controller, final normalized action), sampled RGB every10 steps, per-episode outcome, complete summary. No stop-on-success or filtering. Ideal UWB has zero noise/delay; improvements cannot establish real-UWB robustness. Add explicitly noisy/delayed profile after ideal protocol validated, not silently.

## Decision sequence after this job

1. Check scheduler AND all worker logs/96 rows/hash pairs. Failure is not a policy score. Preserve failed output and fix only diagnosed issue.
2. Compare paired mode and noise effects on SR/CR, action jitter, range/bearing error, collision/lost traces. Missing UWB is proven; causal improvement requires results.
3. Inspect failure clips and initialization sensor camera alignment against training RGB bbox. Don't silently use later GT boxes. Inspect timestamps, SE2 signs/scales, target-guard heading effects. Controllers from other WLA branches are not identical baselines.
4. If mixed input repair suffices, independent confirmation before further training. Otherwise implement evidence-backed correction; full training (8GPU) up to2epochs, with per-epoch checkpoint and the same fixed diagnostics, not blind epoch extension. Preserve data split/model architecture. Add truthful globally averaged loss logging when touching trainer; historical logs rank0 last microbatch only.
5. If gate fails at2epochs, investigate/correct/retrain; do not merely add epochs. All runs independent reproducible snapshots and ledger entries. No infinite hyperparameter sweep. Changes in scientific direction require explicit notice.

## Monitoring

Heartbeat `wa`: every15minutes, current thread. Read current state + this file on each wake. No duplicate jobs. No changes to other WLA projects/automation. Record once-only morning summary after2026-09-30 08:00 Asia/Shanghai with real evidence and remaining limitations. No promise of success.
