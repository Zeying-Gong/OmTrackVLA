# Formal experiment WA_FULL_V1
Authorized by user on 2026-09-28 to train on accessible A800 or H100.
Selected: baidu_bd_a800, 8 A800 GPUs, K8s; no cross-NAS copy.
Image: x5-builder:cuda12.8-isaac5.0.0-v2.test1.
Runtime: existing wla_evt_torch28 (torch 2.8.0+cu128, torchvision 0.23.0+cu128).

Frozen ResNet18 ImageNet1K V1 encoder; trainable target-conditioned temporal
fusion, seven-point XY head and action-conditioned latent dynamics head.
No warm-start policy checkpoint; explicit official pretrained vision weights.
Data: wla_evt_se2_cache_20260925_v2.
Cache metadata SHA256: eb5a52d3478bab1391d12a990dcdc57ce2c77cef787109d29f7f286be797f901.
Train: 780025 windows; 9657 episodes. Heldout: 78958 windows; 1003 episodes.
All source scenes are disjoint. Full initial prompt/timebase audit passed.
One complete epoch, with standard DDP/drop_last discarding 25 tail windows;
780000 training windows consumed, 24375 optimizer steps.
Heldout is unpadded: all 78958 windows evaluated once in each of three modes.
Per-rank batch 4; effective batch 32; AdamW lr 3e-4; FP32; seed 42.
Modes sampled image/point/mixed; world-loss coefficient .1; workers/rank 2.
Point condition is exact simulated UWB; not comparable to RGB-only EVT.
Action target: robot XY metres at 0.1,...,0.7s; no yaw prediction in this baseline.

Acceptance: process completes, finite losses and mode-wise ADE/FDE, complete
checkpoint and provenance, correct heldout counts, verified JSON result ZIP.
SR/TR/CR and Thor latency remain unavailable; effectiveness needs closed loop
and controlled baselines. No numerical performance threshold has been established.
Worker verifies real GPU model and NAS paths at the start of the full task.
Source is pinned by a detached NAS worktree source_full_v1.
Scheduler IDs/actual source commit/output are recorded after submission.
External eight-H100 recipe: same training flags via train_external_h100.sh.
