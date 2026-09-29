# WA-Mobile progress
2026-09-29 16:45+08:00: both formal Pods Pending; no optimizer steps yet.
K8s FailedScheduling: insufficient CPU/full8GPU placement. Platform requests56CPU/472GiB.
Keep queued; no duplicate jobs or platform changes. This is scheduling wait, not training failure.
2026-09-29 16:40/16:41 +08:00: JEPA Job59566 Task70423 and DINO Job59568 Task70425 SUBMITTED.
Each8RTX4090 on baidu_bj_4090; sourceab3ed46d; full22707 updates plus three-mode heldout.
wa/jobs/wm_robot_v1_run.json records config hashes and NAS paths; optimizer evidence pending.
2026-09-29: READY_TO_SUBMIT official JEPA/DINO full robot-domain experiments.
Action recording order and control saturation verified; actual-dt command adapters added.
Full v2 audit retains726631 train/73368 heldout; abnormal transitions excluded and recorded.
Both2A800 real-data optimizer/NCCL tests PASS; peak8.67/8.23GiB;13 regression tests PASS.
Full recipe and external8H100 lane: wa/wm/ROBOT_TRAINING.md; managed two8RTX4090 tasks.
Historical probe-only gates below superseded; no job ID claimed until submission evidence.
2026-09-29: user-approved WLA-compatible route implemented as PARTIAL integration.
Restored original64 MetaQuery/ActionExpert/target head; official JEPA/DINO strict-loaded.
Added UWB polar validation and USS-inspired fusion; retained original SE2 flow+geometry.
Both A800 joint-gradient probes and three-mode world-off inference passed;11 tests passed.
See wa/wm/README.md and wa/results/wm_integration_v2.json; NOT trained/effective yet.
Robot action/time adapter pending; no new formal job. Total model385M/388M, not40M.
Old Job59519 naturally SUCCEEDED; checkpoint and78958-window three-mode offline metrics verified.
2026-09-29: user selected JEPA-WM mainline; DINO-WM comparison, academic usage.
Both official PointMaze pretrained models strict-loaded and ran on A800; 39.7M/42.9M.
Artifacts: wa/results/pretrained_wm_probe_v{1,2}.json; wa/PRETRAINED_WM_PROBE.md.
PARTIAL: synthetic controls only; no tracking-quality claim. Job59519 still RUNNING.
2026-09-29: Job 59125 / Task 69896 FAILED before training: source path
2026-09-29: user requires existing pretrained latent/video world models, not self-built WA.
Audited pinned official DINO-WM/JEPA-WM/V-JEPA2 clones; report wa/WORLD_MODEL_AUDIT.md.
Official DINO predictor20,122,600 parameters; causal/gradient developer tests passed.
12 raw episodes sampled: action records exist; actual dt0.048-0.056s; speed anomaly unresolved.
Replacement readiness PARTIAL; original Job59519 remains untouched (last seen19400/24375).
Old baseline records below are historical, not endorsement of its architecture as mainline.
not visible on Baoding worker. No optimizer steps or trained checkpoint.
Independent checkout on A800 persistent NAS; branch `wa`.
Added ResNet18 prompt/temporal baseline, data adapter and two DDP launchers.
Seven tests and real-data CPU/Gloo checks passed; GPU/Thor remain UNVERIFIED.
2026-09-29 14:12 +08:00: retry Job59519 Task70376 on baidu_bj_4090, 8 GPUs.
TRAINING verified: environment.json confirms eight RTX4090 GPUs and clean source4a73fbd6.
First step 1/24375 loss0.49227056; config170e71e9; completion/effectiveness UNVERIFIED.
A800 retry configuration was superseded before submission; no duplicate job.
Effective batch/lr/objectives unchanged; per-rank RNG and GPU numerics may differ.
External 8xH100 uses the same tools; no access to that machine is assumed.
All 10,660 episode first-frame identities/timebases passed; checkpoint inference passed.
NCCL 2-GPU and pretrained 1-GPU developer checks passed; all seven tests passed.
Official encoder SHA256: f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec.
Run provenance and persistent output: wa/jobs/wa_full_v1_run.json.
Next: worker startup and complete train/heldout artifacts. No effectiveness claim.
