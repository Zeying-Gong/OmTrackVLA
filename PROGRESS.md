# WA-Mobile progress
2026-09-30 00:26+08 controller hypothesis59737/70599 RUNNING2A800; source912db155 configf0fc1e2e. Same24 mixed_zero: measuredUWB range guard and far-heading blend>=.5; hypothesis not proven; weights unchanged.8unit tests PASS.
Paired96 Job59726 SUCCEEDED00:17; initialRGB all equal; image_random8 mixed_random14 image_zero7 mixed_zero16 successes of24. Mixed_zero STT7/8 DT6/8 AT3/8; CR1/8 eachtask. Gate unmet.
Failure traces: AT lost with UWB range>5m bearing~.42rad yet predictedyaw~0 and original guard heading_blend0 above3m. Near collision repeated same scene with range<.7m despite retreat, not solved by heading hypothesis. No realUWB validation or final generalization claim.
2026-09-30 00:07+08 Job59726 RUNNING; random variants24each complete, zero variants pending. Same24 development episodes image_random8success vs mixed_random14success. NOT full96 result.
Task SR image_random STT50% DT12.5% AT37.5%; mixed_random75%/62.5%/37.5%. Mixed CR12.5% eachtask. Gate NOT met. Keep job running; no retraining chosen before zero comparison.
2011 mixed_random logged policy frames include actual ideal_simulated_uwb. Predicted target rangeMAE0.08327m bearingMAE1.428deg; mean normalized action jump0.12898. Geometry accuracy alone does not establish safe tracking.
2026-09-29 23:49+08 paired96 diagnostic59726/70588 RUNNING2A800; sourceefa41778 config8ebe6409; job_59726/task_70588/wa_mixed_diagnostic_v1. No duplicate evaluation.
4A800 diagnostic59720/70582 STOPPED before launch after FailedScheduling GPU/CPU; complete96 scope unchanged. Prior4GPU logs retained.
2026-09-29 23:46+08 mixed paired96 diagnostic59720/70582 SUBMITTED4A800; sourceefa41778 configcca17305; output job_59720/task_70582/wa_mixed_diagnostic_v1.
Confirmed prior closed-loop omitted UWB despite3-mode training; fixed polar RPC. Offlinezero/closedlooprandom difference isolated in4paired variants. NO causal improvement claimed.
Development24 vs confirmation24 in disjoint scenes and no training-scene overlap; fixed plan e7d9b859.10unit tests + actual checkpoint mixed2calls PASS; simulator imports PASS from actual BENCH cwd.
Heartbeat wa every15min created; user permits overnight fixes/retraining/diagnostics. Provisional most-success gate>=80% each task; do not blindly exceed2epochs.
Training59566/70423 SUCCEEDED22707steps; offline metrics exist. Image ADE/FDE .2691/.4697m; mixed .2576/.4487m; no closedloop claim.
Fullimage59678 remains RUNNING unchanged. STT complete63.42%SR/8.04%CR;57invalid starts; DT stillpartial at lastquery. Historical lines below are chronological records, not current state.
2026-09-29T21:13+08 full4215episode eval Job59678/Task70540 RUNNING scheduler;8A800 allocated without prolonged queue.
Source8ebbb30f/config77447239; all8x3 dataset audits PASS. User allows fewer parallel GPUs if8GPU queues. Effectiveness UNVERIFIED.
2026-09-29 user superseded small evaluation with full4215episodes on8GPUs. Job59674/70536 verified STOPPED.
Small job had begun real A800 simulation and rendered initial frames; no completed metrics. Artifacts retained.
Full evaluator preserves image-only policy/controller;8static data shards audited before submitting one full allocation.
2026-09-29 final checkpoint verified22707steps/1epoch; Job59566 RUNNING and offline metrics not yet produced.
Training logged-sample averages(first/middle/last100): loss0.1861/0.1270/0.1211; world0.3166/0.2122/0.2028. NOT validation loss.
TensorBoard installed in separate environment;909 sampled training points served on devpod-a800:6006 via localhost16006 SSH. No continuation job submitted.
2026-09-29 authentication resolved; private ModelScope uploader and H100 verifier running. First1GiB shard uploaded.
Backup plan150archives/1051170files/163832726558sourcebytes. Complete backup NOT yet verified.
JEPA59566 continues:4875/22707steps. See wa/wm/MODELSCOPE_BACKUP.md for paths, process and verification protocol.
2026-09-29 user authorized private ModelScope backup and H100 download alongside current4090 training.
JEPA59566 verified2550/22707steps. Training continues unchanged; no duplicate experiment authorized.
Backup BLOCKED on source authentication: Baidu ms-hub missing API token; H100 logged in. No data uploaded yet.
2026-09-29 user requested minimize migration and list exact files/sizes: stopped data/checkpoint SSH streams.
Partial target files retained, not verified or usable. 4090 JEPA training unaffected.
Public JEPA/DINOv2 weights and code should be downloaded at destination; custom EVT data cannot be substituted by PointMaze.
2026-09-29: JEPA59566 now RUNNING;775/22707 real optimizer steps verified, elapsed447.9s; peak8.73GiB.
H100 migration still incomplete; seek direction before interrupting newly active training for a slower migration.
2026-09-29 H100 CPU-only preflight:13 regression tests PASS in isolated torch2.7/cu128 environment.
Transfer snapshot:538MiB dataset /310MiB original checkpoints; incomplete. GPU/model-load checks pending.
2026-09-29: user authorized8H100 JEPA only. H100 NAS migration actively in progress, not training.
Copied source/dependencies/audit indices; streaming10660episodes/1050874files and original checkpoints.
Direct inter-devpod SSH denied; local stream relay has low throughput. No credentials copied.
H100 isolated torch2.7/cu128+timm1.0.30 environment created; final data/model checks pending.
Ray preinitialized NCCL guard added;13 regression tests PASS on developer. DINO59568 STOPPED.
2026-09-29: user selected JEPA-WM only; DINO Job59568 stop request accepted.
H100 live platform reports26 free GPUs; expected WA dataset cache absent on Alibaba NAS.
Retain JEPA Job59566 while checking migration; no H100 submission and no duplicate run.
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
