# WA-Mobile progress
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
