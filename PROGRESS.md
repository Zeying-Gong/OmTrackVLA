# WA-Mobile progress
2026-09-29: Job 59125 / Task 69896 FAILED before training: source path
not visible on Baoding worker. No optimizer steps or trained checkpoint.
Independent checkout on A800 persistent NAS; branch `wa`.
Added ResNet18 prompt/temporal baseline, data adapter and two DDP launchers.
Seven tests and real-data CPU/Gloo checks passed; GPU/Thor remain UNVERIFIED.
Preparing Beijing A800 retry: 1 GPU, batch32, same full epoch and source 4a73fbd6.
Effective batch/lr/objectives unchanged; per-rank RNG and GPU numerics may differ.
External 8xH100 uses the same tools; no access to that machine is assumed.
All 10,660 episode first-frame identities/timebases passed; checkpoint inference passed.
NCCL 2-GPU and pretrained 1-GPU developer checks passed; all seven tests passed.
Official encoder SHA256: f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec.
Run provenance and persistent output: wa/jobs/wa_full_v1_run.json.
Next: worker startup and complete train/heldout artifacts. No effectiveness claim.
