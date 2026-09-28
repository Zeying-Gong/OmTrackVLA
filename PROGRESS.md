# WA-Mobile progress
2026-09-28: CPU_REAL_BATCH_AND_DDP_PASS; full training not started.
Independent checkout on A800 persistent NAS; branch `wa`.
Added ResNet18 prompt/temporal baseline, data adapter and two DDP launchers.
Seven tests and real-data CPU/Gloo checks passed; GPU/Thor remain UNVERIFIED.
A800/H100/4090 scheduler jobs have not been submitted by this change.
External 8xH100 uses the same tools; no access to that machine is assumed.
All 10,660 episode first-frame identities/timebases passed; checkpoint inference passed.
Next: pretrained weights and full-run approval; see wa/RUNBOOK.md.
