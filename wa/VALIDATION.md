# Development evidence — 2026-09-28
A800 development host, persistent NAS. No scheduler tasks were submitted.
- Seven CPU tests passed (1.359 s): exchange, tamper/path checks, causal prefix,
  invalid coordinate fill invariance, point-mode template invariance, gradients.
- Bash syntax checks passed for both launchers.
- Real dataset: wla_evt_se2_cache_20260925_v2; complete.json SHA256
  eb5a52d3478bab1391d12a990dcdc57ce2c77cef787109d29f7f286be797f901.
- cpu_realbatch_v1: two train/two heldout windows, random frozen encoder,
  one epoch, batch 1. Forward/backward, checkpoint and per-mode report completed.
- cpu_ddp_v1: two CPU/Gloo ranks, four train/four heldout windows, random encoder.
  DDP allreduce and unpadded heldout counting completed; four heldout rows/mode.
Artifacts on NAS:
  /data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/
No reported diagnostic ADE establishes effectiveness. GPU/NCCL and eight H100
devices have NOT been tested.
Full episode prompt audit PASS: train 9,657 episodes / 780,025 windows;
heldout 1,003 episodes / 78,958 windows; zero prompt/timebase failures.
This reads each initial RGB crop and all timestamps, not every RGB frame hash.
Artifact: prompt_audit_v1.json. Checkpoint reload and real first-frame inference
also passed (infer_request_v1.json, infer_result_v1.json).
The real DDP result ZIP was packed and verified successfully.
