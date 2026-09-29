# WLA-compatible world-action integration (PARTIAL)
Update2026-09-29: robot-domain training is now implemented and full jobs submitted.
See ROBOT_TRAINING.md and ../jobs/wm_robot_v1_run.json. The probe-only limitations
below describe the earlier integration milestone, not the new training entrypoint.

This is the user-selected third design, not the older `wa/model.py` ResNet policy.
No Qwen or text encoder is instantiated. JEPA-WM is primary; Meta's official
DINO-WM reproduction is the comparison. Existing baseline files remain unchanged.

## Implemented and checked

- Official pinned DINOv2-S/14 and PointMaze JEPA/DINO predictors, strict hash/key loading.
- Original WLA `MetaQueryTokens` (64 x 2560), 16-block `LayerwiseActionExpert`,
  and target geometry head restored strictly from WLA Job58346 step43203.
- Existing WLA output: seven `[x,y,sin(yaw),cos(yaw)]` points; four Euler steps.
  Existing action training formula: sample-weighted flow MSE + 0.5 geometry SmoothL1.
- UWB interface `[r_metres,theta_radians]`, robot planar forward=0 / left-positive.
  Internal `[r/10,sin(theta),cos(theta),age/0.5,valid]`; reject invalid usable data,
  mask missing/stale inputs before arithmetic. Default freshness0.5s is configurable
  in `polar_features`; this is an initial interface parameter, not sensor validation.
- Visual/image, polar/point, and mixed goal modes. Environmental RGB is always used.
  If no valid goal remains, raise explicitly; a caller must hold/stop or provide
  a separately specified remembered goal, not silently invent a target.
- Causal caller-owned frame windows; no hidden cross-episode memory. Goal crop is
  pooled to4x4 tokens. Valid time offsets end at0; future observations are rejected.
- Training-only QueryToWorld bridge modifies only the last dense frame's features;
  official256-patch layout is preserved. Future targets are detached frozen DINO
  features. Since the encoder is frozen, no EMA update is needed in this version.
- A800 real-image interface diagnostics verify nonzero WM gradients to MetaQuery,
  polar encoder and fusion; action gradients also reach MetaQuery. An exception hook
  verifies all three deployment modes do not execute the world predictor.

## Explicit adapters / changes, not claimed official components

`adapters.py` implements USS-inspired self-attention plus read/write/read fusion.
It is our interface adaptation, NOT copied official USS code or a new pretrained
backbone. Three fusion blocks produce three384-wide states. A documented monotonic
mapping, per-slot normalization and shared projection feed the16 existing WLA action
conditioning slots12..27 at width2560. Slots0..11 are unused compatibility placeholders.
These are NOT28 Qwen layers, nor are old Qwen feature semantics preserved for free.
The original MetaQuery parameters load unchanged, then project2560->384 for fusion.
All fusion, polar, projection and QueryToWorld adapter parameters require training.

Full combined counts are385,199,816 (JEPA) /388,374,350 (DINO), including the original
large WLA action expert and frozen encoder. Do not report the40M world-model count
as the size of the full policy. Latency on Thor/RDK is UNVERIFIED.

## Critical gates still open

The official checkpoint consumes stacked PointMaze action10 / proprio4. Robot controls
must not be silently reshaped into those semantics. `dynamics_loss` currently accepts
only the explicit `pointmaze_pretrained_probe_only` contract. Diagnostics use synthetic
actions/state/SE2 labels and simulated-UWB target labels with real EVT RGB; their losses
are NOT forecast accuracy or tracking quality. No optimizer update was performed.

Before formal training: resolve raw command/timebase anomalies, define robot action
and state adapters, audit temporal intervals and future labels, choose WM objective
weight, then validate the joint optimizer/DDP and target-condition learning. The
old baseline launcher does NOT train this integration. No new full job is submitted.
No LightNav history compression, visibility head, multi-camera fusion or active
search implementation is claimed in this first integration.

## Reproduce on managed A800 or external H100

Use an isolated environment with torch/torchvision/einops/timm/numpy/Pillow/safetensors
and the dependencies of the existing WLA source. Official source pins and checkpoint
hashes are enforced by `loaders.py`. Prepare the layout in `wa/PRETRAINED_WM_PROBE.md`.

```bash
python wa/tools/wm_integration_probe.py \
  --root /YOUR/NAS/WA_ROOT \
  --wla-source /YOUR/NAS/WLA_SOURCE \
  --wla-checkpoint /YOUR/NAS/step-0043203.pt \
  --encoder-weight /YOUR/NAS/dinov2_vits14_pretrain.pth \
  --cache /YOUR/NAS/wla_evt_se2_cache_20260925_v2 \
  --output /YOUR/NAS/results/wm_integration_h100.json
python -m unittest discover -s wa/tests -v
```

The diagnostic accepts paths instead of assuming access to the user's external H100.
The cache currently contains absolute raw episode paths: those must be accessible
at the same paths for this probe. Portability of relocated cache roots is pending.
User's WLA snapshot is a separate required dependency, not a public Git dependency;
this change does not redistribute it or its weights. Result JSON records every WLA
source-file hash and checkpoint hash, plus this integration's source hashes. Return
that JSON via GitHub for comparison. This is a one-GPU developer diagnostic, not an
8-H100 formal launch; that launch remains gated by the checks above.

## Evidence

`wa/results/wm_integration_v1.json` and `wm_integration_v2.json` record independent
developer checks. v1 tested flow-only gradients; v2 tests the preserved flow+geometry
formula and records adapter source hashes. Both preserve the original WLA/WM weights
on disk. Eleven unit/regression tests passed. Model effectiveness remains UNVERIFIED.

Separately, old Job59519/Task70376 finished naturally on8RTX4090. Its full heldout
image/point/mixed ADE is0.3895/0.3625/0.3682m across78958 windows per mode. SR/TR/CR
are null and closed_loop=false; it is a historical ResNet baseline, NOT this model.
