# Pretrained latent world-model migration audit
Date: 2026-09-29. Status: PARTIAL / migration gates pending.

## Scope and decision
User explicitly rejects the self-designed ResNet18 baseline as the desired WA method.
Use an existing action-conditioned latent/video world-model implementation and its
pretrained assets. No text-generating VLM in the online policy. Do not stop Job59519
until a suitable replacement is ready; this audit does not alter its frozen source.

First migration candidate: original DINO-WM. This is a candidate, NOT a demonstrated
dense-crowd tracking solution. JEPA-WM is a research comparison; V-JEPA2.1 small video
encoders remain a fallback requiring action-conditioned post-training.

## Audited official source snapshots
All are clean clones under ../upstream_audit on persistent Beijing NAS.
- DINO-WM: https://github.com/gaoyuezhou/dino_wm
  commit 0a9492fa12044b852ae9e001cc74604b79c8bb0c
- JEPA-WMs: https://github.com/facebookresearch/jepa-wms
  commit 13cf1d9c7e476f53c17714d2e0f1dc239a883ce0
- V-JEPA2: https://github.com/facebookresearch/vjepa2
  commit 204698b45b3712590f06245fbfba32d3be539812

## Code findings
DINO-WM conf/encoder/dino.yaml selects dinov2_vits14 patch tokens. conf/predictor/vit.yaml
selects a six-layer, 16-head predictor, MLP width2048. train.py uses 196 patches at
img_size224 (encoder input is resized to196); default latent width is384+10+10=404.
The unchanged predictor has20,122,600 parameters for3 history frames. This count
EXCLUDES the visual encoder, action/proprio embeddings, decoder and planning costs.
models/visual_world_model.py supports decoder=None; no generated pixels are required
for latent prediction. Its default training config nevertheless enables a decoder.
models/vit.py hardcodes attention masks onto CUDA, so device-safe buffer handling
needs an explicit compatibility patch before portable/DDP deployment.
models/dino.py performs an unpinned torch.hub download; pin the encoder source and
use an explicitly hashed local pretrained file before formal runs.
conf/planner/cem.yaml uses300 candidates and30 optimization iterations (horizon5).
Do not infer end-to-end200ms from backbone size or predictor-only throughput.
Original source LICENSE is MIT; pretrained assets/dependencies require separate review.
Published task dynamics checkpoints cover PointMaze/PushT/Wall, not this EVT domain.

JEPA-WM small configs use DINOv2 ViT-S/14 and a six-layer predictor with action
conditioning; decoded visualization is optional but supplied eval configs enable it.
LICENSE is CC-BY-NC4.0. pyproject.toml requires Python>=3.10,<3.11 and torchvision0.22;
do not install it into or mutate the active Python3.11/torch2.8 training environment.
hubconf.py provides pretrained world-model URLs; no candidate world-model checkpoint
has been downloaded or executed in this audit.

V-JEPA2.1 offers an80M video encoder, but the released V-JEPA2-AC checkpoint uses
ViT-g; configs/train/vitg16/droid-256px-8f.yaml uses a24-layer1024-width predictor.
The small encoder is NOT a released80M action-conditioned navigation world model.

## Executed developer diagnostic and data audit
Artifact: ../artifacts/upstream_audit_20260929.json; script: wa/tools/upstream_code_audit.py.
Unmodified DINO predictor forward/backward passed on a developer A800, random weights.
Perturbing future tokens changed earlier outputs by0; action-feature perturbation
changed visual outputs. This proves wiring/causality, NOT learned dynamics or quality.
Twelve deterministic train episodes (four per teacher) were sampled, not a full audit.
Raw actions.json includes normalized_action and teacher_predicted_waypoints separately.
observations.json includes robot poses and timestamps. Sample intervals are0.048-0.056s,
not nominal0.1s; one interval implies22.868m/s. Causes remain UNVERIFIED: inspect the
collector/control implementation, reset transitions and command scaling before reuse.
Never silently equate commanded actions, realized displacement and future waypoint labels.

## Gates before replacement training
Resolve action/time semantics and transition validity; preserve scene-disjoint splits.
Load official pretrained assets with hashes and test genuine latent rollouts on real frames.
Retain official model/loss initially; separately document target-crop/UWB/control adaptations.
Measure full encoder+prediction+planning cost; Thor/RDK latency and SR/CR remain UNVERIFIED.
Keep both managed8x4090 and external8xH100 launch/result-exchange lanes reproducible.
