# Phase 1 minimal closed loop

Phase 1 trains two isolated streams with one shared visual encoder. It does not treat ordinary ego motion as an expert person-following action.

## Data boundary

- Geometry/dynamics uses InternData-N1 RGB pairs and pose-derived realized SE(2). Natural-language task fields are never loaded.
- Identity uses SAGE3D and TpT. Only the first visible RGB+bbox is converted to an in-memory reference crop; later RGB frames carry bbox/visibility as labels only.
- The adapters never create target crops, caches, manifests, or other files under a source root.
- SAGE3D episodes must be in the root index, contain `_ACCEPTED`, and have `quality.status=accepted`.
- TpT uses video PTS only for chronological identity clips. Its ODOM/GT timing remains blocked from physical-motion supervision.

The fixed split manifest is generated from `group/scene`, `run`, and `sequence` units:

```bash
PY=/data/nfs/share/gzy/miniconda3/envs/omtrackvla/bin/python
$PY scripts/build_phase1_manifest.py --output configs/manifests/phase1_v1.json
```

This command enumerates split units and required metadata. It does not decode or scan all media.

## Baseline objectives

The first executable baseline uses a shared ResNet-18 encoder without downloaded weights:

- identity visibility BCE and normalized bbox regression from an initialization crop plus a GRU over subsequent RGB history;
- inverse dynamics prediction of realized `(x_forward, y_left, yaw)`;
- motion-conditioned forward feature prediction;
- action-free single-step next-feature prediction.

This baseline exists to make the Phase 1 interfaces and benchmark executable. DA3 remains the preferred geometry teacher candidate; `scripts/probe_da3_geometry.py` verifies its depth, confidence, pose, intrinsics, intermediate features, w2c convention, and SE(2) error before its outputs are admitted as pseudo labels.

## Development smoke test

Use one GPU and a small number of split units first:

```bash
CUDA_VISIBLE_DEVICES=0 $PY -m omtrackvla.training.train \
  --phase 1 \
  --config configs/phases/phase1_pretrain.yaml \
  --data-root data \
  --output-dir outputs/training/phase1_smoke/phase_1 \
  --run-manifest outputs/training/phase1_smoke/run_manifest.json \
  --max-units-per-dataset 1 \
  --max-steps 2 \
  --num-workers 0
```

The formal launcher remains:

```bash
bash scripts/run_pipeline_8xh100.sh --phase 1 --run-id phase1_v1
```

Evaluation writes B1-ID, B1-GEO, and a fixed low-data linear B1-PROBE comparison. Rendering writes `phase1_viz.mp4`, where later boxes are explicitly marked `PRED` or `GT` and are never fed back into the model. Gate failure preserves all artifacts and stops the pipeline.
