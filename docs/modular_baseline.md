# Modular Person-Following Baseline

This is one of the repository's two maintained routes. It separates target perception from navigation/control so that perception loss, control loss, privileged upper bounds, and deployable RGB/RGB-D behavior can be evaluated independently.

## Active components

- `oracle_modular_follow.py`: shared target observation, oracle perception, oracle navmesh control, and map/reactive control implementation.
- `oracle_modular_follow_v6.py`: Version 6 policy defaults.
- `oracle_modular_batch.py`: sharded dataset evaluation.
- `rgb_person_perception.py`: detector, association, target initialization, and ReID.
- `modular_obstacle_map.py`: persistent RGB-D obstacle map and collision-aware local waypoint selection.
- `scripts/modular/eval_oracle_modular_8gpu.sh`: maintained batch launcher.
- `tools/modular/monitor_oracle_progress.py`: progress and completion checks.
- `tools/modular/summarize_oracle_modular.py`: SR/TR/CR aggregation.

## Evaluation matrix

- Oracle perception + oracle navmesh control: privileged reference.
- RGB/RGB-D perception + oracle control: isolates perception loss.
- Oracle or coordinate perception + modular control: isolates control loss.
- RGB/RGB-D perception + modular control: deployable modular baseline.

Pure visual, pure coordinate, and hybrid target inputs must remain explicitly labeled. A hybrid run that uses ground-truth coordinates is an upper bound, not a real-robot result.

## Example evaluation

```bash
cd /data/nas_ray/home/zeying.gong/algorithm/repos/OmTrackVLA
PYTHON_BIN=/path/to/python \
TASKS=stt SPLITS=val \
PERCEPTION=oracle CONTROLLER=oracle-navmesh TARGET_MODE=hybrid \
RENDER_BACKEND=egl \
bash scripts/modular/eval_oracle_modular_8gpu.sh
```

Historical controller and perception studies are under `archive/2026-07/` and `archive/2026-08/`. Structured V5 results are preserved in `results/oracle_v5_summary.csv`.
