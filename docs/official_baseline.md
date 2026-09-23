# Official OmTrackVLA Baseline

This is one of the repository's two maintained routes. It preserves the released 0.6B waypoint model, data preparation, training, checkpoint conversion, and EVT-Bench evaluation for STT, DT, and AT.

## Primary entry points

- `make_tracking_data.py`: convert rollout video and pose metadata into training shards.
- `precache_frames.py`: cache DINO/SiGLIP visual tokens.
- `train.py`: train or resume the official waypoint model.
- `tools/official/convert_ckpt_to_hf.py`: convert a native checkpoint to the Hugging Face wrapper.
- `run_eval.py` and `eval.sh`: upstream-compatible evaluation entry points.
- `scripts/official/eval_official_glx.sh`: maintained multi-worker evaluator.
- `scripts/runtime/run_egl.sh`, `run_glx.sh`, and `run_xvfb.sh`: rendering wrappers.

## Protocol-sensitive fixes

- Reset every episode's temporal frame/token state in `trained_agent.py`.
- Switch humanoid avatars per episode rather than only once per worker.
- Keep RGB, panoptic, third-person RGB, then depth sensor registration order under Habitat-Sim 0.3.1 + EGL; the active setup is `spot_agent_simplified_rgbd.yaml`.
- Treat missing planner actions as errors rather than silently falling back to an unrelated controller.
- Keep video saving and step verbosity configurable for full benchmark throughput.

## Example evaluation

```bash
cd /data/nas_ray/home/zeying.gong/algorithm/repos/OmTrackVLA
TASK=stt \
PYTHON_BIN=/path/to/python \
HF_MODEL_DIR=/path/to/checkpoint \
RUN_WRAPPER="$PWD/scripts/runtime/run_egl.sh" \
bash scripts/official/eval_official_glx.sh
```

Do not treat this as an H100-ready command until the worker environment, scene assets, checkpoint, and Ray job configuration have been verified together.

Historical full-run summaries are indexed in `EXPERIMENTS.csv`; detailed dated notes are under `archive/2026-07/` and `archive/2026-08/`.
