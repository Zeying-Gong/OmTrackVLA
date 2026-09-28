> Historical snapshot before DA3 retirement; instructions and RUNNING states below are superseded.

# OmTrackVLA 🤖 👀

**Visual Navigation & Following for Everyone.**

[](https://opensource.org/licenses/Apache-2.0) [](https://www.google.com/search?q=) [](https://www.google.com/search?q=) [](https://arxiv.org/abs/2509.12129)

**OmTrackVLA** is a fully open-source Vision-Language-Action (VLA) stack that turns **monocular video** and **natural-language instructions** into actionable, short-horizon waypoints.

This repository is dedicated to democratizing embodied AI. We have intentionally released our highly efficient **0.6B checkpoint** along with the **full training pipeline**.

### 🚀 Why OmTrackVLA?

  * **Fully Open Source:** We release the model weights, inference code, *and* the training stack—not just the inference wrapper.
  * **Accessible:** Designed to reproduce, fine-tune, and deploy with affordable compute .
  * **Multimodal Control:** Combines learned priors with visual input to guide real or simulated robots via simple text prompts.

> **Acknowledgment:** OmTrackVLA builds on the ideas introduced by the original [TrackVLA project](https://github.com/wsakobe/TrackVLA). Their partially-open release inspired this community-driven effort to keep the ecosystem open so researchers and developers can continue improving the stack together.

## Maintained project scope

This research branch keeps two benchmark baselines:

1. the official OmTrackVLA 0.6B training and EVT-Bench evaluation baseline;
2. the modular person-following baseline with oracle references, RGB/RGB-D perception, ReID, obstacle mapping, and modular control.

A user-authorized minimal DA3 route is also active as a `PARTIAL` experiment. It is separate from the unsuccessful older end-to-end and Hybrid FLUX/online-RL routes retained under `archive/2026-09/failed_routes/`.

Current truth is tracked in `CURRENT_TASK.md`, `PROGRESS.md`, and `EXPERIMENTS.csv`. Maintained usage notes are in:

- `docs/official_baseline.md`
- `docs/modular_baseline.md`
- `docs/h100.md`

### Minimal DA3 person-following route

The model consumes a persistent first-frame RGB plus target bbox, exactly 31 history frames plus the current RGB, and optional two-dimensional polar UWB `(range_m, bearing_rad)`. The prompt expands this to normalized range, `sin/cos` bearing, valid, and age. DA3-SMALL/L11 yields 256 tokens of width 768 per 224x224 frame. A 3x3 RoI produces nine identity tokens; target fusion produces four tokens per frame; the causal temporal sequence has 128 tokens. The public output is seven `(r, theta)` robot waypoints at 0.1-0.7 s. The controller adapter converts these to metric `(x, y)` and prepends an interface-only origin.

The current tiny-data checkpoint fits four windows from an on-disk simulator episode, but the UWB-disabled bbox-swap test is `UNVERIFIED`: changing the selected person did not meaningfully change the path. Treat the route as implementation- and optimization-verified, not identity- or benchmark-verified.

Inspect the real episode without constructing the model:

```bash
/opt/miniforge/envs/isaacray/bin/python scripts/da3/run_da3.py inspect
```

Run the unit checks:

```bash
PYTHONPATH="$PWD/habitat-lab:$PWD" \
  /opt/miniforge/envs/isaacray/bin/python -m unittest tests.test_da3_policy -v
```

The checked Aliyun Ray configurations are:

```bash
export MD_AK_CONFIG_PATH="$PWD/scripts/da3/md_ai_kit_h100_compat.yaml"
TOOL=/opt/miniforge/envs/isaacray/bin/md_ai_kit
$TOOL list --cluster aliyun_sh_h100 --limit 10
$TOOL submit scripts/da3/h100_smoke.yaml
$TOOL submit scripts/da3/h100_overfit.yaml
$TOOL submit scripts/da3/h100_qualitative.yaml
```

Despite the scheduler name `aliyun_sh_h100`, all three verified jobs reported an `NVIDIA L20Z` worker. Do not label their results as H100 results.

Run one offline inference report from the small overfit checkpoint:

```bash
PYTHONPATH="$PWD:/data/nas_ray/home/zeying.gong/algorithm/repos/Depth-Anything-3/src" \
  /opt/miniforge/envs/isaacray/bin/python scripts/da3/run_da3.py infer \
  --checkpoint artifacts/da3_minimal_polar_uwb_20260923/overfit/overfit_checkpoint.pt \
  --output-dir artifacts/da3_minimal_polar_uwb_20260923/infer_frame60 \
  --current-index 60 --no-use-uwb
```

The corrected overfit visualization is under `artifacts/da3_minimal_polar_uwb_20260923/overfit/visualization/`. Polar-UWB qualitative reports are pending regeneration; the older `da3_minimal_20260923/qualitative/` reports used the former external UWB API. Trajectories are not overlaid on RGB because this sample has no verified camera projection calibration.

---

## 📢 News & Updates

- 🔥 We have updated the model weights and refreshed the benchmark scores in the performance table. The latest checkpoint is available at **[omlab/OmTrackVLA-0.6B](https://huggingface.co/omlab/OmTrackVLA-0.6B)**. We will keep updating — **please keep watching this repository!**
- **[Coming Soon]** 📦 We will be releasing the full training data. Stay tuned!

---


## Demo In Action

The system processes video history and text instructions to predict future waypoints. Below are examples of the tracker in action:
<div align="center">
<img src="examples/ex1.gif" width="45%" alt="Tracked clip 1" />
<img src="examples/ex2.gif" width="45%" alt="Tracked clip 2" />
</div>


## 1. Requirements & Environment Setup

1. **Create the Conda env**
   ```bash
   conda create -n omtrack python=3.9 cmake=3.14.0
   conda activate omtrack
   ```
2. **Install Habitat-Sim 0.3.1 (with Bullet)**
   ```bash
   conda install habitat-sim==0.3.1 withbullet -c conda-forge -c aihabitat
   ```
3. **Clone this repository**
   ```bash
   git clone https://github.com/om-ai-lab/OmTrackVLA.git
   cd OmTrackVLA
   ```
4. **Install Habitat-Lab**
   ```bash
   pip install -e habitat-lab
   ```
---

## 2. Dataset Preparation (TrackVLA parity)

The simulator assets and humanoid avatars follow the exact instructions from the original TrackVLA github release. Please accept the respective licenses before downloading.

### 2.1 HM3D + MP3D Scenes
   - Request access through the Habitat links used by TrackVLA: [HM3D](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d) and [MP3D](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#matterport3d-mp3d-dataset).
   - After the providers approve your request, download the archives locally and extract them under `data/scene_datasets` so the final layout matches:
     ```
     data/
       scene_datasets/
         hm3d/
           train/...
           val/...
           minival/...
         mp3d/
           1LXtFkjw3qL/...
           ...
     ```
   - If you stage the downloads elsewhere first, move them with a command such as:
     ```bash
     mv /path/to/hm3d data/scene_datasets/
     mv /path/to/mp3d data/scene_datasets/
     ```
     Ensure the directory names stay lowercase (`hm3d`, `mp3d`) so Habitat configs resolve correctly.

### 2.2 Humanoid Avatars
   - From the repository root run:
     ```bash
     python download_humanoid_data.py
     ```
     This script mirrors TrackVLA’s original helper and should populate `data/humanoids`.
   - If the script fails (e.g., quota errors), manually download `humanoids.zip` from [Google Drive](https://drive.google.com/file/d/1aE_wyvPqvOuVmF8px2vTO3trr70DKf1l/view), then extract it with:
     ```bash
     unzip humanoids.zip -d data/
     ```
     Double-check that `data/humanoids/**` now contains the FBX meshes required by Habitat.

### 2.3 Verification
   - Run `ls data/scene_datasets/{hm3d,mp3d}` to confirm both corpora are present.
   - Re-run any Habitat episodes (e.g., `run_eval.py`) to ensure the simulator can resolve the scene assets and humanoid avatars without missing-file errors.

Keeping this layout in sync with TrackVLA’s published instructions avoids surprises when sharing configs or checkpoints across both projects.

---

## 3. Data Processing Pipeline

### 3.1 Build Dataset (`make_tracking_data.py`)
This script integrates egocentric base velocities to emit future indicator curves, waypoints, and sliding-window JSONL shards. Each episode requires the `<stem>.mp4` RGB capture alongside `<stem>_info.json` pose metadata.

**Quick Start (Sample Data)**

The repo ships with a tiny rollout snapshot under `sim_data/sample`. Generate a ready-to-train dataset with:

```bash
python make_tracking_data.py \
    --input_root sim_data/sample \
    --output_root data/sample
```

This mirrors the `sim_data` tree under `data/sample/frames`, writes JSONL shards to `data/sample/jsonl`, and primes `data/sample/dataset.json` if `--out_file` is set.

**Full Usage**

```bash
python make_tracking_data.py \
    --input_root sim_data/sample \
    --output_root data/sample \
    --history 31 --horizon 8 --dt 0.1 \
    --only_success \
    --instruction "Follow the target person without collision."
```

Episodes are replicated under `frames/seed_xxx/...`, with corresponding `jsonl/seed_xxx/scene/episode.jsonl` files storing history frames (`images`), the current frame (`current`), the instruction, future waypoints (`trajectory`), and velocity commands (`actions`). Use `--out_file` to emit a monolithic dataset JSON in addition to per-episode shards.

### 3.2 Precompute Vision Tokens (`precache_frames.py`)
`train.py` stays I/O bound when every frame already has DINO/SiGLIP embeddings cached. `precache_frames.py` precomputes both fine and coarse tokens so the trainer can mmap `.pt` tensors instead of re-encoding on the fly.

```bash
python precache_frames.py \
    --data_root data/sample \
    --cache_root data/sample/vision_cache \
    --batch_size 8 \
    --image_size 384
```

Tokens are stored as `.half()` tensors to save disk, and any missing coarse tokens are backfilled via pooling when only the fine representation exists. Layout auto-detection handles nested `frames/seed_xxx/...` structures.

---

## 4. Training

`train.py` hosts the Qwen-based planner with masked waypoint losses. Point it at the dataset JSONL directory and the cached vision tokens to kick off optimization.

```bash
python train.py \
    --train_json data/sample/jsonl \
    --cache_root data/sample/vision_cache \
    --out_dir ckpt_sample \
    --epochs 2 \
    --batch_size 8 \
    --n_waypoints 8 \
    --history 31 \
    --lr 2e-5 \
    --mixed_precision \
    --save_trajectories
```

- **Checkpointing:** Every 100 steps a `.pt` file lands in `out_dir`; set `--max_ckpts` to cap retention.
- **Visualization:** Enable `--save_trajectories` to collect `.npz` bundles plus overlay images under `vis/`.
- **Resume:** Pass `--resume` or `--resume_ckpt /path/to/model_epochXX_stepXXXXXX.pt` to pick up training midstream.
- **Scaling knobs:** Adjust `--train_json` to target any directory of JSONL shards, tweak `--history` / `--n_waypoints` for different temporal windows, and flip `--distributed` when launching via `torchrun`.

---

## 5. Evaluation

`eval.sh` fans Habitat-based rollouts across configurable chunks and expects `CKPT`, `HF_MODEL_ID`, or `HF_MODEL_DIR` to define which weights to load. Outputs land under `sim_data/eval/<task>` alongside per-episode videos and metrics. Use `bash kill_eval.sh` to cleanly terminate all spawned jobs.

### 5.1 Using the Pre-trained 0.6B Model
- **Option A — Auto-download**
  ```bash
  HF_MODEL_ID=omlab/OmTrackVLA-0.6B bash eval.sh
  ```
- **Option B — Manual download**
  ```bash
  huggingface-cli download omlab/OmTrackVLA-0.6B --local-dir open_trackvla_hf
  HF_MODEL_DIR=$(pwd)/open_trackvla_hf bash eval.sh
  ```

### 5.2 Evaluating Custom Checkpoints

Point `CKPT` at any artifact produced by `train.py`:

```bash
CKPT=/path/to/model_epoch05_step009000.pt bash eval.sh
```

Tune `CHUNKS`, `NUM_PARALLEL`, or the Habitat config inside `eval.sh` to rebalance throughput vs. coverage.

## 📊 Performance (EVT-Bench)

| Methods      | STT (SR↑ / TR↑ / CR↓) | DT (SR↑ / TR↑ / CR↓) | AT (SR↑ / TR↑ / CR↓) |
|--------------|------------------------|----------------------|----------------------|
| IBVS†        | 42.9 / 56.2 / 3.75     | 10.6 / 28.4 / 6.14   | 15.2 / 39.5 / 4.90   |
| PoliFormer†  | 4.67 / 15.5 / 40.1     | 2.62 / 13.2 / 44.5   | 3.04 / 15.4 / 41.5   |
| EVT          | 24.4 / 39.1 / 42.5     | 3.23 / 11.2 / 47.9   | 17.4 / 21.1 / 45.6   |
| EVT‡         | 32.5 / 49.9 / 40.5     | 15.7 / 35.7 / 53.3   | 18.3 / 21.0 / 44.9   |
| Uni-NaVid (Vicuna-7B)    | 25.7 / 39.5 / 41.9     | 11.3 / 27.4 / 43.5   | 8.26 / 28.6 / 43.7   |
| TrackVLA (Vicuna-7B)     | 85.1 / 78.6 / 1.65     | 57.6 / 63.2 / 5.80   | 50.2 / 63.7 / 17.1   |
| Previous ckpt (0.6B)  | 64.8 / 84.4 / 5.00     | 33.6 / 66.3 / 8.84   | 39.6 / 76.7 / 6.38   |
| Our latest ckpt (0.6B)  | 81.41 / 82.77 / 5.13     | 41.54 / 58.80 / 11.31   | 60.04 / 73.89 / 7.60   |


† Uses GroundingDINO as the open-vocabulary detector. ‡ Uses SoM + GPT-4o as the vision stack (see the TrackVLA paper Table 2).

**Transparency note:** With the updated 0.6B checkpoint, OmTrackVLA now surpasses the 7B TrackVLA baseline on success rate (SR↑) in the AT setting (60.04 vs 50.2) while remaining competitive on STT (81.41 vs 85.1). Tracking rate (TR↑) leads on STT and AT, with a marginal gap on DT. Collision rate (CR↓) is lower than TrackVLA on AT but remains higher on STT and DT — reducing collisions in denser scenarios without sacrificing tracking performance is an active area of improvement. We continue to prioritize this lightweight release to keep reproduction and fine-tuning accessible.



## 📚 Resources & References
- Baseline checkpoint: [omlab/OmTrackVLA-0.6B](https://huggingface.co/omlab/OmTrackVLA-0.6B)
- TrackVLA: Embodied Visual Tracking in the Wild [arXiv:2505.23189](https://arxiv.org/abs/2505.23189)
- Embodied Navigation Foundation Model [arXiv:2509.12129](https://arxiv.org/abs/2509.12129)

## Citation

If you find OmTrackVLA useful in your research or applications, please cite it using the following BibTeX:

```bibtex
@misc{omtrackvla2025,
  author       = {Kyusong Lee, Heting Ying and Tiancheng Zhao},
  title        = {OmTrackVLA: Open-Source Visual Language Action Model for Visual Navigation and Following},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished = {https://github.com/om-ai-lab/OmTrackVLA}
}
```

**Happy tracking!** Contributions and issue reports are welcome via pull requests.
