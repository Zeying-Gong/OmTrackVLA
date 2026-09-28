# WA v0.1: executable baseline
Status: CPU_REAL_BATCH_AND_DDP_PASS; formal training NOT RUN.
This is a small engineering baseline, not the final backbone selection or a
reproduction of USS. ResNet18 is frozen; the identity/geometry fusion, causal
Transformer, seven-point XY head and training-only dynamics head are trainable.
There is no language model, pixel video generator, safety controller or closed-loop
benchmark integration yet. No Thor latency or target-reidentification claim is made.

## Environment
Tested: Python environment with torch 2.8.0+cu128, torchvision 0.23.0+cu128.
Install matching torch/torchvision CUDA wheels on your host, then the remaining
requirements in wa/requirements.txt. No package installation occurs at runtime.
Run from the root: python -m unittest discover -s wa/tests -v.

## Dataset and portable paths
Use the existing audited SE2 cache wla_evt_se2_cache_20260925_v2, NOT the target cache.
It contains train/heldout_{pose,history,episode}.npy, both *_episodes.json, and complete.json.
The raw rollout directories referenced by *_episodes.json must also be present.
Required per rollout: metadata.json, observations.json, and its RGB frame files.
Do not upload this dataset or a checkpoint to Git.
To relocate raw rollouts, pass BOTH --source-prefix OLD_DATASET_ROOT and
--data-root NEW_DATASET_ROOT; cached relative suffixes are preserved.
Cache array hashes and scene split disjointness are checked on launch.
Raw image integrity across hosts must additionally be established during transfer.
The dataset is not distributed by this repository.

The verified existing protocol uses four causal RGB frames near t-1.5,-1,-0.5,0,
persistent RGB first-frame bbox, and seven robot XY points at 0.1,...,0.7 seconds.
Y is left, X is forward, units metres. Yaw is present in the source but omitted in
this first baseline loss/output. RGB is always available in all three modes.
The point input is simulator-derived exact range/bearing, explicitly simulated_uwb;
it is not a measured UWB stream. Noise, delays and outages are future experiments.
Missing/invalid target identity fails rather than silently selecting a different target.
Before a full run, audit all episodes with the command below. Do not silently filter failures.

```bash
python -m wa.audit --cache "$WA_CACHE" --output /persistent/wa-data-audit.json
```

## Pretrained encoder
Formal runner requires an explicit local torchvision ResNet18 state_dict, matching
ResNet18_Weights.IMAGENET1K_V1. It never downloads weights during training.
Official weight URL: https://download.pytorch.org/models/resnet18-f37072fd.pth
Acquire it on an authorized machine and copy it alongside your persistent models.
The full weight file SHA256 is recorded. Random initialization is restricted to
--diagnostic and is NOT evidence of model quality.

## External 8xH100 lane
Set WA_CACHE, WA_OUTPUT (new run directory), WA_ENCODER_WEIGHTS, optionally WA_PYTHON.
The output parent must already exist. Then:

```bash
bash wa/scripts/train_external_h100.sh --epochs 1 --batch-size 4 --mode all --world-weight 0.1
# For relocated raw datasets, append --source-prefix OLD --data-root NEW.
python wa/tools/exchange.py pack --run "$WA_OUTPUT" --output "${WA_OUTPUT}.zip"
python wa/tools/exchange.py verify --bundle "${WA_OUTPUT}.zip"
```

The launcher uses 8 DDP ranks and verifies all eight visible devices report H100.
Effective batch is ranks * per-rank batch. Training drops an incomplete batch;
heldout evaluation is unpadded and counts each window exactly once for each mode.
Every run evaluates Image, Point, Mixed offline ADE/FDE. SR/TR/CR remain null.

## Managed lane
The same runner is wa/scripts/train_managed.sh, with WA_GPUS set to the allocated
GPU count (default 8). Invoke it INSIDE the authorized scheduler task, never for a
long run in a development shell. K8s/Ray platform wrappers require live resource,
image and path verification for the specific approved experiment; none are submitted
by this change. Identical config/data/effective batch is required for comparison.

## Results and limitations
Results: config.json, environment.json, metrics.json, checkpoint.pt. The exchange
tool exports only the JSON files. Upload the ZIP as a GitHub Release asset and share
its URL. Attach the .log separately for failures; checkpoints stay on persistent storage.
A failed process may leave incomplete output; do not present it as a successful run.
There is no resume implementation yet; use a new output directory for each attempt.
CPU diagnostic ADE is a plumbing check using random frozen encoder weights.
Target-switch correctness, crowd safety, GPU DDP/NCCL, edge latency and full dataset
full-image integrity remain unverified. The original NAS first-frame prompt/timebase
audit passed all 10,660 episodes. Re-run it after moving the dataset.

## Offline inference
Run python -m wa.infer --checkpoint RUN/checkpoint.pt --request request.json
--output prediction.json. The output must not already exist.
Request JSON fields:
- mode: image, point, or mixed.
- rgb: list of causal frame paths, oldest first.
- times_s: relative timestamps, ordered, ending at zero.
- target_image and bbox [x1,y1,x2,y2]: persistent initialization target for image/mixed.
- point [range_m,bearing_rad,age_s] and point_valid: for point/mixed.
The output is seven XY points at 0.1 through 0.7 seconds, robot x-forward/y-left.
One-frame startup is supported. Real robot execution requires a separate controller
and collision checking; this CLI produces only offline predictions.
