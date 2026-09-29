# JEPA / DINO robot-domain full training v1

User authorized full training on2026-09-29. This supersedes the probe-only gate in
`wa/wm/README.md`; old PointMaze probe remains available and explicitly named.

## Resolved action contract

Inspected collector `TeacherRecorder.observe/record_action` and evaluation loop:
observation t is recorded, command t is recorded, then `env.step` executes it.
Match commands by `sim_step`, not list position. BaseVelNonCylinderAction clips
each command to[-1,1], then scales forward/left/yaw by15/10/6.28. Its kinematic
integrator uses1/ctrl_freq=0.025s. Recorded physics clock advances approximately
0.048..0.056s, so nominal0.1s must NOT replace observation timestamps.

The new trainable action interface is `[clipped forward,left,yaw,actual_dt/0.1]`
projected4->64->10 into the unchanged official world predictor. A separate4->64->4
interface encodes the preceding command/interval as observable control history.
This is explicit robot-domain adaptation, NOT a claim that commands mean the same
thing as PointMaze actions. Existing official interior and embedding weights initialize
training; the new action/state interface, goal fusion and query bridge are learned.

WM receives four consecutive observed frames ending at current t, corresponding
commands, and predicts the actual next frame. Policy still uses the original four
causal history frames and original seven0.1..0.7s SE2 trajectory labels. Commands,
next frame and privileged geometry labels enter supervision only, not deployment
inputs. UWB is explicitly simulated from robot/target geometry; no real-UWB claim.

Original raw23m/s outliers were real horizontal jumps, not merely vertical settling.
Their exact reset/NavMesh origin is not established. Instead of relabeling them,
exclude any window whose full history-to-label interval crosses displacement above
hypot(15,10)*0.025+0.02m, yaw above6.28*0.025+0.01rad, or dt outside[0.02,0.15]s.
Require four previous command records. This conservative gate is documented;
raw failures remain intact. Preserve the original scene-disjoint partition.

Full audit v2: train726631/780025; heldout73368/78958; zero parsing errors.
V1 incorrectly rejected pre-saturation commands>1; it is retained as superseded.
V2 hashes metadata/observations/actions per episode and records every excluded row
implicitly via the complement of its valid index. Dataset loader rechecks source hashes.
Audit SHA256:3138364dea80543fc83be476e1af8191a4eca3711a9b353591311fc8daf5c2d3.

## Matched full recipe

- Two independent tasks,8RTX4090 each; not one16-GPU K8s task.
- One full epoch; batch2/rank, accumulation2, effective batch32; seed42.
- Original trained WLA step43203 initial heads, SHA256
  0b8f036fd282474d8c9efe9e34e1f4dc56b9282c44295bcda1f16edcf48460e1.
- Frozen pretrained DINOv2 encoder. FP32 trainable modules; BF16 frozen encoding.
- Loss=flow MSE+0.5 target geometry SmoothL1+0.1 normalized future-feature SmoothL1.
- AdamW weight_decay0.01; clip norm1;100-update warmup then cosine to10%.
- LR: original action expert2.5e-6; MetaQuery5e-6; target head5e-5;
  official world modules1e-5; newly initialized interfaces1e-4.
- DDP shuffled training drops the incomplete per-rank batch. Heldout is unpadded:
  every retained row evaluated once in each image/point/mixed mode.
- Four-step flow inference from fixed zero noise for reproducible offline metrics.
- Save model+optimizer every2000 steps and at completion. Frozen encoder weights
  stay external with pinned hash. No overwriting of earlier run directories.
- Report ADE/FDE/yaw error; closed-loop SR/CR and edge latency remain UNVERIFIED.

Expected optimizer updates:22707; the distributed sampler drops7 final training rows.
Model-specific randomness/dropout differs; matching seed does not imply identical
random draws or fully identical newly initialized bridge tensors across variants.

## Preflight evidence

13 regression tests PASS. Each variant ran two actual optimizer updates on2A800
developer GPUs through NCCL, gradient accumulation and all three heldout modes.
JEPA peak allocated8.67GiB; DINO8.23GiB. DINO also tested worker_count2.
These are developer diagnostics, not reduced formal cluster tasks or quality claims.
Full outputs are on NAS under artifacts/{jepa,dino}_robot_ddp_v1.

Source and WLA dependency are frozen separately before submission. The isolated
probe_env imports the already verified torch2.8 runtime read-only, plus timm1.0.30.
No shared runtime was upgraded. Same Beijing NAS and known cuda12.8 worker image.
A800 had4 free GPUs; H100 was reachable but lacked this project's assets;4090 ready.

## External8H100 lane and returning results

Use the same `wa/scripts/train_world.sh`, with all WA_* paths supplied for that
machine (see required variables in the script). Set WA_GPUS=8, WA_WORLD_KIND=jepa
or dino, and a fresh WA_OUTPUT for each. Invoke:

```bash
bash wa/scripts/train_world.sh --lane external-h100 --epochs 1 \
  --batch-size 2 --accumulation 2 --workers 2 --seed 42 --world-weight 0.1
```

If raw data moved, add `--source-prefix /OLD/DATA/ROOT --data-root /NEW/DATA/ROOT`;
retain the original cache manifest and audited index files so hashes stay comparable.
The WLA source snapshot, trained action checkpoint, official source pins/weights,
original EVT cache/raw data and robot_transition_audit_v2 are required assets. They
are not bundled into public GitHub and must already exist on the external machine.
Changing hardware means numerical/throughput differences; not exact bitwise equivalence.

After completion:
```bash
python wa/tools/exchange.py pack --run "$WA_OUTPUT" --output "${WA_OUTPUT}.zip"
python wa/tools/exchange.py verify --bundle "${WA_OUTPUT}.zip"
```

Return this small bundle/config/environment/metrics via GitHub; do not commit datasets
or model weights. Training logs and checkpoints remain on that machine's persistent NAS.
