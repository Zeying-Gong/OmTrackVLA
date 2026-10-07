# Current hard-STT training — Job61609 / Task72803

Verified 2026-10-07: the existing loopback service on devpod-a800:6006 exposes the new `hard_stt_61609` run. HTTP scalar API returned real steps22708..24100 at inspection; this is a live training curve, not a closed-loop score.

```bash
ssh -N -L 16006:127.0.0.1:6006 devpod-a800
```

Open http://127.0.0.1:16006/#scalars and select `hard_stt_61609`. Reuse an existing tunnel if local16006 is already occupied. This check verified the remote service/API; it did not start a new local tunnel.

- JSONL input: `/data/nas_ray/project/md-ak/users/zeying.gong/job_61609/task_72803/wa_hard_stt_train_a800_v1/train.jsonl`.
- Events: `/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/tensorboard_59566/events/hard_stt_61609`.
- Export log: `/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/tb_hard_stt_61609.log`; initial exporter PID2619386.
- Exporter uses the immutable `source_hard_stt_train_v1/wa/tools/tensorboard_jsonl.py`; it does not edit training or old event directories. Existing server PID10620 was reused.
- `train_ddp_accumulation_mean/*` reports the global DDP accumulation-group mean. Rolling100 averages100 logged samples, not100 optimizer steps. Grad norm is before clipping. Offline image/point/mixed metrics appear only after final metrics.json exists.
- Planned parent22707 +37009 new updates =59716 is not yet a completion claim. The logger normally samples every25steps; final checkpoint/metrics, not the last sampled log line, determine completion.
- Full STT/DT/AT closed-loop evaluation remains required. Loss reduction alone does not establish the goal STT>=1289, DT>=1173, AT>=1203 out of1405 each.

## Historical snapshot below — not current job status

# JEPA-WM TensorBoard — Job59566

Training completed22707 optimizer steps (one full epoch), confirmed from final checkpoint metadata. The scheduler job is still RUNNING; final offline metrics.json was absent at inspection. Do not confuse training completion with evaluation completion or method effectiveness.

The separate `tensorboard_env` does not modify the training runtime. Server listens only at127.0.0.1:6006 on `devpod-a800`. Local SSH forwarding was verified HTTP200 and scalar tags loaded.

```bash
ssh -N -L 16006:127.0.0.1:6006 devpod-a800
```

Open http://127.0.0.1:16006/#scalars . A forwarding process was already started on the current control machine; do not start another if the port is occupied by this tunnel.

Paths under `/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/tensorboard_59566`:
- `events/jepa_world_v1`: TensorBoard events.
- `export.log`: JSONL exporter (initial PID10619), automatically appends final validation metrics once available.
- `server.log`: TensorBoard service (initial PID10620).

Charts: raw sampled total/flow/geometry/world losses; rolling100 logged-sample means; unclipped gradient norm; GPU memory; elapsed time. Each sample is rank0's last microbatch every25 steps, NOT a full effective-batch or DDP mean. Rolling100 corresponds to about2500 optimizer steps. Last logged training step22700; checkpoint has22707 because the logger samples every25steps.

Validation will provide final image/point/mixed ADE/FDE/yaw only. There is no historical validation learning curve and no closed-loop SR/collision-rate evidence. Do not synthesize missing validation curves or claim convergence from loss alone. Any continuation recipe should be decided after these metrics and preferably checkpoint-wise validation; no new training has been submitted.
