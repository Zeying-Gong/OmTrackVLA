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
