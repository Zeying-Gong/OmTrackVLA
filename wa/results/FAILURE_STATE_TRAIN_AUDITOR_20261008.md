# Independent failure-state training terminal auditor — 2026-10-08

Status: CPU_DEVELOPMENT_PASS_NOT_REAL_TRAINING_AUDIT. No formal failure-state training Job, new checkpoint or closed-loop SR was produced by this work.

## Purpose and unchanged experiment

The new tool `wa/tools/audit_failure_state_training.py` is an independent end-of-training checker, not a trainer or a replacement for scheduler/worker checks. Its output is accepted only after the planned full run has actual terminal artifacts. Frozen training source remains `199385cd9c826c8f21308ad99b6a8375396d9c90`, with no changes to model, losses, optimizer recipe, controller, physics, data or success criterion.

Draft `wa/jobs/failure_state_train_4090_v1.yaml` remains NOT_SUBMIT_READY; SHA256 `33838ac5dc95bf1a00133cdcb32d28a47224bc73f53bb6ed5fd863ccbf01d035`. The target-4090 real four-update training compatibility check still needs safe development GPU capacity. Existing A800 four-update and completed fixed152 policy-fit evidence are preserved, not rerun or mislabeled as 4090 backward/NCCL evidence.

## Checks implemented

- Read complete terminal metrics before hashing/loading a checkpoint; require final61252 and finite image/point/mixed metrics with exactly73368 heldout windows each. Re-read pinned metrics and reject any change from the initial read.
- Bind the frozen source, full26 training flags, runtime config, eight distinct RTX4090 UUIDs, immutable input graph, fixed152 selection and608 pretraining predictions.
- Reconstruct the immutable three-source plan and compare all four recorded int64 exposure arrays per window, including the old omitted position. Require1233424 total=726631 base+457641 old teacher+49152 recovery.
- Reconstruct eight-rank sampler/DataLoader assignment (154178 positions and77089 microbatches per rank); this is not independent per-rank actual telemetry.
- Check1543 scheduled log rows, original loss composition, resumed five-group cosine LR, final61252 and38545 updates.
- Validate CPU model and AdamW tensor identity/finiteness, resumed parent model+optimizer, unchanged group/hyperparameters, at least one changed model tensor. Every parent optimizer state already at22707 must reach61252; conditional parent states have bounded increments and are explicitly counted, not silently certified as full-step states.
- Rehash pinned inputs before releasing the independent offline-only report. Worker postcheck is supplementary, never a substitute. No new closed-loop or untouched-test claim.

## Executed evidence

Tool SHA256: 862aa8a71693773c00006846297b3d00daa40f5feecda8aeb076228a159873f8
Test SHA256: 389ecf5476b4a10af2ed0c01522cfe862b54d0526618f8901798db962835c10a

Independent tool tests: 34 tests PASS (0.448 seconds). Main directly related regression: 151 tests PASS (3.421 seconds), command:

```bash
cd /data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/checkout
PYTHONDONTWRITEBYTECODE=1 /data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/probe_env/bin/python -m unittest wa.tests.test_audit_failure_state_training wa.tests.test_failure_state_sampling wa.tests.test_failure_state_sampling_runtime wa.tests.test_failure_state_train_args wa.tests.test_failure_state_train_gate wa.tests.test_failure_state_group_fit wa.tests.test_teacher_window_plan
```

Main real-file CPU integration (synthetic launch/runtime envelope, no submitted Job):
1. `validate_config` parsed the actual draft YAML's26 flags; `load_plan` reconstructed actual pinned1233424 positions and original counts. Eight pins; elapsed1.955407729s; session11321 exit0.
2. `bind_inputs` and `bind_fit` read actual NAS arrays/admissions and completed fit, verified6864 recovery rows,73368 original heldout rows,152 selected windows/608 predictions, and47 stable file pins. Elapsed2.642060791s; session48333 exit0.
3. No checkpoint was loaded, no PNG decoded, no policy/GPU inference or training executed by these integrations. The synthetic envelope is a code/data compatibility test, not worker provenance.

Independent review found and corrected two false-PASS possibilities before acceptance: relying only on maximum optimizer step could hide a stopped active parameter/group; first-read metrics could differ from the later pinned content. Regression cases retain these failures. Earlier30 tests and147 related tests passed before these stricter cases were added; they are superseded by the final counts above, not evidence that the original checks were sufficient.

## Resource status and remaining work

Read-only snapshot2026-10-08 19:51:59 Beijing: A8003free/88; bj409080free/144. These are cluster capacities, not allocations. H100's separately mounted NAS was previously checked this day and lacks this project's ready code/environment/data; no migration was done.

Development4090 snapshots19:52:04 and19:52:55: GPU7free7690MiB/util0; GPU6free16252MiB bututil93→88; GPUs0–5free881–2892MiB. No device met the safe short-training criterion (at least10240MiB free and utilization≤5), given prior measured allocated peak7.309GiB plus CUDA/context/reserved headroom. No busy card was borrowed and no other process stopped.

Next: when safe development capacity is available, run the unchanged four-update recipe on4090, record actual kernel/backward/optimizer evidence, bind it into the final config and this auditor's exact input inventory, refresh resources and duplicate-job checks, announce and submit one complete8GPU epoch. Do not submit a cluster smoke task. At terminal time run this auditor against real artifacts, plus scheduler/log review and fixed-fit; only then proceed to fullSTT/DT/AT each1405 on3×8 GPUs.

Current best remains61609: STT1279, DT1178, AT1207 /1405 each. STT still needs10 additional successes. This work changes neither SR nor the evaluation-set-adaptation boundary.
