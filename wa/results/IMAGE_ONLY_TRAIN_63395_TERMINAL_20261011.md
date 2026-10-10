# Image-only same-parent training: Job 63395 terminal audit (2026-10-11)

Status: TRAINING_ARTIFACT_AUDIT_PASS_OFFLINE_ONLY_IMAGE_INPUT. This is not closed-loop success evidence.

- Job 63395 / Task 74738 SUCCEEDED at 07:20:46 Beijing on canonical baidu_bj_a800 K8s; eight distinct NVIDIA A800-SXM4-80GB UUIDs (cc 8.0) were recorded and checked.
- Frozen source: R/source_failure_state_image_only_v1 at commit 76635eb257e43ab2c8194e5b358cac023a3351a0. YAML: C/wa/jobs/failure_state_image_only_a800_v1.yaml SHA256 f5413cf82f95d6d0d032675445ecb58b7d307d59a907268d02400d34b6a4c0ea.
- Parent: J/job_59866/task_70728/wa_history_repeat_v1/checkpoint.pt SHA256 ab39144b0490937ce43c4ab28e2d90cdb0c560f700d3364a29ce1dc9a08b999d, model AND optimizer step 22707. This is an independent one-epoch continuation, not a third epoch from 61609 or 62256.
- Run: J/job_63395/task_74738/wa_failure_state_image_only_a800_v1. Final checkpoint.pt: 4,358,277,769 bytes, SHA256 ac8ca38158d89ab6dcd11a9f604b09d37a3db7db63075091cfd66458a2171efa, step 61252, cumulative two epochs.
- Actual 38,545 new optimizer updates consumed 1,233,424 positions: base 726,631 + retained teacher 457,641 + recovery 49,152. Input modes: image 1,233,424; point 0; mixed 0. Dropped/repeated positions 0.

| Offline input | Windows | ADE m | FDE m | Yaw MAE rad | SR |
| --- | ---: | ---: | ---: | ---: | --- |
| image | 73,368 | 0.265896 | 0.463631 | 0.124653 | null |
| point | 73,368 | 0.261032 | 0.455839 | 0.116104 | null |
| mixed | 73,368 | 0.258561 | 0.450701 | 0.115351 | null |

Prior mixed-trained 62256 checkpoint scored image ADE 0.266651 on the same heldout window count; the 0.000755 m difference is small and does not establish a closed-loop SR gain. Its point/mixed ADE were lower than this image-only model.

Worker emitted one COMPLETE and one WORKER_POSTCHECK_PASS; metrics.json and worker_postcheck.json are present. The independent auditor checked frozen source/YAML and input pins, 8 unique A800 devices, exposure/accounting, final model-plus-optimizer continuation, offline metrics and artifact hashes. Console fatal/OOM/Traceback/error scan returned none; NCCL device-id warnings at init were retained.

Independent report: R/artifacts/failure_state_image_only_training_audit_63395_v1.json, SHA256 619e35c26916c490c164ce956929e10bc82ce72f473643b76a97348c37411203. Auditor status is limited to offline training artifacts. Rank assignments were reconstructed; merged exposure counters were worker-recorded rather than independent per-rank telemetry.

Next gate: fresh full closed-loop STT/DT/AT image/no-UWB evaluations on the exact 1,405-key manifest and eight fixed shards per task, with checkpoint SHA binding and independent per-task media/trajectory audit. No such job has been submitted yet. Preserve the direct-removal 62256 paired comparison separately from this retrained checkpoint; do not infer unseen-test generalization or change model/controller/physics/success threshold.
