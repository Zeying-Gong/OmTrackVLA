# Image-only controlled training preflight — 2026-10-10

Status at 13:06 Beijing: PREPARED_NOT_SUBMITTED. This is an evaluation-set-adaptation experiment, not evidence of unseen-test generalization or an improved closed-loop SR.
Path prefixes: R=/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928; C=R/checkout; J=/data/nas_ray/project/md-ak/users/zeying.gong.

## Reason for the controlled branch

The audited 62256 same-weight direct image/no-UWB results are STT 1124/1405 (80.00%), DT 454/1405 (32.31%), AT 707/1405 (50.32%), versus mixed 1300/1184/1212. Image-only STT is exactly 80.00%, not above 80%. The observer-only visibility analysis reports target semantic zero-pixel frames in 638/739 mixed-success/image-failure DT episodes and 147/527 AT episodes; this is a diagnostic association, not a causal or policy-input claim. Current image mode already uses four causal RGB frames and the episode-zero GT bbox template. Existing point mode is not UWB-only because it still encodes RGB.

## Frozen, single-variable training proposal

- Training source: R/source_failure_state_image_only_v1, commit 76635eb257e43ab2c8194e5b358cac023a3351a0, clean detached worktree. Original 62256 training/evaluation frozen sources are unchanged.
- Launch config: C/wa/jobs/failure_state_image_only_a800_v1.yaml, SHA256 f5413cf82f95d6d0d032675445ecb58b7d307d59a907268d02400d34b6a4c0ea. One K8s shell Task requests 8 A800 GPUs on original NAS, timeout 86400s, no alternate hardware/NAS.
- Initialization: parent59866 model AND optimizer checkpoint J/job_59866/task_70728/wa_history_repeat_v1/checkpoint.pt SHA256 ab39144b0490937ce43c4ab28e2d90cdb0c560f700d3364a29ce1dc9a08b999d at step22707. This is a separate one-epoch branch, not a third epoch after 62256.
- All data, labels, sampling plan, model/loss/LR/controller/physics and seed42 remain the fixed 62256 recipe. Only `--train-input-mode image` forces consumed mode IDs to image. Planned 1,233,424 exposures (726,631 base + 457,641 old teacher + 49,152 recovery) and 38,545 updates to step61252. Expected actual mode exposure is image1,233,424 / point0 / mixed0; worker postcheck and independent auditor must verify, not infer from a flag.
- Dynamic output: J/job_${MD_AK_JOB_ID}/task_${MD_AK_TASK_ID}/wa_failure_state_image_only_a800_v1. No job/task ID exists yet. Checkpoint, full source/config/launch hashes, eight unique A800 UUIDs and actual worker progress are future gates. Heldout ADE/FDE are offline metrics, not SR.
- This cache was assembled for earlier mixed-policy failure recovery; only 68 current image-only STT failures overlap, and no current DT/AT image-specific failure trajectory is represented in it. Therefore image-only modality exposure is an isolating ablation, not a guarantee of DT/AT improvement. If full closed-loop image results remain weak, the next evidence-driven step is image-policy rollout state collection and successful teacher recovery labels, rather than blind extra epochs.

## Bounded new developer evidence

A single A800 developer diagnostic at R/artifacts/failure_state_image_only_developer_20261010_v1 finished exit0 at 12:45 Beijing. It consumed exactly 16 image positions, 0 point and 0 mixed, with four finite-gradient optimizer updates 22708–22711 and only two heldout windows per mode. Its config/actual/environment/metrics/train/log SHA256 respectively are 3053df655ebec91cb4cf25322732ededa7aebeca4a766a735db4f9598c28f561, f08520404a53087ae66e3105e58733f9b0647526f66d60df733e5f0efde28adf, e0e1f499e6015427186413fa1e4bc7844032bfbaedf8d6d633ba52b971295b83, 3ef5c1c83cb852a35719874af3cc55be77099e3073ce967ec4523b6773c7846d, 12e1697d9d532fef0bbb91d2bf536318990871a0d11e9ef2a7bc46d16436824a, 54347ae0401cd546c950735c04f50e11af8f56211bc032c18d6afd8f42c911d7. The YAML separately pins nine new diagnostic files plus the old parent 24-input whitelist. The diagnostic creates no new checkpoint, has no eight-GPU or closed-loop evidence, and cannot demonstrate SR.

Independent CPU auditor C/wa/tools/audit_image_only_training.py is frozen to source/YAML/task identities but has NOT_RUN_ON_REAL_OUTPUT status. New and prior auditor suites pass 55/55 CPU tests; final terminal audit remains pending a real completed job. YAML, shell syntax, embedded Python AST and SHA profile passed. Real A800 scheduler snapshot was 1 free / 88 total GPUs; queue rank/ETA and allocation are unknown. No formal training, pure-UWB, longer-frame, RL or new 4090 evaluation job has been submitted from this preflight.
