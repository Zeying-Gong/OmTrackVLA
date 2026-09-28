# WLA overnight training — 2026-09-28

## Authorization and actual execution

User explicitly authorized overnight formal training and full evaluation, and wants verified results in the morning. WLA remains the main research route; DA3 remains retired. Morning delivery is a status/results checkpoint, not a guarantee that all4215 evaluations will finish.

- Job59097 / Task69868: RUNNING on4A800, submitted22:52:28 Asia/Shanghai. Full2043 train/heldout recovery rollouts with per-frame target UV/visibility and controller diagnostics.
- Job59110 / Task69881: RUNNING on2A800, submitted23:07:42. Full initial-frame target-point pretraining,15347training images/1543heldout images,2epochs. Actual optimizer updates verified:at least200/3838; finite loss and gradients. This is training progress, not efficacy.
- Subsequent four-A800 memory training and full4215 evaluation are authorized and prepared, but not yet submitted. Both upstream completion audits are mandatory gates.
- A30-minute heartbeat in this task follows progress and submits the next complete stage once gates pass. It reports meaningful changes and a morning status after08:00 Asia/Shanghai. It does not reserve GPU workers while waiting.

Execution snapshot: `nas-a800:/data/nas_ray/home/zeying.gong/algorithm/repos/WLA-EVT-20260925`.
The snapshot is not a Git checkout. Declared upstream revision: `cb78954213f5ca32494f13639b8340797144a664`; exact execution is identified by the source manifest, not a claimed new Git commit.
Collection manifest SHA256: `08856dc6ddab0ac591d0d73acd7bb68962a128664849a38c65b795ce1c4a45c5`.
Training manifest SHA256 (442files): `decc680c93287c520a43af1a40f1ce716fffa2960b32cbd6aee823c33b88100d`.
No cross-NAS model/data copy or platform migration is involved.

## Why two stages

Per-step masks are missing in historical trajectories, so new temporal labels require collection. Existing aligned initial panoptic masks are available and were re-audited across every original15347train/1543heldout example, including release/observation hashes and scene separation. They allow target localization learning concurrently with collection.

All these initial frames contain a visible target. Therefore initial pretraining freezes the visibility head, GRU and action-conditioning residual, and trains attention pooling/encode/read/UV only. It cannot learn occlusion recovery or be claimed to improve the unchanged action policy. Report full heldout UV error before/after training; no all-positive visibility accuracy claim.

After full collection, admit only successful collision-free teacher recovery windows with verified identity and action ownership, preserving all rejected raw trajectories. Require at least25training and3heldout recovery episodes per task. Every accepted teacher window appears once per epoch, in ordered within-episode four-step clips; padded steps have zero weight. Student actions are not expert labels. Validation videos and user review annotations never enter training.

## Model and training

Initialize frozen WLA Job58346 step43203, SHA256 `0b8f036fd282474d8c9efe9e34e1f4dc56b9282c44295bcda1f16edcf48460e1`.
Keep original Qwen/MetaQuery/flow action expert/geometry head fixed. Add target-attention pooling, normalized point and visibility heads,256D gated GRU, and initially zero residual into MetaQuery taps. Inference accepts only original text, causal RGB and timestamps; privileged labels never enter inference.

Initial phase:2A800, full2epochs, batch4/GPU, AdamW3e-4, point loss only.
Temporal phase:4A800, full2epochs, batch2clips/GPU, AdamW3e-4, task-balanced loss = flow +0.5geometry +0.2visible-point +0.1visibility,2%warmup/cosine/gradient clipping. Load completed initial grounder before temporal training. Save adapter plus exact frozen-baseline reference; do not overwrite old checkpoints.

Memory resets per clip in training and per episode in serving. Four-step training versus long rollouts is an explicit exposure limitation. Grounding-only ablation is not yet run, so any combined improvement cannot be attributed to memory alone.

## Verification and remaining acceptance

PASS: full initial-frame audit; real four-step single-GPU and dual-GPU two-update checks; new-module gradients and actual parameter changes; checkpoint round-trip; serving state changes causally, resets reproduce identical predictions, privileged request fields rejected. Peak dual-GPU developer memory11.67GiB/rank.
A developer-only checkpoint race was found when both ranks wrote one temporary file, fixed to rank0-only, and the full dual-GPU check passed on retry. Failed logs remain on NAS.
No new model has yet demonstrated tracking improvement.

Temporal training performs full heldout action/grounding evaluation, then runs all4215 frozen benchmark episodes (1405each STT/DT/AT), eight logical shards on four GPUs. Controller remains range_only, seed7 and four flow steps unchanged. Compare SR/TR/CR and coverage against same-protocol WLA58346 and LightNav58433; never infer efficacy from loss.

## Exact remote artifacts

- Collection: `/data/nas_ray/project/md-ak/users/zeying.gong/job_59097/task_69868/wla_grounding_collection`.
- Initial training: `/data/nas_ray/project/md-ak/users/zeying.gong/job_59110/task_69881/wla_grounding_pretrain`.
- Code/config/state under WLA snapshot: `target_memory_train_20260928/{RECIPE.md,source.sha256,DEVELOPMENT_READY.json,BOOTSTRAP_RUN.json,RUN.json}`.
- Initial launch: `bash target_memory_train_20260928/bootstrap_pipeline.sh`, scheduler config `bootstrap_a800.yaml`.
- Next full launch: `bash target_memory_train_20260928/pipeline.sh`, scheduler config `full_a800.yaml`; creates independent `job_<id>/task_<id>/wla_target_memory` NAS output.
- Original WLA58638 evaluation remains running independently and must not be stopped.

## Verified23:42 update
Job59110/Task69881 SUCCEEDED23:24:57, all3838updates/2epochs complete. Full1543heldout normalized UV L2 error improved from0.11451486684019814 to0.029675512296363123 (74.09% reduction versus the untrained point head). This comparison does not measure identification among distractors, temporal recovery or tracking success; action policy is unchanged. Checkpoint grounder.pt SHA256219e963119e526ff901e45ea13582d60107e2326924bf5da34c064dabb86f730 verified against file. Job59097 remains RUNNING,270/2043 completed at23:42. Temporal stage remains WAITING_FOR_FULL_COLLECTION; no additional GPU allocation submitted.

## Verified2026-09-29 05:53 update
Collection59097/69868 SUCCEEDED05:38:21:2043rollouts,176814frames,170275visible/6539invisible;621train/70heldout scenes,zero benchmark scene overlap. Full collection labels audited; qualified ordered teacher clips are admitted by next pipeline.
Both upstream gates passed, frozen442-file source check and checkpoint SHA passed; logged-in md_ai_kit2.0.0 showed8free Beijing A800. No duplicate temporal job existed (unrelated queued59125 is wa_mobile on Baoding and was untouched). Submitted59352/70125 at05:52:38;RUNNING05:52:43 on4A800. Output /data/nas_ray/project/md-ak/users/zeying.gong/job_59352/task_70125/wla_target_memory. Full2epochs then full4215evaluation, unchanged recipe. Optimizer updates not yet verified at this check.
