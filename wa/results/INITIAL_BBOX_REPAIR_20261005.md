# Residual seven initialization repair — 2026-10-05
Status: SUCCEEDED_AUDITED_7_ONLY (61259/72336); broader research PARTIAL. Earlier preparation/failure history retained below.
User authorized repairing/replacing only seven residual invalid rows; no full rerun.
## Evidence
All seven belong to pRbA3pwrgk9: STT27/38, DT27/38, AT27/38/71.
Existing semantic PLY rendering returned zero target pixels. Isolated no-action paired rendering using the RGB scene graph recovered 3706/4509 (STT),3706/4468 (DT),3706/4489/5449 (AT) target pixels.
All seven paired RGB arrays, cameras, agent poses and nonsemantic stage properties are identical. Available depth arrays also identical; DT/AT depth absent, not claimed verified. Visually reviewed all seven mask overlays: visible target human, not another person or through-wall projection.
Habitat Simulator._sanitize_config forces load_semantic_mesh true when semantic sensors exist; merely setting config false did not disable it. Diagnostic process-only wrapper disabled it after sanitation. Shared simulator never edited.
Failed alternative GLB semantic-asset probe asserted because GLB lacks semantic vertex colors; retained artifacts, not adopted.
## Scoped repair
Freeze the corrected initial GT bbox with raw first-RGB hash per task/key in artifacts/initial_bbox_repair_v1/plan.json.
Only seven selected first calls may substitute a zero bbox after exact image-hash match. Valid/nonzero boxes, image mismatch and wrong episode fail closed.
Policy RGB/current ideal UWB and weights/controller unchanged. No future target, later GT boxes or semantic image enter policy.
Closed-loop simulator STILL uses mp3d_semantic_ply_v1 throughout. HumanFollowing and all success/collision thresholds unchanged; renderer ablation is not enabled in evaluation.
This fixes initial annotation only; it does NOT claim all subsequent semantic visibility errors are corrected. Broad semantic-metric replacement is out of this request.
Merge all seven actual outcomes, not only improvements, with unchanged4208 baseline rows; preserve old results and fixed4215 denominator.
## Artifacts
Plan: /data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/initial_bbox_repair_v1/plan.json
SHA256: 2c163cd46a539480c486d9225d477c6e9354f2fd416d9203c7e0e8d5c3f2fcf8
Review: same directory/first_frame_review.png.
Paired reports: artifacts/residual7_no_semantic_mesh_v2_{task}_pRbA3pwrgk9_{episode}/report.json.
Baseline: job61171/task72191/wa_semantic_targeted_v1/combined_episodes.jsonl, immutable SHA in plan.
CPU5 tests pass incl first-only forwarding; two real dataset lane audits select4+3 episodes total7. Dependency hashes pass. Checkpoint hash recheck pending completion.

## First submission retained; lighting-aligned revision
61257/72334 FAILED09:54:52 Beijing,2actualA800,0completed. Strict rawRGBguard failed before first policy action: static probe lacked evaluate_agent's four directional LightInfo entries. Old source60e0d7df and config2a881ef1/plan2c163cd4 remain immutable.
Lighting-aligned seven paired static renders now all pass; bbox pixel geometry unchanged. Re-encoding all seven original static frames using the same imageio/JPEG path exactly matches audited61171 first-frame JPEG SHA.
New immutable plan artifacts/initial_bbox_repair_v2/plan.json SHA6535f7b9a53e399ba7f2323c2d7f72d223146ee77f090223eb260f5885f6269a. Checkpoint full SHA reconfirmed20cc84b3;CPU5boundary tests+4merge cases PASS.
Added actual evaluate_agent first-call check that validates hash/bbox and exits before any action. Must pass all7 before resubmission; no relaxed hash gate or full rerun.

## Final result — 61259/72336
Submitted2026-10-05 10:03:45 Beijing;SUCCEEDED10:07:05,elapsed3m20s,2NVIDIA A800-SXM4-80GB.
Frozen source_initial_bbox_repair_v2 commit1efe5e6131e16bda712c6d5d5ee6a4a1f633a973;configSHA05c74fcbb6c9273b077b3661ec34c4b03373eac6cd65ddad70b51318475c5b15;plan6535f7b9 unchanged.
Seven firstcall tests passed before submission;two worker lanes COMPLETE4+3;no worker Traceback/FATAL/CUDA OOM.
All seven initialized successfully. No claim all seven closed-loop success.

| Task | Episode | Init valid | Success | HumanCollision | Following rate | End |
|---|---|---|---|---|---|---|
| STT |27|true|0|0|0.338462|Normal, below criterion|
| STT |38|true|1|0|0.809524|Normal|
| DT |27|true|0|0|0.323077|Normal, below criterion|
| DT |38|true|1|0|0.788732|Normal|
| AT |27|true|0|0|0.333333|Normal, below criterion|
| AT |38|true|1|0|0.800000|Normal|
| AT |71|true|0|1|0.880000|Collision|

Exactly7 replacements +4208 unmodified rows =4215unique,1405/task.
Total3627success/4215=86.049822% (baseline3624/4215=85.978648%). This is first-frame annotation repair, not training improvement.

| Task | SR % | reference-normalized TR % | HumanCollision CR % | Invalid | macro_TR % |
|---|---:|---:|---:|---:|---:|
| STT |90.818505|87.573986|4.341637|0|92.475068|
| DT |82.419929|78.697793|6.690391|0|81.432605|
| AT |84.911032|85.206358|4.697509|0|88.150881|

All4215 source-image fileSHA/oldRGBpairs andvideo ffprobe metadata/duration PASS (not everyframe decoded).
All4208 unchanged row dictionaries and7 replacement keys checked;summary recomputed identically. Failure rows remain in denominator.
Subsequent semantic visibility scoring deliberately retains previous protocol; residual three27 failures are below the current following criterion, NOT proof that all remaining failures are purely policy failures. No wider metric repair evaluated.
CombinedSHA517937840d3e5da9b42717cf3c16dadbce499bf1c6c203173badb6c8f3b9c08e.
SummarySHA3c6ef1a2039e557aa564a7bdbca7ce0e3a616246f5c483991b137c640138b3c2.
Output /data/nas_ray/project/md-ak/users/zeying.gong/job_61259/task_72336/wa_initial_bbox_repair_v2.
New page/audit /data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/initial_bbox_repair_review_job_61259.
HTTP200 andbrowser7-filter verified. Original videos unedited; separate first-frame annotated-mask contact sheet shows repairedboxes. Existing recorder sees original semantic sensor, not the substituted policy-only initialbbox.
Access: `ssh -N -L 18797:127.0.0.1:18797 devpod-a800`, then http://127.0.0.1:18797/.
Old18796 andbaseline4215 preserved;failed61257 kept. No further task or training launched;monitor remains paused.
