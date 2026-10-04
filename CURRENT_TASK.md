# WA current task — eight-GPU continuation of full mixed validation
Updated: 2026-10-04T10:03+08:00. Status: RUNNING / FULL_RESULTS_UNVERIFIED.

## Scope and acceptance
- User2026-10-04 authorized more GPUs to accelerate full4215 existing validation: STT/DT/AT1405each; no new training or LightNav full evaluation.
- Preserve original seed7/physics/model/controller/success criterion. RGB+initialBBox+ideal simulated polarUWB noise0 delay0, no text.
- HumanCollision target-person distance ever<0.5m, not general wall/doorframe contact. Invalid starts remain in denominator.
- Validation includes previous development/confirmation, not untouchedtest; no product90%/real-UWB/general-superiority claim.
- Final exactly4215 unique(task,key) pairs and per-task1405, model contracts, initialRGB, videos, SR/TR/CR, invalidcounts and HTML audit.
- TR uses reference-step denominator; macro_TR is separate mean following_rate.

## Acceleration and immutable continuation
- Parent60885/71808 deliberately STOPPED2026-10-04 09:14:27 for approved acceleration. Preserve source_full_mixed_learned_yaw_v2 d16c5a9e and all output.
- Parent output /data/nas_ray/project/md-ak/users/zeying.gong/job_60885/task_71808/wa_full_mixed_learned_yaw_v2.
- Frozen2238 complete rows STT830 DT704 AT704; only remaining1977 assigned to8lanes [248,247,247,247,247,247,247,247].
- Snapshot /data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/full_mixed_resume_8gpu_20261004.json SHA66bb0c34f3d44b0a238911e21b21a29502cc85d7076e9a59b2f9048aef1ed33b; embeds rows/sourcefilehashes/provenance. Do not overwrite.
- New60989/71912 RUNNING8A800, submitted09:16:57, actual8 A800-SXM4-80GB server_ready contracts and workerstarts verified.
- Source /data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/source_full_mixed_resume_8gpu_v1 commit5dcbeef5350dc55c345fa7688ec6c2b4ed422e35, clean detached worktree.
- Config wa/jobs/full_mixed_resume_8gpu_v1.yaml SHA1d12415f9f2ecf90f3a31d3b0e80d2cfd099be10d1b1dd7ff3d7c0cffa5a8196,8GPU timeout86400.
- Output /data/nas_ray/project/md-ak/users/zeying.gong/job_60989/task_71912/wa_full_mixed_resume_8gpu_v1.
- Fixed60502/71381 checkpoint step45900 SHA20cc84b3f231ad4056e16b91c52c84cb14e18d886a80324b5f27ffd773be5331; learned_yaw_guard_v1 mixedzero,4samplingsteps. JEPA/MetaQuery/ActionExpert unchanged, worldpredictor training-only.
- Resourcecheck baidu_bj_a80033free before submit; sameNAS, no migration, no effect on otherWLAjobs.
- Developer16CPU tests+8x3real Habitat dataset audits PASS. Evidence /data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/full_mixed_resume_preflight_20261004; dependency/checkpoint hashesPASS.
- 2026-10-04T10:03+08:00 MONITOR60989/71912 RUNNING8A800-SXM4-80GB: new523/1977 lanes[65,70,60,67,64,66,64,67], combined2761/4215(STT1353 DT704 AT704), +225 since09:40. All8workers progressingSTT; workerbytes[170424,177871,162295,180375,171535,177518,171387,181102] increased from prior snapshot, ages3..33s, no fatal/OOM/Traceback. All8ready modelcontractsPASS; frozen2238/sourcehashesunchanged; newoldkeysdisjoint. Invalidinit105(total;22new),allfailure retained. New528mp4files includes active recordings, not completed-video audit. No laneCOMPLETE or fullsummary yet. No code/config/checkpoint mutations or newjobs; continue existing8lane eval then4215 audit/HTML.
- Completion: require8new COMPLETE with1977rows, no overlap with2238snapshot, then combined_episodes.jsonl and summary.json; artifact_root identifies old/new video path. Old8COMPLETE requirement superseded by interrupted parent snapshot.
- Monitorwa updated every20min to60989/71912 and combinedcounts; finalreport/audit thenpause. No duplicatejobs or changing running sources/configs.
- First60883/71806 startupFAILED0episodes retained. Original60885 stop is intentional acceleration, not model failure.
- Original full2GPU phase CURRENT_TASK archived archive/2026-10/CURRENT_TASK_before_8gpu_resume_20261004.md.

## Prior evidence retained
- Confirmation WA60770/71649 20/24 vsLightNav60771/71650 17/24; collision2vs3; invalid0; initialRGB24matched.
- STT6vs7, DT7vs5, AT7vs5. Development60767 17/24 equalLN. Not all-task superiority.
- Full prior phase state archived archive/2026-10/CURRENT_TASK_before_full_mixed_20261003.md.
- Existing confirmation HTML18793, development18792, TB6006 (not full4215 results).
- Forward: ssh -N -L 18793:127.0.0.1:18793 -L 16006:127.0.0.1:6006 devpod-a800
