# WA current task — repair MP3D semantic rendering and fresh full validation
Updated: 2026-10-04. Status: SEMANTIC_FIX_IMPLEMENTED_TESTED / FRESH_FULL_EVAL_PENDING.

## Active repair
- User requested immediate repair of invalid initialization; prior165 allMP3D. Fixed duplicate semantic PLY stage rotation with independent mp3d_semantic_ply_v1 config; RGB/collision frame retained, no shared asset/runtime changes.
- Five paired real static cases across STT/DT/AT +HM3D: RGB/camera/allagent poses/nonsemantic stage identical; STTdepth identical; DT/ATdepth absent.55 target0->5198;AT299 0->3427. Not new SR.
- 18CPUtests PASS;4215 real episode definitions PASS(each721MP3D+684HM3D). Weights/controllers/success criterion unchanged. Report wa/results/SEMANTIC_PLY_FIX_20261004.md.
- Old full81.138790% remains pre-fix protocol evidence; semantic detector also affects HumanFollowing. Need fresh4215, never merge old rows or only replace165.
- Next freeze independent source and submit full8A800 with WA_SEMANTIC_PLY_FIX=mp3d_semantic_ply_v1; no training or LightNav full run. Record scheduler IDs before reporting running.
- Prior full-result monitor remains paused until new full job is verified.

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
- New60989/71912 SUCCEEDED12:45:43 with8A800, submitted09:16:57, actual8 A800-SXM4-80GB server_ready contracts and workerstarts verified.
- Source /data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/source_full_mixed_resume_8gpu_v1 commit5dcbeef5350dc55c345fa7688ec6c2b4ed422e35, clean detached worktree.
- Config wa/jobs/full_mixed_resume_8gpu_v1.yaml SHA1d12415f9f2ecf90f3a31d3b0e80d2cfd099be10d1b1dd7ff3d7c0cffa5a8196,8GPU timeout86400.
- Output /data/nas_ray/project/md-ak/users/zeying.gong/job_60989/task_71912/wa_full_mixed_resume_8gpu_v1.
- Fixed60502/71381 checkpoint step45900 SHA20cc84b3f231ad4056e16b91c52c84cb14e18d886a80324b5f27ffd773be5331; learned_yaw_guard_v1 mixedzero,4samplingsteps. JEPA/MetaQuery/ActionExpert unchanged, worldpredictor training-only.
- Resourcecheck baidu_bj_a80033free before submit; sameNAS, no migration, no effect on otherWLAjobs.
- Developer16CPU tests+8x3real Habitat dataset audits PASS. Evidence /data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/full_mixed_resume_preflight_20261004; dependency/checkpoint hashesPASS.
- 2026-10-04 FINAL60989/71912 SUCCEEDED12:45:43: 8newlanes COMPLETE1977 plus frozen2238 =4215unique, STT/DT/AT1405each; old/new disjoint, prior hashes unchanged, all readycontractsPASS. SR85.693950/77.722420/80.000000%; TR81.424200/74.058413/79.949682%; HumanCollision CR5.053381/7.259786/5.622776%; invalid57/57/51 retained. Overall3420/4215 SR81.138790%,DTbelow80%; not all-task gate or general superiority. InitialRGB4215sha and4215video ffprobe metadata/durationPASS (not every-frame decode). Each task52missingreference steps uses existing actual-step fallback; macro_TR separate. Final summary/combined provenance on NAS; report wa/results/FULL_MIXED_60989.md; HTML artifacts/full_mixed_review_60989 at18794 verified browser4215loaded andATinvalidfilter51. No fatal/OOM or training/newjobs. Monitorwa PAUSED after audit; research remainsPARTIAL.
- Completion: require8new COMPLETE with1977rows, no overlap with2238snapshot, then combined_episodes.jsonl and summary.json; artifact_root identifies old/new video path. Old8COMPLETE requirement superseded by interrupted parent snapshot.
- Monitorwa PAUSED after full completion/audit. No further training/evaluation authorized by this monitor.
- First60883/71806 startupFAILED0episodes retained. Original60885 stop is intentional acceleration, not model failure.
- Original full2GPU phase CURRENT_TASK archived archive/2026-10/CURRENT_TASK_before_8gpu_resume_20261004.md.

## Prior evidence retained
- Confirmation WA60770/71649 20/24 vsLightNav60771/71650 17/24; collision2vs3; invalid0; initialRGB24matched.
- STT6vs7, DT7vs5, AT7vs5. Development60767 17/24 equalLN. Not all-task superiority.
- Full prior phase state archived archive/2026-10/CURRENT_TASK_before_full_mixed_20261003.md.
- Existing confirmation HTML18793, development18792, TB6006 (not full4215 results).
- Forward: ssh -N -L 18793:127.0.0.1:18793 -L 16006:127.0.0.1:6006 devpod-a800

## Final results and access
- Full report wa/results/FULL_MIXED_60989.md; JSON summary/audit alongside.
- Full4215 HTML on18794, independent of old small-sample pages.
- Forward: ssh -N -L 18794:127.0.0.1:18794 devpod-a800 ; open http://127.0.0.1:18794/ .

## Post-evaluation validity diagnostic
- 2026-10-04 INIT_BBOX_DIAGNOSIS PARTIAL_NOT_FIXED:165invalid allzeroBBox/firsttargetpixels0/allactions0;107later visible(35/37/35). Four original firstRGB frames visibly contain people. Developer STT oLBMNvg9in8/55 reproduces all147456semanticpixels=ID160 ceiling with expected1098 and RGB/semantic actualcamera matrices equal; control/12 has6730targetpixels andvalidbox. Specific-start semantic rendering/asset inconsistency isolated, exactasset mechanism and all165coverageunverified. Same semantic facing participates HumanFollowing. No dropping/relabeling rows or changing metrics/policy; current81.138790%retained. Report wa/results/INIT_BBOX_DIAGNOSIS_20261004.md; records165JSON. No formal jobs or retraining; monitor remainsPAUSED.
