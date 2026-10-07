# WA current task — STT target and UWB study
Updated: 2026-10-07 Beijing. Goal ACTIVE; heartbeat wa ACTIVE every20minutes.
Full older task: archive/2026-10/CURRENT_TASK_before_stt_goal_20261007.md; historical watches retained in wa/results.

## Acceptance and latest verified result
- Same1405/task: STT>=1289 (>91.7%), DT>=1173, AT>=1203; then same-weight with/withoutUWB study.
- Latest61609: STT1279 (91.032028%), DT1178 (83.843416%), AT1207 (85.907473%); total3664/4215,invalid0.
- Versus61377: net+3/+5/+4. DT/AT met; STT gap10; goal NOT_MET,not research/product completion.
- Paired gains/regressions STT15/12,DT26/21,AT25/21. No selective reruns or dropped failures.
- Explicit evaluation-set adaptation authorized; never claim untouched-test generalization. WA and LightNav have different inputs.

## Completed61609 training and complete24GPU evaluation — do not duplicate
- Train61609/72803 SUCCEEDED15:50:25Beijing;8A800;independent59866 model+optimizer branch,1new epoch,cumulative2.
- New37009updates,finalstep59716;not61377epoch3;originalarchitecture/loss/controller/physics/criteria unchanged.
- Checkpoint:/data/nas_ray/project/md-ak/users/zeying.gong/job_61609/task_72803/wa_hard_stt_train_a800_v1/checkpoint.pt
- SHA256:c510c04d987d141423401a25323ea353314107a16f4dac5d2901370ab199fa52
- Training source_hard_stt_train_v1 commit8d8efe3aa8a7ce9714913b6f65e5e3c8196eae06;config wa/jobs/hard_stt_train_a800_v1.yaml SHA99f8fc7491b4f7c4c347c73b0ba4f1bf363ff292c76de0d4b29fd9850df19184.
- Actual1184272exposures=726631base+457641teacher;10413hardwindows each3=31239,earlyhard9810;teacher38.643234%,8ranks148034each,one nonhardATOracle taildrop.
- Training audit artifacts/hard_stt_training_audit_61609_v1.json SHA59d5b468eec4283db1a1d87ad7b9a9d1bc2123739cffc38cebfc9a1e14dcc727 PASS24198sourcehashes/optimizer/actualexposure/heldout.
- Heldout73368/mode ADE/FDE image.265734/.463479 point.253940/.442441 mixed.253890/.441945;notSR.
- Eval61653/72847 STT SUCCEEDED18:26:50;61654/72848 DT SUCCEEDED18:50:16;61655/72849 AT SUCCEEDED18:45:01;2026-10-07Beijing.
- Each8actualA800-SXM4-80GB/1405episodes;24COMPLETE/3PARTITION_COMPLETE/4215unique/78sourcehashes/24ready PASS;invalid0.
- Frozen source_student61609_eval24_v1 commit192b57f5e270acfffd8c7c1a4590cb1b257d92a3,unchanged.
- Configs wa/jobs/student61609_{stt,dt,at}_a800_v1.yaml;preflight98CPUtests+actualshortsyntheticinterface+24realdefinitionlanes previouslyPASS,no cluster smoke.
- Run roots:/data/nas_ray/project/md-ak/users/zeying.gong/job_<ID>/task_<ID>/wa_student61609_<task>_a800_v1
- No oldrows reused. NewA800 vsbaselineRTX4090; no crossGPU bitwise-equivalence claim.

## Final result and evidence
- Full report:wa/results/STUDENT61609_FINAL_20261007.json;ledger wa/results/PROJECT_LEDGER_20261007.md.
- Merge/start audit:artifacts/student61609_full_audit_20261007_v1;4215actualrawRGB/dynamicstate pairs against bothteachers PASS.
- Summary SHA223b4b8140e79842408bc7104d16f75b7b4ff1d409354ee547b00e4b49f937f7;combinedSHA0ab45e1b35bb0b8809fcc77fcaabca59b35bab0839507d716c371ce2d5a6f358.
- PairSHA dc98149bdcfb6baf05d1fae167de572b476679f48384dd91ea92e57851645844,identical pinnedteacher evidence.
- Goal artifacts/student61609_goal_20261007_v1/goal_report.json SHA92e388281c3074dee918c411f61f3624e545f19d6c9a06a83393fee2817b93a8;PASS165hashes/NOT_MET10/0/0shortfall.
- Hard outcomes artifacts/student61609_hard_outcomes_20261007_v1/hard_stt_outcomes.json SHAb03ec11a2932ed317d9ebedbd420342f9ac26551b7a39872f4f99e072e681736;PASS168hashes.
- Original91nonzero hardSTT:15recovered/76stillfailed;2zero-window cases stillfailed,separate. All15STTgains belong to originalhard91.
- STT12regressions=7Collision/1Lost/4Normal-no-success;10haveadmittedteacherwindows,2neitherteacher. Current88teacher-solvablestillfailed=86valid+2zero;not guaranteed recoverable from studentstates.
- Remaining76hard=38Collision/26Lost/12Normal;failuresconcentrated21VLzqgDo317F+11ac26ZMwG7aT butno causal claim.
- Fullmedia4215firstJPEGhashes/video-stream+duration PASS;not every-frame decode. artifacts/student61609_review_20261007_v1/audit.json SHA39d0376cccf925c1735264f686c3908cec67cbb2b9cbebc583c06e3da2b36c73.
- NewHTML18799 ondevpod-4090;remoteandlocalforwardHTML/audit/newvideo200verified19:04. Old18798/18797 unchanged.
- Access:ssh -o ExitOnForwardFailure=yes -N -L 18799:127.0.0.1:18799 devpod-4090 ;http://127.0.0.1:18799/
- TRreference-normalized;52missingreference/taskactualfallback;macroTRseparate. CRtargetdistanceever<.5m,notgeneralobstaclecontact.
- Gym/xFormers/SSDsemanticwarningsretained;specifiedfatalpatterns0. Earlierwrongrun/console.logmonitorclaimwithdrawn;actualtaskconsole/.md-ak/workload used. HistoricalSTTupperbound/partialwatchrecords retained.

## Next bounded work — not a new training job yet
- Newcandidate improvesall3tasks butSTT missesgoal. Do not blindlyrepeatweights/LR/epoch3, or startUWB now.
- Fixed88 replay tool wa/tools/replay_teacher_group_fit.py andtests:13CPUtestsPASS;real88window/176pair/1705sourcehash admissionPASS50.50s,no model loaded.
- Referencefit artifacts/teacher_group_fit_61377_20261007_v1.json SHA1463ea39a6fe1acf90fcbd813d1ae23bee6ee0e475a314e5bde174b3fa4f8158.
- Exactold88identities/order retained,notreselectedby61609failure;40hard/16STTsuccess/16DT/16AT,normal+repeatedhistory. Oldstoredperwindowerrors/firstpoints only;newreplaysavesfull7x4predictions/labels/inputhashes.
- Preparation wa/results/TEACHER_FIT_REPLAY_PREPARATION_20261007.json;toolSHA869e5346fb69cfb189746cedc5b72166b0df11b26f3d35c4465256d7650d1a46;no61609fitpredictionsyet.
- BeforeGPUsection recheck actualidle developerGPU. A800allallocated;nas-h100reachable butseparateNAS requiredcode/env/cache/ckptmissing;no migration. 4090sameGPFS GPU6=340MiB/27%tworeads,notassumedidle;otherjobsuntouched.
- Replayonly61609onfixedwindows;old61377errorsreused,no oldmodelrerun. Theninspect12regressions/76persistenthard withpairedtraces before selecting limited nextsampling/recoveryexperiment.
- Parentforanysupportednewtraining remains59866 model+optimizer SHAab39144b0490937ce43c4ab28e2d90cdb0c560f700d3364a29ce1dc9a08b999d;8GPU,maxcumulative2epochs unlessnewauthority.
- Allteacherselectionsuccess-first,thenhigherTR,tieLightNav. Failed/fallbackbranchesnotcorrectlabels;actualexposurelogged.
- EvalSTT/DT/AT each8GPU,total24;followAGENTS resourceorder/NAS/preflight/frozenconfig,noclustersmoke/duplicatejob.
- ModelJEPA/MetaQuery/ActionExpert/originalloss/physics fixed;RGB+initialGTBBox+idealpolarUWB,notext/laterGTboxes/futurepaths;JEPAaux,noonlineMPC.
- AfterSRgate:explicitimage-mode sameweight UWBablation;existing98CPUmodechecks not actualimagebehavior. Requirefinalweightinterface/observerperturbation/realHabitatchecks/fullpaired4215;notmixedzero ornoUWB-retrainingclaim.
- Preserveallcheckpoints/results/failures;GitHubexistingwa announcebeforeeachpush. No weight/data/video upload. CURRENT<=80,PROGRESS<=150.
- TB6006completed61609curves;ssh -N -L 16006:127.0.0.1:6006 devpod-a800 then http://127.0.0.1:16006/#scalars ;not runningtraining.

## Preserved baseline
- 61377checkpoint:/data/nas_ray/project/md-ak/users/zeying.gong/job_61377/task_72474/wa_dual_teacher_train_a800_v1/checkpoint.pt;SHAb5236a21f2d2695780029503c97e339c8350dc4f7337d05ec7e8692b9b9c327f step59065.
- Baselineartifacts/student61377_full_audit_20261007_v1 summarySHA30dfdbd9263ca0ba3615cb1845938de235c7e1b15b2141edb5ac584622f6bbf3;1276/1173/1203;fullpair/mediaPASS.
- LightNav1273/1128/944,Oracle1281/1210/1226,differentinputsystems;currentWAall3exceedLNbutSTTgoalnotmet.
- Old60502step45900/SHA20cc84b3f231ad4056e16b91c52c84cb14e18d886a80324b5f27ffd773be5331 and3627/4215 preserved;no overwrite.
