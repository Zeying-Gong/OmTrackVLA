# WA current task — STT target and UWB study
Updated: 2026-10-08 Beijing. Goal ACTIVE; heartbeat wa ACTIVE every20minutes.
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

## Active bounded work — accepted STT anchor61715; full24GPU evaluation queued
- Realfixed88replay completed onidledevpod4090GPU6:597.856s,1.482GiB,88windows/176predictions/1713sourcehashes PASS. No training/rollout.
- Artifact artifacts/teacher_group_fit_61609_replay_20261007_v1.json SHA3db61f67a648258593382dd2adbfd508a2d8dbe0aeb077648b0e8b51f54d7f73;reference1463ea39 fixedidentities/order unchanged.
- NormalhistoryADE61377->61609:hardCollision.487173->.467611,hardOther.394402->.383305,successSTT.252272->.255145,DT.290127->.310269,AT.400361->.371023. Smallselectedsample/notSR;DTfitworsebutfullSRup.
- Boundedtrace+exposure report wa/results/STUDENT61609_DIAGNOSTIC_20261007.json. Main24new/oldtracepinsPASS plus48info/reviewhashes. Firstflattenedinfo-pathvalidatorfailed;actualscene/episode_info path corrected,originalresultsunchanged.
- 12regressions:10successfulteacher raw972->valid886,early357,eachactual1;2bothfail no labels. Original transient986sum corrected. Noneinoriginalhard91.
- Hax/16 finaldistance.981m failsunchanged1m lowerbound despiteTR1;VLzq/197,/238 forwardcommands withlittlemotion;Vt2/123 andac26/258 visibilitynotrecovered. Notall failuresdoorframes.
- VLzq/238 has149executedactions/148infologs:Lostbreak occursafterstepbeforeinfo-write;lastpostunlogged. info.facing=distance+detectorcombinedmetric,notpurevisibility.
- Persistent76:38Collision/26Lost/12Normal;32terminate<=40allCollision. All38last5meanlongitudinalnegative;18Lost+6Normal forward>0.1 butactualhorizontalpathspeed<0.1m/s. No contactgeometrycausalclaim.
- Fixed-budget candidateA generated/CPUchecked independently:base+allteacheronce;3270earlyhardextra2,7143latehardextra1,886regressionteacherextra1,6257balancedextrasfor1248oldsuccessSTTepisodes. Totalextra20826 unchanged.
- Newv2candidate artifacts/stt_anchor_candidate_61609_20261007_v1 reportbbe21c78;canonicalplan42d33d92. 14CPUtests mainPASS;24111sourcehashes/rederivedeligibility checked;8ranks148034/1184272simulated,notactualtraining.
- Anchorpoolmin6validwindows;17six-windowepisodeallocationsmin30;6257uniqueanchorindices. SametaildropAT/Oracle424632;hard24096early9810late14286;gains3032/persistent21064/regression1772/stableanchors122679.
- Preparationwa/results/STT_ANCHOR_PREPARATION_20261007.json;builder2d011cd7 tests55b5e73e. Initialbinaryfloatoutcome typegatefailedbeforeoutput;strict0/1compatibilityfixed+testsPASS,notdata/trainingfailure.
- v2runtime implemented/strictadmissionPASS;v1order/rejections unchanged. Main66CPUtestsPASS0.387s andfull8rankactual-index/DataLoaderPASS68.13s;notfullimageconsumption.
- PreserveDT/ATdata andold15gains;anchor5/6uniqueagequantiles pereligibleepisode fixedSHAorder,notchosenbynewfit. Insufficientvalidwindows mustfail,no label/filterrelaxation.
- MainshortGPUdev4090GPU6 PASS16samples/4updates22708..22711;peak7.309GiB,train+val18.828s afterslowNAS/modelsetup. All4groupsconsumed;no savedcheckpoint. artifacts/stt_anchor_short_gpu_20261007_v1.
- Runtimepreflight wa/results/STT_ANCHOR_RUNTIME_PREFLIGHT_20261007.json;66CPU+realshortGPU+v2fullsizesimulationauditPASS. CodebackedGitHubwa dd1fec9e2cba0450be7a33387c8a2e550db4cd91.
- ACCEPTED61715/72909:2026-10-08T01:25:14Beijing schedulerJob+TaskSUCCEEDED/workerCOMPLETE/all3modes73368. Finalstep59716,new37009updates,cumulative2epochs. Actual1184272=base726631+teacher457641;fourgroups9810/14286/1772/12514 and8ranks148034each independentlyverified.
- source_stt_anchor_train_v1 clean175Python dd1fec9e;configwa/jobs/stt_anchor_train_a800_v1.yaml SHA4a5b465a9533c779a7990aa754575caee34dc302fa0238ecd62d3471cb3879ef. ReleaseSTT_ANCHOR_RELEASE_20261007.json/configGitHubb240ebd4.
- Checkpoint:/data/nas_ray/project/md-ak/users/zeying.gong/job_61715/task_72909/wa_stt_anchor_train_a800_v1/checkpoint.pt;SHA9631778c81992a349f8315b773f763217a5f387b5206b1bfb97b323839030a6a;no overwrite/duplicate.
- Finaltraining/evalrelease:wa/results/STUDENT61715_RELEASE_20261008.json GitHubd5c0c325;startupwa/results/STUDENT61715_EVAL_STARTUP_20261008.json. Priorstartup/optimization/validationwatches retained;no training restart.
- Frozenaudit_stt_anchor_training PASS24596hashes/actual8rankexposure/parentANDoptimizer/LRloss/heldout;artifacts/stt_anchor_training_audit_61715_v1.json SHAd91dbb222847a30ddb843f895d3cdc84be17a98c871fb1f0f97a5212859aaa80. Auditusedtraindd1fec9e release,notlatercheckout.
- TB stt_anchor_61715:00:45:48 local16006home/runs/tags/loss4HTTP200/latest59700/allreturnedfinite/11tags/no validation scalar. NewagentSSHclosed,remote6006/PIDnotreverified;existingmainobserverlive,no restart. Earlierfullhealth23:52 retained. ssh -o ExitOnForwardFailure=yes -N -L 16006:127.0.0.1:6006 devpod-a800 ;http://127.0.0.1:16006/#scalars
- FinalofflineADE/FDE image.26574965/.46344470 point.25376165/.44215970 mixed.25379827/.44177998;73368each/SRnull. No new61715closedloopresult;last59700lognormal25stepcadence.
- Initialization remains59866 model+optimizer SHAab39144b0490937ce43c4ab28e2d90cdb0c560f700d3364a29ce1dc9a08b999d;1new/cumulative2epochs;not61609epoch3,noLR/loss/controllerchange.
- Reserveoption ifevidenced:boundedstudent-failure-stateOracle/LightNavrecoverycollection,success-first/higherTR/tieLightNav;failedprefix/fallbacknotlabels. Notstartpointteacher successassumed recovery.
- EvalSTT61792/72986 DT61793/72987 AT61794/72988 submitted01:51:39Beijing baidu_4090 each8requested/full1405(total24/4215),NOTduplicates. At01:58:52 allPodsPending/node-/ready0/8/completed0/1405;FailedScheduling11CPU+16GPU+1affinity+1CItaint. Aggregate72free notnodefit;noSR/actualGPUclaim.
- Evalfrozen source_student61715_eval24_v1 clean5a23a982c7ef01f7fdec58faf27f6ea623ed9eeb;114CPU/10dependencySHA/23unchangedfiles/2CPUimports/24realHabitatlanes9b40d627/realweightinterface25fecf25PASS. Configsstudent61715_{stt,dt,at}_4090_v1.yaml backedbeforeonce3fulljobs;no cluster smoke. Monitorwa updatedsame20minthread.
- AfterSRgate only:explicitimage-mode sameweight UWBablation afterrealinterface/observerperturbation/Habitatcheck;notmixedzero ornoUWB-retrainingclaim.
- Preserveallcheckpoints/results/failures;GitHubexistingwa announcebeforeeachpush. No weight/data/video upload. CURRENT<=80,PROGRESS<=150. Goal/20minmonitor remainACTIVE.

## Preserved baseline
- 61377checkpoint:/data/nas_ray/project/md-ak/users/zeying.gong/job_61377/task_72474/wa_dual_teacher_train_a800_v1/checkpoint.pt;SHAb5236a21f2d2695780029503c97e339c8350dc4f7337d05ec7e8692b9b9c327f step59065.
- Baselineartifacts/student61377_full_audit_20261007_v1 summarySHA30dfdbd9263ca0ba3615cb1845938de235c7e1b15b2141edb5ac584622f6bbf3;1276/1173/1203;fullpair/mediaPASS.
- LightNav1273/1128/944,Oracle1281/1210/1226,differentinputsystems;currentWAall3exceedLNbutSTTgoalnotmet.
- Old60502step45900/SHA20cc84b3f231ad4056e16b91c52c84cb14e18d886a80324b5f27ffd773be5331 and3627/4215 preserved;no overwrite.
