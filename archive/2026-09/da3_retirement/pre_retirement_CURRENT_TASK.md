> Historical snapshot before DA3 retirement; instructions and RUNNING states below are superseded.

# Latest user decision — 2026-09-28
Tracking primary route: WLA after matched-ten comparison (9/10 vs DA3 0/10, identical initial RGB).
This supersedes earlier automatic DA3 correction-retraining plans: do not launch further DA3 training without renewed user direction.
Preserve existing artifacts; running collection was not stopped. WLA final candidate evaluation remains in progress.
Evidence and input-interface limits: archive/2026-09/tracking_route_wla_decision_20260928.md.

# Current Task

Status: MIXED_TRAINING_COMPLETE_VERIFIED;paired58582 efficacyFAILED SR0/TR43.0531/CR40 versus official70/70.8591/0;heldout58633SUCCEEDED1003ep78958windowsADE0.751140FDE1.274613mIndependentAuditPASS;14:58correction58613/69363RUNNING8A800;STT_COMPLETE6603attempted6544raw59initRejected202779windows0unexpectedFailures;16of16shardsAndTaskMarkerVerified;AT_COMPLETE6603attempted6543raw60initRejected241709windows16of16shardsAndTaskMarkerVerified0unexpectedFailures;DT_RUNNING1847raw6initRejected58281windows5588boundaries16activeWithin75s0unexpectedFailures. Correction release/train code_v1 FROZEN374filesSHAa5b5e5d55836fb80be59f377c0e173a1b550998dd54d9a5689930a7f36f88dc3;41testExecutionsPASS;fullDataReleaseTrainingAdmissionRetrainAndPostTrainValidationPENDING;no newrelease/trainjobs;see archive/2026-09/mixed_failure_correction_plan.md.

## Authorized objective

- Train the existing DA3 person-following policy on substantial robot-trajectory data: EVT AT/DT/STT, SAGE3D, and eligible TPT.
- All three EVT source tasks: 7,257 each, three teachers in order LightNav -> official 0.6B -> oracle; 65,313 planned attempts, NOT qualified demonstrations.
- The ten-episode limit is only for downstream paired evaluation. Do not launch the full benchmark automatically.
- No scheduler smoke jobs. No architecture replacement, auxiliary heads, target-trajectory labels, commit or push.
- Post-training authorization2026-09-26: verify Job58085 completion/checkpoint -> fix/verify controller timebase -> paired10 EVT validation vs official with frozen IDs/seeds/metrics -> diagnose failures -> if poor, EVT TRAIN ONLY expert-correction/DAgger-style collection + aggregated training -> same paired validation. Validation trajectories/labels never train; preserve train/internal-heldout scene separation; no automatic full benchmark or premature retraining. Implementation and post-training evaluation remain PENDING;current Job58085 unchanged.

## Active training

- Job57528 / Task68138: SUCCEEDED2026-09-25 23:31:54 Asia/Shanghai;4A800;195008updates/2epochs;training elapsed36195.77s;not a mixed-source or generalization success.
- Config: scripts/da3/baidu_a800_evt_joint_train_4gpu.yaml;4ranks,batch2/rank,2complete epochs,lr5e-5,seed7.
- 780,025 unique train windows/9,657 robot trajectories/625scenes; AT1,792/145,894windows; DT1,985/162,471; STT5,880/471,660.
- Natural full-window coverage, no task subsampling:97,504 steps/epoch;195,008 updates total;3explicit DDP padding repeats/epoch.
- Teacher selection:8,044LightNav/1,468official/145oracle train rollouts, best successful collision-free valid-init teacher per task at release time.
- Heldout1,003episodes/78,958windows/71scenes, disjoint across ALL tasks; all three benchmark validation sets excluded.
- Immutable Baidu data: datasets/evt_joint_release_20260925_v1/release_v2. Train SHA bdc2210e620feabd5737d23a338563a2d87cb21ffe58e323ea0e74c2a301913d.
- Immutable code: datasets/evt_joint_release_20260925_v1/code_v2; source manifest SHA929211dc0f9ab858726f6ceaf1e91ed3de5ceae4f9c4c5a7724657c21855574a.
- Initialization: Job57254/Task67822/da3_teacher_train/training/epoch_3.pt; model warm start with fresh optimizer, not resume and not old SAGE weights.
- Output: /data/nas_ray/project/md-ak/users/zeying.gong/job_57528/task_68138/da3_evt_joint_train.
- First optimizer step loss0.1648967564;all4ranks five required gradient groups finite/nonzero;adapter deltaL2=0.00137289497;step_00001.pt saved. Step189loss0.14177558,elapsed42.45s. Initial throughput suggests10-12h,not a completion guarantee.
- SAGE and TPT are NOT in this immutable release. This is a substantive intermediate training run, NOT completion of the full joint-data objective.

## Data preparation and evidence

- Job57443/Task68042 SUCCEEDED07:16:54; full release audit including every robot waypoint interpolation, file existence, timestamps, source identity and scene separation.
- Job57441/Task68040 FAILED: successful rollout summaries did not guarantee a valid init bbox. Fixed eligibility-before-best-teacher selection, retained failed code/release/logs.
- Developer CPU checks PASS: real AT/DT/STT2x32 batches, UWB present/missing, new vectorized audit agrees with original; strict warm-start loading6,683,011 trainable parameters; sampler, hashes and shell/Python syntax.
- Canonical teacher_data.py was empty; restored exact verified Baidu implementation. Frozen previous runs unchanged.
- SAGE materialize/index57369 SUCCEEDED:912archives,34,658sequences;34,643metadata-valid,15rejected;2,147,582potential windows, not yet released.
- SAGE identity audit57442/68041 SUCCEEDED07:21:28;16CPU/0GPU;912archives;34,631identity-audited,12identity/video audit rejections,15metadata rejections. Audited count is NOT bbox-pass count.
- SAGE bbox outcomes:34,103automatic unique crop matches PASS;528low/ambiguous matches quarantined;11video/header failures and1uninformative crop;912scene groups.
- SAGE original identity/RGB audit outputs remain immutable/quarantined. New source-attested candidate overlay: artifacts/sage_scene_contract_20260926_v1/candidates.json;not a training manifest. No prior rejected sequences promoted.
- TPT57444 curl18 failure retained;57560/68170 STOPPED14:59:23 for bounded parallel download and removing RGB dependency;68171 CANCELLED before execution. Prefix5,877,964,936bytes and all logs retained.
- TPT57565/68177 STOPPED15:09:27 after curl18/56 and within-chunk progress loss;RGB complete/prefix/chunks/logs retained. Fixed strict Content-Range + intra-chunk append;mockdisconnect resumed offsets11->12 PASS.
- TPT57567/68179 SUCCEEDED2026-09-26 00:27:44,baidu_a8000GPU. Officialrosbags.zip11054121078bytesMD550c17ba7fdb96467123af32575d0eb5dPASS;0015.bag extracted;6729panorama/6587odom/13176TFmessages;242368repeatedPathposes0conflicts;notfull48.
- TPT RGB COMPLETE:6062/6062 decoded at1920x960;exactGTmatches6062;5334visible/validboxes;duration223.484s;median dt0.033631576s. Artifact job_57565/task_68177/tpt_processing/rgb_audit/{complete.json,frames.csv,index.html}. Not robot supervision or full48 release.
- TPT finalalignment:6062RGBexactpanoramaMatches/allodomCovered;6050with0.7sfuture;nearestodommedian6.610ms/max34.448ms;odommaxgap82.958ms. scripts/da3/audit_tpt_alignment.py SHA38048e068df5dfc93761e3c928c752d66f4bd6355c5d8b99698f1ecb50226383;job57567/task68179/tpt_processing/alignment_audit_v2/{summary.json,alignment.csv}.
- SAGE full RGB57562/68174 SUCCEEDED15:23:07;all912archives/34658records verified by hashes+orderedUIDs,16workers.34103PASS/10230900frames;555prior rejects skipped;no newdecodeFAIL. AT847/DT1161/STT32095;2114386potentialwindows. Summary artifacts/sage_full_rgb_20260925/verified_summary/{summary.json,episodes.csv};training_eligible=0,phase/scene gates remain.
- SAGE source inventory57581/68193 SUCCEEDED16:28:18;all912archive paths+inventorySHA verified;21304828members;0source candidates/0unsafe names/0failures. Output datasets/sage_archive_source_inventory_20260925_v1/summary.json. No collector source recovered;three representative versions also lack explicit RGB phase/timestamps. RGB audit not rerun;training eligibility remains blocked.
- SAGE trajectory quality57600/68214 SUCCEEDED16:40:56,16CPU0GPU;34103diagnosed/555priorRejected.10196797transitions:3007672stop(<.05m/s),5215329turn(>.1rad/s);0grossJump(>.5m),maxstep.1201637m,0poseChainGap,0sourceRecovery/Collision;allsourceSuccess1.96.7901%frames distance1-2m. CSV/JSON datasets/sage_trajectory_quality_20260925_v1. Diagnostic only;no filtering changed;phase/scene gates remain.
- SAGE source USER_CONFIRMED2026-09-26: FLUX sage3d03d7f62c8bdc. Proceed with data-declared30Hz,record-i robot_pos/robot_yaw,zero RGB shift;do not request another exporter. Runtime render latency and exact historical source diffs remain unmeasured,not claimed PASS. Adapter now rejects nonfinite clock/pose and brokenpre chains.9CPUtests+9real version/task sequences27windows PASS;artifacts/sage_record_contract_20260926_v1.
- Full48-sequence TPT RGB/ODOM still not acquired. Local Firefox rosbags tabs were located; usable authenticated download listing was not obtained. Do not request the same URLs again.
- STT57163/67651 SUCCEEDED10:23:17;collection_complete.json=ALL_SOURCES_ATTEMPTED. AT57320/DT57321 not changed in this training migration;existing data/releases untouched.
- STT57254 baseline SUCCEEDED:76,170steps/3epochs/13,951.9s; loss2.49922->0.16240; five required gradients and actual adapter update verified. This is NOT final joint training or generalization.

- Raw TPT image export to conversation denied by auto-review;requested explicit user permission for a few visual checks. No workaround;numeric/CPU processing unaffected.
- Monitoring2026-09-28: Job58085/Task68820 SUCCEEDED2026-09-28T01:27:12AsiaShanghai;679878updates2epochs4ranks;elapsed123467.9903s;epoch_2.pt161560189bytesSHAb1fb2129e9f38202b43ab1d050c78da9e182d56bfb44023ff6d2dec363e027c6;strictCPUloadPASS;generalizationUNVERIFIED. Next: paired10 EVT validation and conditional TRAIN-ONLY expert correction/retraining + retest. Continue30minute reports and prompt failure/completion reports via current Goal, not an independent notifier. Historical15:20/15:50Asia monitoring gap retained; no training restart.

## Protocol / unresolved acceptance

- Persistent initial RGB+3x3RoI identity;32causal visual frames with real times/masks; online DA3-SMALL/L11;4sparse tokens/frame; existing Transformer and7-waypoint diffusion decoder.
- Optional polar UWB(range,bearing),valid/age; GT-derived input explicitly simulated, dropout.5/range noise.05m/age0.
- Predict ROBOT future positions in current robot frame(x forward,y left), times0.1..0.7s. Origin only added at controller boundary.
- No later GT bbox or future visual observations in policy. No GRU/cache-only substitute. Main trajectory loss only.
- User explicitly authorized A800/H100 instead of4090 this turn. Old57445/68044 STOPPED13:26:48 before training,logs retained. New4A800 preserves globalbatch8 and195008updates;2A800 fallback remains unused. No cross-NAS dataset copy.
- Final57528 rank0 loss0.10327129;training/complete.json confirms195008steps;epoch_2.pt SHA2563ca2b3ac99b47091fb551c1d1d1c67c2e19d6709d67dc8e1b82e4c8bab29f63c. SAGE/TPT NOT included;generalization UNVERIFIED.
- TPT geometry improved:official pinned ZED->theta extrinsic + bagTF yields complete base/panorama chain (roundtrip1.11e-16);6587odom/TF pairs agree exactly. Physicalcartreference/clockoffset/sourcequality remain UNVERIFIED;no training release.
- New TPT diagnostics:2017windows/32frames/7points0.1-0.7s,31startupPadded,87speedReview(>2m/s;notfilter);allfinite;6CPUtestsPASS. scripts/da3/audit_tpt_geometry.py;Baidu artifacts/tpt_geometry_20260926_v1 HTML/CSV/JSON;training_eligible=false;notmodelpredictions.
- Verified visualization: ssh -N -L 18772:127.0.0.1:18772 nas-a800 then http://127.0.0.1:18772/. Full48TPTraw unavailable;no newtraining/eval/commit/push. SAGE next: frozen release integration from source-attested candidates,not another source request.
- Learned Who/generalization remain UNVERIFIED. Evaluation adapter native_timed_lookahead_v2 implemented in canonical source only;legacy default preserved;actual physics step*substeps checked against world deltas and native1/ctrl_freq integration;explicit Euler without midpoint translation rotation.23CPUtestsPASS incl nativeHabitat andagentclock/reset;artifacts/controller_timebase_20260926_v2/cpu_tests.log onBaidu. Native controllerSHAa041e98df6e6138dc8605518bc79f6ae1bc402c4f0e29fcc4d1fa7d11d39c08a. Scene-clock/closed-loop checks PENDING;no newevaluation job;frozen training unchanged. See archive/2026-09/controller_timebase_audit_20260926.md.
- Canonical nas-h100 /data/nas_ray/home/zeying.gong/algorithm/repos/OmTrackVLA,base9ad4bd1554f9+dirty. Baidu isolated execution snapshot unchanged outside scoped DA3 work; no new cross-NAS dataset copy.
- SAGE scene contract PASS:34103candidates/912canonical scenes;train31282episodes1939484windows831scenes;heldout2821/174902/81;zero partition leaks/suffix aliases. All6EVTdefinitions checked:703train/101val scenes andzero train/valoverlap. DocumentedInteriorGS(handcrafted) versusHM3D/MP3D provenance supports source-family separation,not mesh-hash identity proof. artifacts/sage_scene_contract_20260926_v1/{summary.json,candidates.json,input_sha256.json};candidateSHAd1c874d3b21f66dce27a96c4d6cabf7b9b5ff1ccc5373c3e73013c7bb9e6f439;training_eligible=false until release integration.

## 2026-09-26 frozen release / mixed integration

- SAGE Job57918/Task68614 SUCCEEDED10:50:31;16CPU0GPU. Frozen datasets/sage_training_release_20260926_v1:train31282/1939484windows/831scenes;heldout2821/174902/81. TrainSHA3cf1b20cefcf6cf941b984021d8cb080ace3fdc38325f3c9fedc29c52574216c;heldoutSHAe365231293977b4e995590546ddd5176581bd14ea6dd84fd9b42f1555a404280. User-attested30Hz/same-record;no instrumented render-phase or physical-collision claim.
- da3_policy/mixed_data.py and train_sage.py --dataset-format mixed added;natural full-window coverage, existing model/loss unchanged.9mixed+9SAGE+5freeze CPU tests PASS;real frozen train/heldout adapter check PASS2x32x3x224x224 and2x7x2. Logs artifacts/sage_training_release_20260926_v1. Historical partial state superseded below: real joint batch now PASS on Baidu;full manifests await all transfer receipts.
- User approved minimal cross-NAS copy. TRANSFER_RUNNING:68209files/62851248760bytes including manifests,Aliyun->Baidu datasets/sage_mixed_transfer_20260926_v1.16 disjoint SSH streams through local control without local data storage;per-file SHA256;existingfiles verified/reused;failedpartial files retained. Direct devpod SSH auth unavailable;use existing control-machine SSH identities,never copy credentials. Sequential metadata scan and multiplex session bottlenecks repaired;no source data altered.
- Mixed code snapshot: Baidu datasets/da3_mixed_release_20260926_v1/code;canonical source remains Aliyun OmTrackVLA9ad4bd1+dirty.9mixed+16model/temporal CPUtests PASS on Baidu;YAML/shell/compile PASS;strict57528epoch2 warmstartPASS195008steps/6683011trainableparameters. Config scripts/da3/baidu_a800_sage_evt_mixed_4gpu.yaml:4A800/globalbatch8/2epochs/lr5e-5/679878updates/3DDP repeats per epoch,freshoptimizer. No model/loss change.
- Next: finish and verify mixed Job58085/Task68820; evaluate the frozen final checkpoint on matched-ten EVT validation after controller-timebase verification. Conditional expert-guided EVT train-set aggregation/retraining is authorized by the latest user instruction; prerequisites and leakage constraints above. TPT remains quarantined.

- 2026-09-26 15:11 Asia/Shanghai: migration16/16 COMPLETE68209files62851248760bytes;full mixed release andCPU_MIXED_PREFLIGHT_PASS. Job58085/Task68820 RUNNING2026-09-26 15:09:02AsiaShanghai;submitted15:08:57;baidu_bj_a8004actualA800-SXM4-80GB driver550.163.01;2epochs/globalbatch8/lr5e-5/seed7;339939stepsPerEpoch679878totalUpdates;freshAdamW modelOnlyWarmstart57528epoch2;all4ranks fiveRequiredGradientGroupsFiniteNonzero;firstMeanLoss0.1858642101;visualAdapterDeltaL2=0.001382117742;step_00001.pt161562802bytes;latestObservedStep471loss0.1047398448elapsed95.15s;bothSAGEandEVTobservedInLogs;generalizationUNVERIFIED. Output /data/nas_ray/project/md-ak/users/zeying.gong/job_58085/task_68820/da3_sage_evt_mixed;full release /data/nas_ray/home/zeying.gong/datasets/da3_mixed_release_20260926_v1/release. Frozen108file sourceSHA35ed25a1148efa94aeb1629189d2cc863614a9290366c56ca5dac98a82c3daea;do not edit. Next monitor full679878updates andcheckpoint completion;then authorized paired10 evaluation only after controller-timebase validation. Details archive/2026-09/mixed_training_58085.md.
- Paired evaluation2026-09-28: Job58582/Task69330 RUNNING02:30:43Asia;1A800,4h;fixed10IDs/seed7/STTval andofficialrerun unchanged. New337file snapshot datasets/da3_mixed_eval_20260928_v3/code sourceSHA756bdd5a6ecda7394ebf31c45ded2797bbf41ebafff43deb244e9ad2b1a6c71c;native_timed_lookahead_v4 uses cumulative physics ticks.39CPUtests+48realpolicyactionsPASS incl extra ticks12/24/36/48. NoSRTRCRyet. See archive/2026-09/mixed_eval_58582_quantized_clock.md.
