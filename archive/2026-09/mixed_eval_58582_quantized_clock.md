> Archived DA3 evidence. All earlier continuation/retraining instructions and live status claims are superseded by the 2026-09-28 retirement decision; Job58613 is STOPPED. See docs/tracking_route_decision.md.

# Quantized action-clock repair and paired evaluation58582

Status: Job58582/Task69330 SUCCEEDED2026-09-28 02:42:48Asia; diagnostic efficacy FAILED. DA3 SR/TR/CR0/43.053129012662275/40;official70/70.8591340940474/0. All20rollouts complete and10initialRGBhashes identical. DA3invalidInit3;remaining7validInit alsofailed(4Collision/3Lost). TRAIN-ONLY expert correction triggered;not implemented/submitted yet. Existing recorder labels executed poses and cannot relabel learner actions as expert corrections;collector needs explicit learner-history/expert-future boundary. Model sample_steps default10 and eval10 agree at source level;checkpointconfig audit pending.

- Submitted2026-09-28 02:30:38AsiaShanghai;RUNNING02:30:43;baidu_bj_a8001actualA800-SXM4-80GB81920MiB driver550.163.01.
- Output: /data/nas_ray/project/md-ak/users/zeying.gong/job_58582/task_69330/mixed_final_paired_ten.
- Snapshot: /data/nas_ray/home/zeying.gong/datasets/da3_mixed_eval_20260928_v3;337codefiles manifestSHA756bdd5a6ecda7394ebf31c45ded2797bbf41ebafff43deb244e9ad2b1a6c71c.
- Config scripts/da3/baidu_a800_mixed_final_paired_ten_clock_v5.yaml SHA65824af55aaf54f92bccf6767514f7cc1be274d78f8fc23396a07b852d15dc61;1GPUK8s14400s. Basecommit9ad4bd1554f97e3a43913bdcd3d4f30af797f6b7 plus recorded uncommitted source;no commit/push.
- Same finalcheckpoint Job58085epoch2SHA b1fb2129e9f38202b43ab1d050c78da9e182d56bfb44023ff6d2dec363e027c6;strictCPUloadPASS.
- Same STTvalSHA8a96fe3be38ab78b0b8985e71645eef337d396be126bbd78ce1c87ffef5a0c63;scene2n8kARJN3HM;IDs119/25/297/136/130/291/16/40/22/273;seed7;both methods rerun. No full1405benchmark.
- DA3 receives initialbbox+missingUWB;official RGB+text. Official controller/model/metric paths unchanged;only DA3 real-time interface repaired. Model architecture/loss/label times unchanged.

## Verified clock repair

- Developer32zero-action trace proved task1/60s plus4simticks of0.008s produces floor(n*73/12) cumulative ticks;actions12and24have0.056s,others0.048s.
- QuantizedActionClock v4 uses Fraction-derived task+sim tick schedule and predicts NEXT interval from cumulative ticks;no fixed average and no hardcoded30/40Hz control duration.
- Runtime physics step/acratio/ctrlfreq/taskSPS must match measured record. Every observed elapsed world time must match cumulative schedule;unexpected/missing ticks still raise.
- 23existingcontrol/nativeHabitat +16calibrated/quantized clock testsPASS (39total). Tests cover300steps, wrongticks,firstaction,reset,params,unchangedlabels andagentASTwiring.
- Bounded developer real finalpolicy run:episode119,48actions,validinitialbbox,48realmodelpredictions,finiteactions,no policyfailure. Each predicted interval matched actualworlddelta;extra0.056ticks at12/24/36/48,maxerror1.8041124150158794e-16. Not a benchmark result.
- Evidence:preflight/{controller_tests.log,clock_tests.log,clock_quantization_trace.json,policy_clock_119.json};source_v4.sha256 and checkpoint_ready.json bind hashes. Worker rechecks admission, source/dependencies/scene metadata/calibration/config/finalweights.
- Old failed58558missingcwdmetadata and58575constant-clock remain intact with logs/configs;no metrics derived from them.

## Remaining acceptance

- Monitor58582 scheduler/logs/NAS and verify all20rollouts,summary/pairedCSV,initialRGBmatching10pairs,SR/TR/CR andfailuretrajectory review.
- If poor, diagnose and useEVT TRAIN ONLY on-policy expert correction/aggregation+formalretrain;samepairedretest;no validationlabels/rollouts in training.
