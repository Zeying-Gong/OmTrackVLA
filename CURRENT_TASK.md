# WA current task — recovery demonstration round complete

2026-10-02 07:24+08 ROUND_COMPLETE / GATE_FAILED: training60502/71381 SUCCEEDED06:59:14; step45900 cumulative2epochs; image/point/mixed offline validation73368windows each COMPLETE. MixedADE0.258341 FDE0.451232 (parent0.262677/0.458386); offline only. Closedloop60509/71388 SUCCEEDED:15/24 vs parent14/24;3gains2regressions;collision3/24both;invalid0;24initialRGBmatch;STT6 DT5 AT4. Below80%gate;noepoch3/RL. 24newvideos and72pairedpage18791 ready;TensorBoard final metrics exported. Heartbeatwa PAUSED confirmed by app. Overall research not achieved; next bounded analysis of gains/regressions before further training. No realUWB/edge/generalization validation.

## Verified protocol
- JEPA / MetaQuery / ActionExpert; RGB + polar simulated UWB; no text. Original losses and controller unchanged.
- 971 valid teacher windows repeated16 times +726631 original windows; recovery exposure2.0933%. Failed branches/prefixes not positive labels.
- Parent59866 model AND optimizer restored; one additional epoch; cumulative2epochs. Source64e90c8eb3099299c2abcb1c194349b6b8a4b3fc.
- Final checkpointSHA256:20cc84b3f231ad4056e16b91c52c84cb14e18d886a80324b5f27ffd773be5331.
- Fixed24 development evaluation; unchanged heldout; ideal simulatedUWB noise0 latency0.

## Artifacts and remaining work
- wa/results/recovery_training_60502.json: final three-mode offline metrics.
- wa/results/recovery_eval_60509.json: paired closedloop outcomes.
- Gains:SByzJLxpRGn/8 all3tasks; regressions:DT VLzqgDo317F/89 and AT auFeVz9Go4m/10.
- Next: examine these paired trajectories and collision geometry; do not infer significance from net1 success or blindly continue training.
- This round completed; >=80% development gate FAILED; independent confirmation/generalization and realUWB remain UNVERIFIED.
- Historical task notes:archive/2026-10/CURRENT_TASK_before_recovery_round_close.md.
