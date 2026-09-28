> Archived DA3 evidence. All earlier continuation/retraining instructions and live status claims are superseded by the 2026-09-28 retirement decision; Job58613 is STOPPED. See docs/tracking_route_decision.md.

# Calibrated v3 evaluation attempt: Job58575 / Task69322

Status: FAILED_NONCONSTANT_ACTION_CLOCK; no valid paired metrics.

- Submitted2026-09-28 02:09:45AsiaShanghai;FAILED02:10:33. 1actualA800-SXM4-80GB driver550.163.01.
- Output: /data/nas_ray/project/md-ak/users/zeying.gong/job_58575/task_69322/mixed_final_paired_ten.
- Frozen candidate: /data/nas_ray/home/zeying.gong/datasets/da3_mixed_eval_20260928_v2.
- Source336files manifestSHA c7e838a405ad7d1208a260160fbea44565a0242af7d6306e90064bb1242fc6ce;only reviewed agent/controller change plus newtests relative to prior335.
- Config scripts/da3/baidu_a800_mixed_final_paired_ten_clock_v4.yaml SHA050667c77f3a9c236848cfb5bb4f47e6330a19dd9d9fc8533764e6d1bb9d329a.
- Final training checkpointSHA b1fb2129e9f38202b43ab1d050c78da9e182d56bfb44023ff6d2dec363e027c6;679878steps2epochs unchanged.
- Independent developer calibration measured8zero-action transitions:all0.048;taskSPS60,physicsStep0.008,acRatio4,ctrlFreq40.
- v3CalibratedActionClock validates calibration fields/timestamps, runtime parameters and every actual world delta. 23old+8newCPU/native testsPASS;separate developer scene3zeroactionsPASS. Receipts/logs in preflight.
- These narrow checks missed variable action durations in longer execution. Formal job initialized scene and GPUmodel, then strictclock raised: World clock delta0.05600000000000005 differs from action duration0.04800000000000003.
- No completed episode;official not started. This is not a policy SR/TR/CR result and does not trigger DAgger.
- Preserve v3snapshot/admission and both failed jobs. Do not weaken assertion or reuse constant8samplecalibration as proof.
- Suspected cause:task1/60s request quantized to0.008s physics ticks with accumulated residual, periodically producing an extra tick;needs long raw-clock evidence before choosing corrected interface.
- Completed bounded developer32zero-action trace via tools/measure_mixed_eval_clock_long.py: actions12and24have0.056s,others0.048s;times0->1.552s. Matches cumulative fixed-tick schedule floor(n*(1/60+4*0.008)/0.008)*0.008, not a constant mean. Output preflight/clock_quantization_trace.json/log. No training labels or benchmark results generated.
- Next: derive/validate variable-clock contract from actual source andtrace;test full quantization cycles and real policy closed-loop on developer before another full paired submission. Keep model/loss/labels/official controller/fixed10IDs/seed/metrics unchanged.
