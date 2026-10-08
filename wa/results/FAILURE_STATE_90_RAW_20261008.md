# Failure-state new90 raw evidence audit — 2026-10-08

Status: PASS_NONRELEASE. This is a collection audit, not a new student success rate, cache admission or training result.

Job61844/Task73066 finished successfully at 2026-10-08 11:24:43 Beijing on eight A800 GPUs. All eight original lanes completed their exact frozen90 keys, disjoint from the previously audited36. No trajectories were rerun for this audit.

| Item | Verified count |
|---|---:|
| Searches | 90 |
| Branches | 637 |
| PNG frames / observations | 64,946 |
| Actions | 64,945 |
| Evidence files rehashed unchanged | 75,855 |
| Repeated-success original candidates | 72 |
| No valid teacher recovery | 18 |
| Candidate teacher windows | 5,357 |

The72 original winners comprise52 Oracle and20 LightNav;70 take over mid-episode and2 from the start. Selection follows success first, then original following rate, ties to LightNav, with an independent successful repeat. Repeat trajectories are not extra demonstrations.

The independent numerical audit retains4,973 valid windows and excludes384 using unchanged filters. Together with old36, the numerical totals are7,396 candidate /6,864 valid /532 excluded, across90 nonempty original episodes. These counts are not yet a training release or a deduplicated new-data count: at least182 full consumed-input-and-label-equivalent windows already occur in the old teacher cache. Full same-task/key equivalence audit is pending collection release.

## Evidence locations and exact hashes

- NAS report: `/data/nas_ray/home/zeying.gong/algorithm/repos/WA-Mobile-Tracking-20260928/artifacts/failure_state_completed_search_audit_61844_v1.json`;34,716,790 bytes;SHA256 `7759d30c623c3f6a3b989c4ba0efa3554344daa17e84df662d52650efb334aa1`.
- Companion `.provenance.json`:269,995 bytes;SHA256 `2a27d5198f35feb95a670b3cbcd8be629eb03e634d17ddf82939318572c97dd7`.
- Numerical report: `wa/results/FAILURE_STATE_90_NUMERIC_20261008.json`;SHA256 `4d0bf78345b46b8fb36e969662f7600c48f35370eb151823829ea0c9fb42a776`.
- Immutable collection source commit: `726544449a5e7ad10c7d246b2bf302d783210c7e`.

Both report files were read back and their hashes, schema, original lane order, source stability and nonrelease flags independently checked. The four-CPU audit completed with exit0 in1,193.265 seconds. All24 owned process spawn/exit PID pairs matched; five checked fatal-log literals counted zero across24 logs. Existing warnings are preserved, not claimed absent.

## Error and scope boundaries

One narrowly classified LightNav MissingRVQ error remains: `VLzqgDo317F/123`, takeover34, sequence59;60 observations and59 actions. The fallback was rejected before execution, terminal result is null and usable windows are zero. The missing-final-l2 error category was not encountered in these90 trajectories, so this run does not establish real-trigger coverage for that category.

This audit checks persisted RGB, action, recorded-state and search evidence; it does not prove complete hidden simulator RNG/contact-state equivalence. Oracle privileged state is teacher-only. WA remains RGB + episode-zero GTBBox template + current ideal simulated polar UWB, with no text or future-target input. The work is explicitly evaluation-set adaptation, not unseen-test generalization.

The next step is independent126 collection admission, cache conversion, actual loader checks and deduplicated exposure planning. Best student61609 remains STT1279 / DT1178 / AT1207 out of1405 each; no new checkpoint or student SR was produced.
