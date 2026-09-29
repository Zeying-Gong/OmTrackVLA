# Epoch2 result: debugging gate failed

Job59826/70688 completed all24 fixed development episodes on2A800 at2026-09-30 05:42:21+08.
Checkpoint59791 step45414/2epochs; immutable source9a8534a0/config75654321.
Same mixed_zero input, original learned_target_guard_v3, simulator ideal instantaneous UWB, no text.
Initial RGB hashes match baseline59726. All initializations valid. Confirmation scenes remain unused.

| Condition | STT SR / CR | DT SR / CR | AT SR / CR | Successes |
|---|---|---|---|---|
| Epoch1 |87.5 /12.5|75 /12.5|37.5 /12.5|16/24|
| Epoch2 |75 /25|62.5 /12.5|37.5 /12.5|14/24|

Percentages; only8 episodes/task, not statistically strong generalization estimates.
Do not extend epoch3 or adopt epoch2 as improved. The A800/4090 training hardware differs; continuation restores AdamW but not original RNG. This is not a controlled architecture comparison.
Current audit finds reverse and turning labels present, not absent. The cached action targets are actual-time interpolated XY and relative yaw at0.1..0.7s. Current inference history uses the same floor-search rule as training (not nearest-neighbor). No timing/sign root cause has yet been established.
Next: quantify near-target failure-state coverage and compare predicted versus teacher trajectories; avoid arbitrary controller changes after measured-heading regression59737.
Training59791 is still doing offline validation at05:50; no offline validation completion claimed here.
