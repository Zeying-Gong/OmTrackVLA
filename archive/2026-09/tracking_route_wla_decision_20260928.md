> Historical diagnostic/decision snapshot. Latest retirement and verified cancellation: docs/tracking_route_decision.md; earlier pending/running instructions below are superseded.

# Tracking route decision — 2026-09-28
User instruction: 简单对比，不行就直接采用 WLA.
Decision: WLA is the primary tracking route. DA3 is deprioritized; do not automatically launch pending DA3 correction retraining or further formal DA3 experiments. Preserve checkpoints, raw collections and failed results. Already running collection Job58613 is not cancelled by this record; no new DA3 training authorized by this comparison.
Evidence: DA3 Job58582 vs completed WLA Job58346 and candidate Job58638. Identical scene2n8kARJN3HM and ten episode IDs119,25,297,136,130,291,16,40,22,273. All initial RGB hashes match by direct comparison.
DA3 success0/10 collision4/10; WLA prior and candidate success9/10 collision0/10. Valid-DA3-init subset: DA3 0/7 vs WLA6/7. Inputs differ: DA3 initial bbox+RGB (UWB missing), WLA text+RGB; heads/controllers/training also differ, so NOT a backbone-only ablation or proof DA3 cannot work.
Completed WLA Job58346 full unique coverage1405/task: SR STT80.2846975088968,DT53.30960854092527,AT49.25266903914591 percent.
Candidate Job58638 remains RUNNING: STT1405 complete SR81.28113879003558; DT/AT incomplete at comparison. Do not declare candidate superior until full results available. Use completed WLA checkpoint as established reference.
This route selection does not silently solve bbox-only/UWB deployment requirements; target interface conversion remains separate work.
