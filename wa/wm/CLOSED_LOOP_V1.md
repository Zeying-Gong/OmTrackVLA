# JEPA-WA image-only closed-loop diagnostic v1

User requested a small closed-loop test after inspecting noisy training loss. This is one complete bounded diagnostic, not a cluster smoke or a full1405episode benchmark.

- Fixed STT val scene2n8kARJN3HM; episodes119,25,297,136,130,291,16,40,22,273; seed7 per episode; unchanged simulator and success/following/collision/finish metrics.
- Final JEPA checkpoint from Job59566, step22707, SHA2562cb78751977cff0a18eee887ae58157fa3f78d6ebf1208534c291a95c5035c52.
- Image mode only: RGB history and one-time target bbox. No language input, UWB, later target boxes or simulator target coordinates. Invalid initialization is retained in denominator, never replaced by a later privileged box.
- Same frozen DINO preprocessing and four causal history times[-1.5,-1,-.5,0] as training. WLA64MetaQueries and original SE2 ActionExpert/target head retained. Learned JEPA dynamics is auxiliary-training-only and OFF at inference.
- Original WLA v3 pose_action + predicted-target geometry guard, unchanged, hash checked. Euler4steps with seeded7+step noise, matching the original closed-loop WLA sampler; offline validation used zero starting noise, so those are different evaluations.
- Preserve all ten outcomes and per4step RGB frames. Compare initial-frame hashes to the frozen prior ten-episode official baseline. Text-vs-bbox input difference prevents claiming identical goal interfaces. One scene cannot establish generalization, dynamic crowd safety, or project-level SR/CR targets.

Checks: strict trained-state key matching except externally loaded encoder; two developer GPU predictions covering bbox initialization and subsequent no-bbox frame; Habitat import; compile/shell syntax;13existing regression tests. Initial import collision with checkout/trained_agent.py was corrected by prioritizing the benchmark root. No full job was submitted before correction.

Source implementation: wa/wm/eval_server.py,eval_agent.py,eval_ten.py; launcher wa/scripts/eval_ten.sh. Full run invokes `bash wa/scripts/eval_ten.sh OUTPUT_ROOT` from a clean frozen checkout. Model server and Habitat share the assigned GPU through a localhost-only RPC endpoint, with separate existing Python environments.

Loss interpretation:909logged points are rank0 finalmicrobatch samples(batch2) every25updates, not full32sample DDP averages. All recorded loss/flow/world/gradnorm values were finite. Last100 loss p10/median/p90/max=0.03686/0.08018/0.24695/0.44650. Random flow noise/time, goal mode, and trajectory difficulty can affect sample variance; this is not proof of convergence or effectiveness. The old log cannot be converted retrospectively to true fullbatch means.
