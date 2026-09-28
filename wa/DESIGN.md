# Design decisions and open questions

USS: https://arxiv.org/abs/2606.25880
LightNav: https://arxiv.org/abs/2608.30935
Cosmos Policy: https://arxiv.org/abs/2601.16163
Wan: https://github.com/Wan-Video/Wan2.2

Use USS prompting/latent-dynamics and LightNav history compression/goal-versus-
affordance separation as design references. These do not establish Thor latency
or dense-crowd safety. No code or pretrained weights from these methods are copied.
USS image-space points are NOT metric UWB coordinates. Separate encoders are needed.
LightNav is a Qwen-based VLM; borrowing its representations does not require its
autoregressive language head. Its reference remote server is not edge evidence.

Candidate online backbone: compact pretrained visual encoder plus causal temporal
fusion and trajectory decoder. Concrete encoder selection is still OPEN pending
licenses, weights, identity sensitivity, feature dimensions and device benchmarks.
Wan/Cosmos are candidate offline teachers/research comparisons, not established
drop-in Qwen replacements. A WA model is not automatically language-free.

Inputs: RGB observations in every mode; persistent target crop/bbox for image mode;
metric target range/bearing with valid/age for point mode; both for mixed mode.
Point-only refers to target specification, not removal of obstacle perception.
Task mode must distinguish follow, navigate-to-point and search; a goal image of
a particular instance is not equivalent to category-level ObjectNav.
Action horizon, points, dt and coordinate convention remain TO BE AUDITED; earlier
8-point/10-Hz suggestions were proposals, not a frozen dataset contract.
Future-frame supervision may be training-only; inference must remain causal.
Test same history/different identity, occlusion recovery, stale/conflicting UWB,
unseen scenes and moving obstacles. Latent alignment alone is not collision proof.
