# Full JEPA-WA closed-loop validation v1

Latest user explicitly superseded the ten-episode diagnostic: stop Job59674 and submit the entire STT/DT/AT validation corpus on eight GPUs in one allocation. Future formal jobs for this WA task must request at least8GPUs unless the user changes this. Never set16 on a single shell/K8s task; that backend supports at most8.

4215episodes total:1405each STT/DT/AT, same existing dataset manifest and eight-way episode assignments, seed7 each. No truncation, filtering, reduced timeout per episode or automatic retries. Eight independent model/simulator lanes share one eight-GPU scheduler allocation. One lane per GPU; this is evaluation of the same final8GPU-trained model, not separate training.

Checkpoint: Job59566 step22707, SHA2562cb78751977cff0a18eee887ae58157fa3f78d6ebf1208534c291a95c5035c52. Model and controller remain those already verified in CLOSED_LOOP_V1.md. Image goal only; no text/UWB or post-initialization GT boxes. Invalid initialization is retained and reported, unlike the older WLA text evaluator's fail-on-invalid behavior. Goal interfaces differ from the original text WLA baseline, so this is not a same-input ablation.

Use original dataset manifest at WLA-EVT-20260925/evt_full_20260926/manifest.json (hash pinned). Metrics: SR; CR; finish; reference-length weighted TR=sum(following_step)/sum(max(total_step,reference_length)); macro following_rate separately. Missing reference lengths use the existing actual-length fallback and their count is reported. Each task must contain exactly1405unique keys; total4215or no complete summary.

Keep per-episode JSONL, partial progress, initial RGB hashes and sampled frames every100simulation steps. Sampling fewer saved images changes visualization density only, not simulation steps, model frequency or metric computation. Unexpected simulator/model errors fail the run with artifacts retained; no silently dropped episodes.

Entrypoints: `wa/scripts/eval_full.sh OUTPUT_ROOT`; `wa/wm/eval_full_launch.py`; `wa/wm/eval_full.py`. Developer static audit checks all eight data shards, not cluster smoke tasks. Existing two-prediction model preflight,13unit tests and actual first-episode rendering from stopped Job59674 establish interface readiness, not efficacy.

Job59674/Task70536 was RUNNING on1A800 and produced initial rollout frames, then STOPPED_USER_SUPERSEDED at explicit request. No ten-episode summary or effectiveness claim exists. Preserve its NAS output.
