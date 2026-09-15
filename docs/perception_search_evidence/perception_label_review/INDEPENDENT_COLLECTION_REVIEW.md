# Independent CPU review of train4 perception sidecar collection

The reviewed implementation is suitable for a separate, label-only replay collection after its frozen remote inputs pass preflight. This review does not admit the output to any optimizer, establish product acceptance, or claim that GPU collection has run. No training samples, v4 loss/schema, model gate, old runtime, or remote GPU process were changed by this reviewer.

The final local check ran **47/47 tests successfully in 0.611 s**. All four implementation files parse as Python 3.9. See `independent_cpu_verification.log` and `independent_review_source_sha256.json` for the reviewed snapshot. Remote Python 3.9 runtime testing, actual input hashing, GPU execution, and independent output admission remain the parent/collector owner's responsibility.

## Scope and denominator

The exact four original trajectories contain 85, 104, 84, and 93 actions. Reset through natural terminal therefore contains **370 observations**. Their permanent role is train. The sidecar references the **19 admitted samples from these long source trajectories**. Four legacy train samples come from different short trajectories and are explicitly excluded; four validation samples stay held out.

Existing prefix coverage is:

| Source | Samples | Longest anchor | Distinct observations with stored RGB/time |
|---|---:|---:|---:|
| AT 401 | 3 | 64 | 65 |
| DT 200 | 5 | 64 | 65 |
| STT 0 | 5 | 32 | 33 |
| STT 3100 | 6 | 64 | 65 |
| Total | 19 | — | **228** |

The overlapping prefixes produce **645 comparison checks**, not 645 distinct observations. There are 142 observations beyond those stored prefixes: 138 nonterminal observations have the source's next-policy RGB mean/std/temporal-difference statistics plus geometry/visibility checks; the four terminal observations have no next source policy call. The terminal observations still have post-action camera, target distance, visibility, and independent same-render audit checks. Do not describe all 370 observations as having original per-pixel or recorded world-time references.

## Confirmed contracts

- RGB and panoptic arrays are copied from the same single sensor-observation dictionary. Camera and world time are read on both sides of the render. The raw panoptic shape/dtype and `.npy` SHA are preserved separately from the canonical squeezed HxW array hash.
- Observation 0 precedes source action 1. Observation k follows source action k and conditions source action k+1 when k<N. Terminal N does not invent another policy call. The RGB statistics comparison uses `source['steps'][k]['policy']` for k<N, matching the original runtime's float64 mean/std and float32 temporal-difference calculation.
- Every stored action is replayed unchanged. GT distance, visibility, semantic IDs and initialization are label/audit inputs only. Failed comparison checks mark failed evidence and do not choose replacement actions. Early natural termination retains partial frame files and progress, then fails.
- The fixed four-entry batch denominator exists before workers start. Native failure cannot drop a source from the denominator; there is no source substitution or retry. Output creation is exclusive. Timeout now remains failure even if termination returns exit code zero with a completed worker result.
- Bbox coordinates use inclusive pixel extrema divided by width/height, matching the frozen existing training normalizer. A full 5x4 mask is `[0,0,.8,.75]`. Single-pixel, one-row and one-column masks have valid visibility supervision but invalid bbox supervision.
- GT sidecars do not certify predicted visual identity, UWB binding, stop, motion permission, ego state, or occluded-versus-out-of-view classification. Their optimizer/product eligibility remains false.

## Findings resolved during review

The original preflight accepted 19 fake/duplicate prefix references and an empty artifact registry; with source pins omitted, changing a source GT distance while retaining its recorded SHA also passed. This was a verifier completeness gap, not evidence that the normal builder generated a bad plan. `review_plan_counterexamples.py` records the original probe construction.

The collector owner corrected the verifier to require consumed protocol/source/config/sample/media references in the artifact registry, freeze authoritative protocol hashes and permanent roles, require 19 unique sample paths, and reconstruct prefix references from the fixed admitted manifest and actual source files. Independent regression tests now cover a valid reference graph, missing source/status/launch/config/dataset pins, missing sample/initial/prefix media pins, duplicate samples, resealed pixel/time references, and changed permanent roles. Those graph tests mock file I/O to isolate verifier logic; they do not substitute for the remote real-file preflight.

The collector initially did not compare source policy RGB statistics outside the stored prefix. It now performs the correctly indexed comparisons and explicitly marks terminal observations as lacking next-policy statistics. A timeout result-accounting edge was also corrected by requiring no recorded worker error for success.

## Integration limitation

The current existing `phase3_sequence_loss` bbox branch does not consume the new independent `bbox_label_valid` field. A future admitted label loader/loss must handle visibility and bbox validity separately so that a visible single pixel is not incorrectly used for bbox regression. This label-only work intentionally does not change v4 training or imply that the sidecars are training-ready.

Independently authored tests are `test_semantic_labels_compatibility.py`, `test_collection_contract_independent.py`, `test_collector_independent.py`, and `test_plan_verifier_independent.py`. The compatibility test checks the exact frozen normalizer SHA and supports both the local snapshot layout and the intended remote bundle layout.
