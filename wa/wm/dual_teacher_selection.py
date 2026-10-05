"""Auditable dual-teacher selection for explicitly authorized in-set adaptation.

This does not relax the legacy training-only collector or release data for training.
Both complete branches must be evaluated from the same replay-verified state.
"""
import math

PAIR_FIELDS = ("task", "key", "takeover_step", "seed", "protocol_sha256",
               "initial_rgb_sha256", "takeover_state_sha256")
EXPERIMENT = "evaluation_set_adaptation_v1"

def select_teacher(lightnav, oracle):
    branches = {"lightnav": lightnav, "oracle": oracle}
    for name, branch in branches.items():
        if branch.get("experiment") != EXPERIMENT:
            raise ValueError("explicit in-set adaptation tag required")
        if branch.get("teacher") != name:
            raise ValueError("teacher identity mismatch")
        if branch.get("complete") is not True or branch.get("replay_verified") is not True:
            raise ValueError("incomplete or unverified branch")
        if branch.get("transport_fallback", False):
            raise ValueError("teacher fallback invalidates comparison")
        for field in PAIR_FIELDS:
            if field not in branch or branch[field] is None:
                raise ValueError("missing pairing evidence: " + field)
        r = branch["result"]
        for field in ("success", "collision", "policy_init_valid"):
            if r.get(field) not in (True, False, 0, 1):
                raise ValueError("missing or invalid result flag: " + field)
        rate = r.get("following_rate")
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or not 0 <= rate <= 1:
            raise ValueError("following_rate must be finite in [0,1]")
        # Preserve the benchmark result; never silently call inconsistent data a success.
        if r["success"] and (r["collision"] or not r["policy_init_valid"]):
            raise ValueError("inconsistent successful branch")
    if any(lightnav[f] != oracle[f] for f in PAIR_FIELDS):
        raise ValueError("branches do not share the same start and protocol")
    valid = [name for name, b in branches.items() if b["result"]["success"]]
    selected = None
    if not valid:
        reason = "both_failed_no_demonstration"
    elif len(valid) == 1:
        selected = valid[0]
        reason = "only_successful_teacher"
    else:
        ln = lightnav["result"]["following_rate"]
        oc = oracle["result"]["following_rate"]
        selected = "oracle" if oc > ln else "lightnav"
        reason = "higher_following_rate" if ln != oc else "equal_rate_fixed_lightnav_tie"
    return {
        "experiment": EXPERIMENT,
        "pair": {f: lightnav[f] for f in PAIR_FIELDS},
        "selected_teacher": selected,
        "reason": reason,
        "results": {name: dict(b["result"]) for name, b in branches.items()},
        "demonstration_candidate": selected is not None,
        "training_released": False,
        "pending": "continuous successful suffix, executed-pose labels and input-boundary audit",
    }
