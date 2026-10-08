"""Explicit execution policy; never changes the v1/v2 selection protocol."""

LEGACY_POLICY = "missing_rvq_v1"
CONTINUATION_POLICY = "missing_rvq_or_final_l2_v1"


def validate_boundary_policy(policy, *, experiment):
    """Reject unknown policies and require v2 identity for the new classifier.

    Imports are local to avoid a collect/protocol_v2 initialization cycle.
    This policy is provenance, not a policy-model input or a success criterion.
    """
    from wa.wm.failure_state_protocol import EXPERIMENT as V1
    from wa.wm.failure_state_teacher_error import EXPERIMENT as V2
    if type(policy) is not str or policy not in (LEGACY_POLICY, CONTINUATION_POLICY):
        raise ValueError("unknown teacher boundary execution policy")
    if experiment not in (V1, V2):
        raise ValueError("unknown teacher boundary experiment")
    if policy == CONTINUATION_POLICY and experiment != V2:
        raise ValueError("missing-final-level policy requires explicit v2 experiment")
    return policy
