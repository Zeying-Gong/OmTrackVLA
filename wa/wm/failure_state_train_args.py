"""CPU-only, opt-in arguments for the bounded failure-state training recipe.

This module never opens paths or verifies artifact contents. Passing it is not
cache admission, sampling-plan admission, optimizer verification, or release.
The training entry retains its existing MD_AK_JOB_ID diagnostic prohibition.
"""
import math
import re

PARENT_SHA256 = "ab39144b0490937ce43c4ab28e2d90cdb0c560f700d3364a29ce1dc9a08b999d"
FIELDS = (
    "failure_state_cache",
    "failure_state_admission_sha256",
    "failure_state_plan",
    "failure_state_plan_admission_sha256",
    "failure_state_dedup_report",
    "failure_state_dedup_sha256",
    "failure_state_old_run",
    "failure_state_old_plan_sha256",
    "failure_state_old_exposure_sha256",
    "failure_state_loader_audit",
    "failure_state_loader_audit_sha256",
)
PIN_FIELDS = {
    "old_plan": "failure_state_old_plan_sha256",
    "old_actual_exposure": "failure_state_old_exposure_sha256",
    "recovery_admission": "failure_state_admission_sha256",
    "dedup_report": "failure_state_dedup_sha256",
}


def add_arguments(parser):
    """Add eleven opt-in string arguments, without changing existing defaults."""
    group = parser.add_argument_group("failure-state evaluation-set adaptation")
    for field in FIELDS:
        group.add_argument("--" + field.replace("_", "-"), default=None,
                           help="Explicit audited recipe input; not an automatic release.")
    return parser


def _path(value, label):
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ValueError(label + ": nonempty path string required")


def _sha(value, label):
    # All project artifact identities use canonical lowercase hexdigests.
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(label + ": canonical 64-character SHA256 required")


def _bundle(arguments):
    values = {field: getattr(arguments, field, None) for field in FIELDS}
    if all(value is None for value in values.values()):
        return None
    missing = [field for field, value in values.items() if value is None]
    if missing:
        raise ValueError("failure-state recipe requires all eleven fields: " + ", ".join(missing))
    for field, value in values.items():
        (_sha if field.endswith("_sha256") else _path)(value, field)
    return values


def validate_training_recipe(a, world=None):
    """Return False for legacy runs; reject any incomplete or changed opt-in.

    world=None performs only the pre-distributed argument check. The caller must
    call again with the actual integer world size before using this recipe.
    Diagnostic runs allow 1 or 8 ranks but must separately pass the entrypoint's
    existing prohibition on diagnostics in a scheduled cluster job.
    """
    if _bundle(a) is None:
        return False
    expected = {
        "kind": "jepa",
        "evaluation_set_adaptation": True,
        "dual_teacher_repeats": 1,
        "completed_epochs": 1,
        "epochs": 1,
        "seed": 42,
        "batch_size": 2,
        "accumulation": 2,
        "world_weight": .1,
        "history_repeat_probability": .25,
    }
    for field, target in expected.items():
        value = getattr(a, field, None)
        if (type(value) is not type(target) or value != target or
                (type(value) is float and not math.isfinite(value))):
            raise ValueError("failure-state fixed recipe: " + field)
    for field in ("dual_teacher_cache", "teacher_window_plan", "resume"):
        _path(getattr(a, field, None), field)
    _sha(getattr(a, "teacher_plan_report_sha256", None), "teacher_plan_report_sha256")
    _sha(getattr(a, "resume_sha256", None), "resume_sha256")
    if a.resume_sha256 != PARENT_SHA256:
        raise ValueError("failure-state initialization requires the fixed 59866 parent")
    for field in ("recovery_cache", "recovery_index"):
        if getattr(a, field, None) is not None:
            raise ValueError("failure-state excludes legacy recovery mix: " + field)
    for field in ("data_root", "source_prefix"):
        if getattr(a, field, None) is not None:
            raise ValueError("failure-state excludes original-source redirection: " + field)
    diagnostic = getattr(a, "diagnostic", None)
    if type(diagnostic) is not bool:
        raise ValueError("diagnostic must be an explicit boolean")
    if world is not None:
        if type(world) is not int or world not in ((1, 8) if diagnostic else (8,)):
            raise ValueError("failure-state requires world=8 (developer diagnostic: 1 or 8)")
    return True


def source_pins_from_args(a):
    """Return the four sampling pins, not their contents or a release decision.

    Disabled legacy arguments return {}; partial or malformed bundles fail
    closed. The runtime must verify the pin values against actual artifacts.
    """
    values = _bundle(a)
    if values is None:
        return {}
    return {key: values[field] for key, field in PIN_FIELDS.items()}
