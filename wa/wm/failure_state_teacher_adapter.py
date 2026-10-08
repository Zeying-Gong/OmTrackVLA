"""Local failure-state adapter: reject fallback, narrowly classify decode errors.

The existing BoundTeacher and LightNav client remain unchanged. A successful
call is forwarded exactly once. Missing-RVQ classification remains the default.
The separately observed missing-final-level response requires explicit opt-in.
All transport guards still apply; every other failure remains fatal. Runtime
integration and real Habitat validation are separate gates.
"""
import ast
import re

from wa.wm.failure_state_teacher_error import TeacherOutputInvalid


def _missing_final_level(message):
    """Recognize only a canonical decoder repr with one level0 and one level1.

    This is not a general rc500 handler. In particular, duplicate, gapped or
    malformed action tokens cannot turn an arbitrary error into a candidate.
    The small length bound keeps literal parsing bounded and fails closed.
    """
    prefix = "got 2 act levels, expected 3 from "
    if type(message) is not str or not message.startswith(prefix):
        return False
    literal = message[len(prefix):]
    if not literal or len(literal) > 4096:
        return False
    try:
        text = ast.literal_eval(literal)
    except (ValueError, SyntaxError, TypeError, RecursionError):
        return False
    if type(text) is not str or repr(text) != literal:
        return False
    tokens = re.findall(r"<act_l([0-9]+)_([0-9]+)>", text)
    return (len(tokens) == 2 and {level for level, _ in tokens} == {"0", "1"}
            and text.count("<act_l") == len(tokens))


class FailureStateTeacherBoundary:
    def __init__(self, bound_teacher, *, teacher_name, allow_missing_final_level=False):
        if type(allow_missing_final_level) is not bool:
            raise ValueError("missing-final-level opt-in must be an explicit bool")
        if teacher_name not in ("lightnav", "oracle"):
            raise ValueError("explicit teacher identity required")
        if getattr(bound_teacher, "allow_released_fallback", None) is not False:
            raise ValueError("fallback execution must be disabled")
        if not hasattr(bound_teacher, "teacher"):
            raise ValueError("bound teacher must retain its actual client")
        self.teacher = bound_teacher
        self.teacher_name = teacher_name
        self.allow_missing_final_level = allow_missing_final_level

    def __getattr__(self, name):
        return getattr(self.teacher, name)

    def act(self, *args, **kwargs):
        raw = self.teacher.teacher
        request_seq = getattr(raw, "seq", None)
        fallback_before = getattr(raw, "episode_fallback_count", 0)
        try:
            return self.teacher.act(*args, **kwargs)
        except RuntimeError as exc:
            # Do not absorb replay errors, unknown RuntimeErrors, subclasses,
            # malformed metadata, rc=0 empty trajectories or reset failures.
            if (type(exc) is not RuntimeError
                    or str(exc) != "TEACHER_TRANSPORT_FALLBACK"
                    or self.teacher_name != "lightnav"):
                raise
            reply = getattr(raw, "reply_error", None)
            if not isinstance(reply, dict):
                raise
            rc, seq, msg = reply.get("rc"), reply.get("seq"), reply.get("msg")
            count = getattr(raw, "episode_fallback_count", None)
            if (type(rc) is not int or rc != 500
                    or type(seq) is not int or seq < 0
                    or type(request_seq) is not int or seq != request_seq
                    or type(getattr(raw, "seq", None)) is not int or raw.seq != seq + 1
                    or type(fallback_before) is not int or fallback_before < 0
                    or type(count) is not int or count != fallback_before + 1
                    or not isinstance(msg, str)
                    or not (re.fullmatch(r"Missing rvq act levels \[[0-9]+(?:, [0-9]+)*\] in .+", msg, re.DOTALL)
                            or (self.allow_missing_final_level
                                and _missing_final_level(msg)))):
                raise
            # BoundTeacher raised before returning an action to ReplayThenTeacher,
            # so the rejected fallback is not an env.step command or a label.
            raise TeacherOutputInvalid(rc=rc, seq=seq, message=msg) from exc
