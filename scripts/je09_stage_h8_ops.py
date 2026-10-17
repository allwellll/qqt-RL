"""H8 keeps H6 labels/loss/verifier; changes only weak courier sampling."""
from scripts import je09_stage_h5c_ops as base_ops
from scripts import je09_stage_h6_ops as h6
from scripts import je09_stage_h8_course as course

for name in dir(h6):
    if not name.startswith('_'):
        globals()[name] = getattr(h6, name)


def collect(*args, **kwargs):
    old_targeted = base_ops.opponent_course.targeted_roles
    old_actions = base_ops.opponent_course.actions
    base_ops.opponent_course.targeted_roles = course.targeted_roles
    base_ops.opponent_course.actions = course.actions
    try:
        return h6.collect(*args, **kwargs)
    finally:
        base_ops.opponent_course.targeted_roles = old_targeted
        base_ops.opponent_course.actions = old_actions
