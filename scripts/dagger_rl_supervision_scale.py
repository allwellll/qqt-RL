"""Single-variable DAgger pressure; all other frozen objective terms persist."""
import math


def scale_loss(base,metrics,coefficient):
    if not math.isfinite(coefficient) or coefficient<0:
        raise ValueError('finite nonnegative supervised coefficient required')
    if coefficient==.2:return base
    return base+(coefficient-.2)*metrics[3]
