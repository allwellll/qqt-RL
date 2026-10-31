"""Descriptive frozen-TRAIN update metrics; no rollout or optimizer calls."""
import numpy as np


def phase_groups(obs, state, live):
    """Partition observed rows, with custody preceding enemy recovery.

    Inactive includes native death/control and the collector terminal latch.
    Own-origin cargo and empty windows without stock remain explicit 'other'.
    """
    obs, state, live = np.asarray(obs), np.asarray(state), np.asarray(live, bool)
    active = live & (state[:, 9] == 1)
    carry = active & (state[:, 14] > 0)
    empty = active & (state[:, 13] == 0)
    drop = empty & np.any(obs[:, 17] > 0, axis=(-2, -1))
    window = (state[:, 10] <= 0) | (state[:, 10] > 1)
    stock = empty & ~drop & window & (state[:, 20] > 0)
    combat = empty & ~drop & ~window
    inactive = ~active
    other = ~(carry | drop | stock | combat | inactive)
    return dict(combat=combat, stock_window=stock, drop=drop, carry=carry,
                inactive=inactive, other=other)


def summarize(advantage, logp_delta, selected):
    """All ratios carry their denominator; empty groups are missing values."""
    a = np.asarray(advantage, float)[selected]
    d = np.asarray(logp_delta, float)[selected]
    nz = a != 0
    return dict(rows=int(len(a)), nonzero_advantage_rows=int(nz.sum()),
                advantage_mean=float(a.mean()) if len(a) else None,
                advantage_variance=float(a.var()) if len(a) else None,
                positive_advantage_fraction=float((a > 0).mean()) if len(a) else None,
                logp_delta_mean=float(d.mean()) if len(a) else None,
                advantage_logp_delta_product_mean=float((a*d).mean()) if len(a) else None,
                positive_alignment_fraction=float((a[nz]*d[nz] > 0).mean()) if nz.any() else None)
