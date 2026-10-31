"""Passive item-chain denominators for a future versioned collector report.

This module neither creates teacher labels nor changes actions, loss or RNG.
The eligible flag is a public precondition, not a proven useful opportunity.
"""
import numpy as np


def summarize(flags, oracle_diagnostics, origins):
    flags = np.asarray(flags)
    oracle = np.asarray(oracle_diagnostics)
    origins = np.asarray(origins)
    if flags.shape[:-1] != origins.shape or flags.shape[-1] != 16:
        raise ValueError('expected existing16-field collector event schema')
    if oracle.shape != origins.shape+(6,):
        raise ValueError('expected existing6-field item-oracle diagnostics')
    if not np.isfinite(flags).all() or not np.isfinite(oracle).all() or not np.isfinite(origins).all():
        raise ValueError('nonfinite telemetry')
    rows = {}
    phases = dict(carry=flags[..., 0] > 0, empty_window=flags[..., 15] > 0,
                  empty_combat=flags[..., 14] > 0)
    phase_count = np.stack(list(phases.values())).sum(0)
    if (phase_count > 1).any():
        raise ValueError('overlapping decision phases would double-count ticks')
    for source, origin in [('natural', origins < 0), ('real_pickup_bank', origins >= 0)]:
        for phase, phase_mask in phases.items():
            mask = origin & phase_mask
            count = lambda condition: int((mask & condition).sum())
            row = dict(ticks=int(mask.sum()),
                       banana_inventory_ticks=count(flags[..., 1] > 0),
                       any_item_ability_legal_ticks=count(flags[..., 2] > 0),
                       static_teacher_precondition_ticks=count(oracle[..., 0] > 0),
                       nearby_owned_banana_ticks=count(oracle[..., 1] > 0),
                       certified_release_label_ticks=count(oracle[..., 2] > 0),
                       certified_contact_label_ticks=count(oracle[..., 3] > 0),
                       selected_any_item_release_ticks=count(flags[..., 3] > 0),
                       selected_banana_release_ticks=count((flags[..., 3] > 0) & (flags[..., 1] > 0)))
            rows[source+'/'+phase] = row
    return dict(schema='passive_item_opportunity_v1', rows=rows,
                phase_unclassified_ticks=int((~np.logical_or.reduce(list(phases.values()))).sum()),
                total_ticks=int(origins.size),
                scope='Static forecast labels and actual student actions, per source/decision phase. '
                      'Legal ability2 includes non-banana inventory. Nearby-owned flag can occur '
                      'outside teacher eligibility. Preconditions are not proven beneficial actions. '
                      'Counts are ticks, not independent chains or event-level conversion cohorts.')
