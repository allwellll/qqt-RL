"""Static, seat-balanced opponent allocation shared by plan and collectors."""


def opponent_counts(count, stage, interceptor_fraction=.75, defenders=1):
    if count < 2 or count % 2:
        raise ValueError('Even environment count required for balanced seats')
    if (stage not in ('interceptor', 'courier') or not 0 <= interceptor_fraction <= 1
            or defenders < 1 or (stage == 'interceptor' and defenders != 1)):
        raise ValueError('Invalid stage/fraction/defender count')
    pairs = 0 if stage == 'interceptor' else round(count * interceptor_fraction / 2)
    if defenders > 1 and pairs < defenders:
        raise ValueError('Need at least one pair of environments per defender')
    per_defender, extra = divmod(pairs, defenders)
    return tuple(2 * (per_defender + (slot < extra)) for slot in range(defenders)) + (count - 2 * pairs,)
