"""Candidate source-stratified item replay; not imported by frozen R4.

Only resamples certified learner labels. Source and arrival IDs are checkpoint
metadata and never policy observations. PPO data remain fresh and unchanged.
"""
import numpy as np
from scripts.dagger_rl_memory import aggregate_rows


def _sources(values, count):
    source = np.asarray(values)
    if (source.shape != (count,) or source.dtype.kind not in 'iu'
            or np.any(source < -1)):
        raise ValueError('aligned integer source IDs required; -1 natural, >=0 Bank')
    return source.astype(np.int32, copy=False)


def aggregate_with_sources(new, sources, previous, *, seed, update, strategy,
                           balance_items, capacity=8192, replay_size=8192,
                           fallback=None, fallback_sources=None):
    if len(new) != 6:
        raise ValueError('six supervised arrays required')
    source = _sources(sources, len(new[0]))
    if len(new[0]) and not np.all(new[-2].any((1, 2))):
        raise ValueError('memory accepts only certified nonempty joint labels')
    seen = 0 if previous is None else previous['seen']
    old_source = None
    if previous is not None:
        if previous.get('source_schema_version') != 1:
            raise ValueError('saved source metadata required; no legacy inference')
        old_source = _sources(previous['source'], len(previous['rows'][0]))
    history, _, stats = aggregate_rows(
        new, previous, seed=seed, update=update, strategy=strategy,
        capacity=capacity, replay_size=replay_size, fallback=fallback)
    mapping = ({} if previous is None else
               dict(zip(previous['arrival_id'].tolist(), old_source.tolist())))
    mapping.update(zip(range(seen, seen + len(source)), source.tolist()))
    retained = np.asarray([mapping[int(i)] for i in history['arrival_id']], np.int32)
    history = dict(history, source_schema_version=1, source=retained)
    pool, origins = history['rows'], retained
    if not len(pool[0]):
        if fallback is None or fallback_sources is None:
            raise ValueError('actual fallback rows need their actual source metadata')
        pool, origins = fallback, _sources(fallback_sources, len(fallback[0]))

    # Reproduce the old draw stream exactly, including route rows and positions.
    rng = np.random.default_rng(seed * 1000 + update)
    item = pool[-1]
    ii, rr = np.flatnonzero(item), np.flatnonzero(~item)
    both = len(ii) > 0 and len(rr) > 0
    if both:
        indices = np.concatenate([rng.choice(ii, replay_size // 2),
                                  rng.choice(rr, replay_size // 2)])
        item_count = replay_size // 2
    else:
        indices = rng.integers(0, len(pool[0]), replay_size)
        item_count = replay_size if len(ii) else 0
    natural, bank = ii[origins[ii] < 0], ii[origins[ii] >= 0]
    active = bool(balance_items and len(natural) and len(bank))
    if active:
        # An independent RNG cannot perturb route draws or the environment.
        balance_rng = np.random.default_rng(np.random.SeedSequence([seed, update, 0xC0B5]))
        selected = np.concatenate([
            balance_rng.choice(natural, item_count // 2),
            balance_rng.choice(bank, item_count - item_count // 2),
        ])
        balance_rng.shuffle(selected)
        indices[:item_count] = selected
    if both:
        rng.shuffle(indices)
    replay = tuple(a[indices] for a in pool)
    birth = (history['arrival_update'][indices] if len(history['rows'][0])
             else np.full(replay_size, -1))
    replay_source = origins[indices]
    stats.update(
        replay_unique_rows=int(np.unique(indices).size),
        replay_oldest_update=int(birth.min()), replay_newest_update=int(birth.max()),
        replay_from_first_update=int((birth == 1).sum()),
        replay_from_current_update=int((birth == update).sum()),
        source_balance_active=active,
        new_item_natural_rows=int((new[-1] & (source < 0)).sum()),
        new_item_bank_rows=int((new[-1] & (source >= 0)).sum()),
        history_item_natural_rows=int((history['rows'][-1] & (retained < 0)).sum()),
        history_item_bank_rows=int((history['rows'][-1] & (retained >= 0)).sum()),
        replay_item_natural_rows=int((replay[-1] & (replay_source < 0)).sum()),
        replay_item_bank_rows=int((replay[-1] & (replay_source >= 0)).sum()),
    )
    return history, replay, stats
