"""Candidate eviction-only DAgger memory with audit metadata, never policy input."""
import numpy as np
from scripts.dagger_rl_reservoir import append as reservoir_append


def aggregate_rows(new, previous, *, seed, update, strategy, capacity=8192,
                   replay_size=8192, fallback=None):
    """Accept already-certified learner rows; keep existing family/RNG sampling."""
    if strategy not in ['fifo', 'reservoir'] or capacity<=0 or replay_size<=0 or replay_size%2:
        raise ValueError('invalid eviction policy or memory/replay size')
    if len(new)!=6 or any(len(a)!=len(new[0]) for a in new):
        raise ValueError('expected six aligned supervised row arrays')
    seen=0 if previous is None else previous['seen']
    if previous is not None and (previous['schema_version']!=1 or previous['strategy']!=strategy):
        raise ValueError('incompatible saved memory schema or eviction policy')
    old=None if previous is None else (*previous['rows'],previous['arrival_id'],previous['arrival_update'])
    n=len(new[0]);arrival=np.arange(seen,seen+n,dtype=np.int64)
    incoming=(*new,arrival,np.full(n,update,dtype=np.int64))
    if strategy=='reservoir':
        # Separate stream: replay and environment random draws stay untouched.
        memory,total=reservoir_append(old,incoming,seen=seen,capacity=capacity,
                                      seed=np.random.SeedSequence([seed,update,0xA631]))
    else:
        memory=tuple(np.concatenate([a,b])[-capacity:] for a,b in zip(old,incoming)) if old is not None else tuple(a[-capacity:].copy() for a in incoming)
        total=seen+n
    history=dict(schema_version=1,strategy=strategy,seen=total,rows=memory[:6],arrival_id=memory[6],arrival_update=memory[7])
    pool=history['rows']
    if not len(pool[0]):
        if fallback is None or not len(fallback[0]):
            raise ValueError('empty memory needs the actual on-policy fallback rows')
        pool=fallback
    rng=np.random.default_rng(seed*1000+update)
    item=pool[-1];ii,rr=np.flatnonzero(item),np.flatnonzero(~item)
    if len(ii) and len(rr):
        indices=np.concatenate([rng.choice(ii,replay_size//2),rng.choice(rr,replay_size//2)])
        rng.shuffle(indices)
    else:
        indices=rng.integers(0,len(pool[0]),replay_size)
    replay=tuple(a[indices] for a in pool)
    birth=history['arrival_update'][indices] if len(history['rows'][0]) else np.full(replay_size,-1)
    stats=dict(new_rows=n,new_item_rows=int(new[-1].sum()),history_rows=len(history['rows'][0]),
               history_item_rows=int(history['rows'][-1].sum()),replay_item_rows=int(replay[-1].sum()),historical_loss_weight=.5,
               memory_strategy=strategy,accepted_seen=total,replay_unique_rows=int(np.unique(indices).size),
               replay_oldest_update=int(birth.min()),replay_newest_update=int(birth.max()),
               replay_from_first_update=int((birth==1).sum()),replay_from_current_update=int((birth==update).sum()))
    return history,replay,stats
