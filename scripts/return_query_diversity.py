"""Select a second certified return deviation while preserving other quotas."""
import jax
import jax.numpy as jnp
from scripts.prestock_fragment_queries import first_deviations as baseline

MIN_GAP = 8


def observed_query_counts(context):
    """Count selected ordinary-return rows on existing host context only.

    This does not rerun the oracle or change labels/losses. A count above two
    is disclosed as diagnostic evidence, never an ability admission gate.
    """
    import numpy as np
    c=np.asarray(context)
    kind3=(c[...,53]>0)&(c[...,54]==3)&(c[...,110]<=0)
    selected=kind3&(c[...,57]<=0)
    terminal=c[...,47]>0
    counts=np.zeros((selected.shape[0],selected.shape[2]),np.int32)
    first=additional=violations=maximum=0
    for tick in range(selected.shape[1]):
        query=selected[:,tick]
        first+=int((query&(counts==0)).sum())
        additional+=int((query&(counts>0)).sum())
        counts+=query
        violations+=int((query&(counts==3)).sum())
        maximum=max(maximum,int(counts.max()))
        counts=np.where(terminal[:,tick],0,counts)
    return dict(ordinary_return_queries=first+additional,first_return_queries=first,
        additional_return_queries=additional,max_queries_per_episode_chunk=maximum,
        kind3_delivery_queries=int((kind3&(c[...,57]>0)).sum()),
        quota_violations=violations,controls_training=False,
        scope='Actual selected collector rows before bounded FIFO selection; per rollout episode chunk, native terminal reset. Actual replay counts are separate; no new oracle query or label import.')


def first_deviations(mismatch,fragment,terminal,preparing,revision,opening,cells):
    """Keep every baseline query plus at most one distinct later return query.

    Inputs already represent native-verified, eligible learner mismatches.
    Cell identity comes from the actual Actor observation, never a reset or
    future target. Phase/revision switches cannot replenish the return quota.
    """
    original=baseline(mismatch,fragment,terminal,preparing,revision,opening)

    def episode(first,mismatch,fragment,terminal,preparing,cells):
        def step(carry,values):
            count,last_tick,last_cell=carry
            first,mismatch,fragment,terminal,preparing,cell,tick=values
            ordinary=(fragment==3)&~preparing
            extra=(mismatch&ordinary&~first&(count==1)
                   &(tick-last_tick>=MIN_GAP)&(cell!=last_cell))
            keep=first|extra
            selected=keep&ordinary
            count=count+selected.astype(jnp.int32)
            last_tick=jnp.where(selected,tick,last_tick)
            last_cell=jnp.where(selected,cell,last_cell)
            count=jnp.where(terminal,0,count)
            last_tick=jnp.where(terminal,-MIN_GAP,last_tick)
            last_cell=jnp.where(terminal,-1,last_cell)
            return (count,last_tick,last_cell),keep

        initial=tuple(map(jnp.int32,(0,-MIN_GAP,-1)))
        return jax.lax.scan(step,initial,(first,mismatch,fragment,terminal,
            preparing,cells,jnp.arange(first.shape[0],dtype=jnp.int32)))[1]

    return jax.vmap(episode,in_axes=(1,1,1,1,1,1),out_axes=1)(
        original,mismatch,fragment,terminal,preparing,cells)
